import logging
from typing import List, Dict, Optional
import numpy as np
import json

import torch
from tqdm import tqdm
import open3d as o3d

from concept_graphs.vlm.OpenAIVerifier import OpenAIVerifier
from concept_graphs.utils import split_camel_preserve_acronyms, aabb_iou

from .BaseMapEngine import BaseMapEngine

log = logging.getLogger(__name__)


class QueryObjects(BaseMapEngine):
    def __init__(
        self, verifier: OpenAIVerifier, receptacles_bbox: Dict, pickupable_to_receptacles: Dict, top_k: int = 5, **kwargs
    ):
        super().__init__(**kwargs)
        self.verifier = verifier
        self.receptacles_bbox = receptacles_bbox
        self.pickupable_to_receptacles = pickupable_to_receptacles
        self.top_k = top_k
        self.receptacle_map_ids = self.get_receptacle_map_ids()
        #{'TVStand|7|0|0___0': 33, 'TVStand|4|4|0___0': None, 'SideTable|4|3': 0, 'DiningTable|6|3|0': 18}

    def get_receptacle_map_ids(self) -> Dict[str, int]:
        """
        For each receptacle OOBB (given as 8 corner points), find the map
        object id whose bounding box has the highest IoU with it.
        """
        # self.receptacles_bbox[receptacle_key]["cornerPoints"] is a list of 8 points

        receptacle_to_map: Dict[str, int] = {}

        for receptacle_key, rec_data in self.receptacles_bbox.items():
            corner_points = rec_data["cornerPoints"]

            # corner_points is already a list of 8 [x, y, z] points
            rec_corners = np.array(corner_points, dtype=np.float32)
            # Negate ya-xis
            rec_corners[:, 1] = -rec_corners[:, 1]

            best_iou = 0.0
            best_idx: Optional[int] = None

            # Iterate over all map bboxes
            for idx, bbox in enumerate(self.bbox):
                # bbox is expected to be an Open3D OrientedBoundingBox
                box_points = np.asarray(bbox.get_box_points(), dtype=np.float32)
                iou = aabb_iou(rec_corners, box_points)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            receptacle_to_map[receptacle_key] = best_idx

        return receptacle_to_map

    def find_receptacle(
        self, object_centroid: List[float], receptacles: Dict
    ) -> Optional[str]:
        """
        Finds the closest receptacle to the object centroid based on Euclidean distance
        to the receptacle's center.
        """
        if not receptacles:
            return None

        obj_x, obj_y, obj_z = object_centroid

        closest_rec = None
        min_dist_sq = float("inf")

        for rec_name, rec_data in receptacles.items():
            c = rec_data["center"]
            # Note: Negate y-axis to be RH frame
            rec_x, rec_y, rec_z = c["x"], -c["y"], c["z"]

            # Calculate squared Euclidean distance (faster than sqrt)
            dist_sq = (
                (obj_x - rec_x) ** 2 + (obj_y - rec_y) ** 2 + (obj_z - rec_z) ** 2
            )

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_rec = rec_name

        return closest_rec

    def process_queries(self, queries: List[str], **kwargs) -> Dict:
        """
        Returns:
        {
          "query_text": {
              "present": bool,
              "map_object_id": int | None,
              "query_timestamp": List[float],
              "present_receptacle_name": str | None,
              "receptacles": [
                    {
                        "receptacle_name": str,
                        "map_object_id": int,
                        "receptacle_timestamps": List[float],
                    },
                    ...
                ]
            }
        }
        """
        results: Dict[str, Dict] = {}

        log.info("Encoding queries...")
        # Encode queries -> (num_queries, feature_dim)
        queries_cleaned = [
            split_camel_preserve_acronyms(q.split("|")[0]) for q in queries
        ]
        text_features = self.ft_extractor.encode_text(queries_cleaned).cpu()

        # 2. Calculate Similarity Matrix: (num_queries, num_map_objects)
        for i, query_text in tqdm(
            enumerate(queries), total=len(queries), desc="Processing Queries"
        ):
            query_feat = text_features[i].unsqueeze(0)  # (1, dim)

            # Calculate similarity against all map objects
            query_feat = query_feat.to(self.device)
            features = self.features.to(self.device)
            sim_scores = self.semantic_sim(query_feat, features).squeeze().cpu()

            # Get Top K indices
            top_k_indices = torch.argsort(sim_scores, descending=True)[: self.top_k]
            top_k_indices = top_k_indices.cpu().numpy()

            # Initialize output structure for this query
            result_entry = {
                "present": False,
                "map_object_id": None,
                "query_timestamp": [],
                "present_receptacle_name": None,
                "receptacles": [],
            }

            # build receptacles list from mapping
            receptacle_names = self.pickupable_to_receptacles[query_text]
            for rec_name in receptacle_names:
                if rec_name == "OOB_FAKE_RECEPTACLE":
                    continue

                # Map name -> map object id
                rec_map_id = self.receptacle_map_ids[rec_name]
                if rec_map_id is None:
                    # In case the receptacle was not mapped properly
                    # We still include the receptacle but without timestamps
                    result_entry["receptacles"].append(
                        {
                            "receptacle_name": rec_name,
                            "map_object_id": None,
                            "receptacle_timestamps": [],
                        }
                    )
                    continue

                rec_timestamps = self.annotations[rec_map_id]["timestamps"]

                result_entry["receptacles"].append(
                    {
                        "receptacle_name": rec_name,
                        "map_object_id": rec_map_id,
                        "receptacle_timestamps": rec_timestamps,
                    }
                )

            # 3. Verify candidates to find where the pickupable currently is
            for idx in top_k_indices:
                idx = int(idx)  # Ensure python int
                obj_data = self.annotations[idx]

                # Get images
                images = self.get_object_images(idx, limit=self.verifier.max_images)
                if len(images) == 0:
                    log.warning(f"No images found for object ID {idx}. Skipping.")
                    continue

                # Verify
                query_text_cleaned = split_camel_preserve_acronyms(
                    query_text.split("|")[0]
                )
                is_match = self.verifier(images, query_text_cleaned)

                if is_match:
                    # 4. Spatial Association: which receptacle does it belong to now?
                    # NOTE from @kumaradityag: It is possible that the rec_name is not the correct receptacle, if the incorrect object was retrieved
                    centroid = obj_data["centroid"]
                    # TODO: this should use the map bbox
                    rec_name = self.find_receptacle(centroid, self.receptacles_bbox)

                    result_entry["present"] = True
                    result_entry["map_object_id"] = idx
                    result_entry["query_timestamp"] = obj_data.get("timestamps", [])
                    result_entry["present_receptacle_name"] = rec_name
                    break

            results[query_text] = result_entry

        return results

    def visualize(self, res_path: str):
        """Visualize all object point clouds with labels in an Open3D window."""
        geometries = []
        
        # Add all point clouds
        for pcd in self.pcd:
            geometries.append(pcd)

        # Read result json file
        with open(res_path, 'r') as f:
            results = json.load(f)

        # Get pickupables
        pickupable_ids = []
        for _, val in results.items():
            if val['present']:
                pickupable_ids.append(val['map_object_id'])
        
        bboxes = [self.bbox[idx] for idx in pickupable_ids]
        
        # Add bounding boxes
        for bbox in bboxes:
            # Assign random color
            color = np.random.rand(3)
            bbox.color = color.tolist()
            geometries.append(bbox)
        
        # Visualize
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Map Objects Visualization",
            width=1024,
            height=768,
        )


class QueryReceptacles(QueryObjects):
    def process_queries(self, queries: List[float]) -> Dict:
        """
        Overrides the base method to always return None since we are querying receptacles.
        """
        results = {}

        log.info("Encoding queries...")
        # Encode queries -> (num_queries, feature_dim)
        queries_cleaned = [
            split_camel_preserve_acronyms(q.split("|")[0]) for q in queries
        ]
        text_features = self.ft_extractor.encode_text(queries_cleaned).cpu()
        mapped_receptacles = list()

        # 2. Calculate Similarity Matrix: (num_queries, num_map_objects)
        for i, query_text in tqdm(
            enumerate(queries), total=len(queries), desc="Processing Queries"
        ):
            query_feat = text_features[i].unsqueeze(0)  # (1, dim)

            # Calculate similarity against all map objects
            # sim shape: (1, num_objects) -> squeeze to (num_objects)
            query_feat = query_feat.to(self.device)
            features = self.features.to(self.device)
            sim_scores = self.semantic_sim(query_feat, features).squeeze().cpu()
            if len(mapped_receptacles) > 0:
                mapped_idx = torch.tensor(mapped_receptacles, dtype=torch.long)
                sim_scores[mapped_idx] = 0.0

            # Get Top K indices
            top_k_indices = torch.argsort(sim_scores, descending=True)[: self.top_k]
            top_k_indices = top_k_indices.cpu().numpy()

            found_details = {
                "present": False,
                "oobb": None,
                "map_object_id": None,
            }
            bboxes = list()
            objects_id = list()

            # 3. Verify candidates
            for idx in top_k_indices:
                idx = int(idx)  # Ensure python int

                # Get images
                images = self.get_object_images(idx, limit=self.verifier.max_images)
                if len(images) == 0:
                    log.warning(f"No images found for object ID {idx}. Skipping.")

                    continue

                # Verify
                query_text_cleaned = split_camel_preserve_acronyms(
                    query_text.split("|")[0]
                )
                is_match = self.verifier(images, query_text_cleaned)

                if is_match:
                    # 4. Save results
                    bbox = self.bbox[idx]
                    vertices = np.asarray(bbox.get_box_points())
                    rotation = bbox.R
                    center = bbox.center
                    extent = bbox.extent

                    objects_id.append(idx)
                    bboxes.append({
                            "center": center.tolist(),
                            "rotation": rotation.flatten().tolist(),
                            "extent": extent.tolist(),
                            "vertices": [v.tolist() for v in vertices],
                        })

                    found_details = {
                        "present": True,
                    }
                    # Blacklist the receptacle as already mapped
                    mapped_receptacles.append(idx)

            found_details["oobb"] = bboxes
            found_details["map_object_id"] = objects_id
            results[query_text] = found_details

        return results

    def visualize(self, res_path: str):
        """Visualize all object point clouds with labels in an Open3D window."""
        geometries = []
        
        # Add all point clouds
        for pcd in self.pcd:
            geometries.append(pcd)

        # Read result json file
        with open(res_path, 'r') as f:
            results = json.load(f)

        # Get pickupables
        receptacle_ids = []
        for _, val in results.items():
            if val['present']:
                receptacle_ids.extend(val['map_object_id'])
    
        bboxes = [self.bbox[idx] for idx in receptacle_ids]
        
        # Add bounding boxes
        for bbox in bboxes:
            # Assign random color
            color = np.random.rand(3)
            bbox.color = color.tolist()
            geometries.append(bbox)
        
        # Visualize
        o3d.visualization.draw_geometries(
            geometries,
            window_name="Map Objects Visualization",
            width=1024,
            height=768,
        )
