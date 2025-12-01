import logging
import shutil
import os
import glob
from natsort import natsorted
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
        self,
        verifier: OpenAIVerifier,
        receptacles_bbox: Dict,
        pickupable_to_receptacles: Dict,
        top_k: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.verifier = verifier
        self.receptacles_bbox = receptacles_bbox
        self.pickupable_to_receptacles = pickupable_to_receptacles
        self.top_k = top_k
        self.receptacle_map_ids = self.get_receptacle_map_ids()

    def get_receptacle_map_ids(self) -> Dict[str, int]:
        """
        For each receptacle OOBB (given as 8 corner points), find the map
        object id whose bounding box has the highest IoU with it.
        """
        # self.receptacles_bbox[receptacle_key]["cornerPoints"] is a list of 8 points

        receptacle_to_map: Dict[str, int] = {}

        for receptacle_key, rec_data in self.receptacles_bbox.items():
            if receptacle_key == "OOB_FAKE_RECEPTACLE":
                receptacle_to_map[receptacle_key] = None
                continue
            corner_points = rec_data["cornerPoints"]

            # corner_points is already a list of 8 [x, y, z] points
            rec_corners = np.array(corner_points, dtype=np.float32)

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

        map_point = np.array(object_centroid)

        closest_rec = None
        min_dist_sq = float("inf")
        # TODO: make assignment not based on centroid but being on centroid or overelapping bbox

        for rec_name, rec_data in receptacles.items():
            corners = rec_data["cornerPoints"]
            rec_center = np.mean(np.array(corners), axis=0)

            dist_sq = np.linalg.norm(map_point - rec_center) ** 2

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
                    rec_name = self.find_receptacle(centroid, self.receptacles_bbox)

                    result_entry["present"] = True
                    result_entry["map_object_id"] = idx
                    result_entry["query_timestamp"] = obj_data.get("timestamps", [])
                    result_entry["present_receptacle_name"] = rec_name
                    break

            results[query_text] = result_entry

        return results

    def save_results(self, results, output_path, sssd_data: object):
        """Save results using SSSD class."""
        import polars as pl
        # TODO: @miguel: Ask charlie how to save this using her thing.
        timestamps = np.concatenate(
            [data._timestamp for data in sssd_data.get_generator_of_selves()]
        )
        priviliged_assignments = np.concatenate(
            [data._assignment for data in sssd_data.get_generator_of_selves()]
        )
        receptacle_to_int = {
            name: i for i, name in enumerate(sssd_data.receptacles_in_scene)
        }
        pickupable_to_int = {
            name: i for i, name in enumerate(sssd_data.pickupables_in_scene)
        }

        # -2 is not valid receptacle, -1 is not observed, 0 is absent and 1 is present
        assignments = np.ones_like(priviliged_assignments, dtype=int) * -2

        for p_name, res_data in results.items():
            p_idx = pickupable_to_int[p_name]
            # Populate receptacles that are either non-observed (-1) or absent (0)
            for receptacle in res_data["receptacles"]:
                rec_name = receptacle["receptacle_name"]
                if rec_name == "OOB_FAKE_RECEPTACLE":
                    continue
                # Populating non-observed for all timestamps (-1)
                r_idx = receptacle_to_int[rec_name]
                assignments[:, p_idx, r_idx] = -1
                # Populate absent for all receptacles (0)
                for t in receptacle["receptacle_timestamps"]:
                    t_idx = np.argmin(np.abs(timestamps - t))
                    assignments[t_idx, p_idx, r_idx] = 0
            # Populate presence (1) if it occurs
            if res_data["present"]:
                rec_name = res_data["present_receptacle_name"]
                r_idx = receptacle_to_int[rec_name]
                # Timestamps where the pickupable was observed
                for t in res_data["query_timestamp"]:
                    t_idx = np.argmin(np.abs(timestamps - t))
                    assignments[t_idx, p_idx, r_idx] = 1
                # Timestamps where the receptacle was observed with the pickupable
                for rec_entry in res_data["receptacles"]:
                    if rec_entry["receptacle_name"] != rec_name:
                        continue
                    for t in rec_entry["receptacle_timestamps"]:
                        t_idx = np.argmin(np.abs(timestamps - t))
                        assignments[t_idx, p_idx, r_idx] = 1

        # Create res directory if it doesn't exist
        if output_path.exists() and output_path.is_dir():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save skeleton, delete old parquets and add new one
        sssd_data.dump_to_parquet(output_path, dump_leftover=True, verbose=False)
        parquet_paths = natsorted(glob.glob(str(output_path / "*.parquet")))
        for pq_path in parquet_paths:
            os.remove(pq_path)

        # Create dummy data for non-used keys
        data = pl.DataFrame({
            "timestamp": timestamps,
            "assignment": assignments
        })
        # Memory efficient way to add empty columns
        data = data.with_columns([
            pl.lit(None, dtype=pl.List(pl.Float64)).alias("position"),
            pl.lit(None, dtype=pl.List(pl.Float64)).alias("rotation"),
            pl.lit(None, dtype=pl.List(pl.Float64)).alias("aabb_center"),
            pl.lit(None, dtype=pl.List(pl.Float64)).alias("aabb_size"),
            pl.lit(None, dtype=pl.List(pl.List(pl.Float64))).alias("aabb_cornerPoints"),
            pl.lit(None, dtype=pl.List(pl.List(pl.Float64))).alias("oobb_cornerPoints"),
        ])
        data.write_parquet(output_path / "scan_0.parquet")

        # Save receptacles map ids
        with open(output_path  / "receptacle_map_ids.json", "w") as f:
            json.dump(self.receptacle_map_ids, f, indent=4)

        # Save pickupable map ids
        pickupable_map_ids = {}
        for p_name, res_data in results.items():
            if res_data["present"]:
                pickupable_map_ids[p_name] = res_data["map_object_id"]
        with open(output_path  / "pickupable_map_ids.json", "w") as f:
            json.dump(pickupable_map_ids, f, indent=4)

        # Save results in original format in case we want to plot
        with open(output_path  / "query_results.json", "w") as f:
            json.dump(results, f, indent=4)

    def visualize(self, res_path: str):
        """Visualize all object point clouds with labels using O3DVisualizer."""
        import open3d.visualization.gui as gui

        # 1. Initialize Application and Visualizer
        app = gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer("Map Objects Visualization", 1024, 768)
        vis.set_background([1.0, 1.0, 1.0, 1.0], bg_image=None)
        vis.show_settings = True
        vis.show_skybox(False)
        vis.enable_raw_mode(True)

        # 2. Add Point Clouds
        for i, pcd in enumerate(self.pcd):
            vis.add_geometry(f"pcd_{i}", pcd)

        # 3. Load Results
        with open(res_path / "query_results.json", "r") as f:
            results = json.load(f)

        # 3. Add Receptacles (From ground truth)
        for r_name, r_data in self.receptacles_bbox.items():

            # A. Extract the corners and convert to numpy array
            corners = np.array(r_data["cornerPoints"], dtype=np.float64)

            # create_from_points computes the tightest box around these 8 points
            bbox = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(corners)
            )

            bbox.color = [0, 0, 1]

            vis.add_geometry(f"gt_receptacle_{r_name}", bbox)
            vis.add_3d_label(bbox.get_center(), f" {r_name} ")

        # --- Pre-processing: Group names by map_id ---
        id_to_names = {}
        for r_name, map_id in self.receptacle_map_ids.items():
            if map_id is None:
                continue
            if map_id not in id_to_names:
                id_to_names[map_id] = []
            id_to_names[map_id].append(r_name)

        # 4. Add Receptacles (with Labels)
        for map_id, name_list in id_to_names.items():
            bbox = self.bbox[map_id]
            bbox.color = [0, 0, 0]

            # Add Geometry only once per ID - happens if a receptacle has multiple names
            vis.add_geometry(f"receptacle_{map_id}", bbox)

            # Combine names into one string
            combined_label = "\n".join(name_list)

            # Add the combined label (if it exists)
            vis.add_3d_label(bbox.get_center(), combined_label)

        # 5. Add Pickupables (with Labels)
        for p_name, data in results.items():
            if data["present"]:
                p_id = data["map_object_id"]
                bbox = self.bbox[p_id]
                bbox.color = [1, 0, 0]

                # Add Geometry
                vis.add_geometry(f"pickup_{p_id}", bbox)

                # Add Label
                vis.add_3d_label(bbox.get_center(), f" {p_name} ")

        # 6. Run Visualization
        vis.reset_camera_to_default()
        app.add_window(vis)
        app.run()


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
                    bboxes.append(
                        {
                            "center": center.tolist(),
                            "rotation": rotation.flatten().tolist(),
                            "extent": extent.tolist(),
                            "vertices": [v.tolist() for v in vertices],
                        }
                    )

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
        with open(res_path, "r") as f:
            results = json.load(f)

        # Get pickupables
        receptacle_ids = []
        for _, val in results.items():
            if val["present"]:
                receptacle_ids.extend(val["map_object_id"])

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
