import logging
from typing import List, Dict, Optional
import json

import torch
from tqdm import tqdm

from concept_graphs.vlm.OpenAIVerifier import OpenAIVerifier
from concept_graphs.utils import split_camel_preserve_acronyms

from .BaseMapEngine import BaseMapEngine

log = logging.getLogger(__name__)


class MapQueryObjects(BaseMapEngine):
    def __init__(
        self, verifier: OpenAIVerifier, receptacles_bbox: Dict, top_k: int = 5, **kwargs
    ):
        super().__init__(**kwargs)
        self.verifier = verifier
        self.receptacles_bbox = receptacles_bbox
        self.top_k = top_k

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

            # Calculate squared Euclidean distance (faster than sqrt)
            dist_sq = (
                (obj_x - c["x"]) ** 2 + (obj_y - c["y"]) ** 2 + (obj_z - c["z"]) ** 2
            )

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_rec = rec_name

        return closest_rec

    def process_queries(self, queries: List[str], **kwargs) -> Dict:
        results = {}

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
            # sim shape: (1, num_objects) -> squeeze to (num_objects)
            query_feat = query_feat.to(self.device)
            features = self.features.to(self.device)
            sim_scores = self.semantic_sim(query_feat, features).squeeze().cpu()

            # Get Top K indices
            top_k_indices = torch.argsort(sim_scores, descending=True)[:self.top_k]
            top_k_indices = top_k_indices.cpu().numpy()

            found_details = {
                "present": False,
                "receptacle": None,
                "timestamps": [],
                "map_object_id": None,
            }

            # 3. Verify candidates
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

                    # 4. Spatial Association
                    centroid = obj_data["centroid"]
                    rec_name = self.find_receptacle(centroid, self.receptacles_bbox)

                    found_details = {
                        "present": True,
                        "receptacle": rec_name,
                        "timestamps": obj_data.get("timestamps", []),
                        "map_object_id": idx,
                    }

                    break

            results[query_text] = found_details

        return results
