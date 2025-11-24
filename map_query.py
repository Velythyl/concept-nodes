import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
import glob
from natsort import natsorted

import hydra
import numpy as np
import torch
import cv2
from omegaconf import DictConfig
from tqdm import tqdm

from concept_graphs.mapping.similarity.semantic import CosineSimilarity01
from concept_graphs.vlm.OpenAIVerifier import OpenAIVerifier
from concept_graphs.perception.ft_extraction.FeatureExtractor import FeatureExtractor

log = logging.getLogger(__name__)

class MapQueryEngine:
    def __init__(
        self, 
        map_path: str, 
        ft_extractor: FeatureExtractor,
        verifier: OpenAIVerifier,
        semantic_sim_metric: CosineSimilarity01,
        device: str = "cuda"
    ):
        self.map_path = Path(map_path)
        self.ft_extractor = ft_extractor
        self.verifier = verifier
        self.semantic_sim = semantic_sim_metric
        
        self.annotations = self._load_annotations()
        self.features = self._load_features()
        self.device = device
        
        log.info(f"Map loaded from {self.map_path}. Found {len(self.annotations)} objects.")

    def _load_annotations(self) -> List[Dict]:
        anno_path = self.map_path / "segments_anno.json"
        
        with open(anno_path, "r") as f:
            data = json.load(f)
        return data.get("segGroups", [])

    def _load_features(self) -> torch.Tensor:
        feat_path = self.map_path / "clip_features.npy"
        
        # Load numpy and convert to torch for the similarity class
        feats_np = np.load(feat_path)
        return torch.from_numpy(feats_np).float()

    def get_object_images(self, object_id: int, limit: int = 5) -> List[np.ndarray]:
        """
        Loads RGB images for a specific object ID from the directory structure.
        Structure: map_dir/segments/{id}/rgb/*.png
        """
        rgb_path = self.map_path / "segments" / str(object_id) / "rgb" / "*.png"

        # Get all pngs, sort them to ensure consistent order
        image_files = natsorted(glob.glob(str(rgb_path)))
        
        
        images = []
        for img_p in image_files[:limit]:
            img = cv2.imread(str(img_p))
            images.append(img)
        
        return images

    def find_receptacle(self, object_centroid: List[float], receptacles: Dict) -> Optional[str]:
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
            dist_sq = (obj_x - c["x"])**2 + (obj_y - c["y"])**2 + (obj_z - c["z"])**2

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_rec = rec_name
        
        return closest_rec

    def process_queries(self, queries: List[str], receptacles_bbox: Dict, top_k: int = 5) -> Dict:
        results = {}
        
        log.info("Encoding queries...")
        # Encode queries -> (num_queries, feature_dim)
        text_features = self.ft_extractor.encode_text(queries).cpu()

        # 2. Calculate Similarity Matrix: (num_queries, num_map_objects)
        
        for i, query_text in tqdm(enumerate(queries), total=len(queries), desc="Processing Queries"):
            query_feat = text_features[i].unsqueeze(0) # (1, dim)
            
            # Calculate similarity against all map objects
            # sim shape: (1, num_objects) -> squeeze to (num_objects)
            query_feat = query_feat.to(self.device)
            features = self.features.to(self.device)
            sim_scores = self.semantic_sim(query_feat, features).squeeze().cpu()
            
            # Get Top K indices
            top_k_indices = torch.argsort(sim_scores, descending=True)[:top_k]
            top_k_indices = top_k_indices.cpu().numpy()

            found = False
            found_details = {
                "present": False,
                "receptacle": None,
                "timestamps": []
            }

            # 3. Verify candidates
            for idx in top_k_indices:
                idx = int(idx) # Ensure python int
                obj_data = self.annotations[idx]
                
                # Get images
                images = self.get_object_images(idx, limit=self.verifier.max_images)
                if len(images) == 0:
                    log.warning(f"No images found for object ID {idx}. Skipping.")
                    continue

                # Verify
                is_match = self.verifier(images, query_text)
                
                if is_match:
                    found = True
                    
                    # 4. Spatial Association
                    centroid = obj_data["centroid"]
                    rec_name = self.find_receptacle(centroid, receptacles_bbox)
                    
                    found_details = {
                        "present": True,
                        "receptacle": rec_name,
                        "timestamps": obj_data.get("timestamps", []),
                        "map_object_id": idx
                    }
                    
                    break
            
            results[query_text] = found_details

        return results


@hydra.main(version_base=None, config_path="conf", config_name="map_query")
def main(cfg: DictConfig):
    
    # Instantiate hydra objects
    dataset = hydra.utils.instantiate(cfg.dataset)
    ft_extractor = hydra.utils.instantiate(cfg.ft_extraction, device=cfg.device)
    verifier = hydra.utils.instantiate(cfg.vlm_verifier)
    
    # Get inputs
    pickupable_names = dataset.get_pickupable_names()
    receptacles_bbox = dataset.get_receptacles_bbox()
    
    map_path = cfg.map_path
        
    semantic_sim = CosineSimilarity01()

    # Initialize Engine
    engine = MapQueryEngine(
        map_path=map_path,
        ft_extractor=ft_extractor,
        verifier=verifier,
        semantic_sim_metric=semantic_sim
    )

    # Run Query Logic
    results = engine.process_queries(
        queries=pickupable_names, 
        receptacles_bbox=receptacles_bbox,
        top_k=cfg.top_k
    )

    # Save Results
    output_path = Path(map_path) / "query_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    
    log.info(f"Results saved to {output_path}")

if __name__ == "__main__":
    main()