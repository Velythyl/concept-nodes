import json
import logging
from pathlib import Path
from typing import List, Dict
from natsort import natsorted
import glob

import numpy as np
import torch
import cv2

from concept_graphs.mapping.similarity.semantic import CosineSimilarity01
from concept_graphs.perception.ft_extraction.FeatureExtractor import FeatureExtractor
from concept_graphs.utils import load_point_cloud

log = logging.getLogger(__name__)


class BaseMapEngine:
    def __init__(
        self,
        map_path: str,
        ft_extractor: FeatureExtractor,
        semantic_sim_metric: CosineSimilarity01,
        device: str = "cuda",
    ):
        self.map_path = Path(map_path)
        self.ft_extractor = ft_extractor
        self.semantic_sim = semantic_sim_metric

        self.annotations = self._load_annotations()
        self.features = self._load_features()
        self.pcd = self._load_point_clouds()
        self.bbox = [pcd.get_oriented_bounding_box() for pcd in self.pcd]
        self.device = device

        log.info(
            f"Map loaded from {self.map_path}. Found {len(self.annotations)} objects."
        )

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

    def _load_point_clouds(self) -> List[np.ndarray]:
        return load_point_cloud(self.map_path)

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

    def process_queries(self, queries: List[str], **kwargs) -> Dict:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def save_results(self, results, output_path, **kwargs):
        """Save results to a JSON file."""
        with open(output_path, "w") as f:
            json.dump(results, f, indent=4)
