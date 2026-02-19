from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import logging

import numpy as np
import open3d as o3d

log = logging.getLogger(__name__)


@dataclass
class ConceptGraphData:
    """Container for concept graph map artifacts."""

    map_path: Path
    pcd_o3d: list[o3d.geometry.PointCloud]
    segments_anno: dict[str, Any] | None
    dense_points: np.ndarray | None
    dense_colors: np.ndarray | None
    arc_states: list[dict] | None
    map_meta: dict[str, Any] | None

    @property
    def num_objects(self) -> int:
        return len(self.pcd_o3d)

    @classmethod
    def load(cls, map_path: str | Path, *, arcs_enabled: bool = True) -> "ConceptGraphData":
        path = Path(map_path)

        pcd = cls._load_sparse_cloud(path)
        segments_anno = cls._load_segments_annotations(path)
        pcd_o3d = cls._split_segments(pcd, segments_anno)

        dense_points, dense_colors = cls._load_dense_cloud(path)
        arc_states = cls._load_arc_states(path, arcs_enabled=arcs_enabled)
        map_meta = cls._load_map_meta(path)

        return cls(
            map_path=path,
            pcd_o3d=pcd_o3d,
            segments_anno=segments_anno,
            dense_points=dense_points,
            dense_colors=dense_colors,
            arc_states=arc_states,
            map_meta=map_meta,
        )

    @staticmethod
    def _load_map_meta(path: Path) -> dict[str, Any] | None:
        meta_path = path / "meta.json"
        if not meta_path.exists():
            return None

        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load map metadata from %s: %s", meta_path, exc)
            return None

        if not isinstance(meta, dict):
            log.warning("Invalid map metadata format at %s", meta_path)
            return None

        return meta

    @staticmethod
    def _load_sparse_cloud(path: Path) -> o3d.geometry.PointCloud | None:
        sparse_path = path / "point_cloud.pcd"
        if not sparse_path.exists():
            log.warning("Segment point cloud not found at %s", sparse_path)
            return None

        pcd = o3d.io.read_point_cloud(str(sparse_path))
        if pcd.is_empty():
            log.warning("Segment point cloud is empty at %s", sparse_path)
            return None

        return pcd

    @staticmethod
    def _load_dense_cloud(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
        dense_path = path / "dense_point_cloud.pcd"
        if not dense_path.exists():
            log.warning("Dense point cloud not found at %s", dense_path)
            return None, None

        dense_pcd = o3d.io.read_point_cloud(str(dense_path))
        if dense_pcd.is_empty():
            log.warning("Dense point cloud is empty at %s", dense_path)
            return None, None
        dense_points = np.asarray(dense_pcd.points).astype(np.float32)
        dense_colors = np.asarray(dense_pcd.colors).astype(np.float32)
        if dense_colors.size == 0 or dense_colors.shape[0] != dense_points.shape[0]:
            dense_colors = None
        return dense_points, dense_colors

    @staticmethod
    def _load_segments_annotations(path: Path) -> dict[str, Any] | None:
        segments_path = path / "segments_anno.json"
        if not segments_path.exists():
            log.warning("Segment annotations not found at %s", segments_path)
            return None

        try:
            with open(segments_path, "r") as f:
                segments_anno = json.load(f)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed to load segment annotations from %s: %s", segments_path, exc)
            return None

        if not isinstance(segments_anno, dict):
            log.warning("Invalid segments_anno format at %s", segments_path)
            return None

        return segments_anno

    @staticmethod
    def _load_arc_states(path: Path, *, arcs_enabled: bool) -> list[dict] | None:
        if not arcs_enabled:
            log.info("Arcs disabled in config, skipping arc loading")
            return None

        arcs_path = path / "arcs.json"
        if not arcs_path.exists():
            log.info("No arcs.json found at %s", arcs_path)
            return None

        with open(arcs_path, "r") as f:
            arc_states = json.load(f)

        total_arcs = sum(len(state.get("arcs", [])) for state in arc_states) if arc_states else 0
        log.info("Loaded %s arc states with %s total arcs from %s", len(arc_states), total_arcs, arcs_path)
        return arc_states

    @staticmethod
    def _split_segments(
        pcd: o3d.geometry.PointCloud | None,
        segments_anno: dict[str, Any] | None,
    ) -> list[o3d.geometry.PointCloud]:
        if pcd is None:
            return []

        if not segments_anno:
            return [pcd]

        seg_groups = segments_anno.get("segGroups", [])
        if not seg_groups:
            return [pcd]

        pcd_o3d = []
        for ann in seg_groups:
            obj = pcd.select_by_index(ann.get("segments", []))
            if not obj.is_empty():
                pcd_o3d.append(obj)

        if not pcd_o3d:
            return [pcd]

        return pcd_o3d
