import sys
from pathlib import Path

# Add parent directory to path so imports work when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import hydra
import torch
from omegaconf import DictConfig
import logging
import os
import json
import hashlib
import tempfile
import gc
from functools import cached_property
from http.server import SimpleHTTPRequestHandler
import socketserver

import numpy as np
import open3d as o3d
import time
import threading

import openai
from openai import OpenAI
import viser
from typing import Any

from concept_graphs.floor_segment import segment_floor_points
from concept_graphs.utils import set_seed
from concept_graphs.viz.utils import similarities_to_rgb
from concept_graphs.mapping.similarity.semantic import CosineSimilarity01
from agent import ChatAgent
from chat_history import ChatManager
from vis.arcs_gui import ArcGUIController, setup_arcs_gui
from vis.cg_dataclass import ConceptGraphData
from vis.gui_fsm import (
    GUIStateController,
    setup_project_selector_gui,
    setup_toggle_gui,
    setup_visualization_gui,
)
from vis.notification_manager import NotificationManager
from vis.queries_and_chat import (
    QueriesAndChatController,
    setup_agent_gui,
    setup_clip_query_gui,
    setup_llm_query_gui,
)
from vis.box_annotator import BoxAnnotatorController, setup_box_annotator_gui

# A logger for this file
log = logging.getLogger(__name__)

LLM_SYSTEM_PROMPT = (
    "You are a retrieval assistant for a 3D scene. Given an abstract user request and a set of"
    " objects described by an id, label, and caption, return up to three object ids that best"
    " satisfy the request. Choose only from the provided ids, sort them by relevance, and return"
    " a JSON object shaped as {\"object_ids\": [id1, id2, id3]}. Return an empty list when nothing fits."
)

SIMILARITY_CMAP = "RdYlGn"


class ViserCallbackManager:
    """Manages point cloud visualizations and callbacks for Viser."""

    def __init__(
        self,
        server: viser.ViserServer,
        pcd_o3d,
        clip_ft,
        ft_extractor=None,
        segments_anno=None,
        dense_points=None,
        dense_colors=None,
        map_path: Path | None = None,
        axes_ordering="xyz",
        floor_axis: str = "z",
        map_meta: dict[str, Any] | None = None,
        llm_client: OpenAI | None = None,
        llm_model: str = "gpt-4o-mini",
        llm_system_prompt: str | None = None,
        arc_states: list[dict] | None = None,
    ):
        self.server = server
        self.ft_extractor = ft_extractor  # Might be None initially
        self.point_size = 0.023
        self.point_count = 1.0
        self.default_axes_ordering = axes_ordering
        self.default_floor_axis = floor_axis
        self.axes_ordering = axes_ordering
        self.floor_axis = floor_axis
        self._apply_map_meta(map_meta)
        self.map_path = Path(map_path) if map_path is not None else None
        self.floor_pcd_path = self.map_path / "floor.pcd" if self.map_path is not None else None
        self.clip_features_file_available = (
            self.map_path is not None and (self.map_path / "clip_features.npy").exists()
        )
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.llm_system_prompt = llm_system_prompt or LLM_SYSTEM_PROMPT
        
        # Store arc states (list of ArcState dicts, each with 'arcs' list)
        self.arc_states = arc_states or []
        self.current_arc_state_index = 0  # Index into arc_states list

        # Store point cloud data
        self.pcd = pcd_o3d or []
        self.num_objects = len(self.pcd)
        self._bootstrap_dense_points = (
            None if dense_points is None else np.asarray(dense_points).astype(np.float32)
        )
        self._bootstrap_dense_colors = (
            None if dense_colors is None else np.asarray(dense_colors).astype(np.float32)
        )
        self.dense_points = None
        self.dense_colors = None
        self.raw_dense_points = None
        self.raw_dense_colors = None
        self.active_dense_cloud_path: Path | None = None
        self.floor_points = None
        self.floor_colors = None
        self.axes_indices = [0, 1, 2]
        self.axes_signs = np.ones((3,), dtype=np.float32)

        # Extract points and colors from Open3D point clouds
        self.points_list = [np.asarray(p.points).astype(np.float32) for p in self.pcd]
        self.og_colors = [np.asarray(p.colors).astype(np.float32) for p in self.pcd]
        
        # Apply axes transformation (supports signed axes like "-x-y-z" or "-xzy")
        if self.axes_ordering != "xyz":
            try:
                indices, signs = self._parse_signed_axes(self.axes_ordering)
                self.axes_indices = indices
                self.axes_signs = signs
                # Transform points
                self.points_list = [pts[:, indices] * signs for pts in self.points_list]
                # Overwrite self.pcd with transformed point clouds
                self.pcd = self._transform_pcd(self.pcd, indices, signs)
            except (KeyError, IndexError, ValueError):
                log.warning("Invalid axes_ordering '%s', using 'xyz'", self.axes_ordering)

        self._rebuild_dense_for_display()
        self._save_runtime_meta()

        # Compute centroids and bounding boxes
        self.centroids = [np.mean(pts, axis=0) for pts in self.points_list]
        self.bboxes = self._compute_bboxes()

        # Generate random colors for each object
        self.random_colors = np.random.rand(self.num_objects, 3).astype(np.float32)

        # Similarity data
        self.sim_query = 0.5 * np.ones(self.num_objects)

        # Semantic features (initialize on CPU; move later if extractor loads async)
        self.semantic_sim = CosineSimilarity01()
        if clip_ft is None:
            clip_ft = np.zeros((self.num_objects, 1), dtype=np.float32)
        clip_ft = np.asarray(clip_ft, dtype=np.float32)
        if clip_ft.ndim == 1:
            clip_ft = clip_ft[None, :]
        if clip_ft.ndim != 2:
            clip_ft = clip_ft.reshape(clip_ft.shape[0], -1)
        self.semantic_tensor = torch.from_numpy(clip_ft).float()

        # Capability flags used to gate UI and behavior when artifacts are missing
        self.has_segment_objects = self.num_objects > 0
        self.has_dense_cloud = self.dense_points is not None and len(self.dense_points) > 0
        self.has_clip_features = (
            self.semantic_tensor.ndim == 2
            and self.semantic_tensor.shape[0] == self.num_objects
            and self.semantic_tensor.shape[1] > 0
            and self.semantic_tensor.numel() > 0
            and self.has_segment_objects
        )
        self.can_run_clip_query = (
            self.has_segment_objects
            and self.has_clip_features
            and self.clip_features_file_available
        )
        self.can_run_llm_query = self.has_segment_objects and self.llm_client is not None
        self.can_show_arcs = bool(self.arc_states) and self.has_segment_objects

        # If extractor was provided up front, finish setup now
        if self.ft_extractor is not None:
            self.set_ft_extractor(self.ft_extractor)

        # Track current visualization handles
        self.pcd_handles = []
        self.bbox_handles = []
        self.centroid_handles = []
        self.id_handles = []
        self.caption_handles = []
        # Per-object label handles keyed by object id (as string)
        self.object_labels_dict: dict[str, Any] = {}

        # Toggle states
        self.bbox_visible = False
        self.centroid_visible = False
        self.ids_visible = False
        self.labels_visible = False
        self.captions_visible = False

        # Stacking mode states (checkboxes)
        self.rgb_enabled = self.has_segment_objects
        self.random_enabled = False
        self.similarity_enabled = False
        self.dense_enabled = self.has_dense_cloud and not self.has_segment_objects
        self.segment_floor_enabled = False
        self.llm_palette_active = False
        self.llm_palette_colors = None

        # Point size for point clouds (shared across layers)
        self.point_noise_enabled = True

        # Small epsilon for point offset to prevent z-fighting
        self.EPS = 0.001 #1e-5

        # Store segment annotations for labels and captions
        self.segments_anno = segments_anno

        # Chat agent (initialized later via set_chat_agent)
        self.chat_agent: ChatAgent | None = None

        # Controllers for GUI state, arcs, and query/chat logic
        self.gui_fsm = GUIStateController(self)
        self.arcs_gui = ArcGUIController(self)
        self.query_chat = QueriesAndChatController(self)

        self.notifications = NotificationManager(server=server)

        # Box annotator controller
        # Gather all points into a single (N, 3) array for ray-casting
        all_pts = np.concatenate(self.points_list, axis=0) if self.points_list else None
        self.box_annotator = BoxAnnotatorController(
            server=server,
            all_points=all_pts,
            dense_points=self.dense_points,
            notification_manager=self.notifications,
        )

    @staticmethod
    def _parse_signed_axes(ordering: str) -> tuple[list[int], np.ndarray]:
        """Parse axes_ordering like '-xzy' into indices and signs."""
        axis_map = {'x': 0, 'y': 1, 'z': 2}
        signed = []
        sign = 1
        for ch in ordering:
            if ch == "-":
                sign = -1
                continue
            idx = axis_map[ch.lower()]
            signed.append((idx, sign))
            sign = 1
        if len(signed) != 3:
            raise ValueError("axes_ordering must specify exactly 3 axes")
        indices = [idx for idx, _ in signed]
        signs = np.array([sgn for _, sgn in signed], dtype=np.float32)
        return indices, signs

    def _clear_cached_text_fields(self):
        """Clear cached text-derived properties that depend on segments annotations."""
        self.__dict__.pop("labels", None)
        self.__dict__.pop("captions", None)
        self.__dict__.pop("search_corpus", None)

    def _apply_map_meta(self, map_meta: dict[str, Any] | None):
        """Set map axis defaults from metadata (if available)."""
        if map_meta is None:
            self.axes_ordering = self.default_axes_ordering
            self.floor_axis = self.default_floor_axis
            return

        axes_ordering = map_meta.get("axes_ordering", self.default_axes_ordering)
        floor_axis = map_meta.get("floor_axis", self.default_floor_axis)
        point_size = map_meta.get("point_size", self.point_size)
        point_count = map_meta.get("point_count", self.point_count)

        self.axes_ordering = str(axes_ordering)
        self.floor_axis = str(floor_axis)
        self.point_size = self._clamp_point_size(point_size)
        self.point_count = self._clamp_point_count(point_count)

    @staticmethod
    def _clamp_point_size(value: float | int) -> float:
        try:
            return float(np.clip(float(value), 0.001, 0.05))
        except (TypeError, ValueError):
            return 0.023

    @staticmethod
    def _clamp_point_count(value: float | int) -> float:
        try:
            return float(np.clip(float(value), 0.0, 1.0))
        except (TypeError, ValueError):
            return 1.0

    def _save_runtime_meta(self):
        """Persist visualization state into map_path/meta.json."""
        if self.map_path is None:
            return

        meta_path = self.map_path / "meta.json"
        meta: dict[str, Any] = {}
        if meta_path.exists():
            try:
                with open(meta_path, "r") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    meta = loaded
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed reading existing map metadata at %s: %s", meta_path, exc)

        meta["axes_ordering"] = str(self.axes_ordering)
        meta["floor_axis"] = str(self.floor_axis)
        meta["point_size"] = float(self.point_size)
        meta["point_count"] = float(self.point_count)

        try:
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed writing map metadata at %s: %s", meta_path, exc)

    def _dense_tmp_path(self) -> Path:
        """Build a deterministic tmp path for downsampled dense clouds."""
        map_key = "unknown"
        if self.map_path is not None:
            map_key = str(self.map_path.resolve())
        digest = hashlib.sha1(map_key.encode("utf-8")).hexdigest()[:12]
        pct = int(round(self.point_count * 1000.0))
        name = f"cg_dense_{digest}_{pct:04d}.pcd"
        return Path(tempfile.gettempdir()) / name

    def _load_full_dense_source(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Load full dense cloud from disk, with bootstrap fallback during initialization."""
        if self.map_path is not None:
            dense_path = self.map_path / "dense_point_cloud.pcd"
            if dense_path.exists():
                try:
                    dense_pcd = o3d.io.read_point_cloud(str(dense_path))
                    if not dense_pcd.is_empty():
                        points = np.asarray(dense_pcd.points).astype(np.float32)
                        colors = np.asarray(dense_pcd.colors).astype(np.float32) if dense_pcd.has_colors() else None
                        if colors is not None and len(colors) != len(points):
                            colors = None
                        return points, colors
                    log.warning("Dense point cloud is empty at %s; falling back to bootstrap data", dense_path)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Failed reading dense point cloud from %s: %s", dense_path, exc)

        if self._bootstrap_dense_points is None or len(self._bootstrap_dense_points) == 0:
            return None, None
        colors = self._bootstrap_dense_colors
        if colors is not None and len(colors) != len(self._bootstrap_dense_points):
            colors = None
        return self._bootstrap_dense_points, colors

    def _clear_display_dense_cloud(self):
        """Drop currently displayed dense cloud arrays to release memory."""
        self.raw_dense_points = None
        self.raw_dense_colors = None
        self.dense_points = None
        self.dense_colors = None
        self.active_dense_cloud_path = None
        self.floor_points = None
        self.floor_colors = None

    def _rebuild_dense_for_display(self):
        """Build the currently displayed dense cloud from full dense cloud and point_count."""
        self._clear_display_dense_cloud()

        source_points, source_colors = self._load_full_dense_source()
        if source_points is None or len(source_points) == 0:
            self._bootstrap_dense_points = None
            self._bootstrap_dense_colors = None
            return

        raw_points = source_points
        raw_colors = source_colors

        n_points = len(raw_points)
        if self.point_count < 1.0:
            keep = max(1, int(np.floor(self.point_count * n_points)))
            indices = np.linspace(0, n_points - 1, num=keep, dtype=np.int64)
            selected_points = raw_points[indices]
            selected_colors = raw_colors[indices] if raw_colors is not None else None

            tmp_path = self._dense_tmp_path()
            tmp_pcd = o3d.geometry.PointCloud()
            tmp_pcd.points = o3d.utility.Vector3dVector(selected_points.astype(np.float64, copy=False))
            if selected_colors is not None and len(selected_colors) == len(selected_points):
                tmp_pcd.colors = o3d.utility.Vector3dVector(selected_colors.astype(np.float64, copy=False))
            if not o3d.io.write_point_cloud(str(tmp_path), tmp_pcd):
                log.warning("Failed writing downsampled dense cloud to %s", tmp_path)

            raw_points = selected_points
            raw_colors = selected_colors
            self.active_dense_cloud_path = tmp_path
        else:
            self.active_dense_cloud_path = (
                self.map_path / "dense_point_cloud.pcd" if self.map_path is not None else None
            )

        self.raw_dense_points = raw_points
        self.raw_dense_colors = raw_colors
        self.dense_points = self._apply_axes_transform(raw_points)
        if raw_colors is not None and len(raw_colors) == len(raw_points):
            self.dense_colors = raw_colors
        else:
            self.dense_colors = None

        self.floor_points = None
        self.floor_colors = None

        self._bootstrap_dense_points = None
        self._bootstrap_dense_colors = None
        del source_points
        del source_colors
        gc.collect()

    def _load_map_data(self, cg: ConceptGraphData, map_path: Path):
        """Load map-dependent state into this manager without recreating GUI/controller objects."""
        self.map_path = Path(map_path)
        self._apply_map_meta(cg.map_meta)
        self.floor_pcd_path = self.map_path / "floor.pcd"
        self.clip_features_file_available = (self.map_path / "clip_features.npy").exists()
        self.arc_states = cg.arc_states or []
        self.current_arc_state_index = 0

        self.pcd = cg.pcd_o3d or []
        self.num_objects = len(self.pcd)
        self._bootstrap_dense_points = (
            None if cg.dense_points is None else np.asarray(cg.dense_points).astype(np.float32)
        )
        self._bootstrap_dense_colors = (
            None if cg.dense_colors is None else np.asarray(cg.dense_colors).astype(np.float32)
        )
        self.dense_points = None
        self.dense_colors = None
        self.raw_dense_points = None
        self.raw_dense_colors = None
        self.active_dense_cloud_path = None
        self.floor_points = None
        self.floor_colors = None
        self.axes_indices = [0, 1, 2]
        self.axes_signs = np.ones((3,), dtype=np.float32)

        self.points_list = [np.asarray(p.points).astype(np.float32) for p in self.pcd]
        self.og_colors = [np.asarray(p.colors).astype(np.float32) for p in self.pcd]

        if self.axes_ordering != "xyz":
            try:
                indices, signs = self._parse_signed_axes(self.axes_ordering)
                self.axes_indices = indices
                self.axes_signs = signs
                self.points_list = [pts[:, indices] * signs for pts in self.points_list]
                self.pcd = self._transform_pcd(self.pcd, indices, signs)
            except (KeyError, IndexError, ValueError):
                log.warning("Invalid axes_ordering '%s', using 'xyz'", self.axes_ordering)

        self._rebuild_dense_for_display()
        self._save_runtime_meta()

        self.centroids = [np.mean(pts, axis=0) for pts in self.points_list]
        self.bboxes = self._compute_bboxes()
        self.random_colors = np.random.rand(self.num_objects, 3).astype(np.float32)
        self.sim_query = 0.5 * np.ones(self.num_objects)
        self.llm_palette_active = False
        self.llm_palette_colors = None

        clip_ft = cg.clip_features
        if clip_ft is None:
            clip_ft = np.zeros((self.num_objects, 1), dtype=np.float32)
        clip_ft = np.asarray(clip_ft, dtype=np.float32)
        if clip_ft.ndim == 1:
            clip_ft = clip_ft[None, :]
        if clip_ft.ndim != 2:
            clip_ft = clip_ft.reshape(clip_ft.shape[0], -1)
        self.semantic_tensor = torch.from_numpy(clip_ft).float()

        self.has_segment_objects = self.num_objects > 0
        self.has_dense_cloud = self.dense_points is not None and len(self.dense_points) > 0
        self.has_clip_features = (
            self.semantic_tensor.ndim == 2
            and self.semantic_tensor.shape[0] == self.num_objects
            and self.semantic_tensor.shape[1] > 0
            and self.semantic_tensor.numel() > 0
            and self.has_segment_objects
        )
        self.can_run_clip_query = (
            self.has_segment_objects
            and self.has_clip_features
            and self.clip_features_file_available
        )
        self.can_run_llm_query = self.has_segment_objects and self.llm_client is not None
        self.can_show_arcs = bool(self.arc_states) and self.has_segment_objects

        self.segments_anno = cg.segments_anno
        self._clear_cached_text_fields()

        all_pts = np.concatenate(self.points_list, axis=0) if self.points_list else None
        self.box_annotator.all_points = all_pts
        self.box_annotator.dense_points = self.dense_points
        self.box_annotator.set_cg_bboxes(self.bboxes)

    def switch_to_map(self, map_path: Path, cfg: DictConfig):
        """Switch to another saved map in-process and refresh scene/UI."""
        self.gui_fsm.switch_to_map(map_path, cfg)

    def notify_clients(self, title: str, body: str, *, client=None, **kwargs):
        """Send a notification to one client or all connected clients."""
        self.notifications.notify(title=title, body=body, client=client, **kwargs)

    def set_chat_agent(self, chat_agent: ChatAgent):
        """Set the ChatAgent instance for handling chat messages."""
        self.query_chat.set_chat_agent(chat_agent)
    
    def get_current_timestep_from_slider(self) -> tuple[int, float]:
        """Get the current day and hour from the arc state slider."""
        return self.arcs_gui.get_current_timestep_from_slider()

    def set_ft_extractor(self, ft_extractor):
        """Called when the feature extractor finishes loading in the background."""
        self.query_chat.set_ft_extractor(ft_extractor)

    def set_llm_client(self, llm_client: OpenAI, llm_model: str | None = None):
        """Attach the OpenAI client once credentials are available."""
        self.query_chat.set_llm_client(llm_client, llm_model)

    def offset_by_eps(self, points: np.ndarray) -> np.ndarray:
        """Add small random offsets to points to prevent z-fighting.
        
        Samples N points uniformly from [-EPS, +EPS] and adds them to the input.
        
        Args:
            points: Input point cloud of shape (N, 3)
            
        Returns:
            Point cloud with small random offsets applied
        """
        if not self.point_noise_enabled:
            return points
        offsets = np.random.uniform(-self.EPS, self.EPS, 
            size=points.shape
        ).astype(np.float32)
        return points + offsets

    def _compute_bboxes(self):
        """Compute oriented bounding boxes for each point cloud."""
        bboxes = []
        for pcd in self.pcd:
            try:
                obb = pcd.get_oriented_bounding_box()
                center = np.array(obb.center)
                extent = np.array(obb.extent)
                rotation = np.array(obb.R)
                bboxes.append({
                    "center": center,
                    "extent": extent,
                    "rotation": rotation,
                })
            except Exception:
                # Fallback to axis-aligned bbox
                aabb = pcd.get_axis_aligned_bounding_box()
                center = np.array(aabb.get_center())
                extent = np.array(aabb.get_extent())
                bboxes.append({
                    "center": center,
                    "extent": extent,
                    "rotation": np.eye(3),
                })
        return bboxes
    
    def _transform_pcd(self, pcd_list, indices, signs):
        """Transform Open3D point clouds with axis reordering and sign flipping.
        
        Creates new point clouds with transformed points while preserving colors.
        
        Args:
            pcd_list: List of Open3D point clouds
            indices: Axis indices for reordering (e.g., [0, 2, 1] for xyz)
            signs: Sign flips as numpy array of shape (3,) with values ±1
            
        Returns:
            List of new Open3D point clouds with transformed points
        """
        transformed_pcd = []
        for pcd in pcd_list:
            # Transform points
            points = np.asarray(pcd.points).astype(np.float32)
            points_transformed = points[:, indices] * signs
            
            # Create new point cloud with transformed points
            new_pcd = o3d.geometry.PointCloud()
            new_pcd.points = o3d.utility.Vector3dVector(points_transformed)
            
            # Preserve colors if they exist
            if pcd.has_colors():
                colors = np.asarray(pcd.colors).astype(np.float32)
                new_pcd.colors = o3d.utility.Vector3dVector(colors)
            
            transformed_pcd.append(new_pcd)
        
        return transformed_pcd

    def _apply_axes_transform(self, points: np.ndarray) -> np.ndarray:
        """Apply the configured axis permutation/sign transform to points."""
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        return points[:, self.axes_indices] * self.axes_signs

    def _set_floor_cloud_from_raw(self, raw_points: np.ndarray, raw_colors: np.ndarray | None):
        """Store floor cloud in visualization coordinates for rendering."""
        self.floor_points = self._apply_axes_transform(raw_points.astype(np.float32, copy=False))
        if raw_colors is not None and len(raw_colors) == len(raw_points):
            self.floor_colors = np.asarray(raw_colors).astype(np.float32, copy=False)
        else:
            self.floor_colors = np.tile(
                np.array([0.2, 0.8, 0.2], dtype=np.float32),
                (len(raw_points), 1),
            )

    def _load_or_create_floor_cloud(self):
        """Load floor.pcd if present; otherwise segment from dense cloud and save it."""
        if self.floor_points is not None:
            return

        if self.floor_pcd_path is not None and self.floor_pcd_path.exists():
            floor_pcd = o3d.io.read_point_cloud(str(self.floor_pcd_path))
            raw_points = np.asarray(floor_pcd.points).astype(np.float32)
            raw_colors = np.asarray(floor_pcd.colors).astype(np.float32) if floor_pcd.has_colors() else None
            self._set_floor_cloud_from_raw(raw_points, raw_colors)
            return

        if self.dense_points is None or self.raw_dense_points is None:
            log.warning("Cannot segment floor: dense point cloud is not available.")
            self.floor_points = np.zeros((0, 3), dtype=np.float32)
            self.floor_colors = np.zeros((0, 3), dtype=np.float32)
            return

        floor_mask = segment_floor_points(
            self.dense_points,
            axes_ordering=self.axes_ordering,
            floor_axis=self.floor_axis,
        )
        raw_points = self.raw_dense_points[floor_mask]
        raw_colors = self.raw_dense_colors[floor_mask] if self.raw_dense_colors is not None else None

        if self.floor_pcd_path is not None:
            floor_pcd = o3d.geometry.PointCloud()
            floor_pcd.points = o3d.utility.Vector3dVector(raw_points.astype(np.float64, copy=False))
            if raw_colors is not None and len(raw_colors) == len(raw_points):
                floor_pcd.colors = o3d.utility.Vector3dVector(raw_colors.astype(np.float64, copy=False))
            o3d.io.write_point_cloud(str(self.floor_pcd_path), floor_pcd)
            log.info("Saved segmented floor cloud to %s", self.floor_pcd_path)

        self._set_floor_cloud_from_raw(raw_points, raw_colors)
    
    def _extract_text_field(self, field: str, *, fallback_field: str | None = None):
        """Extract a text field from segment annotations with optional fallback."""
        if self.segments_anno is None:
            return ["" for _ in range(self.num_objects)]

        values = []
        for seg_group in self.segments_anno.get("segGroups", []):
            value = seg_group.get(field, "")
            if fallback_field and (not value or value == "empty"):
                value = seg_group.get(fallback_field, "")
            values.append(value)

        return values

    @cached_property
    def labels(self):
        return self._extract_text_field("label", fallback_field="caption")

    @cached_property
    def captions(self):
        return self._extract_text_field("caption")

    @cached_property
    def search_corpus(self):
        corpus = []
        for label, caption in zip(self.labels, self.captions):
            pieces = [label.strip(), caption.strip()]
            combined = " ".join([p for p in pieces if p])
            corpus.append(combined)
        return corpus

    def _resolve_object_id(self, identifier):
        """Resolve an object identifier that may be an index or label."""
        if identifier is None:
            return None
        if isinstance(identifier, int):
            return identifier
        if isinstance(identifier, str):
            candidate = identifier.strip()
            try:
                return int(candidate)
            except ValueError:
                pass
            for idx, label in enumerate(self.labels):
                if label and label.strip().lower() == candidate.lower():
                    return idx
        return None

    def register_chat_markdown(self, markdown_handle):
        """Attach the HTML/markdown handle used for chat history display."""
        self.query_chat.register_chat_markdown(markdown_handle)

    def clear_chat(self):
        """Clear the chat history (both UI display and agent context)."""
        self.query_chat.clear_chat()

    def add_chat_message(self, sender: str, message: str):
        """Append a chat message and refresh the markdown view."""
        self.query_chat.add_chat_message(sender, message)

    def set_cmap_html_handle(self, handle):
        """Store reference to the colormap HTML handle created in setup_gui."""
        self.gui_fsm.set_cmap_html_handle(handle)

    def add_point_clouds(self):
        """Add all point clouds to the scene based on enabled modes."""
        self.remove_point_clouds()
        
        # Collect all enabled layers
        layers = []
        
        if self.dense_enabled and self.dense_points is not None:
            # Dense mode renders separately
            dense_points = self.offset_by_eps(self.dense_points)
            dense_colors = self.dense_colors if self.dense_colors is not None else np.tile(
                np.array([0.6, 0.6, 0.6], dtype=np.float32), (len(dense_points), 1)
            )
            dense_colors_uint8 = (np.clip(dense_colors, 0, 1) * 255).astype(np.uint8)
            handle = self.server.scene.add_point_cloud(
                name="/pointclouds/dense",
                points=dense_points,
                colors=dense_colors_uint8,
                point_size=self.point_size,
                point_shape="circle",
            )
            self.pcd_handles.append(handle)
        
        if self.rgb_enabled:
            layers.append(("rgb", self.og_colors))
        
        if self.random_enabled:
            random_colors = [
                np.tile(self.random_colors[i], (len(self.points_list[i]), 1))
                for i in range(self.num_objects)
            ]
            layers.append(("random", random_colors))
        
        if self.similarity_enabled:
            if self.llm_palette_active and self.llm_palette_colors is not None:
                sim_colors = [
                    np.tile(self.llm_palette_colors[i], (len(self.points_list[i]), 1)).astype(np.float32)
                    for i in range(self.num_objects)
                ]
            else:
                rgb = similarities_to_rgb(self.sim_query, cmap_name=SIMILARITY_CMAP)
                sim_colors = [
                    np.tile(np.array(rgb[i]) / 255.0, (len(self.points_list[i]), 1)).astype(np.float32)
                    for i in range(self.num_objects)
                ]
            layers.append(("similarity", sim_colors))

        if self.segment_floor_enabled:
            try:
                self._load_or_create_floor_cloud()
            except Exception as exc:
                log.warning("Floor segmentation/loading failed: %s", exc)
                self.floor_points = np.zeros((0, 3), dtype=np.float32)
                self.floor_colors = np.zeros((0, 3), dtype=np.float32)

            if self.floor_points is not None and len(self.floor_points) > 0:
                floor_points = self.offset_by_eps(self.floor_points)
                floor_colors = self.floor_colors
                if floor_colors is None or len(floor_colors) != len(floor_points):
                    floor_colors = np.tile(
                        np.array([0.2, 0.8, 0.2], dtype=np.float32),
                        (len(floor_points), 1),
                    )
                floor_colors_uint8 = (np.clip(floor_colors, 0, 1) * 255).astype(np.uint8)
                handle = self.server.scene.add_point_cloud(
                    name="/pointclouds/floor",
                    points=floor_points,
                    colors=floor_colors_uint8,
                    point_size=self.point_size,
                    point_shape="circle",
                )
                self.pcd_handles.append(handle)
        
        # Add each layer with epsilon offset
        for layer_name, colors_list in layers:
            for i, (pts, colors) in enumerate(zip(self.points_list, colors_list)):
                # Offset points to prevent z-fighting between layers
                pts_offset = self.offset_by_eps(pts)
                
                # Expand colors to per-point if uniform
                if colors.ndim == 1:
                    colors_expanded = np.tile(colors, (len(pts), 1))
                else:
                    colors_expanded = colors

                # Convert to uint8 RGB
                colors_uint8 = (np.clip(colors_expanded, 0, 1) * 255).astype(np.uint8)

                handle = self.server.scene.add_point_cloud(
                    name=f"/pointclouds/{layer_name}_pcd_{i}",
                    points=pts_offset,
                    colors=colors_uint8,
                    point_size=self.point_size,
                    point_shape="circle",
                )
                self.pcd_handles.append(handle)

    def remove_point_clouds(self):
        """Remove all point cloud handles."""
        handles = self.pcd_handles
        self.pcd_handles = []
        for handle in handles:
            try:
                handle.remove()
            except KeyError:
                # Scene node already removed by a previous refresh; safe to ignore.
                pass
            except Exception as exc:  # noqa: BLE001
                log.debug("Ignoring point cloud handle removal error: %s", exc)

    def update_point_clouds(self):
        """Update point cloud colors."""
        self.add_point_clouds()

    def add_bboxes(self):
        """Add bounding boxes to the scene."""
        self.remove_bboxes()
        for i, bbox in enumerate(self.bboxes):
            # Create a wireframe box using line segments
            center = bbox["center"]
            extent = bbox["extent"]
            rotation = bbox["rotation"]

            # Create box corners in local coordinates
            half = extent / 2
            corners_local = np.array([
                [-half[0], -half[1], -half[2]],
                [half[0], -half[1], -half[2]],
                [half[0], half[1], -half[2]],
                [-half[0], half[1], -half[2]],
                [-half[0], -half[1], half[2]],
                [half[0], -half[1], half[2]],
                [half[0], half[1], half[2]],
                [-half[0], half[1], half[2]],
            ])

            # Transform to world coordinates
            corners_world = (rotation @ corners_local.T).T + center

            # Define edges (pairs of corner indices)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
                (4, 5), (5, 6), (6, 7), (7, 4),  # top face
                (0, 4), (1, 5), (2, 6), (3, 7),  # vertical edges
            ]

            # Create line segments in shape (N, 2, 3) for N line segments
            points = np.array([
                [corners_world[e1], corners_world[e2]]
                for e1, e2 in edges
            ], dtype=np.float32)

            handle = self.server.scene.add_line_segments(
                name=f"/bboxes/bbox_{i}",
                points=points,
                colors=np.array([0, 0, 0], dtype=np.uint8),  # Black
                line_width=2.0,
            )
            self.bbox_handles.append(handle)

    def remove_bboxes(self):
        """Remove all bounding box handles."""
        for handle in self.bbox_handles:
            handle.remove()
        self.bbox_handles = []

    def add_centroids(self):
        """Add centroid spheres to the scene."""
        self.remove_centroids()
        for i, centroid in enumerate(self.centroids):
            color = self.random_colors[i]
            color_uint8 = (np.clip(color, 0, 1) * 255).astype(np.uint8)

            handle = self.server.scene.add_icosphere(
                name=f"/centroids/centroid_{i}",
                radius=0.03,
                position=centroid.astype(np.float32),
                color=tuple(color_uint8),
            )
            self.centroid_handles.append(handle)

    def remove_centroids(self):
        """Remove all centroid handles."""
        for handle in self.centroid_handles:
            handle.remove()
        self.centroid_handles = []

    def add_ids(self):
        """Add text IDs to the scene."""
        self.remove_ids()
        for i, centroid in enumerate(self.centroids):
            handle = self.server.scene.add_label(
                name=f"/ids/id_{i}",
                text=str(i),
                position=centroid.astype(np.float32) + np.array([0, 0, 0.05], dtype=np.float32),
            )
            self.id_handles.append(handle)

    def remove_ids(self):
        """Remove all id handles."""
        for handle in self.id_handles:
            handle.remove()
        self.id_handles = []

    def remove_labels(self):
        """Remove all label handles."""
        # Remove any per-object labels managed via object_labels_dict
        if self.object_labels_dict:
            for handle in list(self.object_labels_dict.values()):
                try:
                    handle.remove()
                except Exception:
                    pass
            self.object_labels_dict = {}

    def toggle_bbox(self):
        """Toggle bounding box visibility."""
        self.gui_fsm.toggle_bbox()

    def toggle_centroids(self):
        """Toggle centroid visibility."""
        self.gui_fsm.toggle_centroids()

    def toggle_ids(self):
        """Toggle ID visibility."""
        self.gui_fsm.toggle_ids()

    def toggle_labels(self):
        """Toggle label visibility."""
        self.gui_fsm.toggle_labels()

    def add_label_for_object(self, object_id: int, text: str | None = None, offset: np.ndarray | None = None):
        """Add (or replace) a label for a single object.

        Args:
            object_id: Index of the object to label
            text: Optional label text; defaults to `self.labels[object_id]`
            offset: Optional position offset; defaults to small +Z offset

        Returns:
            The created label handle, or None if no label was added
        """
        key = str(object_id)

        # Remove existing per-object label if present
        if key in self.object_labels_dict:
            try:
                self.object_labels_dict[key].remove()
            except Exception:
                pass
            del self.object_labels_dict[key]

        # Resolve text; skip if empty
        label_text = text if text is not None else (
            self.labels[object_id] if 0 <= object_id < self.num_objects else ""
        )
        if not label_text:
            return None

        # Position with default small Z offset
        base_pos = self.centroids[object_id].astype(np.float32)
        if offset is None:
            offset = np.array([0, 0, 0.1], dtype=np.float32)

        handle = self.server.scene.add_label(
            name=f"/labels/label_{object_id}",
            text=label_text,
            position=base_pos + offset,
        )
        self.object_labels_dict[key] = handle
        return handle

    def add_captions(self):
        """Add text captions to the scene."""
        self.remove_captions()
        for i, (centroid, caption) in enumerate(zip(self.centroids, self.captions)):
            if caption:  # Only add if caption is not empty
                handle = self.server.scene.add_label(
                    name=f"/captions/caption_{i}",
                    text=caption,
                    position=centroid.astype(np.float32) + np.array([0, 0, 0.15], dtype=np.float32),
                )
                self.caption_handles.append(handle)

    def remove_captions(self):
        """Remove all caption handles."""
        for handle in self.caption_handles:
            handle.remove()
        self.caption_handles = []

    def toggle_captions(self):
        """Toggle caption visibility."""
        self.gui_fsm.toggle_captions()

    def get_current_arcs(self) -> list[dict]:
        """Get arcs from the currently selected ArcState."""
        return self.arcs_gui.get_current_arcs()

    def get_current_arc_state_info(self) -> str:
        """Get a markdown string describing the current arc state (day/hour)."""
        return self.arcs_gui.get_current_arc_state_info()

    def set_arc_state_index(self, index: int):
        """Set the current arc state index and refresh arc visualization."""
        self.arcs_gui.set_arc_state_index(index)

    def set_arc_state_slider(self, slider):
        """Store reference to the arc state slider."""
        self.arcs_gui.set_arc_state_slider(slider)

    def add_forward_arcs(self):
        """Add forward arcs to the scene using the forward colormap."""
        self.arcs_gui.add_forward_arcs()

    def remove_forward_arcs(self):
        """Remove all forward arc handles."""
        self.arcs_gui.remove_forward_arcs()

    def add_dependency_arcs(self):
        """Add dependency arcs to the scene using the dependency colormap."""
        self.arcs_gui.add_dependency_arcs()

    def remove_dependency_arcs(self):
        """Remove all dependency arc handles."""
        self.arcs_gui.remove_dependency_arcs()

    def remove_arcs(self):
        """Remove all arc handles (both forward and dependency)."""
        self.arcs_gui.remove_arcs()

    def toggle_forward_arcs(self):
        """Toggle forward arc visibility."""
        self.arcs_gui.toggle_forward_arcs()

    def toggle_dependency_arcs(self):
        """Toggle dependency arc visibility."""
        self.arcs_gui.toggle_dependency_arcs()

    def toggle_all_arcs(self):
        """Toggle all arcs (both forward and dependency)."""
        self.arcs_gui.toggle_all_arcs()

    def toggle_rgb_mode(self, enabled: bool):
        """Toggle RGB color mode."""
        self.gui_fsm.toggle_rgb_mode(enabled)

    def toggle_random_mode(self, enabled: bool):
        """Toggle random color mode."""
        self.gui_fsm.toggle_random_mode(enabled)

    def toggle_similarity_mode(self, enabled: bool):
        """Toggle similarity color mode."""
        self.gui_fsm.toggle_similarity_mode(enabled)

    def toggle_dense_mode(self, enabled: bool):
        """Toggle dense point cloud mode."""
        return self.gui_fsm.toggle_dense_mode(enabled)

    def toggle_point_noise(self, enabled: bool):
        """Toggle random point offsets used to reduce z-fighting."""
        self.gui_fsm.toggle_point_noise(enabled)

    def toggle_floor_segment(self, enabled: bool):
        """Toggle floor segmentation overlay mode."""
        self.gui_fsm.toggle_floor_segment(enabled)

    def set_point_size(self, value: float):
        """Set point size and refresh point clouds."""
        self.point_size = self._clamp_point_size(value)
        self._save_runtime_meta()
        self.update_point_clouds()

    def set_point_count(self, value: float):
        """Set dense point fraction, persist metadata, and refresh rendered cloud."""
        self.remove_point_clouds()
        self._clear_display_dense_cloud()
        gc.collect()
        self.point_count = self._clamp_point_count(value)
        self._save_runtime_meta()
        self._rebuild_dense_for_display()
        self.has_dense_cloud = self.dense_points is not None and len(self.dense_points) > 0
        self.box_annotator.dense_points = self.dense_points
        self.add_point_clouds()

    def enable_similarity_mode(self):
        """Enable and check similarity mode after a query."""
        self.gui_fsm.enable_similarity_mode()

    def set_similarity_checkbox(self, checkbox):
        """Store reference to the similarity checkbox for enabling/disabling."""
        self.gui_fsm.set_similarity_checkbox(checkbox)

    def query(self, query_text: str, client=None):
        """Perform CLIP query and update similarity visualization."""
        return self.query_chat.query(query_text, client=client)

    def llm_query(self, query_text: str, client=None):
        """Send query to OpenAI and color top matches."""
        return self.query_chat.llm_query(query_text, client=client)

    def respond(self, message: str) -> str:
        """Answer chat messages using the ChatAgent (if available) or fallback to LLM."""
        return self.query_chat.respond(message)



def setup_gui(server: viser.ViserServer, manager: ViserCallbackManager, cfg: DictConfig):
    """Set up the Viser GUI controls."""
    gui_cfg = cfg.gui
    box_annotator_refresh = None

    if gui_cfg.visualization.visible:
        setup_visualization_gui(server, manager, gui_cfg, SIMILARITY_CMAP)

    if gui_cfg.toggles.visible:
        setup_toggle_gui(server, manager, gui_cfg)

    if cfg.arcs_enabled and gui_cfg.arcs.visible:
        setup_arcs_gui(server, manager, gui_cfg)

    if gui_cfg.clip_query.visible:
        setup_clip_query_gui(server, manager, gui_cfg)

    if gui_cfg.llm_query.visible:
        setup_llm_query_gui(server, manager, gui_cfg)

    if gui_cfg.agent.visible:
        setup_agent_gui(server, manager, gui_cfg)

    # Box annotator panel (always visible)
    box_annotator_cfg = gui_cfg.get("box_annotator", {})
    if box_annotator_cfg.get("visible", True):
        save_dir = Path(cfg.get("map_path", "."))
        # Provide CG bboxes to the box annotator for auto-adjust feature
        manager.box_annotator.set_cg_bboxes(manager.bboxes)
        box_annotator_refresh = setup_box_annotator_gui(
            server,
            manager.box_annotator,
            save_dir=save_dir,
            expanded=box_annotator_cfg.get("expanded", False),
        )

    return box_annotator_refresh


def _build_video_html(video_filename: str, video_url: str, description: str) -> str:
    """Build HTML for a video player with consistent styling.
    
    Args:
        video_filename: Name of the video file (e.g., "rgb.mp4")
        video_url: Full HTTP URL to the video
        description: Description text to display below the video
        
    Returns:
        HTML string for the video element
    """
    return f"""
<div style="margin: 12px 0; padding: 0 8px;">
  <video width="100%" controls style="border-radius: 8px; border: 1px solid rgba(0, 0, 0, 0.2);">
    <source src="{video_url}" type="video/mp4">
    Your browser does not support the video tag.
  </video>
  <div style="margin-top: 8px; font-size: 11px; color: #666; font-family: sans-serif;">
    {description}
  </div>
</div>
"""


def _build_missing_video_html(video_filename: str, video_path: Path) -> str:
    """Build HTML for missing video placeholder with warning styling.
    
    Args:
        video_filename: Name of the video file (e.g., "rgb.mp4")
        video_path: Full path to where the video should be
        
    Returns:
        HTML string for the missing video warning
    """
    return f"""
<div style="margin: 12px 8px; padding: 12px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 4px; font-size: 12px; font-family: sans-serif; color: #856404;">
  <strong>⚠️ Video not found</strong><br/>
  <code>{video_filename}</code> not found at:<br/>
  <code style="font-size: 10px;">{video_path}</code>
</div>
"""


def _add_video_to_gui(server: viser.ViserServer, video_filename: str, video_path: Path, 
                      description: str, video_port: int = 8766):
    """Add a video element to the GUI, showing either the video or a missing file warning.
    
    Args:
        server: Viser server instance
        video_filename: Name of the video file (e.g., "rgb.mp4")
        video_path: Full path to the video file
        description: Description text to display below the video
        video_port: Port number for the HTTP video server
    """
    if video_path.exists():
        video_url = f"http://localhost:{video_port}/{video_filename}"
        html = _build_video_html(video_filename, video_url, description)
        return server.gui.add_html(html)
    else:
        html = _build_missing_video_html(video_filename, video_path)
        return server.gui.add_html(html)


def setup_data_collection_folder(server: viser.ViserServer, map_path: Path, cfg: DictConfig, video_port: int = 8766):
    """Set up the Data Collection folder with video players for rgb, depth, and rgbd recordings."""
    gui_cfg = cfg.gui
    if not gui_cfg.data_collection.visible:
        return None

    videos = [
        ("rgb.mp4", "RGB recording from data collection"),
        ("depth.mp4", "Depth recording from data collection"),
        ("rgbd.mp4", "RGBD recording from data collection"),
    ]

    html_handles = []

    def _render_video_html(video_filename: str, description: str, selected_map_path: Path) -> str:
        video_path = selected_map_path / video_filename
        if video_path.exists():
            token = str(int(selected_map_path.stat().st_mtime_ns)) if selected_map_path.exists() else "0"
            video_url = f"http://localhost:{video_port}/{video_filename}?map={token}"
            return _build_video_html(video_filename, video_url, description)
        return _build_missing_video_html(video_filename, video_path)
    with server.gui.add_folder("Data Collection", expand_by_default=gui_cfg.data_collection.expanded):
        for video_filename, description in videos:
            initial_html = _render_video_html(video_filename, description, map_path)
            html_handles.append(server.gui.add_html(initial_html))

    def refresh(selected_map_path: Path):
        for handle, (video_filename, description) in zip(html_handles, videos):
            handle.content = _render_video_html(video_filename, description, selected_map_path)

    return refresh


def _read_notes_from_meta(map_path: Path) -> str:
    """Read notes string from map metadata."""
    meta_path = map_path / "meta.json"
    if not meta_path.exists():
        return ""
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed reading map metadata at %s: %s", meta_path, exc)
        return ""

    if not isinstance(meta, dict):
        return ""

    notes = meta.get("notes", "")
    return notes if isinstance(notes, str) else str(notes)


def _write_notes_to_meta(map_path: Path, notes: str):
    """Persist notes into map metadata under the `notes` field."""
    meta_path = map_path / "meta.json"
    meta: dict[str, Any] = {}

    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                meta = loaded
        except Exception as exc:  # noqa: BLE001
            log.warning("Failed reading existing map metadata at %s: %s", meta_path, exc)

    meta["notes"] = notes

    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    except Exception as exc:  # noqa: BLE001
        log.warning("Failed writing map metadata at %s: %s", meta_path, exc)


def setup_notes_folder(server: viser.ViserServer, map_path: Path, cfg: DictConfig):
    """Set up a Notes folder with a raw multiline text area synced to meta.json."""
    gui_cfg = cfg.gui
    notes_cfg = gui_cfg.get("notes", {})
    expanded = notes_cfg.get("expanded", False)
    notes_map_ref = {"path": map_path.resolve()}

    with server.gui.add_folder("Notes", expand_by_default=expanded):
        notes_input = server.gui.add_text(
            "Notes",
            initial_value=_read_notes_from_meta(notes_map_ref["path"]),
            multiline=True,
        )

    @notes_input.on_update
    def _(event):
        _write_notes_to_meta(notes_map_ref["path"], notes_input.value)

    def refresh(selected_map_path: Path):
        notes_map_ref["path"] = selected_map_path.resolve()
        notes_input.value = _read_notes_from_meta(notes_map_ref["path"])

    return refresh


def _has_any_videos(map_path: Path) -> bool:
    for video_filename in ("rgb.mp4", "depth.mp4", "rgbd.mp4"):
        if (map_path / video_filename).exists():
            return True
    return False


@hydra.main(version_base=None, config_path="../conf", config_name="visualizer")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    path = Path(cfg.map_path)
    cg = ConceptGraphData.load(path, arcs_enabled=cfg.arcs_enabled)

    log.info("Loading map with a total of %s objects", cg.num_objects)
    if cg.arc_states:
        log.info("Loaded %s arc states", len(cg.arc_states))

    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm_client = None
    if openai.api_key:
        llm_client = OpenAI()
        log.info("OpenAI client initialized for LLM queries")
    else:
        log.warning("OPENAI_API_KEY not set; LLM query and agent features will be disabled")

    # Create Viser server
    viser_port = int(os.getenv("VISER_PORT", "8765"))
    server = viser.ViserServer(host="0.0.0.0", port=viser_port)
    log.info("Viser server started at http://localhost:%s", viser_port)

    # Create callback manager
    manager = ViserCallbackManager(
        server=server,
        pcd_o3d=cg.pcd_o3d,
        clip_ft=cg.clip_features,
        ft_extractor=None,
        segments_anno=cg.segments_anno,
        dense_points=cg.dense_points,
        dense_colors=cg.dense_colors,
        map_path=path,
        axes_ordering=cfg.get("axes_ordering", "xyz"),
        floor_axis=cfg.get("floor_axis", "z"),
        map_meta=cg.map_meta,
        llm_client=llm_client,
        llm_model=cfg.llm_model,
        llm_system_prompt=LLM_SYSTEM_PROMPT,
        arc_states=cg.arc_states,
    )
    cg.dense_points = None
    cg.dense_colors = None
    del cg
    if llm_client is not None:
        manager.set_llm_client(llm_client, cfg.llm_model)
    
    # Initialize the ChatManager and ChatAgent with Langchain
    chat_manager = ChatManager()
    # Set display callback to update the UI when messages are added
    chat_manager.set_display_callback(manager.add_chat_message)
    
    if llm_client is not None and manager.has_segment_objects:
        agent_cfg = cfg.get("agent", None)
        chat_agent = ChatAgent(
            manager=manager,
            cfg=agent_cfg,
            get_current_timestep=manager.get_current_timestep_from_slider,
            chat_manager=chat_manager,
        )
        manager.set_chat_agent(chat_agent)
        log.info("ChatAgent initialized with Langchain tools and chat history")
    else:
        log.info("ChatAgent disabled (requires OPENAI_API_KEY and object data)")

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        manager.notify_clients(
            title="Connected",
            body="You are now connected to the visualizer.",
            client=client,
            auto_close_seconds=2.0,
        )

        if manager.can_run_clip_query and manager.ft_extractor is None:
            manager.notify_clients(
                title="CLIP model",
                body="CLIP model is still loading...",
                client=client,
                color="yellow",
                with_close_button=True,
            )

    # Add initial point clouds
    manager.add_point_clouds()

    # Project selector (switch between saved maps)
    setup_project_selector_gui(server, manager, cfg, path)

    # Setup GUI controls
    box_annotator_refresh = setup_gui(server, manager, cfg)
    
    # Start HTTP server for video in background
    video_port = int(os.getenv("VISER_VIDEO_PORT", "8766"))

    video_root_ref = {"path": path.resolve()}

    def start_video_server():
        """Start a simple HTTP server to serve the rgb.mp4 file."""
        class VideoHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(video_root_ref["path"]), **kwargs)
            
            def log_message(self, format, *args):
                # Suppress HTTP server logs to keep output clean
                pass
            
            def copyfile(self, source, outputfile):
                # Handle BrokenPipeError gracefully when client disconnects
                try:
                    super().copyfile(source, outputfile)
                except BrokenPipeError:
                    # Client disconnected before file transfer completed
                    pass
        
        try:
            with socketserver.TCPServer(("", video_port), VideoHandler) as httpd:
                log.info("Video server started at http://localhost:%s", video_port)
                httpd.serve_forever()
        except BrokenPipeError:
            # Client disconnected - ignore
            pass
        except Exception as e:
            log.error(f"Failed to start video server: {e}")
    
    video_server_thread = threading.Thread(target=start_video_server, daemon=True)
    video_server_thread.start()
    
    # Setup Data Collection folder with video
    refresh_data_collection = setup_data_collection_folder(server, path, cfg, video_port=video_port)
    refresh_notes = setup_notes_folder(server, path, cfg)
    manager.gui_fsm.configure_map_switch_refreshes(
        video_root_ref=video_root_ref,
        refresh_data_collection=refresh_data_collection,
        refresh_notes=refresh_notes,
        box_annotator_refresh=box_annotator_refresh,
    )

    # Background-load the CLIP model so UI is responsive immediately
    def load_model_background():
        log.info("Starting background loading of CLIP model...")
        if not manager.can_run_clip_query:
            log.info("Skipping CLIP model load: CLIP features or objects are unavailable")
            return
        try:
            extractor = hydra.utils.instantiate(cfg.ft_extraction)
            manager.set_ft_extractor(extractor)
        except Exception as e:
            log.error(f"Failed to load CLIP model: {e}")
            manager.notify_clients(
                title="CLIP model",
                body="Failed to load CLIP model; search is unavailable.",
                color="red",
                with_close_button=True,
            )
            return
        manager.notify_clients(
            title="CLIP model loaded!",
            body="CLIP model has been successfully loaded; search is available.",
            color="green",
            with_close_button=True,
        )
        log.info("Finished background loading of CLIP model.")


    loader_thread = threading.Thread(target=load_model_background, daemon=True)
    loader_thread.start()

    # Keep server running
    log.info("Visualization ready. Open http://localhost:8080 in your browser.")
    log.info("Press Ctrl+C to exit.")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down...")


if __name__ == "__main__":
    main()
