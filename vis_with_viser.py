import hydra
import torch
from omegaconf import DictConfig
import logging
import json
import html
import re
import os
from functools import cached_property
from http.server import HTTPServer, SimpleHTTPRequestHandler
import socketserver

from pathlib import Path
import numpy as np
import open3d as o3d
import time
import threading
from matplotlib.pyplot import get_cmap

import openai
from openai import OpenAI
import viser
from typing import Any

from concept_graphs.utils import set_seed
from concept_graphs.viz.utils import similarities_to_rgb
from concept_graphs.mapping.similarity.semantic import CosineSimilarity01

# A logger for this file
log = logging.getLogger(__name__)

LLM_SYSTEM_PROMPT = (
    "You are a retrieval assistant for a 3D scene. Given an abstract user request and a set of"
    " objects described by an id, label, and caption, return up to three object ids that best"
    " satisfy the request. Choose only from the provided ids, sort them by relevance, and return"
    " a JSON object shaped as {\"object_ids\": [id1, id2, id3]}. Return an empty list when nothing fits."
)

SIMILARITY_CMAP = "RdYlGn"
FORWARD_ARC_CMAP = "winter"  # Top half: blue-green to green
DEPENDENCY_ARC_CMAP = "autumn"  # Bottom half: red to orangeish


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
        axes_ordering="-xzy",
        llm_client: OpenAI | None = None,
        llm_model: str = "gpt-4o-mini",
        llm_system_prompt: str | None = None,
        arc_states: list[dict] | None = None,
    ):
        self.server = server
        self.ft_extractor = ft_extractor  # Might be None initially
        self.axes_ordering = axes_ordering
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.llm_system_prompt = llm_system_prompt or LLM_SYSTEM_PROMPT
        
        # Store arc states (list of ArcState dicts, each with 'arcs' list)
        self.arc_states = arc_states or []
        self.current_arc_state_index = 0  # Index into arc_states list

        # Store point cloud data
        self.pcd = pcd_o3d
        self.num_objects = len(self.pcd)
        self.dense_points = dense_points
        self.dense_colors = dense_colors

        # Save original point clouds before any transformation
        self.pcd_original = self.pcd
        
        # Extract points and colors from Open3D point clouds
        self.points_list = [np.asarray(p.points).astype(np.float32) for p in self.pcd]
        self.og_colors = [np.asarray(p.colors).astype(np.float32) for p in self.pcd]
        
        # Apply axes transformation (supports signed axes like "-x-y-z" or "-xzy")
        if axes_ordering != "xyz":
            axis_map = {'x': 0, 'y': 1, 'z': 2}

            def _parse_signed_axes(ordering: str):
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

            try:
                indices, signs = _parse_signed_axes(axes_ordering)
                # Transform points
                self.points_list = [pts[:, indices] * signs for pts in self.points_list]
                if self.dense_points is not None:
                    self.dense_points = self.dense_points[:, indices] * signs
                # Overwrite self.pcd with transformed point clouds
                self.pcd = self._transform_pcd(self.pcd, indices, signs)
            except (KeyError, IndexError, ValueError):
                log.warning(f"Invalid axes_ordering '{axes_ordering}', using 'xyz'")

        # Compute centroids and bounding boxes
        self.centroids = [np.mean(pts, axis=0) for pts in self.points_list]
        self.bboxes = self._compute_bboxes()

        # Generate random colors for each object
        self.random_colors = np.random.rand(self.num_objects, 3).astype(np.float32)

        # Similarity data
        self.sim_query = 0.5 * np.ones(self.num_objects)

        # Semantic features (initialize on CPU; move later if extractor loads async)
        self.semantic_sim = CosineSimilarity01()
        self.semantic_tensor = torch.from_numpy(clip_ft).float()

        # If extractor was provided up front, finish setup now
        if self.ft_extractor is not None:
            self.set_ft_extractor(self.ft_extractor)

        # Track current visualization handles
        self.pcd_handles = []
        self.bbox_handles = []
        self.centroid_handles = []
        self.id_handles = []
        self.label_handles = []
        self.caption_handles = []
        self.arc_handles = []
        # Per-object label handles keyed by object id (as string)
        self.object_labels_dict: dict[str, Any] = {}

        # Toggle states
        self.bbox_visible = False
        self.centroid_visible = False
        self.ids_visible = False
        self.labels_visible = False
        self.captions_visible = False
        self.forward_arcs_visible = False
        self.dependency_arcs_visible = False

        # Arc handles split by type
        self.forward_arc_handles = []
        self.dependency_arc_handles = []

        # GUI slider handle for arc states (set by setup_gui)
        self.arc_state_slider = None

        # Stacking mode states (checkboxes)
        self.rgb_enabled = True
        self.random_enabled = False
        self.similarity_enabled = False
        self.dense_enabled = False
        self.llm_palette_active = False
        self.llm_palette_colors = None
        
        # GUI checkbox handles (set by setup_gui)
        self.similarity_checkbox = None

        # Similarity legend HTML handle (set by setup_gui)
        self.cmap_html_handle = None

        # Small epsilon for point offset to prevent z-fighting
        self.EPS = 0.001 #1e-5

        # Store segment annotations for labels and captions
        self.segments_anno = segments_anno

        # Chat history for GUI chat panel
        self.chat_messages = []
        self.chat_markdown_handle = None

    def notify_clients(self, title: str, body: str, *, client=None, **kwargs):
        """Send a notification to one client or all connected clients."""
        targets = [client] if client is not None else list(self.server.get_clients().values())
        if not targets:
            log.debug("No connected clients to notify: %s - %s", title, body)
            return

        for target in targets:
            try:
                target.add_notification(title=title, body=body, **kwargs)
            except Exception as exc:  # noqa: BLE001
                log.warning("Failed to send notification to client %s: %s", getattr(target, "client_id", "unknown"), exc)

    def set_ft_extractor(self, ft_extractor):
        """Called when the feature extractor finishes loading in the background."""
        self.ft_extractor = ft_extractor
        device = getattr(self.ft_extractor, "device", "cpu")

        # Move pre-loaded CLIP features to model device
        self.semantic_tensor = self.semantic_tensor.to(device)

        msg = f"CLIP Model loaded on {device}. Search is now active."
        log.info(msg)
        self.notify_clients(
            title="System",
            body=msg,
            color="green",
            auto_close_seconds=6.0,
        )

    def set_llm_client(self, llm_client: OpenAI, llm_model: str | None = None):
        """Attach the OpenAI client once credentials are available."""
        self.llm_client = llm_client
        if llm_model:
            self.llm_model = llm_model

        msg = f"LLM client ready using model '{self.llm_model}'."
        log.info(msg)
        self.notify_clients(
            title="LLM",
            body=msg,
            color="green",
            auto_close_seconds=6.0,
        )

    def offset_by_eps(self, points: np.ndarray) -> np.ndarray:
        """Add small random offsets to points to prevent z-fighting.
        
        Samples N points uniformly from [-EPS, +EPS] and adds them to the input.
        
        Args:
            points: Input point cloud of shape (N, 3)
            
        Returns:
            Point cloud with small random offsets applied
        """
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
        self.chat_markdown_handle = markdown_handle
        self._update_chat_markdown()

    def add_chat_message(self, sender: str, message: str):
        """Append a chat message and refresh the markdown view."""
        clean_message = message.strip()
        if not clean_message:
            return
        self.chat_messages.append((sender, clean_message))
        self._update_chat_markdown()

    def _update_chat_markdown(self):
        """Render chat history into the markdown handle."""
        if self.chat_markdown_handle is None:
            return

        content = self._render_chat_html()
        self.chat_markdown_handle.content = content

    def _render_chat_html(self) -> str:
        """Render chat messages inside a scrollable container."""
        if not self.chat_messages:
            body = "<em>No messages yet.</em>"
        else:
            pieces = []
            for sender, msg in self.chat_messages:
                safe_sender = html.escape(sender)
                safe_msg = html.escape(msg).replace("\n", "<br>")
                pieces.append(f"<h3>{safe_sender}</h3><p>{safe_msg}</p>")
            body = "".join(pieces)

        return (
            "<div style=\"max-height: 320px; overflow-y: auto; padding-right: 8px;\">"
            f"{body}"
            "</div>"
        )

    def set_cmap_html_handle(self, handle):
        """Store reference to the colormap HTML handle created in setup_gui."""
        self.cmap_html_handle = handle

    def _set_cmap_visibility(self, visible: bool):
        if self.cmap_html_handle is not None:
            self.cmap_html_handle.visible = visible

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
                point_size=0.01,
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
                    point_size=0.01,
                    point_shape="circle",
                )
                self.pcd_handles.append(handle)

    def remove_point_clouds(self):
        """Remove all point cloud handles."""
        for handle in self.pcd_handles:
            handle.remove()
        self.pcd_handles = []

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

    def add_labels(self):
        """Add text labels to the scene."""
        self.remove_labels()
        for i, (centroid, label) in enumerate(zip(self.centroids, self.labels)):
            if label:  # Only add if label is not empty
                handle = self.server.scene.add_label(
                    name=f"/labels/label_{i}",
                    text=label,
                    position=centroid.astype(np.float32) + np.array([0, 0, 0.1], dtype=np.float32),
                )
                self.label_handles.append(handle)

    def remove_labels(self):
        """Remove all label handles."""
        for handle in self.label_handles:
            handle.remove()
        self.label_handles = []

        # Also remove any per-object labels managed via object_labels_dict
        if self.object_labels_dict:
            for handle in list(self.object_labels_dict.values()):
                try:
                    handle.remove()
                except Exception:
                    pass
            self.object_labels_dict = {}

    def toggle_bbox(self):
        """Toggle bounding box visibility."""
        if self.bbox_visible:
            self.remove_bboxes()
        else:
            self.add_bboxes()
        self.bbox_visible = not self.bbox_visible

    def toggle_centroids(self):
        """Toggle centroid visibility."""
        if self.centroid_visible:
            self.remove_centroids()
        else:
            self.add_centroids()
        self.centroid_visible = not self.centroid_visible

    def toggle_ids(self):
        """Toggle ID visibility."""
        if self.ids_visible:
            self.remove_ids()
        else:
            self.add_ids()
        self.ids_visible = not self.ids_visible

    def toggle_labels(self):
        """Toggle label visibility."""
        if self.labels_visible:
            self.remove_labels()
        else:
            # Add labels per-object using the helper
            for i in range(self.num_objects):
                self.add_label_for_object(i)
        self.labels_visible = not self.labels_visible

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
        if self.captions_visible:
            self.remove_captions()
        else:
            self.add_captions()
        self.captions_visible = not self.captions_visible

    def get_current_arcs(self) -> list[dict]:
        """Get arcs from the currently selected ArcState."""
        if not self.arc_states or self.current_arc_state_index >= len(self.arc_states):
            return []
        return self.arc_states[self.current_arc_state_index].get("arcs", [])

    def get_current_arc_state_info(self) -> str:
        """Get a markdown string describing the current arc state (day/hour)."""
        if not self.arc_states or self.current_arc_state_index >= len(self.arc_states):
            return "**No arc states**"
        state = self.arc_states[self.current_arc_state_index]
        day = state.get("current_day", 0)
        hour = state.get("current_hour", 0)
        return f"**Day {day}, Hour {hour:.1f}**"

    def set_arc_state_index(self, index: int):
        """Set the current arc state index and refresh arc visualization."""
        if not self.arc_states:
            return
        self.current_arc_state_index = max(0, min(index, len(self.arc_states) - 1))
        # Refresh arcs if any are visible
        if self.forward_arcs_visible:
            self.add_forward_arcs()
        if self.dependency_arcs_visible:
            self.add_dependency_arcs()

    def set_arc_state_slider(self, slider):
        """Store reference to the arc state slider."""
        self.arc_state_slider = slider

    def _get_all_arcs_with_resolved_ids(self) -> list[tuple[int, dict]]:
        """Get all arcs (both forward and dependency) with resolved source/target IDs.
        
        Returns:
            List of (arc_index, arc_dict_with_ids) tuples for all valid arcs.
        """
        arcs = self.get_current_arcs()
        if not arcs:
            return []
        
        all_arcs_with_ids = []
        for i, arc in enumerate(arcs):
            raw_source = arc.get("source")
            raw_target = arc.get("target")
            source_id = self._resolve_object_id(raw_source)
            target_id = self._resolve_object_id(raw_target)
            
            # Skip invalid arcs
            if source_id is None or target_id is None:
                continue
            if source_id >= self.num_objects or target_id >= self.num_objects:
                continue
            
            arc_with_ids = arc.copy()
            arc_with_ids["source_id"] = source_id
            arc_with_ids["target_id"] = target_id
            all_arcs_with_ids.append((i, arc_with_ids))
        
        return all_arcs_with_ids

    def _detect_arc_overlaps_all(self) -> dict:
        """Detect overlapping arcs across ALL arc types and assign offsets.
        
        This considers both forward and dependency arcs together to properly
        handle bidirectional cases where S->T might be forward and T->S might
        be dependency.
        
        Returns a dict mapping arc index to (offset_multiplier, total_overlaps, is_reversed).
        """
        all_arcs = self._get_all_arcs_with_resolved_ids()
        if not all_arcs:
            return {}
        
        # Group arcs by their source-target pair (canonicalized for bidirectional detection)
        arc_groups = {}  # Maps (min_id, max_id) -> list of (arc_idx, is_reversed) tuples
        
        for idx, arc in all_arcs:
            source = arc.get("source_id")
            target = arc.get("target_id")
            
            # Create canonical key (sorted IDs) to detect bidirectional arcs
            canonical_source = min(source, target)
            canonical_target = max(source, target)
            key = (canonical_source, canonical_target)
            
            # Track whether this arc is "reversed" relative to canonical direction
            is_reversed = source > target
            
            if key not in arc_groups:
                arc_groups[key] = []
            arc_groups[key].append((idx, is_reversed))
        
        # Assign offsets to overlapping arcs
        offset_map = {}  # Maps arc_idx -> (offset_multiplier, total_overlaps, is_reversed)
        for key, arc_info_list in arc_groups.items():
            num_arcs = len(arc_info_list)
            if num_arcs > 1:
                log.info(f"Arc group {key}: {num_arcs} arcs -> {arc_info_list}")
                # Multiple arcs between same objects - spread them out
                for i, (arc_idx, is_reversed) in enumerate(arc_info_list):
                    # Distribute offsets: -n, -(n-1), ..., -1, 0, 1, ..., n
                    # For even count: offset from center, for odd: center one at 0
                    if num_arcs % 2 == 1:
                        # Odd number: center arc gets 0, others distributed around it
                        offset_multiplier = i - num_arcs // 2
                    else:
                        # Even number: no arc at exact center
                        offset_multiplier = i - num_arcs // 2 + 0.5
                    
                    offset_map[arc_idx] = (offset_multiplier, num_arcs, is_reversed)
                    log.info(f"  Arc {arc_idx}: offset={offset_multiplier}, reversed={is_reversed}")
        
        if offset_map:
            log.info(f"Final offset_map: {offset_map}")
        return offset_map

    def _detect_arc_overlaps(self, typed_arcs: list) -> dict:
        """Detect overlapping arcs and assign offsets to separate them.
        
        Returns a dict mapping arc index to (offset_multiplier, total_overlaps) where:
        - offset_multiplier: integer offset (-n to +n) to apply to this arc
        - total_overlaps: total number of arcs in this overlap group
        
        For bidirectional arcs (S->T and T->S), we use a consistent perpendicular direction
        based on the canonical (sorted) source-target pair to ensure arcs are separated.
        """
        # Use the global overlap detection that considers ALL arc types
        return self._detect_arc_overlaps_all()

    def _add_arcs_by_type(self, arc_type: str, cmap_name: str, name_prefix: str) -> list:
        """Add arcs of a specific type to the scene.
        
        Args:
            arc_type: 'forward' or 'dependency'
            cmap_name: Name of the colormap to use
            name_prefix: Prefix for scene object names
            
        Returns:
            List of created handles
        """
        handles = []
        arcs = self.get_current_arcs()
        
        if not arcs:
            return handles
        
        cmap = get_cmap(cmap_name)
        
        # Filter arcs by type and add resolved IDs for overlap detection
        typed_arcs_with_ids = []
        for i, arc in enumerate(arcs):
            if arc.get("arc_type") != arc_type:
                continue
                
            raw_source = arc.get("source")
            raw_target = arc.get("target")
            source_id = self._resolve_object_id(raw_source)
            target_id = self._resolve_object_id(raw_target)
            
            # Validate source and target indices early
            if source_id is None or target_id is None:
                log.warning(
                    f"Arc {i} missing or unresolved source/target ({raw_source} -> {raw_target}), skipping"
                )
                continue
            if source_id >= self.num_objects or target_id >= self.num_objects:
                log.warning(f"Arc {i} has invalid object indices ({source_id} -> {target_id}), skipping")
                continue
            
            # Store resolved IDs in arc dict for overlap detection
            arc_with_ids = arc.copy()
            arc_with_ids["source_id"] = source_id
            arc_with_ids["target_id"] = target_id
            typed_arcs_with_ids.append((i, arc_with_ids))
        
        # Detect overlapping arcs and get offset information
        offset_map = self._detect_arc_overlaps(typed_arcs_with_ids)
        
        for i, arc in typed_arcs_with_ids:
            source_id = arc["source_id"]
            target_id = arc["target_id"]
            weight = arc.get("weight", 0.5)
            label = arc.get("label", "")
            
            # Get source and target centroids
            source_pos = self.centroids[source_id]
            target_pos = self.centroids[target_id]

            # Add labels for the interacting objects
            self.add_label_for_object(source_id)
            self.add_label_for_object(target_id)
            
            # Compute arc height based on distance between objects
            distance = np.linalg.norm(target_pos - source_pos)
            arc_height = min(3, distance * 0.4)  # Arc rises proportionally to distance
            
            # Compute midpoint and raise it above the scene
            midpoint = (source_pos + target_pos) / 2
            midpoint_raised = midpoint.copy()
            midpoint_raised[2] += arc_height  # Raise in Z (up) direction
            
            # Compute perpendicular offset for overlapping arcs (including bidirectional S->T, T->S)
            perp_offset = np.zeros(3, dtype=np.float32)
            if i in offset_map:
                offset_multiplier, total_overlaps, is_reversed = offset_map[i]
                
                # Use CANONICAL arc direction (from lower ID to higher ID) to ensure
                # consistent perpendicular direction for bidirectional arcs.
                # This way S->T and T->S use the same perpendicular reference.
                if is_reversed:
                    canonical_direction = source_pos - target_pos  # Flip to canonical
                else:
                    canonical_direction = target_pos - source_pos
                
                arc_direction_norm = np.linalg.norm(canonical_direction[:2])  # Use only XY plane
                
                if arc_direction_norm > 1e-6:
                    # Perpendicular vector in XY plane (rotate 90 degrees left of canonical direction)
                    perp_direction = np.array([-canonical_direction[1], canonical_direction[0], 0])
                    perp_direction = perp_direction / np.linalg.norm(perp_direction)
                    
                    # Dynamically scale offset based on number of overlapping arcs
                    # Total spread is capped but grows with more arcs, then divided evenly
                    max_total_spread = min(2.0, distance * 0.4)  # Max spread across all arcs
                    min_arc_spacing = 0.15  # Minimum spacing between adjacent arcs
                    
                    # Calculate spacing: use larger of even distribution or minimum spacing
                    # For N arcs, we need (N-1) gaps to cover the spread
                    even_spacing = max_total_spread / max(1, total_overlaps - 1) if total_overlaps > 1 else 0
                    arc_spacing = max(min_arc_spacing, even_spacing)
                    
                    # Apply offset: multiplier is centered around 0, so multiply by spacing
                    offset_distance = offset_multiplier * arc_spacing
                    
                    # Store the offset vector to apply to all control points
                    perp_offset = perp_direction * offset_distance
            
            # Apply offset to the midpoint
            midpoint_raised += perp_offset
            
            # Create control points for Catmull-Rom spline
            # We need at least 4 points for Catmull-Rom; use extended endpoints
            # Apply perpendicular offset to all control points so bidirectional arcs are fully separated
            post_source = source_pos + np.array([0, 0, arc_height * 0.1]) + perp_offset
            pre_target = target_pos + np.array([0, 0, arc_height * 0.1]) + perp_offset
            
            control_points = np.array([
                source_pos,
                post_source,
                midpoint_raised,
                pre_target,
                target_pos,
            ], dtype=np.float32)
            
            # Get color from colormap based on weight
            # For forward arcs: use top half of winter (0.5 to 1.0)
            # For dependency arcs: use bottom half of autumn (0.0 to 0.5)
            if arc_type == "forward":
                cmap_value = 0.5 + weight * 0.5  # Map [0,1] to [0.5,1]
            else:  # dependency
                cmap_value = weight * 0.5  # Map [0,1] to [0,0.5]
            r, g, b, _ = cmap(cmap_value)
            color_uint8 = (int(r * 255), int(g * 255), int(b * 255))
            
            # Add the spline
            handle = self.server.scene.add_spline_catmull_rom(
                name=f"/{name_prefix}/arc_{i}",
                positions=control_points,
                tension=0.5,
                line_width=3.0,
                color=color_uint8,
                segments=50,
            )
            handles.append(handle)

            # Sample points along the spline to add arrow heads
            spline_points = self._sample_catmull_rom_spline(control_points, num_samples=50)
            
            # Add arrow heads at intervals along the spline
            arrow_interval = 5  # Add an arrow every N points
            arrow_length = 0.15
            arrow_angle = np.radians(30)  # 30 degrees offset
            
            for j in range(arrow_interval, len(spline_points) - 1, arrow_interval):
                point = spline_points[j]
                next_point = spline_points[j + 1]
                
                # Compute tangent direction (pointing towards target)
                tangent = next_point - point
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm < 1e-6:
                    continue
                tangent = tangent / tangent_norm
                
                # Find a perpendicular vector (in the plane containing tangent and up)
                up = np.array([0.0, 0.0, 1.0])
                perp = np.cross(tangent, up)
                perp_norm = np.linalg.norm(perp)
                if perp_norm < 1e-6:
                    # Tangent is parallel to up, use a different reference
                    up = np.array([1.0, 0.0, 0.0])
                    perp = np.cross(tangent, up)
                    perp_norm = np.linalg.norm(perp)
                perp = perp / perp_norm
                
                # Create arrow wing directions (rotated from -tangent by +/- arrow_angle)
                back_dir = -tangent
                
                # Rotate back_dir around perp by +/- arrow_angle
                cos_a = np.cos(arrow_angle)
                sin_a = np.sin(arrow_angle)
                
                # Wing 1: rotate back_dir by +arrow_angle around perp
                wing1_dir = back_dir * cos_a + np.cross(perp, back_dir) * sin_a + perp * np.dot(perp, back_dir) * (1 - cos_a)
                wing1_dir = wing1_dir / np.linalg.norm(wing1_dir)
                
                # Wing 2: rotate back_dir by -arrow_angle around perp  
                wing2_dir = back_dir * cos_a - np.cross(perp, back_dir) * sin_a + perp * np.dot(perp, back_dir) * (1 - cos_a)
                wing2_dir = wing2_dir / np.linalg.norm(wing2_dir)
                
                # Create a second perpendicular axis (perpendicular to both tangent and perp)
                perp2 = np.cross(tangent, perp)
                perp2 = perp2 / np.linalg.norm(perp2)
                
                # Wing 3: rotate back_dir by +arrow_angle around perp2
                wing3_dir = back_dir * cos_a + np.cross(perp2, back_dir) * sin_a + perp2 * np.dot(perp2, back_dir) * (1 - cos_a)
                wing3_dir = wing3_dir / np.linalg.norm(wing3_dir)
                
                # Wing 4: rotate back_dir by -arrow_angle around perp2
                wing4_dir = back_dir * cos_a - np.cross(perp2, back_dir) * sin_a + perp2 * np.dot(perp2, back_dir) * (1 - cos_a)
                wing4_dir = wing4_dir / np.linalg.norm(wing4_dir)
                
                # Create line segments for the arrow wings
                wing1_start = point + wing1_dir * arrow_length
                wing2_start = point + wing2_dir * arrow_length
                wing3_start = point + wing3_dir * arrow_length
                wing4_start = point + wing4_dir * arrow_length
                
                # Add arrow wings
                for wing_idx, wing_start in enumerate([wing1_start, wing2_start, wing3_start, wing4_start], 1):
                    arrow_handle = self.server.scene.add_line_segments(
                        name=f"/{name_prefix}/arc_{i}/arrow_{j}_wing{wing_idx}",
                        points=np.array([[wing_start, point]], dtype=np.float32),
                        colors=color_uint8,
                        line_width=2.0,
                    )
                    handles.append(arrow_handle)

            # Add label at the midpoint if provided
            if label:
                label_handle = self.server.scene.add_label(
                    name=f"/{name_prefix}/arc_label_{i}",
                    text=f"{label}",
                    position=midpoint_raised.astype(np.float32) + np.array([0, 0, 0.3], dtype=np.float32),
                )
                handles.append(label_handle)
        
        return handles

    def add_forward_arcs(self):
        """Add forward arcs to the scene using the forward colormap."""
        self.remove_forward_arcs()
        self.forward_arc_handles = self._add_arcs_by_type("forward", FORWARD_ARC_CMAP, "forward_arcs")

    def remove_forward_arcs(self):
        """Remove all forward arc handles."""
        for handle in self.forward_arc_handles:
            handle.remove()
        self.forward_arc_handles = []

    def add_dependency_arcs(self):
        """Add dependency arcs to the scene using the dependency colormap."""
        self.remove_dependency_arcs()
        self.dependency_arc_handles = self._add_arcs_by_type("dependency", DEPENDENCY_ARC_CMAP, "dependency_arcs")

    def remove_dependency_arcs(self):
        """Remove all dependency arc handles."""
        for handle in self.dependency_arc_handles:
            handle.remove()
        self.dependency_arc_handles = []

    def remove_arcs(self):
        """Remove all arc handles (both forward and dependency)."""
        self.remove_forward_arcs()
        self.remove_dependency_arcs()
        # Also clear old-style arc_handles if any remain
        for handle in self.arc_handles:
            handle.remove()
        self.arc_handles = []

    def toggle_forward_arcs(self):
        """Toggle forward arc visibility."""
        if self.forward_arcs_visible:
            self.remove_forward_arcs()
        else:
            self.add_forward_arcs()
        self.forward_arcs_visible = not self.forward_arcs_visible

    def toggle_dependency_arcs(self):
        """Toggle dependency arc visibility."""
        if self.dependency_arcs_visible:
            self.remove_dependency_arcs()
        else:
            self.add_dependency_arcs()
        self.dependency_arcs_visible = not self.dependency_arcs_visible

    def toggle_all_arcs(self):
        """Toggle all arcs (both forward and dependency)."""
        # Determine target state: if any are visible, hide all; otherwise show all
        any_visible = self.forward_arcs_visible or self.dependency_arcs_visible
        
        if any_visible:
            self.remove_forward_arcs()
            self.remove_dependency_arcs()
            self.forward_arcs_visible = False
            self.dependency_arcs_visible = False
        else:
            self.add_forward_arcs()
            self.add_dependency_arcs()
            self.forward_arcs_visible = True
            self.dependency_arcs_visible = True

    def toggle_rgb_mode(self, enabled: bool):
        """Toggle RGB color mode."""
        self.rgb_enabled = enabled
        self.update_point_clouds()
        self._update_centroid_colors()

    def toggle_random_mode(self, enabled: bool):
        """Toggle random color mode."""
        self.random_enabled = enabled
        self.update_point_clouds()
        self._update_centroid_colors()

    def toggle_similarity_mode(self, enabled: bool):
        """Toggle similarity color mode."""
        self.similarity_enabled = enabled
        # If unchecking, disable the checkbox
        if not enabled and self.similarity_checkbox is not None:
            self.similarity_checkbox.disabled = True
        self._set_cmap_visibility(enabled)
        self.update_point_clouds()
        self._update_centroid_colors()

    def toggle_dense_mode(self, enabled: bool):
        """Toggle dense point cloud mode."""
        if enabled and self.dense_points is None:
            self.notify_clients(
                title="Dense Point Cloud",
                body="Dense point cloud is not available at this map path.",
                color="yellow",
                with_close_button=True,
                auto_close_seconds=6.0,
            )
            log.warning("Dense point cloud requested but not found.")
            return False
        self.dense_enabled = enabled
        self.update_point_clouds()
        return True

    def enable_similarity_mode(self):
        """Enable and check similarity mode after a query."""
        self.similarity_enabled = True
        if self.similarity_checkbox is not None:
            self.similarity_checkbox.disabled = False
            self.similarity_checkbox.value = True
        self._set_cmap_visibility(True)
        self.update_point_clouds()
        self._update_centroid_colors()

    def _sample_catmull_rom_spline(self, control_points: np.ndarray, num_samples: int = 50) -> np.ndarray:
        """Sample points along a Catmull-Rom spline.
        
        Args:
            control_points: Array of shape (N, 3) with control points
            num_samples: Total number of samples along the entire spline
            
        Returns:
            Array of shape (num_samples, 3) with sampled points
        """
        n = len(control_points)
        if n < 4:
            return control_points
        
        # Catmull-Rom spline interpolation
        def catmull_rom_point(p0, p1, p2, p3, t, tension=0.5):
            """Compute a point on a Catmull-Rom spline segment."""
            t2 = t * t
            t3 = t2 * t
            
            # Catmull-Rom basis matrix (with tension parameter)
            s = (1 - tension) / 2
            
            return (
                p1 +
                (-s * p0 + s * p2) * t +
                (2*s * p0 + (s-3) * p1 + (3-2*s) * p2 - s * p3) * t2 +
                (-s * p0 + (2-s) * p1 + (s-2) * p2 + s * p3) * t3
            )
        
        samples = []
        # Number of segments is n-3 for Catmull-Rom (we use overlapping windows of 4 points)
        num_segments = n - 3
        samples_per_segment = max(1, num_samples // num_segments)
        
        for i in range(num_segments):
            p0, p1, p2, p3 = control_points[i:i+4]
            for j in range(samples_per_segment):
                t = j / samples_per_segment
                point = catmull_rom_point(p0, p1, p2, p3, t)
                samples.append(point)
        
        # Add the final point
        samples.append(control_points[-2])  # p2 of the last segment
        
        return np.array(samples, dtype=np.float32)

    def _update_centroid_colors(self):
        """Update centroid colors to match current scheme."""
        if self.centroid_visible:
            self.remove_centroids()
            self.add_centroids()

    def set_similarity_checkbox(self, checkbox):
        """Store reference to the similarity checkbox for enabling/disabling."""
        self.similarity_checkbox = checkbox

    def query(self, query_text: str, client=None):
        """Perform CLIP query and update similarity visualization."""
        if self.ft_extractor is None:
            msg = "CLIP model is still loading... please wait."
            log.warning(msg)
            self.notify_clients(
                title="CLIP model",
                body=msg,
                color="yellow",
                with_close_button=True,
                auto_close_seconds=5.0,
                client=client,
            )
            return

        # Reset any LLM-specific palette so CLIP queries use the inferno colormap.
        self.llm_palette_active = False
        self.llm_palette_colors = None

        query_ft = self.ft_extractor.encode_text([query_text])
        self.sim_query = self.semantic_sim(query_ft, self.semantic_tensor)
        self.sim_query = self.sim_query.squeeze().cpu().numpy()
        self.enable_similarity_mode()
        log.info(f"Query: '{query_text}' - Top match: Object {np.argmax(self.sim_query)}")

    def _build_llm_messages(self, query_text: str):
        """Construct system+user messages for OpenAI completion."""
        object_lines = []
        for idx, (label, caption) in enumerate(zip(self.labels, self.captions)):
            safe_label = label if label else "unknown"
            safe_caption = caption if caption else "unknown"
            object_lines.append(f"{idx}: label='{safe_label}' | caption='{safe_caption}'")

        user_message = (
            "Given an abstract user request, choose up to three objects that best satisfy it. "
            "Only use the provided ids and keep them in best-first order."
            f"\nUser request: {query_text}\nObjects:\n" + "\n".join(object_lines)
        )

        return [
            {"role": "system", "content": self.llm_system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _extract_object_ids(self, content: str) -> list[int]:
        """Parse object ids from LLM JSON response; tolerate loose text."""
        ids: list[int] = []

        try:
            parsed = json.loads(content)
            candidate_ids = parsed.get("object_ids", []) if isinstance(parsed, dict) else []
            for cand in candidate_ids:
                if isinstance(cand, (int, float)):
                    ids.append(int(cand))
        except json.JSONDecodeError:
            # Fallback to any integers in the text
            ids = [int(match) for match in re.findall(r"\d+", content)]

        seen = set()
        filtered = []
        for idx in ids:
            if 0 <= idx < self.num_objects and idx not in seen:
                filtered.append(idx)
                seen.add(idx)

        return filtered[:3]

    def _apply_llm_palette(self, ranked_indices):
        """Apply fixed palette to centroids and force similarity view on."""
        palette = [
            np.array([0.0, 0.8, 0.0], dtype=np.float32),
            np.array([0.95, 0.85, 0.0], dtype=np.float32),
            np.array([0.9, 0.1, 0.1], dtype=np.float32),
        ]
        default_color = np.zeros(3, dtype=np.float32)
        centroid_colors = []

        # Build an explicit similarity score vector so similarity mode still has data
        # while the point colors come from the fixed palette.
        similarity_scores = np.zeros(self.num_objects, dtype=np.float32)
        score_steps = [1.0, 0.66, 0.33]

        for obj_idx in range(self.num_objects):
            color = default_color
            if ranked_indices:
                if obj_idx == ranked_indices[0]:
                    color = palette[0]
                    similarity_scores[obj_idx] = score_steps[0]
                elif len(ranked_indices) > 1 and obj_idx == ranked_indices[1]:
                    color = palette[1]
                    similarity_scores[obj_idx] = score_steps[1]
                elif len(ranked_indices) > 2 and obj_idx == ranked_indices[2]:
                    color = palette[2]
                    similarity_scores[obj_idx] = score_steps[2]

            centroid_colors.append(color.astype(np.float32))

        # Drive the similarity layer so the GUI visibly switches on.
        self.sim_query = similarity_scores
        self.llm_palette_active = True
        self.llm_palette_colors = np.array(centroid_colors, dtype=np.float32)
        self.enable_similarity_mode()

        # Keep centroid colors aligned with the palette without changing base random colors.
        if self.centroid_visible:
            self.remove_centroids()
            original_random = self.random_colors.copy()
            self.random_colors = np.array(centroid_colors, dtype=np.float32)
            self.add_centroids()
            self.random_colors = original_random

    def llm_query(self, query_text: str, client=None):
        """Send query to OpenAI and color top matches."""
        if self.llm_client is None:
            msg = "LLM client is still initializing."
            self.notify_clients(
                title="LLM Query",
                body=msg,
                client=client,
                color="yellow",
                with_close_button=True,
                auto_close_seconds=6.0,
            )
            log.warning(msg)
            return

        messages = self._build_llm_messages(query_text)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            msg = f"OpenAI call failed: {exc}"
            log.error(msg)
            self.notify_clients(
                title="LLM Query",
                body="LLM request failed; check logs and credentials.",
                client=client,
                color="red",
                with_close_button=True,
            )
            return

        ranked = self._extract_object_ids(content)
        if not ranked:
            msg = "LLM returned no valid object ids for this query."
            self.notify_clients(
                title="LLM Query",
                body=msg,
                client=client,
                color="yellow",
                with_close_button=True,
                auto_close_seconds=6.0,
            )
            log.warning(msg)
            return

        self._apply_llm_palette(ranked)
        summary = ", ".join(self.labels[idx] for idx in ranked)
        result_msg = f"LLM query: '{query_text}' → {summary}"
        log.info(result_msg)
        return summary

    def respond(self, message: str) -> str:
        """Answer chat messages by forwarding to the LLM selection pipeline."""
        user_text = message.strip()
        if not user_text:
            return ""

        if self.llm_client is None:
            reply = "LLM client is still initializing. Please try again shortly."
            self.add_chat_message("Agent", reply)
            return reply

        messages = self._build_llm_messages(user_text)

        try:
            response = self.llm_client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            reply = f"LLM chat failed: {exc}"
            log.error(reply)
            self.add_chat_message("Agent", reply)
            return reply

        ranked = self._extract_object_ids(content)
        if ranked:
            self._apply_llm_palette(ranked)
            reply = f"Top objects: {', '.join(self.labels[i] for i in ranked)}"
        else:
            reply = "No suitable objects found for that request."

        self.add_chat_message("Agent", reply)
        return reply

    def listen(self):
        """Start listening for audio input (currently unimplemented)."""
        log.info("Listen method called - audio input not yet implemented")
        # TODO: Implement audio input functionality
        pass


def load_point_cloud(path):
    """Load point cloud, dense cloud arrays, segment annotations, and arcs."""
    path = Path(path)
    pcd = o3d.io.read_point_cloud(str(path / "point_cloud.pcd"))
    dense_path = path / "dense_point_cloud.pcd"
    dense_points = None
    dense_colors = None
    if dense_path.exists():
        dense_pcd = o3d.io.read_point_cloud(str(dense_path))
        dense_points = np.asarray(dense_pcd.points).astype(np.float32)
        dense_colors = np.asarray(dense_pcd.colors).astype(np.float32)
    else:
        log.warning("Dense point cloud not found at %s", dense_path)

    with open(path / "segments_anno.json", "r") as f:
        segments_anno = json.load(f)

    # Load arc states if available (new format: list of ArcState dicts)
    arcs_path = path / "arcs.json"
    arc_states = None
    if arcs_path.exists():
        with open(arcs_path, "r") as f:
            arc_states = json.load(f)
        total_arcs = sum(len(s.get("arcs", [])) for s in arc_states) if arc_states else 0
        log.info(f"Loaded {len(arc_states)} arc states with {total_arcs} total arcs from {arcs_path}")
    else:
        log.info("No arcs.json found at %s", arcs_path)

    # Build individual point clouds for each segment
    pcd_o3d = []
    for ann in segments_anno["segGroups"]:
        obj = pcd.select_by_index(ann["segments"])
        pcd_o3d.append(obj)

    return pcd_o3d, segments_anno, dense_points, dense_colors, arc_states


def setup_gui(server: viser.ViserServer, manager: ViserCallbackManager):
    """Set up the Viser GUI controls."""

    # Add folder for visualization controls
    with server.gui.add_folder("Visualization"):
        # Color mode checkboxes (stackable)
        rgb_checkbox = server.gui.add_checkbox("RGB Colors", initial_value=True)
        random_checkbox = server.gui.add_checkbox("Random Colors", initial_value=False)
        similarity_checkbox = server.gui.add_checkbox("Similarity Colors", initial_value=False)
        similarity_checkbox.disabled = True  # Initially disabled until a query is run
        dense_checkbox = server.gui.add_checkbox("Dense Point Cloud", initial_value=False)
        
        # Store reference to similarity checkbox in manager
        manager.set_similarity_checkbox(similarity_checkbox)
        
        # Colormap legend (visible only when similarity mode is active)
        cmap = get_cmap(SIMILARITY_CMAP)
        stops = np.linspace(0.0, 1.0, 9)
        color_stops = []
        for frac in stops:
            r, g, b, _ = cmap(frac)
            color_stops.append(
                f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)}) {frac * 100:.1f}%"
            )
        
        gradient_css = ", ".join(color_stops)
        cmap_html_content = f"""
<div style="margin-top: 12px; padding: 0 8px; font-family: sans-serif; font-size: 11px;">
  <div style="height: 14px; border-radius: 7px; overflow: hidden; border: 1px solid rgba(0, 0, 0, 0.2); background: linear-gradient(90deg, {gradient_css}); box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);"></div>
  <div style="display: flex; justify-content: space-between; margin-top: 4px; font-weight: 600; color: #666;">
    <span>Low</span>
    <span>High</span>
  </div>
</div>
"""
        cmap_html = server.gui.add_html(cmap_html_content, order=1000.0)
        cmap_html.visible = False
        manager.set_cmap_html_handle(cmap_html)

        @rgb_checkbox.on_update
        def _(event):
            manager.toggle_rgb_mode(rgb_checkbox.value)

        @random_checkbox.on_update
        def _(event):
            manager.toggle_random_mode(random_checkbox.value)

        @similarity_checkbox.on_update
        def _(event):
            manager.toggle_similarity_mode(similarity_checkbox.value)

        @dense_checkbox.on_update
        def _(event):
            success = manager.toggle_dense_mode(dense_checkbox.value)
            if not success:
                dense_checkbox.value = False

    with server.gui.add_folder("Toggles"):
        # Toggle buttons
        bbox_btn = server.gui.add_button("Toggle Bounding Boxes")
        centroid_btn = server.gui.add_button("Toggle Centroids")
        id_btn = server.gui.add_button("Toggle IDs")
        label_btn = server.gui.add_button("Toggle Labels")
        caption_btn = server.gui.add_button("Toggle Captions")

        @bbox_btn.on_click
        def _(_):
            manager.toggle_bbox()

        @centroid_btn.on_click
        def _(_):
            manager.toggle_centroids()

        @id_btn.on_click
        def _(_):
            manager.toggle_ids()

        @label_btn.on_click
        def _(_):
            manager.toggle_labels()

        @caption_btn.on_click
        def _(_):
            manager.toggle_captions()

    with server.gui.add_folder("Arcs"):
        # Slider for selecting arc state (timestep)
        num_states = len(manager.arc_states) if manager.arc_states else 1
        arc_state_slider = server.gui.add_slider(
            "Timestep",
            min=0,
            max=max(0, num_states - 1),
            step=1,
            initial_value=0,
        )
        manager.set_arc_state_slider(arc_state_slider)
        
        # Display current state info
        arc_state_label = server.gui.add_markdown(
            content=manager.get_current_arc_state_info(),
        )
        
        @arc_state_slider.on_update
        def _(event):
            manager.set_arc_state_index(int(arc_state_slider.value))
            arc_state_label.content = manager.get_current_arc_state_info()
        
        # Toggle buttons for different arc types
        forward_arcs_btn = server.gui.add_button("Toggle Forward Arcs")
        dependency_arcs_btn = server.gui.add_button("Toggle Dependency Arcs")
        all_arcs_btn = server.gui.add_button("Toggle All Arcs")

        @forward_arcs_btn.on_click
        def _(_):
            manager.toggle_forward_arcs()

        @dependency_arcs_btn.on_click
        def _(_):
            manager.toggle_dependency_arcs()

        @all_arcs_btn.on_click
        def _(_):
            manager.toggle_all_arcs()

    with server.gui.add_folder("CLIP Query"):
        # Query input
        query_input = server.gui.add_text("Query", initial_value="")
        query_btn = server.gui.add_button("Search")

        @query_btn.on_click
        def _(event):
            if query_input.value.strip():
                manager.query(query_input.value.strip(), client=event.client)

    with server.gui.add_folder("LLM Query", expand_by_default=False):
        llm_query_input = server.gui.add_text("Query", initial_value="")
        llm_query_btn = server.gui.add_button("Search")

        @llm_query_btn.on_click
        def _(event):
            if llm_query_input.value.strip():
                manager.llm_query(llm_query_input.value.strip(), client=event.client)

    with server.gui.add_folder("Chat", expand_by_default=False):
        # Use HTML handle so we can wrap messages in a scrollable container.
        chat_history = server.gui.add_html("<em>No messages yet.</em>")
        manager.register_chat_markdown(chat_history)

        chat_input = server.gui.add_text("Message", initial_value="")
        
        # State for mic button
        mic_state = {"listening": False}
        
        send_btn = server.gui.add_button("Send")
        mic_btn = server.gui.add_button("🎤", hint="Hold to record audio")

        @send_btn.on_click
        def _(event):
            user_text = chat_input.value.strip()
            if not user_text:
                return

            manager.add_chat_message("You", user_text)
            manager.respond(user_text)
            chat_input.value = ""

        @mic_btn.on_click
        def _(event):
            """Toggle audio mode."""
            mic_state["listening"] = not mic_state["listening"]
            
            if mic_state["listening"]:
                # Start listening: change to green background
                try:
                    mic_btn.color = "green"
                except AttributeError:
                    # If style is read-only, try alternative approach
                    pass
                send_btn.disabled = True
                manager.listen()
            else:
                # Stop listening: revert to default background
                try:
                    mic_btn.color = ""
                except AttributeError:
                    pass
                send_btn.disabled = False


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
                      description: str, video_port: int = 8081) -> None:
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
        server.gui.add_html(html)
    else:
        html = _build_missing_video_html(video_filename, video_path)
        server.gui.add_html(html)


def setup_data_collection_folder(server: viser.ViserServer, map_path: Path, video_port: int = 8081):
    """Set up the Data Collection folder with video players for rgb, depth, and rgbd recordings."""
    with server.gui.add_folder("Data Collection", expand_by_default=False):
        videos = [
            ("rgb.mp4", "RGB recording from data collection"),
            ("depth.mp4", "Depth recording from data collection"),
            ("rgbd.mp4", "RGBD recording from data collection"),
        ]
        
        for video_filename, description in videos:
            video_path = map_path / video_filename
            _add_video_to_gui(server, video_filename, video_path, description, video_port)


@hydra.main(version_base=None, config_path="conf", config_name="visualizer")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    path = Path(cfg.map_path)
    clip_ft = np.load(path / "clip_features.npy")
    pcd_o3d, segments_anno, dense_points, dense_colors, arc_states = load_point_cloud(path)

    log.info(f"Loading map with a total of {len(pcd_o3d)} objects")
    if arc_states:
        log.info(f"Loaded {len(arc_states)} arc states")

    openai.api_key = os.getenv("OPENAI_API_KEY")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set")
    llm_client = OpenAI()
    log.info("OpenAI client initialized for LLM queries")

    # Create Viser server
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    log.info("Viser server started at http://localhost:8080")

    # Create callback manager
    manager = ViserCallbackManager(
        server=server,
        pcd_o3d=pcd_o3d,
        clip_ft=clip_ft,
        ft_extractor=None,
        segments_anno=segments_anno,
        dense_points=dense_points,
        dense_colors=dense_colors,
        llm_client=llm_client,
        llm_model=cfg.llm_model,
        llm_system_prompt=LLM_SYSTEM_PROMPT,
        arc_states=arc_states,
    )
    manager.set_llm_client(llm_client, cfg.llm_model)

    @server.on_client_connect
    def _(client: viser.ClientHandle):
        manager.notify_clients(
            title="Connected",
            body="You are now connected to the visualizer.",
            client=client,
            auto_close_seconds=10.0,
        )

        if manager.ft_extractor is None:
            manager.notify_clients(
                title="CLIP model",
                body="CLIP model is still loading...",
                client=client,
                color="yellow",
                with_close_button=True,
            )

    # Add initial point clouds
    manager.add_point_clouds()

    # Setup GUI controls
    setup_gui(server, manager)
    
    # Start HTTP server for video in background
    def start_video_server():
        """Start a simple HTTP server to serve the rgb.mp4 file."""
        class VideoHandler(SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(path), **kwargs)
            
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
            with socketserver.TCPServer(("", 8081), VideoHandler) as httpd:
                log.info("Video server started at http://localhost:8081")
                httpd.serve_forever()
        except BrokenPipeError:
            # Client disconnected - ignore
            pass
        except Exception as e:
            log.error(f"Failed to start video server: {e}")
    
    video_server_thread = threading.Thread(target=start_video_server, daemon=True)
    video_server_thread.start()
    
    # Setup Data Collection folder with video
    setup_data_collection_folder(server, path, video_port=8081)

    # Background-load the CLIP model so UI is responsive immediately
    def load_model_background():
        log.info("Starting background loading of CLIP model...")
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
