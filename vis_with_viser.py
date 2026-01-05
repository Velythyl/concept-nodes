import hydra
import torch
from omegaconf import DictConfig
import logging
import json
import html
import re
import os
from functools import cached_property

from pathlib import Path
import numpy as np
import open3d as o3d
import time
import threading

import openai
from openai import OpenAI
import viser

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
    ):
        self.server = server
        self.ft_extractor = ft_extractor  # Might be None initially
        self.axes_ordering = axes_ordering
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.llm_system_prompt = llm_system_prompt or LLM_SYSTEM_PROMPT

        # Store point cloud data
        self.pcd = pcd_o3d
        self.num_objects = len(self.pcd)
        self.dense_points = dense_points
        self.dense_colors = dense_colors

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
                self.points_list = [pts[:, indices] * signs for pts in self.points_list]
                if self.dense_points is not None:
                    self.dense_points = self.dense_points[:, indices] * signs
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

        # Toggle states
        self.bbox_visible = False
        self.centroid_visible = False
        self.ids_visible = False
        self.labels_visible = False
        self.captions_visible = False

        # Stacking mode states (checkboxes)
        self.rgb_enabled = True
        self.random_enabled = False
        self.similarity_enabled = False
        self.dense_enabled = False
        self.llm_palette_active = False
        self.llm_palette_colors = None
        
        # GUI checkbox handles (set by setup_gui)
        self.similarity_checkbox = None

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
                rgb = similarities_to_rgb(self.sim_query, cmap_name="inferno")
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
            self.add_labels()
        self.labels_visible = not self.labels_visible

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
        self.update_point_clouds()
        self._update_centroid_colors()

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
    """Load point cloud, dense cloud arrays, and segment annotations."""
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

    # Build individual point clouds for each segment
    pcd_o3d = []
    for ann in segments_anno["segGroups"]:
        obj = pcd.select_by_index(ann["segments"])
        pcd_o3d.append(obj)

    return pcd_o3d, segments_anno, dense_points, dense_colors


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


@hydra.main(version_base=None, config_path="conf", config_name="visualizer")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    path = Path(cfg.map_path)
    clip_ft = np.load(path / "clip_features.npy")
    pcd_o3d, segments_anno, dense_points, dense_colors = load_point_cloud(path)

    log.info(f"Loading map with a total of {len(pcd_o3d)} objects")

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
