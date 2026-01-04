import hydra
import torch
from omegaconf import DictConfig
import logging
import json
import html
import difflib
import re

from pathlib import Path
import numpy as np
import open3d as o3d
import copy
import time
import threading
import os

import viser
import viser.transforms as tf

from concept_graphs.utils import load_map, set_seed
from concept_graphs.viz.utils import similarities_to_rgb
from concept_graphs.mapping.similarity.semantic import CosineSimilarity01

# A logger for this file
log = logging.getLogger(__name__)


class ViserCallbackManager:
    """Manages point cloud visualizations and callbacks for Viser."""

    def __init__(self, server: viser.ViserServer, pcd_o3d, clip_ft, ft_extractor=None, segments_anno=None, axes_ordering="xzy"):
        self.server = server
        self.ft_extractor = ft_extractor  # Might be None initially
        self.axes_ordering = axes_ordering

        # Store point cloud data
        self.pcd = pcd_o3d
        self.num_objects = len(self.pcd)

        # Extract points and colors from Open3D point clouds
        self.points_list = [np.asarray(p.points).astype(np.float32) for p in self.pcd]
        self.og_colors = [np.asarray(p.colors).astype(np.float32) for p in self.pcd]
        
        # Apply axes transformation
        if axes_ordering != "xyz":
            axis_map = {'x': 0, 'y': 1, 'z': 2}
            try:
                indices = [axis_map[ax.lower()] for ax in axes_ordering]
                self.points_list = [pts[:, indices] for pts in self.points_list]
            except (KeyError, IndexError):
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
        self.current_color_mode = "rgb"  # "rgb", "random", "similarity"

        # Current colors (start with original colors)
        self.current_colors = [c.copy() for c in self.og_colors]
        
        # Store segment annotations for labels and captions
        self.segments_anno = segments_anno
        self.labels = self._extract_labels()
        self.captions = self._extract_captions()
        self.search_corpus = self._build_text_corpus()

        # Chat history for GUI chat panel
        self.chat_messages = []
        self.chat_markdown_handle = None

        # Fixed palette for LLM search highlights (green, yellow, red, black)
        self.llm_palette = [
            np.array([0.0, 0.8, 0.0], dtype=np.float32),  # top match
            np.array([0.95, 0.85, 0.0], dtype=np.float32),  # second
            np.array([0.9, 0.1, 0.1], dtype=np.float32),  # third
        ]
        self.llm_default_color = np.zeros(3, dtype=np.float32)

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
        self.add_chat_message("System", msg)

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
    
    def _extract_labels(self):
        """Extract labels from segment annotations."""
        if self.segments_anno is None:
            return ["" for _ in range(self.num_objects)]
        
        labels = []
        for seg_group in self.segments_anno.get("segGroups", []):
            label = seg_group.get("label", "")
            if label == "empty":
                # Try caption if label is empty
                label = seg_group.get("caption", "")
            labels.append(label)
        
        return labels
    
    def _extract_captions(self):
        """Extract captions from segment annotations."""
        if self.segments_anno is None:
            return ["" for _ in range(self.num_objects)]
        
        captions = []
        for seg_group in self.segments_anno.get("segGroups", []):
            caption = seg_group.get("caption", "")
            captions.append(caption)
        
        return captions

    def _build_text_corpus(self):
        """Combine labels and captions for lightweight text search."""
        corpus = []
        for label, caption in zip(self.labels, self.captions):
            pieces = [label.strip(), caption.strip()]
            combined = " ".join([p for p in pieces if p])
            corpus.append(combined)
        return corpus

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(text.lower().split())

    @staticmethod
    def _tokenize(text: str):
        return set(re.findall(r"\b\w+\b", text))

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
        """Add all point clouds to the scene."""
        self.remove_point_clouds()
        for i, (pts, colors) in enumerate(zip(self.points_list, self.current_colors)):
            # Expand colors to per-point if uniform
            if colors.ndim == 1:
                colors_expanded = np.tile(colors, (len(pts), 1))
            else:
                colors_expanded = colors

            # Convert to uint8 RGB
            colors_uint8 = (np.clip(colors_expanded, 0, 1) * 255).astype(np.uint8)

            handle = self.server.scene.add_point_cloud(
                name=f"/pointclouds/pcd_{i}",
                points=pts,
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

    def set_rgb_colors(self):
        """Set original RGB colors."""
        self.current_colors = [c.copy() for c in self.og_colors]
        self.current_color_mode = "rgb"
        self.update_point_clouds()
        self._update_centroid_colors()

    def set_random_colors(self):
        """Set random uniform colors per object."""
        self.current_colors = [
            np.tile(self.random_colors[i], (len(self.points_list[i]), 1))
            for i in range(self.num_objects)
        ]
        self.current_color_mode = "random"
        self.update_point_clouds()
        self._update_centroid_colors()

    def set_similarity_colors(self):
        """Set similarity-based colors."""
        rgb = similarities_to_rgb(self.sim_query, cmap_name="inferno")
        self.current_colors = [
            np.tile(np.array(rgb[i]) / 255.0, (len(self.points_list[i]), 1)).astype(np.float32)
            for i in range(self.num_objects)
        ]
        self.current_color_mode = "similarity"
        self.update_point_clouds()

        # Also update centroid colors
        if self.centroid_visible:
            self.remove_centroids()
            # Temporarily override random_colors for centroids
            original_random = self.random_colors.copy()
            self.random_colors = np.array([np.array(c) / 255.0 for c in rgb], dtype=np.float32)
            self.add_centroids()
            self.random_colors = original_random

    def _update_centroid_colors(self):
        """Update centroid colors to match current scheme."""
        if self.centroid_visible:
            self.remove_centroids()
            self.add_centroids()

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

        query_ft = self.ft_extractor.encode_text([query_text])
        self.sim_query = self.semantic_sim(query_ft, self.semantic_tensor)
        self.sim_query = self.sim_query.squeeze().cpu().numpy()
        self.set_similarity_colors()
        log.info(f"Query: '{query_text}' - Top match: Object {np.argmax(self.sim_query)}")

    def _score_llm_query(self, query_text: str) -> np.ndarray:
        """Approximate text similarity over labels+captions without a model."""
        norm_query = self._normalize_text(query_text)
        if not norm_query:
            return np.zeros(self.num_objects, dtype=np.float32)

        query_tokens = self._tokenize(norm_query)
        scores = np.zeros(self.num_objects, dtype=np.float32)

        for idx, entry in enumerate(self.search_corpus):
            if not entry:
                continue

            norm_entry = self._normalize_text(entry)
            entry_tokens = self._tokenize(norm_entry)

            # Blend overlap and sequence similarity to rank lightweight matches.
            overlap = 0.0
            if entry_tokens:
                overlap = len(query_tokens & entry_tokens) / max(len(query_tokens | entry_tokens), 1)

            seq_ratio = difflib.SequenceMatcher(None, norm_query, norm_entry).ratio()
            scores[idx] = 0.6 * seq_ratio + 0.4 * overlap

        return scores

    def _apply_llm_palette(self, ranked_indices):
        """Apply fixed palette: green, yellow, red, else black."""
        centroid_colors = []
        updated_colors = []

        for obj_idx in range(self.num_objects):
            color = self.llm_default_color
            if ranked_indices:
                if obj_idx == ranked_indices[0]:
                    color = self.llm_palette[0]
                elif len(ranked_indices) > 1 and obj_idx == ranked_indices[1]:
                    color = self.llm_palette[1]
                elif len(ranked_indices) > 2 and obj_idx == ranked_indices[2]:
                    color = self.llm_palette[2]

            color = color.astype(np.float32)
            updated_colors.append(np.tile(color, (len(self.points_list[obj_idx]), 1)))
            centroid_colors.append(color)

        self.current_colors = updated_colors
        self.current_color_mode = "llm"
        self.update_point_clouds()

        if self.centroid_visible:
            self.remove_centroids()
            original_random = self.random_colors.copy()
            self.random_colors = np.array(centroid_colors, dtype=np.float32)
            self.add_centroids()
            self.random_colors = original_random

    def llm_query(self, query_text: str, client=None):
        """Search labels+captions, highlight top 3 with fixed colors."""
        scores = self._score_llm_query(query_text)
        if scores.max(initial=0.0) <= 0.0:
            msg = "No textual metadata available to match this query."
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

        ranked = list(np.argsort(scores)[::-1][:3])
        self._apply_llm_palette(ranked)

        summary = ", ".join([f"{idx} ({scores[idx]:.2f})" for idx in ranked])
        log.info(f"LLM query: '{query_text}' - Top: {summary}")
        self.notify_clients(
            title="LLM Query",
            body=f"Top match: {ranked[0]} (score {scores[ranked[0]]:.2f})",
            client=client,
            auto_close_seconds=5.0,
        )

    def respond(self, message: str) -> str:
        """Generate an agent response (placeholder)."""
        reply = "OK"
        self.add_chat_message("Agent", reply)
        return reply


def load_point_cloud(path):
    """Load point cloud and segment annotations."""
    path = Path(path)
    pcd = o3d.io.read_point_cloud(str(path / "point_cloud.pcd"))

    with open(path / "segments_anno.json", "r") as f:
        segments_anno = json.load(f)

    # Build individual point clouds for each segment
    pcd_o3d = []
    for ann in segments_anno["segGroups"]:
        obj = pcd.select_by_index(ann["segments"])
        pcd_o3d.append(obj)

    return pcd_o3d, segments_anno


def setup_gui(server: viser.ViserServer, manager: ViserCallbackManager):
    """Set up the Viser GUI controls."""

    # Add folder for visualization controls
    with server.gui.add_folder("Visualization"):
        # Color mode buttons
        rgb_btn = server.gui.add_button("RGB Colors")
        random_btn = server.gui.add_button("Random Colors")
        #sim_btn = server.gui.add_button("Similarity Colors")

        @rgb_btn.on_click
        def _(_):
            manager.set_rgb_colors()

        @random_btn.on_click
        def _(_):
            manager.set_random_colors()

        #@sim_btn.on_click
        #def _(_):
        #    manager.set_similarity_colors()

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
        chat_input = server.gui.add_text("Message", initial_value="", multiline=True)
        send_btn = server.gui.add_button("Send")

        manager.register_chat_markdown(chat_history)

        @send_btn.on_click
        def _(_):
            user_message = chat_input.value.strip()
            if not user_message:
                return

            manager.add_chat_message("User", user_message)
            manager.respond(user_message)
            chat_input.value = ""


def setup_audio_mode(server: viser.ViserServer, manager: ViserCallbackManager, cfg: DictConfig):
    """Set up audio mode with voice commands."""
    from openai import OpenAI
    import openai
    from listen_for_keyword import VoskModel

    openai.api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI()
    vosk_model = VoskModel(f"{cfg.cache_dir}/vosk")

    # Add chat panel to GUI
    with server.gui.add_folder("Voice Chat"):
        status_label = server.gui.add_text("Status", initial_value="Listening for 'hey rise'...")
        last_query = server.gui.add_text("Last Query", initial_value="")

    def run_audio_routine():
        while True:
            try:
                # Listen for wake word
                status_label.value = "Listening for 'hey rise'..."
                vosk_model.listen_for_keywords(keywords=["hey", "rise"])

                # Listen for command
                status_label.value = "Listening for command (say 'please' when done)..."
                hit, record_file = vosk_model.listen_for_keywords(keywords=["please"], record=True)

                # Transcribe
                status_label.value = "Transcribing..."
                with open(record_file, "rb") as open_record_file:
                    transcription = client.audio.transcriptions.create(
                        model="gpt-4o-transcribe",
                        file=open_record_file,
                        response_format="text"
                    )

                # Execute query
                query = transcription.strip()
                last_query.value = query
                log.info(f"Voice query: {query}")
                manager.query(query)

            except Exception as e:
                log.error(f"Audio routine error: {e}")
                status_label.value = f"Error: {e}"
                time.sleep(1)

    # Start audio thread
    audio_thread = threading.Thread(target=run_audio_routine, daemon=True)
    audio_thread.start()



@hydra.main(version_base=None, config_path="conf", config_name="visualizer")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    path = Path(cfg.map_path)
    clip_ft = np.load(path / "clip_features.npy")
    pcd_o3d, segments_anno = load_point_cloud(path)

    log.info(f"Loading map with a total of {len(pcd_o3d)} objects")

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
    )

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

    # Handle different modes
    if cfg.mode == "audio":
        setup_audio_mode(server, manager, cfg)

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
