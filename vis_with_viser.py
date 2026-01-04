"""
Visualizer using Viser (web-based 3D visualization).
This is a port of visualizer.py from Open3D to Viser.
"""

import hydra
import torch
from omegaconf import DictConfig
import logging
import json

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

    def __init__(self, server: viser.ViserServer, pcd_o3d, clip_ft, ft_extractor):
        self.server = server
        self.ft_extractor = ft_extractor

        # Store point cloud data
        self.pcd = pcd_o3d
        self.num_objects = len(self.pcd)

        # Extract points and colors from Open3D point clouds
        self.points_list = [np.asarray(p.points).astype(np.float32) for p in self.pcd]
        self.og_colors = [np.asarray(p.colors).astype(np.float32) for p in self.pcd]

        # Compute centroids and bounding boxes
        self.centroids = [np.mean(pts, axis=0) for pts in self.points_list]
        self.bboxes = self._compute_bboxes()

        # Generate random colors for each object
        self.random_colors = np.random.rand(self.num_objects, 3).astype(np.float32)

        # Similarity data
        self.sim_query = 0.5 * np.ones(self.num_objects)

        # Semantic features
        device = ft_extractor.device if ft_extractor is not None else "cpu"
        self.semantic_tensor = torch.from_numpy(clip_ft).to(device)
        self.semantic_sim = CosineSimilarity01()

        # Track current visualization handles
        self.pcd_handles = []
        self.bbox_handles = []
        self.centroid_handles = []
        self.label_handles = []

        # Toggle states
        self.bbox_visible = False
        self.centroid_visible = False
        self.labels_visible = False
        self.current_color_mode = "rgb"  # "rgb", "random", "similarity"

        # Current colors (start with original colors)
        self.current_colors = [c.copy() for c in self.og_colors]

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

            # Create line segments
            points = []
            for e1, e2 in edges:
                points.append(corners_world[e1])
                points.append(corners_world[e2])
            points = np.array(points, dtype=np.float32)

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

    def add_labels(self):
        """Add text labels to the scene."""
        self.remove_labels()
        for i, centroid in enumerate(self.centroids):
            handle = self.server.scene.add_label(
                name=f"/labels/label_{i}",
                text=str(i),
                position=centroid.astype(np.float32) + np.array([0, 0, 0.05], dtype=np.float32),
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

    def toggle_labels(self):
        """Toggle label visibility."""
        if self.labels_visible:
            self.remove_labels()
        else:
            self.add_labels()
        self.labels_visible = not self.labels_visible

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

    def query(self, query_text: str):
        """Perform CLIP query and update similarity visualization."""
        if self.ft_extractor is None:
            log.warning("No feature extractor provided.")
            return

        query_ft = self.ft_extractor.encode_text([query_text])
        self.sim_query = self.semantic_sim(query_ft, self.semantic_tensor)
        self.sim_query = self.sim_query.squeeze().cpu().numpy()
        self.set_similarity_colors()
        log.info(f"Query: '{query_text}' - Top match: Object {np.argmax(self.sim_query)}")


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

    return pcd_o3d


def setup_gui(server: viser.ViserServer, manager: ViserCallbackManager):
    """Set up the Viser GUI controls."""

    # Add folder for visualization controls
    with server.gui.add_folder("Visualization"):
        # Color mode buttons
        rgb_btn = server.gui.add_button("RGB Colors")
        random_btn = server.gui.add_button("Random Colors")
        sim_btn = server.gui.add_button("Similarity Colors")

        @rgb_btn.on_click
        def _(_):
            manager.set_rgb_colors()

        @random_btn.on_click
        def _(_):
            manager.set_random_colors()

        @sim_btn.on_click
        def _(_):
            manager.set_similarity_colors()

    with server.gui.add_folder("Toggles"):
        # Toggle buttons
        bbox_btn = server.gui.add_button("Toggle Bounding Boxes")
        centroid_btn = server.gui.add_button("Toggle Centroids")
        label_btn = server.gui.add_button("Toggle Labels")

        @bbox_btn.on_click
        def _(_):
            manager.toggle_bbox()

        @centroid_btn.on_click
        def _(_):
            manager.toggle_centroids()

        @label_btn.on_click
        def _(_):
            manager.toggle_labels()

    with server.gui.add_folder("CLIP Query"):
        # Query input
        query_input = server.gui.add_text("Query", initial_value="")
        query_btn = server.gui.add_button("Search")

        @query_btn.on_click
        def _(_):
            if query_input.value.strip():
                manager.query(query_input.value.strip())


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


def take_screenshot(manager: ViserCallbackManager, path: Path):
    """Take screenshots in different color modes (requires manual capture in Viser)."""
    log.info("Screenshot mode - Viser runs in browser. Use browser screenshot tools.")
    log.info("Setting up color modes for manual capture...")

    # RGB mode
    manager.set_rgb_colors()
    log.info("RGB colors set - take screenshot now (cg_rgb.png)")
    time.sleep(2)

    # Random color mode
    manager.set_random_colors()
    log.info("Random colors set - take screenshot now (cg_random_color.png)")
    time.sleep(2)

    log.info("Screenshot setup complete.")


@hydra.main(version_base=None, config_path="conf", config_name="visualizer")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    path = Path(cfg.map_path)
    clip_ft = np.load(path / "clip_features.npy")
    pcd_o3d = load_point_cloud(path)
    ft_extractor = (
        hydra.utils.instantiate(cfg.ft_extraction)
        if hasattr(cfg, "ft_extraction")
        else None
    )

    log.info(f"Loading map with a total of {len(pcd_o3d)} objects")

    # Create Viser server
    server = viser.ViserServer(host="0.0.0.0", port=8080)
    log.info("Viser server started at http://localhost:8080")

    # Create callback manager
    manager = ViserCallbackManager(
        server=server,
        pcd_o3d=pcd_o3d,
        clip_ft=clip_ft,
        ft_extractor=ft_extractor,
    )

    # Add initial point clouds
    manager.add_point_clouds()

    # Setup GUI controls
    setup_gui(server, manager)

    # Handle different modes
    if cfg.mode == "audio":
        setup_audio_mode(server, manager, cfg)
    elif cfg.mode == "offline_screenshot":
        take_screenshot(manager, path)

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
