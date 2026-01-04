import hydra
import torch
from omegaconf import DictConfig
import logging
import json

from pathlib import Path
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import copy

from concept_graphs.utils import load_map, set_seed
from concept_graphs.viz.utils import similarities_to_rgb
from concept_graphs.mapping.similarity.semantic import CosineSimilarity01
import threading
import os
import sys
import time

from listen_for_keyword import VoskModel

# A logger for this file
log = logging.getLogger(__name__)


class CallbackManager:
    def __init__(self, pcd_o3d, clip_ft, ft_extractor, mode):
        self.ft_extractor = ft_extractor
        self.mode = mode

        # Geometries
        self.pcd = pcd_o3d
        self.bbox = [pcd.get_oriented_bounding_box() for pcd in self.pcd]

        self.pcd_names = [f"pcd_{i}" for i in range(len(self.pcd))]
        self.bbox_names = [f"bbox_{i}" for i in range(len(self.pcd))]
        self.centroid_names = [f"centroid_{i}" for i in range(len(self.pcd))]
        self.label_names = [str(i) for i in range(len(self.pcd))]
        self.centroids = []
        self.label_coord = []
        for p in self.pcd:
            centroid = o3d.geometry.TriangleMesh.create_sphere(radius=0.03)
            c = np.mean(np.asarray(p.points), axis=0)
            centroid.translate(c)
            self.centroids.append(centroid)
            self.label_coord.append(c)

        # Change color to black
        for b in self.bbox:
            b.color = (0, 0, 0)

        # Colorings
        self.og_colors = [
            o3d.utility.Vector3dVector(copy.deepcopy(p.colors)) for p in self.pcd
        ]
        self.sim_query = 0.5 * np.ones(len(self.pcd))
        self.random_colors = np.random.rand(len(self.pcd), 3)

        # Color centroids
        for c, color in zip(self.centroids, self.random_colors):
            c.paint_uniform_color(color)

        # Similarities
        device = ft_extractor.device if ft_extractor is not None else "cpu"
        self.semantic_tensor = torch.from_numpy(clip_ft).to(device)
        self.semantic_sim = CosineSimilarity01()

        # Toggles
        self.bbox_toggle = False
        self.centroid_toggle = False
        self.number_toggle = False

    def add_geometries(self, vis, geometry_names, geometries):
        if self.mode == "keycallback":
            for geometry in geometries:
                vis.add_geometry(geometry)
        elif self.mode in ["gui"]:
            for name, geometry in zip(geometry_names, geometries):
                vis.add_geometry(name, geometry)
        elif self.mode in ["audio", "offline_screenshot"]:
            # vis is the OffscreenRenderer.scene
            for name, geometry in zip(geometry_names, geometries):
                mat = rendering.MaterialRecord()
                mat.shader = "defaultUnlit"
                vis.add_geometry(name, geometry, mat)

    def remove_geometries(self, vis, geometry_names, geometries):
        if self.mode == "keycallback":
            for geometry in geometries:
                vis.remove_geometry(geometry)
        elif self.mode in ["gui", "audio", "offline_screenshot"]:
            for name in geometry_names:
                vis.remove_geometry(name)

    def update_geometries(self, vis, geometry_names, geometries):
        if self.mode == "keycallback":
            for geometry in geometries:
                vis.update_geometry(geometry)
        elif self.mode in ["gui", "audio", "offline_screenshot"]:
            self.remove_geometries(vis, geometry_names, geometries)
            self.add_geometries(vis, geometry_names, geometries)

    def toggle_bbox(self, vis):
        if not self.bbox_toggle:
            self.add_geometries(vis, self.bbox_names, self.bbox)
        else:
            self.remove_geometries(vis, self.bbox_names, self.bbox)
        self.bbox_toggle = not self.bbox_toggle

    def toggle_centroids(self, vis):
        if not self.centroid_toggle:
            self.add_geometries(vis, self.centroid_names, self.centroids)
        else:
            self.remove_geometries(vis, self.centroid_names, self.centroids)
        self.centroid_toggle = not self.centroid_toggle

    def toggle_numbers(self, vis):
        if not self.number_toggle:
            for c, n in zip(self.label_coord, self.label_names):
                vis.add_3d_label(c, n)
        else:
            vis.clear_3d_labels()
        self.number_toggle = not self.number_toggle

    def toggle_sim(self, vis):
        rgb = similarities_to_rgb(self.sim_query, cmap_name="inferno")
        for p, c, color in zip(self.pcd, self.centroids, rgb):
            p.paint_uniform_color(np.array(color) / 255)
            c.paint_uniform_color(np.array(color) / 255)
        self.update_geometries(vis, self.pcd_names, self.pcd)
        if self.centroid_toggle:
            self.update_geometries(vis, self.centroid_names, self.centroids)

    def toggle_random_color(self, vis):
        for p, c, color in zip(self.pcd, self.centroids, self.random_colors):
            p.paint_uniform_color(color)
            c.paint_uniform_color(color)
        self.update_geometries(vis, self.pcd_names, self.pcd)
        if self.centroid_toggle:
            self.update_geometries(vis, self.centroid_names, self.centroids)

    def toggle_rgb(self, vis):
        for p, c in zip(self.pcd, self.og_colors):
            p.colors = c
        self.update_geometries(vis, self.pcd_names, self.pcd)

    def query(self, vis, query=None):
        if self.ft_extractor is None:
            log.warning("No feature extractor provided.")
            return
        if query is None:
            query = input("Enter query: ")
        query_ft = self.ft_extractor.encode_text([query])
        self.sim_query = self.semantic_sim(query_ft, self.semantic_tensor)
        self.sim_query = self.sim_query.squeeze().cpu().numpy()
        self.toggle_sim(vis)

    def view(self, vis):
        pass
        obj_id = input("Object Id: ")
        obj_id = int(obj_id)
        pass

    def register_callbacks(self, vis):
        if self.mode == "keycallback":
            vis.register_key_callback(ord("R"), self.toggle_rgb)
            vis.register_key_callback(ord("Z"), self.toggle_random_color)
            vis.register_key_callback(ord("S"), self.toggle_sim)
            vis.register_key_callback(ord("B"), self.toggle_bbox)
            vis.register_key_callback(ord("C"), self.toggle_centroids)
            vis.register_key_callback(ord("Q"), self.query)
            vis.register_key_callback(ord("V"), self.view)
        else:
            vis.add_action("RGB", self.toggle_rgb)
            vis.add_action("Random Color", self.toggle_random_color)
            vis.add_action("Similarity", self.toggle_sim)
            vis.add_action("Toggle Bbox", self.toggle_bbox)
            vis.add_action("Toggle Centroid", self.toggle_centroids)
            vis.add_action("Toggle Number", self.toggle_numbers)
            vis.add_action("CLIP Query", self.query)
            vis.add_action("View Segments", self.view)


def load_point_cloud(path):
    path = Path(path)
    pcd = o3d.io.read_point_cloud(str(path / "point_cloud.pcd"))

    with open(path / "segments_anno.json", "r") as f:
        segments_anno = json.load(f)

    # Build a pcd with random colors
    pcd_o3d = []

    for ann in segments_anno["segGroups"]:
        obj = pcd.select_by_index(ann["segments"])
        pcd_o3d.append(obj)

    return pcd_o3d


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

    # Callback Manager
    manager = CallbackManager(
        pcd_o3d=pcd_o3d, clip_ft=clip_ft, ft_extractor=ft_extractor, mode=cfg.mode
    )

    # Visualizer
    if cfg.mode == "keycallback":
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=f"Open3D", width=1280, height=720)

        manager.add_geometries(vis, manager.pcd_names, manager.pcd)
        manager.register_callbacks(vis)
        vis.run()
    elif cfg.mode == "gui":
        app = gui.Application.instance
        app.initialize()

        vis = o3d.visualization.O3DVisualizer("Open3D - 3D Text", 1024, 768)
        vis.set_background([1.0, 1.0, 1.0, 1.0], bg_image=None)
        vis.show_settings = True
        vis.show_skybox(False)
        vis.enable_raw_mode(True)
        manager.add_geometries(vis, manager.pcd_names, manager.pcd)
        manager.register_callbacks(vis)
        # for idx in range(0, len(points.points)):
        #     vis.add_3d_label(points.points[idx], "{}".format(idx))
        vis.reset_camera_to_default()

        app.add_window(vis)
        app.run()
    elif cfg.mode == "audio":
        app = gui.Application.instance
        app.initialize()
        window = gui.Application.instance.create_window(
            "Open3D", 1024, 768)
        w = window

        # 3D widget
        _scene = gui.SceneWidget()
        _scene.scene = rendering.Open3DScene(w.renderer)

        em = w.theme.font_size
        separation_height = int(round(0.5 * em))
        chat_panel = gui.Vert(
            0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))
        chat_panel.add_child(gui.Label("Rise Orchestration"))

        # Initialize fake chat history
        chat_history = [
            {"user": "Alice", "message": "Can you identify the objects in this scene?"},
            {"user": "Bob", "message": "I see several furniture items and decorations."},
            {"user": "Alice", "message": "What are the colors of the dominant objects?"},
            {"user": "Bob", "message": "The main colors are brown, white, and gray."},
            {"user": "Alice", "message": "Thanks for the analysis!"}
        ]

        # Define colors for each user
        user_colors = {
            "Alice": gui.Color(0.2, 0.4, 0.8, 1.0),  # Blue
            "Bob": gui.Color(0.8, 0.2, 0.2, 1.0),    # Red
        }

        messages_to_show = chat_history[-5:] if len(chat_history) > 5 else chat_history
        for chat in messages_to_show:
            msg_label = gui.Label(f"{chat['user']}: {chat['message']}")
            user = chat['user']
            if user in user_colors:
                msg_label.text_color = user_colors[user]
            chat_panel.add_child(msg_label)
        chat_panel.add_fixed(separation_height)

        def _on_layout(layout_context):
            r = window.content_rect
            _scene.frame = r
            width = 24 * layout_context.theme.font_size
            height = min(
                r.height,
                2 * chat_panel.calc_preferred_size(
                    layout_context, gui.Widget.Constraints()).height)
            chat_panel.frame = gui.Rect(r.get_right() - width, r.y, width, height)
        w.set_on_layout(_on_layout)
        w.add_child(_scene)
        w.add_child(chat_panel)

        manager.add_geometries(_scene.scene, manager.pcd_names, manager.pcd )

        render_app = lambda : app.run_one_tick()

        from openai import OpenAI
        import openai
        openai.api_key = os.getenv("OPENAI_API_KEY")
        client = OpenAI()
        vosk_model = VoskModel(f"{cfg.cache_dir}/vosk")

        audio_routine_results = []
        def run_audio_routine():
            hit = vosk_model.listen_for_keywords(keywords=["hey", "rise"])
            hit, record_file = vosk_model.listen_for_keywords(keywords=["please"], record=True)

            with open(record_file, "rb") as open_record_file:
                transcription = client.audio.transcriptions.create(
                    model="gpt-4o-transcribe",
                    file=open_record_file,
                    response_format="text"
                )

            audio_routine_results.append(transcription)
            print(transcription)
        audio_routine = threading.Thread(target=run_audio_routine)
        audio_routine.start()
        while True:
            if not audio_routine.is_alive():
                query = audio_routine_results.pop()
                manager.query(_scene.scene, query)
                render_app()
                audio_routine = threading.Thread(target=run_audio_routine)
                audio_routine.start()

            render_app()
            time.sleep(0.1)


    elif cfg.mode == "offline_screenshot":
        width, height = 1024, 768
        vis = rendering.OffscreenRenderer(width, height)
        scene = vis.scene
        scene.set_background([1.0, 1.0, 1.0, 1.0])

        # Add all point clouds using CallbackManager
        manager.add_geometries(scene, manager.pcd_names, manager.pcd)

        # Camera setup: fit to geometry
        bounds = scene.bounding_box
        center = bounds.get_center()
        extent = bounds.get_extent()
        eye = center + [0, 0, max(extent)]
        up = [0, 1, 0]
        scene.camera.look_at(center, eye, up)

        # --- RGB screenshot ---
        manager.toggle_rgb(scene)
        img = vis.render_to_image()
        o3d.io.write_image(str(path / "cg_rgb.png"), img)

        # --- Random color screenshot ---
        manager.toggle_random_color(scene)
        img = vis.render_to_image()
        o3d.io.write_image(str(path / "cg_random_color.png"), img)

        print("Screenshots taken (headless).")

    else:
        raise ValueError("Invalid mode.")


if __name__ == "__main__":
    main()
