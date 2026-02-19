import logging
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Callable

import numpy as np
from matplotlib.pyplot import get_cmap
from omegaconf import DictConfig

if TYPE_CHECKING:  # pragma: no cover
    from vis.vis_with_viser import ViserCallbackManager
    from vis.box_annotator import BoxAnnotatorController

log = logging.getLogger(__name__)


class GUIStateController:
    """Tracks GUI toggle state and manages interactions between display layers."""

    def __init__(self, manager: "ViserCallbackManager") -> None:
        self.manager = manager
        self.map_switch_handlers: list[Callable[[Path], None]] = []
        self.similarity_checkbox = None
        self.cmap_html_handle = None
        self.rgb_checkbox = None
        self.random_checkbox = None
        self.dense_checkbox = None
        self.point_noise_checkbox = None
        self.floor_segment_checkbox = None
        self.point_size_slider = None
        self.point_count_slider = None
        self.bbox_btn = None
        self.centroid_btn = None
        self.id_btn = None
        self.label_btn = None
        self.caption_btn = None
        self._map_switch_lock = threading.Lock()

    def set_similarity_checkbox(self, checkbox):
        """Store reference to the similarity checkbox for enabling/disabling."""
        self.similarity_checkbox = checkbox

    def set_cmap_html_handle(self, handle):
        """Store reference to the colormap HTML handle created in setup_gui."""
        self.cmap_html_handle = handle

    def set_visual_controls(
        self,
        *,
        rgb_checkbox,
        random_checkbox,
        similarity_checkbox,
        dense_checkbox,
        point_noise_checkbox,
        floor_segment_checkbox,
        point_size_slider,
        point_count_slider,
    ):
        """Register visualization control handles for runtime refreshes."""
        self.rgb_checkbox = rgb_checkbox
        self.random_checkbox = random_checkbox
        self.similarity_checkbox = similarity_checkbox
        self.dense_checkbox = dense_checkbox
        self.point_noise_checkbox = point_noise_checkbox
        self.floor_segment_checkbox = floor_segment_checkbox
        self.point_size_slider = point_size_slider
        self.point_count_slider = point_count_slider

    def set_toggle_controls(self, *, bbox_btn, centroid_btn, id_btn, label_btn, caption_btn):
        """Register toggle button handles for runtime refreshes."""
        self.bbox_btn = bbox_btn
        self.centroid_btn = centroid_btn
        self.id_btn = id_btn
        self.label_btn = label_btn
        self.caption_btn = caption_btn

    def refresh_controls(self):
        """Refresh GUI control enabled/value states after map changes."""
        has_any_cloud = self.manager.has_segment_objects or self.manager.has_dense_cloud
        controls_enabled = self.manager.has_segment_objects

        if self.point_size_slider is not None:
            self.point_size_slider.disabled = not has_any_cloud
            self.point_size_slider.value = self.manager.point_size
        if self.point_count_slider is not None:
            self.point_count_slider.disabled = not self.manager.has_dense_cloud
            self.point_count_slider.value = self.manager.point_count
        if self.point_noise_checkbox is not None:
            self.point_noise_checkbox.disabled = not has_any_cloud
            self.point_noise_checkbox.value = self.manager.point_noise_enabled

        if self.rgb_checkbox is not None:
            self.rgb_checkbox.disabled = not self.manager.has_segment_objects
            self.rgb_checkbox.value = self.manager.rgb_enabled
        if self.random_checkbox is not None:
            self.random_checkbox.disabled = not self.manager.has_segment_objects
            self.random_checkbox.value = self.manager.random_enabled
        if self.similarity_checkbox is not None:
            self.similarity_checkbox.disabled = not self.manager.has_segment_objects
            self.similarity_checkbox.value = self.manager.similarity_enabled
        if self.dense_checkbox is not None:
            self.dense_checkbox.disabled = not self.manager.has_dense_cloud
            self.dense_checkbox.value = self.manager.dense_enabled
        if self.floor_segment_checkbox is not None:
            self.floor_segment_checkbox.disabled = not self.manager.has_dense_cloud
            self.floor_segment_checkbox.value = self.manager.segment_floor_enabled

        if self.bbox_btn is not None:
            self.bbox_btn.disabled = not controls_enabled
        if self.centroid_btn is not None:
            self.centroid_btn.disabled = not controls_enabled
        if self.id_btn is not None:
            self.id_btn.disabled = not controls_enabled
        if self.label_btn is not None:
            self.label_btn.disabled = not controls_enabled
        if self.caption_btn is not None:
            self.caption_btn.disabled = not controls_enabled

        self._set_cmap_visibility(self.manager.similarity_enabled)

    def _set_cmap_visibility(self, visible: bool):
        if self.cmap_html_handle is not None:
            self.cmap_html_handle.visible = visible

    def _update_centroid_colors(self):
        """Update centroid colors to match current scheme."""
        if self.manager.centroid_visible:
            self.manager.remove_centroids()
            self.manager.add_centroids()

    def toggle_bbox(self):
        """Toggle bounding box visibility."""
        if not self.manager.has_segment_objects:
            return
        if self.manager.bbox_visible:
            self.manager.remove_bboxes()
        else:
            self.manager.add_bboxes()
        self.manager.bbox_visible = not self.manager.bbox_visible

    def toggle_centroids(self):
        """Toggle centroid visibility."""
        if not self.manager.has_segment_objects:
            return
        if self.manager.centroid_visible:
            self.manager.remove_centroids()
        else:
            self.manager.add_centroids()
        self.manager.centroid_visible = not self.manager.centroid_visible

    def toggle_ids(self):
        """Toggle ID visibility."""
        if not self.manager.has_segment_objects:
            return
        if self.manager.ids_visible:
            self.manager.remove_ids()
        else:
            self.manager.add_ids()
        self.manager.ids_visible = not self.manager.ids_visible

    def toggle_labels(self):
        """Toggle label visibility."""
        if not self.manager.has_segment_objects:
            return
        if self.manager.labels_visible:
            self.manager.remove_labels()
        else:
            # Add labels per-object using the helper
            for i in range(self.manager.num_objects):
                self.manager.add_label_for_object(i)
        self.manager.labels_visible = not self.manager.labels_visible

    def toggle_captions(self):
        """Toggle caption visibility."""
        if not self.manager.has_segment_objects:
            return
        if self.manager.captions_visible:
            self.manager.remove_captions()
        else:
            self.manager.add_captions()
        self.manager.captions_visible = not self.manager.captions_visible

    def toggle_rgb_mode(self, enabled: bool):
        """Toggle RGB color mode."""
        if enabled and not self.manager.has_segment_objects:
            return
        self.manager.rgb_enabled = enabled
        self.manager.update_point_clouds()
        self._update_centroid_colors()

    def toggle_random_mode(self, enabled: bool):
        """Toggle random color mode."""
        if enabled and not self.manager.has_segment_objects:
            return
        self.manager.random_enabled = enabled
        self.manager.update_point_clouds()
        self._update_centroid_colors()

    def toggle_similarity_mode(self, enabled: bool):
        """Toggle similarity color mode."""
        if enabled and not self.manager.has_segment_objects:
            return
        self.manager.similarity_enabled = enabled
        # If unchecking, disable the checkbox
        if not enabled and self.similarity_checkbox is not None:
            self.similarity_checkbox.disabled = True
        self._set_cmap_visibility(enabled)
        self.manager.update_point_clouds()
        self._update_centroid_colors()

    def toggle_dense_mode(self, enabled: bool) -> bool:
        """Toggle dense point cloud mode."""
        if enabled and self.manager.dense_points is None:
            self.manager.notify_clients(
                title="Dense Point Cloud",
                body="Dense point cloud is not available at this map path.",
                color="yellow",
                with_close_button=True,
                auto_close_seconds=2.0,
            )
            log.warning("Dense point cloud requested but not found.")
            return False
        self.manager.dense_enabled = enabled
        self.manager.update_point_clouds()
        return True

    def toggle_point_noise(self, enabled: bool):
        """Toggle random point offsets used to reduce z-fighting."""
        self.manager.point_noise_enabled = enabled
        self.manager.update_point_clouds()

    def toggle_floor_segment(self, enabled: bool):
        """Toggle floor segmentation overlay mode."""
        if enabled and not self.manager.has_dense_cloud:
            return
        self.manager.segment_floor_enabled = enabled
        self.manager.update_point_clouds()

    def enable_similarity_mode(self):
        """Enable and check similarity mode after a query."""
        self.manager.similarity_enabled = True
        if self.similarity_checkbox is not None:
            self.similarity_checkbox.disabled = False
            self.similarity_checkbox.value = True
        self._set_cmap_visibility(True)
        self.manager.update_point_clouds()
        self._update_centroid_colors()

    def register_map_switch_handler(self, handler: Callable[[Path], None] | None):
        """Register a callback invoked after a map is successfully switched."""
        if callable(handler):
            self.map_switch_handlers.append(handler)

    def configure_map_switch_refreshes(
        self,
        *,
        video_root_ref: dict | None = None,
        refresh_data_collection: Callable[[Path], None] | None = None,
        refresh_notes: Callable[[Path], None] | None = None,
        box_annotator_refresh: Callable[[Path], None] | None = None,
    ):
        """Register standard map-switch refresh behavior used by the app."""

        def _refresh(new_map_path: Path):
            if video_root_ref is not None:
                video_root_ref["path"] = new_map_path.resolve()
            if callable(refresh_data_collection):
                refresh_data_collection(new_map_path)
            if callable(refresh_notes):
                refresh_notes(new_map_path)
            if callable(box_annotator_refresh):
                box_annotator_refresh(new_map_path)

        self.register_map_switch_handler(_refresh)

    def _run_map_switch_handlers(self, selected_map: Path):
        """Run all registered map-switch handlers safely."""
        for handler in self.map_switch_handlers:
            try:
                handler(selected_map)
            except Exception as exc:  # noqa: BLE001
                log.warning("map switch handler failed: %s", exc)

    def switch_to_map(self, map_path: Path, cfg: DictConfig):
        """Switch to another saved map in-process and refresh scene/UI."""
        if not self._map_switch_lock.acquire(blocking=False):
            log.info("Map switch already in progress; skipping request for %s", map_path)
            return

        try:
            selected = Path(map_path).resolve()
            current = self.manager.map_path.resolve() if self.manager.map_path is not None else None
            if current is not None and selected == current:
                return

            self.manager.notify_clients(
                title="Project Selector",
                body=f"Loading map: {selected}",
                color="blue",
                auto_close_seconds=2.0,
            )

            from vis.cg_dataclass import ConceptGraphData

            try:
                cg = ConceptGraphData.load(selected, arcs_enabled=cfg.arcs_enabled)
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to load map %s: %s", selected, exc)
                self.manager.notify_clients(
                    title="Project Selector",
                    body=f"Failed to load map: {selected.name}",
                    color="red",
                    with_close_button=True,
                )
                return

            mode_state = {
                "rgb": self.manager.rgb_enabled,
                "random": self.manager.random_enabled,
                "similarity": self.manager.similarity_enabled,
                "dense": self.manager.dense_enabled,
                "segment_floor": self.manager.segment_floor_enabled,
            }
            visibility_state = {
                "bbox": self.manager.bbox_visible,
                "centroid": self.manager.centroid_visible,
                "ids": self.manager.ids_visible,
                "labels": self.manager.labels_visible,
                "captions": self.manager.captions_visible,
                "forward_arcs": self.manager.arcs_gui.forward_arcs_visible,
                "dependency_arcs": self.manager.arcs_gui.dependency_arcs_visible,
            }

            self.manager.remove_arcs()
            self.manager.remove_captions()
            self.manager.remove_labels()
            self.manager.remove_ids()
            self.manager.remove_centroids()
            self.manager.remove_bboxes()
            self.manager.remove_point_clouds()

            self.manager.bbox_visible = False
            self.manager.centroid_visible = False
            self.manager.ids_visible = False
            self.manager.labels_visible = False
            self.manager.captions_visible = False
            self.manager.arcs_gui.forward_arcs_visible = False
            self.manager.arcs_gui.dependency_arcs_visible = False

            self.manager._load_map_data(cg, selected)

            self.manager.rgb_enabled = mode_state["rgb"] and self.manager.has_segment_objects
            self.manager.random_enabled = mode_state["random"] and self.manager.has_segment_objects
            self.manager.similarity_enabled = mode_state["similarity"] and self.manager.has_segment_objects
            self.manager.dense_enabled = mode_state["dense"] and self.manager.has_dense_cloud
            self.manager.segment_floor_enabled = (
                mode_state["segment_floor"] and self.manager.has_dense_cloud
            )

            if not (
                self.manager.rgb_enabled
                or self.manager.random_enabled
                or self.manager.similarity_enabled
                or self.manager.dense_enabled
            ):
                if self.manager.has_segment_objects:
                    self.manager.rgb_enabled = True
                elif self.manager.has_dense_cloud:
                    self.manager.dense_enabled = True

            self.manager.add_point_clouds()

            self.manager.bbox_visible = visibility_state["bbox"] and self.manager.has_segment_objects
            self.manager.centroid_visible = (
                visibility_state["centroid"] and self.manager.has_segment_objects
            )
            self.manager.ids_visible = visibility_state["ids"] and self.manager.has_segment_objects
            self.manager.labels_visible = visibility_state["labels"] and self.manager.has_segment_objects
            self.manager.captions_visible = (
                visibility_state["captions"] and self.manager.has_segment_objects
            )
            self.manager.arcs_gui.forward_arcs_visible = (
                visibility_state["forward_arcs"] and self.manager.can_show_arcs
            )
            self.manager.arcs_gui.dependency_arcs_visible = (
                visibility_state["dependency_arcs"] and self.manager.can_show_arcs
            )

            if self.manager.bbox_visible:
                self.manager.add_bboxes()
            if self.manager.centroid_visible:
                self.manager.add_centroids()
            if self.manager.ids_visible:
                self.manager.add_ids()
            if self.manager.labels_visible:
                for object_id in range(self.manager.num_objects):
                    self.manager.add_label_for_object(object_id)
            if self.manager.captions_visible:
                self.manager.add_captions()
            if self.manager.arcs_gui.forward_arcs_visible:
                self.manager.add_forward_arcs()
            if self.manager.arcs_gui.dependency_arcs_visible:
                self.manager.add_dependency_arcs()

            self.refresh_controls()
            self.manager.arcs_gui.refresh_controls()
            self.manager.query_chat.refresh_controls()

            self._run_map_switch_handlers(selected)

            self.refresh_controls()
            self.manager.arcs_gui.refresh_controls()
            self.manager.query_chat.refresh_controls()

            self.manager.notify_clients(
                title="Project Selector",
                body=f"Loaded map: {selected.name}",
                color="green",
                auto_close_seconds=2.0,
            )
            log.info("Switched map in-process to: %s", selected)
        finally:
            self._map_switch_lock.release()


def setup_visualization_gui(server, manager: "ViserCallbackManager", gui_cfg, cmap_name: str):
    """Set up the visualization controls (color layers + colormap legend)."""
    with server.gui.add_folder("Visualization", expand_by_default=gui_cfg.visualization.expanded):
        point_size_slider = server.gui.add_slider(
            "Point Size",
            min=0.001,
            max=0.05,
            step=0.001,
            initial_value=manager.point_size,
        )
        point_count_slider = server.gui.add_slider(
            "Point Count",
            min=0.1,
            max=1.0,
            step=0.01,
            initial_value=manager.point_count,
        )

        # Color mode checkboxes (stackable)
        rgb_checkbox = server.gui.add_checkbox("RGB Colors", initial_value=manager.rgb_enabled)
        random_checkbox = server.gui.add_checkbox("Random Colors", initial_value=False)
        similarity_checkbox = server.gui.add_checkbox("Similarity Colors", initial_value=False)
        similarity_checkbox.disabled = True
        dense_checkbox = server.gui.add_checkbox("Dense Point Cloud", initial_value=manager.dense_enabled)
        point_noise_checkbox = server.gui.add_checkbox(
            "Random Point Noise", initial_value=manager.point_noise_enabled
        )
        floor_segment_checkbox = server.gui.add_checkbox(
            "segment floor", initial_value=manager.segment_floor_enabled
        )

        has_any_cloud = manager.has_segment_objects or manager.has_dense_cloud
        point_size_slider.disabled = not has_any_cloud
        point_count_slider.disabled = not manager.has_dense_cloud
        point_noise_checkbox.disabled = not has_any_cloud

        rgb_checkbox.disabled = not manager.has_segment_objects
        random_checkbox.disabled = not manager.has_segment_objects
        dense_checkbox.disabled = not manager.has_dense_cloud
        floor_segment_checkbox.disabled = not manager.has_dense_cloud

        # Store reference to similarity checkbox in manager
        manager.set_similarity_checkbox(similarity_checkbox)
        manager.gui_fsm.set_visual_controls(
            rgb_checkbox=rgb_checkbox,
            random_checkbox=random_checkbox,
            similarity_checkbox=similarity_checkbox,
            dense_checkbox=dense_checkbox,
            point_noise_checkbox=point_noise_checkbox,
            floor_segment_checkbox=floor_segment_checkbox,
            point_size_slider=point_size_slider,
            point_count_slider=point_count_slider,
        )

        # Colormap legend (visible only when similarity mode is active)
        cmap = get_cmap(cmap_name)
        stops = np.linspace(0.0, 1.0, 9)
        color_stops = []
        for frac in stops:
            r, g, b, _ = cmap(frac)
            color_stops.append(
                f"rgb({int(r * 255)}, {int(g * 255)}, {int(b * 255)}) {frac * 100:.1f}%"
            )

        gradient_css = ", ".join(color_stops)
        cmap_html_content = f"""
<div style=\"margin-top: 12px; padding: 0 8px; font-family: sans-serif; font-size: 11px;\">
  <div style=\"height: 14px; border-radius: 7px; overflow: hidden; border: 1px solid rgba(0, 0, 0, 0.2); background: linear-gradient(90deg, {gradient_css}); box-shadow: 0 1px 2px rgba(0, 0, 0, 0.2);\"></div>
  <div style=\"display: flex; justify-content: space-between; margin-top: 4px; font-weight: 600; color: #666;\">
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

        @point_noise_checkbox.on_update
        def _(event):
            manager.toggle_point_noise(point_noise_checkbox.value)

        @floor_segment_checkbox.on_update
        def _(event):
            manager.toggle_floor_segment(floor_segment_checkbox.value)

        @point_size_slider.on_update
        def _(event):
            manager.set_point_size(point_size_slider.value)

        @point_count_slider.on_update
        def _(event):
            manager.set_point_count(point_count_slider.value)


def setup_toggle_gui(server, manager: "ViserCallbackManager", gui_cfg):
    """Set up the GUI toggle buttons for display layers."""
    with server.gui.add_folder("Toggles", expand_by_default=gui_cfg.toggles.expanded):
        bbox_btn = server.gui.add_button("Toggle Bounding Boxes")
        centroid_btn = server.gui.add_button("Toggle Centroids")
        id_btn = server.gui.add_button("Toggle IDs")
        label_btn = server.gui.add_button("Toggle Labels")
        caption_btn = server.gui.add_button("Toggle Captions")

        controls_enabled = manager.has_segment_objects
        bbox_btn.disabled = not controls_enabled
        centroid_btn.disabled = not controls_enabled
        id_btn.disabled = not controls_enabled
        label_btn.disabled = not controls_enabled
        caption_btn.disabled = not controls_enabled
        manager.gui_fsm.set_toggle_controls(
            bbox_btn=bbox_btn,
            centroid_btn=centroid_btn,
            id_btn=id_btn,
            label_btn=label_btn,
            caption_btn=caption_btn,
        )

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


def reload_box_annotator_for_map(
    controller: "BoxAnnotatorController",
    selected_map_path: Path,
    *,
    filename: str,
    show_bboxes: bool,
):
    """Reload box annotations for the newly selected map."""
    target = Path(selected_map_path) / filename

    controller.delete_all()
    if target.exists():
        controller.load(target)
    else:
        controller._set_status("No annotation file for this project")
        controller._mark_synced()

    controller.set_bboxes_visible(show_bboxes)


def _is_saved_map_dir(path: Path) -> bool:
    """Return True when a folder looks like a saved concept-graph map."""
    return (path / "point_cloud.pcd").exists() or (path / "dense_point_cloud.pcd").exists()


def _discover_saved_maps(cfg: DictConfig, current_map_path: Path) -> list[Path]:
    """Discover saved map folders from likely output roots."""
    candidate_roots: list[Path] = []

    cfg_output_dir = cfg.get("output_dir", None)
    if cfg_output_dir:
        candidate_roots.append(Path(cfg_output_dir))

    candidate_roots.append(current_map_path.parent)
    candidate_roots.append(Path.cwd() / "outputs")
    candidate_roots.append(Path.cwd() / "DATAPIPES" / "OUTPUT")

    discovered: list[Path] = []
    for root in candidate_roots:
        if not root.exists() or not root.is_dir():
            continue
        if _is_saved_map_dir(root):
            discovered.append(root)
        for cloud_file in root.rglob("point_cloud.pcd"):
            discovered.append(cloud_file.parent)
        for dense_cloud_file in root.rglob("dense_point_cloud.pcd"):
            discovered.append(dense_cloud_file.parent)

    discovered.append(current_map_path)

    unique_by_resolved: dict[Path, Path] = {}
    for map_dir in discovered:
        try:
            resolved = map_dir.resolve()
        except Exception:
            resolved = map_dir
        if resolved not in unique_by_resolved and _is_saved_map_dir(resolved):
            unique_by_resolved[resolved] = resolved

    return sorted(unique_by_resolved.values(), key=lambda p: p.as_posix())


def _format_map_option(map_path: Path) -> str:
    """Format a map path label for GUI dropdown options."""
    cwd = Path.cwd().resolve()
    try:
        return str(map_path.resolve().relative_to(cwd))
    except Exception:
        return str(map_path.resolve())


def setup_project_selector_gui(server, manager: "ViserCallbackManager", cfg: DictConfig, current_map_path: Path):
    """Add Project Selector controls for switching between saved maps."""
    saved_maps = _discover_saved_maps(cfg, current_map_path)
    if not saved_maps:
        return

    current_resolved = current_map_path.resolve()
    current_index = 0
    for idx, map_path in enumerate(saved_maps):
        if map_path.resolve() == current_resolved:
            current_index = idx
            break

    option_labels = [_format_map_option(path) for path in saved_maps]
    label_to_path = {label: path for label, path in zip(option_labels, saved_maps)}
    index_ref = {"value": current_index}

    def _map_name_markdown(map_path: Path) -> str:
        map_name = map_path.resolve().name
        return f"**Map Name:** {map_name}"

    def _reload_with_map(selected_map: Path):
        selected_resolved = selected_map.resolve()
        log.info("Project Selector switching to map: %s", selected_resolved)
        manager.switch_to_map(selected_resolved, cfg)

    with server.gui.add_folder("Project Selector", expand_by_default=True):
        previous_btn = server.gui.add_button("Previous")
        next_btn = server.gui.add_button("Next")
        map_dropdown = server.gui.add_dropdown(
            "Map",
            options=option_labels,
            initial_value=option_labels[current_index],
        )
        map_name_markdown = server.gui.add_markdown(_map_name_markdown(saved_maps[current_index]))

        def _sync_map_name_display(map_path: Path):
            map_name_markdown.content = _map_name_markdown(map_path)

        manager.gui_fsm.register_map_switch_handler(_sync_map_name_display)

        @previous_btn.on_click
        def _(_event):
            next_index = (index_ref["value"] - 1) % len(saved_maps)
            index_ref["value"] = next_index
            map_dropdown.value = option_labels[next_index]

        @next_btn.on_click
        def _(_event):
            next_index = (index_ref["value"] + 1) % len(saved_maps)
            index_ref["value"] = next_index
            map_dropdown.value = option_labels[next_index]

        @map_dropdown.on_update
        def _(_event):
            selected_label = map_dropdown.value
            selected_map = label_to_path.get(selected_label)
            if selected_map is None:
                return
            for idx, candidate in enumerate(saved_maps):
                if candidate == selected_map:
                    index_ref["value"] = idx
                    break
            _sync_map_name_display(selected_map)
            _reload_with_map(selected_map)
