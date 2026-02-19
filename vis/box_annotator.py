"""3D Bounding Box Annotator overlay for Viser.

Provides a web-based 3D bounding box annotation tool on top of point cloud
visualizations, similar to labelCloud but running entirely in the browser via
viser.

Key features:
- Click on point cloud to place a new bounding box
- Central transform gizmo for move + rotate
- Per-face drag handles for intuitive resizing
- GUI panel for label editing, numeric dimension inputs, box list management
- Save / load annotations as JSON
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np
import viser
import viser.transforms as vtf
from vis.gui_fsm import reload_box_annotator_for_map

log = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from vis.notification_manager import NotificationManager

# ──────────────────────────────────────────────────────────────────────────────
# Data model
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class Box3D:
    """A single oriented 3D bounding box annotation."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    label: str = "object"
    center: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    dimensions: np.ndarray = field(
        default_factory=lambda: np.array([0.5, 0.5, 0.5], dtype=np.float64)
    )
    wxyz: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    )
    colour: str = "green"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "label": self.label,
            "center": self.center.tolist(),
            "dimensions": self.dimensions.tolist(),
            "wxyz": self.wxyz.tolist(),
            "colour": self.colour,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Box3D:
        return cls(
            id=d["id"],
            label=d.get("label", "object"),
            center=np.array(d["center"], dtype=np.float64),
            dimensions=np.array(d["dimensions"], dtype=np.float64),
            wxyz=np.array(d["wxyz"], dtype=np.float64),
            colour=str(d.get("colour", "green")),
        )


# ──────────────────────────────────────────────────────────────────────────────
# Per-box scene handles
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _BoxSceneHandles:
    """Viser scene handles associated with a single box."""

    box_handle: Any = None  # viser.BoxHandle
    wire_handle: Any = None  # wireframe BoxHandle
    gizmo_handle: Any = None  # viser.TransformControlsHandle
    label_handle: Any = None  # viser.LabelHandle
    # 8 corner resize handles
    corner_handles: list[Any] = field(default_factory=list)
    corner_gizmo_handles: list[Any] = field(default_factory=list)
    # Tracking for central gizmo delta detection
    prev_gizmo_wxyz: np.ndarray | None = None
    prev_gizmo_position: np.ndarray | None = None


@dataclass
class _MultiSelectionHandles:
    """Viser scene handles for aggregate multi-selection controls."""

    gizmo_handle: Any = None
    wire_handle: Any = None
    corner_handles: list[Any] = field(default_factory=list)
    corner_gizmo_handles: list[Any] = field(default_factory=list)
    prev_gizmo_wxyz: np.ndarray | None = None
    prev_gizmo_position: np.ndarray | None = None
    bbox_min: np.ndarray | None = None
    bbox_max: np.ndarray | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Quaternion helpers
# ──────────────────────────────────────────────────────────────────────────────


def _quat_to_rotmat(wxyz: np.ndarray) -> np.ndarray:
    """Convert (w, x, y, z) quaternion to 3×3 rotation matrix."""
    w, x, y, z = wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3×3 rotation matrix to (w, x, y, z) quaternion."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _corner_gizmo_wxyz(box_wxyz: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Rotation for a corner gizmo so its axes point outward from the box.

    We align the gizmo axes to signed box-local axes. If the resulting
    orientation is a reflection (det < 0), swap two axes to recover a
    proper rotation while keeping all axes pointing outward.
    """
    R_box = _quat_to_rotmat(box_wxyz)
    signed_axes = R_box @ np.diag(signs)
    if np.linalg.det(signed_axes) < 0:
        signed_axes = signed_axes.copy()
        signed_axes[:, [0, 1]] = signed_axes[:, [1, 0]]
    return _rotmat_to_quat(signed_axes)


# ──────────────────────────────────────────────────────────────────────────────
# Controller
# ──────────────────────────────────────────────────────────────────────────────

# The 8 box corners as sign triples (±1 for each local axis)
_CORNER_SIGNS = [
    np.array([+1, +1, +1], dtype=np.float64),
    np.array([+1, +1, -1], dtype=np.float64),
    np.array([+1, -1, +1], dtype=np.float64),
    np.array([+1, -1, -1], dtype=np.float64),
    np.array([-1, +1, +1], dtype=np.float64),
    np.array([-1, +1, -1], dtype=np.float64),
    np.array([-1, -1, +1], dtype=np.float64),
    np.array([-1, -1, -1], dtype=np.float64),
]

# Vertical gap above the box centre for the central gizmo (metres, global Y-up)
_GIZMO_GAP = 2.0


def _gizmo_global_offset() -> np.ndarray:
    """Fixed world-space offset so the gizmo always floats above the box
    along the global Y axis, regardless of box orientation."""
    return np.array([0.0, 0.0, _GIZMO_GAP], dtype=np.float64)

# Colors
_BOX_COLOR = (0, 200, 50)
_BOX_COLOR_SELECTED = (255, 165, 0)
_HANDLE_COLOR = (255, 180, 0)
_HANDLE_COLOR_HOVER = (255, 220, 100)
_BOX_OPACITY = 0.25
_BOX_OPACITY_SELECTED = 0.0

_NAMED_COLOURS: dict[str, tuple[int, int, int]] = {
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "gray": (128, 128, 128),
    "grey": (128, 128, 128),
    "red": (255, 0, 0),
    "green": (0, 200, 50),
    "blue": (0, 114, 255),
    "yellow": (255, 215, 0),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 105, 180),
    "cyan": (0, 200, 200),
    "magenta": (255, 0, 255),
    "brown": (139, 69, 19),
}


def _parse_colour_to_rgb(colour: str) -> tuple[int, int, int]:
    value = str(colour).strip()
    if not value:
        return _BOX_COLOR

    lower = value.lower()
    if lower in _NAMED_COLOURS:
        return _NAMED_COLOURS[lower]

    hex_value = lower[1:] if lower.startswith("#") else lower
    if len(hex_value) == 3 and all(c in "0123456789abcdef" for c in hex_value):
        hex_value = "".join(c * 2 for c in hex_value)
    if len(hex_value) == 6 and all(c in "0123456789abcdef" for c in hex_value):
        return (
            int(hex_value[0:2], 16),
            int(hex_value[2:4], 16),
            int(hex_value[4:6], 16),
        )

    return _BOX_COLOR


class BoxAnnotatorController:
    """Manages creation, selection, manipulation, and persistence of 3D boxes."""

    def __init__(
        self,
        server: viser.ViserServer,
        all_points: np.ndarray | None = None,
        dense_points: np.ndarray | None = None,
        notification_manager: "NotificationManager | None" = None,
    ):
        self.server = server
        # Flattened point cloud for ray-casting (N, 3)
        self.all_points = all_points
        # Dense point cloud (N, 3) for selection tools
        self.dense_points = dense_points
        self.notification_manager = notification_manager

        self._lock = threading.Lock()
        self._boxes: dict[str, Box3D] = {}
        self._handles: dict[str, _BoxSceneHandles] = {}
        self._selected_id: str | None = None
        self._selected_ids: set[str] = set()
        self._next_idx: int = 0

        # GUI element references (set later by setup_box_annotator_gui)
        self._label_input: Any = None
        self._colour_input: Any = None
        self._dim_x: Any = None
        self._dim_y: Any = None
        self._dim_z: Any = None
        self._pos_x: Any = None
        self._pos_y: Any = None
        self._pos_z: Any = None
        self._euler_roll: Any = None
        self._euler_pitch: Any = None
        self._euler_yaw: Any = None
        self._box_dropdown: Any = None
        self._status_text: Any = None
        self._sync_indicator_text: Any = None
        self._property_controls: list[Any] = []

        # CG bounding boxes for auto-adjust feature (set via set_cg_bboxes)
        self._cg_bboxes: list[dict] | None = None

        # Flag: is click-to-place mode active?
        self._placing_mode = False

        # Re-entrancy guard: skip cascading callbacks while updating state
        self._updating = False

        self._suppress_multi_gizmo_events = False

        # Flag: is box selection mode active?
        self._selecting_mode = False
        self._multi_selecting_mode = False

        # Flag: is box-by-corner mode active?
        self._corner_mode = False
        self._corner_points: list[np.ndarray] = []
        self._corner_label_handle: Any = None
        self._corner_label_name: str | None = None
        self._corner_client: viser.ClientHandle | None = None

        # Flag: is box-by-select mode active?
        self._box_by_select_mode = False

        # Global visibility toggle for all annotation bbox visuals.
        self._show_bboxes = True

        # Persistence state: True when current boxes match saved file state.
        self._is_synced_to_file = True

        # Aggregate controls for multi-selection transforms.
        self._multi_handles = _MultiSelectionHandles()
        self._prev_selected_ids: set[str] = set()
        self._global_multi_select_rotation = False
        
    def set_global_multi_select_rotation(self, enabled: bool):
        self._global_multi_select_rotation = bool(enabled)

    # ── helpers ───────────────────────────────────────────────────────────

    @property
    def selected_box(self) -> Box3D | None:
        if len(self._selected_ids) != 1:
            return None
        if self._selected_id is None:
            return None
        return self._boxes.get(self._selected_id)

    @property
    def selected_boxes(self) -> list[Box3D]:
        return [self._boxes[box_id] for box_id in self._selected_ids if box_id in self._boxes]

    @property
    def is_multi_selected(self) -> bool:
        return len(self._selected_ids) > 1

    def set_properties_enabled(self, enabled: bool):
        for control in self._property_controls:
            try:
                control.disabled = not enabled
            except Exception:
                pass

    def _compute_selection_aabb(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
        boxes = self.selected_boxes
        if len(boxes) == 0:
            return None

        world_corners = []
        for box in boxes:
            R = _quat_to_rotmat(box.wxyz)
            for signs in _CORNER_SIGNS:
                local_off = signs * (box.dimensions / 2)
                world_corners.append(box.center + R @ local_off)

        corners = np.stack(world_corners, axis=0)
        bbox_min = corners.min(axis=0)
        bbox_max = corners.max(axis=0)
        center = (bbox_min + bbox_max) / 2.0
        dims = np.maximum(bbox_max - bbox_min, 0.02)
        return center, dims, bbox_min, bbox_max

    def _set_per_box_controls_visible(self, box_id: str, visible: bool):
        h = self._handles.get(box_id)
        if h is None:
            return
        for obj in [h.gizmo_handle, *h.corner_handles, *h.corner_gizmo_handles]:
            if obj is None:
                continue
            try:
                obj.visible = visible and self._show_bboxes
            except Exception:
                pass

    def _remove_multi_selection_handles(self):
        for obj in [
            self._multi_handles.gizmo_handle,
            self._multi_handles.wire_handle,
            *self._multi_handles.corner_handles,
            *self._multi_handles.corner_gizmo_handles,
        ]:
            if obj is None:
                continue
            try:
                obj.remove()
            except Exception:
                pass
        self._multi_handles = _MultiSelectionHandles()

    def _resolve_multi_gizmo_wxyz(self) -> np.ndarray:
        """Resolve the aggregate gizmo orientation using single-box semantics.

        Prefer the current multi gizmo orientation if present; otherwise use
        the orientation of the first selected box.
        """
        if self._multi_handles.prev_gizmo_wxyz is not None:
            return self._multi_handles.prev_gizmo_wxyz.copy()
        boxes = self.selected_boxes
        if len(boxes) > 0:
            return boxes[0].wxyz.copy()
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def _create_or_update_multi_selection_handles(self):
        if len(self._selected_ids) <= 1:
            self._remove_multi_selection_handles()
            return

        aabb = self._compute_selection_aabb()
        if aabb is None:
            self._remove_multi_selection_handles()
            return
        center, dims, bbox_min, bbox_max = aabb

        if self._multi_handles.gizmo_handle is None:
            prefix = "/annotations/multi_selection"
            gizmo_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            gizmo_pos = center + _gizmo_global_offset()

            wire = self.server.scene.add_box(
                name=f"{prefix}/wire",
                color=_BOX_COLOR_SELECTED,
                dimensions=tuple(dims.tolist()),
                wireframe=True,
                wxyz=(1.0, 0.0, 0.0, 0.0),
                position=tuple(center.tolist()),
            )

            gizmo = self.server.scene.add_transform_controls(
                name=f"{prefix}/gizmo",
                scale=0.35,
                line_width=2.0,
                wxyz=tuple(gizmo_wxyz.tolist()),
                position=tuple(gizmo_pos.tolist()),
                disable_sliders=True,
                depth_test=False,
                opacity=0.9,
            )

            @gizmo.on_update
            def _on_multi_gizmo_update(event: viser.TransformControlsEvent):
                self._on_multi_central_gizmo_moved(event)

            @gizmo.on_drag_end
            def _on_multi_gizmo_drag_end(event: viser.TransformControlsEvent):
                self._on_multi_central_gizmo_drag_end()

            corner_handles: list[Any] = []
            corner_gizmos: list[Any] = []
            multi_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
            for ci, signs in enumerate(_CORNER_SIGNS):
                world_pos = center + signs * (dims / 2.0)
                corner_wxyz = _corner_gizmo_wxyz(multi_wxyz, signs)
                corner_gizmo = self.server.scene.add_transform_controls(
                    name=f"{prefix}/corner_{ci}/gizmo",
                    scale=0.22,
                    line_width=2.5,
                    position=tuple(world_pos.tolist()),
                    wxyz=tuple(corner_wxyz.tolist()),
                    disable_rotations=True,
                    disable_axes=False,
                    disable_sliders=True,
                    depth_test=False,
                    opacity=0.7,
                )
                corner_gizmos.append(corner_gizmo)

                _ci = ci
                _signs = signs.copy()

                @corner_gizmo.on_update
                def _on_multi_corner_update(
                    event: viser.TransformControlsEvent,
                    _ci_=_ci,
                    _signs_=_signs,
                ):
                    self._on_multi_corner_handle_moved(_ci_, _signs_, event)

                @corner_gizmo.on_drag_end
                def _on_multi_corner_drag_end(event: viser.TransformControlsEvent):
                    self._on_multi_corner_drag_end()

            self._multi_handles = _MultiSelectionHandles(
                gizmo_handle=gizmo,
                wire_handle=wire,
                corner_handles=corner_handles,
                corner_gizmo_handles=corner_gizmos,
                prev_gizmo_wxyz=gizmo_wxyz.copy(),
                prev_gizmo_position=gizmo_pos.copy(),
                bbox_min=bbox_min.copy(),
                bbox_max=bbox_max.copy(),
            )
        else:
            self._set_multi_handles_from_aabb(
                bbox_min,
                bbox_max,
                update_gizmo_pose=True,
            )

        visible = self._show_bboxes
        try:
            self._multi_handles.gizmo_handle.visible = visible
        except Exception:
            pass
        if self._multi_handles.wire_handle is not None:
            try:
                self._multi_handles.wire_handle.visible = visible
            except Exception:
                pass
        for obj in [*self._multi_handles.corner_handles, *self._multi_handles.corner_gizmo_handles]:
            try:
                obj.visible = visible
            except Exception:
                pass

    def _set_multi_handles_from_aabb(
        self,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        *,
        update_gizmo_pose: bool,
    ):
        """Update aggregate multi-selection handles from an explicit world AABB."""
        if self._multi_handles.gizmo_handle is None:
            return

        bbox_min = np.asarray(bbox_min, dtype=np.float64)
        bbox_max = np.asarray(bbox_max, dtype=np.float64)
        center = (bbox_min + bbox_max) / 2.0
        dims = np.maximum(bbox_max - bbox_min, 0.02)
        gizmo_wxyz = self._multi_handles.prev_gizmo_wxyz
        if gizmo_wxyz is None:
            gizmo_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        if self._multi_handles.wire_handle is not None:
            self._multi_handles.wire_handle.position = tuple(center.tolist())
            self._multi_handles.wire_handle.wxyz = (1.0, 0.0, 0.0, 0.0)
            self._multi_handles.wire_handle.dimensions = tuple(dims.tolist())

        if update_gizmo_pose:
            self._set_gizmo_pose_from_center(
                self._multi_handles,
                center,
                gizmo_wxyz,
            )

        for ci, signs in enumerate(_CORNER_SIGNS):
            world_pos = center + signs * (dims / 2.0)
            corner_wxyz = _corner_gizmo_wxyz(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64), signs)
            wp = tuple(world_pos.tolist())
            if ci < len(self._multi_handles.corner_handles):
                self._multi_handles.corner_handles[ci].position = wp
            if ci < len(self._multi_handles.corner_gizmo_handles):
                self._multi_handles.corner_gizmo_handles[ci].position = wp
                self._multi_handles.corner_gizmo_handles[ci].wxyz = tuple(corner_wxyz.tolist())

        self._multi_handles.bbox_min = bbox_min.copy()
        self._multi_handles.bbox_max = bbox_max.copy()

    def _selection_changed(self):
        prev_selected_ids = set(self._prev_selected_ids)

        if len(self._selected_ids) == 1:
            self._selected_id = next(iter(self._selected_ids))
        else:
            self._selected_id = None

        selected_set = set(self._selected_ids)
        for box_id in self._boxes:
            self._update_box_color(box_id, selected=(box_id in selected_set))

        if self._box_dropdown is not None:
            self._refresh_dropdown()

        if len(self._selected_ids) == 1 and self._selected_id in self._boxes:
            self._sync_gui_to_box(self._boxes[self._selected_id])

        self.set_properties_enabled(len(self._selected_ids) <= 1)

        if len(self._selected_ids) > 1:
            for box_id in self._selected_ids:
                self._set_per_box_controls_visible(box_id, False)
            self._create_or_update_multi_selection_handles()
        else:
            self._remove_multi_selection_handles()

            # During multi-selection drags we intentionally skip per-box gizmo
            # updates. When leaving multi-selection, bring those gizmos back to
            # the current box poses before making controls visible again.
            if len(prev_selected_ids) > 1:
                for box_id in prev_selected_ids:
                    box = self._boxes.get(box_id)
                    handles = self._handles.get(box_id)
                    if box is None or handles is None:
                        continue
                    self._set_gizmo_pose_from_center(handles, box.center, box.wxyz)

            for box_id in self._boxes:
                self._set_per_box_controls_visible(box_id, True)

        self._prev_selected_ids = set(self._selected_ids)

    def set_selection(self, ids: list[str] | set[str]):
        self._selected_ids = {box_id for box_id in ids if box_id in self._boxes}
        self._selection_changed()

    def clear_selection(self):
        self._selected_ids = set()
        self._selection_changed()

    def toggle_selected(self, box_id: str):
        if box_id not in self._boxes:
            return
        if box_id in self._selected_ids:
            self._selected_ids.remove(box_id)
        else:
            self._selected_ids.add(box_id)
        self._selection_changed()

    def _set_status(self, text: str):
        if self._status_text is not None:
            self._status_text.content = text

    def _set_sync_indicator(self):
        if self._sync_indicator_text is None:
            return
        if self._is_synced_to_file:
            self._sync_indicator_text.content = "**🟢 Synced to file**"
        else:
            self._sync_indicator_text.content = "**◌ Out of sync**"

    def _mark_out_of_sync(self):
        self._is_synced_to_file = False
        self._set_sync_indicator()

    def _mark_synced(self):
        self._is_synced_to_file = True
        self._set_sync_indicator()

    def _notify(self, title: str, body: str, *, client: viser.ClientHandle):
        if self.notification_manager is None:
            log.warning("Notification manager is not configured; skipping notification.")
            return
        self.notification_manager.notify(title=title, body=body, client=client)

    # ── point-cloud ray intersection ─────────────────────────────────────

    def _ray_nearest_point(
        self,
        ray_origin: np.ndarray,
        ray_dir: np.ndarray,
        points: np.ndarray | None = None,
    ) -> np.ndarray | None:
        """Find the point cloud point closest to the given ray.

        Projects every point onto the ray, computes the perpendicular distance,
        and returns the closest one.  Falls back to a point 3 units along the
        ray if no point cloud is available.
        """
        pts = self.all_points if points is None else points
        if pts is None or len(pts) == 0:
            # Fallback: place 3 units in front
            return ray_origin + ray_dir * 3.0

        pts = pts  # (N, 3)
        diff = pts - ray_origin  # (N, 3)
        t = np.dot(diff, ray_dir)  # (N,)
        # Only consider points in front of the camera
        mask = t > 0.0
        if not np.any(mask):
            return ray_origin + ray_dir * 3.0

        proj = ray_origin + t[:, None] * ray_dir  # (N, 3)
        dists = np.linalg.norm(pts - proj, axis=1)  # (N,)
        # Penalise points behind camera
        dists[~mask] = np.inf
        best = int(np.argmin(dists))
        return pts[best].copy()

    def _project_points_to_screen(
        self, points: np.ndarray, camera: viser.CameraHandle
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project world-space points to normalized screen coordinates.

        Returns (proj, valid_mask) where proj is (N, 2) in [0, 1] range.
        Uses OpenCV image coordinates: (0, 0) is upper-left, (1, 1) is lower-right.
        """
        R_camera_world = vtf.SE3.from_rotation_and_translation(
            vtf.SO3(camera.wxyz), camera.position
        ).inverse()

        homog = np.hstack([points, np.ones((points.shape[0], 1))])
        points_cam = (R_camera_world.as_matrix() @ homog.T).T[:, :3]

        z = points_cam[:, 2]
        valid = z > 1e-6
        points_cam = points_cam[valid]

        fov, aspect = camera.fov, camera.aspect
        proj = points_cam[:, :2] / points_cam[:, 2].reshape(-1, 1)
        proj /= np.tan(fov / 2)
        proj[:, 0] /= aspect

        # Move origin to upper-left corner and scale to [0, 1].
        # This matches viser's OpenCV image coordinates convention.
        proj = (1 + proj) / 2
        return proj, valid

    def _compute_aabb_from_points(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute axis-aligned box (center, dimensions, wxyz) from points."""
        mins = points.min(axis=0)
        maxs = points.max(axis=0)
        center = (mins + maxs) / 2.0
        dims = np.maximum(maxs - mins, 0.02)
        wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return center, dims, wxyz

    def _compute_oobb_from_points_and_camera(
        self, points: np.ndarray, camera: viser.CameraHandle
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute oriented bounding box aligned to camera view.

        The box is oriented so that its local X/Y axes align with the camera's
        screen axes, giving minimal extents from the current viewing angle.

        Returns (center, dimensions, wxyz) in world coordinates.
        """
        # Get camera rotation (world -> camera)
        R_camera_world = vtf.SO3(camera.wxyz).inverse()
        R_world_camera = vtf.SO3(camera.wxyz)

        # Transform points to camera space
        points_cam = (R_camera_world.as_matrix() @ points.T).T

        # Compute AABB in camera space (minimal extents from this view)
        mins_cam = points_cam.min(axis=0)
        maxs_cam = points_cam.max(axis=0)
        center_cam = (mins_cam + maxs_cam) / 2.0
        dims = np.maximum(maxs_cam - mins_cam, 0.02)

        # Transform center back to world space
        center_world = R_world_camera.as_matrix() @ center_cam

        # The box orientation is the camera's orientation
        wxyz = camera.wxyz.copy()

        return center_world.astype(np.float64), dims.astype(np.float64), wxyz.astype(np.float64)

    def _augment_rect_select_points(
        self,
        selected_pts: np.ndarray,
        camera: viser.CameraHandle,
        rect_lo: np.ndarray,
        rect_hi: np.ndarray,
    ) -> np.ndarray:
        """Add synthetic points at the rect corners for robust box fitting.

        We back-project the 2D rectangle corners to 3D at the min/max depth
        of the selected points. This makes the fitted box respect the full
        rectangle even if corner points are missing.
        """
        if selected_pts.size == 0:
            return selected_pts

        cam_se3 = vtf.SE3.from_rotation_and_translation(
            vtf.SO3(camera.wxyz), camera.position
        )
        cam_se3_inv = cam_se3.inverse()

        homog = np.hstack([selected_pts, np.ones((selected_pts.shape[0], 1))])
        points_cam = (cam_se3_inv.as_matrix() @ homog.T).T[:, :3]

        z_vals = points_cam[:, 2]
        if z_vals.size == 0:
            return selected_pts

        z_min = float(np.min(z_vals))
        z_max = float(np.max(z_vals))
        if z_min <= 1e-6:
            z_min = max(z_min, 1e-3)

        fov = camera.fov
        aspect = camera.aspect
        tan_half_fov = np.tan(fov / 2)

        rect_corners = np.array(
            [
                [rect_lo[0], rect_lo[1]],
                [rect_lo[0], rect_hi[1]],
                [rect_hi[0], rect_lo[1]],
                [rect_hi[0], rect_hi[1]],
            ],
            dtype=np.float64,
        )

        synthetic_cam = []
        for depth in (z_min, z_max):
            ndc = rect_corners * 2.0 - 1.0
            x = ndc[:, 0] * tan_half_fov * aspect * depth
            y = ndc[:, 1] * tan_half_fov * depth
            z = np.full_like(x, depth)
            synthetic_cam.append(np.stack([x, y, z], axis=1))

        synthetic_cam = np.vstack(synthetic_cam)
        homog_cam = np.hstack(
            [synthetic_cam, np.ones((synthetic_cam.shape[0], 1))]
        )
        synthetic_world = (cam_se3.as_matrix() @ homog_cam.T).T[:, :3]

        return np.vstack([selected_pts, synthetic_world])

    # ── box lifecycle ────────────────────────────────────────────────────

    def create_box(self, center: np.ndarray, label: str | None = None) -> Box3D:
        """Create a new box at *center* and add it to the scene."""
        box = Box3D(
            center=center.astype(np.float64),
            label=label or f"box_{self._next_idx}",
        )
        self._next_idx += 1
        with self._lock:
            self._boxes[box.id] = box
        self._add_box_to_scene(box)
        # Refresh dropdown first, then select - this ensures the new box
        # appears in the dropdown options before we try to select it
        self._refresh_dropdown_options_only()
        self.select_box(box.id)
        self._mark_out_of_sync()
        log.info("Created box %s at %s", box.id, center)
        return box

    def delete_box(self, box_id: str):
        """Remove a box from the scene and from internal state."""
        with self._lock:
            box = self._boxes.pop(box_id, None)
            handles = self._handles.pop(box_id, None)
        if handles:
            self._remove_scene_handles(handles)
        if box_id in self._selected_ids:
            self._selected_ids.remove(box_id)
        self._selection_changed()
        if box:
            self._mark_out_of_sync()
            log.info("Deleted box %s (%s)", box.id, box.label)

    def delete_selected(self):
        ids = list(self._selected_ids)
        for box_id in ids:
            self.delete_box(box_id)

    def delete_all(self):
        ids = list(self._boxes.keys())
        for bid in ids:
            self.delete_box(bid)

    # ── selection ────────────────────────────────────────────────────────

    def select_box(self, box_id: str | None):
        """Select a box by id (or deselect if None)."""
        if self._updating:
            return
        self._updating = True
        try:
            if box_id is None:
                self.clear_selection()
            elif box_id in self._boxes:
                self.set_selection([box_id])
            else:
                self.clear_selection()
        finally:
            self._updating = False

    def _sync_gui_to_box(self, box: Box3D):
        """Push box state into the GUI inputs (label, dimensions)."""
        was_updating = self._updating
        self._updating = True
        try:
            if self._label_input is not None:
                self._label_input.value = box.label
            if self._colour_input is not None:
                self._colour_input.value = box.colour
            if self._pos_x is not None:
                self._pos_x.value = round(float(box.center[0]), 4)
            if self._pos_y is not None:
                self._pos_y.value = round(float(box.center[1]), 4)
            if self._pos_z is not None:
                self._pos_z.value = round(float(box.center[2]), 4)
            if self._dim_x is not None:
                self._dim_x.value = float(box.dimensions[0])
            if self._dim_y is not None:
                self._dim_y.value = float(box.dimensions[1])
            if self._dim_z is not None:
                self._dim_z.value = float(box.dimensions[2])
            # Euler angles (degrees) derived from the box quaternion
            rpy = vtf.SO3(wxyz=box.wxyz).as_rpy_radians()
            if self._euler_roll is not None:
                self._euler_roll.value = round(float(np.degrees(rpy.roll)), 2)
            if self._euler_pitch is not None:
                self._euler_pitch.value = round(float(np.degrees(rpy.pitch)), 2)
            if self._euler_yaw is not None:
                self._euler_yaw.value = round(float(np.degrees(rpy.yaw)), 2)
        finally:
            self._updating = was_updating

    # ── scene rendering ──────────────────────────────────────────────────

    def _add_box_to_scene(self, box: Box3D):
        """Render a box in the Viser scene with gizmo + face handles."""
        bid = box.id
        prefix = f"/annotations/box_{bid}"

        # Semi-transparent filled box
        box_handle = self.server.scene.add_box(
            name=f"{prefix}/mesh",
            color=_parse_colour_to_rgb(box.colour),
            dimensions=tuple(box.dimensions.tolist()),
            wireframe=False,
            opacity=_BOX_OPACITY,
            wxyz=tuple(box.wxyz.tolist()),
            position=tuple(box.center.tolist()),
            side="double",
        )

        # Wireframe overlay for visibility
        wire_handle = self.server.scene.add_box(
            name=f"{prefix}/wire",
            color=_parse_colour_to_rgb(box.colour),
            dimensions=tuple(box.dimensions.tolist()),
            wireframe=True,
            wxyz=tuple(box.wxyz.tolist()),
            position=tuple(box.center.tolist()),
        )

        R = _quat_to_rotmat(box.wxyz)

        # Central transform gizmo (translate + rotate) — hovers above the
        # box along the global Y axis so it doesn't overlap corner handles.
        gizmo_world_pos = box.center + _gizmo_global_offset()
        gizmo = self.server.scene.add_transform_controls(
            name=f"{prefix}/gizmo",
            scale=0.35,
            line_width=2.0,
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=tuple(gizmo_world_pos.tolist()),
            disable_sliders=True,
            depth_test=False,
            opacity=0.9,
        )

        @gizmo.on_update
        def _on_gizmo_update(
            event: viser.TransformControlsEvent, _bid=bid
        ):
            self._on_central_gizmo_moved(_bid, event)

        @gizmo.on_drag_end
        def _on_gizmo_drag_end(
            event: viser.TransformControlsEvent, _bid=bid
        ):
            self._on_central_gizmo_drag_end(_bid)

        # Label floating above box
        label_offset = np.array([0.0, 0.0, box.dimensions[2] / 2 + 0.08])
        label_world = box.center + R @ label_offset
        label_handle = self.server.scene.add_label(
            name=f"{prefix}/label",
            text=box.label,
            position=tuple(label_world.tolist()),
        )

        # Corner resize gizmos (8 corners)
        corner_handles: list[Any] = []
        corner_gizmo_handles: list[Any] = []
        for ci, signs in enumerate(_CORNER_SIGNS):
            # Position in world space
            local_offset = signs * (box.dimensions / 2)
            world_pos = box.center + R @ local_offset
            corner_wxyz = _corner_gizmo_wxyz(box.wxyz, signs)

            # Translate-only gizmo: arrows hidden, plane sliders active
            corner_gizmo = self.server.scene.add_transform_controls(
                name=f"{prefix}/corner_{ci}/gizmo",
                scale=0.22,
                line_width=2.5,
                position=tuple(world_pos.tolist()),
                wxyz=tuple(corner_wxyz.tolist()),
                disable_rotations=True,
                disable_axes=False,
                disable_sliders=True,
                depth_test=False,
                opacity=0.7,
            )
            corner_gizmo_handles.append(corner_gizmo)

            # Closure capture
            _ci = ci
            _signs = signs.copy()

            @corner_gizmo.on_update
            def _on_corner_update(
                event: viser.TransformControlsEvent,
                _bid=bid,
                _ci_=_ci,
                _signs_=_signs,
            ):
                self._on_corner_handle_moved(_bid, _ci_, _signs_, event)

        handles = _BoxSceneHandles(
            box_handle=box_handle,
            wire_handle=wire_handle,
            gizmo_handle=gizmo,
            label_handle=label_handle,
            corner_handles=corner_handles,
            corner_gizmo_handles=corner_gizmo_handles,
            prev_gizmo_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
            prev_gizmo_position=gizmo_world_pos.copy(),
        )
        self._handles[bid] = handles
        self._set_handles_visibility(handles, self._show_bboxes)
        if bid in self._selected_ids and self.is_multi_selected:
            self._set_per_box_controls_visible(bid, False)

    def _remove_scene_handles(self, h: _BoxSceneHandles):
        """Remove all viser scene nodes for a box."""
        for attr in ("box_handle", "wire_handle", "gizmo_handle", "label_handle"):
            obj = getattr(h, attr, None)
            if obj is not None:
                try:
                    obj.remove()
                except Exception:
                    pass
        for lst in (h.corner_handles, h.corner_gizmo_handles):
            for obj in lst:
                try:
                    obj.remove()
                except Exception:
                    pass
        # Also clean up the wireframe (it's a sibling under the same prefix)
        # viser's remove_by_name with parent prefix would be ideal; handles
        # cover the main objects.

    def _set_handles_visibility(self, h: _BoxSceneHandles, visible: bool):
        """Set visibility for all scene nodes associated with a single box."""
        for attr in ("box_handle", "wire_handle", "gizmo_handle", "label_handle"):
            obj = getattr(h, attr, None)
            if obj is not None:
                try:
                    obj.visible = visible
                except Exception:
                    pass
        for lst in (h.corner_handles, h.corner_gizmo_handles):
            for obj in lst:
                if obj is not None:
                    try:
                        obj.visible = visible
                    except Exception:
                        pass

    def _update_box_color(self, box_id: str, selected: bool):
        h = self._handles.get(box_id)
        box = self._boxes.get(box_id)
        box_colour = _parse_colour_to_rgb(box.colour if box else "green")
        fill_opacity = _BOX_OPACITY_SELECTED if selected else _BOX_OPACITY
        if h and h.box_handle:
            h.box_handle.color = box_colour
            h.box_handle.opacity = fill_opacity
        if h and h.wire_handle:
            h.wire_handle.color = box_colour

    def _rebuild_box_visuals(self, box: Box3D):
        """Tear down and recreate all scene elements for a box."""
        bid = box.id
        old_handles = self._handles.pop(bid, None)
        if old_handles:
            self._remove_scene_handles(old_handles)
        self._add_box_to_scene(box)
        if bid in self._selected_ids:
            self._update_box_color(bid, selected=True)
        if self.is_multi_selected:
            self._create_or_update_multi_selection_handles()

    def set_bboxes_visible(self, visible: bool):
        """Show/hide all rendered bboxes and their controls."""
        self._show_bboxes = bool(visible)
        for handles in self._handles.values():
            self._set_handles_visibility(handles, self._show_bboxes)
        if self._multi_handles.gizmo_handle is not None:
            try:
                self._multi_handles.gizmo_handle.visible = self._show_bboxes
            except Exception:
                pass
        if self._multi_handles.wire_handle is not None:
            try:
                self._multi_handles.wire_handle.visible = self._show_bboxes
            except Exception:
                pass
        for obj in [*self._multi_handles.corner_handles, *self._multi_handles.corner_gizmo_handles]:
            try:
                obj.visible = self._show_bboxes
            except Exception:
                pass

    @property
    def bboxes_visible(self) -> bool:
        return self._show_bboxes

    # ── gizmo callbacks ──────────────────────────────────────────────────

    def _on_corner_handle_moved(
        self,
        box_id: str,
        corner_idx: int,
        signs: np.ndarray,
        event: viser.TransformControlsEvent,
    ):
        """Handle a corner resize handle being dragged.

        The opposite corner (all signs flipped) stays fixed.  The box is
        resized and re-centred so that the dragged corner follows the
        handle while the opposite corner remains stationary.
        """
        box = self._boxes.get(box_id)
        if box is None:
            return

        handle_pos_world = np.array(event.target.position, dtype=np.float64)
        R = _quat_to_rotmat(box.wxyz)

        # Opposite corner in world space (current dims)
        opp_local = -signs * (box.dimensions / 2)
        opp_world = box.center + R @ opp_local

        # Vector from opposite corner to dragged position, in local frame
        delta_local = R.T @ (handle_pos_world - opp_world)

        # New dimensions = absolute span in each local axis
        new_dims = np.abs(delta_local)
        new_dims = np.maximum(new_dims, 0.02)  # minimum size

        # New centre = midpoint between opposite corner and dragged corner
        box.center = (opp_world + handle_pos_world) / 2.0
        box.dimensions = new_dims

        # Rebuild visuals (fast path)
        self._update_box_scene_fast(box)
        # Update GUI
        if box.id in self._selected_ids and not self.is_multi_selected:
            self._sync_gui_to_box(box)
        self._mark_out_of_sync()

    def _update_box_scene_fast(self, box: Box3D):
        """Update positions/dimensions of existing scene handles without full rebuild."""
        self._update_box_in_place(box, update_gizmo=True, update_dimensions=True)

    # ── central gizmo callbacks ──────────────────────────────────────────

    def _set_gizmo_pose_from_center(
        self,
        handles: _BoxSceneHandles | _MultiSelectionHandles,
        center: np.ndarray,
        wxyz: np.ndarray,
    ):
        """Set gizmo pose from object center using shared single-box logic.

        The gizmo is always axis-aligned (identity quaternion), hovering 2m above
        the box center. The ``wxyz`` parameter is accepted for API compatibility
        but ignored; we always reset to axis-aligned so rotations apply as deltas.
        """
        if handles.gizmo_handle is None:
            return
        gizmo_world = center + _gizmo_global_offset()
        identity_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        suppress_events = handles is self._multi_handles
        if suppress_events:
            self._suppress_multi_gizmo_events = True
        try:
            handles.gizmo_handle.position = tuple(gizmo_world.tolist())
            handles.gizmo_handle.wxyz = tuple(identity_wxyz.tolist())
            handles.prev_gizmo_position = gizmo_world.copy()
            handles.prev_gizmo_wxyz = identity_wxyz.copy()
        finally:
            if suppress_events:
                self._suppress_multi_gizmo_events = False

    def _apply_box_rotation_delta(
        self,
        box: Box3D,
        R_delta: np.ndarray,
        *,
        pivot_world: np.ndarray | None = None,
        update_gizmo: bool = False,
    ):
        """Apply an incremental world-space rotation to a box.

        If ``pivot_world`` is provided, the box center is rotated around that
        world-space pivot before updating orientation.
        """
        if pivot_world is not None:
            box.center = pivot_world + R_delta @ (box.center - pivot_world)
        R_box_new = R_delta @ _quat_to_rotmat(box.wxyz)
        box.wxyz = _rotmat_to_quat(R_box_new)
        self._update_box_in_place(box, update_gizmo=update_gizmo)

    def _apply_box_orientation_from_gizmo_delta(
        self,
        box: Box3D,
        prev_gizmo_wxyz: np.ndarray,
        new_gizmo_wxyz: np.ndarray,
        *,
        update_gizmo: bool = False,
    ):
        """Apply gizmo orientation delta to a single box orientation.

        This is the canonical orientation behavior used by a single-box gizmo,
        and is reused by multi-selection orientation changes.
        """
        R_old = _quat_to_rotmat(prev_gizmo_wxyz)
        R_new = _quat_to_rotmat(new_gizmo_wxyz)
        R_delta = R_new @ R_old.T
        self._apply_box_rotation_delta(box, R_delta, update_gizmo=update_gizmo)

    def _apply_box_translation_delta(
        self,
        box: Box3D,
        delta: np.ndarray,
        *,
        update_gizmo: bool = False,
    ):
        """Apply a world-space translation delta to a box."""
        box.center = box.center + delta
        self._update_box_in_place(box, update_gizmo=update_gizmo)

    def _on_central_gizmo_moved(
        self,
        box_id: str,
        event: viser.TransformControlsEvent,
    ):
        """Handle the central transform gizmo being dragged (translate or rotate)."""
        box = self._boxes.get(box_id)
        h = self._handles.get(box_id)
        if box is None or h is None:
            return

        new_pos = np.array(event.target.position, dtype=np.float64)
        new_wxyz = np.array(event.target.wxyz, dtype=np.float64)

        prev_pos = h.prev_gizmo_position
        prev_wxyz = h.prev_gizmo_wxyz

        if prev_pos is None or prev_wxyz is None:
            h.prev_gizmo_position = new_pos.copy()
            h.prev_gizmo_wxyz = new_wxyz.copy()
            return

        wxyz_changed = np.linalg.norm(new_wxyz - prev_wxyz) > 1e-6
        pos_changed = np.linalg.norm(new_pos - prev_pos) > 1e-6

        if wxyz_changed:
            # ── Rotation: rotate the box around its own centre ────────
            # Update visuals *except* gizmo (user is still dragging)
            self._apply_box_orientation_from_gizmo_delta(
                box,
                prev_wxyz,
                new_wxyz,
                update_gizmo=False,
            )

            h.prev_gizmo_wxyz = new_wxyz.copy()
            h.prev_gizmo_position = new_pos.copy()

            # Keep euler angle GUI in sync
            if box_id in self._selected_ids and not self.is_multi_selected:
                self._sync_gui_to_box(box)

            self._mark_out_of_sync()

        elif pos_changed:
            # ── Translation: move box so gizmo stays above it ─────────
            delta = new_pos - prev_pos

            # Update visuals; gizmo is already where the user put it
            self._apply_box_translation_delta(box, delta, update_gizmo=False)

            h.prev_gizmo_position = new_pos.copy()

            if box.id in self._selected_ids and not self.is_multi_selected:
                self._sync_gui_to_box(box)
            self._mark_out_of_sync()

    def _on_multi_central_gizmo_drag_end(self):
        self._reset_multi_gizmo_pose()

    def _on_multi_central_gizmo_moved(
        self,
        event: viser.TransformControlsEvent,
    ):
        if len(self._selected_ids) <= 1:
            return
        if self._suppress_multi_gizmo_events:
            return
        h = self._multi_handles
        if h.gizmo_handle is None:
            return

        new_pos = np.array(event.target.position, dtype=np.float64)
        new_wxyz = np.array(event.target.wxyz, dtype=np.float64)

        prev_pos = h.prev_gizmo_position
        prev_wxyz = h.prev_gizmo_wxyz
        if prev_pos is None or prev_wxyz is None:
            h.prev_gizmo_position = new_pos.copy()
            h.prev_gizmo_wxyz = new_wxyz.copy()
            return

        wxyz_changed = np.linalg.norm(new_wxyz - prev_wxyz) > 1e-6
        pos_changed = np.linalg.norm(new_pos - prev_pos) > 1e-6

        boxes = self.selected_boxes
        if len(boxes) <= 1:
            return

        if wxyz_changed:
            R_old = _quat_to_rotmat(prev_wxyz)
            R_new = _quat_to_rotmat(new_wxyz)
            R_delta = R_new @ R_old.T

            if self._global_multi_select_rotation:
                if h.bbox_min is not None and h.bbox_max is not None:
                    pivot_world = (h.bbox_min + h.bbox_max) / 2.0
                else:
                    pivot_world = prev_pos - _gizmo_global_offset()

                for box in boxes:
                    self._apply_box_rotation_delta(
                        box,
                        R_delta,
                        pivot_world=pivot_world,
                        update_gizmo=False,
                    )
            else:
                for box in boxes:
                    self._apply_box_orientation_from_gizmo_delta(
                        box,
                        prev_wxyz,
                        new_wxyz,
                        update_gizmo=False,
                    )

            h.prev_gizmo_wxyz = new_wxyz.copy()
            h.prev_gizmo_position = new_pos.copy()
            aabb = self._compute_selection_aabb()
            if aabb is not None:
                _, _, bbox_min, bbox_max = aabb
                self._set_multi_handles_from_aabb(
                    bbox_min,
                    bbox_max,
                    update_gizmo_pose=False,
                )
            self._mark_out_of_sync()
        elif pos_changed:
            delta = new_pos - prev_pos
            for box in boxes:
                self._apply_box_translation_delta(box, delta, update_gizmo=False)

            h.prev_gizmo_position = new_pos.copy()
            aabb = self._compute_selection_aabb()
            if aabb is not None:
                _, _, bbox_min, bbox_max = aabb
                self._set_multi_handles_from_aabb(
                    bbox_min,
                    bbox_max,
                    update_gizmo_pose=False,
                )
            self._mark_out_of_sync()

    def _on_central_gizmo_drag_end(self, box_id: str):
        self._reset_single_gizmo_pose(box_id)

    def _on_multi_corner_handle_moved(
        self,
        corner_idx: int,
        signs: np.ndarray,
        event: viser.TransformControlsEvent,
    ):
        if len(self._selected_ids) <= 1:
            return
        h = self._multi_handles
        if h.bbox_min is None or h.bbox_max is None:
            return

        old_min = h.bbox_min.copy()
        old_max = h.bbox_max.copy()
        old_dims = np.maximum(old_max - old_min, 1e-6)

        handle_pos_world = np.array(event.target.position, dtype=np.float64)
        dragged_corner = np.where(signs > 0, old_max, old_min)

        # Constrain scaling to the dominant moved axis to avoid coupling
        # between axes from drag jitter.
        delta = handle_pos_world - dragged_corner
        axis = int(np.argmax(np.abs(delta)))

        opposite_axis = old_min[axis] if signs[axis] > 0 else old_max[axis]
        moved_axis = float(handle_pos_world[axis])

        new_min = old_min.copy()
        new_max = old_max.copy()
        if signs[axis] > 0:
            new_min[axis] = opposite_axis
            new_max[axis] = max(moved_axis, opposite_axis + 0.02)
        else:
            new_max[axis] = opposite_axis
            new_min[axis] = min(moved_axis, opposite_axis - 0.02)

        old_extent_axis = max(float(old_dims[axis]), 1e-6)
        new_extent_axis = max(float(new_max[axis] - new_min[axis]), 0.02)
        uniform_scale = new_extent_axis / old_extent_axis

        # World direction of this scaling interaction.
        scale_dir = np.zeros(3, dtype=np.float64)
        scale_dir[axis] = 1.0 if signs[axis] > 0 else -1.0

        for box in self.selected_boxes:
            R = _quat_to_rotmat(box.wxyz)
            old_box_dims = box.dimensions.copy()

            # Keep aspect ratio for each contained box via uniform scaling.
            # Clamp by minimum edge size while preserving uniformity.
            min_uniform = np.max(0.02 / np.maximum(old_box_dims, 1e-9))
            box_scale = max(uniform_scale, float(min_uniform))

            # Static point: corner furthest along scaling direction.
            corners_world = []
            for corner_signs in _CORNER_SIGNS:
                local_off = corner_signs * (old_box_dims / 2.0)
                corners_world.append(box.center + R @ local_off)
            corners_world = np.stack(corners_world, axis=0)
            static_idx = int(np.argmin(corners_world @ scale_dir))
            static_corner = corners_world[static_idx]

            # Move center toward/away from static corner at container scale rate.
            box.center = static_corner + (box.center - static_corner) * box_scale
            box.dimensions = old_box_dims * box_scale
            self._update_box_in_place(box, update_gizmo=False, update_dimensions=True)

        # Redraw aggregate handles from the dragged container AABB directly,
        # independent of any fitting back to child boxes.
        self._set_multi_handles_from_aabb(
            new_min,
            new_max,
            update_gizmo_pose=True,
        )
        self._mark_out_of_sync()

    def _on_multi_corner_drag_end(self):
        if len(self._selected_ids) <= 1:
            return
        self._create_or_update_multi_selection_handles()

    def _update_box_in_place(
        self,
        box: Box3D,
        update_gizmo: bool = True,
        update_dimensions: bool = False,
    ):
        """Move / rotate existing scene handles without tearing them down."""
        h = self._handles.get(box.id)
        if h is None:
            return

        R = _quat_to_rotmat(box.wxyz)
        pos = tuple(box.center.tolist())
        wxyz = tuple(box.wxyz.tolist())

        # Box mesh + wireframe
        if h.box_handle is not None:
            h.box_handle.position = pos
            h.box_handle.wxyz = wxyz
            if update_dimensions:
                h.box_handle.dimensions = tuple(box.dimensions.tolist())
        if h.wire_handle is not None:
            h.wire_handle.position = pos
            h.wire_handle.wxyz = wxyz
            if update_dimensions:
                h.wire_handle.dimensions = tuple(box.dimensions.tolist())

        # Label
        label_offset = np.array([0.0, 0.0, box.dimensions[2] / 2 + 0.08])
        label_world = box.center + R @ label_offset
        if h.label_handle is not None:
            h.label_handle.position = tuple(label_world.tolist())

        # Corner gizmos
        for ci, signs in enumerate(_CORNER_SIGNS):
            local_off = signs * (box.dimensions / 2)
            world_pos = box.center + R @ local_off
            wp = tuple(world_pos.tolist())
            if ci < len(h.corner_handles):
                h.corner_handles[ci].position = wp
            if ci < len(h.corner_gizmo_handles):
                h.corner_gizmo_handles[ci].position = wp
                corner_wxyz = _corner_gizmo_wxyz(box.wxyz, signs)
                h.corner_gizmo_handles[ci].wxyz = tuple(corner_wxyz.tolist())

        # Central gizmo
        if update_gizmo and h.gizmo_handle is not None:
            self._set_gizmo_pose_from_center(h, box.center, box.wxyz)

        if self.is_multi_selected and box.id in self._selected_ids:
            self._set_per_box_controls_visible(box.id, False)

    def _reset_single_gizmo_pose(self, box_id: str):
        box = self._boxes.get(box_id)
        handles = self._handles.get(box_id)
        if box is None or handles is None:
            return
        self._set_gizmo_pose_from_center(handles, box.center, box.wxyz)
        if box_id in self._selected_ids:
            self._update_box_color(box_id, selected=True)

    def _reset_multi_gizmo_pose(self):
        if len(self._selected_ids) <= 1:
            return
        h = self._multi_handles
        if h.gizmo_handle is None:
            return
        aabb = self._compute_selection_aabb()
        if aabb is None:
            return
        center, _, bbox_min, bbox_max = aabb
        self._set_gizmo_pose_from_center(h, center, np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
        h.bbox_min = bbox_min.copy()
        h.bbox_max = bbox_max.copy()

    # ── GUI callbacks ────────────────────────────────────────────────────

    def on_label_changed(self, new_label: str):
        if self._updating:
            return
        box = self.selected_box
        if box is None:
            return
        box.label = new_label
        h = self._handles.get(box.id)
        if h and h.label_handle:
            h.label_handle.text = new_label
        self._refresh_dropdown()
        self._mark_out_of_sync()

    def on_colour_changed(self, new_colour: str):
        if self._updating:
            return
        box = self.selected_box
        if box is None:
            return
        box.colour = str(new_colour).strip() or "green"
        self._update_box_color(box.id, selected=(box.id in self._selected_ids))
        self._mark_out_of_sync()

    def on_center_changed(self, axis: int, value: float):
        """One of the center position inputs was edited — update box position."""
        if self._updating:
            return
        box = self.selected_box
        if box is None:
            return
        box.center[axis] = value
        self._rebuild_box_visuals(box)
        if box.id in self._selected_ids:
            self._update_box_color(box.id, selected=True)
        self._mark_out_of_sync()

    def on_dimension_changed(self, axis: int, value: float):
        if self._updating:
            return
        box = self.selected_box
        if box is None:
            return
        value = max(value, 0.02)
        box.dimensions[axis] = value
        self._rebuild_box_visuals(box)
        if box.id in self._selected_ids:
            self._update_box_color(box.id, selected=True)
        self._mark_out_of_sync()

    def on_euler_changed(self):
        """One of the euler angle inputs was edited — update box orientation."""
        if self._updating:
            return
        box = self.selected_box
        if box is None:
            return
        if self._euler_roll is None or self._euler_pitch is None or self._euler_yaw is None:
            return
        roll = np.radians(self._euler_roll.value)
        pitch = np.radians(self._euler_pitch.value)
        yaw = np.radians(self._euler_yaw.value)
        box.wxyz = vtf.SO3.from_rpy_radians(roll, pitch, yaw).wxyz
        self._rebuild_box_visuals(box)
        if box.id in self._selected_ids:
            self._update_box_color(box.id, selected=True)
        self._mark_out_of_sync()

    def on_dropdown_changed(self, selected_label: str):
        """Dropdown selection changed → select the corresponding box."""
        if self._updating:
            return
        # Label format: "id: label"
        if not selected_label or selected_label in ("(none)", "(multiple)"):
            self.select_box(None)
            return
        box_id = selected_label.split(":")[0].strip()
        if box_id in self._boxes:
            self.select_box(box_id)

    # ── dropdown helper ──────────────────────────────────────────────────

    def _refresh_dropdown_options_only(self):
        """Update dropdown options without changing selection."""
        if self._box_dropdown is None:
            return
        self._updating = True
        try:
            options = ["(none)"] + [
                f"{b.id}: {b.label}" for b in self._boxes.values()
            ]
            if "(multiple)" not in options:
                options.insert(1, "(multiple)")
            self._box_dropdown.options = options
        finally:
            self._updating = False

    def _refresh_dropdown(self):
        if self._box_dropdown is None:
            return
        self._updating = True
        try:
            options = ["(none)"] + [
                f"{b.id}: {b.label}" for b in self._boxes.values()
            ]
            if "(multiple)" not in options:
                options.insert(1, "(multiple)")
            self._box_dropdown.options = options
            if len(self._selected_ids) > 1:
                self._box_dropdown.value = "(multiple)"
            elif self._selected_id and self._selected_id in self._boxes:
                box = self._boxes[self._selected_id]
                self._box_dropdown.value = f"{box.id}: {box.label}"
            else:
                self._box_dropdown.value = "(none)"
        finally:
            self._updating = False

    # ── click-to-place ───────────────────────────────────────────────────

    def enter_placement_mode(self, client: viser.ClientHandle):
        """Activate click-to-place: next scene click creates a box."""
        self.cancel_selection()
        self._corner_mode = False
        self._box_by_select_mode = False
        self._placing_mode = True
        self._set_status("🎯 Click on point cloud to place a box...")

        @client.scene.on_pointer_event(event_type="click")
        def _on_click(event: viser.ScenePointerEvent):
            if not self._placing_mode:
                return
            self._placing_mode = False
            client.scene.remove_pointer_callback()
            self._set_status("")

            ray_origin = np.array(event.ray_origin)
            ray_dir = np.array(event.ray_direction)
            ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-12)

            hit = self._ray_nearest_point(ray_origin, ray_dir)
            if hit is not None:
                self.create_box(hit)

        @client.scene.on_pointer_callback_removed
        def _():
            self._placing_mode = False
            self._set_status("")

    def cancel_placement(self):
        self._placing_mode = False
        self._set_status("")

    # ── click-to-place (8 corners) ─────────────────────────────────────

    def _clear_corner_label(self):
        if self._corner_label_handle is not None:
            try:
                self._corner_label_handle.remove()
            except Exception:
                pass
        self._corner_label_handle = None

    def _update_corner_label(self, text: str, position: np.ndarray):
        label_pos = position + np.array([0.0, 0.0, 0.02], dtype=np.float64)
        if self._corner_label_handle is None:
            if self._corner_label_name is None:
                self._corner_label_name = (
                    f"/annotations/corner_progress_{uuid.uuid4().hex[:8]}"
                )
            self._corner_label_handle = self.server.scene.add_label(
                name=self._corner_label_name,
                text=text,
                position=tuple(label_pos.tolist()),
            )
            return
        self._corner_label_handle.text = text
        self._corner_label_handle.position = tuple(label_pos.tolist())

    def _finalize_corner_box(self):
        if len(self._corner_points) < 8:
            return
        pts = np.stack(self._corner_points, axis=0)
        center, dims, wxyz = self._compute_aabb_from_points(pts)

        box = self.create_box(center)
        box.dimensions = dims.astype(np.float64)
        box.wxyz = wxyz.astype(np.float64)
        self._rebuild_box_visuals(box)
        if box.id in self._selected_ids:
            self._sync_gui_to_box(box)

        self._set_status("✓ Box created from 8 corners")

    def enter_corner_box_mode(self, client: viser.ClientHandle):
        """Activate box-by-corner: click 8 points to define a box."""
        self.cancel_placement()
        self.cancel_selection()
        self._corner_mode = True
        self._corner_points = []
        self._corner_client = client
        self._set_status("🎯 Click 8 corners to create a box (0/8)")

        client.scene.remove_pointer_callback()
        self._clear_corner_label()
        self._notify(
            title="Box By Corner",
            body="Selected Point 0 out of 8",
            client=client,
        )

        @client.scene.on_pointer_event(event_type="click")
        def _on_click(event: viser.ScenePointerEvent):
            if not self._corner_mode:
                return
            ray_origin = np.array(event.ray_origin)
            ray_dir = np.array(event.ray_direction)
            ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-12)

            hit = self._ray_nearest_point(
                ray_origin, ray_dir, points=self.dense_points
            )
            if hit is None:
                return
            self._corner_points.append(hit)
            self._notify(
                title="Box By Corner",
                body=f"Selected Point {len(self._corner_points)} out of 8",
                client=client,
            )
            self._set_status(
                f"🎯 Click 8 corners to create a box ({len(self._corner_points)}/8)"
            )

            if len(self._corner_points) >= 8:
                self._corner_mode = False
                client.scene.remove_pointer_callback()
                self._finalize_corner_box()
                self._clear_corner_label()
                self._corner_points = []
                self._corner_client = None
                self._set_status("")
            elif self._corner_label_handle is not None:
                self._corner_label_handle.text = f"{len(self._corner_points)}/8"

        @client.scene.on_pointer_callback_removed
        def _():
            if self._corner_mode:
                self._corner_mode = False
                self._corner_points = []
                self._corner_client = None
                self._clear_corner_label()
                self._set_status("")

    # ── click-to-select ──────────────────────────────────────────────────

    def enter_selection_mode(self, client: viser.ClientHandle):
        """Activate click-to-select: next scene click selects closest box."""
        self.cancel_placement()
        self._corner_mode = False
        self._box_by_select_mode = False
        self._multi_selecting_mode = False
        self._selecting_mode = True
        self._set_status("🎯 Click near a box to select it...")

        @client.scene.on_pointer_event(event_type="click")
        def _on_click(event: viser.ScenePointerEvent):
            if not self._selecting_mode:
                return
            self._selecting_mode = False
            client.scene.remove_pointer_callback()
            self._set_status("")

            ray_origin = np.array(event.ray_origin)
            ray_dir = np.array(event.ray_direction)
            ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-12)

            # Find the box whose center is closest to the click ray
            best_id = None
            best_dist = float("inf")
            for box_id, box in self._boxes.items():
                # Distance from box center to ray
                diff = box.center - ray_origin
                t = np.dot(diff, ray_dir)
                if t < 0:
                    continue  # Box is behind camera
                closest_on_ray = ray_origin + t * ray_dir
                dist = np.linalg.norm(box.center - closest_on_ray)
                if dist < best_dist:
                    best_dist = dist
                    best_id = box_id

            if best_id is not None:
                self.select_box(best_id)
                self._refresh_dropdown()
                self._set_status(f"Selected box: {self._boxes[best_id].label}")
            else:
                self._set_status("No box found near click")

        @client.scene.on_pointer_callback_removed
        def _():
            self._selecting_mode = False
            self._set_status("")

    def enter_multi_selection_mode(self, client: viser.ClientHandle):
        """Activate rectangle-drag multi-select mode for multiple boxes."""
        self.cancel_placement()
        self._corner_mode = False
        self._box_by_select_mode = False
        self._selecting_mode = False
        self._multi_selecting_mode = True
        self._set_status("🟦 Drag a rectangle to multi-select boxes...")

        @client.scene.on_pointer_event(event_type="rect-select")
        def _on_rect_select(event: viser.ScenePointerEvent):
            if not self._multi_selecting_mode:
                return
            self._multi_selecting_mode = False
            client.scene.remove_pointer_callback()

            if len(self._boxes) == 0:
                self._set_status("⚠️ No boxes available")
                return

            camera = event.client.camera
            box_ids = list(self._boxes.keys())
            centers = np.stack([self._boxes[box_id].center for box_id in box_ids], axis=0)
            proj, valid_mask = self._project_points_to_screen(centers, camera)
            if proj.size == 0:
                self._set_status("⚠️ No boxes in view")
                return

            p0 = np.array(event.screen_pos[0], dtype=np.float64)
            p1 = np.array(event.screen_pos[1], dtype=np.float64)
            lo = np.minimum(p0, p1)
            hi = np.maximum(p0, p1)

            inside = (
                (proj[:, 0] >= lo[0])
                & (proj[:, 0] <= hi[0])
                & (proj[:, 1] >= lo[1])
                & (proj[:, 1] <= hi[1])
            )

            visible_indices = np.flatnonzero(valid_mask)
            selected_indices = visible_indices[inside]
            if selected_indices.size == 0:
                self.clear_selection()
                self._set_status("⚠️ No boxes selected")
                return

            selected_ids = [box_ids[i] for i in selected_indices]
            self.set_selection(selected_ids)
            self._set_status(f"🎯 Multi-select active: {len(self._selected_ids)} selected")

        @client.scene.on_pointer_callback_removed
        def _():
            self._multi_selecting_mode = False
            self._set_status("")

    def cancel_selection(self):
        self._selecting_mode = False
        self._multi_selecting_mode = False
        self._set_status("")

    # ── box-by-select (rect-select) ─────────────────────────────────────

    def enter_box_by_select_mode(self, client: viser.ClientHandle):
        """Activate rect-select to fit an OOBB to selected dense points."""
        self.cancel_placement()
        self.cancel_selection()
        self._corner_mode = False
        self._box_by_select_mode = True

        self._set_status("🟦 Drag a rectangle to select points...")

        @client.scene.on_pointer_event(event_type="rect-select")
        def _on_rect_select(event: viser.ScenePointerEvent):
            if not self._box_by_select_mode:
                return
            self._box_by_select_mode = False
            client.scene.remove_pointer_callback()
            self._set_status("")

            if self.dense_points is None or len(self.dense_points) == 0:
                self._set_status("⚠️ Dense point cloud not available")
                return

            camera = event.client.camera
            proj, valid_mask = self._project_points_to_screen(
                self.dense_points, camera
            )
            if proj.size == 0:
                self._set_status("⚠️ No points in view")
                return

            p0 = np.array(event.screen_pos[0], dtype=np.float64)
            p1 = np.array(event.screen_pos[1], dtype=np.float64)
            lo = np.minimum(p0, p1)
            hi = np.maximum(p0, p1)

            inside = (
                (proj[:, 0] >= lo[0])
                & (proj[:, 0] <= hi[0])
                & (proj[:, 1] >= lo[1])
                & (proj[:, 1] <= hi[1])
            )

            selected_idx = np.flatnonzero(valid_mask)
            selected_idx = selected_idx[inside]
            if selected_idx.size == 0:
                self._set_status("⚠️ No points selected")
                return

            selected_pts = self.dense_points[selected_idx]
            selected_pts = self._augment_rect_select_points(
                selected_pts, camera, lo, hi
            )
            center, dims, wxyz = self._compute_oobb_from_points_and_camera(
                selected_pts, camera
            )

            box = self.create_box(center)
            box.dimensions = dims.astype(np.float64)
            box.wxyz = wxyz.astype(np.float64)
            self._rebuild_box_visuals(box)
            if box.id in self._selected_ids:
                self._sync_gui_to_box(box)
            self._set_status("✓ Box created from selection")

        @client.scene.on_pointer_callback_removed
        def _():
            if self._box_by_select_mode:
                self._box_by_select_mode = False
                self._set_status("")

    # ── CG bbox integration ──────────────────────────────────────────────

    def set_cg_bboxes(self, bboxes: list[dict]):
        """Set the ConceptGraph bounding boxes for auto-adjust feature.
        
        Each bbox should be a dict with 'center' and 'extent' keys (numpy arrays).
        """
        self._cg_bboxes = bboxes

    def auto_adjust_to_nearest_cg_bbox(self) -> bool:
        """Adjust selected box to match the nearest CG bounding box.
        
        Returns True if adjustment was made, False otherwise.
        """
        box = self.selected_box
        if box is None:
            self._set_status("⚠️ No box selected")
            return False

        if self._cg_bboxes is None or len(self._cg_bboxes) == 0:
            self._set_status("⚠️ No CG bboxes available")
            return False

        # Find nearest CG bbox by center distance
        best_idx = None
        best_dist = float("inf")
        for i, cg_bbox in enumerate(self._cg_bboxes):
            cg_center = np.array(cg_bbox["center"])
            dist = np.linalg.norm(box.center - cg_center)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx is None:
            self._set_status("⚠️ Could not find nearest CG bbox")
            return False

        cg_bbox = self._cg_bboxes[best_idx]
        cg_center = np.array(cg_bbox["center"], dtype=np.float64)
        cg_extent = np.array(cg_bbox["extent"], dtype=np.float64)

        # Update box position and dimensions to match CG bbox AABB
        box.center = cg_center
        box.dimensions = cg_extent
        # Reset rotation to identity (axis-aligned)
        box.wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # Rebuild visuals and sync GUI
        self._rebuild_box_visuals(box)
        if box.id in self._selected_ids:
            self._update_box_color(box.id, selected=True)
            self._sync_gui_to_box(box)

        self._set_status(f"✓ Adjusted to CG bbox (dist={best_dist:.2f}m)")
        self._mark_out_of_sync()
        return True

    # ── persistence ──────────────────────────────────────────────────────

    def save(self, path: str | Path):
        path = Path(path)
        data = [b.to_dict() for b in self._boxes.values()]
        path.write_text(json.dumps(data, indent=2))
        log.info("Saved %d boxes to %s", len(data), path)
        self._set_status(f"💾 Saved {len(data)} boxes")
        self._mark_synced()

    def load(self, path: str | Path):
        path = Path(path)
        if not path.exists():
            log.warning("Annotation file not found: %s", path)
            self._set_status("⚠️ File not found")
            return
        data = json.loads(path.read_text())
        # Clear current boxes
        self.delete_all()
        for d in data:
            box = Box3D.from_dict(d)
            self._next_idx = max(self._next_idx, int(box.label.split("_")[-1]) + 1) if box.label.startswith("box_") else self._next_idx
            with self._lock:
                self._boxes[box.id] = box
            self._add_box_to_scene(box)
        self._refresh_dropdown()
        log.info("Loaded %d boxes from %s", len(data), path)
        self._set_status(f"📂 Loaded {len(data)} boxes")
        self._mark_synced()

    # ── public helpers ───────────────────────────────────────────────────

    @property
    def box_count(self) -> int:
        return len(self._boxes)

    def get_boxes(self) -> list[Box3D]:
        return list(self._boxes.values())


# ──────────────────────────────────────────────────────────────────────────────
# GUI setup
# ──────────────────────────────────────────────────────────────────────────────


def setup_box_annotator_gui(
    server: viser.ViserServer,
    controller: BoxAnnotatorController,
    *,
    save_dir: Path | None = None,
    expanded: bool = True,
):
    """Build the sidebar GUI panel for the box annotator.

    Parameters
    ----------
    server : viser.ViserServer
    controller : BoxAnnotatorController
    save_dir : Path, optional
        Directory where annotation JSON files are saved / loaded.
    expanded : bool
        Whether the folder starts expanded.
    """
    if save_dir is None:
        save_dir = Path(".")

    save_dir_ref = {"path": Path(save_dir)}

    def _reload_for_map(selected_map_path: Path):
        save_dir_ref["path"] = Path(selected_map_path)
        reload_box_annotator_for_map(
            controller,
            selected_map_path,
            filename=filename_input.value,
            show_bboxes=show_bboxes_checkbox.value,
        )

    with server.gui.add_folder("📦 Box Annotator", expand_by_default=expanded):
        # Persistent file-sync indicator (always visible)
        sync_indicator = server.gui.add_markdown("**🟢 Synced to file**")
        controller._sync_indicator_text = sync_indicator
        controller._set_sync_indicator()

        # Status line
        status = server.gui.add_markdown(
            "*Ready — click 'Draw Box' to begin*"
        )
        controller._status_text = status

        # ── Creation / deletion buttons ──────────────────────────────────
        with server.gui.add_folder("Actions", expand_by_default=True):
            show_bboxes_checkbox = server.gui.add_checkbox(
                "Show Bboxes", initial_value=True
            )
            draw_box_icon = getattr(viser.Icon, "RECTANGLE", viser.Icon.POINTER)
            select_rect_btn = server.gui.add_button("Draw Box", icon=draw_box_icon)
            auto_adjust_btn = server.gui.add_button(
                "Auto-Adjust Box", icon=viser.Icon.BOX_ALIGN_BOTTOM
            )
            dup_btn = server.gui.add_button("Duplicate Selected", icon=viser.Icon.COPY)
            del_btn = server.gui.add_button("Delete Selected", icon=viser.Icon.TRASH)
            del_all_btn = server.gui.add_button(
                "Delete All", color="red", icon=viser.Icon.TRASH_X
            )

        # ── Selection ─────────────────────────────────────────────────────
        select_btn = server.gui.add_button("Select Box", icon=viser.Icon.POINTER)
        multi_select_icon = getattr(viser.Icon, "RECTANGLE", viser.Icon.POINTER)
        multi_select_btn = server.gui.add_button("Select Box", icon=multi_select_icon)
        global_multi_rotate_checkbox = server.gui.add_checkbox(
            "Global Multi-Select Rotation", initial_value=False
        )
        deselect_all_btn = server.gui.add_button("Deselect All Boxes")
        dropdown = server.gui.add_dropdown(
            "Select Box",
            options=["(none)", "(multiple)"],
            initial_value="(none)",
        )
        dropdown.disabled = True
        controller._box_dropdown = dropdown

        # ── Properties ───────────────────────────────────────────────────
        with server.gui.add_folder("Properties", expand_by_default=False):
            label_input = server.gui.add_text("Label", initial_value="object")
            colour_input = server.gui.add_text("Colour", initial_value="green")
            server.gui.add_markdown("*Use a name (e.g., `green`) or hex (e.g., `#00ff00`)*")
            pos_x = server.gui.add_number("Pos X", initial_value=0.0, step=0.01)
            pos_y = server.gui.add_number("Pos Y", initial_value=0.0, step=0.01)
            pos_z = server.gui.add_number("Pos Z", initial_value=0.0, step=0.01)
            dim_x = server.gui.add_number("Size X", initial_value=0.5, min=0.02, max=50.0, step=0.01)
            dim_y = server.gui.add_number("Size Y", initial_value=0.5, min=0.02, max=50.0, step=0.01)
            dim_z = server.gui.add_number("Size Z", initial_value=0.5, min=0.02, max=50.0, step=0.01)
            euler_roll = server.gui.add_number("Roll  (X°)", initial_value=0.0, min=-180.0, max=180.0, step=0.5)
            euler_pitch = server.gui.add_number("Pitch (Y°)", initial_value=0.0, min=-180.0, max=180.0, step=0.5)
            euler_yaw = server.gui.add_number("Yaw   (Z°)", initial_value=0.0, min=-180.0, max=180.0, step=0.5)

        controller._label_input = label_input
        controller._colour_input = colour_input
        controller._pos_x = pos_x
        controller._pos_y = pos_y
        controller._pos_z = pos_z
        controller._dim_x = dim_x
        controller._dim_y = dim_y
        controller._dim_z = dim_z
        controller._euler_roll = euler_roll
        controller._euler_pitch = euler_pitch
        controller._euler_yaw = euler_yaw
        controller._property_controls = [
            label_input,
            colour_input,
            pos_x,
            pos_y,
            pos_z,
            dim_x,
            dim_y,
            dim_z,
            euler_roll,
            euler_pitch,
            euler_yaw,
        ]

        # ── Save / Load ──────────────────────────────────────────────────
        with server.gui.add_folder("Save / Load", expand_by_default=True):
            filename_input = server.gui.add_text(
                "Filename", initial_value="annotated_bboxes.json"
            )
            save_btn = server.gui.add_button("Save", icon=viser.Icon.DEVICE_FLOPPY)
            load_btn = server.gui.add_button("Re/Load", icon=viser.Icon.UPLOAD)

        # ── Wire up callbacks ────────────────────────────────────────────

        @show_bboxes_checkbox.on_update
        def _(_):
            controller.set_bboxes_visible(show_bboxes_checkbox.value)

        @select_rect_btn.on_click
        def _(event: viser.GuiEvent):
            if event.client is not None:
                controller.enter_box_by_select_mode(event.client)

        @dup_btn.on_click
        def _(_):
            boxes = controller.selected_boxes
            if len(boxes) == 0:
                return
            offset = np.array([0.3, 0.0, 0.0])
            new_ids: list[str] = []
            for box in boxes:
                new_box = controller.create_box(
                    box.center + offset, label=box.label + "_copy"
                )
                new_box.dimensions = box.dimensions.copy()
                new_box.wxyz = box.wxyz.copy()
                new_box.colour = box.colour
                controller._rebuild_box_visuals(new_box)
                new_ids.append(new_box.id)
            controller.set_selection(new_ids)

        @del_btn.on_click
        def _(_):
            controller.delete_selected()

        @del_all_btn.on_click
        def _(_):
            controller.delete_all()

        @select_btn.on_click
        def _(event: viser.GuiEvent):
            if event.client is not None:
                controller.enter_selection_mode(event.client)

        @multi_select_btn.on_click
        def _(event: viser.GuiEvent):
            if event.client is not None:
                controller.enter_multi_selection_mode(event.client)

        @global_multi_rotate_checkbox.on_update
        def _(_):
            controller.set_global_multi_select_rotation(
                global_multi_rotate_checkbox.value
            )

        @deselect_all_btn.on_click
        def _(_):
            controller.clear_selection()

        @auto_adjust_btn.on_click
        def _(_):
            controller.auto_adjust_to_nearest_cg_bbox()

        @label_input.on_update
        def _(_):
            controller.on_label_changed(label_input.value)

        @colour_input.on_update
        def _(_):
            controller.on_colour_changed(colour_input.value)

        @pos_x.on_update
        def _(_):
            controller.on_center_changed(0, pos_x.value)

        @pos_y.on_update
        def _(_):
            controller.on_center_changed(1, pos_y.value)

        @pos_z.on_update
        def _(_):
            controller.on_center_changed(2, pos_z.value)

        @dim_x.on_update
        def _(_):
            controller.on_dimension_changed(0, dim_x.value)

        @dim_y.on_update
        def _(_):
            controller.on_dimension_changed(1, dim_y.value)

        @dim_z.on_update
        def _(_):
            controller.on_dimension_changed(2, dim_z.value)

        @euler_roll.on_update
        def _(_):
            controller.on_euler_changed()

        @euler_pitch.on_update
        def _(_):
            controller.on_euler_changed()

        @euler_yaw.on_update
        def _(_):
            controller.on_euler_changed()

        @save_btn.on_click
        def _(_):
            controller.save(save_dir_ref["path"] / filename_input.value)

        @load_btn.on_click
        def _(_):
            controller.load(save_dir_ref["path"] / filename_input.value)
            controller.set_bboxes_visible(show_bboxes_checkbox.value)

        # Auto-load default annotations on startup when available.
        default_annotations_path = save_dir_ref["path"] / "annotated_bboxes.json"
        if default_annotations_path.exists():
            controller.load(default_annotations_path)
            controller.set_bboxes_visible(True)
            show_bboxes_checkbox.value = True

        controller.set_properties_enabled(True)

    return _reload_for_map
