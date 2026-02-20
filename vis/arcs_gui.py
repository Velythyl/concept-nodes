import logging
from typing import TYPE_CHECKING

import numpy as np
from matplotlib.pyplot import get_cmap

if TYPE_CHECKING:  # pragma: no cover
    from vis.vis_with_viser import ViserCallbackManager

log = logging.getLogger(__name__)

FORWARD_ARC_CMAP = "winter"  # Top half: blue-green to green
DEPENDENCY_ARC_CMAP = "autumn"  # Bottom half: red to orangeish


class ArcGUIController:
    """Handles arc visualization (splines) and related GUI state."""

    def __init__(self, manager: "ViserCallbackManager") -> None:
        self.manager = manager
        self.arc_state_slider = None
        self.arc_state_label = None
        self.forward_arcs_checkbox = None
        self.dependency_arcs_checkbox = None
        self.all_arcs_checkbox = None
        self._syncing_arc_controls = False
        self.forward_arc_handles = []
        self.dependency_arc_handles = []
        self.arc_handles = []
        self.forward_arcs_visible = False
        self.dependency_arcs_visible = False

    def get_current_timestep_from_slider(self) -> tuple[int, float]:
        """Get the current day and hour from the arc state slider.

        Returns:
            Tuple of (current_day, current_hour) from the currently selected arc state.
        """
        if not self.manager.arc_states:
            return (1, 0.0)

        idx = self.manager.current_arc_state_index
        if idx < 0 or idx >= len(self.manager.arc_states):
            idx = 0

        state = self.manager.arc_states[idx]
        day = state.get("current_day", 1)
        hour = state.get("current_hour", 0.0)
        return (day, hour)

    def set_arc_state_slider(self, slider):
        """Store reference to the arc state slider."""
        self.arc_state_slider = slider

    def set_arc_controls(self, *, slider, label, forward_btn, dependency_btn, all_btn):
        """Store arc control handles for runtime refreshes."""
        self.arc_state_slider = slider
        self.arc_state_label = label
        self.forward_arcs_checkbox = forward_btn
        self.dependency_arcs_checkbox = dependency_btn
        self.all_arcs_checkbox = all_btn
        self._sync_arc_checkbox_values()

    def _sync_arc_checkbox_values(self):
        """Sync checkbox values to current arc visibility without triggering handlers."""
        self._syncing_arc_controls = True
        try:
            if self.forward_arcs_checkbox is not None:
                self.forward_arcs_checkbox.value = self.forward_arcs_visible
            if self.dependency_arcs_checkbox is not None:
                self.dependency_arcs_checkbox.value = self.dependency_arcs_visible
            if self.all_arcs_checkbox is not None:
                self.all_arcs_checkbox.value = (
                    self.forward_arcs_visible and self.dependency_arcs_visible
                )
        finally:
            self._syncing_arc_controls = False

    def refresh_controls(self):
        """Refresh arc GUI controls after map changes."""
        num_states = len(self.manager.arc_states) if self.manager.arc_states else 1
        if self.manager.current_arc_state_index >= num_states:
            self.manager.current_arc_state_index = max(0, num_states - 1)

        if self.arc_state_slider is not None:
            self.arc_state_slider.min = 0
            self.arc_state_slider.max = max(0, num_states - 1)
            self.arc_state_slider.step = 1
            self.arc_state_slider.value = int(self.manager.current_arc_state_index)
            self.arc_state_slider.disabled = not self.manager.can_show_arcs

        if self.arc_state_label is not None:
            self.arc_state_label.content = self.get_current_arc_state_info()

        if self.forward_arcs_checkbox is not None:
            self.forward_arcs_checkbox.disabled = not self.manager.can_show_arcs
        if self.dependency_arcs_checkbox is not None:
            self.dependency_arcs_checkbox.disabled = not self.manager.can_show_arcs
        if self.all_arcs_checkbox is not None:
            self.all_arcs_checkbox.disabled = not self.manager.can_show_arcs

        self._sync_arc_checkbox_values()

    def get_current_arcs(self) -> list[dict]:
        """Get arcs from the currently selected ArcState."""
        if not self.manager.arc_states or self.manager.current_arc_state_index >= len(self.manager.arc_states):
            return []
        return self.manager.arc_states[self.manager.current_arc_state_index].get("arcs", [])

    def get_current_arc_state_info(self) -> str:
        """Get a markdown string describing the current arc state (day/hour)."""
        if not self.manager.arc_states or self.manager.current_arc_state_index >= len(self.manager.arc_states):
            return "**No arc states**"
        state = self.manager.arc_states[self.manager.current_arc_state_index]
        day = state.get("current_day", 0)
        hour = state.get("current_hour", 0)
        return f"**Day {day}, Hour {hour:.1f}**"

    def set_arc_state_index(self, index: int):
        """Set the current arc state index and refresh arc visualization."""
        if not self.manager.arc_states:
            return
        self.manager.current_arc_state_index = max(0, min(index, len(self.manager.arc_states) - 1))
        # Refresh arcs if any are visible
        if self.forward_arcs_visible:
            self.add_forward_arcs()
        if self.dependency_arcs_visible:
            self.add_dependency_arcs()

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
            source_id = self.manager._resolve_object_id(raw_source)
            target_id = self.manager._resolve_object_id(raw_target)

            # Skip invalid arcs
            if source_id is None or target_id is None:
                continue
            if source_id >= self.manager.num_objects or target_id >= self.manager.num_objects:
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
                log.info("Arc group %s: %s arcs -> %s", key, num_arcs, arc_info_list)
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
                    log.info("  Arc %s: offset=%s, reversed=%s", arc_idx, offset_multiplier, is_reversed)

        if offset_map:
            log.info("Final offset_map: %s", offset_map)
        return offset_map

    def _detect_arc_overlaps(self, typed_arcs: list) -> dict:
        """Detect overlapping arcs and assign offsets to separate them."""
        # Use the global overlap detection that considers ALL arc types
        return self._detect_arc_overlaps_all()

    def _sample_catmull_rom_spline(self, control_points: np.ndarray, num_samples: int = 50) -> np.ndarray:
        """Sample points along a Catmull-Rom spline."""
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
                (2 * s * p0 + (s - 3) * p1 + (3 - 2 * s) * p2 - s * p3) * t2 +
                (-s * p0 + (2 - s) * p1 + (s - 2) * p2 + s * p3) * t3
            )

        samples = []
        # Number of segments is n-3 for Catmull-Rom (we use overlapping windows of 4 points)
        num_segments = n - 3
        samples_per_segment = max(1, num_samples // num_segments)

        for i in range(num_segments):
            p0, p1, p2, p3 = control_points[i:i + 4]
            for j in range(samples_per_segment):
                t = j / samples_per_segment
                point = catmull_rom_point(p0, p1, p2, p3, t)
                samples.append(point)

        # Add the final point
        samples.append(control_points[-2])  # p2 of the last segment

        return np.array(samples, dtype=np.float32)

    def _add_arcs_by_type(self, arc_type: str, cmap_name: str, name_prefix: str) -> list:
        """Add arcs of a specific type to the scene."""
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
            source_id = self.manager._resolve_object_id(raw_source)
            target_id = self.manager._resolve_object_id(raw_target)

            # Validate source and target indices early
            if source_id is None or target_id is None:
                log.warning(
                    "Arc %s missing or unresolved source/target (%s -> %s), skipping",
                    i,
                    raw_source,
                    raw_target,
                )
                continue
            if source_id >= self.manager.num_objects or target_id >= self.manager.num_objects:
                log.warning("Arc %s has invalid object indices (%s -> %s), skipping", i, source_id, target_id)
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
            source_pos = self.manager.centroids[source_id]
            target_pos = self.manager.centroids[target_id]

            # Add labels for the interacting objects
            self.manager.add_label_for_object(source_id)
            self.manager.add_label_for_object(target_id)

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
            handle = self.manager.server.scene.add_spline_catmull_rom(
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
                    arrow_handle = self.manager.server.scene.add_line_segments(
                        name=f"/{name_prefix}/arc_{i}/arrow_{j}_wing{wing_idx}",
                        points=np.array([[wing_start, point]], dtype=np.float32),
                        colors=color_uint8,
                        line_width=2.0,
                    )
                    handles.append(arrow_handle)

            # Add label at the midpoint if provided
            if label:
                label_handle = self.manager.server.scene.add_label(
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
        self.set_forward_arcs_visible(not self.forward_arcs_visible)

    def toggle_dependency_arcs(self):
        """Toggle dependency arc visibility."""
        self.set_dependency_arcs_visible(not self.dependency_arcs_visible)

    def toggle_all_arcs(self):
        """Toggle all arcs (both forward and dependency)."""
        # Determine target state: if any are visible, hide all; otherwise show all
        any_visible = self.forward_arcs_visible or self.dependency_arcs_visible
        self.set_all_arcs_visible(not any_visible)

    def set_forward_arcs_visible(self, visible: bool):
        """Set forward arc visibility to an explicit state."""
        if self.forward_arcs_visible == visible:
            return
        if visible:
            self.add_forward_arcs()
        else:
            self.remove_forward_arcs()
        self.forward_arcs_visible = visible
        self._sync_arc_checkbox_values()

    def set_dependency_arcs_visible(self, visible: bool):
        """Set dependency arc visibility to an explicit state."""
        if self.dependency_arcs_visible == visible:
            return
        if visible:
            self.add_dependency_arcs()
        else:
            self.remove_dependency_arcs()
        self.dependency_arcs_visible = visible
        self._sync_arc_checkbox_values()

    def set_all_arcs_visible(self, visible: bool):
        """Set all arc visibility state."""
        self.set_forward_arcs_visible(visible)
        self.set_dependency_arcs_visible(visible)
        self._sync_arc_checkbox_values()


def setup_arcs_gui(server, manager, gui_cfg):
    """Set up the GUI controls for arc visualization."""
    arc_gui = manager.arcs_gui
    with server.gui.add_folder("Arcs", expand_by_default=gui_cfg.arcs.expanded):
        # Slider for selecting arc state (timestep)
        num_states = len(manager.arc_states) if manager.arc_states else 1
        arc_state_slider = server.gui.add_slider(
            "Timestep",
            min=0,
            max=max(0, num_states - 1),
            step=1,
            initial_value=0,
        )
        arc_gui.set_arc_state_slider(arc_state_slider)

        # Display current state info
        arc_state_label = server.gui.add_markdown(
            content=arc_gui.get_current_arc_state_info(),
        )

        @arc_state_slider.on_update
        def _(event):
            arc_gui.set_arc_state_index(int(arc_state_slider.value))
            arc_state_label.content = arc_gui.get_current_arc_state_info()

        # Checkboxes for different arc types
        forward_arcs_btn = server.gui.add_checkbox(
            "Forward Arcs", initial_value=arc_gui.forward_arcs_visible
        )
        dependency_arcs_btn = server.gui.add_checkbox(
            "Dependency Arcs", initial_value=arc_gui.dependency_arcs_visible
        )
        all_arcs_btn = server.gui.add_checkbox(
            "All Arcs",
            initial_value=arc_gui.forward_arcs_visible and arc_gui.dependency_arcs_visible,
        )

        arc_gui.set_arc_controls(
            slider=arc_state_slider,
            label=arc_state_label,
            forward_btn=forward_arcs_btn,
            dependency_btn=dependency_arcs_btn,
            all_btn=all_arcs_btn,
        )

        arcs_enabled = manager.can_show_arcs
        arc_state_slider.disabled = not arcs_enabled
        forward_arcs_btn.disabled = not arcs_enabled
        dependency_arcs_btn.disabled = not arcs_enabled
        all_arcs_btn.disabled = not arcs_enabled

        @forward_arcs_btn.on_update
        def _(_):
            if arc_gui._syncing_arc_controls:
                return
            arc_gui.set_forward_arcs_visible(forward_arcs_btn.value)

        @dependency_arcs_btn.on_update
        def _(_):
            if arc_gui._syncing_arc_controls:
                return
            arc_gui.set_dependency_arcs_visible(dependency_arcs_btn.value)

        @all_arcs_btn.on_update
        def _(_):
            if arc_gui._syncing_arc_controls:
                return
            arc_gui.set_all_arcs_visible(all_arcs_btn.value)
