# Multi-Select Boxes Refactor TODOs

This plan is based on current behavior in `vis/box_annotator.py`:
- single-selection state is `BoxAnnotatorController._selected_id`
- click-to-select is `enter_selection_mode()`
- scene controls are created per-box in `_add_box_to_scene()`
- per-box controls are updated in `_update_box_in_place()` / `_rebuild_box_visuals()`
- Properties UI is built in `setup_box_annotator_gui()`

---

## Goals (from product request)

- Add a new button under `Select Box` for multi-selection mode (`rectangle icon` + `Select Box`).
- Allow selecting many boxes and editing them together.
- When selection size is `> 1`:
  - hide orientation + corner handles from all selected boxes,
  - show one aggregate set of orientation + corner handles using the *selection bounding box* logic (without rendering that bbox),
  - disable Properties UI section.

---

## P0: State model + mode wiring

- [ ] Replace single selection state with multi-capable state:
  - keep `_selected_id` for backward compatibility during migration,
  - add `_selected_ids: set[str]` as source of truth,
  - add helper APIs:
    - `selected_boxes` (list of `Box3D`)
    - `is_multi_selected` (len > 1)
    - `set_selection(ids)` / `clear_selection()` / `toggle_selected(id)`.

- [ ] Add multi-select mode flag and lifecycle:
  - add `_multi_selecting_mode` flag,
  - ensure mutual exclusion with `_selecting_mode`, `_placing_mode`, `_corner_mode`, `_box_by_select_mode`,
  - update `cancel_selection()` to clear both select modes.

- [ ] Keep dropdown behavior deterministic with multi-select:
  - in multi-select (`>1`), dropdown should not pretend to represent all selections,
  - define fallback value (`(multiple)` or `(none)`), and keep options refresh safe.

---

## P1: GUI changes

- [ ] In `setup_box_annotator_gui()`, add new button directly under existing pointer `Select Box` button:
  - label: `Select Box`
  - icon: rectangle icon from `viser.Icon` (confirm exact enum available in runtime).

- [ ] Wire button callback to a new controller entrypoint:
  - `enter_multi_selection_mode(client)`.

- [ ] Track all Properties controls in one collection for easy enable/disable:
  - include `label_input`, position, size, and euler inputs,
  - add helper `set_properties_enabled(enabled: bool)`.

- [ ] Disable Properties section whenever selected count is `>1`; re-enable for `0/1`.

---

## P2: Multi-select picking behavior

- [ ] Implement `enter_multi_selection_mode(client)`:
  - use click picking logic similar to `enter_selection_mode()`,
  - each click toggles closest box membership in `_selected_ids`,
  - keep mode active across multiple clicks until pointer callback is removed/canceled,
  - show status text with selected count.

- [ ] Ensure current single-select mode remains unchanged:
  - pointer `Select Box` should still select exactly one box and exit mode.

- [ ] Update any paths that assume one selected box:
  - `delete_selected()` behavior when many selected (delete all selected),
  - `duplicate selected` behavior (either disable for multi-select or duplicate all selected in a predictable offset).

---

## P3: Aggregate transform/resize handles

- [ ] Introduce aggregate selection handles state (new dataclass or fields):
  - one central gizmo handle,
  - 8 corner gizmo handles + spheres,
  - prev transform tracking (similar to per-box `prev_gizmo_*`).

- [ ] Add helper to compute logical selection OBB/AABB from selected boxes:
  - input: selected box centers/corners,
  - output: aggregate center/dimensions/wxyz used only for control placement.

- [ ] On transition to multi-select (`len > 1`):
  - hide per-box orientation + corner gizmos for selected boxes,
  - keep visual box meshes/wires visible,
  - create/show aggregate handles positioned from selection-bounds logic.

- [ ] On transition back to single/none:
  - remove aggregate handles,
  - restore normal per-box handle visibility for selected single box.

- [ ] Implement aggregate central gizmo update callback:
  - translation applies same world delta to every selected box center,
  - rotation applies same rotation delta around aggregate pivot to:
    - each selected box center,
    - each selected box orientation.

- [ ] Implement aggregate corner drag update callback:
  - derive scale/change from aggregate corner movement,
  - apply consistent resize transform to all selected boxes,
  - enforce min dimensions per box (`>= 0.02`).

- [ ] Reuse existing fast update path where possible:
  - batch call `_update_box_in_place()` for all selected,
  - avoid full `_rebuild_box_visuals()` during drag.

---

## P4: Selection visuals + property synchronization

- [ ] Update color logic for multi-select:
  - all selected boxes use selected color,
  - deselected return to default color.

- [ ] Keep GUI sync semantics clear:
  - for 1 selected: `_sync_gui_to_box()` as today,
  - for >1 selected: avoid writing mixed values into inputs; disable properties.

- [ ] Audit callbacks that currently use `selected_box`:
  - `on_label_changed`, `on_center_changed`, `on_dimension_changed`, `on_euler_changed`, `auto_adjust_to_nearest_cg_bbox`,
  - guard against accidental writes when multi-select is active.

---

## P5: Cleanup + regression checks

- [ ] Add focused helper methods to reduce branching:
  - `_selection_changed()` as single place to refresh colors, dropdown, properties enablement, and handles.

- [ ] Manual validation checklist:
  - single select still works,
  - multi-select button can add/remove boxes by click,
  - when 2+ selected: no per-box orientation/corner handles visible,
  - aggregate handles appear and move/rotate/scale all selected boxes,
  - Properties section is disabled only for 2+ selected,
  - returning to 1 selected restores per-box handles + Properties.

- [ ] Save/load sanity:
  - save after multi-transform and verify loaded boxes preserve final positions, sizes, orientations.

---

## Implementation notes (risk reducers)

- Prefer introducing multi-select with compatibility shims first, then remove `_selected_id` dependencies in a second pass.
- Keep pointer-callback teardown behavior explicit to avoid mode leaks between select/place/rect-select tools.
- Avoid updating GUI input values while disabled to prevent stale callback cascades.
