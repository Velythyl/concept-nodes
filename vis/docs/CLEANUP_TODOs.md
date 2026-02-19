# VIS Cleanup TODOs (Modularity + Maintainability)

This backlog is based on a crawl of the `vis` package.

## Snapshot: why cleanup is worth it

- `vis/vis_with_viser.py` is very large (1266 LOC) and combines:
  - data loading/state mutation
  - rendering details
  - GUI orchestration
  - app bootstrap / background workers / HTTP video serving
- `vis/box_annotator.py` is very large (1595 LOC) and mixes:
  - geometry/math utilities
  - scene-handle lifecycle
  - GUI callbacks
  - persistence (save/load)
  - mode state machine logic
- `GUIStateController.switch_to_map()` in `vis/gui_fsm.py` is a long orchestration path that does many responsibilities in one method.
- A layering leak exists: `vis/box_annotator.py` imports `reload_box_annotator_for_map` from `vis/gui_fsm.py`, coupling feature logic to GUI-state module.

---

## P0 (High impact, low risk): do first

- [ ] **Deduplicate map-switch refresh calls**
  - In `GUIStateController.switch_to_map()`, `refresh_controls()` / `arcs_gui.refresh_controls()` / `query_chat.refresh_controls()` are run twice.
  - Keep one post-switch refresh path to reduce accidental drift.

- [ ] **Extract map-switch orchestration into smaller private steps**
  - Split `switch_to_map()` into focused helpers:
    - `_capture_view_state()`
    - `_clear_scene()`
    - `_load_map_and_apply_state()`
    - `_restore_visibility_overlays()`
    - `_refresh_all_controls()`
  - Keep behavior unchanged; target readability and safer edits.

- [ ] **Centralize mode/visibility state into typed dataclasses**
  - Replace parallel booleans on manager (`rgb_enabled`, `bbox_visible`, etc.) with:
    - `DisplayModeState`
    - `OverlayVisibilityState`
  - Reduces “forgot to update one bool” bugs and simplifies save/restore.
  - Shouldnt state management be in the FSM anyway?

- [ ] **Move duplicated axes parsing logic to one utility**
  - `_parse_signed_axes()` appears both nested in `__init__` and as a static method.
  - Keep one canonical implementation and reuse it.

---

## P1 (Medium effort): modularize boundaries

- [ ] **Split `ViserCallbackManager` into role-focused components**
  - Suggested extraction:
    - `MapStateLoader` (load/apply map data into state)
    - `PointCloudRenderer` (all point/bbox/label/caption draw/remove)
    - `FeatureQueryService` (CLIP + LLM query operations)
    - `AppRuntimeCoordinator` (threads, startup/shutdown wiring)
  - Keep `ViserCallbackManager` as thin façade for backward compatibility.

- [ ] **Extract video-serving concerns out of `vis_with_viser.py`**
  - Introduce `VideoServerService` for HTTP serving + path switching.
  - Keep GUI HTML rendering separate from server lifecycle.

- [ ] **Create a shared “capability evaluator” module**
  - Capability flags (`has_segment_objects`, `can_run_clip_query`, etc.) are recomputed in multiple places.
  - Add a single function/class that computes all capability flags from state.

- [ ] **Replace ad-hoc dicts for arc state with typed model**
  - Introduce dataclasses/pydantic-style typed structures for:
    - Arc state (`current_day`, `current_hour`, `arcs`)
    - Arc edge (`source`, `target`, `arc_type`, `weight`, `label`)
  - Improves validation and IDE support.

- [ ] **Reduce direct cross-controller mutation**
  - Current controllers mutate manager internals directly.
  - Add explicit manager methods/events for key transitions (mode change, map loaded, query result applied).

---

## P2 (Bigger refactor): box annotator decomposition

- [ ] **Split `box_annotator.py` into submodules**
  - Proposed structure:
    - `box_annotator/model.py` (`Box3D`, lightweight state)
    - `box_annotator/math3d.py` (quat/rot helpers, fitting utils)
    - `box_annotator/scene.py` (Viser handle creation/update/remove)
    - `box_annotator/io.py` (save/load JSON)
    - `box_annotator/controller.py` (interaction/state transitions)
    - `box_annotator/gui.py` (UI construction + callback wiring)

- [ ] **Introduce mode enum/state machine for annotator modes**
  - Replace multiple booleans (`_placing_mode`, `_selecting_mode`, `_corner_mode`, `_box_by_select_mode`) with a single enum + transition table.
  - Prevent conflicting active modes.

- [ ] **Separate high-frequency scene updates from full rebuild paths**
  - Preserve existing fast path (`_update_box_scene_fast`) and make rebuild path explicit/rare.
  - Document when each path should be used.

---

## P3 (Quality guardrails)

- [ ] **Add targeted unit tests around fragile logic**
  - `queries_and_chat._extract_object_ids()` (loose JSON parsing + fallback)
  - `arcs_gui._detect_arc_overlaps_all()` (bidirectional overlap offsets)
  - `gui_fsm._discover_saved_maps()` (root discovery + de-dup)
  - `cg_dataclass._load_clip_features()` (shape repair/padding/truncation)

- [ ] **Add integration smoke test for map switching**
  - Validate that switching maps preserves selected modes and visibility where applicable.
  - Validate controls re-enable/disable correctly based on capabilities.

- [ ] **Standardize user-facing error handling**
  - Use structured result objects (success/failure + message) instead of mixed return `None/False/str` patterns.
  - Keep notifications in one thin adapter.

- [ ] **Normalize logging style**
  - Replace mixed f-strings and `%s` formatting with one style (prefer parameterized `%s` in logger calls).

---

## Optional “quick wins” (can be done anytime)

- [ ] Rename GUI label `"segment floor"` to `"Segment Floor"` for consistency.
- [ ] Remove dead helper `_add_video_to_gui()` if no longer used.
- [ ] Add `vis/docs/ARCHITECTURE.md` with module boundaries and data flow diagram.

---

## Suggested rollout plan

1. **Week 1:** P0 items only (safe, behavior-preserving).
2. **Week 2:** P1 extraction of services behind existing APIs.
3. **Week 3+:** P2 annotator split in small PRs (`math3d` → `io` → `gui` → controller cleanup).
4. **Parallel:** P3 tests added as each refactor lands.

---

## Definition of done for this cleanup

- New contributors can locate responsibilities quickly (one concern per module).
- Map switch logic is readable and testable end-to-end.
- Box annotator is split into focused files with minimal cross-layer imports.
- Capability/state transitions are typed and centralized.
- Regression tests cover query parsing, arc overlap behavior, and map switching.
