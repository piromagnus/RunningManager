# Planned Reapplied Changes

1. **Interval utilities module**
   - Add `services/interval_utils.py` providing normalize/serialize helpers, action descriptions, and cloning utilities for interval steps.
   - Introduce accompanying unit tests in `tests/test_interval_utils.py`.
   - Add `ui/__init__.py` placeholder for new UI helpers package.

2. **Shared interval editor component**
   - Add `ui/interval_editor.py` with Streamlit widgets for loop-based interval editing (add/remove loops/actions, pre/post blocks, between-block).
   - Use new editor in planner and session template creator.

3. **Planner service enhancements**
   - Update `services/planner_service.py` to rely on `normalize_steps`, support new structure for distance/duration/ascent estimation, and compute weekly distance-equivalent totals.
   - Update planner-related tests (`tests/test_planner_service.py`) accordingly.

4. **Metrics service adjustments**
   - Modify `services/metrics_service.py` to use normalized steps when building heart-rate segments.

5. **Planner presenter updates**
   - Extend `services/planner_presenter.py` to build compact card sections (pre/loops/between/post) using `describe_action`.
   - Update tests (`tests/test_planner_presenter.py`) to match new meta/sections.

6. **Planner UI changes**
   - Refactor `pages/Planner.py` to:
     * import and use `render_interval_editor`.
     * support the new **RACE** session type with distance/ascent/target time inputs.
     * show richer interval cards with sections and updated CSS.
     * include distance-equivalent in weekly totals caption.

7. **Session template creator**
   - Update `pages/SessionCreator.py` to use shared interval editor, add Race form, and align CSS with planner.

8. **Session detail page**
   - Update `pages/Session.py` to display interval breakdown using normalized steps and show aggregate duration/distance/ascent captions.

9. **Planned sessions strip styling**
   - Revise `pages/Activities.py` to apply status-based styling (future blue, today green, earlier-this-week orange, older red) to unlinked planned session cards/buttons while leaving activity feed cards unchanged.

10. **New planner infrastructure**
    - Ensure newly introduced helpers/tests (`services/interval_utils.py`, `ui/interval_editor.py`, `tests/test_interval_utils.py`) are imported and referenced where required.
