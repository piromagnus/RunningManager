# Widgets

Reusable Streamlit UI components for forms, selectors, and panels.

## Files

| File | Purpose |
|------|---------|
| `athlete_selector.py` | Athlete dropdown selection |
| `session_forms.py` | Session type form renderers |
| `template_selector.py` | Template selection UI |
| `template_actions.py` | Save/delete/schedule template actions |
| `session_importer.py` | Import planned sessions as templates |
| `comparison_panel.py` | Activity comparison display |
| `comparison_table.py` | Detailed segment comparison table |
| `week_view.py` | Planner week view renderer |

## session_forms.py

Form renderers for session types:

| Function | Session Type |
|----------|--------------|
| `render_fundamental_form()` | FUNDAMENTAL_ENDURANCE |
| `render_long_run_form()` | LONG_RUN |
| `render_race_form()` | RACE |
| `render_interval_form()` | INTERVAL_SIMPLE |

Common signature:
```python
def render_*_form(
    athlete_id: Optional[str],
    payload: Dict[str, Any],
    planner: PlannerService,
) -> Tuple[Dict[str, Any], Optional[float]]:
    # Returns (form_result, distance_eq_preview)
```

## template_selector.py

- `render_template_selector(athlete_id)`: Template dropdown with preview
- Manages template state and selection

## template_actions.py

- `render_save_template_action()`: Save current form as template
- `render_delete_template_action()`: Delete selected template
- `render_schedule_template_action()`: Schedule template to date

## comparison_panel.py

- `render_comparison_panel(planned, actual)`: Side-by-side metrics
- Shows planned vs actual distance, duration, ascent

## comparison_table.py

- `render_comparison_table(comparison_df, planned_segments_df, pacer_service)`: Segment detail table

## week_view.py

- `render_week_view(...)`: Planner week cards and totals

## Widget Patterns

State keys use widget-specific prefixes:
- `creator-*`: Session creator widgets
- `template-*`: Template management widgets

## Related Files

- `pages/Planner.py`: Uses session forms
- `pages/SessionCreator.py`: Uses all form widgets
- `pages/Activity.py`: Uses comparison panel
- `services/planner_service.py`: Planning calculations
- `ui/interval_editor.py`: Interval step editing

## Maintaining This File

Update when:
- Adding new form widgets
- Adding new selector components
- Changing widget API signatures
- Adding new session type forms
