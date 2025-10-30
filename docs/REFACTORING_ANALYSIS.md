# Codebase Refactoring Analysis

## Executive Summary

This document provides a comprehensive analysis of the Running Manager codebase in preparation for refactoring. The goal is to:

- Extract widgets and graph visualizations into dedicated folders
- Keep files under 500 lines
- Eliminate duplicate functions
- Identify unused code
- Improve code organization

## Files Over 500 Lines (Targets for Refactoring)

| File | Lines | Primary Purpose | Refactoring Priority |
|------|-------|-----------------|---------------------|
| `pages/Planner.py` | 1006 | Weekly planning UI, session forms | HIGH |
| `pages/Dashboard.py` | 906 | Training load charts, speed/HR visualizations | HIGH |
| `services/strava_service.py` | 867 | Strava OAuth and sync | MEDIUM |
| `pages/Activity.py` | 809 | Activity detail view, maps, elevation profiles | HIGH |
| `services/metrics_service.py` | 783 | Metrics computation pipeline | MEDIUM |
| `pages/SessionCreator.py` | 692 | Session template creation forms | HIGH |
| `pages/Analytics.py` | 571 | Planned vs actual analytics charts | HIGH |
| `services/speed_profile_service.py` | 561 | Speed profile processing | LOW |
| `services/analytics_service.py` | 539 | Analytics data loading/shaping | LOW |
| `pages/Activities.py` | 533 | Activity feed UI, linking dialogs | MEDIUM |

## Function Catalog by File

### pages/Planner.py (1006 lines)

**Functions:**

- `_format_week_duration()` - Format seconds to hours:minutes
- `_format_session_duration()` - Format seconds to duration string
- `_coerce_float()` - Coerce value to float with default
- `_coerce_int()` - Coerce value to int with default
- `_default_template_title()` - Generate default template title
- `_clean_optional()` - Clean optional values
- `_reset_planner_state()` - Reset planner session state
- `_apply_planner_prefill()` - Prefill form from template/session
- `_get_dialog_factory()` - Get Streamlit dialog factory
- `_should_prompt_template_save()` - Check if template save prompt needed
- `_render_template_prompt_if_needed()` - Render template save dialog
- `build_session_row()` - Build session row dict
- `get_sessions_df_cached()` - Cached sessions dataframe
- `get_threshold_names_cached()` - Cached threshold names
- `get_session_templates_cached()` - Cached session templates
- `_update_week_from_picker()` - Update week from date picker

**Purpose:** Weekly session planning interface with form handling, template management, and week view rendering.

**Refactoring Opportunities:**

- Extract form rendering functions → `widgets/session_forms.py`
- Extract card rendering → `widgets/session_cards.py`
- Extract helper functions (`_coerce_*`, `_format_*`) → `utils/helpers.py`

### pages/Dashboard.py (906 lines)

**Functions:**

- `_select_athlete()` - Athlete selector widget
- `_load_daily_metrics()` - Load and process daily metrics
- `_training_load_chart()` - Create training load Altair chart
- `_load_hr_speed_data()` - Load HR vs Speed data with clustering

**Purpose:** Dashboard with training load trends, speed-equivalent scatter plots, and HR vs Speed analysis.

**Refactoring Opportunities:**

- Extract `_training_load_chart()` → `graph/training_load.py`
- Extract speed scatter plot → `graph/speed_scatter.py`
- Extract HR vs Speed clustering → `graph/hr_speed.py`
- Extract athlete selector → `widgets/athlete_selector.py`

### services/strava_service.py (867 lines)

**Functions:** (Class methods)

- `authorization_url()` - Generate OAuth URL
- `exchange_code()` - Exchange auth code for tokens
- `preview_sync_last_n_days()` - Preview sync without downloading
- `sync_last_n_days()` - Sync activities from Strava
- `sync_last_14_days()` - Convenience wrapper
- `rebuild_from_cache()` - Rebuild from cached raw JSON
- `_ensure_access_token()` - Ensure valid access token
- `_refresh_token()` - Refresh OAuth token
- `_iter_recent_activities()` - Iterator over recent activities
- `_get_activity()` - Fetch activity detail
- `_get_streams()` - Fetch activity streams
- `_save_raw_activity()` - Save raw JSON
- `_save_timeseries()` - Save timeseries CSV
- `_map_activity_row()` - Map Strava API to activity row
- `_load_tokens()` - Load encrypted tokens
- `_save_tokens()` - Save encrypted tokens
- `_request_json()` - Generic API request handler
- `_log_api_call()` - Log API calls for rate limiting
- `_rate_status_from_headers()` - Parse rate limit headers
- `get_rate_status()` - Get current rate limit status
- `get_rate_log()` - Get recent API call log

**Purpose:** Strava OAuth integration, activity syncing, rate limit management.

**Refactoring Opportunities:**

- Extract token management → `services/oauth_strava.py` (if split needed)
- Rate limiting could stay as it's tightly coupled

### pages/Activity.py (809 lines)

**Functions:**

- `main()` - Main page entry point
- `_format_duration()` - Format seconds to duration string
- `_render_summary()` - Render activity summary metrics
- `_select_map_style()` - Map style selector widget
- `_render_link_panel()` - Render planned vs actual comparison
- `_format_comparison_value()` - Format comparison metric value
- `_format_comparison_delta()` - Format comparison delta
- `_render_timeseries()` - Render timeseries charts
- `_render_elevation_profile()` - Render elevation profile with grade
- `_get_grade_category()` - Map grade to category
- `_plot_interactive_elevation()` - Plot elevation with grade coloring
- `_plot_grade_histogram()` - Plot grade distribution histogram
- `_build_map_deck()` - Build pydeck map visualization
- `_render_map()` - Render map with selected style

**Purpose:** Activity detail page with metrics, maps, timeseries, and elevation profiles.

**Refactoring Opportunities:**

- Extract map rendering → `graph/maps.py`
- Extract elevation profile → `graph/elevation_profile.py`
- Extract timeseries charts → `graph/timeseries.py`
- Extract comparison panel → `widgets/comparison_panel.py`
- Extract map style selector → `widgets/map_selector.py`

### services/metrics_service.py (783 lines)

**Functions:** (Class methods)

- `recompute_all()` - Recompute all metrics
- `recompute_athlete()` - Recompute for one athlete
- `recompute_for_activities()` - Recompute for specific activities
- `_compute_activity_metrics()` - Compute per-activity metrics
- `_compute_planned_metrics()` - Compute planned session metrics
- `_build_weekly_metrics()` - Build weekly aggregates
- `_build_daily_metrics()` - Build daily aggregates with rolling windows
- `_compute_trimp()` - Compute TRIMP from HR
- `_compute_planned_trimp()` - Compute TRIMP for planned sessions
- `_resolve_activity_category()` - Categorize sport type
- `_normalize_category()` - Normalize category string
- `_bike_eq_factors()` - Get bike distance-equivalent factors
- `_compute_bike_distance_eq()` - Compute bike DistEq
- `_planned_segments()` - Extract segments from planned intervals
- `_avg_hr_for_target()` - Get average HR for target threshold

**Purpose:** Metrics computation pipeline for activities, planned sessions, weekly, and daily aggregates.

**Refactoring Opportunities:**

- Extract TRIMP computation → `services/trimp_service.py`
- Extract bike DistEq → `services/bike_metrics.py` (if grows)
- Weekly/daily aggregation could be split if needed

### pages/SessionCreator.py (692 lines)

**Functions:**

- `_coerce_float()` - Coerce to float (DUPLICATE)
- `_coerce_int()` - Coerce to int (DUPLICATE)
- `_format_duration()` - Format duration (DUPLICATE)
- `list_templates()` - List session templates (cached)
- `list_sessions()` - List recent sessions (cached)
- `_default_template_title()` - Default title generator (DUPLICATE)
- `_clean_notes()` - Clean notes field
- `_render_fundamental_form()` - Render fundamental endurance form
- `_render_long_run_form()` - Render long run form
- `_render_race_form()` - Render race form
- `_render_interval_form()` - Render interval form
- `_render_session_form()` - Main form router
- `_session_payload_for_save()` - Build session payload

**Purpose:** Session template creation interface with forms for different session types.

**Refactoring Opportunities:**

- Extract form rendering functions → `widgets/session_forms.py` (share with Planner)
- Extract helper functions → `utils/helpers.py`

### pages/Analytics.py (571 lines)

**Functions:**

- `_select_athlete()` - Athlete selector (DUPLICATE)
- `_load_saved_activity_types()` - Load activity type filters from settings
- `_to_date_series()` - Convert series to date (helper)

**Purpose:** Planned vs actual analytics with weekly/daily comparisons and category filters.

**Refactoring Opportunities:**

- Extract chart creation → `graph/analytics_charts.py`
- Extract athlete selector → `widgets/athlete_selector.py` (shared)

### services/speed_profile_service.py (561 lines)

**Class:** `SpeedProfileService`
**Purpose:** Process timeseries to compute speed profiles, HR-speed relationships, clustering.

**Refactoring Opportunities:**

- Could split clustering logic if it grows
- Generally well-contained

### services/analytics_service.py (539 lines)

**Class:** `AnalyticsService`
**Purpose:** Load and shape analytics data (daily/weekly ranges, planned vs actual).

**Refactoring Opportunities:**

- Data loading is cohesive, could stay as-is

### pages/Activities.py (533 lines)

**Functions:**

- `_format_session_type_label()` - Format session type
- `_escape_text()` - HTML escape text
- `_trigger_rerun()` - Trigger Streamlit rerun
- `_format_duration()` - Format duration (DUPLICATE)
- `_format_datetime()` - Format datetime
- `_format_sport_type()` - Format sport type
- `_planned_card_status()` - Get planned session status text
- `_dialog_factory()` - Get dialog factory (DUPLICATE)
- `main()` - Main page entry
- `_render_planned_strip()` - Render unlinked planned sessions strip
- `_open_link_dialog()` - Open linking dialog
- `_format_candidate_label()` - Format link candidate label
- `_render_activity_card()` - Render activity feed card

**Purpose:** Activity feed page with linking functionality.

**Refactoring Opportunities:**

- Extract activity card → `widgets/activity_cards.py`
- Extract planned strip → `widgets/planned_strip.py`
- Extract linking dialog → `widgets/linking_dialog.py`

## Duplicate Functions Across Files

### Type Coercion Functions

**Duplicated in:**

- `pages/Planner.py`: `_coerce_float()`, `_coerce_int()`
- `pages/SessionCreator.py`: `_coerce_float()`, `_coerce_int()`
- `pages/Settings.py`: `_coerce_float()`, `_coerce_int()`
- `services/metrics_service.py`: `_safe_float()`, `_safe_int()` (similar)
- `services/lap_metrics_service.py`: `_safe_float()`, `_safe_int()`
- `services/activity_detail_service.py`: `_coerce_float()`, `_coerce_int()`
- `services/activity_feed_service.py`: `_coerce_float()`, `_coerce_int()`
- `services/linking_service.py`: `_coerce_float()`, `_coerce_int()`

**Action:** Consolidate → `utils/coercion.py`

### Duration Formatting

**Duplicated in:**

- `pages/Planner.py`: `_format_week_duration()`, `_format_session_duration()`
- `pages/SessionCreator.py`: `_format_duration()`
- `pages/Activity.py`: `_format_duration()`
- `pages/Activities.py`: `_format_duration()`
- `services/interval_utils.py`: `format_duration_label()`

**Action:** Consolidate → `utils/formatting.py` (extend existing)

### Template Title Generation

**Duplicated in:**

- `pages/Planner.py`: `_default_template_title()`
- `pages/SessionCreator.py`: `_default_template_title()`

**Action:** Consolidate → `utils/helpers.py` or `services/templates_service.py`

### Dialog Factory

**Duplicated in:**

- `pages/Planner.py`: `_get_dialog_factory()`
- `pages/Activities.py`: `_dialog_factory()`

**Action:** Consolidate → `utils/ui_helpers.py`

### Athlete Selector

**Duplicated in:**

- `pages/Dashboard.py`: `_select_athlete()`
- `pages/Analytics.py`: `_select_athlete()`

**Action:** Extract → `widgets/athlete_selector.py`

## Widgets to Extract

### widgets/session_forms.py

- `_render_fundamental_form()` (from SessionCreator)
- `_render_long_run_form()` (from SessionCreator)
- `_render_race_form()` (from SessionCreator)
- `_render_interval_form()` (from SessionCreator)
- Form handling logic from Planner

### widgets/session_cards.py

- Card rendering from Planner week view
- Card model building (from `planner_presenter.py`)

### widgets/activity_cards.py

- `_render_activity_card()` (from Activities)
- Activity feed card rendering

### widgets/planned_strip.py

- `_render_planned_strip()` (from Activities)
- Unlinked sessions strip

### widgets/linking_dialog.py

- `_open_link_dialog()` (from Activities)
- `_format_candidate_label()` (from Activities)
- Link candidate selection UI

### widgets/comparison_panel.py

- `_render_link_panel()` (from Activity)
- `_format_comparison_value()` (from Activity)
- `_format_comparison_delta()` (from Activity)

### widgets/map_selector.py

- `_select_map_style()` (from Activity)
- Map style selection logic

### widgets/athlete_selector.py

- `_select_athlete()` (from Dashboard, Analytics)
- Shared athlete selection widget

## Graphs to Extract

### graph/training_load.py

- `_training_load_chart()` (from Dashboard)
- Training load time series charts
- Acute/chronic band visualization

### graph/speed_scatter.py

- Speed-equivalent scatter plot (from Dashboard)
- Duration vs SpeedEq with HR coloring

### graph/hr_speed.py

- `_load_hr_speed_data()` (from Dashboard)
- HR vs Speed clustering visualization
- Regression line and error bars

### graph/elevation_profile.py

- `_render_elevation_profile()` (from Activity)
- `_plot_interactive_elevation()` (from Activity)
- `_plot_grade_histogram()` (from Activity)
- `_get_grade_category()` (from Activity)

### graph/timeseries.py

- `_render_timeseries()` (from Activity)
- HR, pace, elevation time series charts

### graph/maps.py

- `_build_map_deck()` (from Activity)
- `_render_map()` (from Activity)
- Pydeck map visualization

### graph/analytics_charts.py

- Planned vs actual charts (from Analytics)
- Weekly/daily comparison visualizations
- Stacked bar charts, line charts

## Utils Extensions

### utils/coercion.py (NEW)

- `coerce_float()` - Unified float coercion
- `coerce_int()` - Unified int coercion
- `safe_float()` - Safe float with error handling
- `safe_int()` - Safe int with error handling

### utils/formatting.py (EXTEND)

- `format_duration()` - Unified duration formatting
- `format_week_duration()` - Week-level duration
- Keep existing: `fmt_km()`, `fmt_m()`, `fmt_decimal()`, etc.

### utils/ui_helpers.py (NEW)

- `get_dialog_factory()` - Streamlit dialog factory
- `trigger_rerun()` - Trigger Streamlit rerun helper
- Other UI utility functions

### utils/helpers.py (NEW)

- `default_template_title()` - Template title generation
- `clean_optional()` - Clean optional values
- `clean_notes()` - Clean notes field
- Other generic helpers

## Proposed Folder Structure

```bash
RunningManager/
├── widgets/
│   ├── __init__.py
│   ├── athlete_selector.py      # Shared athlete selector
│   ├── session_forms.py          # Session form rendering
│   ├── session_cards.py          # Session card widgets
│   ├── activity_cards.py         # Activity feed cards
│   ├── planned_strip.py          # Unlinked sessions strip
│   ├── linking_dialog.py         # Activity linking UI
│   ├── comparison_panel.py       # Planned vs actual comparison
│   └── map_selector.py            # Map style selector
├── graph/
│   ├── __init__.py
│   ├── training_load.py          # Training load charts
│   ├── speed_scatter.py          # Speed-equivalent scatter
│   ├── hr_speed.py               # HR vs Speed analysis
│   ├── elevation_profile.py       # Elevation profiles
│   ├── timeseries.py             # Timeseries visualizations
│   ├── maps.py                   # Map visualizations
│   └── analytics_charts.py       # Analytics charts
├── utils/
│   ├── coercion.py               # NEW: Type coercion utilities
│   ├── ui_helpers.py             # NEW: UI helper functions
│   ├── helpers.py                # NEW: Generic helpers
│   ├── formatting.py             # EXTEND: Duration formatting
│   ├── config.py                 # Existing
│   ├── crypto.py                 # Existing
│   ├── ids.py                    # Existing
│   ├── time.py                   # Existing
│   ├── styling.py                # Existing
│   └── auth_state.py             # Existing
├── pages/                        # Streamlit pages (simplified)
├── services/                     # Business logic (mostly unchanged)
└── persistence/                  # Data layer (unchanged)
```

## Files to Tag for Removal/Review

### Potentially Unused

- `config.py` (root) - Check if used vs `utils/config.py`
- `serialization.py` in services - Verify usage

### Candidates for Consolidation

- Multiple `_coerce_*` implementations → consolidate
- Multiple `_format_duration` implementations → consolidate
- Duplicate athlete selectors → consolidate

## Refactoring Plan

### Phase 1: Extract Utilities

1. Create `utils/coercion.py` - Consolidate all coercion functions
2. Extend `utils/formatting.py` - Add duration formatting
3. Create `utils/ui_helpers.py` - Extract UI utilities
4. Create `utils/helpers.py` - Extract generic helpers
5. Update all imports across codebase

### Phase 2: Extract Widgets

1. Create `widgets/athlete_selector.py` - Shared athlete selector
2. Create `widgets/session_forms.py` - Extract session forms
3. Create `widgets/session_cards.py` - Extract session cards
4. Create `widgets/activity_cards.py` - Extract activity cards
5. Create `widgets/planned_strip.py` - Extract planned strip
6. Create `widgets/linking_dialog.py` - Extract linking dialog
7. Create `widgets/comparison_panel.py` - Extract comparison panel
8. Create `widgets/map_selector.py` - Extract map selector

### Phase 3: Extract Graphs

1. Create `graph/training_load.py` - Extract training load charts
2. Create `graph/speed_scatter.py` - Extract speed scatter plot
3. Create `graph/hr_speed.py` - Extract HR vs Speed analysis
4. Create `graph/elevation_profile.py` - Extract elevation profiles
5. Create `graph/timeseries.py` - Extract timeseries charts
6. Create `graph/maps.py` - Extract map visualizations
7. Create `graph/analytics_charts.py` - Extract analytics charts

### Phase 4: Refactor Pages

1. Refactor `pages/Planner.py` - Use extracted widgets/utils
2. Refactor `pages/Dashboard.py` - Use extracted graphs/widgets
3. Refactor `pages/Activity.py` - Use extracted graphs/widgets
4. Refactor `pages/SessionCreator.py` - Use extracted widgets/utils
5. Refactor `pages/Analytics.py` - Use extracted graphs/widgets
6. Refactor `pages/Activities.py` - Use extracted widgets

### Phase 5: Cleanup

1. Remove duplicate functions
2. Verify all imports
3. Run tests
4. Check linting (ruff)
5. Verify file sizes < 500 lines

## Testing Strategy

1. Run existing test suite after each phase
2. Test each extracted module independently
3. Integration tests for widget/graph usage
4. Visual regression testing for UI components

## Risk Assessment

- **Low Risk:** Utility extraction (well-isolated)
- **Medium Risk:** Widget extraction (UI dependencies)
- **Medium Risk:** Graph extraction (data dependencies)
- **High Risk:** Page refactoring (many dependencies)

## Success Criteria

- All files under 500 lines
- No duplicate functions
- Ruff linting passes
- All tests pass
- No functional regressions
- Improved code organization
