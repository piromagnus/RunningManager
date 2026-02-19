# Services Layer

Domain services for planning, analytics, metrics, and external integrations.

## Files

| File | Purpose |
|------|---------|
| `planner_service.py` | Planning estimations (pace, distance, DistEq) |
| `metrics_service.py` | Full metrics pipeline (activities, daily, weekly) |
| `analytics_service.py` | Weekly/daily data loading, planned vs actual |
| `interval_utils.py` | Interval step normalization and serialization |
| `timeseries_service.py` | Activity timeseries loading + cached metrics |
| `strava_service.py` | Strava OAuth, sync, caching |
| `garmin_import_service.py` | Garmin import (stub) |
| `garmin_export_service.py` | TCX export for intervals |
| `dashboard_data_service.py` | Dashboard data preprocessing |
| `activity_feed_service.py` | Activity list building |
| `activity_detail_service.py` | Single activity detail loading |
| `lap_metrics_service.py` | Lap-level metrics extraction |
| `linking_service.py` | PlannedSession ↔ Activity linking |
| `templates_service.py` | Session template CRUD |
| `session_templates_service.py` | Template payload management |
| `speed_profile_service.py` | Speed profile computation |
| `pacer_service.py` | Race pacing calculations |
| `speed_profile/preprocessing.py` | GPS preprocessing helpers |
| `speed_profile/hr_speed_analysis.py` | HR vs speed analysis helpers |
| `speed_profile/minetti.py` | Minetti energy cost formulas |
| `speed_profile/profile_computation.py` | Speed profile computations |
| `speed_profile/persistence.py` | Speed profile CSV persistence |
| `pacer/segmentation.py` | Pacer segmentation + metrics helpers |
| `pacer/segment_merger.py` | Segment merging strategies |
| `pacer/preprocessing.py` | Pacer GPX preprocessing |
| `pacer/aid_station_stats.py` | Aid station stats helpers |
| `pacer/race_persistence.py` | Race pacing CSV persistence |
| `pacer/activity_comparison.py` | Planned vs actual comparison + linking |
| `pacer/__init__.py` | PacerService facade |
| `planner_presenter.py` | Week planning presentation layer |
| `serialization.py` | JSON serialization utilities |

## Key Service APIs

### PlannerService
- `derive_from_distance(athlete_id, km, ascent)`: Estimate duration from distance
- `derive_from_duration(athlete_id, sec, ascent)`: Estimate distance from duration
- `compute_interval_totals(athlete_id, steps_json)`: Total distance/duration for intervals
- `distance_eq(km, ascent)`: Distance-equivalent calculation

### MetricsComputationService
- `recompute_all(athlete_id=None)`: Full pipeline recomputation
- `recompute_athlete(athlete_id)`: Single athlete recomputation
- `recompute_for_activities(activity_ids)`: Incremental recomputation

Key metrics:
- `distanceEqKm = distanceKm + ascentM * distanceEqFactor`
- TRIMP: HR reserve weighting with exponential factor
- Categories: `RUN`, `TRAIL_RUN`, `HIKE`, `RIDE`, `BACKCOUNTRY_SKI`

### StravaService
- `authorization_url(state)`: OAuth initiation
- `exchange_code(code)`: Token exchange
- `sync_activities(athlete_id)`: Incremental sync
- `rebuild_cache(athlete_id)`: Full cache rebuild

### AnalyticsService
- `load_weekly_data(athlete_id, weeks)`: Weekly aggregates
- `load_daily_data(athlete_id, start, end)`: Daily range data

### SpeedProfileService
- `preprocess_timeseries(df)`: GPS-based preprocessing (distance, speed, grade, elevation)
- `compute_speed_eq_column(df)`: Add speed_eq_km_h using Minetti energy cost model
- `process_timeseries(activity_id, strategy)`: Full HR/speed analysis with clustering
- `save_metrics_ts(activity_id, result)`: Save HR analysis results (hr_smooth, hr_shifted, cluster)
- `compute_and_save_elevation_metrics(activity_id)`: Compute elevation metrics, preserves existing HR columns
- `compute_all_metrics_ts(activity_id)`: **Main entry point** - computes both HR analysis + elevation metrics
- `load_elevation_metrics(activity_id)`: Load cached elevation metrics from metrics_ts
- `get_or_compute_elevation_metrics(activity_id)`: Get cached metrics or compute and save

### TimeseriesService
- `load(activity_id)`: Load raw timeseries DataFrame
- `load_metrics_ts(activity_id)`: Load cached metrics_ts DataFrame
- `has_elevation_metrics(activity_id)`: Check if cached elevation metrics are available

## Session Types

| Type | Description |
|------|-------------|
| `FUNDAMENTAL_ENDURANCE` | Easy runs (pace/hr target) |
| `LONG_RUN` | Extended duration runs |
| `INTERVAL_SIMPLE` | Structured intervals (warmup, loops, cooldown) |
| `RACE` | Race sessions |

## Interval Steps Schema

Legacy format (deprecated):
```json
{"warmupSec": 600, "cooldownSec": 300, "repeats": [...]}
```

Current format:
```json
{
  "warmupSec": 600,
  "cooldownSec": 300,
  "betweenLoopRecoverSec": 60,
  "preBlocks": [...],
  "loops": [{"repeats": 5, "actions": [...]}],
  "postBlocks": [...]
}
```

## Related Files

- `persistence/repositories.py`: Data access
- `utils/time.py`: Date/time helpers
- `utils/coercion.py`: Type conversion
- `tests/test_*.py`: Service tests

## Invariants

- **TRIMP categories**: Only `RUN`, `TRAIL_RUN`, `HIKE`, `BACKCOUNTRY_SKI` for training load
- **Bike DistEq**: Special factors from settings (distance, ascent, descent)
- **Tokens**: Always encrypted with Fernet

## Maintaining This File

Update when:
- Adding new services
- Changing key API signatures
- Adding new session types
- Modifying metrics formulas
