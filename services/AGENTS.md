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
| `trail_performance_model.py` | Trail digital-twin notebook helpers |
| `strava_service.py` | Strava OAuth, sync, caching, merge enrichment |
| `strava_archive_service.py` | Strava GDPR ZIP archive import (same IDs/storage as sync) |
| `garmin_import_service.py` | Garmin import (stub) |
| `garmin_export_service.py` | TCX export for intervals |
| `dashboard_data_service.py` | Dashboard data preprocessing |
| `trail_digital_twin_pipeline.py` | Configurable trail digital-twin fitting/report pipeline |
| `trail_digital_twin_benchmark.py` | Benchmark sweep expansion, leaderboard extraction, and HTML reporting |
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
| `pacer/activity_comparison.py` | Planned vs actual comparison, cache, linking |
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
- `recompute_planned_incremental(athlete_id, affected_date)`: Planned + weekly incremental rebuild
- `_ensure_dependencies(activity_ids)`: Ensure `metrics_ts`, speed profile, and lap metrics exist
- `_ensure_hr_zones(activity_ids)`: Backfill missing HR zone borders from earliest impacted date

Key metrics:
- `distanceEqKm = distanceKm + ascentM * distanceEqFactor`
- TRIMP: HR reserve weighting with exponential factor
- Categories: `RUN`, `TRAIL_RUN`, `HIKE`, `RIDE`, `BACKCOUNTRY_SKI`

### StravaService
- `authorization_url(state)`: OAuth initiation
- `exchange_code(athlete_id, code)`: Token exchange
- `sync_last_n_days(athlete_id, days)`: Incremental sync; incomplete cache (missing timeseries / empty laps / empty polyline) is **merged** from API into the same `activityId`
- `rebuild_from_cache(athlete_id)`: Cache rebuild with single metrics pass
- `merge_and_save_raw` / `save_timeseries_dataframe` / `upsert_activity_row_from_detail`: shared fill-empty writers used by archive import
- Sync/rebuild ensure `metrics_ts` + speed profile + lap metrics + HR zones

### StravaArchiveService
- `import_strava_archive(zip_source, athlete_id, progress_callback=None)`: Import official Strava GDPR ZIP (`activities.csv` + `activities/*.{fit,gpx,tcx}[.gz]`)
- Uses Strava **Activity ID** as `activityId` (same as API sync → no duplicates)
- Fills missing artifacts only; re-import of complete data → `already_complete`
- Writes `raw/strava/{id}.json`, `timeseries/{id}.csv`, `activities.csv`; then `_apply_sync_metrics`

### HrZonesService
- `backfill_all_borders(athlete_id=None)`: Full zone-border rebuild
- `backfill_borders_from_date(athlete_id, from_date, to_date=None)`: Incremental backfill window
- `get_or_compute_zones(activity_id)`: Loads zones or lazily backfills missing summaries

### AnalyticsService
- `load_weekly_data(athlete_id, weeks)`: Weekly aggregates
- `load_daily_data(athlete_id, start, end)`: Daily range data
- `activity_category_breakdown(...)`: Actual metric totals by activity category

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

### Trail Performance Model
- `prepare_raw_timeseries_for_segments(...)`: Fast raw GPS/HR/elevation preparation for notebook segment aggregation
- `segment_timeseries(df)`: Aggregate processed activity streams into 1 km course segments; includes distance-weighted integrated GAP, mixed climb/descent diagnostics, `stationaryTimeShare`, and `actualMovingTimeSec`
- `apply_segment_exclusion(...)`: Flag immobile segments that are flat on the altitude–time profile (`absAltitudeRateMph`) via `isFitEligible` + `exclusionReason`; requires low altitude rate AND (low `meanSpeedEqKmh` OR high `stationaryTimeShare`) so climbing/descending time profiles stay in the fit set
- `grid_search_model(...)`: Fit paper-style `alpha` and fatigue/pacing-decay parameter
- `leave_one_out_grid_search(...)`: Race/activity-level LOO validation for notebook experiments
- `top_hrr_hard_trailrun_ids(...)`: Build top hard TrailRun subset by average HR reserve
- `select_best_activity_by_dates(...)`: Build configurable selected-date race subset
- `compute_redi_load_features(...)`: REDI slow/fast/balance load features
- `attach_previous_daily_features(...)`: Attach strictly previous-day load/readiness features
- `add_in_activity_trimp_features(...)`: Segment TRIMP plus cumulative/decayed acute load
- `route_segments_from_points(...)`: Convert GPX-like route points to route-only segments; ignores GPX timestamps because planned-route exports may contain synthetic timing
- `estimate_hrr_duration_envelope(...)`: Empirical max-duration table by average HRR band
- `estimate_hrr_duration_power_law(...)`: Monotone representative-window fit for sustainable HRR over duration; supports performance-weighted frontier fitting
- `max_duration_for_hrr_power_law(...)`: Invert fitted HRR-duration power law for pre-race feasibility checks
- `simulate_constant_hrr_route(...)`: Pre-race route prediction with constant HRR and cumulative predicted acute TRIMP fatigue by default
- `simulate_observed_hrr_segments(...)`: Completed-activity segment prediction using observed segment HRR without actual-time TRIMP leakage; cumulative fatigue by default
- `sweep_constant_hrr_route(...)`: Constant-HRR candidate sweep with endurance-envelope feasibility and configurable fatigue input column
- `select_best_constant_hrr(...)`: Fastest feasible HRR choice from a sweep
- `segment_grid_search_model(...)`: Extension-only segment-level grid search with race-summed metrics
- `predict_hrr_trimp_segment_times(...)`: Constrained HRR-linear plus raw acute-load fatigue speed equation; supports decayed TRIMP, cumulative TRIMP, progress, or decayed-plus-secondary fatigue input, and treats `trimp_scale` as deprecated/ignored
- `hrr_trimp_grid_search_model(..., fit_mask_col=None)`: Small-grid constrained HRR-TRIMP calibration; optional fit mask optimizes on cleaned segments while still predicting full race
- `leave_one_out_hrr_trimp_grid_search(..., fit_mask_col=None)`: All-activity LOO; fit on eligible segments, score full race (plus fit-eligible diagnostics)
- `forbidden_anonymized_columns(...)`: Guard direct identifiers from paper feature exports
- `fit_linear_regression(...)`: Lightweight HR and segment model fitting for notebook analysis

### Trail Digital Twin Pipeline
- `load_config(path)`: Read and validate the YAML extension pipeline config
- `run_pipeline(config, project_root=None, config_path=None)`: Build features, fit configured model variants, and return tables; `execution.jobs` parallelizes independent cohort/objective Stage 0-3 fits
- `write_outputs(result, output_dir)`: Write CSV assets, manifest, and self-contained HTML report
- `table_hrr_trimp_grid_search`: Exported Stage 1-3 HRR/TRIMP alpha-kappa grid-search cells with physiology bounds and MAE metrics
- `table_segment_type_metrics`: Stage 3 segment-level MAE, bias, MAPE, R2, and counts by terrain family
- `segment_exclusion` config: optional altitude–time flat immobile fit mask (`isFitEligible`); keys `min_mean_speed_eq_kmh`, `max_stationary_time_share`, `max_abs_altitude_rate_mph`; full-race LOO evaluation retained
- Default config path: `configs/trail_digital_twin_extensions.yaml`
- Segment-exclusion A/B: `configs/trail_digital_twin_benchmark_segment_exclusion.yaml`

### Trail Digital Twin Benchmark
- `load_benchmark_config(path)`: Read and validate grouped benchmark sweep YAML
- `expand_benchmark_runs(base_config, benchmark_config, output_dir, max_runs=None)`: Materialize per-run pipeline configs
- `summarize_benchmark_result(run, result, selection)`: Extract Stage 3 benchmark metrics from a pipeline result
- `execute_benchmark_run(run, selection, project_root, config_path)`: Run one isolated benchmark experiment for sequential or parallel scheduling
- `combine_benchmark_tables(...)`: Build aggregate run, leaderboard, manifest, and plan tables
- `write_benchmark_outputs(tables, output_dir, metadata)`: Write aggregate CSV assets and self-contained HTML report
- `read_benchmark_output_tables(output_dir)`: Reload aggregate benchmark CSV assets for report-only rebuilds
- `write_benchmark_html(tables, output_dir, metadata)`: Rebuild only the self-contained benchmark HTML report
- `benchmark_hrr_trimp_grid_search`: Aggregate per-run HRR/TRIMP alpha, fatigue, and secondary-fatigue grid cells
- `benchmark_segment_type_metrics`: Aggregate per-run segment-type metrics when pipeline runs export them
- `benchmark_activity_error_strata`: Aggregate Stage 3 LOO error cuts by duration, in-race TRIMP, HRR-frequency, and terrain family
- `benchmark_bootstrap_uncertainty`: Deterministic bootstrap confidence intervals for Stage 3 LOO MAE and bias

### PacerService
- `save_race(...)`: Persists race and invalidates comparison cache for this race
- `compare_race_segments_with_activity(race_id, activity_id, timeseries_df)`: Uses on-disk cache
- Link/unlink race-to-activity invalidates cached comparisons for that activity

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
