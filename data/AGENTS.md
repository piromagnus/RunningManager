# Data Directory

CSV-backed persistence layer for Running Manager. All data uses `.` as decimal separator.

## Directory Structure

| Path | Description |
|------|-------------|
| `activities.csv` | Imported activities from Strava/Garmin |
| `activities_metrics.csv` | Computed metrics per activity (DistEq, TRIMP) |
| `planned_sessions.csv` | Coach-created training sessions |
| `planned_metrics.csv` | Computed metrics per planned session |
| `links.csv` | Activity ↔ PlannedSession links with RPE |
| `daily_metrics.csv` | Per-day aggregates + acute/chronic windows |
| `weekly_metrics.csv` | Per-week aggregates + adherence |
| `athlete.csv` | Athlete profiles (hrRest, hrMax) |
| `settings.csv` | Coach settings (distanceEqFactor, bike factors) |
| `thresholds.csv` | HR/pace zones per athlete |
| `session_templates.csv` | Saved session templates (JSON payloads) |
| `goals.csv` | Race/goal targets |
| `races.csv` | Race definitions for pacing |
| `tokens.csv` | Encrypted OAuth tokens |

## Subdirectories

| Path | Description |
|------|-------------|
| `timeseries/{activityId}.csv` | Per-activity streams (hr, pace, elevation, lat/lon) |
| `metrics_ts/{activityId}.csv` | Preprocessed timeseries: HR analysis (hr_smooth, hr_shifted, speed_smooth, cluster) + elevation (speedeq_smooth, grade_ma_10, elevationM_ma_5, cumulated_distance) |
| `speed_profil/{activityId}.csv` | Speed profile data per activity |
| `raw/strava/{activityId}.json` | Cached Strava API responses |
| `race_pacing/` | Race pacing segment data |
| `exp_perf_predictions/` | Trail digital twin benchmark experiment outputs |

### exp_perf_predictions/

- Aggregate CSVs and HTML reports are versioned (leaderboard, stage metrics, fitted parameters, etc.)
- `benchmark_hrr_trimp_grid_search.csv` is gitignored (hundreds of MB; recomputable from benchmark reruns)
- Synthesis report: `docs/science/trail_digital_twin_experiment_synthesis_report.md`
- Session review: `docs/science/trail_digital_twin_session_benchmark_review.md`
- Session CSV: `exp_perf_predictions/session_benchmark_review/`
- Regenerate synthesis: `uv run python scripts/synthesize_trail_digital_twin_experiments.py`
- Regenerate session review: `uv run python scripts/synthesize_trail_digital_twin_sessions.py`

## Key Schema References

### activities.csv
- ID: `activityId`
- Fields: `athleteId`, `source`, `sportType`, `name`, `startTime`, `distanceKm`, `elapsedSec`, `movingSec`, `ascentM`, `avgHr`, `maxHr`, `hasTimeseries`, `polyline`, `rawJsonPath`, `raceId`

### planned_sessions.csv
- ID: `plannedSessionId`
- Fields: `athleteId`, `date`, `type`, `plannedDistanceKm`, `plannedDurationSec`, `plannedAscentM`, `targetType`, `targetLabel`, `notes`, `templateTitle`, `raceName`, `stepEndMode`, `stepsJson`
- Session types: `FUNDAMENTAL_ENDURANCE`, `LONG_RUN`, `INTERVAL_SIMPLE`, `RACE`

### links.csv
- ID: `linkId`
- Links `plannedSessionId` ↔ `activityId`
- Stores `rpe(1-10)` and `comments`

### daily_metrics.csv
- ID: `dailyId`
- Rolling windows: `acuteDistanceKm`, `chronicDistanceKm`, `acuteTrimp`, `chronicTrimp`

## Invariants

- **Decimal separator**: Always `.` (never locale comma)
- **Headers**: Defined in `persistence/repositories.py`
- **Locking**: Use `CsvStorage` for all I/O (portalocker)
- **Migrations**: Follow `_migrate_headers_if_needed` pattern when adding columns

## Related Files

- `persistence/repositories.py`: Header definitions and CRUD
- `persistence/csv_storage.py`: Thread-safe CSV I/O
- `services/metrics_service.py`: Metrics recomputation pipeline

## Maintaining This File

Update when:
- Adding new CSV tables
- Adding columns to existing tables
- Creating new subdirectories
- Changing ID columns or key relationships
