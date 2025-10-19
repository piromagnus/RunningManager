Running Manager — Data Sync and Metrics Pipeline

Overview
- Trigger: Settings page button “Synchroniser les N derniers jours”.
- Source: Strava (OAuth2) for activity list, detail and streams (time series).
- Storage: CSVs under `data/` with file locking; raw JSON cache in `data/raw/strava/`; per-activity time series in `data/timeseries/`.
- Locale: Storage uses `.` for decimals; UI formatting is fr-FR only.

Key Components
- `services/strava_service.py`: OAuth, manual sync, raw/time-series persistence, and post-sync metrics recompute.
- `services/metrics_service.py`: Computes `activities_metrics.csv`, `daily_metrics.csv`, `weekly_metrics.csv`.
- `persistence/`: CSV abstraction (`CsvStorage`) and repositories (headers and CRUD).
- `pages/Settings.py`: UI to connect Strava and trigger manual sync (uses `stravaSyncDays`).

Current Pipeline (before update)
1) On click, UI reads `stravaSyncDays` from `settings.csv` and calls `StravaService.sync_last_n_days(athlete_id, days)`.
2) The service lists recent activities from Strava after the cutoff date.
3) For activities not found in `activities.csv` and not cached in `data/raw/strava/`, it fetches detail and streams, persists raw JSON + time series, creates a row in `activities.csv`.
4) If any new activity was imported from the API, it recomputes metrics via `MetricsComputationService.recompute_for_activities()` (which updates activity, daily and weekly metrics for the concerned athlete).
5) A separate UI action exists to “Rebuild from cache”, which rebuilds `activities.csv` from `data/raw/strava/` only.

Updated Pipeline (this change)
Goal: Ensure the last N days are fully represented locally and metrics are up to date, while avoiding unnecessary network calls.

1) Determine window
   - Read `N = stravaSyncDays` from `settings.csv` (UI already provides a default).
   - Compute `after_ts = now - N days`.

2) List activities (API)
   - Call Strava `GET /athlete/activities?after=...` to enumerate candidate activity IDs in the window.

3) Ensure raw cache and time series
   - For each activity ID:
     - If `data/raw/strava/{id}.json` does not exist, fetch activity detail and streams, then persist:
       - Save raw JSON to `data/raw/strava/{id}.json`.
       - Build time series CSV in `data/timeseries/{id}.csv` from returned streams.
     - If raw JSON already exists, do not re-fetch (respect cache); time series is not re-fetched on cache hits.

4) Ensure tabular activities
   - For each activity ID in the window, ensure a row exists in `activities.csv`:
     - If missing, build it from the available detail (raw cache or freshly fetched) and persist.
     - Use repository header order from `persistence/repositories.py`.

5) Compute metrics
   - If any new `activities.csv` rows were created (either from API or cache), run `MetricsComputationService.recompute_for_activities(ids)`.
   - This recomputes and merges:
     - `activities_metrics.csv` (per-activity)
     - `daily_metrics.csv` (only affected athlete’s days are overwritten)
     - `weekly_metrics.csv` (only affected athlete’s weeks are overwritten)

6) Results and UX
   - The sync call returns the list of IDs newly downloaded from the API (cache misses). Activities created from cache are not counted in this list to keep backward compatibility with existing UI/tests.

Invariants Kept
- CSV writes keep `.` decimal separator; file locking via `CsvStorage`.
- No logging of raw secrets; tokens encrypted at rest using Fernet.
- Headers and column order kept per repositories with idempotent rewrites.

Primary Code Paths
- UI trigger: `pages/Settings.py:249` → `StravaService.sync_last_n_days`.
- Sync implementation: `services/strava_service.py:81` (`sync_last_n_days`).
- Metrics pipeline: `services/metrics_service.py` → recompute per athlete (activity → daily/weekly).

Notes
- “Rebuild from cache” remains available to fully reconstruct `activities.csv` solely from raw JSON if needed.
- The updated sync populates missing `activities.csv` rows for cached activities within the target window and updates metrics accordingly, without extra network calls.

UI Improvement
- Settings now displays a combined message after sync: number downloaded from Strava and number created from cache within the window, plus a short “Dernier résumé de synchronisation” with counts and a compact list of IDs.
