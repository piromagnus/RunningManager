# Data Flow Reference

Full data flow documentation for Running Manager.
Traces how data is gathered, computed, stored, and displayed for every user-facing action.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [CSV Data Model](#2-csv-data-model)
3. [Data Ingestion: Strava Sync](#3-data-ingestion-strava-sync)
4. [Data Ingestion: Rebuild from Cache](#4-data-ingestion-rebuild-from-cache)
5. [Metrics Computation Pipeline](#5-metrics-computation-pipeline)
6. [HR Zones Recomputation](#6-hr-zones-recomputation)
7. [Speed Profile & Elevation Metrics](#7-speed-profile--elevation-metrics)
8. [Planner: Session CRUD](#8-planner-session-crud)
9. [Activity Linking](#9-activity-linking)
10. [Race Pacer](#10-race-pacer)
11. [Dashboard (Read-Only)](#11-dashboard-read-only)
12. [Analytics (Read-Only)](#12-analytics-read-only)
13. [Activity Detail Page (Detail + Actions)](#13-activity-detail-page-detail--actions)
14. [Settings: Save Settings](#14-settings-save-settings)
15. [CSV Modification Matrix](#15-csv-modification-matrix)

---

## 1. Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│  Streamlit Pages (UI)                                            │
│  Dashboard │ Planner │ Activities │ Activity │ Analytics │ ...    │
└──────┬──────┬────────┬───────────┬──────────┬───────────┬────────┘
       │      │        │           │          │           │
┌──────▼──────▼────────▼───────────▼──────────▼───────────▼────────┐
│  Services Layer                                                   │
│  MetricsService │ StravaService │ AnalyticsService │ PacerService │
│  SpeedProfileService │ HrZonesService │ LinkingService │ ...      │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  Persistence Layer                                                │
│  CsvStorage (portalocker) + Typed Repositories (BaseRepo)         │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│  data/                                                            │
│  CSV tables │ timeseries/ │ metrics_ts/ │ speed_profil/ │ raw/    │
└──────────────────────────────────────────────────────────────────┘
```

- **Pages** handle UI rendering and user interactions
- **Services** contain all business logic and computation
- **Persistence** provides thread-safe CSV I/O with file locking
- **data/** stores all CSV tables and per-activity files

---

## 2. CSV Data Model

### Main Tables

| File | ID Column | Written By |
|------|-----------|------------|
| `activities.csv` | `activityId` | StravaService |
| `activities_metrics.csv` | `activityId` | MetricsService |
| `planned_sessions.csv` | `plannedSessionId` | Planner page |
| `planned_metrics.csv` | `plannedSessionId` | MetricsService |
| `links.csv` | `linkId` | LinkingService |
| `daily_metrics.csv` | `dailyId` | MetricsService |
| `weekly_metrics.csv` | `weekStartDate` | MetricsService |
| `athlete.csv` | `athleteId` | Athlete page |
| `settings.csv` | `coachId` | Settings page |
| `thresholds.csv` | `thresholdId` | Athlete page |
| `tokens.csv` | `athleteId` | StravaService |
| `races.csv` | `raceId` | PacerService |

### Per-Activity Directories

| Directory | Content | Written By |
|-----------|---------|------------|
| `timeseries/{id}.csv` | Raw streams (hr, pace, elevation, lat, lon) | StravaService |
| `metrics_ts/{id}.csv` | Cached metrics (hr_smooth, clusters, speedeq, grade) | SpeedProfileService |
| `speed_profil/{id}.csv` | Max speed profile curves | SpeedProfileService |
| `hr_zones/{id}.csv` | Zone time summaries | HrZonesService |
| `raw/strava/{id}.json` | Cached Strava API responses | StravaService |
| `race_pacing/{id}_segments.csv` | Race pacing segments | PacerService |

### Relationships

```
activities.csv ──┬── activities_metrics.csv   (1:1 on activityId)
                 │
                 ├── links.csv               (N:1 on activityId)
                 │       │
planned_sessions.csv ────┘                   (N:1 on plannedSessionId)
                 │
                 └── planned_metrics.csv      (1:1 on plannedSessionId)

activities_metrics.csv ──→ daily_metrics.csv  (aggregated by date)
                         ──→ weekly_metrics.csv (aggregated by ISO week)
```

---

## 3. Data Ingestion: Strava Sync

**Trigger:** Settings page → `"Synchroniser les {N} derniers jours"` button

### Call Chain

```
Settings.py: st.button("Synchroniser les {sync_days} derniers jours")
  → strava_service.preview_sync_last_n_days(athlete_id, sync_days)   [optional preview]
  → strava_service.sync_last_n_days(athlete_id, sync_days)
      ├── For each activity NOT in cache:
      │     → Strava API: GET /athlete/activities (paginated)
      │     → Strava API: GET /activities/{id}           (detail)
      │     → Strava API: GET /activities/{id}/streams   (timeseries)
      │     → Write raw/strava/{id}.json
      │     → Write timeseries/{id}.csv
      │     → lap_metrics.compute_and_store()
      │     → _map_activity_row() → activities.csv row
      │
      ├── For each activity already in cache:
      │     → Read raw/strava/{id}.json
      │     → lap_metrics.compute_and_store() (best effort)
      │     → _map_activity_row() → activities.csv row (if missing)
      │
      ├── metrics_service.recompute_for_activities(created_rows)      [pass 1]
      │     → ensures missing dependencies (metrics_ts, speed profile, laps)
      │     → updates activities_metrics/daily_metrics/weekly_metrics
      │
      ├── For each new activity with timeseries:
      │     → speed_profile_service.compute_all_metrics_ts(id)
      │     → speed_profile_service.compute_and_store_speed_profile(id)
      │
      ├── metrics_service.recompute_for_activities(created_rows)      [pass 2]
      │     → refreshes hrSpeedShift after fresh metrics_ts
      │
      └── hr_zones_service.backfill_borders_from_date(athlete_id, earliest_new_date)
            → recomputes zone borders for all activities from that date onward
            → writes hr_zones/{id}.csv + hrZone_z*_upper columns
```

### Strava API Endpoints

| Step | Endpoint | Purpose |
|------|----------|---------|
| 1 | `GET /athlete/activities?after={ts}&per_page=200` | List activities in time window |
| 2 | `GET /activities/{id}` | Full activity detail |
| 3 | `GET /activities/{id}/streams?keys=time,distance,altitude,heartrate,cadence,latlng,velocity_smooth` | Activity streams |
| Refresh | `POST https://www.strava.com/oauth/token` | Token refresh if expired |

### Data Mapping: Strava → `activities.csv`

| Column | Source |
|--------|--------|
| `activityId` | `detail.id` |
| `athleteId` | Parameter |
| `source` | `"strava"` |
| `sportType` | `detail.sport_type` or `detail.type` |
| `name` | `detail.name` |
| `startTime` | `detail.start_date_local` |
| `distanceKm` | `detail.distance / 1000` |
| `elapsedSec` | `detail.elapsed_time` |
| `movingSec` | `detail.moving_time` |
| `ascentM` | `detail.total_elevation_gain` |
| `avgHr` | `detail.average_heartrate` |
| `maxHr` | `detail.max_heartrate` |
| `hasTimeseries` | `True` if streams exist |
| `polyline` | `detail.map.summary_polyline` |
| `rawJsonPath` | Relative path to `raw/strava/{id}.json` |

### Data Mapping: Streams → `timeseries/{id}.csv`

| Column | Source |
|--------|--------|
| `timestamp` | `start_date + time_offsets` |
| `hr` | `heartrate` stream |
| `paceKmh` | `velocity_smooth × 3.6` |
| `elevationM` | `altitude` stream |
| `cadence` | `cadence` stream |
| `lat` | `latlng[0]` |
| `lon` | `latlng[1]` |

### Files Written

| File | Action |
|------|--------|
| `raw/strava/{id}.json` | Created (one per new activity) |
| `timeseries/{id}.csv` | Created (one per activity with streams) |
| `activities.csv` | Rows appended |
| `laps/{id}.csv` | Created/updated when raw details include laps |
| `metrics_ts/{id}.csv` | Created/updated for new activities with timeseries |
| `speed_profil/{id}.csv` | Created/updated for new activities with timeseries |
| `activities_metrics.csv` | Rows upserted |
| `daily_metrics.csv` | Recomputed from earliest affected date |
| `weekly_metrics.csv` | Recomputed from earliest affected week |
| `hr_zones/{id}.csv` | Created/updated from earliest new activity date onward |

---

## 4. Data Ingestion: Rebuild from Cache

**Trigger:** Settings page → `"Reconstruire les activités depuis le cache Strava"` button

### Call Chain

```
Settings.py: st.button("Reconstruire les activités depuis le cache Strava")
  → strava_service.rebuild_from_cache(athlete_id, progress_callback)
      1. Scan data/raw/strava/*.json
      2. For each cached JSON:
         → Build activity row via _map_activity_row()
         → Compute lap metrics
      3. Merge rebuilt rows with existing activities.csv
      4. Write full activities.csv
      5. Delete metrics_ts/{id}.csv  (for all cached activities)
      6. Delete speed_profil/{id}.csv (for all cached activities)
      7. For each activity with timeseries:
         → speed_profile_service.compute_all_metrics_ts(id)
         → speed_profile_service.compute_and_store_speed_profile(id)
      8. metrics_service.recompute_for_activities(all_ids) [single pass]
         → rebuilds activities_metrics/daily_metrics/weekly_metrics
         → backfills missing HR zones from earliest impacted date
```

### Key Differences from Sync

| Aspect | Sync | Rebuild from Cache |
|--------|------|--------------------|
| API calls | Yes (for missing activities) | None |
| Source | Strava API + cache | Only `raw/strava/*.json` |
| Timeseries | Creates new | Keeps existing |
| `metrics_ts/` | Not touched | Deleted and recomputed |
| `speed_profil/` | Created for new | Deleted and recomputed |
| Metrics passes | 2 targeted passes on new IDs | 1 pass after full TS rebuild |
| HR zones | Backfilled from earliest new activity date | Backfilled during incremental recompute |

### Files Written

| File | Action |
|------|--------|
| `activities.csv` | Full rewrite (merge) |
| `metrics_ts/{id}.csv` | Deleted then recomputed |
| `speed_profil/{id}.csv` | Deleted then recomputed |
| `activities_metrics.csv` | Recomputed |
| `daily_metrics.csv` | Recomputed |
| `weekly_metrics.csv` | Recomputed |
| `hr_zones/{id}.csv` | Created/updated for impacted date range |

---

## 5. Metrics Computation Pipeline

### 5.1 Recompute All (`recompute_all`)

**Trigger:** Settings page → `"Recompute weekly & daily metrics"` button

```
metrics_service.recompute_all(athlete_id=None)
  → _target_athletes()                          # All athletes
  → _ensure_dependencies(all_activity_ids)      # metrics_ts + speed_profile + laps (if missing)
  → _recompute_for_athletes(ids, replace_all=True)
      For each athlete:
        → _load_hr_profiles()                    # athlete.csv (hrRest, hrMax)
        → _compute_activity_metrics()            # activities.csv → formulas
        → _compute_planned_metrics()             # planned_sessions.csv → formulas
        → _build_weekly_metrics()                # Aggregate by ISO week
        → _build_daily_metrics()                 # Aggregate by day + rolling
        → _persist_frame() × 4                   # Write all CSVs
  → _ensure_hr_zones(all_activity_ids)          # hr_zones backfill from earliest impacted date
```

**Input CSVs:** `activities.csv`, `athlete.csv`, `settings.csv`, `planned_sessions.csv`, `thresholds.csv`, `timeseries/{id}.csv`, `raw/strava/{id}.json`

**Output CSVs:** `activities_metrics.csv`, `planned_metrics.csv`, `weekly_metrics.csv`, `daily_metrics.csv`, `laps/{id}.csv`, `metrics_ts/{id}.csv`, `speed_profil/{id}.csv`, `hr_zones/{id}.csv`

### 5.2 Recompute for Activities (`recompute_for_activities`)

**Trigger:** Strava sync (new activities), Save Settings (factor change), Recompute zones

```
metrics_service.recompute_for_activities(activity_ids)
  → _ensure_dependencies(activity_ids)          # backfill missing metrics_ts/laps/speed profiles
  → _load_hr_profiles()
  → _compute_activity_metrics()               # Only for given IDs
  → Upsert into activities_metrics.csv         # Remove old + concat new
  → _resolve_impacted_start_date()             # Earliest date among updated
  → _build_weekly_metrics_from_date()          # From impacted week onward
  → _build_daily_metrics_from_date()           # From impacted date (27-day lookback)
  → _persist_frame()
  → _ensure_hr_zones(activity_ids)             # backfill zone borders from earliest impacted date
```

### 5.3 Recompute Single Activity

**Trigger:** Activities page → `"Recalculer métriques"` button (per activity card)

```
metrics_service.recompute_single_activity(activity_id)
  → _ensure_dependencies([activity_id])
  → _compute_activity_metrics()                # One activity
  → activity_metrics.update(id, row)           # Single row upsert
  → _build_weekly_metrics()                    # Full rebuild for athlete
  → _build_daily_metrics()                     # Full rebuild for athlete
  → _persist_frame()
  → _ensure_hr_zones([activity_id])
```

### 5.4 Recompute for Categories

**Trigger:** Settings page → `"Save Settings"` when DistEq factors change

```
metrics_service.recompute_for_categories(categories)
  → Filter activities.csv by category
  → recompute_for_activities(matching_ids)
```

Category mapping when factors change:
- `distanceEqFactor` changed → recompute `RUN`, `TRAIL_RUN`, `HIKE`
- `bikeEq*` changed → recompute `RIDE`
- `skiEq*` changed → recompute `BACKCOUNTRY_SKI`

### 5.5 Activity-Level Metrics Formulas

**Distance-Equivalent (`distanceEqKm`):**

| Category | Formula |
|----------|---------|
| RUN, TRAIL_RUN, HIKE | `distanceKm + ascentM × distanceEqFactor` |
| RIDE | `distance × bikeEqDistance + ascent × bikeEqAscent + descent × bikeEqDescent` |
| BACKCOUNTRY_SKI | `distance × skiEqDistance + ascent × skiEqAscent + descent × skiEqDescent` |

**TRIMP (Training Impulse):**

```
hrr = (avg_hr - hr_rest) / (hr_max - hr_rest)    # clamped [0, 1.2]
duration_hours = duration_sec / 3600
trimp = duration_hours × hrr × 0.64 × exp(1.92 × hrr)
```

Returns 0 if HR profile is invalid or missing.

**HR Speed Shift (`hrSpeedShift`):**
- Only computed when `hasTimeseries=True` and timeseries contains `hr`
- GPS-based: `SpeedProfileService.preprocess_timeseries` → `compute_hr_speed_shift`
- Non-GPS: uses `paceKmh` as speed proxy

### 5.6 Daily Metrics Aggregation

**Filter:** Only `TRAINING_LOAD_CATEGORIES` = {`RUN`, `TRAIL_RUN`, `HIKE`, `BACKCOUNTRY_SKI`}

**Per-day sums:** `distanceKm`, `distanceEqKm`, `timeSec`, `trimp`, `ascentM`

**Date range:** Full range from min to max activity date, zero-filled for no-activity days.

**Rolling windows (pandas `.rolling()`):**

| Window | Columns | Formula |
|--------|---------|---------|
| Acute (7 days) | `acuteDistanceKm`, `acuteTimeSec`, `acuteDistanceEqKm`, `acuteTrimp`, `acuteAscentM` | `.rolling(7, min_periods=1).mean()` |
| Chronic (28 days) | `chronicDistanceKm`, `chronicTimeSec`, `chronicDistanceEqKm`, `chronicTrimp`, `chronicAscentM` | `.rolling(28, min_periods=1).mean()` |

### 5.7 Weekly Metrics Aggregation

**Per ISO week, combines planned + actual:**

| Column | Source |
|--------|--------|
| `plannedTimeSec` | Sum from `planned_metrics.csv` |
| `plannedDistanceKm` | Sum from `planned_metrics.csv` |
| `plannedDistanceEqKm` | Sum from `planned_metrics.csv` |
| `actualTimeSec` | Sum from `activities_metrics.csv` |
| `actualDistanceKm` | Sum from `activities_metrics.csv` |
| `actualDistanceEqKm` | Sum from `activities_metrics.csv` |
| `adherencePct` | `min(actualDistanceEqKm / plannedDistanceEqKm × 100, 999)` |

---

## 6. HR Zones Recomputation

**Triggers:**
- Settings page → `"Recompute zones"` button (full refresh)
- Strava sync → automatic `backfill_borders_from_date` from earliest imported activity date
- Metrics recompute flows (`recompute_all`, `recompute_for_activities`, `recompute_single_activity`)
- Activity detail page → lazy backfill when zone summary is missing

### Call Chain

```
Settings.py: st.button("Recompute zones")
  → _recompute_zone_artifacts(athlete_id, n_cluster, hr_zone_count, hr_zone_window_days)
      1. metrics_service.recompute_for_activities(all_activity_ids)
      2. For each activity:
         → speed_profile_service.process_timeseries(id, strategy="cluster", n_clusters=N)
         → speed_profile_service.save_metrics_ts(id, result)
         → speed_profile_service.compute_and_save_elevation_metrics(id)
      3. hr_zones_service.backfill_all_borders(athlete_id)
         For each activity (last 90 days, RUN/TRAIL_RUN):
           → GMM clustering on HR samples → zone borders
           → Write hr_zones/{id}.csv
           → Update activities_metrics.csv (hrZone_z*_upper columns)

Activity.py (lazy)
  → hr_zones_service.get_or_compute_zones(activity_id)
      → If summary missing:
           - Find last computed zone date before activity
           - backfill_borders_from_date(athlete_id, from_date, to_date=activity_date)
      → Reload summary + borders and render charts
```

### HR Zone Computation Logic

1. **Input:** HR samples from `timeseries/{id}.csv` + cluster data from `metrics_ts/{id}.csv`
2. **Window:** Last `hr_zone_window_days` (default 90) days, filtered to RUN/TRAIL_RUN
3. **Method:** Gaussian Mixture Model (GMM) on HR values to find zone boundaries
4. **Output per activity:**
   - `hr_zones/{id}.csv`: zone summaries (zone, lower_hr, upper_hr, time_seconds, avg_speed, avg_speedeq)
   - `activities_metrics.csv`: zone border columns (`hrZone_z1_upper`, ..., `hrZone_z5_upper`)

### Date-Window Backfill

`HrZonesService.backfill_borders_from_date(athlete_id, from_date, to_date=None)`:
- Filters `activities_metrics.csv` to athlete + date window
- Sorts by `startDate`, `activityId`
- Recomputes and persists borders + summary per activity
- Used for sync incremental updates and lazy Activity-page fill

### Files Written

| File | Action |
|------|--------|
| `activities_metrics.csv` | Updated (metrics + zone borders) |
| `daily_metrics.csv` | Recomputed |
| `weekly_metrics.csv` | Recomputed |
| `metrics_ts/{id}.csv` | Rewritten (clusters + elevation) |
| `hr_zones/{id}.csv` | Created/updated per activity |

---

## 7. Speed Profile & Elevation Metrics

### 7.1 Speed Profile Computation

**Trigger:** Automatic during Strava sync; on-demand on Activity detail page if missing

```
speed_profile_service.compute_and_store_speed_profile(activity_id)
  → timeseries_service.load(activity_id)           # timeseries/{id}.csv
  → preprocess_timeseries(df)                       # GPS preprocessing
  → Compute max speed windows (various durations)
  → Write speed_profil/{id}.csv
```

### 7.2 Full Metrics TS Computation

**Trigger:** Rebuild from cache; Recompute zones

```
speed_profile_service.compute_all_metrics_ts(activity_id)
  → process_timeseries(id, strategy="cluster")      # HR/speed analysis
  → save_metrics_ts(id, result)                      # hr_smooth, hr_shifted, cluster
  → compute_and_save_elevation_metrics(id)           # speedeq_smooth, grade_ma, elevation_ma
```

### 7.3 Elevation Metrics Caching

**Used by:** Activity page elevation profile, Dashboard speed scatter

```
preprocess_for_elevation_profile(df, service, activity_id)
  → Check metrics_ts/{id}.csv for cached columns
  → If cached: return directly
  → If not: compute in memory (does NOT save to avoid overwriting HR data)
```

Persistent save only via `compute_and_save_elevation_metrics()` or `compute_all_metrics_ts()`.

### `metrics_ts/{id}.csv` Columns

| Column Group | Source | Columns |
|-------------|--------|---------|
| HR Analysis | `process_timeseries()` | `hr_smooth`, `hr_shifted`, `speed_smooth`, `cluster` |
| Elevation | `compute_and_save_elevation_metrics()` | `speedeq_smooth`, `grade_ma_10`, `elevationM_ma_5`, `cumulated_distance` |

---

## 8. Planner: Session CRUD

**Location:** Planner page

### Create Session

```
Planner.py: Submit form
  → sessions_repo.create(payload)
      → Write row to planned_sessions.csv
  → metrics_service.recompute_planned_incremental(athlete_id, session_date)
      → planned_metrics.csv   (full athlete recompute)
      → weekly_metrics.csv    (from impacted ISO week onward)
```

### Edit Session

```
Planner.py: Edit form submit
  → sessions_repo.update(plannedSessionId, payload)
      → Update row in planned_sessions.csv
  → metrics_service.recompute_planned_incremental(athlete_id, session_date)
      → planned_metrics.csv   (full athlete recompute)
      → weekly_metrics.csv    (from impacted ISO week onward)
```

### Delete Session

```
Planner.py: Delete button
  → sessions_repo.delete(plannedSessionId)
      → Remove row from planned_sessions.csv
  → metrics_service.recompute_planned_incremental(athlete_id, deleted_session_date)
      → planned_metrics.csv   (full athlete recompute)
      → weekly_metrics.csv    (from impacted ISO week onward)
```

### Incremental Strategy

- Planned metrics are recomputed for the athlete (table is small).
- Weekly metrics are rebuilt only from `iso_week_start(affected_date)` onward.
- `daily_metrics.csv` is not touched (daily table is based on actual activities only).

### Session Types & Templates

- **Session types:** `FUNDAMENTAL_ENDURANCE`, `LONG_RUN`, `INTERVAL_SIMPLE`, `RACE`
- **Week templates** stored in `templates.csv`: apply a full week of sessions at once
- **Session templates** stored in `session_templates.csv`: reusable session payloads (JSON)
- `stepsJson` column in `planned_sessions.csv` stores interval structures

### Planned Metrics

| Column | Formula |
|--------|---------|
| `plannedDistanceEqKm` | `plannedDistanceKm + plannedAscentM × distanceEqFactor` |
| `plannedTrimp` | TRIMP from threshold HR targets and planned duration |

---

## 9. Activity Linking

**Location:** Activities page → "Associer" button on planned session cards

### Link Activity to Planned Session

```
Activities.py: "Associer cette activité" button in dialog
  → link_service.create_link(athlete_id, activity_id, planned_session_id, window_days=14)
      → Validate: activity and session belong to same athlete
      → Validate: activity date within ±window_days of session date
      → links_repo.create(row)
          → Write row to links.csv
```

### Unlink

```
link_service.delete_link(link_id)
  → links_repo.delete(link_id)
      → Remove row from links.csv
```

### Candidate Matching

`suggest_for_planned_session()` ranks candidate activities by:
- Date proximity to planned session
- Distance similarity
- Duration similarity
- Returns `match_score` (0-1)

### Effect on Analytics

`AnalyticsService.daily_range()` uses `links.csv` to move planned metric values to the activity date (instead of the planned date) when a link exists. This ensures the "planned vs actual" comparison aligns correctly.

---

## 10. Race Pacer

**Location:** RacePacing page

### Import GPX and Segment

```
RacePacing.py: Upload GPX file
  → gpx_parser.parse_gpx_to_timeseries(gpx_bytes)
      → Parse trkpt elements (lat, lon, elevation, timestamp)
      → Return DataFrame

  → pacer_service.preprocess_timeseries_for_pacing(df)
      → Distance computation (haversine)
      → Elevation smoothing
      → Grade computation
      → Outlier filtering
      → cumulated_distance, distanceEq

  → pacer_service.segment_course(metrics_df, aid_stations_km)
      → Grade-based segmentation
      → Aid-station splits at configured km points
      → Merge small segments
      → Compute per-segment: distanceKm, distanceEqKm, elevGainM, elevLossM
```

### Edit Speeds & Recompute Times

```
RacePacing.py: "Recalculer les temps" button
  → For each segment:
      → pacer_service.compute_segment_time(distanceEq, distance, speedEq, speed)
      → Update timeSec
  → Session state update (no CSV write)
```

### Save Race

```
RacePacing.py: "Enregistrer la course" button
  → pacer_service.save_race(name, aid_km, segments_df, race_id, aid_stations_times)
      → Write/update races.csv (race metadata)
      → Write race_pacing/{race_id}_segments.csv (segment details)
      → Invalidate race_pacing/{race_id}_*_comparison.csv cache files
```

### Link Race to Activity

```
Activity.py: "Lier un pacing" button
  → pacer_service.link_race_to_activity(activity_id, race_id)
      → Write/update race-pacing-link.csv
      → Invalidate race_pacing/*_{activity_id}_comparison.csv

Activity.py: "Délier" button
  → pacer_service.unlink_race_from_activity(activity_id)
      → Remove row from race-pacing-link.csv
      → Invalidate race_pacing/*_{activity_id}_comparison.csv
```

### Pacing vs Actual Comparison

```
Activity.py: "Comparaison pacing vs réel" tab
  → pacer_service.compare_race_segments_with_activity(race_id, activity_id, timeseries_df)
      → If race_pacing/{race_id}_{activity_id}_comparison.csv exists: load cached result
      → Else:
           - Preprocess activity timeseries
           - Map planned segments onto actual GPS trace
           - Compute delta time, delta speed per segment
           - Save cache file race_pacing/{race_id}_{activity_id}_comparison.csv
  → st.spinner("Calcul de la comparaison pacing vs réel...")
  → render_comparison_elevation_profile()
  → render_delta_bar_chart()
  → render_comparison_table()
```

### Files Written

| File | Action |
|------|--------|
| `races.csv` | Row created/updated |
| `race_pacing/{id}_segments.csv` | Created/updated |
| `race_pacing/{race_id}_{activity_id}_comparison.csv` | Created/read/deleted (cache) |
| `race-pacing-link.csv` | Row created/deleted |

---

## 11. Dashboard (Read-Only)

**Location:** Dashboard page

The Dashboard **reads** data but **never writes** to CSV files.

### Tabs and Data Sources

| Tab | Data Source | Service |
|-----|-----------|---------|
| **Charge** (Training Load) | `daily_metrics.csv` + `planned_metrics.csv` + `activities_metrics.csv` | `AnalyticsService.daily_range()` |
| **SpeedEq** scatter | `activities_metrics.csv` + `activities.csv` | Direct CSV read |
| **FC vs Vitesse** | `metrics_ts/{id}.csv` (cluster data) | `dashboard_data_service.load_hr_speed_data()` |
| **Décalage HR** | `activities_metrics.csv` (hrSpeedShift column) | Direct CSV read |
| **Profil de vitesse** | `speed_profil/{id}.csv` | `dashboard_data_service.load_aggregated_speed_profile()` |
| **Nuage de vitesse max** | `speed_profil/{id}.csv` | `dashboard_data_service.load_speed_profile_cloud()` |
| **Zones HR** borders | `activities_metrics.csv` (hrZone_z*_upper columns) | Direct CSV read |
| **Vitesse par zone** | `hr_zones/{id}.csv` | `hr_zones_service.build_zone_speed_evolution()` |

### Training Load Chart Logic

The "Charge" tab computes acute/chronic curves including planned values:

1. Load `daily_range()` from `AnalyticsService` (actual + planned daily values)
2. Compute `actual_acute` = 7-day rolling mean of actual values
3. Compute `actual_chronic` = 28-day rolling mean of actual values
4. For future dates: blend actual (past) + planned (future) for projected curves
5. Display band at 75%-150% of chronic load

---

## 12. Analytics (Read-Only)

**Location:** Analytics page

The Analytics page **reads** data but **never writes** to CSV files (except saving activity type preferences to `settings.csv`).

### Weekly Planned vs Actual

```
AnalyticsService.weekly_range(athlete_id, metric_label, selected_types, start_date, end_date)
  → Load planned_metrics.csv (grouped by ISO week)
  → Load activities_metrics.csv (grouped by ISO week, filtered by category)
  → Build weekly grid with planned_value and actual_value
  → analytics.build_planned_vs_actual_segments() splits into:
      "Réalisé" (actual up to plan), "Au-dessus du plan" (excess), "Plan manquant" (shortfall)
```

### Daily Planned vs Actual

```
AnalyticsService.daily_range(athlete_id, metric_label, selected_types, start_date, end_date)
  → Load planned_metrics.csv (by date)
  → Load activities_metrics.csv (by date, filtered by category)
  → Use links.csv to reschedule planned values to activity dates
  → Build daily grid with planned_value and actual_value
  → analytics.build_planned_vs_actual_segments() for stacked bars
```

### HR Zone Distribution

- Weekly view: `hr_zones_service.build_weekly_zone_data()` → stacked bars by week
- Session view: `hr_zones_service.load_zone_summary()` per activity → stacked bars by session

### Data Sources

| Chart | Read From |
|-------|-----------|
| Weekly bars | `weekly_metrics.csv`, `planned_metrics.csv`, `activities_metrics.csv` |
| Daily bars | `planned_metrics.csv`, `activities_metrics.csv`, `links.csv` |
| HR zone distribution | `hr_zones/{id}.csv` |
| Summary metrics | `activities_metrics.csv` |

---

## 13. Activity Detail Page (Detail + Actions)

**Location:** Activity page (navigated from Activities feed)

Displays activity data and now exposes direct maintenance actions.

### Data Loading

```
activity_detail_service.get_detail(athlete_id, activity_id)
  → activities.csv           (activity row)
  → activities_metrics.csv   (metrics: trimp, distEq, hrShift)
  → links.csv                (linked planned session info)
  → planned_sessions.csv     (comparison panel data)
  → timeseries/{id}.csv      (decode polyline for map)
```

### Direct Actions

| UI Action | Service Call | Files Written |
|-----------|--------------|---------------|
| `Recalculer métriques` | `metrics_service.recompute_single_activity(activity_id)` | `activities_metrics.csv`, `daily_metrics.csv`, `weekly_metrics.csv`, optional `metrics_ts/`, `speed_profil/`, `laps/`, `hr_zones/` |
| `Associer une séance` | `link_service.create_link(...)` | `links.csv` |
| `Lier/Délier un pacing` | `pacer_service.link_race_to_activity()` / `unlink_race_from_activity()` | `race-pacing-link.csv`, comparison cache invalidation |

### Timeseries Tab

| Chart | Source |
|-------|--------|
| HR over time | `timeseries/{id}.csv` + `metrics_ts/{id}.csv` (hr_smooth, cluster) |
| Speed over time | `timeseries/{id}.csv` |
| Elevation over time | `timeseries/{id}.csv` |
| SpeedEq over time | `metrics_ts/{id}.csv` (speedeq_smooth) |
| Elevation profile | `metrics_ts/{id}.csv` (grade_ma_10, elevationM_ma_5, cumulated_distance) |
| Grade histogram | Computed from elevation profile data |
| Cluster means (HR vs Speed / SpeedEq) | `metrics_ts/{id}.csv` grouped by `cluster` |

### Speed Profile Section

```
speed_profile_service.load_speed_profile(activity_id)
  → If missing: compute_and_store_speed_profile(activity_id)
  → Display max speed curve chart
```

### HR Zones Section

```
hr_zones_service.get_or_compute_zones(activity_id)
  → Load hr_zones/{id}.csv
  → If missing: lazy backfill from last computed zone date to current activity date
  → Display zone timeseries, zone time bars, zone speed table
```

### Interval Comparison Tab

Shown only when linked planned session is `INTERVAL_SIMPLE`:
```
interval_comparison_service.compare(activity_id, planned_session)
  → Parse stepsJson from planned session
  → Match planned intervals to activity laps
  → Display comparison table (planned time vs actual, pace, HR)
```

---

## 14. Settings: Save Settings

**Trigger:** Settings page → `"Save Settings"` button

### Call Chain

```
Settings.py: st.button("Save Settings")
  → settings_repo.update("coach-1", payload)
      → Write to settings.csv
  → If distanceEqFactor changed:
      → metrics_service.recompute_for_categories(["RUN", "TRAIL_RUN", "HIKE"])
  → If bikeEq* changed:
      → metrics_service.recompute_for_categories(["RIDE"])
  → If skiEq* changed:
      → metrics_service.recompute_for_categories(["BACKCOUNTRY_SKI"])
```

### Settings Payload

| Key | Purpose |
|-----|---------|
| `distanceEqFactor` | Trail D+ factor (default 0.01) |
| `bikeEqDistance`, `bikeEqAscent`, `bikeEqDescent` | Bike DistEq factors |
| `skiEqDistance`, `skiEqAscent`, `skiEqDescent` | Ski DistEq factors |
| `stravaSyncDays` | Number of days to sync with Strava |
| `nCluster` | Number of GMM clusters for HR analysis |
| `hrZoneCount` | Number of HR zones (2-5) |
| `hrZoneWindowDays` | Window for zone border computation (days) |

---

## 15. CSV Modification Matrix

Summary of which CSV files are modified by each user action.

| Action | activities | act_metrics | planned_sessions | planned_metrics | links | daily_metrics | weekly_metrics | settings | metrics_ts | speed_profil | hr_zones | races | race_pacing |
|--------|:---------:|:-----------:|:----------------:|:--------------:|:-----:|:-------------:|:--------------:|:--------:|:----------:|:------------:|:--------:|:-----:|:-----------:|
| **Strava Sync** | W | W | | | | W | W | | W | W | W | | |
| **Rebuild from Cache** | W | W | | | | W | W | | W | W | W | | |
| **Recompute All** | | W | | W | | W | W | | W* | W* | W | | |
| **Recompute Single Activity** | | W | | | | W | W | | W* | W* | W | | |
| **Recompute Zones** | | W | | | | W | W | | W | | W | | |
| **Save Settings** | | W* | | | | W* | W* | W | | | | | |
| **Create Planned Session** | | | W | W | | | W | | | | | | |
| **Edit Planned Session** | | | W | W | | | W | | | | | | |
| **Delete Planned Session** | | | W | W | | | W | | | | | | |
| **Link Activity** | | | | | W | | | | | | | | |
| **Unlink Activity** | | | | | W | | | | | | | | |
| **Save Race** | | | | | | | | | | | | W | W |
| **Open Pacing Comparison Tab** | | | | | | | | | | | | | W* |
| **Link Race to Activity** | | | | | | | | | | | | | W† |
| **Unlink Race** | | | | | | | | | | | | | W† |

`W` = Written, `W*` = Conditionally written, `W†` = `race-pacing-link.csv` write/delete
