# AGENTS

Guidelines for AI coding agents working in this repository (Running Manager).

## Project Overview

- Streamlit multi-page app to manage trail running coaching for a single coach/athlete MVP
- Key domains: weekly planning, session templates, Strava/Garmin import, manual linking with RPE, analytics, TCX export for intervals
- Storage: CSVs under `data/` with pandas; per-activity timeseries in `data/timeseries/`
- Locale: UI shows fr-FR formatting (decimal comma); CSV persists with `.`

See `.taskmaster/docs/prd.txt` for full requirements and domain rules.

## Repository Layout (selected)

- `app.py`: Streamlit entry
- `pages/`: UI pages (Planner, Activities, Athlete, Goals, Analytics, Settings)
- `services/`: domain services (planner, analytics, templates, timeseries; Strava/Garmin stubs present)
- `persistence/`: CSV abstraction and repositories (CRUD over CSV files)
- `utils/`: config loading, crypto, formatting, ids, time helpers
- `data/`: CSV tables and timeseries
- `tests/`: pytest suite and fixtures

## Setup

- Python 3.11+
- Install dependencies via uv
  - With uv lockfile: `uv sync` (preferred)
  - or 'uv pip install -r requirements.txt'

- Environment variables (load via `.env`, see `utils/config.py`):
  - `DATA_DIR` (default `./data`)
  - `STRAVA_CLIENT_ID`, `STRAVA_CLIENT_SECRET`, `STRAVA_REDIRECT_URI` (if enabling Strava OAuth)
  - `ENCRYPTION_KEY` (Fernet key for token at-rest encryption; required before storing tokens)

Run the app: `uv run streamlit run app.py`

## Commands

- Use uv command to replace python. Eg : "uv run" instead of "python3".
- Tests: `pytest`
- Lint (Ruff configured): run ruff if available; line length 100, rules E,F,I

## Invariants & Domain Rules (must-keep)

- CSV persistence
  - Always write `.` as decimal separator; never write locale-specific commas
  - Maintain headers per repo classes in `persistence/repositories.py`; when adding fields, ensure header migration logic keeps column order
  - Use file locking for reads/writes (`portalocker`) as implemented in `persistence/csv_storage.py`
- Formatting (UI-only)
  - Use `utils/formatting.py` for fr-FR display; do not mix UI formatting into storage
- Secrets
  - Never log raw secrets/tokens; use `utils/config.redact()` when necessary
  - Use `utils/crypto.get_fernet(ENCRYPTION_KEY)` and `encrypt_text/decrypt_text` for token storage
- Planning estimates
  - Use `PlannerService` helpers to estimate fundamental distance, interval durations/distances/ascent
  - Respect session types: `FUNDAMENTAL_ENDURANCE`, `LONG_RUN`, `INTERVAL_SIMPLE`
- Distance-equivalent metrics and intense vs easy classification are defined in PRD; align analytics implementations to PRD logic

## Coding Style

- Match existing style (see `pyproject.toml`):
  - Line length 100; keep imports ordered (Ruff I)
  - Prefer early returns; shallow nesting; avoid broad try/except
  - Use descriptive identifiers; avoid single-letter names
  - Keep comments only for non-obvious rationale or constraints

## Security & Privacy

- Do not hardcode secrets; load via environment; `.env` supported
- Redact secrets in logs and errors (never echo tokens)
- Avoid writing PII to test artifacts or snapshots

## Data & Migrations

- When adding CSV fields:
  - Update repo headers in `persistence/repositories.py`
  - If file migration is needed, follow the `PlannedSessionsRepo._migrate_headers_if_needed` pattern
  - Keep operations idempotent; re-running should not corrupt data

## Testing

- Write/adjust pytest tests for any behavior change; keep tests deterministic
- Use provided fakes in `tests/conftest.py` (portalocker stub, Babel stub)
- Prefer small, focused tests validating domain invariants and edge cases

## Streamlit & UI

- Keep display formatting via `utils/formatting`; avoid mixing storage/compute with UI
- Planner-specific computations should live in `services/planner_service.py` or presenters, not directly in pages

## External Integrations

- Strava: implement OAuth and 14-day import in `services/strava_service.py`; persist raw JSON and extract metrics to `activities.csv`
- Garmin: use `garminconnect` for ingest in `services/garmin_import_service.py` per PRD; ensure tokens are encrypted if persisted
- TCX export: only interval workouts (MVP); step end mode `auto|lap` from planned session

## Task Workflows (Taskmaster)

- Use `.taskmaster/` if managing tasks; start with `parse_prd` for initial tasks; expand and track subtasks per dev workflow
- Use mcp server to get the task list, update it and eventually expand tasks but never modify the file inside '.taskmaster/'

## How to Make Changes Safely (Agent Checklist)

- Get the task list from the mcp server and the details of the task to do
- Collect informations needed to implement the task
- Propose a small plan; implement minimally; preserve formatting and headers
- Add/adjust tests; run `pytest`
- For secrets or tokens, enforce redaction and encryption rules
- If adding CSV columns, update headers and migration logic
- Update the task list in the mcp server
- Explain what you did directly in the chat.
- Use a markdown file only when the user asks for it.

## Common Pitfalls

- Writing commas to CSV (must remain `.`); use UI helpers for display only
- Logging raw secrets or storing tokens without `ENCRYPTION_KEY`
- Bypassing locking on CSV writes; always go through `CsvStorage`

## Module Map & Responsibilities

- `utils/`
  - `config.py`: `.env` loading, directory provisioning, config dataclass, `redact()` helper. Reads `DATA_DIR`, Strava vars, `ENCRYPTION_KEY`, and `MAPBOX_API_KEY`.
  - `formatting.py`: fr-FR display helpers for numbers and units; storage must always keep `.` decimals.
  - `crypto.py`: Fernet helpers and safe decrypt errors; used for token-at-rest encryption.
  - `ids.py`: UUID-based `new_id()`.
  - `time.py`: ISO week helpers and `today_local()`.
  - `styling.py`: theme application utilities for Streamlit pages.
  - `auth_state.py`: session state bootstrap utilities.

- `persistence/`
  - `csv_storage.py`: pandas-based CSV IO with `portalocker` (shared/exclusive). Ensures parent dirs, header handling, and upsert/append semantics.
  - `repositories.py`: thin repos per table (headers, id column, migration pattern for header additions).

- `services/`
  - `planner_service.py`: core planning estimations (fundamental pace, intervals duration/distance/ascent, distance-equivalent), weekly totals.
  - `interval_utils.py`: interval steps normalisation, description and serialisation for editor/preview.
  - `analytics_service.py`: load/shape weekly data, compute stacked planned vs actual segments, daily/weekly range APIs.
  - `metrics_service.py`: full metrics pipeline recomputation (activities, planned, daily, weekly), TRIMP, category normalisation, bike DistEq rules.
  - `timeseries_service.py`: load per-activity timeseries CSV.
  - `strava_service.py`: OAuth flow, token storage (encrypted), raw JSON + streams caching, rate-limit logging, incremental sync and cache rebuild.
  - `lap_metrics_service.py`, `linking_service.py`, `activity_feed_service.py`, `templates_service.py`, `session_templates_service.py`, `serialization.py`: domain utilities for laps, linking, feed building and templates.
  - `garmin_import_service.py`: stub placeholder for future ingestion.

- `pages/`
  - `Planner.py`: week editor (sessions CRUD, template apply/save), interval editor integration, cached lookups (`st.cache_data`).
  - `Dashboard.py`: training load (acute/chronic) time series and SpeedEq scatter; Altair charts.
  - `Analytics.py`: planned vs actual (weekly/daily) with category filters and persisted preferences.
  - `Activities.py`: activity feed, planned-unlinked strip, link dialog and navigation to details.
  - `Activity.py`, `Athlete.py`, `Goals.py`, `Session.py`, `SessionCreator.py`, `Settings.py`: supporting pages (settings include Strava OAuth and metrics recompute).

- `ui/`
  - `interval_editor.py`: interval step editor rendering and state wiring (used by Planner and creator flows).

- Root
  - `app.py`: entrypoint, page config, theme and locale setup, safe env preview.
  - `config.py` (root): global constants like `METRICS` for page configs.

## CSV Tables & Repositories (headers and IDs)

-
- Activities: `activities.csv` (id: `activityId`) — core fields: athlete, source, sportType, name, startTime, distanceKm, elapsedSec, movingSec, ascentM, avgHr, hasTimeseries, polyline, rawJsonPath.
- Planned sessions: `planned_sessions.csv` (id: `plannedSessionId`) — athlete, date, type, plannedDistanceKm, plannedDurationSec, plannedAscentM, targetType/Label, notes, stepEndMode, stepsJson.
- Linking: `links.csv` (id: `linkId`) — plannedSessionId, activityId, matchScore, rpe(1-10), comments.
- Activities metrics: `activities_metrics.csv` (id: `activityId`) — per-activity aggregates + DistEq and TRIMP.
- Planned metrics: `planned_metrics.csv` (id: `plannedSessionId`) — per-session planned aggregates incl. DistEq and TRIMP.
- Weekly metrics: `weekly_metrics.csv` (id: `weekStartDate`) — week bounds, planned/actual aggregates, adherence, counters.
- Daily metrics: `daily_metrics.csv` (id: `dailyId`) — per-day sums plus rolling acute/chronic windows.
- Thresholds: `thresholds.csv` (id: `thresholdId`) — named HR/pace zones.
- Goals: `goals.csv` (id: `goalId`).
- Templates: `templates.csv` (id: `templateId`).
- Session templates: `session_templates.csv` (id: `templateId`) — JSON payloads for interval/simple sessions.
- Athletes: `athlete.csv` (id: `athleteId`).
- Settings: `settings.csv` (id: `coachId`) — units, distanceEqFactor, Strava sync window, analytics activity types, bike DistEq factors.
- Tokens: `tokens.csv` (id: `athleteId`, composite with `provider`) — encrypted tokens and expiry.

- Timeseries: `data/timeseries/{activityId}.csv` — sampled streams: timestamp, hr, paceKmh, elevationM, cadence, lat, lon.
- Raw Strava: `data/raw/strava/{activityId}.json` — full activity payloads.

## UI Patterns & Caching

- Use `utils.styling.apply_theme()` and `set_page_config` at page top.
- Locale: call `set_locale("fr_FR")` for display; never mix display formatting into persistence.
- Prefer `st.cache_data` with explicit TTL for DataFrame-loading helpers; use `st.cache_resource` for long-lived clients (repos, services) when appropriate.
- Maintain state keys under a consistent prefix per page (e.g., `planner_*`, `analytics_*`). Clear cache/state after writes to keep UI reactive.
- CSS customisations are injected via `st.markdown(..., unsafe_allow_html=True)`; centralise reusable styles in `utils/styling.py` when possible.

## Analytics & Metrics (domain)

- Distance-equivalent: `distanceEqKm = distanceKm + ascentM * distanceEqFactor`, default factor from `settings.csv` (`distanceEqFactor`, default 0.01).
- TRIMP: computed using HR reserve weighting with exponential factor; planned TRIMP estimated from session targets/steps; actual TRIMP computed per-activity.
- Bike DistEq: when metric is DistEq and category is RIDE, override per-activity using bike factors from settings; optional descent contribution from timeseries.
- Weekly/Daily aggregation: use `MetricsComputationService` to recompute for specific athletes or all; dashboards and analytics pages read from these tables.

## Strava Integration (key constraints)

- OAuth: `authorization_url(state)` and `exchange_code` require `STRAVA_CLIENT_ID/SECRET/REDIRECT_URI` and `ENCRYPTION_KEY`.
- Tokens are encrypted with Fernet and stored in `tokens.csv`; never log raw values; use `redact()` for any previews.
- Sync windows: listing with pagination (200/page), detail+streams for cache-misses; caches raw JSON and streams to CSV; maintains simple rate-limit log and status with short/daily windows.
- Incremental recompute: after creating new `activities.csv` rows, recompute metrics only for affected athletes or activities.

## Data directories & environment

- Data dir: `DATA_DIR` (default `./data`) — created on boot; includes `timeseries/`, `raw/strava/`, `laps/`.
- Map tiles: optional `MAPBOX_API_KEY` for enhanced backgrounds on Activity maps; safe to omit (fallback to default styles).

## Additional Testing Notes

- Pytest suite under `tests/` with focused coverage on services, persistence, and presenters. Use fakes in `tests/conftest.py` for portalocker and Babel.
- Keep tests deterministic; avoid network calls; inject fakes for time and external sessions where needed.