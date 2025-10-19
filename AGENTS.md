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

## Task Workflows (Taskmaster optional)

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

## Common Pitfalls

- Writing commas to CSV (must remain `.`); use UI helpers for display only
- Logging raw secrets or storing tokens without `ENCRYPTION_KEY`
- Bypassing locking on CSV writes; always go through `CsvStorage`


