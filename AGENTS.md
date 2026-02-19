# AGENTS

Guidelines for AI coding agents working in this repository (Running Manager).

## Project Overview

- Streamlit multi-page app for trail running coaching (single coach/athlete MVP)
- Key domains: weekly planning, session templates, Strava/Garmin import, manual linking with RPE, analytics, TCX export
- Storage: CSVs under `data/` with pandas; per-activity timeseries in `data/timeseries/`
- Locale: UI shows fr-FR formatting (decimal comma); CSV persists with `.`

See `.taskmaster/docs/prd.txt` for full requirements.

## Directory Navigation

| Directory | Purpose | See |
|-----------|---------|-----|
| `data/` | CSV tables and timeseries | [data/AGENTS.md](data/AGENTS.md) |
| `persistence/` | CSV storage and repositories | [persistence/AGENTS.md](persistence/AGENTS.md) |
| `services/` | Domain services | [services/AGENTS.md](services/AGENTS.md) |
| `utils/` | Config, formatting, helpers | [utils/AGENTS.md](utils/AGENTS.md) |
| `pages/` | Streamlit UI pages | [pages/AGENTS.md](pages/AGENTS.md) |
| `widgets/` | Reusable UI components | [widgets/AGENTS.md](widgets/AGENTS.md) |
| `graph/` | Visualization components | [graph/AGENTS.md](graph/AGENTS.md) |
| `ui/` | Complex UI widgets | [ui/AGENTS.md](ui/AGENTS.md) |
| `tests/` | Pytest suite | [tests/AGENTS.md](tests/AGENTS.md) |

## Data Model Overview

```
athlete.csv ──────────────────────────────────────────────────────┐
     │                                                            │
     ├── activities.csv ─── activities_metrics.csv                │
     │        │                    │                              │
     │        └────── links.csv ───┘                              │
     │                    │                                       │
     ├── planned_sessions.csv ─── planned_metrics.csv             │
     │                                                            │
     ├── daily_metrics.csv (acute/chronic windows)                │
     │                                                            │
     ├── weekly_metrics.csv (aggregates)                          │
     │                                                            │
     ├── thresholds.csv (HR/pace zones)                           │
     │                                                            │
     └── settings.csv (distanceEqFactor, bike factors)            │
                                                                  │
session_templates.csv ────────────────────────────────────────────┘
```

### Key Tables

| Table | ID Column | Purpose |
|-------|-----------|---------|
| `activities.csv` | `activityId` | Imported activities (Strava/Garmin) |
| `planned_sessions.csv` | `plannedSessionId` | Coach-planned sessions |
| `links.csv` | `linkId` | Activity ↔ PlannedSession with RPE |
| `activities_metrics.csv` | `activityId` | Per-activity DistEq, TRIMP |
| `daily_metrics.csv` | `dailyId` | Daily aggregates + acute/chronic |
| `weekly_metrics.csv` | `weekStartDate` | Weekly aggregates |

### Session Types

| Type | Description |
|------|-------------|
| `FUNDAMENTAL_ENDURANCE` | Easy runs with pace/hr target |
| `LONG_RUN` | Extended duration runs |
| `INTERVAL_SIMPLE` | Structured intervals (warmup, loops, cooldown) |
| `RACE` | Race sessions |

### Metrics Formulas

- **Distance-equivalent**: `distanceEqKm = distanceKm + ascentM * distanceEqFactor`
- **TRIMP**: HR reserve weighting with exponential factor
- **Training load categories**: `RUN`, `TRAIL_RUN`, `HIKE`, `BACKCOUNTRY_SKI`

## Setup

```bash
# Python 3.11+
uv sync                           # or: uv pip install -r requirements.txt
uv run streamlit run app.py       # Run app
pytest                            # Run tests
```

Environment variables (`.env`):
- `DATA_DIR` (default `./data`)
- `STRAVA_CLIENT_ID`, `STRAVA_CLIENT_SECRET`, `STRAVA_REDIRECT_URI`
- `ENCRYPTION_KEY` (Fernet key for token encryption)
- `MAPBOX_API_KEY` (optional, for maps)

## Commands

| Command | Description |
|---------|-------------|
| `uv run streamlit run app.py` | Run application |
| `pytest` | Run test suite |
| `ruff check .` | Lint code |

## Invariants (must-keep)

### CSV Persistence
- Always write `.` as decimal separator (never locale comma)
- Use `CsvStorage` for all I/O (portalocker locking)
- Headers defined in `persistence/repositories.py`

### Formatting
- Use `utils/formatting.py` for fr-FR display only
- Never mix UI formatting into storage

### Secrets
- Never log raw secrets/tokens
- Use `utils/crypto.encrypt_text/decrypt_text` for token storage
- Use `utils/config.redact()` for logging

### Code Style
- Line length 100
- License header on all Python files:
```python
"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""
```

## Agent Checklist

1. Read relevant AGENTS.md files for context
3. Implement minimally; preserve formatting and headers
4. Add license headers to new Python files
5. Add/adjust tests; run `pytest`

## Maintaining AGENTS.md Files

**Update when:**
- Adding new modules or files
- Changing key function signatures
- Adding new data tables or columns
- Modifying critical invariants
- Solving significant bugs (document pitfalls)
- Adding new features

**What to update:**
- Root `AGENTS.md`: Overall structure, data model, invariants
- Folder `AGENTS.md`: File-specific details, APIs, relationships

**Format rules:**
- Headers + bullets (no paragraphs)
- Tables for structured data
- Code blocks for commands and schemas
- No filler or obvious instructions

## Common Pitfalls

- Writing commas to CSV (must remain `.`)
- Logging raw secrets or storing tokens without `ENCRYPTION_KEY`
- Bypassing locking on CSV writes (always use `CsvStorage`)
- Mixing UI formatting into persistence layer

