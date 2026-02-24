# Pages

Streamlit UI pages for the Running Manager application.

## Files

| File | Purpose |
|------|---------|
| `Dashboard.py` | Training load charts, SpeedEq scatter |
| `Planner.py` | Weekly session planning, interval editor |
| `Activities.py` | Activity feed, linking dialog |
| `Activity.py` | Single activity detail, elevation profile |
| `Analytics.py` | Planned vs actual charts |
| `SessionCreator.py` | Template creation wizard |
| `Athlete.py` | Athlete profile management |
| `Goals.py` | Race/goal management |
| `Settings.py` | App settings, Strava OAuth, metrics recompute |
| `Session.py` | Session detail view |
| `RacePacing.py` | Race pacing strategy |

## Page Structure Pattern

```python
"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""
import streamlit as st
from utils.styling import apply_theme

st.set_page_config(page_title="Page Name", layout="wide")
apply_theme()

# Page content...
```

## Key Pages

### Dashboard.py
- Training load (acute/chronic) time series
- SpeedEq scatter visualization
- Uses `dashboard_data_service` and `dashboard_state`

### Planner.py
- Week navigation and session CRUD
- Template apply/save actions
- Interval editor integration
- Uses `recompute_planned_incremental(athlete_id, affected_date)` after session/template writes
- State prefix: `planner_*`

### Activities.py
- Activity feed with filters
- Unlinked planned sessions strip
- Link dialog for manual linking
- Navigation to Activity detail

### Activity.py
- Tabbed layout:
  - `🫀 Zones HR`
  - `📈 Cluster / Profil / QQ`
  - `🗺️ Élévation & Carte`
- Comparison panel for planned vs actual
- Action buttons: `Recalculer métriques` and `Associer une séance`
- Per-activity HR cluster means chart (speed/speed-eq toggle + std bars)
- Cluster regression line is red with displayed equation
- QQ-line row for HR, speed, and speed-eq
- HR zones loaded lazily (`get_or_compute_zones`) when summary is missing
- Pacing comparison uses on-disk cache + spinner feedback

### Analytics.py
- Weekly/daily planned vs actual bars
- Category filters
- Metric selection (Distance, DistEq, Time, TRIMP)
- State prefix: `analytics_*`

### Settings.py
- Strava OAuth flow
- Metrics recomputation trigger
- Distance-equivalent factor configuration
- Bike/ski DistEq factors
- Strava sync/rebuild now refresh metrics dependencies and HR zones automatically

## State Management

- Use `st.session_state` with page-specific prefixes
- Clear cache after writes: `st.cache_data.clear()`
- Common patterns:
  - `st.cache_data`: DataFrame loading
  - `st.cache_resource`: Long-lived services

## Related Files

- `widgets/`: Reusable form components
- `graph/`: Chart rendering
- `services/`: Data loading and computation
- `ui/interval_editor.py`: Interval step editor

## Code Organization

- Keep pages under 500 lines
- Extract forms to `widgets/`
- Extract charts to `graph/`
- Keep business logic in `services/`

## Maintaining This File

Update when:
- Adding new pages
- Changing page responsibilities
- Modifying state key patterns
- Adding new page-level features
