# Utilities

General-purpose helpers for config, formatting, time, crypto, and UI.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Environment loading, DATA_DIR provisioning |
| `formatting.py` | fr-FR display formatting (decimal comma) |
| `crypto.py` | Fernet encryption for tokens |
| `ids.py` | UUID generation |
| `time.py` | ISO week helpers, date parsing |
| `coercion.py` | Type conversion (safe_float, safe_int) |
| `helpers.py` | General helpers (clean_optional) |
| `styling.py` | Streamlit theme application |
| `auth_state.py` | Session state bootstrap |
| `ui_helpers.py` | UI utilities (trigger_rerun, dialogs) |
| `dashboard_state.py` | Dashboard date range state |
| `elevation_preprocessing.py` | Elevation profile data prep |
| `segments.py` | Segment merging for plots |
| `gpx_parser.py` | GPX file parsing |

## Key Functions

### config.py
- `load_config()`: Returns `AppConfig` dataclass
- `redact(value)`: Masks secrets for logging
- Environment vars: `DATA_DIR`, `STRAVA_CLIENT_ID`, `STRAVA_CLIENT_SECRET`, `STRAVA_REDIRECT_URI`, `ENCRYPTION_KEY`, `MAPBOX_API_KEY`

### formatting.py (UI only)
- `fmt_decimal(value, decimals)`: Format with locale comma
- `fmt_pace(kmh)`: Format as min:sec/km
- `fmt_m(meters)`: Format meters with unit
- `format_session_duration(sec)`: Human-readable duration

**Important**: Never use formatting functions in persistence layer.

### crypto.py
- `get_fernet(key)`: Get Fernet instance
- `encrypt_text(text, key)`: Encrypt string
- `decrypt_text(encrypted, key)`: Decrypt string

### time.py
- `today_local()`: Current date in local timezone
- `iso_week_start(year, week)`: Monday of ISO week
- `parse_timestamp(ts)`: Parse ISO timestamp
- `to_date(value)`: Convert to date object
- `ensure_datetime(value)`: Convert to datetime

### coercion.py
- `safe_float(value, default)`: Safe float conversion
- `safe_int(value, default)`: Safe int conversion
- `coerce_float(value, default)`: Coerce with fallback
- `coerce_int(value, default)`: Coerce with fallback

### segments.py
- `merge_small_segments(df, min_distance)`: Merge short segments
- `merge_adjacent_same_color(df)`: Combine adjacent same-color segments

## Related Files

- `services/metrics_service.py`: Uses time and coercion
- `pages/*.py`: Use formatting and styling
- `graph/elevation.py`: Uses segments utilities

## Invariants

- **Formatting**: Only for UI display, never storage
- **Decimal separator**: Storage uses `.`, UI uses `,` (fr-FR)
- **Secrets**: Always use `redact()` in logs
- **Encryption**: Require `ENCRYPTION_KEY` before storing tokens

## Maintaining This File

Update when:
- Adding new utility modules
- Adding new formatting functions
- Changing environment variable requirements
- Adding new coercion helpers
