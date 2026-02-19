# Utilities

General-purpose helpers for config, formatting, time, crypto, and UI.

## Files

| File | Purpose |
|------|---------|
| `config.py` | Environment loading, DATA_DIR provisioning |
| `constants.py` | Shared UI/analytics constants |
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
| `elevation_preprocessing.py` | Elevation profile data prep with caching |
| `elevation.py` | Elevation math helpers (avg grade) |
| `grade_classification.py` | Grade classification utilities |
| `segments.py` | Segment merging for plots |
| `series_filters.py` | Outlier filtering for numeric series |
| `gpx_parser.py` | GPX file parsing |
| `metrics_formulas.py` | TRIMP and physiology formulas |
| `timeseries_preprocessing.py` | Shared moving average, distance, grade helpers |

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
- `compute_segment_time(distance_eq_km, distance_km, speed_eq_kmh, speed_kmh)`: Segment time helper

### metrics_formulas.py
- `compute_trimp_hr_reserve(avg_hr, duration_sec, hr_rest, hr_max)`: TRIMP formula
- `compute_trimp_hr_reserve_from_profile(avg_hr, duration_sec, hr_profile)`: HR profile helper

### timeseries_preprocessing.py
- `moving_average(df, window_size, col)`: Centered smoothing
- `distance(df, lat_col, lon_col)`: Haversine distance (km)
- `cumulated_distance(df, distance_col)`: Cumulative distance
- `time_from_timestamp(df, timestamp_col)`: Timestamp parsing
- `duration(df, timestamp_col)`: Durations and cumulated durations
- `speed(df, distance_col, time_col)`: Speed in km/h
- `elevation(df, elevation_col)`: Elevation diffs and gain/loss
- `grade(df, distance_col, elevation_col)`: Grade computation

### elevation.py
- `compute_avg_grade(elev_gain_m, elev_loss_m, distance_km)`: Average grade helper

### coercion.py
- `safe_float(value, default)`: Safe float conversion
- `safe_int(value, default)`: Safe int conversion
- `coerce_float(value, default)`: Coerce with fallback
- `coerce_int(value, default)`: Coerce with fallback

### segments.py
- `merge_small_segments(df, min_distance)`: Merge short segments
- `merge_adjacent_same_color(df)`: Combine adjacent same-color segments

### elevation_preprocessing.py
- `preprocess_for_elevation_profile(df, service, activity_id)`: Preprocess timeseries for elevation profile
  - If `activity_id` provided, loads from cached `metrics_ts` if available
  - If not cached, computes and saves: `speedeq_smooth`, `grade_ma_10`, `elevationM_ma_5`, `cumulated_distance`

### grade_classification.py
- `classify_grade_pacer_5cat(grade, elevation_delta_per_km)`: 5-category classification for pacer
- `classify_grade_elevation_8cat(grade)`: 8-category classification for elevation charts

### series_filters.py
- `filter_series_outliers(df, value_col, reference_col, window, sigma)`: Hampel-style filter

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
