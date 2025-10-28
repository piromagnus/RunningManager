## Production Readiness Plan for Running Manager

This document proposes targeted improvements to make the app clearer, faster, and more production-ready. Items are grouped by area with concrete actions and references.

### 1) Architecture and Code Clarity

- Enforce consistent page bootstrapping
  - Create `utils/ui.py` with helpers: `init_page(title, layout='wide', locale='fr_FR')` calling `st.set_page_config`, `apply_theme`, `set_locale`.
  - Replace duplicated boot code in pages (`Planner.py`, `Analytics.py`, `Dashboard.py`, `Activities.py`, `Settings.py`).

- Centralise CSS
  - Move large style blocks into `utils/styling.py` or `ui/styles.css` loaded via `st.markdown(Path(...).read_text(), unsafe_allow_html=True)`.

- Align Mapbox env name
  - Standardise on `MAPBOX_API_KEY` (already used by `utils.config`). Update docs/README and any page using Mapbox.

- Strengthen type hints
  - Add return types and argument annotations for public methods in `services/*` and `persistence/*` to improve IDE support and readability.

### 2) Data Layer and CSV Robustness

- Header migrations
  - Ensure all repos with evolving schemas implement `_migrate_headers_if_needed` like `ActivitiesRepo` and `PlannedSessionsRepo`.

- Strict dtypes at read time
  - Extend `CsvStorage.read_csv` to accept `parse_dates` and column converters for hot paths (e.g., metrics tables).

- Idempotent writes
  - Keep all write paths going through `CsvStorage` with locking; document this rule in `AGENTS.md` (done).

### 3) Performance and Caching

- Streamlit caching
  - Use `st.cache_data` for dataframes with short TTL (5–60s) and `.clear()` after writes.
  - Use `st.cache_resource` for repositories/services that build internal caches or clients.

- Chart performance
  - Prefer Altair aggregations in the transform pipeline (`transform_aggregate`, `transform_joinaggregate`) when possible.
  - Downsample large time series with server-side sampling or `plotly-resampler` where interactivity on dense series is required.

- I/O formats (optional)
  - Keep CSV for canonical storage (project invariant). For internal temp caches, consider Arrow/Parquet to speed repeated loads (not persisted as source of truth).

### 4) Analytics Consistency

- Distance-equivalent
  - Keep the invariant `distanceEqKm = distanceKm + ascentM * distanceEqFactor`. Surface `distanceEqFactor` in `Settings.py` (already present).

- Bike equivalence
  - Document the RIDE overrides and optional descent contribution from timeseries; expose factors in settings (already present) and in `README.md`.

- Rolling windows
  - Standardise: acute = 7-day mean, chronic = 28-day mean. Confirm this is applied consistently in both analytics and dashboard views.

### 5) Strava Integration Hardening

- Token lifecycle
  - Ensure refresh happens 5 minutes before expiry (implemented). Add warning UI if `ENCRYPTION_KEY` is missing.

- Rate limits
  - Keep local JSON log and computed status (present). Expose short wait countdown in UI (done) and provide a guidance tooltip.

- Sync previews
  - Keep `preview_sync_last_n_days` warnings for large windows; cap UI at 31 days by default.

### 6) Testing and CI

- Unit tests
  - Add tests for `AnalyticsService.daily_range/weekly_range` edge cases (empty ranges, single-day windows, category filters).
  - Add tests for `PlannerService.estimate_*` to validate interval estimation math and step normalisation.

- Determinism
  - Keep tests network-free; inject fakes for time and HTTP clients.

- CI suggestions
  - Add GitHub Actions workflow: `uv sync`, `ruff check`, `pytest -q`.

### 7) Observability and Errors

- Logging
  - Use `streamlit.logger.get_logger` and standard module loggers in services (already used). Avoid logging sensitive data; use `redact()`.

- User feedback
  - Prefer `st.status`/`st.spinner` around long operations (Strava sync, metrics recompute). Promote errors to friendly `st.error` with a short remediation.

### 8) Security and Secrets

- Secrets handling
  - No hardcoded secrets; `.env` is required for Strava and `ENCRYPTION_KEY`.
  - Never log secrets; use `utils.config.redact(value)` when displaying identifiers.

### 9) Deployment

- Containerisation
  - Provide a minimal Dockerfile:
    - Base: `python:3.11-slim` with `uv` installation
    - Copy project, run `uv sync --frozen`
    - `CMD ["uv", "run", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]`

- Reverse proxy
  - Front with Nginx or a platform LB; enable HTTPS and set proper `X-Forwarded-*` headers.

- Filesystem
  - Mount persistent volume for `DATA_DIR` and ensure proper permissions.

### 10) UI/UX Enhancements

- Global filters
  - Provide a top-level athlete picker persisted across pages via `st.session_state`.

- Accessibility and i18n
  - Use descriptive tooltips and aria-friendly labels where applicable.

---

## References (best practices)

- Streamlit performance and caching: `st.cache_data`, `st.cache_resource` (Streamlit docs)
- Altair transforms and aggregation: Altair user guide
- Handling large time series: plotly-resampler (downsampling)
- Production deployment: container + reverse proxy (Nginx) patterns


