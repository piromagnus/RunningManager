# Garmin Connect activity import

RunningManager imports Garmin activities through the `garminconnect` Python API and
stores them using the app's existing `activities.csv` and per-activity timeseries
formats. It does not create or query GarminDB's separate SQLite databases.

## Setup

1. Set `GARMIN_EMAIL` and `GARMIN_PASSWORD` in RunningManager's local `.env`.
2. Set `GARMIN_DOMAIN=garmin.com` (or `garmin.cn` for accounts on that domain).
3. Optionally set `GARMIN_TOKEN_STORE` to change the default token store path under `DATA_DIR`.
4. Restart RunningManager, open **Settings → Garmin**, choose the lookback window, and sync.

GarminDB's `GarminConnectConfig.json` is no longer needed by RunningManager's sync.
RunningManager stores refreshed Garmin authentication tokens under `DATA_DIR`;
credentials are read from `.env` and are never logged. If Garmin requests MFA,
enter the current code in the optional field on the Settings page. Sync is
incremental: activity IDs already present in RunningManager are skipped. New
activity summaries, raw detail JSON, and available time-series samples are stored
under the configured `DATA_DIR`; metrics are recomputed after the import.

For activity details, Garmin's `directSpeed` values (meters/second) are converted
to the `paceKmh` field expected by RunningManager (kilometers/hour).
