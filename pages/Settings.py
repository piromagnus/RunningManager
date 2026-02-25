"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import datetime as dt
import math
from pathlib import Path

import streamlit as st

from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo, SettingsRepo, TokensRepo
from services.hr_zones_service import HrZonesService
from services.metrics_service import MetricsComputationService
from services.speed_profile_service import SpeedProfileService
from services.strava_service import StravaService
from services.timeseries_service import TimeseriesService
from utils.coercion import coerce_float, coerce_int
from utils.config import load_config
from utils.formatting import set_locale
from utils.ids import new_id
from utils.styling import apply_theme
from utils.ui_helpers import trigger_rerun

st.set_page_config(page_title="Running Manager - Settings")
apply_theme()
st.title("Settings")

cfg = load_config()
set_locale("fr_FR")
storage = CsvStorage(base_dir=Path(cfg.data_dir))
settings_repo = SettingsRepo(storage)
tokens_repo = TokensRepo(storage)
athletes_repo = AthletesRepo(storage)
state_file = Path(cfg.data_dir) / ".strava_state"


def _first_athlete_id() -> str | None:
    df = athletes_repo.list()
    if df.empty:
        return None
    return str(df.iloc[0]["athleteId"])


def _strava_missing_config() -> list[str]:
    missing: list[str] = []
    if not cfg.strava_client_id:
        missing.append("STRAVA_CLIENT_ID")
    if not cfg.strava_client_secret:
        missing.append("STRAVA_CLIENT_SECRET")
    if not cfg.strava_redirect_uri:
        missing.append("STRAVA_REDIRECT_URI")
    if not cfg.encryption_key:
        missing.append("ENCRYPTION_KEY")
    return missing


def _load_state_file() -> str | None:
    try:
        value = state_file.read_text(encoding="utf-8").strip()
        return value or None
    except FileNotFoundError:
        return None


def _save_state_file(value: str) -> None:
    state_file.write_text(value, encoding="utf-8")


def _render_link(label: str, href: str) -> None:
    st.markdown(
        f'<a href="{href}" target="_self" class="stLinkButton">{label}</a>',
        unsafe_allow_html=True,
    )


def _strava_service() -> StravaService | None:
    missing = _strava_missing_config()
    athlete_id = _first_athlete_id()
    if missing or not athlete_id:
        return None
    try:
        return StravaService(storage=storage, config=cfg)
    except Exception as exc:  # pragma: no cover - defensive guard for encryption issues
        st.error(f"Impossible d'initialiser Strava: {exc}")
        return None


strava_service = _strava_service()
athlete_id = _first_athlete_id()
metrics_service = MetricsComputationService(storage)

existing_settings = settings_repo.get("coach-1") or {}
distance_eq_default = coerce_float(existing_settings.get("distanceEqFactor"), 0.01)
strava_days_default = coerce_int(existing_settings.get("stravaSyncDays"), 14)
bike_eq_dist_default = coerce_float(existing_settings.get("bikeEqDistance"), 0.3)
bike_eq_ascent_default = coerce_float(existing_settings.get("bikeEqAscent"), 0.02)
bike_eq_descent_default = coerce_float(existing_settings.get("bikeEqDescent"), 0.0)
ski_eq_dist_default = coerce_float(existing_settings.get("skiEqDistance"), 1.0)
ski_eq_ascent_default = coerce_float(existing_settings.get("skiEqAscent"), distance_eq_default)
ski_eq_descent_default = coerce_float(existing_settings.get("skiEqDescent"), 0.0)
n_cluster_default = max(2, coerce_int(existing_settings.get("nCluster"), cfg.n_cluster))
hr_zone_count_default = max(2, min(5, coerce_int(existing_settings.get("hrZoneCount"), 5)))
hr_zone_window_days_default = max(1, coerce_int(existing_settings.get("hrZoneWindowDays"), 90))


def _float_changed(old: float, new: float) -> bool:
    return not math.isclose(float(old), float(new), rel_tol=1e-9, abs_tol=1e-9)


def _recompute_zone_artifacts(
    *,
    athlete_id: str | None,
    n_cluster: int,
    hr_zone_count: int,
    hr_zone_window_days: int,
) -> tuple[int, int, int]:
    if not athlete_id:
        return 0, 0, 0

    athlete_activities = metrics_service.activities.list(athleteId=athlete_id)
    if athlete_activities.empty or "activityId" not in athlete_activities.columns:
        return 0, 0, 0

    activity_ids = sorted(
        {
            str(activity_id)
            for activity_id in athlete_activities["activityId"].dropna().astype(str).tolist()
            if str(activity_id).strip()
        }
    )
    if not activity_ids:
        return 0, 0, 0

    # Refresh activity metrics first so hrSpeedShift and source rows are up-to-date.
    metrics_service.recompute_for_activities(activity_ids)

    speed_profile_service = SpeedProfileService(cfg)
    ts_service = TimeseriesService(cfg)
    hr_zones_service = HrZonesService(
        storage=storage,
        ts_service=ts_service,
        speed_profile_service=speed_profile_service,
        zone_count=hr_zone_count,
        window_days=hr_zone_window_days,
    )

    cluster_recomputed = 0
    for activity_id in activity_ids:
        result = speed_profile_service.process_timeseries(
            activity_id,
            strategy="cluster",
            n_clusters=int(n_cluster),
        )
        if result is not None:
            speed_profile_service.save_metrics_ts(activity_id, result)
            cluster_recomputed += 1
        speed_profile_service.compute_and_save_elevation_metrics(activity_id)

    borders_recomputed = hr_zones_service.backfill_all_borders(athlete_id=athlete_id)
    return len(activity_ids), cluster_recomputed, borders_recomputed


st.subheader("Coach Settings")
units = st.selectbox("Units", ["metric"], index=0, help="Metric units only in MVP")

st.markdown("#### Paramètres DistEq")
dist_cols = st.columns(7)
with dist_cols[0]:
    st.caption("Trail D+")
    distance_eq = st.number_input(
        "Trail D+",
        min_value=0.0,
        max_value=0.1,
        value=float(distance_eq_default),
        step=0.001,
        format="%.3f",
        label_visibility="collapsed",
        key="dist_eq_factor",
    )
with dist_cols[1]:
    st.caption("Bike dist")
    bike_eq_distance = st.number_input(
        "Bike dist",
        min_value=0.0,
        max_value=5.0,
        value=float(bike_eq_dist_default),
        step=0.01,
        format="%.2f",
        label_visibility="collapsed",
        key="bike_eq_distance",
    )
with dist_cols[2]:
    st.caption("Bike D+")
    bike_eq_ascent = st.number_input(
        "Bike D+",
        min_value=0.0,
        max_value=1.0,
        value=float(bike_eq_ascent_default),
        step=0.001,
        format="%.3f",
        label_visibility="collapsed",
        key="bike_eq_ascent",
    )
with dist_cols[3]:
    st.caption("Bike D-")
    bike_eq_descent = st.number_input(
        "Bike D-",
        min_value=0.0,
        max_value=1.0,
        value=float(bike_eq_descent_default),
        step=0.001,
        format="%.3f",
        label_visibility="collapsed",
        key="bike_eq_descent",
    )
with dist_cols[4]:
    st.caption("Ski dist")
    ski_eq_distance = st.number_input(
        "Ski dist",
        min_value=0.0,
        max_value=5.0,
        value=float(ski_eq_dist_default),
        step=0.01,
        format="%.2f",
        label_visibility="collapsed",
        key="ski_eq_distance",
    )
with dist_cols[5]:
    st.caption("Ski D+")
    ski_eq_ascent = st.number_input(
        "Ski D+",
        min_value=0.0,
        max_value=1.0,
        value=float(ski_eq_ascent_default),
        step=0.001,
        format="%.3f",
        label_visibility="collapsed",
        key="ski_eq_ascent",
    )
with dist_cols[6]:
    st.caption("Ski D-")
    ski_eq_descent = st.number_input(
        "Ski D-",
        min_value=0.0,
        max_value=1.0,
        value=float(ski_eq_descent_default),
        step=0.001,
        format="%.3f",
        label_visibility="collapsed",
        key="ski_eq_descent",
    )

st.markdown("#### Paramètres zones HR")
zone_cols = st.columns(3)
with zone_cols[0]:
    st.caption("n_cluster")
    n_cluster = st.number_input(
        "n_cluster",
        min_value=2,
        max_value=10,
        value=int(n_cluster_default),
        step=1,
        label_visibility="collapsed",
        key="n_cluster_value",
    )
with zone_cols[1]:
    st.caption("n_zones")
    hr_zone_count = st.number_input(
        "n_zones",
        min_value=2,
        max_value=5,
        value=int(hr_zone_count_default),
        step=1,
        label_visibility="collapsed",
        key="hr_zone_count_value",
    )
with zone_cols[2]:
    st.caption("window_size (jours)")
    hr_zone_window_days = st.number_input(
        "window_size",
        min_value=7,
        max_value=365,
        value=int(hr_zone_window_days_default),
        step=1,
        label_visibility="collapsed",
        key="hr_zone_window_days_value",
    )

strava_sync_days = st.number_input(
    "Jours à synchroniser avec Strava",
    min_value=1,
    value=int(strava_days_default),
    step=1,
    help="Nombre de jours remontés lors d'une synchronisation manuelle Strava.",
)
if int(strava_sync_days) > 31:
    st.warning(
        "Plus de 31 jours sélectionnés. L'API Strava peut appliquer des limites de débit; "
        "la synchronisation peut être plus lente et incomplète si la fenêtre est très large."
    )

btn_save, btn_recompute_metrics, btn_recompute_zones = st.columns(3)
save_settings = btn_save.button("Save Settings", use_container_width=True)
recompute_metrics = btn_recompute_metrics.button(
    "Recompute weekly & daily metrics",
    use_container_width=True,
)
recompute_zones = btn_recompute_zones.button("Recompute zones", use_container_width=True)

if save_settings:
    payload = {
        "coachId": "coach-1",
        "units": units,
        "distanceEqFactor": float(distance_eq),
        "stravaSyncDays": int(strava_sync_days),
        "analyticsActivityTypes": existing_settings.get("analyticsActivityTypes", ""),
        "bikeEqDistance": float(bike_eq_distance),
        "bikeEqAscent": float(bike_eq_ascent),
        "bikeEqDescent": float(bike_eq_descent),
        "skiEqDistance": float(ski_eq_distance),
        "skiEqAscent": float(ski_eq_ascent),
        "skiEqDescent": float(ski_eq_descent),
        "nCluster": int(n_cluster),
        "hrZoneCount": int(hr_zone_count),
        "hrZoneWindowDays": int(hr_zone_window_days),
    }
    categories_to_recompute: set[str] = set()
    if _float_changed(distance_eq_default, payload["distanceEqFactor"]):
        categories_to_recompute.update(["RUN", "TRAIL_RUN", "HIKE"])
    if any(
        _float_changed(old, new)
        for old, new in (
            (bike_eq_dist_default, payload["bikeEqDistance"]),
            (bike_eq_ascent_default, payload["bikeEqAscent"]),
            (bike_eq_descent_default, payload["bikeEqDescent"]),
        )
    ):
        categories_to_recompute.add("RIDE")
    if any(
        _float_changed(old, new)
        for old, new in (
            (ski_eq_dist_default, payload["skiEqDistance"]),
            (ski_eq_ascent_default, payload["skiEqAscent"]),
            (ski_eq_descent_default, payload["skiEqDescent"]),
        )
    ):
        categories_to_recompute.add("BACKCOUNTRY_SKI")
    settings_repo.update("coach-1", payload)
    existing_settings.update(payload)
    if categories_to_recompute:
        metrics_service.recompute_for_categories(sorted(categories_to_recompute))
        st.success(
            "Settings saved. Recalcul DistEq appliqué à: "
            + ", ".join(sorted(categories_to_recompute))
        )
    else:
        st.success("Settings saved")

if recompute_metrics:
    metrics_service.recompute_all()
    st.success("Métriques recalculées.")

if recompute_zones:
    if not athlete_id:
        st.warning("Aucun athlète disponible pour recalculer les zones.")
    else:
        with st.spinner("Recalcul des clusters et zones HR..."):
            total, clusters, borders = _recompute_zone_artifacts(
                athlete_id=athlete_id,
                n_cluster=int(n_cluster),
                hr_zone_count=int(hr_zone_count),
                hr_zone_window_days=int(hr_zone_window_days),
            )
        st.success(
            f"Zones recalculées pour {borders} activité(s) "
            f"(clusters recalculés: {clusters}/{total})."
        )

if "strava_state" not in st.session_state:
    persisted = _load_state_file()
    if persisted:
        st.session_state["strava_state"] = persisted
        print(f"[Strava UI] Restored state {persisted}")
    else:
        st.session_state["strava_state"] = new_id()
        _save_state_file(st.session_state["strava_state"])
        print(f"[Strava UI] Generated state {st.session_state['strava_state']}")

params = st.query_params


def _qp_first(key: str, default: str = "") -> str:
    raw = params.get(key, default)
    if isinstance(raw, list):
        if not raw:
            return default
        return str(raw[0])
    if raw is None:
        return default
    return str(raw)

if strava_service and athlete_id:
    if "code" in params:
        returned_state = _qp_first("state", "")
        expected_state = st.session_state.get("strava_state") or _load_state_file()
        if expected_state and returned_state == expected_state:
            code = _qp_first("code", "")
            try:
                strava_service.exchange_code(athlete_id, code)
                st.session_state["strava_flash"] = (
                    "success",
                    "Compte Strava connecté avec succès. Vous pouvez lancer une synchronisation",
                )
                st.session_state["strava_state"] = new_id()
                _save_state_file(st.session_state["strava_state"])
            except Exception as exc:  # pragma: no cover - runtime API failures
                st.session_state["strava_flash"] = (
                    "error",
                    f"Échec de l'échange du code Strava : {exc}",
                )
        else:
            print(
                "[Strava UI] State mismatch",
                {
                    "expected": expected_state,
                    "returned": returned_state,
                    "query_params": params,
                },
            )
            st.session_state["strava_flash"] = (
                "error",
                "Requête Strava invalide (state différent). Merci de réessayer la connexion.",
            )
        params.clear()
        trigger_rerun()
    elif "error" in params:
        description = _qp_first("error_description", "") or _qp_first("message", "")
        st.session_state["strava_flash"] = (
            "error",
            f"Strava a refusé l'autorisation : {description or _qp_first('error', '')}",
        )
        params.clear()
        trigger_rerun()

flash = st.session_state.pop("strava_flash", None)

st.subheader("Integrations")

if flash:
    level, message = flash
    if level == "success":
        st.success(message)
    elif level == "warning":
        st.warning(message)
    else:
        st.error(message)

st.markdown("### Strava")
missing_config = _strava_missing_config()

if missing_config:
    env_list = ", ".join(missing_config)
    st.info(
        "Configurez les variables d'environnement manquantes pour Strava avant de poursuivre : "
        f"{env_list}."
    )
elif not athlete_id:
    st.warning("Ajoutez un athlète avant de connecter Strava.")
elif not strava_service:
    st.error("Strava n'est pas disponible. Vérifiez la configuration.")
else:
    token_df = tokens_repo.list(athleteId=athlete_id)
    token_df = token_df[token_df.get("provider") == "strava"] if not token_df.empty else token_df
    connected = not token_df.empty
    state = st.session_state.get("strava_state")
    auth_url = strava_service.authorization_url(state=state, approval_prompt="auto")
    print("[Strava UI] Authorization URL", auth_url)

    if connected:
        row = token_df.iloc[0]
        expires_raw = row.get("expiresAt")
        expires_label = None
        if expires_raw:
            try:
                expires_dt = dt.datetime.fromtimestamp(float(expires_raw), tz=dt.timezone.utc)
                expires_label = expires_dt.astimezone().strftime("%d/%m/%Y %H:%M")
            except Exception:
                expires_label = str(expires_raw)
        st.success("Compte Strava connecté.")
        if expires_label:
            st.caption(f"Expiration du token : {expires_label}")
        sync_days = coerce_int(existing_settings.get("stravaSyncDays"), int(strava_sync_days))
        if int(sync_days) > 31:
            st.warning(
                "Fenêtre > 31 jours: attention aux limites de l'API Strava."
            )
        if st.button(f"Synchroniser les {sync_days} derniers jours", type="primary"):
            try:
                # Preview to estimate API cost and warn if needed
                try:
                    preview = strava_service.preview_sync_last_n_days(athlete_id, sync_days)
                    missing = int(preview.get("missing_raw", 0))
                    est_req = int(preview.get("est_total_requests", 0))
                    waits = int(preview.get("est_additional_waits", 0))
                    hits_daily = bool(preview.get("est_hits_daily_limit", False))
                    if missing > 100:
                        msg = (
                            f"{missing} activités à télécharger (~{est_req} requêtes). "
                            f"Cela peut nécessiter {waits} attente(s) de 15 minutes"
                        )
                        if hits_daily:
                            msg += ", et potentiellement atteindre la limite quotidienne (1000)."
                        st.warning(msg)
                except Exception:
                    # Best-effort preview; continue even if it fails
                    pass

                imported = strava_service.sync_last_n_days(athlete_id, sync_days)
                stats = getattr(strava_service, "last_sync_stats", {}) or {}
                downloaded = int(stats.get("downloaded_count", len(imported) if imported else 0))
                from_cache = int(stats.get("created_from_cache_count", 0))
                total_created = int(stats.get("created_rows_count", downloaded + from_cache))
                if downloaded or from_cache:
                    st.success(
                        f"{downloaded} téléchargée(s) + {from_cache} créée(s) depuis le cache "
                        f"sur les {sync_days} derniers jours (total {total_created})."
                    )
                else:
                    st.info(f"Aucune nouvelle activité sur les {sync_days} derniers jours.")
                # Persist lightweight summary for display
                st.session_state["strava_last_sync"] = {
                    "days": stats.get("days", sync_days),
                    "downloaded_count": downloaded,
                    "created_from_cache_count": from_cache,
                    "created_rows_count": total_created,
                    # Keep IDs but cap display later for readability
                    "downloaded_ids": list(stats.get("downloaded_ids", [])),
                    "created_from_cache_ids": list(stats.get("created_from_cache_ids", [])),
                }
            except Exception as exc:  # pragma: no cover - runtime API failures
                st.error(f"La synchronisation Strava a échoué : {exc}")
        if st.button(
            "Reconstruire les activités depuis le cache Strava", key="strava-rebuild-cache"
        ):
            try:
                # Create a status container for progress tracking
                status_container = st.status("Reconstruction en cours...", expanded=True)
                
                with status_container:
                    st.write("Reconstruction des activités depuis le cache Strava...")
                    
                    # Create progress elements inside the status container
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    current_activity = [None]
                    total_activities = [0]
                    
                    def update_progress(current: int, total: int, activity_name: str) -> None:
                        """Update progress bar and status text."""
                        total_activities[0] = total
                        current_activity[0] = activity_name
                        if total > 0:
                            progress = current / total
                            progress_bar.progress(progress)
                            status_text.text(f"Traitement des métriques timeseries:"
                                f"{current}/{total} - {activity_name}")
                        else:
                            status_text.text("Préparation de la reconstruction...")
                    
                    rebuilt = strava_service.rebuild_from_cache(athlete_id,
                        progress_callback=update_progress)
                    
                    if total_activities[0] > 0:
                        progress_bar.progress(1.0)
                        status_text.text(f"Terminé ! {total_activities[0]} activité(s) traitées.")
                    
                    st.write(f"✓ {len(rebuilt)} activité(s) recréée(s) depuis le cache Strava.")
                
                st.success(f"{len(rebuilt)} activité(s) recréée(s) depuis le cache Strava.")
                status_container.update(label="Reconstruction terminée", state="complete")
            except Exception as exc:  # pragma: no cover - runtime API failures
                st.error(f"Reconstruction depuis le cache impossible : {exc}")
        _render_link("Gérer l'autorisation Strava", auth_url)
        # Détails récents des appels API
        with st.expander("Détails appels API Strava (récents)"):
            try:
                recent = strava_service.get_rate_log(limit=8)
            except Exception:
                recent = []
            if not recent:
                st.caption("Aucun appel récent enregistré.")
            else:
                for e in recent:
                    ts = e.get("ts", "")
                    method = e.get("method", "")
                    endpoint = e.get("endpoint", "")
                    status = e.get("status", "")
                    usage_s = e.get("usage_short")
                    usage_d = e.get("usage_daily")
                    extra = (f" · usage15={usage_s}" if usage_s is not None else "") + (
                        f" · usageJour={usage_d}" if usage_d is not None else ""
                    )
                    st.caption(f"{ts} · {method} {endpoint} → {status}{extra}")
        # API rate status
        try:
            rate = strava_service.get_rate_status()
        except Exception:
            rate = {}
        if rate:
            short_used = rate.get("short_used")
            short_limit = rate.get("short_limit")
            daily_used = rate.get("daily_used")
            daily_limit = rate.get("daily_limit")
            wait_sec = int(rate.get("wait_seconds") or 0)
            if short_used is not None and short_limit is not None:
                st.caption(f"API Strava (15 min): {short_used}/{short_limit}")
            if daily_used is not None and daily_limit is not None:
                st.caption(f"API Strava (jour): {daily_used}/{daily_limit}")
            if wait_sec > 0:
                minutes = wait_sec // 60
                seconds = wait_sec % 60
                st.warning(f"Fenêtre 15 min saturée. Réessaie dans ~{minutes:02d}:{seconds:02d}.")
        # Last sync summary (if any)
        last_sync = st.session_state.get("strava_last_sync")
        if last_sync:
            st.caption("Résumé de la dernière synchronisation")
            dl = int(last_sync.get("downloaded_count", 0))
            fc = int(last_sync.get("created_from_cache_count", 0))
            total = int(last_sync.get("created_rows_count", dl + fc))
            days = int(last_sync.get("days", sync_days))
            st.info(
                f"{dl} téléchargée(s), {fc} depuis cache · fenêtre: {days}j"
                f"· total lignes créées: {total}."
            )
            # Show a concise list of IDs for quick inspection (trim if long)
            max_ids = 6
            downloaded_ids = [str(x) for x in (last_sync.get("downloaded_ids") or [])][:max_ids]
            cached_ids = [str(x) for x in (last_sync.get("created_from_cache_ids") or [])][:max_ids]
            if downloaded_ids:
                st.caption(
                    "IDs téléchargées: " + ", ".join(downloaded_ids) + ("…" if dl > max_ids else "")
                )
            if cached_ids:
                st.caption(
                    "IDs depuis cache: " + ", ".join(cached_ids) + ("…" if fc > max_ids else "")
                )
    else:
        st.warning("Aucun compte Strava connecté.")
        _render_link("Connecter Strava", auth_url)
        st.caption(
            "Après l'autorisation Strava, vous reviendrez sur cette page"
        )

st.markdown("### Garmin")
st.info("Connexion Garmin à venir.")
