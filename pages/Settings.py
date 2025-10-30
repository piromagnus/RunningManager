"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import datetime as dt
from pathlib import Path

import streamlit as st

from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo, SettingsRepo, TokensRepo
from services.metrics_service import MetricsComputationService
from services.strava_service import StravaService
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


st.subheader("Coach Settings")
units = st.selectbox("Units", ["metric"], index=0, help="Metric units only in MVP")
distance_eq = st.number_input(
    "Distance-eq factor (km per meter ascent)",
    min_value=0.0,
    max_value=0.1,
    value=float(distance_eq_default),
    step=0.001,
    help="Default: 0.01 (100 m ascent = 1.0 km)",
)

st.markdown("#### Équivalences vélo (DistEq)")
bike_eq_distance = st.number_input(
    "Facteur distance (vélo)",
    min_value=0.0,
    max_value=5.0,
    value=float(bike_eq_dist_default),
    step=0.01,
    help="Contribution de la distance pour le vélo (par défaut 0.30).",
)
bike_eq_ascent = st.number_input(
    "Facteur D+ (vélo)",
    min_value=0.0,
    max_value=1.0,
    value=float(bike_eq_ascent_default),
    step=0.001,
    help="Contribution du dénivelé positif pour le vélo (par défaut 0.02).",
)
bike_eq_descent = st.number_input(
    "Facteur D- (vélo)",
    min_value=0.0,
    max_value=1.0,
    value=float(bike_eq_descent_default),
    step=0.001,
    help="Contribution du dénivelé négatif pour le vélo (par défaut 0.00).",
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

if st.button("Save Settings"):
    payload = {
        "coachId": "coach-1",
        "units": units,
        "distanceEqFactor": float(distance_eq),
        "stravaSyncDays": int(strava_sync_days),
        "analyticsActivityTypes": existing_settings.get("analyticsActivityTypes", ""),
        "bikeEqDistance": float(bike_eq_distance),
        "bikeEqAscent": float(bike_eq_ascent),
        "bikeEqDescent": float(bike_eq_descent),
    }
    settings_repo.update("coach-1", payload)
    existing_settings.update(payload)
    st.success("Settings saved")

if st.button("Recompute weekly & daily metrics"):
    metrics_service.recompute_all()
    st.success("Métriques recalculées.")

if "strava_state" not in st.session_state:
    persisted = _load_state_file()
    if persisted:
        st.session_state["strava_state"] = persisted
        print(f"[Strava UI] Restored state {persisted}")
    else:
        st.session_state["strava_state"] = new_id()
        _save_state_file(st.session_state["strava_state"])
        print(f"[Strava UI] Generated state {st.session_state['strava_state']}")

params = st.experimental_get_query_params()

if strava_service and athlete_id:
    if "code" in params:
        returned_state = params.get("state", [""])[0]
        expected_state = st.session_state.get("strava_state") or _load_state_file()
        if expected_state and returned_state == expected_state:
            code = params["code"][0]
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
        st.experimental_set_query_params()
        trigger_rerun()
    elif "error" in params:
        description = params.get("error_description", params.get("message", [""]))[0]
        st.session_state["strava_flash"] = (
            "error",
            f"Strava a refusé l'autorisation : {description or params['error'][0]}",
        )
        st.experimental_set_query_params()
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
