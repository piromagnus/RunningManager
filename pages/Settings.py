import datetime as dt
from pathlib import Path

import streamlit as st

from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo, SettingsRepo, TokensRepo
from services.strava_service import StravaService
from utils.config import load_config
from utils.formatting import set_locale
from utils.ids import new_id


st.set_page_config(page_title="Running Manager - Settings")
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


def _trigger_rerun() -> None:
    rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun:
        rerun()


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


st.subheader("Coach Settings")
units = st.selectbox("Units", ["metric"], index=0, help="Metric units only in MVP")
distance_eq = st.number_input(
    "Distance-eq factor (km per meter ascent)", min_value=0.0, max_value=0.1, value=0.01, step=0.001,
    help="Default: 0.01 (100 m ascent = 1.0 km)"
)

if st.button("Save Settings"):
    settings_repo.update(
        "coach-1",
        {"coachId": "coach-1", "units": units, "distanceEqFactor": distance_eq},
    )
    st.success("Settings saved")

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
                    "Compte Strava connecté avec succès. Vous pouvez lancer une synchronisation manuelle.",
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
        _trigger_rerun()
    elif "error" in params:
        description = params.get("error_description", params.get("message", [""]))[0]
        st.session_state["strava_flash"] = (
            "error",
            f"Strava a refusé l'autorisation : {description or params['error'][0]}",
        )
        st.experimental_set_query_params()
        _trigger_rerun()

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
        if st.button("Synchroniser les 14 derniers jours", type="primary"):
            try:
                imported = strava_service.sync_last_14_days(athlete_id)
                if imported:
                    st.success(f"{len(imported)} activité(s) importée(s).")
                else:
                    st.info("Aucune nouvelle activité sur les 14 derniers jours.")
            except Exception as exc:  # pragma: no cover - runtime API failures
                st.error(f"La synchronisation Strava a échoué : {exc}")
        _render_link("Gérer l'autorisation Strava", auth_url)
    else:
        st.warning("Aucun compte Strava connecté.")
        _render_link("Connecter Strava", auth_url)
        st.caption(
            "Après l'autorisation Strava, vous reviendrez sur cette page pour finaliser l'enregistrement du token."
        )

st.markdown("### Garmin")
st.info("Connexion Garmin à venir.")
