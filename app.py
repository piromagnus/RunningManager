import streamlit as st
from utils.config import load_config, redact
from utils.formatting import set_locale, fmt_km, fmt_speed_kmh
from utils.styling import apply_theme
from utils.auth_state import init_session_state



def _ensure_placeholders():
    # Demonstrate formatting without performing heavy imports
    st.write(fmt_km(12.3), "|", fmt_speed_kmh(9.4))


def main():
    st.set_page_config(page_title="Running Manager", layout="wide")
    cfg = load_config()
    set_locale("fr_FR")
    init_session_state()
    apply_theme()

    st.title("Running Manager")
    st.caption("Use the sidebar to navigate between pages.")

    with st.expander("Environment (sanitized)", expanded=False):
        st.write({
            "DATA_DIR": str(cfg.data_dir),
            "STRAVA_CLIENT_ID": redact(cfg.strava_client_id),
            "STRAVA_REDIRECT_URI": cfg.strava_redirect_uri or "",
        })

    _ensure_placeholders()


if __name__ == "__main__":
    main()
