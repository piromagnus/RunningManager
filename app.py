"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

import streamlit as st
from streamlit.logger import get_logger

from utils.auth_state import init_session_state
from utils.config import load_config, redact
from utils.formatting import set_locale
from utils.styling import apply_theme

logger = get_logger(__name__)


def render_home() -> None:
    cfg = st.session_state["app_config"]
    st.title("Running Manager")
    st.caption("Use the sidebar to navigate between pages.")

    with st.expander("Environment (sanitized)", expanded=False):
        st.write(
            {
                "DATA_DIR": str(cfg.data_dir),
                "STRAVA_CLIENT_ID": redact(cfg.strava_client_id),
                "STRAVA_REDIRECT_URI": cfg.strava_redirect_uri or "",
            }
        )


def main():
    st.set_page_config(page_title="Running Manager", layout="wide")
    # load_dotenv()
    cfg = load_config()
    # if cfg.mapbox_token:
    #     os.environ.setdefault("MAPBOX_API_KEY", cfg.mapbox_token)
    #     os.environ.setdefault("MAPBOX_TOKEN", cfg.mapbox_token)
    set_locale("fr_FR")
    init_session_state()
    apply_theme()
    st.session_state.setdefault("app_config", cfg)
    st.session_state.setdefault("mapbox_token", cfg.mapbox_token)
    logger.debug("cfg.mapbox: %s", cfg.mapbox_token)

    pages = [
        st.Page(render_home, title="Running Manager"),
        st.Page("pages/Activities.py", title="Activities"),
        st.Page("pages/Activity.py", title="Activity"),
        st.Page("pages/Dashboard.py", title="Dashboard"),
        st.Page("pages/Analytics.py", title="Analytics"),
        st.Page("pages/RacePacing.py", title="Race Pacing"),
        st.Page("pages/Athlete.py", title="Athlete"),
        st.Page("pages/Settings.py", title="Settings"),
    ]
    st.navigation(pages).run()


if __name__ == "__main__":
    main()
