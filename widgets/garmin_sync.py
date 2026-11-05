"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Streamlit controls for importing activities through the Garmin Connect API.
"""

from __future__ import annotations

import streamlit as st

from persistence.csv_storage import CsvStorage
from services.garmin_import_api import GarminImportService
from utils.config import Config


def render_garmin_sync(
    storage: CsvStorage,
    config: Config,
    athlete_id: str | None,
) -> None:
    """Render GarminDB configuration status and the manual activity sync action."""
    if not athlete_id:
        st.warning("Ajoutez un athlète avant de synchroniser Garmin.")
        return

    service = GarminImportService(storage=storage, config=config)
    if not service.is_ready:
        st.info("Renseignez GARMIN_EMAIL et GARMIN_PASSWORD dans le fichier .env puis redémarrez RunningManager.")
        return

    mfa_code = st.text_input(
        "Code MFA Garmin (uniquement si demandé)",
        type="password",
        key="garmin-mfa-code",
    )
    days = st.number_input(
        "Jours à synchroniser avec Garmin",
        min_value=1,
        max_value=365,
        value=14,
        step=1,
        key="garmin-sync-days",
    )
    if st.button(
        f"Synchroniser les {int(days)} derniers jours avec Garmin",
        key="garmin-sync",
        type="primary",
    ):
        try:
            with st.spinner("Synchronisation Garmin en cours..."):
                imported = service.sync_last_n_days(
                    athlete_id,
                    int(days),
                    mfa_code=mfa_code or None,
                )
        except Exception as exc:
            st.error(f"La synchronisation Garmin a échoué : {exc}")
            return

        stats = service.last_sync_stats
        imported_count = len(imported)
        if imported_count:
            st.success(
                f"{imported_count} activité(s) Garmin importée(s), dont "
                f"{stats.get('timeseries_count', 0)} avec données détaillées."
            )
            st.cache_data.clear()
        else:
            st.info(f"Aucune nouvelle activité Garmin sur les {int(days)} derniers jours.")
        if stats.get("invalid_date_count"):
            st.warning(f"{stats['invalid_date_count']} activité(s) ignorée(s) car leur date de début était invalide.")
