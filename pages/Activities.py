"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import datetime as dt
from html import escape
from typing import Callable, Optional

import streamlit as st

from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo
from services.activity_feed_service import (
    ActivityFeedItem,
    ActivityFeedService,
    PlannedSessionCard,
)
from services.linking_service import LinkingService
from services.metrics_service import MetricsComputationService
from utils.config import load_config
from utils.constants import SESSION_TYPE_LABELS_FR
from utils.formatting import fmt_decimal, fmt_km, fmt_m, format_duration, set_locale
from utils.styling import apply_theme

st.set_page_config(page_title="Running Manager - Activités", layout="wide")
apply_theme()
st.title("Flux d'activités")

def _format_session_type_label(raw: Optional[str]) -> str:
    if not raw:
        return "Session"
    return SESSION_TYPE_LABELS_FR.get(raw.upper(), raw.replace("_", " ").title())


def _escape_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return escape(str(value)).replace("\n", "<br/>")



def _trigger_rerun() -> None:
    rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun:
        rerun()


def _format_datetime(ts: Optional[dt.datetime]) -> str:
    if ts is None:
        return "-"
    display = ts
    try:
        display = ts.tz_convert("Europe/Paris")
    except Exception:
        pass
    return display.strftime("%d/%m/%Y · %H:%M")


def _format_sport_type(raw: str) -> str:
    if not raw:
        return "-"
    normalized = raw.replace("_", " ").replace("-", " ").strip()
    return normalized.capitalize()


def _planned_card_status(session_date: object) -> str:
    if isinstance(session_date, dt.datetime):
        session_date = session_date.date()
    elif isinstance(session_date, str):
        try:
            session_date = dt.date.fromisoformat(session_date)
        except Exception:
            session_date = dt.date.today()
    elif not isinstance(session_date, dt.date):
        session_date = dt.date.today()
    today = dt.date.today()
    if session_date > today:
        return "future"
    if session_date == today:
        return "today"
    week_start = today - dt.timedelta(days=today.weekday())
    if session_date >= week_start:
        return "week"
    return "past"


def _dialog_factory() -> Optional[Callable]:
    if hasattr(st, "dialog"):
        return getattr(st, "dialog")
    if hasattr(st, "experimental_dialog"):
        return getattr(st, "experimental_dialog")
    return None


def main() -> None:
    cfg = load_config()
    set_locale("fr_FR")
    storage = CsvStorage(base_dir=cfg.data_dir)
    ath_repo = AthletesRepo(storage)
    feed_service = ActivityFeedService(storage)
    link_service = LinkingService(storage)
    metrics_service = MetricsComputationService(storage)

    ath_df = ath_repo.list()
    if ath_df.empty:
        st.warning("Aucun athlète disponible.")
        st.stop()

    athlete_options = {
        f"{row.get('name') or 'Sans nom'} ({row.get('athleteId')})": row.get("athleteId")
        for _, row in ath_df.iterrows()
    }
    selected_label = st.selectbox("Athlète", list(athlete_options.keys()))
    athlete_id = athlete_options.get(selected_label)
    if not athlete_id:
        st.stop()

    sport_types = feed_service.available_sport_types(athlete_id)
    active_types: Optional[set[str]]
    if sport_types:
        label_map = {code: _format_sport_type(code) for code in sport_types}
        preferred_defaults = {"RUN", "TRAIL_RUN", "TRAILRUN", "HIKE", "RIDE", "BIKE","BACKCOUNTRY_SKI"}
        default_codes = []
        for code in sport_types:
            normalized = code.replace("_", "").upper()
            if normalized in {p.replace("_", "").upper() for p in preferred_defaults}:
                default_codes.append(code)
        if not default_codes:
            default_codes = sport_types
        default_labels = [label_map[code] for code in default_codes]
        selected_labels = st.multiselect(
            "Types d'activités",
            options=[label_map[code] for code in sport_types],
            default=default_labels,
            help=(
                "Filtre les cartes du flux par type d'activité. Vider la sélection pour "
                "afficher toutes les activités."
            ),
        )
        if selected_labels:
            active_types = {code for code, label in label_map.items() if label in selected_labels}
        else:
            active_types = set(sport_types)
    else:
        active_types = None

    DEFAULT_BATCH = 10
    state_key = "activities_feed_limit"
    if state_key not in st.session_state:
        st.session_state[state_key] = DEFAULT_BATCH

    planned_cards = feed_service.get_unlinked_planned_sessions(
        athlete_id,
        reference_date=dt.date.today(),
        lookback_days=7,
        lookahead_days=21,
        max_items=8,
    )

    if planned_cards:
        st.subheader("Séances planifiées non liées")
        _render_planned_strip(planned_cards, athlete_id, link_service)
    else:
        st.caption("Toutes les séances planifiées récentes sont déjà liées à une activité.")

    batch_limit = st.session_state[state_key]
    raw_feed = feed_service.get_feed(
        athlete_id,
        limit=batch_limit * 3,
        offset=0,
    )
    if active_types:
        filtered = [
            item for item in raw_feed if (item.sport_type or "").strip().upper() in active_types
        ]
    else:
        filtered = raw_feed
    feed_items = filtered[:batch_limit]

    if not feed_items:
        st.info("Aucune activité enregistrée pour cet athlète.")
        st.stop()

    for item in feed_items:
        _render_activity_card(item, metrics_service)

    if len(filtered) > batch_limit:
        if st.button("Charger plus", type="secondary"):
            st.session_state[state_key] += DEFAULT_BATCH
            _trigger_rerun()


def _render_planned_strip(
    cards: list[PlannedSessionCard],
    athlete_id: str,
    link_service: LinkingService,
) -> None:
    max_cols = min(len(cards), 4) or 1
    for idx in range(0, len(cards), max_cols):
        chunk = cards[idx : idx + max_cols]
        columns = st.columns(len(chunk))
        for card, col in zip(chunk, columns):
            with col:
                metrics_parts = []
                if card.planned_distance_km is not None:
                    metrics_parts.append(fmt_km(card.planned_distance_km))
                if card.planned_duration_sec is not None:
                    metrics_parts.append(
                        format_duration(card.planned_duration_sec, include_seconds=True)
                    )
                if card.planned_ascent_m is not None:
                    metrics_parts.append(fmt_m(card.planned_ascent_m))
                metrics = " • ".join(metrics_parts) if metrics_parts else "—"
                status = _planned_card_status(card.date)
                type_key = (card.session_type or "").strip().upper()
                type_label = _format_session_type_label(card.session_type)
                date_label = card.date.strftime("%d/%m/%Y")
                template_title = (card.template_title or "").strip()
                race_name = (card.race_name or "").strip()
                notes_text = (card.notes or "").strip()

                title_text = template_title or type_label
                subtitle_lines: list[str] = []
                if type_key == "RACE":
                    title_text = race_name or template_title or type_label
                    if template_title and template_title != title_text:
                        subtitle_lines.append(template_title)
                    if type_label and type_label not in subtitle_lines:
                        subtitle_lines.append(type_label)
                elif type_key in {"FUNDAMENTAL_ENDURANCE", "LONG_RUN"}:
                    title_text = type_label
                    if template_title and template_title.lower() != title_text.lower():
                        subtitle_lines.append(template_title)
                else:
                    title_text = template_title or type_label
                    if template_title and type_label and template_title != type_label:
                        subtitle_lines.append(type_label)
                    elif not template_title and type_label:
                        subtitle_lines.append(type_label)

                subtitle_html = "".join(
                    f'<div class="secondary">{_escape_text(line)}</div>'
                    for line in subtitle_lines
                    if line
                )
                notes_html = (
                    f'<div class="notes">{_escape_text(notes_text)}</div>' if notes_text else ""
                )
                card_classes = [f"planned-card status-{status}"]
                if type_key == "RACE":
                    card_classes.append("race")
                title_html = _escape_text(title_text) or _escape_text(type_label) or "Session"
                card_class_str = " ".join(card_classes)
                card_html = (
                    f'<div class="{card_class_str}">'
                    f"<h4>{title_html}</h4>"
                    f'<div class="date">{date_label}</div>'
                    f"{subtitle_html}"
                    f"{notes_html}"
                    f'<div class="metrics">{metrics}</div>'
                    "</div>"
                )
                st.markdown(card_html, unsafe_allow_html=True)
                st.markdown(
                    f'<div class="planned-card-button status-{status}">', unsafe_allow_html=True
                )
                clicked = st.button(
                    "Associer",
                    key=f"link-{card.planned_session_id}",
                    use_container_width=True,
                )
                st.markdown("</div>", unsafe_allow_html=True)
                if clicked:
                    _open_link_dialog(card, athlete_id, link_service)


def _open_link_dialog(
    card: PlannedSessionCard,
    athlete_id: str,
    link_service: LinkingService,
) -> None:
    factory = _dialog_factory()
    if not factory:
        st.warning("Version de Streamlit sans prise en charge des dialogues.")
        return

    title = f"Associer la séance du {card.date.strftime('%d/%m/%Y')}"

    @factory(title)
    def _dialog() -> None:
        candidates = link_service.suggest_for_planned_session(
            athlete_id, card.planned_session_id, window_days=14
        )
        if not candidates:
            st.info("Aucune activité candidate dans la fenêtre ±14 jours.")
            return

        options = {
            _format_candidate_label(candidate): candidate.activity_id for candidate in candidates
        }
        selected = st.radio(
            "Activités correspondantes",
            list(options.keys()),
            index=0,
        )
        selected_id = options[selected]

        if st.button("Associer cette activité", type="primary"):
            try:
                link_service.create_link(
                    athlete_id,
                    selected_id,
                    card.planned_session_id,
                    window_days=14,
                )
                st.success("Activité liée avec succès.")
                _trigger_rerun()
            except Exception as exc:
                st.error(f"Impossible de créer le lien : {exc}")

    _dialog()


def _format_candidate_label(candidate) -> str:
    start = candidate.start_time
    label_date = "-"
    if start is not None:
        try:
            label_date = start.tz_convert("Europe/Paris").strftime("%d/%m/%Y %H:%M")
        except Exception:
            label_date = start.strftime("%d/%m/%Y %H:%M")
    distance = fmt_km(candidate.distance_km) if candidate.distance_km is not None else "-"
    duration = format_duration(candidate.moving_sec, include_seconds=True)
    score = (
        f"{fmt_decimal(candidate.match_score * 100, 1)}% de correspondance"
        if candidate.match_score is not None
        else "Score inconnu"
    )
    return f"{label_date} · {distance} · {duration} · {score}"


def _render_activity_card(
    item: ActivityFeedItem, metrics_service: MetricsComputationService
) -> None:
    status_icon = ""
    if item.linked:
        tick = "✅"
        tooltip_text = (
            f"Score de lien : {fmt_decimal(item.match_score, 2)}"
            if item.match_score is not None
            else "Activité liée"
        )
        status_icon = f'<span title="{tooltip_text}">{tick}</span>'

    planned_type_raw = (item.planned_session_type or "").strip()
    planned_type = planned_type_raw.upper()
    planned_type_label = _format_session_type_label(planned_type_raw) if planned_type_raw else ""
    template_title = (item.planned_session_template_title or "").strip()
    race_name = (item.planned_session_race_name or "").strip()
    notes_text = (item.planned_session_notes or "").strip()
    activity_name = (item.name or "").strip()

    title_text = activity_name or template_title or race_name or planned_type_label or "-"

    subtitle_lines: list[str] = []

    def _append_subtitle(text: Optional[str]) -> None:
        if not text:
            return
        candidate = text.strip()
        if candidate and candidate not in subtitle_lines:
            subtitle_lines.append(candidate)

    if planned_type == "RACE":
        _append_subtitle(race_name or template_title)
        _append_subtitle(planned_type_label)
    else:
        if template_title:
            _append_subtitle(template_title)
        elif planned_type_label:
            _append_subtitle(planned_type_label)

    subtitle_html = "".join(
        f'<div class="subtitle-line">{_escape_text(line)}</div>' for line in subtitle_lines if line
    )

    header_html = (
        '<div class="header">'
        '<div class="title-block">'
        f"<h3>{_escape_text(title_text)}</h3>"
        f"{subtitle_html}"
        "</div>"
        f'<div class="status">{status_icon}</div>'
        "</div>"
    )

    type_label = _format_sport_type(item.sport_type)
    meta_lines = [f'<div class="meta-line">{_escape_text(_format_datetime(item.start_time))}</div>']
    if planned_type == "RACE" and race_name and race_name not in subtitle_lines:
        meta_lines.append(
            f'<div class="meta-line race-name">Course : {_escape_text(race_name)}</div>'
        )
    if notes_text:
        meta_lines.append(f'<div class="meta-line notes">{_escape_text(notes_text)}</div>')
    type_tag_html = (
        f'<div class="activity-tag"><span>{_escape_text(type_label)}</span></div>'
        if type_label and type_label != "-"
        else ""
    )
    meta_html = '<div class="meta">' + "".join(meta_lines) + type_tag_html + "</div>"

    metrics = [
        ("Distance", fmt_km(item.distance_km) if item.distance_km is not None else "-"),
        (
            "Durée",
            format_duration(item.moving_sec, include_seconds=True)
            if item.moving_sec is not None
            else format_duration(item.elapsed_sec, include_seconds=True),
        ),
        ("D+", fmt_m(item.ascent_m) if item.ascent_m is not None else "-"),
        ("FC moy", fmt_decimal(item.avg_hr, 0) if item.avg_hr is not None else "-"),
        ("TRIMP", fmt_decimal(item.trimp, 1) if item.trimp is not None else "-"),
        (
            "Dist. équiv.",
            fmt_decimal(item.distance_eq_km, 1) if item.distance_eq_km is not None else "-",
        ),
    ]
    metrics_html = '<div class="metrics">'
    for label, value in metrics:
        metrics_html += (
            '<div class="metric">'
            f'<span class="label">{label}</span>'
            f'<span class="value">{value}</span>'
            "</div>"
        )
    metrics_html += "</div>"

    card_classes = ["activity-card"]
    if item.linked:
        card_classes.append("linked")
    if planned_type == "RACE":
        card_classes.append("race")
    elif planned_type in {"FUNDAMENTAL_ENDURANCE", "LONG_RUN"}:
        card_classes.append("endurance")
    container_class = " ".join(card_classes)

    st.markdown(
        f'<div class="{container_class}">{header_html}{meta_html}{metrics_html}</div>',
        unsafe_allow_html=True,
    )
    col_open, col_recompute = st.columns(2)
    with col_open:
        if st.button(
            "Ouvrir l'activité",
            key=f"open-{item.activity_id}",
            type="secondary",
            use_container_width=True,
        ):
            st.session_state["activity_view_id"] = item.activity_id
            st.session_state["activity_view_athlete"] = item.athlete_id
            qp = st.query_params
            qp["activityId"] = item.activity_id
            qp["athleteId"] = item.athlete_id
            st.switch_page("pages/Activity.py")
    with col_recompute:
        if st.button(
            "Recalculer métriques",
            key=f"recompute-{item.activity_id}",
            use_container_width=True,
        ):
            metrics_service.recompute_single_activity(item.activity_id)
            st.success("Métriques recalculées.")
            _trigger_rerun()


main()
