from __future__ import annotations

import datetime as dt
from typing import Callable, Optional

import streamlit as st

from utils.config import load_config
from utils.formatting import fmt_decimal, fmt_km, fmt_m, set_locale
from utils.styling import apply_theme
from persistence.csv_storage import CsvStorage
from persistence.repositories import AthletesRepo
from services.activity_feed_service import ActivityFeedItem, ActivityFeedService, PlannedSessionCard
from services.linking_service import LinkingService


st.set_page_config(page_title="Running Manager - Activités", layout="wide")
apply_theme()
st.title("Flux d'activités")

_CARD_STYLES = """
<style>
.planned-strip {display:flex; gap:0.75rem; overflow-x:auto; padding-bottom:0.5rem;}
.planned-card {min-width:220px; border:1px solid rgba(228,204,160,0.35); border-radius:12px; padding:0.75rem; background-color:rgba(41,61,86,0.85);}
.planned-card h4 {font-size:0.95rem; margin:0 0 0.35rem 0; color:#e4cca0;}
.planned-card .secondary {color:#60ac84; font-size:0.8rem;}
.planned-card .metrics {margin-top:0.25rem; font-size:0.85rem; color:#f8fafc;}
.activity-card {border:1px solid rgba(228,204,160,0.25); border-radius:12px; padding:1rem 1rem 0.75rem 1rem; margin-bottom:0.9rem; background:rgba(41,61,86,0.88);}
.activity-card.linked {border:2px solid #60ac84; box-shadow:0 0 0 1px rgba(96,172,132,0.35);}
.activity-card .header {display:flex; justify-content:space-between; align-items:flex-start; gap:0.75rem;}
.activity-card .header h3 {margin:0; font-size:1.15rem; color:#e4cca0;}
.activity-card .header .status {font-size:1.4rem;}
.activity-card .header .status span {display:inline-flex; align-items:center;}
.activity-card .header .status span[title] {cursor:help;}
.activity-card .meta {color:#d4acb4; font-size:0.82rem; margin-bottom:0.75rem;}
.activity-card .metrics {display:flex; flex-wrap:wrap; gap:1.4rem; margin-bottom:0.5rem;}
.metric {display:flex; flex-direction:column;}
.metric .label {font-size:0.72rem; text-transform:uppercase; color:#d4acb4; letter-spacing:0.05em;}
.metric .value {font-size:1.05rem; color:#f8fafc;}
.activity-tag {display:inline-flex; align-items:center; gap:0.4rem; font-size:0.78rem; color:#f8fafc;}
.activity-tag span {background:#04813c; color:#f8fafc; padding:0.25rem 0.5rem; border-radius:8px; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em;}
</style>
"""


def _trigger_rerun() -> None:
    rerun = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun:
        rerun()


def _format_duration(seconds: Optional[float]) -> str:
    if seconds in (None, "", float("nan")):
        return "-"
    total = int(float(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h{minutes:02d}"
    if minutes:
        return f"{minutes}m{secs:02d}"
    return f"{secs}s"


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

    st.markdown(_CARD_STYLES, unsafe_allow_html=True)

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
        preferred_defaults = {"RUN", "TRAIL_RUN", "TRAILRUN", "HIKE", "RIDE", "BIKE"}
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
            help="Filtre les cartes du flux par type d'activité. Vider la sélection pour afficher toutes les activités.",
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
        _render_activity_card(item)

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
                    metrics_parts.append(_format_duration(card.planned_duration_sec))
                if card.planned_ascent_m is not None:
                    metrics_parts.append(fmt_m(card.planned_ascent_m))
                metrics = " • ".join(metrics_parts) if metrics_parts else "—"
                st.markdown(
                    (
                        '<div class="planned-card">'
                        f"<h4>{card.date.strftime('%d/%m/%Y')} · {card.session_type or ''}</h4>"
                        f'<div class="secondary">{card.target_label or "Sans cible"}</div>'
                        f'<div class="metrics">{metrics}</div>'
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                if st.button(
                    "Associer",
                    key=f"link-{card.planned_session_id}",
                    use_container_width=True,
                ):
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
            _format_candidate_label(candidate): candidate.activity_id
            for candidate in candidates
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
    duration = _format_duration(candidate.moving_sec)
    score = (
        f"{fmt_decimal(candidate.match_score * 100, 1)}% de correspondance"
        if candidate.match_score is not None
        else "Score inconnu"
    )
    return f"{label_date} · {distance} · {duration} · {score}"


def _render_activity_card(item: ActivityFeedItem) -> None:
    status_icon = ""
    if item.linked:
        tick = "✅"
        tooltip_text = (
            f"Score de lien : {fmt_decimal(item.match_score, 2)}"
            if item.match_score is not None
            else "Activité liée"
        )
        status_icon = f'<span title="{tooltip_text}">{tick}</span>'

    header_html = (
        '<div class="header">'
        f"<h3>{item.name}</h3>"
        f'<div class="status">{status_icon}</div>'
        "</div>"
    )

    type_label = _format_sport_type(item.sport_type)
    meta_html = (
        '<div class="meta">'
        f"{_format_datetime(item.start_time)}"
        f'<div class="activity-tag"><span>{type_label}</span></div>'
        "</div>"
    )

    metrics = [
        ("Distance", fmt_km(item.distance_km) if item.distance_km is not None else "-"),
        (
            "Durée",
            _format_duration(item.moving_sec)
            if item.moving_sec is not None
            else _format_duration(item.elapsed_sec),
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

    container_class = "activity-card linked" if item.linked else "activity-card"

    st.markdown(
        f'<div class="{container_class}">{header_html}{meta_html}{metrics_html}</div>',
        unsafe_allow_html=True,
    )
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


main()
