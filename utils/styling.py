"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import altair as alt
import streamlit as st

THEME_CSS = """
<style>
:root {
    --rm-deep-blue: #293d56;
    --rm-maroon: #9e4836;
    --rm-forest: #04813c;
    --rm-emerald: #60ac84;
    --rm-sand: #e4cca0;
    --rm-rose: #d4acb4;
    --rm-text-primary: #f8fafc;
    --rm-text-secondary: #e4cca0;
    --rm-surface: rgba(41, 61, 86, 0.96);
    --rm-surface-alt: rgba(41, 61, 86, 0.88);
    --rm-surface-soft: rgba(20, 32, 48, 0.7);
    --rm-shadow: 0 18px 30px rgba(8, 14, 24, 0.38);
}

html, body, [data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, rgba(41, 61, 86, 0.88), rgba(15, 23, 42, 0.94));
    color: var(--rm-text-primary);
}

[data-testid="stHeader"] {
    background: rgba(9, 16, 28, 0.6);
    backdrop-filter: blur(6px);
    border-bottom: 1px solid rgba(228, 204, 160, 0.2);
}

[data-testid="stSidebar"] > div {
    background: linear-gradient(180deg, rgba(41, 61, 86, 0.95), rgba(17, 24, 39, 0.96));
    color: var(--rm-text-primary);
    border-right: 1px solid rgba(228, 204, 160, 0.2);
}

[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] p {
    color: var(--rm-text-secondary);
}

label, legend {
    color: var(--rm-text-secondary) !important;
}

.stSelectbox > div[data-baseweb="select"],
.stMultiSelect > div[data-baseweb="select"],
.stTextInput > div > div,
.stNumberInput > div > div,
.stDateInput > div > div,
.stTimeInput > div > div {
    color: var(--rm-text-primary);
    background-color: rgba(33, 51, 74, 0.85);
    border: 1px solid rgba(228, 204, 160, 0.25);
    border-radius: 12px;
}

.stSelectbox div[data-baseweb="select"] > div,
.stSelectbox div[data-baseweb="select"] input {
    color: var(--rm-text-primary) !important;
}

.stSelectbox ul[role="listbox"],
.stMultiSelect ul[role="listbox"] {
    background-color: rgba(33, 51, 74, 0.95);
    border: 1px solid rgba(228, 204, 160, 0.25);
}

.stSelectbox li[role="option"],
.stMultiSelect li[role="option"] {
    color: var(--rm-text-primary) !important;
    background-color: rgba(33, 51, 74, 0.85);
}

.stSelectbox li[role="option"]:hover,
.stMultiSelect li[role="option"]:hover {
    background-color: rgba(96, 172, 132, 0.2);
    color: var(--rm-text-primary) !important;
}

.stSelectbox > div[data-baseweb="select"]:focus-within,
.stMultiSelect > div[data-baseweb="select"]:focus-within,
.stTextInput > div > div:focus-within,
.stNumberInput > div > div:focus-within,
.stDateInput > div > div:focus-within,
.stTimeInput > div > div:focus-within {
    border-color: rgba(96, 172, 132, 0.9);
    box-shadow: 0 0 0 2px rgba(96, 172, 132, 0.25);
}

.stMultiSelect [data-baseweb="tag"] {
    background: rgba(158, 72, 54, 0.85);
    color: var(--rm-text-primary);
    border: 1px solid rgba(148, 44, 28, 0.5);
    border-radius: 12px;
}

.stButton button {
    background: linear-gradient(135deg, var(--rm-emerald), #2f8a62);
    color: #08141c;
    border: none;
    border-radius: 14px;
    padding: 0.45rem 1.4rem;
    font-weight: 600;
    box-shadow: var(--rm-shadow);
}

.stButton button:hover {
    background: linear-gradient(135deg, #72cfa2, var(--rm-emerald));
    color: #08141c;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 0.4rem;
    background: rgba(9, 16, 28, 0.45);
    padding: 0.35rem;
    border-radius: 14px;
    border: 1px solid rgba(228, 204, 160, 0.18);
}

.stTabs [data-baseweb="tab"] {
    font-weight: 600;
    color: var(--rm-text-secondary);
}

.stTabs [aria-selected="true"] {
    background: rgba(96, 172, 132, 0.2);
    color: var(--rm-text-primary);
    border-radius: 10px;
}

.stMetric {
    background: rgba(12, 20, 33, 0.55);
    border-radius: 16px;
    padding: 0.75rem 1.1rem;
    border: 1px solid rgba(228, 204, 160, 0.18);
    box-shadow: var(--rm-shadow);
}

.stMetric label {
    color: var(--rm-text-secondary) !important;
}

.stMetric [data-testid="stMetricValue"] {
    color: var(--rm-text-primary);
}

.stAlert {
    border-radius: 14px;
    border: 1px solid rgba(228, 204, 160, 0.26) !important;
    background: rgba(41, 61, 86, 0.55) !important;
    color: var(--rm-text-primary);
}

.stExpander {
    background: rgba(18, 28, 41, 0.65);
    border-radius: 12px;
    border: 1px solid rgba(228, 204, 160, 0.22);
    color: var(--rm-text-primary);
}

.stProgress .st-bo {
    background-color: rgba(96, 172, 132, 0.35);
}

.stProgress .st-c0 {
    background-color: var(--rm-emerald);
}

.stSlider > div > div > div {
    background: rgba(96, 172, 132, 0.3);
}

.stSlider > div > div > div[data-baseweb="slider"] > div:last-child {
    background: var(--rm-emerald);
    border: 2px solid rgba(8, 14, 24, 0.6);
}

.stDataFrame {
    background: rgba(18, 28, 41, 0.65);
}

.stDataFrame [data-testid="StyledTable"] {
    border: 1px solid rgba(228, 204, 160, 0.2);
}

.stPlotlyChart, .stVegaLiteChart, .stAltairChart {
    background: rgba(18, 28, 41, 0.72);
    border-radius: 22px;
    padding: 1.0rem 1.0rem 1.2rem;
    border: 1px solid rgba(228, 204, 160, 0.24);
    box-shadow: 0 18px 36px rgba(8, 14, 24, 0.35);
    margin: 1.2rem 0rem;
    width: 100%;
    box-sizing: border-box;
    overflow: visible !important;
}

.stPlotlyChart > div,
.stVegaLiteChart > div,
.stAltairChart > div {
    width: 100%;
    max-width: 100%;
    margin: 0 auto;
    overflow: visible;
    box-sizing: border-box;
}

.stPlotlyChart iframe,
.stVegaLiteChart iframe,
.stAltairChart iframe,
.stAltairChart canvas,
.stVegaLiteChart canvas {
    border-radius: 16px;
    box-sizing: border-box;
    max-width: 100%;
    width: 100% !important;
    height: auto !important;
    border: none !important;
    margin: 0 !important;
    display: block !important;
    overflow: visible !important;
}

.rm-chart-shell {
    width: 100%;
    display: flex;
    justify-content: center;
    box-sizing: border-box;
}

.rm-chart-shell .stAltairChart {
    width: 100%;
    max-width: 100%;
    margin: 0 auto;
    box-sizing: border-box;
}

.rm-chart-shell .stAltairChart > div {
    width: 100% !important;
    box-sizing: border-box;
    overflow: visible !important;
}

.block-container {
    padding-top: 1.5rem;
}
</style>
"""

_ALTAIR_THEME_REGISTERED = False

CARD_STYLES_CSS = """
<style>
.planned-strip {display:flex; gap:0.75rem; overflow-x:auto; padding-bottom:0.5rem;}
.planned-card {min-width:220px; border-radius:12px; padding:0.85rem; box-shadow:0 6px 14px rgba(8,47,73,0.28); border:1px solid transparent;}
.planned-card h4 {font-size:0.98rem; margin:0 0 0.35rem 0; color:#f8fafc;}
.planned-card .secondary {font-size:0.82rem;}
.planned-card .date {font-size:0.78rem; color:#e2e8f0;}
.planned-card .race-name {margin-top:0.25rem; font-size:0.8rem; color:#fcd34d; font-weight:600;}
.planned-card .notes {margin-top:0.3rem; font-size:0.78rem; color:#e2e8f0; opacity:0.9;}
.planned-card .metrics {margin-top:0.35rem; font-size:0.88rem; color:#f8fafc;}
.planned-card.status-future {background:linear-gradient(135deg, rgba(14,116,144,0.95), rgba(8,47,73,0.95)); border-color:rgba(14,165,233,0.5);}
.planned-card.status-future .secondary {color:#a5f3fc;}
.planned-card.status-today {background:linear-gradient(135deg, rgba(22,163,74,0.95), rgba(5,46,22,0.95)); border-color:rgba(34,197,94,0.6);}
.planned-card.status-today .secondary {color:#bbf7d0;}
.planned-card.status-week {background:linear-gradient(135deg, rgba(249,115,22,0.95), rgba(124,45,18,0.95)); border-color:rgba(251,146,60,0.6);}
.planned-card.status-week .secondary {color:#fed7aa;}
.planned-card.status-past {background:linear-gradient(135deg, rgba(220,38,38,0.95), rgba(127,29,29,0.95)); border-color:rgba(248,113,113,0.6);}
.planned-card.status-past .secondary {color:#fecaca;}
.planned-card.race {background:linear-gradient(135deg, rgba(253,224,71,0.92), rgba(202,138,4,0.9)); border-color:rgba(250,204,21,0.65); box-shadow:0 6px 18px rgba(202,138,4,0.35);}
.planned-card.race .secondary {color:#fde68a;}
.planned-card.race .date {color:#fef3c7;}
.planned-card.race .notes {color:#fff9c2;}
.planned-card-button {margin-top:0.55rem;}
.planned-card-button button {width:100%; font-weight:600; border-width:1px;}
.planned-card-button.status-future button {background:#22d3ee; color:#082f49; border-color:#0ea5e9;}
.planned-card-button.status-future button:hover {background:#0ea5e9; color:#f8fafc;}
.planned-card-button.status-today button {background:#22c55e; color:#052e16; border-color:#16a34a;}
.planned-card-button.status-today button:hover {background:#16a34a; color:#f0fdf4;}
.planned-card-button.status-week button {background:#fb923c; color:#451a03; border-color:#f97316;}
.planned-card-button.status-week button:hover {background:#f97316; color:#fff7ed;}
.planned-card-button.status-past button {background:#ef4444; color:#450a0a; border-color:#dc2626;}
.planned-card-button.status-past button:hover {background:#dc2626; color:#fef2f2;}
.activity-card {border:1px solid rgba(228,204,160,0.25); border-radius:12px; padding:1rem 1rem 0.75rem 1rem; margin-bottom:0.9rem; background:rgba(41,61,86,0.88);}
.activity-card.linked {border:2px solid #60ac84; box-shadow:0 0 0 1px rgba(96,172,132,0.35);}
.activity-card .header {display:flex; justify-content:space-between; align-items:flex-start; gap:0.75rem;}
.activity-card .header .title-block {display:flex; flex-direction:column; gap:0.1rem;}
.activity-card .header h3 {margin:0; font-size:1.15rem; color:#e4cca0;}
.activity-card .header .subtitle-line {margin-top:0.1rem; font-size:0.9rem; color:#d4acb4;}
.activity-card.race {background:linear-gradient(135deg, rgba(253,224,71,0.92), rgba(202,138,4,0.9)); border:2px solid rgba(250,204,21,0.65); box-shadow:0 6px 18px rgba(202,138,4,0.35);}
.activity-card.race .header h3 {color:#fef3c7;}
.activity-card.race .header .subtitle-line {color:#fde68a;}
.activity-card.race .meta .meta-line {color:#fff7d6;}
.activity-card.race .meta .notes {color:#fff9c2;}
.activity-card .header .status {font-size:1.4rem;}
.activity-card .header .status span {display:inline-flex; align-items:center;}
.activity-card .header .status span[title] {cursor:help;}
.activity-card .meta {color:#d4acb4; font-size:0.82rem; margin-bottom:0.75rem; display:flex; flex-direction:column; gap:0.35rem;}
.activity-card .meta .meta-line {display:flex; align-items:center; gap:0.35rem;}
.activity-card .meta .activity-name {color:#e4cca0; font-weight:600;}
.activity-card .meta .race-name {color:#fcd34d; font-weight:600;}
.activity-card .meta .notes {color:#e2e8f0; font-size:0.78rem; opacity:0.9;}
.activity-card .metrics {display:flex; flex-wrap:wrap; gap:1.4rem; margin-bottom:0.5rem;}
.metric {display:flex; flex-direction:column;}
.metric .label {font-size:0.72rem; text-transform:uppercase; color:#d4acb4; letter-spacing:0.05em;}
.metric .value {font-size:1.05rem; color:#f8fafc;}
.activity-tag {display:inline-flex; align-items:center; gap:0.4rem; font-size:0.78rem; color:#f8fafc;}
.activity-tag span {background:#04813c; color:#f8fafc; padding:0.25rem 0.5rem; border-radius:8px; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.05em;}
.rm-card {padding:10px 12px; border:1px solid rgba(255,255,255,0.08); border-radius:8px; margin-bottom:10px; background:rgba(0,0,0,0.15);}
.rm-card-header {font-size:0.95rem; font-weight:600; margin-bottom:4px;}
.rm-card-meta {font-size:0.8rem; color:rgba(255,255,255,0.7); margin-bottom:4px;}
.rm-card-section {background:rgba(255,255,255,0.04); border-radius:6px; padding:6px 8px; margin-top:6px;}
.rm-card-section-title {font-size:0.75rem; font-weight:600; text-transform:uppercase; margin-bottom:2px;}
.rm-card-section-body {font-size:0.8rem; line-height:1.3;}
.rm-card-actions {display:flex; gap:6px; margin-top:6px;}
</style>
"""

INTERVAL_EDITOR_CSS = """
<style>
.rm-loop-card {background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:8px 10px; margin-bottom:10px;}
.rm-interval-action {background:rgba(255,255,255,0.02); border-radius:6px; padding:6px 8px; margin-bottom:6px;}
.rm-interval-editor .stNumberInput label,
.rm-interval-editor .stSelectbox label,
.rm-interval-editor .stTextInput label {font-size:0.75rem;}
</style>
"""


def _enable_altair_theme() -> None:
    global _ALTAIR_THEME_REGISTERED
    if _ALTAIR_THEME_REGISTERED:
        return

    def _theme() -> dict:
        return {
            "config": {
                "padding": 10,
                "autosize": {"type": "fit", "contains": "padding"},
                "background": "rgba(18, 28, 41, 0.01)",
                "view": {
                    "continuousWidth": 400,
                    "continuousHeight": 300,
                    "strokeWidth": 0,
                    "fill": "rgba(18, 28, 41, 0.8)",
                },
                "axis": {
                    "labelColor": "#f4f7fb",
                    "titleColor": "#e4cca0",
                    "gridColor": "rgba(96, 172, 132, 0.15)",
                    "domainColor": "rgba(228, 204, 160, 0.25)",
                    "tickColor": "rgba(228, 204, 160, 0.25)",
                },
                "legend": {
                    "labelColor": "#f4f7fb",
                    "titleColor": "#e4cca0",
                    "padding": 4,
                    "offset": 5,
                    "titlePadding": 2,
                    "labelLimit": 120,
                    "orient": "right",
                    "direction": "vertical",
                    "titleLimit": 120,
                },
                "title": {
                    "color": "#e4cca0",
                    "fontSize": 16,
                    "fontWeight": 600,
                },
                "range": {
                    "category": [
                        "#60ac84",
                        "#9e4836",
                        "#04813c",
                        "#d4acb4",
                        "#e4cca0",
                        "#293d56",
                    ],
                },
                "mark": {
                    "color": "#60ac84",
                },
            }
        }

    alt.themes.register("running_manager_theme", _theme)
    alt.themes.enable("running_manager_theme")
    _ALTAIR_THEME_REGISTERED = True


def apply_theme() -> None:
    """Inject global CSS theme for Streamlit pages."""
    st.markdown(THEME_CSS + CARD_STYLES_CSS + INTERVAL_EDITOR_CSS, unsafe_allow_html=True)
    _enable_altair_theme()
