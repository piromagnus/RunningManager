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
    padding: 1.4rem 2.6rem 1.6rem 2.0rem;
    border: 1px solid rgba(228, 204, 160, 0.24);
    box-shadow: 0 18px 36px rgba(8, 14, 24, 0.35);
    overflow: visible !important;
}

.stPlotlyChart > div,
.stVegaLiteChart > div,
.stAltairChart > div {
    width: 100% !important;
    overflow: visible !important;
    box-sizing: border-box;
}

.stPlotlyChart iframe,
.stVegaLiteChart iframe,
.stAltairChart iframe,
.stAltairChart canvas,
.stVegaLiteChart canvas {
    border-radius: 16px;
    box-sizing: border-box;
    width: 100% !important;
    height: 100% !important;
    border: none !important;
    margin: 0 !important;
    display: block !important;
    overflow: visible !important;
}

.block-container {
    padding-top: 1.5rem;
}
</style>
"""

_ALTAIR_THEME_REGISTERED = False


def _enable_altair_theme() -> None:
    global _ALTAIR_THEME_REGISTERED
    if _ALTAIR_THEME_REGISTERED:
        return

    def _theme() -> dict:
        return {
            "config": {
                "padding": {"top": 28, "right": 100, "bottom": 28, "left": 56},
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
                    "padding": 12,
                    "offset": 16,
                    "titlePadding": 6,
                    "labelLimit": 220,
                    "orient": "right",
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
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    _enable_altair_theme()
