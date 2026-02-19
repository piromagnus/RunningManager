"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

from typing import Optional

import altair as alt
import pandas as pd

DEFAULT_ZONE_COLORS = [
    "#3b82f6",
    "#22c55e",
    "#eab308",
    "#f97316",
    "#ef4444",
]


def build_zone_domain(zone_count: int) -> list[str]:
    safe_count = max(2, int(zone_count))
    return [f"Z{idx}" for idx in range(1, safe_count + 1)]


def build_zone_colors(zone_count: int) -> list[str]:
    safe_count = max(2, int(zone_count))
    if safe_count <= len(DEFAULT_ZONE_COLORS):
        return DEFAULT_ZONE_COLORS[:safe_count]
    repeated: list[str] = []
    while len(repeated) < safe_count:
        repeated.extend(DEFAULT_ZONE_COLORS)
    return repeated[:safe_count]


def get_zone_scale(zone_count: int = 5) -> alt.Scale:
    return alt.Scale(domain=build_zone_domain(zone_count), range=build_zone_colors(zone_count))


def _with_zone_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "zone_label" not in out.columns and "zone" in out.columns:
        out["zone"] = pd.to_numeric(out["zone"], errors="coerce")
        out["zone_label"] = out["zone"].map(lambda value: f"Z{int(value)}" if pd.notna(value) else None)
    return out


def _resolve_zone_domain(df: pd.DataFrame) -> list[str]:
    if "zone_label" in df.columns:
        labels = df["zone_label"].dropna().astype(str).unique().tolist()
        parsed: list[tuple[int, str]] = []
        for label in labels:
            if not label.startswith("Z"):
                continue
            try:
                parsed.append((int(label[1:]), label))
            except ValueError:
                continue
        if parsed:
            return [label for _, label in sorted(parsed, key=lambda item: item[0])]
    if "zone" in df.columns:
        zones = pd.to_numeric(df["zone"], errors="coerce").dropna().astype(int).unique().tolist()
        if zones:
            return [f"Z{int(zone)}" for zone in sorted(zones)]
    return build_zone_domain(5)


def _zone_scale_from_domain(domain: list[str]) -> alt.Scale:
    return alt.Scale(domain=domain, range=build_zone_colors(len(domain)))


def render_hr_zones_timeseries(df: pd.DataFrame) -> Optional[alt.Chart]:
    if df is None or df.empty:
        return None
    if "minutes" not in df.columns:
        return None
    if "zone" not in df.columns and "zone_label" not in df.columns:
        return None

    working = _with_zone_label(df)
    hr_col = "hr"
    if hr_col not in working.columns:
        hr_col = "hr_smooth" if "hr_smooth" in working.columns else ""
    if not hr_col:
        return None

    working[hr_col] = pd.to_numeric(working[hr_col], errors="coerce")
    working["minutes"] = pd.to_numeric(working["minutes"], errors="coerce")
    working = working.dropna(subset=["minutes", hr_col, "zone_label"])
    if working.empty:
        return None
    zone_domain = _resolve_zone_domain(working)
    zone_scale = _zone_scale_from_domain(zone_domain)

    return (
        alt.Chart(working)
        .mark_point(size=12)
        .encode(
            x=alt.X("minutes:Q", title="Temps (min)"),
            y=alt.Y(f"{hr_col}:Q", title="FC (bpm)"),
            color=alt.Color("zone_label:N", title="Zone", scale=zone_scale),
            tooltip=[
                alt.Tooltip("zone_label:N", title="Zone"),
                alt.Tooltip("minutes:Q", title="Temps (min)", format=".1f"),
                alt.Tooltip(f"{hr_col}:Q", title="FC (bpm)", format=".1f"),
            ],
        )
        .properties(height=260)
    )


def render_zone_time_bars(summary_df: pd.DataFrame) -> Optional[alt.Chart]:
    if summary_df is None or summary_df.empty:
        return None

    working = _with_zone_label(summary_df)
    if "time_seconds" not in working.columns:
        return None
    working["time_seconds"] = pd.to_numeric(working["time_seconds"], errors="coerce").fillna(0.0)
    working["time_minutes"] = working["time_seconds"] / 60.0
    for col in ("hr_mean", "hr_min", "hr_max"):
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
    working = working.sort_values("zone")
    zone_domain = _resolve_zone_domain(working)
    zone_scale = _zone_scale_from_domain(zone_domain)

    return (
        alt.Chart(working)
        .mark_bar()
        .encode(
            y=alt.Y("zone_label:N", title="Zone", sort=zone_domain),
            x=alt.X("time_minutes:Q", title="Temps (min)"),
            color=alt.Color("zone_label:N", title="Zone", scale=zone_scale),
            tooltip=[
                alt.Tooltip("zone_label:N", title="Zone"),
                alt.Tooltip("time_minutes:Q", title="Temps (min)", format=".1f"),
                alt.Tooltip("hr_mean:Q", title="FC moy", format=".1f"),
                alt.Tooltip("hr_min:Q", title="FC min", format=".0f"),
                alt.Tooltip("hr_max:Q", title="FC max", format=".0f"),
            ],
        )
        .properties(height=260)
    )


def render_zone_speed_table(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()

    working = _with_zone_label(summary_df).copy()
    numeric_cols = [
        "time_seconds",
        "hr_mean",
        "hr_min",
        "hr_max",
        "avg_speed_kmh",
        "avg_speedeq_kmh",
    ]
    for col in numeric_cols:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")
        else:
            working[col] = pd.NA

    working["time_minutes"] = working["time_seconds"].fillna(0.0) / 60.0
    working["hr_range"] = working.apply(
        lambda row: (
            f"{row['hr_min']:.0f}-{row['hr_max']:.0f}"
            if pd.notna(row["hr_min"]) and pd.notna(row["hr_max"])
            else "-"
        ),
        axis=1,
    )
    display = (
        working.sort_values("zone")[
            [
                "zone_label",
                "hr_mean",
                "hr_range",
                "time_minutes",
                "avg_speed_kmh",
                "avg_speedeq_kmh",
            ]
        ]
        .rename(
            columns={
                "zone_label": "Zone",
                "hr_mean": "FC moyenne (bpm)",
                "hr_range": "Plage FC (bpm)",
                "time_minutes": "Temps (min)",
                "avg_speed_kmh": "Vitesse moy (km/h)",
                "avg_speedeq_kmh": "Vitesse eq moy (km/h)",
            }
        )
        .reset_index(drop=True)
    )
    return display


def render_weekly_zone_stacked(
    weekly_df: pd.DataFrame,
    *,
    absolute: bool = True,
    chart_width: int = 860,
) -> Optional[alt.Chart]:
    if weekly_df is None or weekly_df.empty:
        return None

    working = _with_zone_label(weekly_df)
    value_col = "time_minutes" if absolute else "pct_time"
    if value_col not in working.columns:
        return None

    working[value_col] = pd.to_numeric(working[value_col], errors="coerce").fillna(0.0)
    zone_domain = _resolve_zone_domain(working)
    zone_scale = _zone_scale_from_domain(zone_domain)
    y_title = "Temps (min)" if absolute else "% du temps hebdo"
    value_format = ".1f" if absolute else ".2f"

    return (
        alt.Chart(working)
        .mark_bar()
        .encode(
            x=alt.X("weekLabel:N", title="Semaine"),
            y=alt.Y(f"{value_col}:Q", title=y_title),
            color=alt.Color("zone_label:N", title="Zone", scale=zone_scale),
            order=alt.Order("zone:Q"),
            tooltip=[
                alt.Tooltip("weekLabel:N", title="Semaine"),
                alt.Tooltip("zone_label:N", title="Zone"),
                alt.Tooltip(f"{value_col}:Q", title=y_title, format=value_format),
                alt.Tooltip("time_minutes:Q", title="Temps (min)", format=".1f"),
                alt.Tooltip("pct_time:Q", title="% temps", format=".2f"),
            ],
        )
        .properties(height=320, width=chart_width)
    )


def render_zone_speed_evolution(
    evolution_df: pd.DataFrame,
    *,
    metric: str = "speed",
    chart_width: int = 860,
) -> Optional[alt.Chart]:
    if evolution_df is None or evolution_df.empty:
        return None

    working = _with_zone_label(evolution_df)
    field = "avg_speed_kmh" if metric == "speed" else "avg_speedeq_kmh"
    if field not in working.columns:
        return None

    if "date" not in working.columns:
        if "weekStartDate" not in working.columns:
            return None
        working["date"] = pd.to_datetime(working["weekStartDate"], errors="coerce")
    else:
        working["date"] = pd.to_datetime(working["date"], errors="coerce")

    working[field] = pd.to_numeric(working[field], errors="coerce")
    working = working.dropna(subset=["date", "zone_label", field]).sort_values("date")
    if working.empty:
        return None

    zone_domain = _resolve_zone_domain(working)
    zone_scale = _zone_scale_from_domain(zone_domain)
    y_title = "Vitesse (km/h)" if metric == "speed" else "Vitesse eq (km/h)"
    base = alt.Chart(working).encode(
        x=alt.X("date:T", title="Semaine"),
        y=alt.Y(f"{field}:Q", title=y_title),
        color=alt.Color("zone_label:N", title="Zone", scale=zone_scale),
        tooltip=[
            alt.Tooltip("date:T", title="Semaine"),
            alt.Tooltip("zone_label:N", title="Zone"),
            alt.Tooltip(f"{field}:Q", title=y_title, format=".2f"),
        ],
    )
    return (base.mark_line() + base.mark_point(size=30)).properties(height=320, width=chart_width)


def render_zone_borders_chart(
    borders_df: pd.DataFrame,
    *,
    chart_width: int = 860,
) -> Optional[alt.Chart]:
    if borders_df is None or borders_df.empty:
        return None
    working = borders_df.copy()
    if "startDate" not in working.columns:
        return None
    working["date"] = pd.to_datetime(working["startDate"], errors="coerce")
    border_cols = [col for col in working.columns if col.startswith("hrZone_z") and col.endswith("_upper")]
    if not border_cols:
        return None
    for col in border_cols:
        working[col] = pd.to_numeric(working[col], errors="coerce")
    long_df = working.melt(
        id_vars=["date"],
        value_vars=border_cols,
        var_name="border_key",
        value_name="hr_border",
    )
    long_df = long_df.dropna(subset=["date", "hr_border"])
    if long_df.empty:
        return None
    long_df["border_idx"] = (
        long_df["border_key"].str.extract(r"hrZone_z(\d+)_upper", expand=False).astype(int)
    )
    long_df["border_label"] = long_df["border_idx"].map(lambda idx: f"Bord Z{idx}/Z{idx + 1}")
    label_domain = [
        f"Bord Z{idx}/Z{idx + 1}" for idx in sorted(long_df["border_idx"].dropna().astype(int).unique())
    ]
    color_range = build_zone_colors(max(2, len(label_domain) + 1))[: len(label_domain)]
    return (
        alt.Chart(long_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("hr_border:Q", title="Bordure FC (bpm)"),
            color=alt.Color(
                "border_label:N",
                title="Frontière",
                scale=alt.Scale(domain=label_domain, range=color_range),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("border_label:N", title="Frontière"),
                alt.Tooltip("hr_border:Q", title="FC (bpm)", format=".1f"),
            ],
        )
        .properties(height=320, width=chart_width)
    )
