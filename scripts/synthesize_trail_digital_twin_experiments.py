"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Synthesize trail digital twin experiment leaderboards into a markdown report.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_EXP_DIR = REPO_ROOT / "data" / "exp_perf_predictions"
DEFAULT_OUTPUT = (
    REPO_ROOT / "docs" / "science" / "trail_digital_twin_experiment_synthesis_report.md"
)

PHYSIOLOGY_COLS = [
    "hrrReference",
    "hrrMinFactor",
    "hrrMaxFactor",
    "decayLambda",
    "minFatigueFactor",
    "ctlWeight",
    "tsbWeight",
]

METRIC_COLS = [
    "meanStage3MaeMin",
    "meanStage3MapePct",
    "meanStage3R2",
    "meanAbsBiasMin",
]

HYPOTHESIS_FOLDERS = {
    "trail_digital_twin_hypothesis_screen",
    "trail_digital_twin_hypothesis_refined",
    "trail_digital_twin_minetti075_probe",
}

HISTORICAL_ORDER = [
    "trail_digital_twin_reference_loo",
    "trail_digital_twin_best_loo",
    "trail_digital_twin_low_ref_screen",
    "trail_digital_twin_refine_screen",
    "trail_digital_twin_wide_screen",
    "trail_digital_twin_refined_screen",
    "trail_digital_twin_boundary_wide",
    "trail_digital_twin_boundary_refined",
    "trail_digital_twin_boundary_best_profile",
    "trail_digital_twin_best_factor_sweep",
    "trail_digital_twin_hypothesis_screen",
    "trail_digital_twin_hypothesis_refined",
    "trail_digital_twin_minetti075_probe",
]


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt(value: Any, digits: int = 2) -> str:
    parsed = _safe_float(value)
    if parsed is None:
        return "—"
    return f"{parsed:.{digits}f}"


def _count_runs(folder: Path) -> int:
    plan_path = folder / "benchmark_plan.csv"
    if plan_path.exists():
        try:
            plan = pd.read_csv(plan_path)
            if not plan.empty:
                return int(len(plan))
        except Exception:
            pass
    manifest_path = folder / "benchmark_manifest.csv"
    if manifest_path.exists():
        try:
            manifest = pd.read_csv(manifest_path)
            if not manifest.empty:
                return int(len(manifest))
        except Exception:
            pass
    runs_dir = folder / "runs"
    if runs_dir.is_dir():
        return sum(1 for path in runs_dir.iterdir() if path.is_dir())
    return 0


def _best_row(leaderboard: pd.DataFrame) -> pd.Series | None:
    if leaderboard.empty:
        return None
    working = leaderboard.copy()
    if "rank" in working.columns:
        working["rank"] = pd.to_numeric(working["rank"], errors="coerce")
        ranked = working.dropna(subset=["rank"]).sort_values("rank")
        if not ranked.empty:
            return ranked.iloc[0]
    if "meanStage3MaeMin" in working.columns:
        working["meanStage3MaeMin"] = pd.to_numeric(
            working["meanStage3MaeMin"], errors="coerce"
        )
        scored = working.dropna(subset=["meanStage3MaeMin"]).sort_values(
            "meanStage3MaeMin"
        )
        if not scored.empty:
            return scored.iloc[0]
    return working.iloc[0]


def _scan_experiment(folder: Path) -> dict[str, Any] | None:
    leaderboard_path = folder / "benchmark_leaderboard.csv"
    if not leaderboard_path.exists():
        return None
    try:
        leaderboard = pd.read_csv(leaderboard_path)
    except Exception:
        return None
    if leaderboard.empty:
        return None

    best = _best_row(leaderboard)
    if best is None:
        return None

    row: dict[str, Any] = {
        "experimentFolder": folder.name,
        "runCount": _count_runs(folder),
        "leaderboardRows": int(len(leaderboard)),
        "hasHtmlReport": (folder / "trail_digital_twin_benchmark_report.html").exists(),
        "runId": str(best.get("runId", "")),
        "experimentGroup": str(best.get("experimentGroup", "")),
        "experimentName": str(best.get("experimentName", "")),
        "status": str(best.get("status", "")),
        "description": str(best.get("description", "")),
    }
    for col in METRIC_COLS + PHYSIOLOGY_COLS:
        row[col] = _safe_float(best.get(col))
    return row


def scan_experiments(exp_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not exp_dir.is_dir():
        return pd.DataFrame()
    for folder in sorted(exp_dir.iterdir()):
        if not folder.is_dir():
            continue
        scanned = _scan_experiment(folder)
        if scanned is not None:
            rows.append(scanned)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.sort_values(
        "meanStage3MaeMin", ascending=True, na_position="last"
    ).reset_index(drop=True)
    frame["globalRank"] = range(1, len(frame) + 1)
    return frame


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    # Left-align the first column (usually a name/id); right-align metrics.
    align = ["---"] + ["---:"] * (len(headers) - 1)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(align) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _profile_cell(row: pd.Series) -> str:
    parts = [
        f"hrr={_fmt(row.get('hrrReference'), 2)}",
        f"min={_fmt(row.get('hrrMinFactor'), 2)}",
        f"max={_fmt(row.get('hrrMaxFactor'), 2)}",
        f"λ={_fmt(row.get('decayLambda'), 2)}",
        f"floor={_fmt(row.get('minFatigueFactor'), 2)}",
    ]
    return ", ".join(parts)


def build_report(frame: pd.DataFrame) -> str:
    today = date.today().isoformat()
    lines: list[str] = [
        "# Trail Digital Twin Experiment Synthesis Report",
        "",
        f"Generated: {today}",
        "",
        "Cross-experiment summary of Stage 3 leave-one-out activity MAE across",
        "`data/exp_perf_predictions/`. Primary metric is **mean Stage 3 LOO MAE (min)**;",
        "lower is better. Aggregate scores average configured cohorts",
        "(`hardTrailRun`, `hardRunOrTrailRun`, `top10HardTrailByHRR`, `selectedDateRaces`).",
        "",
        "## Definitions",
        "",
        "- **HRR_ref (`hrr_reference`)**: fraction of heart-rate reserve at which the",
        "  runner is expected to sustain the reference speed (VMA / flat threshold proxy).",
        "- **Decay λ (`decay_lambda`)**: exponential decay rate for in-race TRIMP used",
        "  by Stage 3 fatigue states.",
        "- **Min fatigue factor**: floor on the fatigue multiplier so long races cannot",
        "  collapse predicted speed to zero.",
        "- Open modeling question: how to treat elapsed time or segments that include",
        "  recovery / walking without clear HR-effort continuity.",
        "",
        "## Executive Summary",
        "",
        "Operational default for continued building (current-code hypothesis winner):",
        "",
        "| factor | recommended value |",
        "| --- | ---: |",
        "| HRR reference | 0.88 |",
        "| HRR min factor | 0.30 |",
        "| HRR max factor | 1.00 |",
        "| Decay lambda | 0.20 |",
        "| Min fatigue factor | 0.60 |",
        "| Readiness | ctl_weight=0.05, tsb_weight=0.10 |",
        "| Fatigue state | plain decayed TRIMP (keep muscular secondary term in grid only) |",
        "",
        "Absolute best archived aggregate remains `trail_digital_twin_boundary_wide`",
        "(~10.49 min MAE, `hrr_reference=0.85`), but the hypothesis screen/refined",
        "batches are the strongest **current-code** validation and should drive defaults.",
        "",
        "Detailed H1–H5 verdicts live in",
        "`docs/science/trail_digital_twin_benchmark_hyperparameter_report.md`.",
        "",
    ]

    if frame.empty:
        lines.extend(
            [
                "## Experiment Catalog",
                "",
                "No `benchmark_leaderboard.csv` files found under",
                "`data/exp_perf_predictions/`.",
                "",
            ]
        )
        return "\n".join(lines)

    # Global leaderboard
    global_rows: list[list[str]] = []
    for _, row in frame.iterrows():
        global_rows.append(
            [
                str(int(row["globalRank"])),
                str(row["experimentFolder"]),
                _fmt(row.get("meanStage3MaeMin")),
                _fmt(row.get("meanStage3MapePct")),
                _fmt(row.get("meanStage3R2"), 3),
                _profile_cell(row),
                str(int(row.get("runCount") or 0)),
            ]
        )
    lines.extend(
        [
            "## Global Leaderboard (Best Run Per Experiment)",
            "",
            _markdown_table(
                [
                    "rank",
                    "experiment",
                    "MAE min",
                    "MAPE %",
                    "R2",
                    "winner profile",
                    "runs",
                ],
                global_rows,
            ),
            "",
        ]
    )

    # Hypothesis batch
    hyp = frame[frame["experimentFolder"].isin(HYPOTHESIS_FOLDERS)].copy()
    if not hyp.empty:
        hyp_rows: list[list[str]] = []
        for _, row in hyp.sort_values("meanStage3MaeMin").iterrows():
            hyp_rows.append(
                [
                    str(row["experimentFolder"]),
                    str(row.get("experimentGroup") or "—"),
                    _fmt(row.get("meanStage3MaeMin")),
                    _fmt(row.get("meanStage3MapePct")),
                    _fmt(row.get("meanStage3R2"), 3),
                    _profile_cell(row),
                ]
            )
        lines.extend(
            [
                "## Hypothesis Batch Summary",
                "",
                "Stage 1 screen → Stage 2 refined confirmation, plus Minetti ±75% GAP probe.",
                "",
                _markdown_table(
                    [
                        "experiment",
                        "winner group",
                        "MAE min",
                        "MAPE %",
                        "R2",
                        "profile",
                    ],
                    hyp_rows,
                ),
                "",
                "| hypothesis | verdict |",
                "| --- | --- |",
                "| H1 high-reference family | confirmed (`hrr_reference≈0.88`) |",
                "| H2 muscular secondary fatigue | keep in grid; default usually zero |",
                "| H3 hard-trail low-reference | rejected as default; diagnostic only |",
                "| H4 stress/terrain residual strata | confirmed as remaining error drivers |",
                "| H5 bootstrap uncertainty | overlapping intervals; avoid overfit to sub-min diffs |",
                "",
            ]
        )

    # Historical lineage
    lineage = frame[frame["experimentFolder"].isin(HISTORICAL_ORDER)].copy()
    if not lineage.empty:
        order_map = {name: idx for idx, name in enumerate(HISTORICAL_ORDER)}
        lineage["lineageOrder"] = lineage["experimentFolder"].map(
            lambda name: order_map.get(str(name), 999)
        )
        lineage = lineage.sort_values("lineageOrder")
        lineage_rows: list[list[str]] = []
        for _, row in lineage.iterrows():
            lineage_rows.append(
                [
                    str(row["experimentFolder"]),
                    _fmt(row.get("meanStage3MaeMin")),
                    _fmt(row.get("hrrReference"), 2),
                    _fmt(row.get("decayLambda"), 2),
                    _fmt(row.get("minFatigueFactor"), 2),
                ]
            )
        lines.extend(
            [
                "## Historical Sweep Lineage",
                "",
                "Chronological search path from early LOO baselines through boundary and",
                "hypothesis campaigns.",
                "",
                _markdown_table(
                    [
                        "experiment",
                        "best MAE min",
                        "hrr_ref",
                        "decay_λ",
                        "fatigue floor",
                    ],
                    lineage_rows,
                ),
                "",
            ]
        )

    # Minetti probe note
    minetti = frame[frame["experimentFolder"] == "trail_digital_twin_minetti075_probe"]
    if not minetti.empty:
        m = minetti.iloc[0]
        lines.extend(
            [
                "## Minetti ±75% Probe",
                "",
                f"- Best aggregate Stage 3 LOO MAE: **{_fmt(m.get('meanStage3MaeMin'))} min**",
                f"- Profile: `{_profile_cell(m)}`",
                "- Wider GAP grade clamp (±75%) plus spike filtering helps steep-terrain",
                "  segment metrics but did **not** improve the aggregate LOO score versus",
                "  the hypothesis winner (~10.84 min).",
                "- Treat as a terrain-model experiment until a full resweep confirms a",
                "  new optimum.",
                "",
            ]
        )

    # Experiment catalog
    catalog_rows: list[list[str]] = []
    for _, row in frame.sort_values("experimentFolder").iterrows():
        catalog_rows.append(
            [
                str(row["experimentFolder"]),
                str(int(row.get("runCount") or 0)),
                str(int(row.get("leaderboardRows") or 0)),
                "yes" if bool(row.get("hasHtmlReport")) else "no",
                str(row.get("status") or "—"),
            ]
        )
    lines.extend(
        [
            "## Experiment Catalog",
            "",
            _markdown_table(
                ["folder", "planned/runs", "leaderboard rows", "HTML", "best status"],
                catalog_rows,
            ),
            "",
            "## Next Build Steps",
            "",
            "1. Promote high-reference defaults (`0.88 / 0.30 / 1.00 / 0.20 / 0.60`) into",
            "   `configs/trail_digital_twin_extensions.yaml` when ready for production.",
            "2. Keep `hrr_reference=0.85` and hard-trail `min_fatigue_factor=0.50` as",
            "   cohort-specific challengers, not the global default.",
            "3. Leave muscular secondary fatigue in the Stage 3 grid; do not hard-code it.",
            "4. Investigate residual MAE on high-duration / high-TRIMP hard trails",
            "   (stress-duration and terrain mechanics).",
            "5. Optionally re-evaluate Minetti clamp with a dedicated terrain sweep.",
            "",
            "## Source Artifacts",
            "",
            "- Aggregate CSVs under `data/exp_perf_predictions/*/benchmark_*.csv`",
            "- HTML reports: `*/trail_digital_twin_benchmark_report.html`",
            "- Detailed findings:",
            "  `docs/science/trail_digital_twin_benchmark_hyperparameter_report.md`",
            "- Research notes: `docs/science/trail_digital_twin_research_documentation.md`",
            "- Regenerator: `uv run python scripts/synthesize_trail_digital_twin_experiments.py`",
            "",
            "Note: `benchmark_hrr_trimp_grid_search.csv` files are gitignored (large,",
            "recomputable from benchmark reruns).",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Synthesize trail digital twin experiment leaderboards."
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=DEFAULT_EXP_DIR,
        help="Directory containing experiment folders (default: data/exp_perf_predictions)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Markdown report path",
    )
    args = parser.parse_args(argv)

    exp_dir = args.exp_dir if args.exp_dir.is_absolute() else REPO_ROOT / args.exp_dir
    output = args.output if args.output.is_absolute() else REPO_ROOT / args.output

    frame = scan_experiments(exp_dir)
    report = build_report(frame)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Wrote {output} ({len(frame)} experiments)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
