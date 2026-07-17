"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Generate the Stage 2 refined benchmark from Stage 1 hypothesis-screen outputs.
"""

from __future__ import annotations

import argparse
import itertools
import re
import sys
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import yaml
except ImportError as exc:  # pragma: no cover - CLI dependency check.
    yaml = None
    YAML_IMPORT_ERROR = exc
else:
    YAML_IMPORT_ERROR = None


HIGH_REFERENCE_GRIDS: dict[str, list[float]] = {
    "physiology.hrr_reference": [0.82, 0.85, 0.88],
    "physiology.hrr_max_factor": [1.00, 1.10, 1.20],
    "physiology.decay_lambda": [0.20, 0.25, 0.30],
    "physiology.min_fatigue_factor": [0.50, 0.60, 0.70],
}
HARD_TRAIL_GRIDS: dict[str, list[float]] = {
    "physiology.hrr_reference": [0.55, 0.60, 0.65],
    "physiology.hrr_min_factor": [0.30, 0.40, 0.50],
    "physiology.hrr_max_factor": [1.40, 1.60, 1.80],
    "physiology.decay_lambda": [0.10, 0.15, 0.20],
}


def _repo_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _slug(value: object) -> str:
    text = f"{float(value):.3g}" if isinstance(value, (float, int)) else str(value)
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def _float_value(row: Mapping[str, object], column: str) -> float:
    value = row.get(column)
    if value is None or pd.isna(value):
        raise ValueError(f"missing required selected value: {column}")
    return float(value)


def _column_for_path(dotted_path: str) -> str:
    mapping = {
        "physiology.hrr_reference": "hrrReference",
        "physiology.hrr_min_factor": "hrrMinFactor",
        "physiology.hrr_max_factor": "hrrMaxFactor",
        "physiology.decay_lambda": "decayLambda",
        "physiology.min_fatigue_factor": "minFatigueFactor",
    }
    return mapping[dotted_path]


def _nearest_grid_index(value: float, grid: Sequence[float]) -> int:
    return min(range(len(grid)), key=lambda index: (abs(grid[index] - value), index))


def _adjacent_grid_values(value: float, grid: Sequence[float]) -> list[float]:
    index = _nearest_grid_index(value, grid)
    start = max(0, index - 1)
    end = min(len(grid), index + 2)
    return [float(item) for item in grid[start:end]]


def _candidate_overrides(
    selected: Mapping[str, object],
    grids: Mapping[str, Sequence[float]],
    *,
    cap: int,
    fixed: Mapping[str, float] | None = None,
) -> list[dict[str, float]]:
    selected_values = {
        dotted_path: _float_value(selected, _column_for_path(dotted_path))
        for dotted_path in grids
    }
    candidate_values = {
        dotted_path: _adjacent_grid_values(selected_values[dotted_path], grid)
        for dotted_path, grid in grids.items()
    }
    rows: list[dict[str, float]] = []
    for values in itertools.product(*(candidate_values[key] for key in grids)):
        row = dict(zip(grids.keys(), values))
        if fixed:
            row.update(fixed)
        rows.append({key: float(value) for key, value in row.items()})

    def sort_key(row: Mapping[str, float]) -> tuple[float, tuple[float, ...]]:
        distance = 0.0
        for dotted_path, grid in grids.items():
            selected_index = _nearest_grid_index(selected_values[dotted_path], grid)
            row_index = _nearest_grid_index(float(row[dotted_path]), grid)
            distance += abs(row_index - selected_index)
        return distance, tuple(float(row[key]) for key in sorted(row))

    return sorted(rows, key=sort_key)[:cap]


def _run_name(prefix: str, index: int, overrides: Mapping[str, float]) -> str:
    parts = [prefix, f"{index:03d}"]
    for key in [
        "physiology.hrr_reference",
        "physiology.hrr_min_factor",
        "physiology.hrr_max_factor",
        "physiology.decay_lambda",
        "physiology.min_fatigue_factor",
    ]:
        if key in overrides:
            parts.append(f"{key.split('.')[-1]}_{_slug(overrides[key])}")
    return "_".join(parts)[:96]


def _variant_runs(prefix: str, overrides_list: Sequence[Mapping[str, float]]) -> list[dict[str, object]]:
    return [
        {
            "name": _run_name(prefix, index, overrides),
            "overrides": {key: float(value) for key, value in overrides.items()},
        }
        for index, overrides in enumerate(overrides_list)
    ]


def _best_high_reference_run(stage1_dir: Path) -> dict[str, object]:
    leaderboard = pd.read_csv(stage1_dir / "benchmark_leaderboard.csv")
    data = leaderboard[
        leaderboard["experimentGroup"].astype(str).eq("h1_high_reference_confirmation")
    ].copy()
    if data.empty:
        raise ValueError("no h1_high_reference_confirmation rows found in Stage 1 leaderboard")
    data["meanStage3MaeMin"] = pd.to_numeric(data["meanStage3MaeMin"], errors="coerce")
    data["meanStage3MapePct"] = pd.to_numeric(data["meanStage3MapePct"], errors="coerce")
    data["maxStage3MaeMin"] = pd.to_numeric(data["maxStage3MaeMin"], errors="coerce")
    data = data.dropna(subset=["meanStage3MaeMin"])
    if data.empty:
        raise ValueError("no ranked high-reference Stage 1 rows found")
    return data.sort_values(
        ["meanStage3MaeMin", "meanStage3MapePct", "maxStage3MaeMin", "runId"],
        na_position="last",
    ).iloc[0].to_dict()


def _best_hard_trail_run(stage1_dir: Path) -> dict[str, object]:
    stage_metrics = pd.read_csv(stage1_dir / "benchmark_stage_metrics.csv")
    data = stage_metrics[
        stage_metrics["experimentGroup"].astype(str).eq("h3_hard_trail_low_reference")
        & stage_metrics["cohort"].astype(str).eq("hardTrailRun")
        & stage_metrics["stage"].astype(str).eq("Stage 3 HRR speed ratio LOO")
        & stage_metrics["fitObjective"].astype(str).eq("activity")
    ].copy()
    if data.empty:
        raise ValueError("no hardTrailRun H3 Stage 3 LOO rows found in Stage 1 metrics")
    data["maeMin"] = pd.to_numeric(data["maeMin"], errors="coerce")
    data["mapePct"] = pd.to_numeric(data["mapePct"], errors="coerce")
    data = data.dropna(subset=["maeMin"])
    if data.empty:
        raise ValueError("no ranked hard-trail Stage 1 rows found")
    return data.sort_values(["maeMin", "mapePct", "runId"], na_position="last").iloc[0].to_dict()


def _common_overrides() -> dict[str, object]:
    return {
        "fitting.enabled_objectives": ["activity"],
        "fitting.validation_modes": ["in_sample", "loo"],
        "fitting.hrr_trimp_alpha_grid": [0.85, 0.90, 0.95, 1.00, 1.05],
        "fitting.hrr_trimp_kappa_grid": [0.20, 0.30, 0.40, 0.60, 0.80],
        "fitting.hrr_trimp_secondary_kappa_grid": [0.00, 0.10, 0.20],
        "fitting.fatigue_models": ["linear", "exponential"],
        "fitting.stage3_fatigue_states": [
            {
                "fatigue_state": "decayed",
                "acute_trimp_col": "decayedTrimpBefore",
                "label": "decayed TRIMP",
            },
            {
                "fatigue_state": "cumulative",
                "acute_trimp_col": "cumTrimpBefore",
                "label": "cumulative TRIMP",
            },
            {
                "fatigue_state": "decayed_progress",
                "acute_trimp_col": "decayedTrimpBefore",
                "label": "decayed exp + progress",
                "fatigue_models": ["exponential"],
                "secondary_acute_trimp_col": "progress",
                "secondary_fatigue_model": "exponential",
            },
            {
                "fatigue_state": "decayed_cumulative",
                "acute_trimp_col": "decayedTrimpBefore",
                "label": "decayed exp + cumulative TRIMP",
                "fatigue_models": ["exponential"],
                "secondary_acute_trimp_col": "cumTrimpBefore",
                "secondary_fatigue_model": "exponential",
            },
        ],
        "readiness.ctl_weight": 0.05,
        "readiness.tsb_weight": 0.10,
        "readiness.ctl_factor_min": 0.90,
        "readiness.ctl_factor_max": 1.10,
    }


def build_refined_config(
    stage1_dir: Path,
    *,
    output_dir: str = "data/exp_perf_predictions/trail_digital_twin_hypothesis_refined",
    high_cap: int = 36,
    hard_cap: int = 36,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    stage1_dir = _repo_path(stage1_dir)
    best_high = _best_high_reference_run(stage1_dir)
    best_hard = _best_hard_trail_run(stage1_dir)
    high_overrides = _candidate_overrides(
        best_high,
        HIGH_REFERENCE_GRIDS,
        cap=high_cap,
        fixed={"physiology.hrr_min_factor": 0.30},
    )
    hard_overrides = _candidate_overrides(
        best_hard,
        HARD_TRAIL_GRIDS,
        cap=hard_cap,
        fixed={"physiology.min_fatigue_factor": 0.50},
    )
    config: dict[str, object] = {
        "execution": {
            "output_dir": output_dir,
            "per_run_csv": False,
            "per_run_html": False,
            "disable_inner_robustness": True,
            "disable_inner_segment_grid": True,
            "write_html": True,
            "fail_fast": False,
            "jobs": 24,
            "common_overrides": _common_overrides(),
        },
        "selection": {
            "primary_stage": "Stage 3 HRR speed ratio LOO",
            "fallback_stage": "Stage 3 HRR speed ratio",
            "aggregate_objectives": ["activity"],
            "aggregate_cohorts": [
                "hardTrailRun",
                "hardRunOrTrailRun",
                "top10HardTrailByHRR",
                "selectedDateRaces",
            ],
        },
        "groups": [
            {
                "id": "h1_high_reference_refined",
                "mode": "variants",
                "description": "Stage 2 refined high-reference family from Stage 1 aggregate winner.",
                "runs": _variant_runs("h1_refined", high_overrides),
            },
            {
                "id": "h3_hard_trail_refined",
                "mode": "variants",
                "description": "Stage 2 refined hard-trail family from Stage 1 hardTrailRun winner.",
                "runs": _variant_runs("h3_refined", hard_overrides),
            },
        ],
    }
    return config, best_high, best_hard


def write_refined_config(config: Mapping[str, object], output_path: Path) -> Path:
    if yaml is None:
        raise RuntimeError("PyYAML is required to write benchmark configs") from YAML_IMPORT_ERROR
    output_path = _repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(dict(config), sort_keys=False))
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Stage 2 hypothesis-refined benchmark YAML.")
    parser.add_argument(
        "--stage1-dir",
        type=Path,
        default=Path("data/exp_perf_predictions/trail_digital_twin_hypothesis_screen"),
        help="Stage 1 benchmark output directory.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("configs/trail_digital_twin_benchmark_hypothesis_refined.yaml"),
        help="Path to write the Stage 2 benchmark YAML.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/exp_perf_predictions/trail_digital_twin_hypothesis_refined",
        help="Stage 2 benchmark output directory stored in the generated YAML.",
    )
    parser.add_argument("--high-cap", type=int, default=36, help="Maximum high-reference refined runs.")
    parser.add_argument("--hard-cap", type=int, default=36, help="Maximum hard-trail refined runs.")
    parser.add_argument("--quiet", action="store_true", help="Print only the written config path.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config, best_high, best_hard = build_refined_config(
        args.stage1_dir,
        output_dir=args.output_dir,
        high_cap=args.high_cap,
        hard_cap=args.hard_cap,
    )
    path = write_refined_config(config, args.output_config)
    if args.quiet:
        print(path)
        return
    print(f"Wrote Stage 2 benchmark config: {path}")
    print(f"Selected H1 aggregate winner: {best_high.get('runId')}")
    print(f"Selected H3 hardTrailRun winner: {best_hard.get('runId')}")
    print(
        "Planned refined runs: "
        f"{len(config['groups'][0]['runs']) + len(config['groups'][1]['runs'])}"
    )


if __name__ == "__main__":
    main()
