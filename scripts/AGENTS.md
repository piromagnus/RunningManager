# Scripts

Runnable analysis and maintenance tools.

## Files

| File | Purpose |
|------|---------|
| `estimate_prerace_hrr.py` | GPX pre-race estimate by constant HRR sweep using binned or power-law HRR-duration feasibility |
| `predict_race_constant_hrr.py` | Hold-out Stage 3 fit + constant-hard HRR prediction on race_pacing/GPX or activity timeseries profiles (LUT, Grésivaudan, Échappée Belle, Passerelles) |
| `trail_digital_twin_extensions.py` | YAML-configured trail digital-twin fitting, CSV export, and HTML report |
| `trail_digital_twin_benchmark.py` | Large benchmark sweep runner for trail digital-twin config variants |
| `generate_trail_digital_twin_hypothesis_refined.py` | Generate Stage 2 hypothesis benchmark YAML from Stage 1 CSV winners |
| `synthesize_trail_digital_twin_experiments.py` | Aggregate experiment leaderboards into a synthesis markdown report |
| `synthesize_trail_digital_twin_sessions.py` | Session-level LOO consolidation + useful-elements review report |

## Conventions

- Keep scripts runnable from the repository root with `uv run python`.
- Reuse domain helpers from `services/` instead of duplicating model logic.
- Write outputs only when an explicit `--output-dir` is provided.
- Do not print secrets or raw platform tokens.

## Pre-Race HRR Estimate

```bash
uv run python scripts/estimate_prerace_hrr.py --duration-model power-law --output-dir /tmp/prerace \
  "divers/30km - LUT By Night .gpx" "divers/trail-du-gresivaudan-2026.gpx"
```

- Use `--duration-model bin` for the empirical HRR-band max-duration guardrail
- Use `--duration-model power-law` to fit a monotone representative-window HRR-duration frontier
- Power-law fitting defaults to `--power-law-weight-mode performance` to prioritize best frontier performances
- Use `--fatigue-input-col cumTrimpBefore` for monotone in-race fatigue, or `decayedTrimpBefore` to mirror benchmark-selected decayed fatigue variants
- Power-law runs write `hrr_duration_power_law_curve.html` next to the CSV outputs
- `route_estimates.csv` includes combined time uncertainty and HRR sustainability-risk metrics
- Pass optimized physiology knobs explicitly when evaluating benchmark-derived settings

## Trail Digital Twin Extensions

```bash
uv run python scripts/trail_digital_twin_extensions.py \
  --config configs/trail_digital_twin_extensions.yaml
```

- Config owns paths, cohorts, physiology/readiness weights, grids, fitting objectives, robustness, and output toggles
- Default output directory is configured in `configs/trail_digital_twin_extensions.yaml`
- Use `--output-dir /path/to/run` for ad hoc runs that should not update science assets
- Use `--jobs N` to run independent cohort/objective Stage 0-3 fits in parallel; benchmark runs force this inner setting back to 1
- Use `--validate-config` to check YAML without fitting models

## Trail Digital Twin Benchmark

```bash
uv run python scripts/trail_digital_twin_benchmark.py \
  --base-config configs/trail_digital_twin_extensions.yaml \
  --benchmark-config configs/trail_digital_twin_benchmark.yaml
```

- Benchmark config expands the compact best-setup fatigue-floor and decay-lambda sweeps by default
- Default benchmark disables per-run HTML; tune `disable_inner_robustness` and `disable_inner_segment_grid` for faster screening runs
- Use `--dry-run` to write the expanded plan without fitting models
- Use `--html-only` to rebuild the aggregate benchmark HTML report from saved CSVs without fitting models
- Use `--max-runs N` for smoke tests before the full sweep
- Use `--jobs N` to execute independent benchmark runs in parallel; the default is `execution.jobs` from the benchmark YAML
- The benchmark includes both `hardTrailRun` and mixed `hardRunOrTrailRun` cohort coverage

## Trail Digital Twin Hypothesis Refinement

```bash
uv run python scripts/generate_trail_digital_twin_hypothesis_refined.py
```

- Reads Stage 1 `benchmark_leaderboard.csv` and `benchmark_stage_metrics.csv`
- Selects the best H1 aggregate run and best H3 `hardTrailRun` run
- Writes `configs/trail_digital_twin_benchmark_hypothesis_refined.yaml`
- Use after running `configs/trail_digital_twin_benchmark_hypothesis_screen.yaml`

## Trail Digital Twin Experiment Synthesis

```bash
uv run python scripts/synthesize_trail_digital_twin_experiments.py
```

- Scans `data/exp_perf_predictions/*/benchmark_leaderboard.csv`
- Writes `docs/science/trail_digital_twin_experiment_synthesis_report.md`
- Use `--exp-dir` / `--output` to override paths

## Trail Digital Twin Session Review

```bash
uv run python scripts/synthesize_trail_digital_twin_sessions.py
```

- Joins Stage 3 activity LOO predictions with `activities.csv`
- Pulls hypothesis-winner cohort MAE, fitted params, strata, and terrain metrics
- Writes `docs/science/trail_digital_twin_session_benchmark_review.md`
- Also writes `data/exp_perf_predictions/session_benchmark_review/session_benchmark_review.csv`
- Defaults: `--session-source trail_digital_twin_boundary_best_profile`,
  `--hyper-source trail_digital_twin_hypothesis_refined`
