# Trail Digital Twin Experiment Synthesis Report

Generated: 2026-07-17

Cross-experiment summary of Stage 3 leave-one-out activity MAE across
`data/exp_perf_predictions/`. Primary metric is **mean Stage 3 LOO MAE (min)**;
lower is better. Aggregate scores average configured cohorts
(`hardTrailRun`, `hardRunOrTrailRun`, `top10HardTrailByHRR`, `selectedDateRaces`).

## Definitions

- **HRR_ref (`hrr_reference`)**: HRR at which the effort multiplier \(E=1\)
  (\(E=\mathrm{clip}(\mathrm{HRR}/\mathrm{HRR}_{\mathrm{ref}},\,h_{\min},\,h_{\max})\)).
  With `hrr_max_factor=1.0` it is also the effort ceiling. It is **not** “HRR at
  VMA”; VMA (`vma_flat_kmh`) is a separate flat-speed anchor, and flat fresh
  speed at \(E=1\) is \(v_{\mathrm{VMA}}\cdot\alpha\).
- **Decay λ (`decay_lambda`)**: exponential decay rate for in-race TRIMP used
  by Stage 3 fatigue states.
- **Min fatigue factor**: floor on the fatigue multiplier so long races cannot
  collapse predicted speed to zero.
- Open modeling question: how to treat elapsed time or segments that include
  recovery / walking without clear HR-effort continuity.

## Executive Summary

Operational default for continued building (current-code hypothesis winner):

| factor | recommended value |
| --- | ---: |
| HRR reference | 0.88 |
| HRR min factor | 0.30 |
| HRR max factor | 1.00 |
| Decay lambda | 0.20 |
| Min fatigue factor | 0.60 |
| Readiness | ctl_weight=0.05, tsb_weight=0.10 |
| Fatigue state | plain decayed TRIMP (keep muscular secondary term in grid only) |

Absolute best archived aggregate remains `trail_digital_twin_boundary_wide`
(~10.49 min MAE, `hrr_reference=0.85`), but the hypothesis screen/refined
batches are the strongest **current-code** validation and should drive defaults.

Detailed H1–H5 verdicts live in
`docs/science/trail_digital_twin_benchmark_hyperparameter_report.md`.

## Global Leaderboard (Best Run Per Experiment)

| rank | experiment | MAE min | MAPE % | R2 | winner profile | runs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | trail_digital_twin_boundary_wide | 10.49 | 7.76 | 0.982 | hrr=0.85, min=0.30, max=1.00, λ=0.25, floor=0.50 | 427 |
| 2 | trail_digital_twin_hypothesis_refined | 10.84 | 7.91 | 0.982 | hrr=0.88, min=0.30, max=1.00, λ=0.20, floor=0.60 | 48 |
| 3 | trail_digital_twin_hypothesis_screen | 10.84 | 7.91 | 0.982 | hrr=0.88, min=0.30, max=1.00, λ=0.20, floor=0.60 | 165 |
| 4 | trail_digital_twin_boundary_refined | 11.09 | 8.44 | 0.982 | hrr=0.85, min=0.30, max=1.20, λ=0.25, floor=0.50 | 64 |
| 5 | trail_digital_twin_minetti075_probe | 11.17 | 8.33 | 0.982 | hrr=0.88, min=0.30, max=1.00, λ=0.20, floor=0.60 | 1 |
| 6 | trail_digital_twin_best_factor_sweep | 11.33 | 8.26 | 0.980 | hrr=0.85, min=0.30, max=1.20, λ=0.25, floor=0.60 | 23 |
| 7 | trail_digital_twin_wide_screen | 11.66 | 8.73 | 0.977 | hrr=0.80, min=0.30, max=1.00, λ=0.15, floor=0.50 | 257 |
| 8 | trail_digital_twin_refined_screen | 12.59 | 9.04 | 0.975 | hrr=0.75, min=0.30, max=1.00, λ=0.15, floor=0.50 | 56 |
| 9 | trail_digital_twin_refine_screen | 17.03 | 10.12 | 0.953 | hrr=0.55, min=0.30, max=1.80, λ=0.30, floor=0.50 | 27 |
| 10 | trail_digital_twin_low_ref_screen | 17.81 | 13.43 | 0.963 | hrr=0.60, min=0.40, max=1.60, λ=0.30, floor=0.50 | 30 |
| 11 | trail_digital_twin_best_loo | 19.00 | 12.61 | 0.948 | hrr=0.55, min=0.30, max=1.80, λ=0.30, floor=0.50 | 1 |
| 12 | trail_digital_twin_reference_loo | 26.30 | 23.71 | 0.942 | hrr=0.70, min=0.30, max=1.00, λ=0.30, floor=0.50 | 1 |

## Hypothesis Batch Summary

Stage 1 screen → Stage 2 refined confirmation, plus Minetti ±75% GAP probe.

| experiment | winner group | MAE min | MAPE % | R2 | profile |
| --- | ---: | ---: | ---: | ---: | ---: |
| trail_digital_twin_hypothesis_refined | h1_high_reference_refined | 10.84 | 7.91 | 0.982 | hrr=0.88, min=0.30, max=1.00, λ=0.20, floor=0.60 |
| trail_digital_twin_hypothesis_screen | h1_high_reference_confirmation | 10.84 | 7.91 | 0.982 | hrr=0.88, min=0.30, max=1.00, λ=0.20, floor=0.60 |
| trail_digital_twin_minetti075_probe | h1_high_reference_refined | 11.17 | 8.33 | 0.982 | hrr=0.88, min=0.30, max=1.00, λ=0.20, floor=0.60 |

| hypothesis | verdict |
| --- | --- |
| H1 high-reference family | confirmed (`hrr_reference≈0.88`) |
| H2 muscular secondary fatigue | keep in grid; default usually zero |
| H3 hard-trail low-reference | rejected as default; diagnostic only |
| H4 stress/terrain residual strata | confirmed as remaining error drivers |
| H5 bootstrap uncertainty | overlapping intervals; avoid overfit to sub-min diffs |

## Historical Sweep Lineage

Chronological search path from early LOO baselines through boundary and
hypothesis campaigns.

| experiment | best MAE min | hrr_ref | decay_λ | fatigue floor |
| --- | ---: | ---: | ---: | ---: |
| trail_digital_twin_reference_loo | 26.30 | 0.70 | 0.30 | 0.50 |
| trail_digital_twin_best_loo | 19.00 | 0.55 | 0.30 | 0.50 |
| trail_digital_twin_low_ref_screen | 17.81 | 0.60 | 0.30 | 0.50 |
| trail_digital_twin_refine_screen | 17.03 | 0.55 | 0.30 | 0.50 |
| trail_digital_twin_wide_screen | 11.66 | 0.80 | 0.15 | 0.50 |
| trail_digital_twin_refined_screen | 12.59 | 0.75 | 0.15 | 0.50 |
| trail_digital_twin_boundary_wide | 10.49 | 0.85 | 0.25 | 0.50 |
| trail_digital_twin_boundary_refined | 11.09 | 0.85 | 0.25 | 0.50 |
| trail_digital_twin_best_factor_sweep | 11.33 | 0.85 | 0.25 | 0.60 |
| trail_digital_twin_hypothesis_screen | 10.84 | 0.88 | 0.20 | 0.60 |
| trail_digital_twin_hypothesis_refined | 10.84 | 0.88 | 0.20 | 0.60 |
| trail_digital_twin_minetti075_probe | 11.17 | 0.88 | 0.20 | 0.60 |

## Minetti ±75% Probe

- Best aggregate Stage 3 LOO MAE: **11.17 min**
- Profile: `hrr=0.88, min=0.30, max=1.00, λ=0.20, floor=0.60`
- Wider GAP grade clamp (±75%) plus spike filtering helps steep-terrain
  segment metrics but did **not** improve the aggregate LOO score versus
  the hypothesis winner (~10.84 min).
- Treat as a terrain-model experiment until a full resweep confirms a
  new optimum.

## Experiment Catalog

| folder | planned/runs | leaderboard rows | HTML | best status |
| --- | ---: | ---: | ---: | ---: |
| trail_digital_twin_best_factor_sweep | 23 | 23 | yes | success |
| trail_digital_twin_best_loo | 1 | 1 | yes | success |
| trail_digital_twin_boundary_refined | 64 | 64 | yes | success |
| trail_digital_twin_boundary_wide | 427 | 427 | yes | success |
| trail_digital_twin_hypothesis_refined | 48 | 48 | yes | success |
| trail_digital_twin_hypothesis_screen | 165 | 165 | yes | success |
| trail_digital_twin_low_ref_screen | 30 | 30 | yes | success |
| trail_digital_twin_minetti075_probe | 1 | 1 | yes | success |
| trail_digital_twin_reference_loo | 1 | 1 | yes | success |
| trail_digital_twin_refine_screen | 27 | 27 | yes | success |
| trail_digital_twin_refined_screen | 56 | 55 | yes | success |
| trail_digital_twin_wide_screen | 257 | 256 | yes | success |

## Next Build Steps

1. Promote high-reference defaults (`0.88 / 0.30 / 1.00 / 0.20 / 0.60`) into
   `configs/trail_digital_twin_extensions.yaml` when ready for production.
2. Keep `hrr_reference=0.85` and hard-trail `min_fatigue_factor=0.50` as
   cohort-specific challengers, not the global default.
3. Leave muscular secondary fatigue in the Stage 3 grid; do not hard-code it.
4. Investigate residual MAE on high-duration / high-TRIMP hard trails
   (stress-duration and terrain mechanics).
5. Optionally re-evaluate Minetti clamp with a dedicated terrain sweep.
6. Run `configs/trail_digital_twin_benchmark_segment_exclusion.yaml` to fit on
   cleaned (non-stationary) segments then score full-race LOO; review with
   `scripts/synthesize_trail_digital_twin_sessions.py`.
7. Session-level residual triage:
   `docs/science/trail_digital_twin_session_benchmark_review.md`.

## Source Artifacts

- Aggregate CSVs under `data/exp_perf_predictions/*/benchmark_*.csv`
- HTML reports: `*/trail_digital_twin_benchmark_report.html`
- Detailed findings:
  `docs/science/trail_digital_twin_benchmark_hyperparameter_report.md`
- Research notes: `docs/science/trail_digital_twin_research_documentation.md`
- Regenerator: `uv run python scripts/synthesize_trail_digital_twin_experiments.py`

Note: `benchmark_hrr_trimp_grid_search.csv` files are gitignored (large,
recomputable from benchmark reruns).
