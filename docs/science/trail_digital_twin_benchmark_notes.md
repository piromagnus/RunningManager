# Trail Digital Twin Benchmark Notes

## 2026-06-18 HRR Reference and Hard-Run Cohort Screen

Context:

- Added `hardRunOrTrailRun`, a hard activity cohort that accepts both `RUN` and `TRAIL_RUN` categories with usable segments.
- Kept `hardTrailRun` unchanged for continuity with the trail-only manuscript tables.
- Screened low HRR references and wider HRR effort multiplier caps with in-sample Stage 3 metrics.
- Verified the best low-reference setting with a tight leave-one-out run.

Screen setup:

- Base cohorts: `hardTrailRun`, `hardRunOrTrailRun`.
- Objective: `activity`.
- Screen validation: `in_sample`.
- Tight verification validation: `in_sample`, `loo`.
- Segment-grid and inner robustness loops were disabled for benchmark throughput.

First screen:

| result | hrr_reference | hrr_min_factor | hrr_max_factor | mean Stage 3 MAE min |
| --- | ---: | ---: | ---: | ---: |
| best averaged screen | 0.60 | 0.40 | 1.60 | 17.81 |
| best hardRunOrTrailRun screen | 0.50 | 0.20 | 1.60 | 13.63 |
| best hardTrailRun screen | 0.60 | 0.40 | 1.60 | 20.21 |

Refinement screen:

| result | hrr_reference | hrr_min_factor | hrr_max_factor | mean Stage 3 MAE min |
| --- | ---: | ---: | ---: | ---: |
| best averaged refinement | 0.55 | 0.30 | 1.80 | 17.03 |
| tied averaged refinement | 0.55 | 0.30 | 2.00 | 17.03 |

The best refinement selected cumulative exponential acute fatigue for both cohorts.

Tight LOO comparison:

| setting | cohort | Stage 3 LOO MAE min | MAPE pct | R2 |
| --- | --- | ---: | ---: | ---: |
| HRR ref 0.55, min 0.30, max 1.80 | hardTrailRun | 22.18 | 13.64 | 0.959 |
| HRR ref 0.55, min 0.30, max 1.80 | hardRunOrTrailRun | 15.83 | 11.59 | 0.938 |
| HRR ref 0.70, min 0.30, max 1.00 | hardTrailRun | 28.12 | 19.83 | 0.946 |
| HRR ref 0.70, min 0.30, max 1.00 | hardRunOrTrailRun | 24.48 | 27.60 | 0.938 |

Interpretation:

- Allowing HRR effort above 1.0 matters. The original `hrr_max_factor: 1.0` clips all above-reference effort and loses useful signal.
- A smaller HRR reference around 0.55-0.60 plus a wider max cap around 1.6-1.8 improved both the hard trail cohort and the mixed hard run/trail cohort in the targeted LOO check.
- `hardRunOrTrailRun` benefits strongly from the lower reference setting, suggesting the HRR effort normalization helps non-trail RUN activities with different effort distributions.
- These results came from compact grids, so the full benchmark should still include wider alpha/kappa/readiness sweeps before changing manuscript claims.

Artifacts:

- `/tmp/trail_digital_twin_low_ref_screen`
- `/tmp/trail_digital_twin_refine_screen`
- `/tmp/trail_digital_twin_best_loo`
- `/tmp/trail_digital_twin_reference_loo`

## 2026-06-18 Two-Stage Wide and Refined Benchmark

Context:

- Reduced the broad benchmark from the previous 872-run shape to a 257-run wide in-sample screen.
- Added benchmark `execution.common_overrides` so shared validation, objective, and grid settings are not duplicated in every group.
- Ran a second 56-run refined pass with `in_sample` plus `loo` validation based on the wide-screen winner regions.
- Used 24 benchmark workers for both runs.

Wide screen setup:

- Config: `configs/trail_digital_twin_benchmark.yaml`.
- Output: `data/exp_perf_predictions/trail_digital_twin_wide_screen`.
- Runs: 257 success, 0 failed.
- Ranking stage: in-sample `Stage 3 HRR speed ratio`.
- Ranking cohorts: `hardTrailRun`, `hardRunOrTrailRun`, `top10HardTrailByHRR`, `selectedDateRaces`.

Wide screen top results:

| rank | group | hrr_reference | hrr_min_factor | hrr_max_factor | trimp_scale | decay_lambda | mean MAE min | max MAE min |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | hrr_trimp_scale | 0.80 | 0.30 | 1.00 | 2.5 | 0.15 | 11.66 | 20.27 |
| 2 | hrr_trimp_scale | 0.80 | 0.30 | 1.00 | 2.5 | 0.05 | 12.01 | 20.18 |
| 3 | hrr_trimp_scale | 0.80 | 0.30 | 1.00 | 5.0 | 0.05 | 12.14 | 20.30 |
| 5 | hrr_factor_bounds | 0.70 | 0.40 | 1.60 | 10.0 | 0.30 | 12.63 | 21.10 |

Cohort-specific wide-screen signal:

- `hardRunOrTrailRun` still preferred a low-reference, wide-clip setting: best MAE 13.16 min at `hrr_reference: 0.40`, `hrr_min_factor: 0.60`, `hrr_max_factor: 2.00`.
- `hardTrailRun` preferred high reference and low acute TRIMP scale: best MAE 20.18 min at `hrr_reference: 0.80`, `trimp_scale: 2.5`, `decay_lambda: 0.05`.
- The top 50 wide-screen runs were dominated by `hrr_factor_bounds` and `hrr_trimp_scale`; default readiness weights remained globally competitive.

Refined LOO setup:

- Config: `configs/trail_digital_twin_benchmark_refined.yaml`.
- Output: `data/exp_perf_predictions/trail_digital_twin_refined_screen`.
- Runs: 56 success, 0 failed.
- Ranking stage: `Stage 3 HRR speed ratio LOO`.
- Ranking objective: `activity`; segment objective was included as a separate comparison group.
- Slowest run: `objective_both`, 690.6 sec; segment-enabled LOO is the expensive tail.

Refined LOO top results:

| rank | group | hrr_reference | hrr_min_factor | hrr_max_factor | trimp_scale | decay_lambda | mean LOO MAE min | max LOO MAE min |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | high_reference_low_trimp | 0.75 | 0.30 | 1.00 | 2.0 | 0.15 | 12.59 | 22.61 |
| 2 | high_reference_low_trimp | 0.75 | 0.30 | 1.00 | 2.0 | 0.05 | 12.66 | 22.61 |
| 3 | high_reference_low_trimp | 0.80 | 0.30 | 1.00 | 2.0 | 0.05 | 12.96 | 20.80 |
| 4 | high_reference_low_trimp | 0.75 | 0.30 | 1.00 | 2.5 | 0.15 | 13.02 | 22.61 |
| 5 | high_reference_low_trimp | 0.85 | 0.30 | 1.00 | 2.5 | 0.15 | 13.11 | 22.45 |

Best refined run by cohort:

| cohort | LOO MAE min | MAPE pct | R2 |
| --- | ---: | ---: | ---: |
| hardRunOrTrailRun | 13.93 | 10.42 | 0.961 |
| hardTrailRun | 22.61 | 14.13 | 0.967 |
| selectedDateRaces | 8.49 | 5.22 | 0.995 |
| top10HardTrailByHRR | 5.33 | 6.23 | 0.987 |

Objective comparison at the refined balanced setting:

| objective | hardRunOrTrailRun MAE | hardTrailRun MAE | selectedDateRaces MAE | top10HardTrailByHRR MAE |
| --- | ---: | ---: | ---: | ---: |
| activity | 17.11 | 22.79 | 16.80 | 5.96 |
| segment | 13.77 | 22.47 | 13.92 | 5.35 |

Fatigue-state findings:

- Across all refined LOO rows, cumulative exponential fatigue had the best median MAE: 16.99 min.
- In best-per-run/cohort selections, cumulative exponential was selected most often: 93 selections, median 14.60 min.
- Decayed fatigue remains useful for specific cohorts: the hard-trail minimum used decayed exponential at 20.73 min, and selected-date race minima also benefited from decayed variants.
- Keep both cumulative and decayed states in comparison tables; use cumulative exponential as the default stability candidate unless a cohort-specific objective is selected.

Readiness findings:

- Global refined winners kept `ctl_weight: 0.05` and `tsb_weight: 0.10`.
- `top10HardTrailByHRR` improved with stronger readiness weighting: best LOO MAE 3.98 min at `ctl_weight: 0.20`, `tsb_weight: 0.20`.
- Do not promote heavier readiness weights globally from this batch; they look cohort-specific.

Interpretation:

- The LOO refinement moved the best global activity configuration toward `hrr_reference: 0.75`, `hrr_min_factor: 0.30`, `hrr_max_factor: 1.00`, `trimp_scale: 2.0`, and `decay_lambda: 0.05-0.15`.
- The earlier low-reference, wide-clip setting remains useful for mixed hard runs in the in-sample screen, but it did not beat the high-reference low-TRIMP region in the refined activity LOO ranking.
- Segment objective fitting is promising at the balanced setting, but it was not swept across the high-reference low-TRIMP winner region in this batch.
- The best refined activity run hit the upper bound of the refined alpha grid for `hardRunOrTrailRun` and `hardTrailRun` (`alpha: 0.80`, `fatigueCoef: 0.4`), so the next micro-pass should reopen the alpha grid above 0.80 around the high-reference low-TRIMP region.

Artifacts:

- `data/exp_perf_predictions/trail_digital_twin_wide_screen/benchmark_leaderboard.csv`
- `data/exp_perf_predictions/trail_digital_twin_wide_screen/trail_digital_twin_benchmark_report.html`
- `data/exp_perf_predictions/trail_digital_twin_refined_screen/benchmark_leaderboard.csv`
- `data/exp_perf_predictions/trail_digital_twin_refined_screen/trail_digital_twin_benchmark_report.html`

## 2026-06-18 Boundary-Expanded Wide and Refined Benchmark

Context:

- Expanded the previous refined extrema proportionally and deliberately over-shot the ranges before refining.
- Reopened the high-reference region, lower TRIMP scale, larger decay lambda, wider HRR caps, and larger Stage 3 alpha/kappa grids.
- Ran a 427-run in-sample wide screen followed by a 64-run LOO refinement with 24 workers.
- Kept the activity objective for this pass so the refined LOO comparison stayed tractable.

Boundary-wide setup:

- Config: `configs/trail_digital_twin_benchmark_boundary_wide.yaml`.
- Output: `data/exp_perf_predictions/trail_digital_twin_boundary_wide`.
- Runs: 427 success, 0 failed.
- Ranking stage: in-sample `Stage 3 HRR speed ratio`.

Boundary-wide top results:

| rank | group | hrr_reference | hrr_min_factor | hrr_max_factor | trimp_scale | decay_lambda | mean MAE min | max MAE min |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | high_ref_trimp_decay_surface | 0.85 | 0.30 | 1.00 | 1.0 | 0.25 | 10.49 | 18.30 |
| 2 | high_ref_trimp_decay_surface | 0.85 | 0.30 | 1.00 | 1.5 | 0.10 | 10.51 | 17.94 |
| 3 | high_ref_trimp_decay_surface | 0.85 | 0.30 | 1.00 | 1.0 | 0.10 | 10.54 | 18.38 |
| 4 | high_ref_clip_surface | 0.65 | 0.40 | 1.40 | 2.0 | 0.15 | 10.61 | 18.96 |

Boundary-wide cohort signal:

| cohort | best MAE min | hrr_reference | hrr_min_factor | hrr_max_factor | trimp_scale | decay_lambda |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hardRunOrTrailRun | 11.62 | 0.65 | 0.40 | 1.40 | 2.0 | 0.15 |
| hardTrailRun | 17.94 | 0.85 | 0.30 | 1.00 | 1.5 | 0.10 |
| selectedDateRaces | 7.25 | 0.775 | 0.30 | 1.00 | 0.75 | 0.10 |
| top10HardTrailByHRR | 3.85 | 0.65 | 0.30 | 1.00 | 1.0 | 0.25 |

Boundary-refined setup:

- Config: `configs/trail_digital_twin_benchmark_boundary_refined.yaml`.
- Output: `data/exp_perf_predictions/trail_digital_twin_boundary_refined`.
- Runs: 64 success, 0 failed.
- Ranking stage: `Stage 3 HRR speed ratio LOO`.
- Expanded Stage 3 grids: `alpha = 0.70..1.20`, `fatigueCoef = 0.20..1.20`.

Boundary-refined LOO top results:

| rank | group | hrr_reference | hrr_min_factor | hrr_max_factor | trimp_scale | decay_lambda | mean LOO MAE min | max LOO MAE min |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | high_ref_clip_probe | 0.85 | 0.30 | 1.20 | 1.0 | 0.25 | 11.09 | 20.64 |
| 2 | high_ref_clip_probe | 0.85 | 0.30 | 1.40 | 1.0 | 0.25 | 11.09 | 20.64 |
| 3 | boundary_refined_anchors | 0.85 | 0.30 | 1.00 | 1.0 | 0.25 | 11.14 | 20.73 |
| 9 | high_ref_low_trimp_refine | 0.85 | 0.30 | 1.00 | 0.75 | 0.25 | 11.18 | 20.72 |

Boundary-refined best run by cohort:

| cohort | LOO MAE min | MAPE pct | R2 | best region |
| --- | ---: | ---: | ---: | --- |
| hardRunOrTrailRun | 11.35 | 8.79 | 0.972 | `hrr_reference=0.85`, `hrr_max_factor=1.2`, `trimp_scale=1.0`, `decay_lambda=0.25` |
| hardTrailRun | 17.98 | 9.98 | 0.969 | `hrr_reference=0.60`, `hrr_min_factor=0.40`, `hrr_max_factor=1.8`, `trimp_scale=2.0`, `decay_lambda=0.15` |
| selectedDateRaces | 7.35 | 4.68 | 0.994 | `hrr_reference=0.85`, `trimp_scale=0.75`, `decay_lambda=0.25` |
| top10HardTrailByHRR | 4.11 | 5.65 | 0.993 | `hrr_reference=0.90`, `trimp_scale=0.75`, `decay_lambda=0.25` |

Fatigue-state findings:

- The wide pass was dominated by decayed fatigue among best selections: decayed exponential median 7.61 min and decayed linear median 8.05 min.
- In the refined LOO pass, all LOO variants favored decayed states by median: decayed exponential 12.51 min and decayed linear 12.85 min.
- In best-per-run/cohort refined selections, cumulative exponential had the best median among its selected rows, but decayed models were selected more often.
- The global top refined run selected decayed linear for `hardRunOrTrailRun`, `hardTrailRun`, and `selectedDateRaces`, and decayed exponential for `top10HardTrailByHRR`.

Boundary interpretation:

- The previous best region was still under-expanded. The new refined best improved mean LOO MAE from 12.59 min to 11.09 min.
- The global LOO region moved from `hrr_reference=0.75`, `trimp_scale=2.0`, `decay_lambda=0.15` to `hrr_reference=0.85`, `trimp_scale=1.0`, `decay_lambda=0.25`.
- The best global `hrr_reference=0.85` is no longer at the upper tested reference bound because `0.90` was tested and ranked lower overall.
- The global best `hrr_max_factor` benefits slightly from reopening above 1.0, but the tie between 1.2 and 1.4 suggests the useful effect saturates quickly.
- The Stage 3 top-five refined fits no longer hit the reopened `alpha` or `fatigueCoef` upper bounds: top-fit `alpha` is 0.9-1.0 and `fatigueCoef` is 0.4-0.8.
- Keep the low-reference wide-clip family for hard-trail cohort-specific work; it still wins `hardTrailRun` LOO even though it is not the global average winner.

Artifacts:

- `data/exp_perf_predictions/trail_digital_twin_boundary_wide/benchmark_leaderboard.csv`
- `data/exp_perf_predictions/trail_digital_twin_boundary_wide/trail_digital_twin_benchmark_report.html`
- `data/exp_perf_predictions/trail_digital_twin_boundary_refined/benchmark_leaderboard.csv`
- `data/exp_perf_predictions/trail_digital_twin_boundary_refined/trail_digital_twin_benchmark_report.html`

## 2026-06-18 Distribution-Informed Hyperparameter Report

Context:

- Added a technical report that combines benchmark hyperparameter effects with cohort distribution statistics.
- Generated a fresh best-setting profile run at `hrr_reference=0.85`, `hrr_min_factor=0.30`, `hrr_max_factor=1.20`, `trimp_scale=1.0`, `decay_lambda=0.25`, `ctl_weight=0.05`, and `tsb_weight=0.10`.
- Computed cohort-level distributions for HRR amplitude, HRR threshold frequency, in-race TRIMP, pre-race TRIMP load states, terrain family mix, and residual/error strata.

Key distribution findings:

- `hardRunOrTrailRun` has 103 activities, median duration 64.6 min, median ascent 208 m, median activity HRR 0.724, and 38.2% of segment time at HRR >= 0.70.
- `hardTrailRun` has 45 activities, median duration 128.0 min, median ascent 782 m, median activity HRR 0.693, and only 28.9% of segment time at HRR >= 0.70.
- `top10HardTrailByHRR` is a narrow high-intensity slice: 90.2% of segment time is at HRR >= 0.70 and 36.2% at HRR >= 0.80.
- `selectedDateRaces` has the highest in-race TRIMP median at 5.05 and a P90 of 8.76, which explains why long-race residuals remain sensitive to fatigue-state choice.

Key model implications:

- The global best remains the high-reference region: `hrr_reference=0.85`, `trimp_scale=1.0`, and `decay_lambda=0.25`.
- The hard-trail-specific winner remains lower-reference and wider-clip because the trail-only cohort is longer, steeper, and has lower high-HRR frequency than the mixed hard-run cohort.
- Error increases sharply in the highest in-race TRIMP and duration tertiles; this is the next model weakness to target.
- More experiments are required, but they should be narrow and stress-stratified rather than another unconstrained wide search.

Recommended next experiments:

- Segment-objective sweep around `hrr_reference=0.82-0.88`, `hrr_max_factor=1.05-1.35`, `trimp_scale=0.75-1.25`, and `decay_lambda=0.20-0.30`.
- Hard-trail-specific low-reference verification around `hrr_reference=0.55-0.65`, `hrr_max_factor=1.4-2.0`, `trimp_scale=1.5-2.5`, and `decay_lambda=0.10-0.20`.
- Add duration-tertile, in-race-TRIMP-tertile, and HRR-frequency-tertile MAE to future benchmark reports.

Artifacts:

- `docs/science/trail_digital_twin_benchmark_hyperparameter_report.md`
- `docs/science/paper_assets/trail_digital_twin_cohort_distribution_summary.csv`
- `docs/science/paper_assets/trail_digital_twin_cohort_terrain_distribution.csv`
- `docs/science/paper_assets/trail_digital_twin_cohort_residual_distribution.csv`
- `docs/science/paper_assets/trail_digital_twin_cohort_error_strata.csv`
- `docs/science/paper_assets/trail_digital_twin_refinedTop20_hyperparameter_counts.csv`
- `docs/science/paper_assets/trail_digital_twin_wideTop50_hyperparameter_counts.csv`
- `data/exp_perf_predictions/trail_digital_twin_boundary_best_profile/`
