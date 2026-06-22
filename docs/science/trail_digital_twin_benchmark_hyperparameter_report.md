# Trail Digital Twin Hyperparameter Benchmark Report

## Technical Summary

- The best global activity-level setting from the boundary-expanded LOO refinement is `hrr_reference=0.85`, `hrr_min_factor=0.30`, `hrr_max_factor=1.20-1.40`, `trimp_scale=1.0`, `decay_lambda=0.25`, `ctl_weight=0.05`, and `tsb_weight=0.10`. It reached 11.09 min mean Stage 3 LOO MAE across the four benchmark cohorts, improving the previous refined best of 12.59 min.
- The expanded search fixed the previous boundary problem for Stage 3 fit parameters. The top-five refined fits use `alpha=0.9-1.0` and `fatigueCoef=0.4-0.8`, below the reopened upper grid limit of 1.20.
- Cohort distributions explain why one setting does not dominate every cohort. `hardRunOrTrailRun` is flatter and shorter with more high-HRR segment time, while `hardTrailRun` is longer, steeper, and has a wider HRR amplitude but lower high-HRR frequency. The low-reference wide-clip family still wins `hardTrailRun` specifically.
- More experiments are required, but they should be narrow: sweep segment-objective fitting around the new high-reference region, verify a hard-trail low-reference profile, and add duration/TRIMP-stratified validation rather than running another broad global search immediately.

## Scope and Metric Definitions

Primary comparison metric is Stage 3 activity-level leave-one-out MAE in minutes. Lower is better. The benchmark aggregate averages four configured cohorts: `hardTrailRun`, `hardRunOrTrailRun`, `top10HardTrailByHRR`, and `selectedDateRaces`.

Distribution metrics use the fresh profile run at the global best setting:

- Output: `data/exp_perf_predictions/trail_digital_twin_boundary_best_profile`.
- Activity distribution source: `anonymized_activity_features.csv`.
- Segment distribution source: `anonymized_segment_features.csv`.
- HRR amplitude is summarized as P90-P10.
- In-race TRIMP is the sum of leakage-free segment TRIMP inside an activity.
- Pre-race TRIMP load states are the model's pre-activity `ctl`, `tsb`, `trimpRediSlow`, and `trimpRediBalance` features. They are model-state units, not raw race TRIMP totals.

The cohorts overlap. For example, `top10HardTrailByHRR` is a subset of hard trail activities, and selected-date races can also appear inside broader hard cohorts.

## Best Hyperparameter Region

| question | evidence | interpretation |
| --- | --- | --- |
| Best global refined setting | Top refined LOO runs tie at `hrr_reference=0.85`, `hrr_min_factor=0.30`, `hrr_max_factor=1.20-1.40`, `trimp_scale=1.0`, `decay_lambda=0.25`, mean MAE 11.09 min. | Promote this as the current global activity-level default candidate. |
| HRR reference | In `high_ref_low_trimp_refine`, mean MAE by reference was 12.64 at 0.80, 11.50 at 0.85, and 11.89 at 0.90. | `0.85` is supported; `0.90` was tested and did not improve the aggregate. |
| HRR min/max clip | In `high_ref_clip_probe`, `hrr_min_factor=0.30` was best by mean MAE; `hrr_max_factor=1.20` and `1.40` tied. | Reopening max above 1.0 helps slightly, but the useful effect saturates by 1.20. |
| TRIMP scale | Refined top-20 runs used `trimp_scale=1.0` in 65% of rows, `1.5` in 25%, and `0.75` or `2.0` only rarely. | Use `1.0` globally; treat `0.75` and `1.5` as local alternatives. |
| Decay lambda | Refined top-20 runs used `decay_lambda=0.25` in 75% of rows. | `0.25` is the current best acute-fatigue decay candidate. |
| Readiness weights | Top refined rows all keep `ctl_weight=0.05`, `tsb_weight=0.10`; the `0.20/0.20` anchor worsened mean MAE to 13.01. | Do not promote heavier readiness weights globally. |

## Cohort Distribution Evidence

| cohort | n activities | duration median/P90 min | activity HRR median | segment HRR P90-P10 | time HRR >= 0.70 | time HRR >= 0.80 | in-race TRIMP median/P90 | CTL median | TSB median |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardRunOrTrailRun | 103 | 64.6 / 232.3 | 0.724 | 0.330 | 38.2% | 10.1% | 1.88 / 6.50 | 0.688 | 0.053 |
| hardTrailRun | 45 | 128.0 / 497.9 | 0.693 | 0.372 | 28.9% | 6.7% | 2.79 / 7.05 | 1.011 | 0.034 |
| selectedDateRaces | 18 | 150.0 / 323.0 | 0.752 | 0.249 | 53.6% | 13.7% | 5.05 / 8.76 | 0.973 | 0.036 |
| top10HardTrailByHRR | 10 | 74.6 / 181.8 | 0.768 | 0.158 | 90.2% | 36.2% | 3.08 / 6.67 | 0.980 | 0.082 |

The mixed hard-run cohort has many shorter, flatter activities but a high segment-HRR frequency. That supports the high `hrr_reference=0.85` region: many mixed-cohort segments sit near or above the reference, so the HRR speed-ratio term has enough signal without requiring a very wide clip cap.

The hard-trail cohort has the widest segment-HRR amplitude and much longer high-tail duration. It also has lower high-HRR frequency than the mixed cohort. That helps explain why its cohort-specific LOO winner remains a lower-reference, wider-clip profile: `hrr_reference=0.60`, `hrr_min_factor=0.40`, `hrr_max_factor=1.8`, `trimp_scale=2.0`, `decay_lambda=0.15`.

## Terrain and HRR Frequency

| cohort | dominant terrain by time | interpretation |
| --- | --- | --- |
| hardRunOrTrailRun | flat 43.1%, climb 18.6%, descent 12.8% | Flatter distribution makes HRR speed ratio more important than grade-only cost for many activities. |
| hardTrailRun | climb 26.4%, flat 20.2%, descent 18.4%, mixed 14.2% | More balanced terrain and longer climbs make a single HRR reference less stable. |
| selectedDateRaces | flat 26.4%, climb 23.7%, mixed 16.9%, descent 16.1% | Long race efforts combine high TRIMP accumulation with mixed terrain; fatigue-state choice matters more. |
| top10HardTrailByHRR | climb 33.4%, flat 21.5%, descent 18.4% | High-HRR trail efforts are mostly climb-heavy and consistently intense. |

## Error Distribution Under the Best Global Setting

| cohort | mean absolute error min | median absolute error min | P90 absolute error min | mean bias min | over-predicted share |
| --- | ---: | ---: | ---: | ---: | ---: |
| hardRunOrTrailRun | 8.90 | 4.93 | 24.73 | 2.44 | 55.3% |
| hardTrailRun | 17.00 | 13.00 | 38.84 | 11.87 | 77.8% |
| selectedDateRaces | 12.24 | 6.08 | 26.51 | -9.50 | 22.2% |
| top10HardTrailByHRR | 4.81 | 3.92 | 8.96 | -0.40 | 50.0% |

The global best is well balanced for `hardRunOrTrailRun` and `top10HardTrailByHRR`, but it over-predicts hard-trail duration and under-predicts selected-date races. That is a practical reason to keep cohort-specific calibration in the next pass instead of relying only on the aggregate leaderboard.

Error increases with stress exposure:

- In `hardRunOrTrailRun`, the top in-race TRIMP tertile has 17.0 min MAE versus 3.5 min in the lowest tertile.
- In `hardTrailRun`, the top in-race TRIMP tertile has 26.1 min MAE versus 8.7 min in the lowest tertile.
- In `selectedDateRaces`, the top duration/TRIMP tertile is strongly under-predicted. The top in-race TRIMP tertile has -19.7 min mean bias.

This suggests that the acute fatigue term is directionally useful but still too simple for long race stress. A single decayed or cumulative scalar does not fully describe late-race degradation in the longest activities.

## Fatigue-State Impact

Across all refined LOO variants:

| fatigue state | shape | count | median MAE min | min MAE min |
| --- | --- | ---: | ---: | ---: |
| decayed | exponential | 248 | 12.51 | 4.29 |
| decayed | linear | 248 | 12.85 | 4.11 |
| cumulative | exponential | 248 | 13.07 | 5.81 |
| cumulative | linear | 244 | 13.41 | 5.65 |

The distribution favors decayed fatigue overall, and the global top refined run selects decayed linear for `hardRunOrTrailRun`, `hardTrailRun`, and `selectedDateRaces`, plus decayed exponential for `top10HardTrailByHRR`.

The caveat is cohort interaction. In broad medians, selected-date races sometimes prefer cumulative exponential, while the global top setting still uses decayed linear for that cohort. This is not a contradiction; it means fatigue state and physiology hyperparameters interact. The next sweep should test fatigue state inside the cohort-specific refined regions, not only as isolated one-off variants.

## What Looks Best

- Global activity default: `hrr_reference=0.85`, `hrr_min_factor=0.30`, `hrr_max_factor=1.20`, `trimp_scale=1.0`, `decay_lambda=0.25`, decayed fatigue enabled, `ctl_weight=0.05`, `tsb_weight=0.10`.
- Hard-trail candidate: low-reference wide-clip family around `hrr_reference=0.60`, `hrr_min_factor=0.40`, `hrr_max_factor=1.4-1.8`, `trimp_scale=2.0`, `decay_lambda=0.15`.
- High-HRR trail candidate: high reference around `0.90`, low TRIMP scale around `0.75-1.5`, `decay_lambda=0.25`.
- Fatigue default: keep both decayed linear and decayed exponential in the grid. Do not remove cumulative fatigue yet for selected-date race diagnostics.

## What Does Not Look Promising

- Old broad defaults around `hrr_reference=0.70`, `trimp_scale=10.0`, and `decay_lambda=0.30` are no longer competitive.
- Globally heavier readiness weights are not supported by this batch. The `ctl_weight=0.20`, `tsb_weight=0.20` anchor is worse than the default readiness pair.
- Very wide HRR max caps above 1.4 do not improve the global high-reference region. They are useful mainly in low-reference hard-trail variants.
- Another blind wide search is not the best next step. The current error distribution points to cohort-specific and stress-stratified refinement.

## Recommended Next Experiments

1. Run a focused segment-objective sweep around the global region:
   - `hrr_reference`: 0.82, 0.85, 0.88
   - `hrr_max_factor`: 1.05, 1.20, 1.35
   - `trimp_scale`: 0.75, 1.00, 1.25
   - `decay_lambda`: 0.20, 0.25, 0.30
   - objectives: `activity`, `segment`, and `both` only for the final shortlist.

2. Run a hard-trail cohort-specific verification:
   - `hrr_reference`: 0.55, 0.60, 0.65
   - `hrr_min_factor`: 0.30, 0.40, 0.50
   - `hrr_max_factor`: 1.40, 1.60, 1.80, 2.00
   - `trimp_scale`: 1.5, 2.0, 2.5
   - `decay_lambda`: 0.10, 0.15, 0.20

3. Add stress-stratified validation:
   - Report MAE by duration tertile, in-race TRIMP tertile, and HRR-frequency tertile.
   - Promote a setting only if it improves the long-duration/high-TRIMP tail without degrading the median activity too much.

4. Add uncertainty estimates before manuscript-level claims:
   - Bootstrap activity-level MAE by cohort.
   - Repeat LOO with the selected-date races separated from the training target when the evaluation is race-specific.
   - Keep cohort overlap explicit so `top10HardTrailByHRR` does not overstate evidence independent from `hardTrailRun`.

## Supporting Artifacts

- `docs/science/paper_assets/trail_digital_twin_cohort_distribution_summary.csv`
- `docs/science/paper_assets/trail_digital_twin_cohort_terrain_distribution.csv`
- `docs/science/paper_assets/trail_digital_twin_cohort_residual_distribution.csv`
- `docs/science/paper_assets/trail_digital_twin_cohort_error_strata.csv`
- `docs/science/paper_assets/trail_digital_twin_refinedTop20_hyperparameter_counts.csv`
- `docs/science/paper_assets/trail_digital_twin_wideTop50_hyperparameter_counts.csv`
- `data/exp_perf_predictions/trail_digital_twin_boundary_best_profile/`
- `data/exp_perf_predictions/trail_digital_twin_boundary_refined/`
- `data/exp_perf_predictions/trail_digital_twin_boundary_wide/`
