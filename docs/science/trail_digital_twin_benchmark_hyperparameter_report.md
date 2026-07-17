# Trail Digital Twin Benchmark Findings

Last updated: 2026-06-23.

## Technical Summary

- The strongest archived profile across all saved benchmark folders remains the older
  boundary-wide high-reference profile: `hrr_reference=0.85`, `hrr_min_factor=0.30`,
  `hrr_max_factor=1.00`, `decay_lambda=0.25`, `min_fatigue_factor=0.50`,
  `ctl_weight=0.05`, and `tsb_weight=0.10`. It reached 10.49 min mean Stage 3 MAE
  across the four benchmark cohorts, but should be revalidated under the current
  benchmark surface before being treated as the production default.
- The two-stage hypothesis benchmark found a stronger current-code recipe than the
  previous combined-fatigue sweep: `hrr_reference=0.88`, `hrr_min_factor=0.30`,
  `hrr_max_factor=1.00`, `decay_lambda=0.20`, and `min_fatigue_factor=0.60`.
  It reached 10.845 min mean Stage 3 LOO MAE, 7.91% mean MAPE, and 0.982 mean R2.
- The high-reference family is confirmed for the aggregate objective. The hard-trail
  low-reference family is not the default: its best hard-trail run had lower MAPE but
  worse MAE and a large negative bias, so it is a diagnostic family rather than the
  main recipe.
- The main modeling result is stable: Stage 3 HRR speed-ratio modeling dominates the
  reproduction/CTL/REDI-only stages. In the refined hypothesis batch, average Stage 3
  LOO MAE was 12.87 min versus 25.83 min for Stage 2 REDI LOO, 25.68 min for Stage 1
  CTL LOO, and 32.93 min for the Stage 0 reproduction LOO.
- Decayed in-race TRIMP is now the safest default fatigue representation. In the
  refined hypothesis batch, the best aggregate run selected plain `decayed` fatigue
  for all four cohorts, with no positive secondary muscular term.
- The muscular-fatigue term should stay in the grid, not be hard-coded. Positive
  secondary coefficients were selected in only 3 of 192 final Stage 3 rows in the
  refined hypothesis batch.
- A one-run Minetti clamp probe widened the GAP grade range from +/-45% to +/-75% and
  filtered raw grade spikes above 100%. It improved steep-terrain segment metrics, but
  worsened the aggregate Stage 3 LOO score, so it should be treated as a terrain-model
  experiment until a full resweep confirms a new optimum.

## Scope and Metrics

Primary benchmark metric is activity-level Stage 3 leave-one-out MAE in minutes.
Lower is better. The aggregate benchmark averages these configured cohorts:

- `hardTrailRun`
- `hardRunOrTrailRun`
- `top10HardTrailByHRR`
- `selectedDateRaces`

The cohorts overlap. `top10HardTrailByHRR` is a high-intensity subset of hard trail
activities, and selected-date races can also appear in broader hard cohorts. Treat
aggregate results as a model-selection guide, not an independent sample-size claim.

Key source artifacts:

- `data/exp_perf_predictions/trail_digital_twin_best_factor_sweep/`
- `data/exp_perf_predictions/trail_digital_twin_boundary_wide/`
- `data/exp_perf_predictions/trail_digital_twin_boundary_refined/`
- `data/exp_perf_predictions/trail_digital_twin_boundary_best_profile/`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_screen/`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_refined/`
- `configs/trail_digital_twin_extensions.yaml`
- `configs/trail_digital_twin_benchmark.yaml`
- `configs/trail_digital_twin_benchmark_hypothesis_screen.yaml`
- `configs/trail_digital_twin_benchmark_hypothesis_refined.yaml`

## Best Observed Benchmark Profiles

| benchmark folder | best mean Stage 3 MAE min | mean MAPE pct | mean R2 | best profile interpretation |
| --- | ---: | ---: | ---: | --- |
| `trail_digital_twin_boundary_wide` | 10.49 | 7.76 | 0.982 | Best historical global result; high HRR reference, no upper HRR boost above 1.0. |
| `trail_digital_twin_hypothesis_screen` | 10.84 | 7.91 | 0.982 | Best current-code screen; confirms the high-reference family and moves the optimum toward `hrr_reference=0.88`. |
| `trail_digital_twin_hypothesis_refined` | 10.84 | 7.91 | 0.982 | Stage 2 confirmation of the screen winner; same aggregate optimum. |
| `trail_digital_twin_boundary_refined` | 11.09 | 8.44 | 0.982 | Best refined LOO confirmation before the new muscular-fatigue variants. |
| `trail_digital_twin_best_factor_sweep` | 11.33 | 8.26 | 0.980 | Best current run with the combined-fatigue model family enabled. |
| `trail_digital_twin_wide_screen` | 11.66 | 8.73 | 0.977 | Earlier wide screen; useful mostly as a search-routing artifact. |
| `trail_digital_twin_refined_screen` | 12.59 | 9.04 | 0.975 | Earlier refined LOO pass before boundary expansion. |
| `trail_digital_twin_best_loo` | 19.00 | 12.61 | 0.948 | Low-reference LOO check; not globally competitive. |
| `trail_digital_twin_reference_loo` | 26.30 | 23.71 | 0.942 | Matched reference baseline; clearly obsolete. |

The boundary-wide winner is still the absolute best archived result, but the
hypothesis benchmark is the best current-code validation pass and should drive the
operational default. The right interpretation is not that the older boundary-wide
profile is wrong; it is that the current fitting surface prefers a slightly higher HRR
reference, a faster decay, and a higher fatigue floor.

## Two-Stage Hypothesis Validation

The hypothesis batch directly tested the five recommendations from the previous report.
Stage 1 screened 165 runs; Stage 2 refined the Stage 1 high-reference and hard-trail
winners into 48 runs. The HTML reports contain the analysis-ready visual evidence:

- `data/exp_perf_predictions/trail_digital_twin_hypothesis_screen/trail_digital_twin_benchmark_report.html`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_refined/trail_digital_twin_benchmark_report.html`

| hypothesis | verdict | evidence | modeling implication |
| --- | --- | --- | --- |
| H1 high-reference family | confirmed | Screen and refined winners both use `hrr_reference=0.88`, `hrr_min_factor=0.30`, `hrr_max_factor=1.00`, `decay_lambda=0.20`, `min_fatigue_factor=0.60`; refined mean Stage 3 LOO MAE is 10.845 min. | Use high-reference as the global default family. |
| H2 alpha/kappa and muscular-fatigue grid | partially confirmed | Best refined run selects `alpha=0.90` and plain decayed fatigue for all cohorts; positive secondary fatigue is selected in only 3 of 192 final Stage 3 rows. | Keep the secondary-muscle term available but default to zero unless validation selects it. |
| H3 hard-trail low-reference family | rejected as default, retained as diagnostic | Best low-reference hard-trail run reaches 19.43 min MAE, versus 18.84 min for the best high-reference hard-trail row; it has lower MAPE but a large negative bias. | Do not use low-reference/wide-clip as the default. Use it to diagnose hard-trail bias and HRR ceiling assumptions. |
| H4 stress and terrain residual strata | confirmed | For the best refined aggregate run, high-duration hard-trail activities have 28.71 min MAE versus 5.15 min in the low-duration tertile; high in-race TRIMP has 30.65 min MAE versus 8.94 min in the low tertile. | Remaining errors are stress-duration and terrain-mechanics errors, not only fatigue-state errors. |
| H5 bootstrap uncertainty | confirmed | Top aggregate runs have overlapping cohort-level intervals; for the winner, hardTrailRun MAE is 19.03 min with 14.41-24.06 min bootstrap interval, and hardRunOrTrailRun is 12.24 min with 9.66-15.16 min interval. | Do not promote sub-minute leaderboard differences as robust. |

Stage 2 confirms the screen winner instead of finding a new basin. Within 0.50 min of
the refined optimum, every run is high-reference with `hrr_reference=0.88`,
`hrr_min_factor=0.30`, `hrr_max_factor in {1.00,1.10}`,
`decay_lambda in {0.20,0.25}`, and `min_fatigue_factor in {0.50,0.60}`. Within
1.00 min, `hrr_reference=0.85` also appears, but the low-reference hard-trail family
does not enter the aggregate frontier.

## Recommended Default Profile

Use this as the current production/default candidate for activity-level prediction:

| factor | recommended value | evidence and guideline |
| --- | ---: | --- |
| HRR reference | `0.88` for the current aggregate default; keep `0.85` as the hard-trail/race-specific challenger | Stage 2 confirms `0.88` for aggregate MAE. `0.85` remains close and gives the best hardTrailRun row. |
| HRR minimum factor | `0.30` | Universal across the high-reference frontier. Low-reference hard-trail runs needed `0.50`, but did not beat high-reference MAE. |
| HRR maximum factor | `1.00` default, with `1.10` as the first alternate | Best refined aggregate run uses `1.00`; near-frontier runs include `1.10`. `1.20` is no longer needed for the current aggregate default. |
| Decay lambda | `0.20` default, keep `0.25` in confirmation grids | Best refined aggregate run uses `0.20`; `0.25` is within 0.25 min of the optimum and matches older winners. |
| Minimum fatigue factor | `0.60` aggregate default; test `0.50` for hard-trail-specific fits | `0.60` wins aggregate MAE; `0.50` gives the best hardTrailRun rows. |
| Readiness weights | `ctl_weight=0.05`, `tsb_weight=0.10` | All global winners keep the light readiness pair. Heavier readiness is cohort-specific at best. |
| Readiness clipping | `ctl_factor_min=0.90`, `ctl_factor_max=1.10` | Current global winner uses this narrow clip. Do not widen without a dedicated readiness sweep. |
| Fit objective | `activity` | The global ranking uses activity-level LOO. Segment objective is promising but not yet swept enough for default promotion. |

## Best Stage 3 Parameters by Cohort

Best refined hypothesis aggregate winner:

| cohort | alpha | fatigue state | fatigue shape | primary coef | secondary load | secondary coef | race MAE min | race MAPE pct | race R2 |
| --- | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| `hardTrailRun` | 0.90 | `decayed` | exponential | 0.60 | none | 0.00 | 19.03 | 11.93 | 0.972 |
| `hardRunOrTrailRun` | 0.90 | `decayed` | linear | 0.40 | none | 0.00 | 11.69 | 9.43 | 0.973 |
| `top10HardTrailByHRR` | 0.90 | `decayed` | exponential | 0.30 | none | 0.00 | 4.24 | 5.72 | 0.991 |
| `selectedDateRaces` | 0.90 | `decayed` | linear | 0.30 | none | 0.00 | 7.87 | 4.50 | 0.995 |

Absolute historical boundary-wide winner:

| cohort | alpha | fatigue state | fatigue shape | fatigue coef | race MAE min | race MAPE pct | race R2 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `hardTrailRun` | 1.00 | `decayed` | exponential | 0.80 | 18.30 | 10.39 | 0.969 |
| `hardRunOrTrailRun` | 1.00 | `decayed` | exponential | 0.80 | 11.96 | 9.63 | 0.973 |
| `top10HardTrailByHRR` | 0.90 | `decayed` | linear | 0.30 | 3.99 | 5.97 | 0.991 |
| `selectedDateRaces` | 1.00 | `decayed` | linear | 0.40 | 7.70 | 5.03 | 0.993 |

Interpretation:

- `alpha=0.90` is the best current aggregate setting across all four cohorts. Keep
  the wider `0.85-1.05` grid because `0.85` is still the most common final selection
  across all refined rows, and older boundary-wide profiles selected `0.90-1.00`.
- The best aggregate run disabling the secondary term in every cohort is important:
  the muscular-fatigue hypothesis is not rejected, but it is not a universal term.
- The hard-trail-specific frontier prefers `min_fatigue_factor=0.50`, while the
  aggregate frontier prefers `0.60`. Treat this as the main cohort-specific split.

## Fatigue-State Findings

Refined hypothesis LOO comparison across fatigue-state candidates:

| fatigue state | shape | mean LOO MAE min | median LOO MAE min | mean MAPE pct | mean R2 |
| --- | --- | ---: | ---: | ---: | ---: |
| `decayed_cumulative` | exponential + cumulative TRIMP | 13.63 | 13.30 | 9.10 | 0.970 |
| `decayed` | linear | 13.70 | 13.23 | 9.12 | 0.970 |
| `decayed_progress` | exponential + progress | 13.77 | 13.45 | 9.06 | 0.970 |
| `decayed` | exponential | 13.84 | 13.12 | 9.21 | 0.970 |
| `cumulative` | exponential | 14.23 | 14.02 | 9.75 | 0.970 |
| `cumulative` | linear | 14.34 | 13.93 | 9.76 | 0.970 |

Previous combined-fatigue sweep final selected Stage 3 rows:

| selected fatigue state | selected shape | selection count |
| --- | --- | ---: |
| `decayed` | exponential | 39 |
| `decayed` | linear | 19 |
| `cumulative` | exponential | 16 |
| `decayed_progress` | exponential + progress | 13 |
| `cumulative` | linear | 3 |
| `decayed_cumulative` | exponential + cumulative TRIMP | 1 |
| `progress` | linear | 1 |

Refined hypothesis batch final selected Stage 3 rows:

| selected fatigue state | selected shape | selection count |
| --- | --- | ---: |
| `decayed` | exponential | 65 |
| `decayed` | linear | 60 |
| `cumulative` | exponential | 32 |
| `cumulative` | linear | 30 |
| `decayed_progress` | exponential + progress | 5 |

The candidate-level table is intentionally close: the four best states are separated
by only 0.21 min mean LOO MAE. Secondary fatigue was selected only 3 times in 192
refined final rows, always at `secondaryFatigueCoef=0.10`. The best aggregate run
selected no secondary term in any cohort.

Guidelines:

- Keep `decayed` linear and `decayed` exponential as the default candidates.
- Keep `decayed_progress` as an optional muscular-fatigue candidate, but do not force it.
- Keep `decayed_cumulative` for stress testing and manuscript diagnostics, not as the
  sole default recipe.
- Keep pure `cumulative` for race-specific diagnostics and as a contrast against
  decayed short-term fatigue.
- Do not use pure `progress` as a default. It remains a diagnostic baseline.
- Include `secondaryFatigueCoef=0.0` in every muscular-fatigue grid. The benchmark
  often chooses to disable the secondary term.
- If the secondary term is positive, `0.10` is the most defensible starting value.

## Stage-Level Model Choice

Refined hypothesis batch mean results by stage:

| stage | mean MAE min | median MAE min | mean R2 | interpretation |
| --- | ---: | ---: | ---: | --- |
| Stage 3 HRR speed ratio | 12.62 | 11.79 | 0.971 | Best in-sample stage. |
| Stage 3 HRR speed ratio LOO | 12.87 | 12.43 | 0.970 | Best validated stage; use for model selection. |
| Stage 1 TRIMP fatigue CTL | 25.53 | 24.69 | 0.873 | CTL readiness alone is not enough. |
| Stage 2 TRIMP fatigue REDI | 25.54 | 24.88 | 0.870 | REDI readiness alone is not enough. |
| Stage 1 TRIMP fatigue CTL LOO | 25.68 | 24.69 | 0.872 | Validated CTL-only stage. |
| Stage 2 TRIMP fatigue REDI LOO | 25.83 | 25.05 | 0.869 | Validated REDI-only stage. |
| Stage 0 reproduction Stage 3 | 31.38 | 35.22 | 0.906 | Baseline reproduction. |
| Stage 0 reproduction Stage 3 LOO | 32.93 | 37.25 | 0.894 | Baseline validation. |

Stage 3 is the only current modeling family that should be used for prediction. The
earlier stages are useful only as ablations and manuscript comparison baselines.

## Terrain Error Pattern

Best refined hypothesis aggregate winner, weighted across benchmark cohorts by segment count:

| terrain label | mean MAE min per segment | mean MAPE pct | mean bias min | mean R2 | guideline |
| --- | ---: | ---: | ---: | ---: | --- |
| Steep ascent | 3.56 | 22.65 | +3.07 | -0.18 | Main over-prediction risk; keep GAP/grade diagnostics visible. |
| Steep descent | 3.33 | 34.16 | -3.31 | -0.31 | Main under-prediction risk; descent recovery and technical descent cost remain weak. |
| Ascent | 1.85 | 17.66 | +1.26 | 0.54 | Better than steep ascent but still biased slow. |
| Mixed climb/descent | 1.38 | 14.32 | +0.12 | 0.75 | Current mixed terrain treatment is acceptable. |
| Descent | 1.36 | 16.32 | -1.07 | 0.59 | Descent is still biased fast. |
| Flat | 0.84 | 14.18 | +0.37 | 0.78 | Best-supported segment family. |

The model is not only a fatigue model. Terrain-specific residuals remain large enough
that a fatigue improvement can be hidden by steep ascent/descent error. Future fatigue
claims should report terrain-stratified residuals.

## Minetti 75 Percent Clamp Probe

The first refined high-reference run was rerun with Minetti GAP clamped at +/-75%
instead of +/-45%, plus raw GPS grade outlier filtering: samples with absolute grade
above 100% are interpolated before smoothing. Output:

- `data/exp_perf_predictions/trail_digital_twin_minetti075_probe/`

| metric | old +/-45% clamp | new +/-75% clamp + outlier filter | delta |
| --- | ---: | ---: | ---: |
| Aggregate Stage 3 LOO MAE min | 10.845 | 11.173 | +0.328 |
| Aggregate Stage 3 LOO MAPE pct | 7.915 | 8.329 | +0.414 |
| Steep climb segment MAE min | 3.560 | 3.240 | -0.320 |
| Steep climb segment bias min | +3.068 | +2.710 | -0.357 |
| Steep descent segment MAE min | 3.330 | 3.195 | -0.135 |
| Steep descent segment bias min | -3.309 | -3.142 | +0.167 |
| Flat segment MAE min | 0.836 | 0.936 | +0.100 |

Interpretation: the wider Minetti range moves steep terrain in the right direction, but
it is not a free aggregate improvement. It likely changes the selected Stage 3
fatigue/HRR coefficients and should be evaluated with a full refined resweep before
being promoted as the default.

## What Is Robust, Universal, or Athlete-Specific

Robust across the current benchmark:

- Stage 3 HRR speed-ratio modeling is the only family ready for prediction. CTL/REDI
  readiness-only stages remain ablation baselines.
- Light readiness scaling is safe: `ctl_weight=0.05`, `tsb_weight=0.10`, and clipping
  to `0.90-1.10`. Treat this as a small correction term, not the main model.
- `hrr_min_factor=0.30` is stable for the high-reference frontier.
- `decayed` fatigue should be in every default grid. It is the selected state for the
  best refined aggregate run in all four cohorts.
- The Stage 3 selection grid can stay compact:
  `alpha={0.85,0.90,0.95,1.00,1.05}`,
  `fatigueCoef={0.20,0.30,0.40,0.60,0.80}`,
  `secondaryFatigueCoef={0.00,0.10,0.20}`.

Almost universal defaults for this athlete and current code:

- `hrr_reference=0.88`, `hrr_min_factor=0.30`, `hrr_max_factor=1.00`,
  `decay_lambda=0.20`, `min_fatigue_factor=0.60`.
- `alpha=0.90` and plain decayed fatigue for aggregate race-time prediction.
- `hrr_max_factor` is weakly identified between `1.00` and `1.10`; do not over-interpret
  this knob unless a new athlete shows systematic HRR clipping.

Fine-tune per athlete:

- `hrr_reference`: this encodes the athlete's sustainable race HRR. It should move with
  physiology, heat, altitude, taper, and race duration.
- `decay_lambda`: this is short-term cardio recovery speed. Athletes who recover quickly
  after descents/rests can support a larger lambda; slow-recovery athletes need a smaller
  lambda.
- `min_fatigue_factor`: this is the floor on how much the model can slow the athlete.
  The aggregate winner prefers `0.60`, but hard-trail rows prefer `0.50`.
- Stage 3 `alpha` and `fatigueCoef`: these translate HRR and in-race load into speed.
  They should be fit from each athlete's race history rather than copied blindly.
- Terrain cost and descent handling: steep ascent/descent errors are large enough that
  athlete-specific technical skill and downhill tolerance probably matter more than small
  fatigue-coefficient changes.

Very specific to this dataset:

- The exact 10.845 min MAE and the `0.88/0.20/0.60` optimum are from overlapping cohorts
  on one athlete's available activity history. They are good defaults for this project,
  not population-level constants.
- The low-reference hard-trail family reduces MAPE but introduces a large negative bias.
  That pattern may be specific to the current hard-trail subset and should not be used as
  a generic hard-trail rule.

## Where the Errors Come From

The main error driver is not the fatigue equation alone. For the best refined aggregate
run:

- Long and high-load activities dominate activity-level error. In `hardTrailRun`,
  high-duration activities have 28.71 min MAE versus 5.15 min in the low-duration
  tertile. High in-race TRIMP has 30.65 min MAE versus 8.94 min in the low-TRIMP tertile.
- Steep ascent and steep descent dominate segment-level residuals. Steep ascent is
  biased too slow (`+3.07` min per segment on average); steep descent is biased too fast
  (`-3.31` min per segment on average). That sign pattern is physically coherent:
  uphill cost is still too punitive or pacing is under-modeled, while downhill technical
  cost, braking, surface, and recovery are under-penalized.
- HRR-frequency alone does not explain the errors. In hard trail, low HRR>=0.70 share
  still has high MAE, which points to terrain, hiking, technicality, pauses, and route
  specificity rather than only cardio load.
- Bootstrap intervals are wide enough that small leaderboard deltas are noise. For the
  refined winner, hardTrailRun MAE is 19.03 min with a 14.41-24.06 min bootstrap
  interval; hardRunOrTrailRun is 12.24 min with a 9.66-15.16 min interval.

Practical consequence: future improvement should target terrain mechanics and
activity-context features before adding more fatigue-state complexity.

## Modeling Guidelines

Use this decision rule for the next modeling pass:

1. Select by `Stage 3 HRR speed ratio LOO`, activity objective, averaged over the four
   benchmark cohorts.
2. Use the high-reference family as the global default:
   `hrr_reference=0.88`, `hrr_min_factor=0.30`, `hrr_max_factor in {1.00,1.10}`,
   `decay_lambda in {0.20,0.25}`, `min_fatigue_factor in {0.50,0.60}`.
3. Fit cohort-specific Stage 3 fatigue states rather than forcing a single fatigue
   form across cohorts.
4. Keep the fatigue grid narrow and interpretable:
   `alpha in {0.85,0.90,0.95,1.00,1.05}`,
   `fatigueCoef in {0.20,0.30,0.40,0.60,0.80}`,
   `secondaryFatigueCoef in {0.00,0.10,0.20}`.
5. Keep these fatigue states in the next confirmation run:
   `decayed` linear, `decayed` exponential, `decayed_progress` exponential, and
   `decayed_cumulative` exponential.
6. Keep pure `cumulative` only for race-specific diagnostics and ablations unless it
   wins again in selected-date race validation.
7. Do not increase readiness weights globally. Use `ctl_weight=0.05` and
   `tsb_weight=0.10` until a dedicated readiness benchmark proves otherwise.
8. Report error by cohort, terrain family, duration tertile, and in-race TRIMP tertile
   before treating a fatigue change as a real improvement.

## Recommended Next Experiments

The five previous recommendations were executed in
`trail_digital_twin_hypothesis_screen` and `trail_digital_twin_hypothesis_refined`.
The next batch should stop expanding physiology grids and instead isolate the remaining
error sources.

1. Terrain-mechanics correction:
   - Fit separate steep-ascent and steep-descent correction terms.
   - Run a full refined resweep with the +/-75% Minetti clamp and grade-spike filter,
     because the one-run probe improved steep terrain while worsening the aggregate.
   - Add descent technicality proxies: grade variance, switchbacks/curvature if route
     geometry supports it, surface proxy when available, and sustained descent length.
   - Validate by terrain family first, then by activity-level MAE.

2. Route-context and long-duration correction:
   - Add duration and cumulative-distance interactions to the Stage 3 model.
   - Test whether long high-TRIMP residuals are systematic after controlling for terrain.
   - Keep the benchmark strata tables as required outputs.

3. Athlete-specific calibration protocol:
   - Fit `hrr_reference`, `decay_lambda`, and `min_fatigue_factor` per athlete using
     race-only or hard-trail-only folds.
   - Compare a global prior plus athlete-specific shrinkage against fully independent
     per-athlete fits once more athletes are available.

4. Race-day feasibility model:
   - Combine the current route prediction with the HRR-duration power-law envelope.
   - Penalize candidate HRR plans that exceed sustainable duration, rather than relying
     only on constant HRR sweep feasibility.

5. Uncertainty and fold design:
   - Keep deterministic bootstrap intervals, but add blocked folds by race/date so
     overlapping cohorts do not make results look more certain than they are.
   - Report winner differences only when they clear the bootstrap interval overlap or
     repeat across blocked folds.

## Practical Defaults for Current Use

If a single config must be chosen now:

```yaml
physiology:
  hrr_reference: 0.88
  hrr_min_factor: 0.30
  hrr_max_factor: 1.00
  decay_lambda: 0.20
  min_fatigue_factor: 0.60

readiness:
  ctl_weight: 0.05
  tsb_weight: 0.10
  ctl_factor_min: 0.90
  ctl_factor_max: 1.10

fitting:
  enabled_objectives:
    - activity
  validation_modes:
    - in_sample
    - loo
  hrr_trimp_alpha_grid:
    - 0.85
    - 0.90
    - 0.95
    - 1.00
    - 1.05
  hrr_trimp_kappa_grid:
    - 0.20
    - 0.30
    - 0.40
    - 0.60
    - 0.80
  hrr_trimp_secondary_kappa_grid:
    - 0.00
    - 0.10
    - 0.20
```

This is a conservative operational default for the current code path. For a hard-trail
only race prediction, keep a local challenger with `hrr_reference=0.85`,
`hrr_max_factor=1.10`, `decay_lambda=0.20`, and `min_fatigue_factor=0.50`, because that
family produced the best hardTrailRun MAE even though it did not win the aggregate.

## Claims to Avoid

- Do not claim that the muscular-fatigue term is globally proven. The evidence supports
  keeping it as a candidate and using it for hard-trail stress, not forcing it everywhere.
- Do not claim that the low-reference hard-trail family is better. It has an interesting
  MAPE/bias profile, but it loses the primary hardTrailRun MAE comparison.
- Do not claim that cumulative fatigue beats decayed fatigue globally. It is useful for
  selected race diagnostics, but decayed forms dominate selections and historical winners.
- Do not treat the archived 10.49 min boundary-wide score and the current 10.845 min
  hypothesis score as the same benchmark surface. The archived score remains important,
  but the current-code validation should drive operational defaults.
- Do not claim that readiness scaling is paper-derived. The local readiness factors are
  pragmatic bounded multipliers; paper-backed training-response models motivate the
  structure but do not provide these exact coefficients.
- Do not rank models on in-sample Stage 3 alone. Use LOO as the primary benchmark.
- Do not ignore terrain residuals when interpreting fatigue changes. Steep ascents and
  steep descents remain the largest per-segment error sources.

## Supporting Artifacts

- `data/exp_perf_predictions/trail_digital_twin_best_factor_sweep/benchmark_leaderboard.csv`
- `data/exp_perf_predictions/trail_digital_twin_best_factor_sweep/benchmark_fitted_parameters.csv`
- `data/exp_perf_predictions/trail_digital_twin_best_factor_sweep/benchmark_stage3_fatigue.csv`
- `data/exp_perf_predictions/trail_digital_twin_best_factor_sweep/benchmark_stage_metrics.csv`
- `data/exp_perf_predictions/trail_digital_twin_best_factor_sweep/benchmark_segment_type_metrics.csv`
- `data/exp_perf_predictions/trail_digital_twin_best_factor_sweep/trail_digital_twin_benchmark_report.html`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_screen/benchmark_leaderboard.csv`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_screen/benchmark_hrr_trimp_grid_search.csv`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_screen/benchmark_activity_error_strata.csv`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_screen/benchmark_bootstrap_uncertainty.csv`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_screen/trail_digital_twin_benchmark_report.html`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_refined/benchmark_leaderboard.csv`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_refined/benchmark_stage_metrics.csv`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_refined/benchmark_fitted_parameters.csv`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_refined/benchmark_hrr_trimp_grid_search.csv`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_refined/benchmark_segment_type_metrics.csv`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_refined/benchmark_activity_error_strata.csv`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_refined/benchmark_bootstrap_uncertainty.csv`
- `data/exp_perf_predictions/trail_digital_twin_hypothesis_refined/trail_digital_twin_benchmark_report.html`
- `data/exp_perf_predictions/trail_digital_twin_minetti075_probe/benchmark_leaderboard.csv`
- `data/exp_perf_predictions/trail_digital_twin_minetti075_probe/benchmark_segment_type_metrics.csv`
- `data/exp_perf_predictions/trail_digital_twin_minetti075_probe/trail_digital_twin_benchmark_report.html`
- `data/exp_perf_predictions/trail_digital_twin_boundary_wide/benchmark_leaderboard.csv`
- `data/exp_perf_predictions/trail_digital_twin_boundary_refined/benchmark_leaderboard.csv`
- `data/exp_perf_predictions/trail_digital_twin_boundary_best_profile/table_fitted_parameters.csv`
- `data/exp_perf_predictions/trail_digital_twin_boundary_best_profile/table_stage3_fatigue_state_comparison.csv`
- `data/exp_perf_predictions/trail_digital_twin_boundary_best_profile/table_stage3_ablation.csv`
