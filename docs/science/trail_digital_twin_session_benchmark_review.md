# Trail Digital Twin Session-Level Benchmark Review

Generated: 2026-07-20

Consolidates hyperparameter-tuning winners with **session (activity/race)** LOO
errors, highlighting the most useful elements for continued modeling.

## Most Useful Elements

1. **Operational physiology defaults** (current-code hypothesis winner):
   `hrr_reference=0.88`,
   `hrr_min_factor=0.30`,
   `hrr_max_factor=1.00`,
   `decay_lambda=0.20`,
   `min_fatigue_factor=0.60`.
   Aggregate Stage 3 LOO MAE ≈ **10.84 min**
   (MAPE 7.91%, R² 0.982).
2. **Stage 3 HRR speed-ratio + decayed TRIMP** dominates earlier stages;
   muscular secondary fatigue usually stays at 0 but should remain in the grid.
3. **Session residuals** are the actionable unit: large positive errors often mean
   the athlete was slower than the model (stops, nutrition, pacing issues,
   device-open idle time); large negatives can mean unusually strong execution.
4. **Hard-trail high-duration / high-TRIMP strata** remain the main residual risk;
   terrain families (steep descent / steep climb) still drive segment MAE.
5. **Segment stationary exclusion** (new): fit on moving segments, then score the
   full race so excluded idle time still appears as informative residual.

## Hyperparameter Winner Snapshot

- Source experiment: `trail_digital_twin_hypothesis_refined`
- Winner run: `000_h1_high_reference_refined_h1_refined_000_hrr_reference_0_88_hrr_min_factor_0_3_hrr_max_factor_1_decay_lamb`

### Cohort Stage 3 LOO (winner run)

| cohort | MAE min | MAPE % | R2 | bias min |
| --- | ---: | ---: | ---: | ---: |
| top10HardTrailByHRR | 4.24 | 5.72 | 0.991 | -1.68 |
| selectedDateRaces | 7.87 | 4.50 | 0.995 | 3.73 |
| hardRunOrTrailRun | 12.24 | 9.52 | 0.970 | 1.98 |
| hardTrailRun | 19.03 | 11.93 | 0.972 | 3.05 |

### Fitted Stage 3 parameters (winner run)

| cohort | alpha | fatigueCoef | model | state | secondary |
| --- | ---: | ---: | --- | --- | ---: |
| hardTrailRun | 0.90 | 0.60 | exponential | decayed | 0.00 |
| hardRunOrTrailRun | 0.90 | 0.40 | linear | decayed | 0.00 |
| top10HardTrailByHRR | 0.90 | 0.30 | exponential | decayed | 0.00 |
| selectedDateRaces | 0.90 | 0.30 | linear | decayed | 0.00 |

## Session-Level LOO Review

Source LOO table: `trail_digital_twin_segment_exclusion` (Stage 3 HRR speed ratio LOO, activity objective).
Note: archived session LOO may use an earlier physiology profile than the
hypothesis winner; use it for residual triage, then re-run with the winner
config when timeseries are available.

### Cohort session summary

| cohort | n | MAE min | median |abs| | MAPE % | bias min | worst |abs| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hardRunOrTrailRun | 103 | 10.76 | 5.43 | 9.06 | 2.31 | 96.84 |
| hardTrailRun | 45 | 16.98 | 9.05 | 10.00 | 0.88 | 103.22 |
| selectedDateRaces | 18 | 10.94 | 7.42 | 6.18 | 3.53 | 62.16 |
| top10HardTrailByHRR | 10 | 4.81 | 4.51 | 6.84 | -0.84 | 9.48 |

### Largest absolute session residuals

| activity | name | cohort | actual min | pred min | error min | error % | α | κ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15018253166 | Revanche aux 7 laux | hardTrailRun | 516.38 | 413.17 | -103.22 | -19.99 | 0.85 | 0.20 |
| 15018253166 | Revanche aux 7 laux | hardRunOrTrailRun | 516.38 | 419.54 | -96.84 | -18.75 | 0.85 | 0.30 |
| 15179557231 | La croix et les lacs | hardRunOrTrailRun | 559.05 | 494.21 | -64.84 | -11.60 | 0.85 | 0.30 |
| 14953411411 | Moyen duc 2025 | selectedDateRaces | 551.45 | 489.29 | -62.16 | -11.27 | 0.90 | 0.30 |
| 16043688740 | GTV J1 : Saint-Nizier - Abri de Carette | hardRunOrTrailRun | 470.17 | 410.78 | -59.38 | -12.63 | 0.85 | 0.30 |
| 15114097922 | Petite sortie reco de l’EB pour finir le week-en | hardRunOrTrailRun | 563.65 | 509.41 | -54.24 | -9.62 | 0.85 | 0.30 |
| 15114097922 | Petite sortie reco de l’EB pour finir le week-en | hardTrailRun | 563.65 | 515.05 | -48.60 | -8.62 | 0.85 | 0.20 |
| 17481444994 | Trail du Grésivaudan 2026 : Le Grand V | hardTrailRun | 212.98 | 260.79 | 47.80 | 22.44 | 0.85 | 0.20 |
| 15179557231 | La croix et les lacs | hardTrailRun | 559.05 | 516.59 | -42.46 | -7.60 | 0.85 | 0.20 |
| 17481444994 | Trail du Grésivaudan 2026 : Le Grand V | hardRunOrTrailRun | 212.98 | 252.98 | 40.00 | 18.78 | 0.85 | 0.30 |
| 16043688740 | GTV J1 : Saint-Nizier - Abri de Carette | hardTrailRun | 470.17 | 430.63 | -39.54 | -8.41 | 0.85 | 0.20 |
| 16043688538 | GTV J2 : Abri de Carette - Die 1/2 | hardTrailRun | 307.37 | 346.68 | 39.32 | 12.79 | 0.85 | 0.20 |

### Hard-trail error strata (hypothesis winner)

| strata | value | n | MAE min | bias min |
| --- | --- | ---: | ---: | ---: |
| in_race_trimp_tertile | high | 18 | 30.65 | -3.96 |
| duration_tertile | high | 20 | 28.71 | -1.98 |
| dominant_terrain | Steep descent | 1 | 26.08 | 26.08 |
| hrr70_share_tertile | low | 22 | 25.61 | -4.32 |
| dominant_terrain | Flat | 11 | 20.50 | 16.45 |
| dominant_terrain | Ascent | 21 | 20.43 | -2.07 |
| hrr70_share_tertile | high | 10 | 18.71 | 18.02 |
| dominant_terrain | Steep ascent | 6 | 16.82 | -8.93 |
| dominant_terrain | Descent | 3 | 16.80 | 16.80 |
| duration_tertile | mid | 15 | 15.37 | 10.56 |
| in_race_trimp_tertile | mid | 19 | 12.27 | 8.59 |
| in_race_trimp_tertile | low | 8 | 8.94 | 5.66 |

### Hard-trail terrain segment MAE (hypothesis winner)

| terrain | segments | MAE min | bias min | MAPE % |
| --- | ---: | ---: | ---: | ---: |
| Steep ascent | 61 | 4.20 | 3.85 | 25.68 |
| Steep descent | 67 | 3.42 | -3.39 | 33.70 |
| Ascent | 189 | 2.15 | 1.66 | 19.68 |
| Mixed climb/descent | 110 | 1.69 | 0.37 | 16.77 |
| Descent | 192 | 1.41 | -1.00 | 16.33 |
| Flat | 218 | 1.17 | 0.54 | 17.74 |

## Segment Exclusion Experiment (executed)

Source: `data/exp_perf_predictions/trail_digital_twin_segment_exclusion/`

Zero-distance GPS dwell is now kept inside 1 km segments (`stationaryTimeShare`).
Hyperparameters are fit on cleaned segments; LOO still scores the **full race**.

| run | full-race mean Stage3 MAE min | fit-eligible session MAE min | excluded segs | excluded time |
| --- | ---: | ---: | ---: | ---: |
| baseline_no_exclusion | 10.87 | 14.97 | 0 | 0 |
| exclude speed&lt;3 / share&gt;0.40 | 12.32 | 10.50 | 64 | 1625 min |
| exclude speed&lt;4 / share&gt;0.30 | 12.15 | 10.22 | 156 | 3076 min |

Findings:
- Fit-eligible MAE improves under exclusion (~10.2–10.5 min vs ~15 min baseline
  fit-eligible diagnostic), confirming idle/aid segments were contaminating the fit.
- Full-race aggregate MAE gets worse (+1.3 to +1.5 min): remaining residual is
  informative non-model time (stops / device-open), not a signal to ignore.
- `hardTrailRun` full-race MAE slightly improves (16.98 → 16.53); `selectedDateRaces`
  and `top10HardTrailByHRR` degrade on full-race scoring when stops are common.

Default exclusion knobs after this sweep: `min_mean_speed_kmh=3.0`,
`max_stationary_time_share=0.40` (disabled by default in production config).

## Interpretation Guide

- **Positive error** (pred > actual): model too slow / athlete faster than twin.
- **Negative error** (pred < actual): model too fast / athlete slower — often
  includes aid stops, walking, or device-open idle segments.
- Prefer diagnosing large residuals with segment exclusion enabled: if fit-eligible
  MAE improves while full-race MAE stays high, the gap is non-model time.

## Next Experiments

1. Keep exclusion as a diagnostic fit mode; do not replace full-race evaluation.
2. Inspect high `excludedTimeSec` sessions in
   `trail_digital_twin_segment_exclusion/runs/*/activity_loo_predictions.csv`.
3. Optionally cohort-specific thresholds (hard trails vs road races).

## Regenerator

```bash
uv run python scripts/synthesize_trail_digital_twin_sessions.py \
  --session-source trail_digital_twin_segment_exclusion
```
