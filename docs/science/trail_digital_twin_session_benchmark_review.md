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

Source LOO table: `trail_digital_twin_boundary_best_profile` (Stage 3 HRR speed ratio LOO, activity objective).
Note: archived session LOO may use an earlier physiology profile than the
hypothesis winner; use it for residual triage, then re-run with the winner
config when timeseries are available.

### Cohort session summary

| cohort | n | MAE min | median |abs| | MAPE % | bias min | worst |abs| |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| hardRunOrTrailRun | 103 | 11.35 | 4.87 | 8.79 | 1.39 | 123.75 |
| hardTrailRun | 45 | 20.64 | 16.41 | 13.61 | 8.53 | 103.54 |
| selectedDateRaces | 18 | 7.88 | 3.76 | 5.45 | -4.73 | 52.86 |
| top10HardTrailByHRR | 10 | 4.51 | 2.84 | 5.89 | -0.04 | 8.66 |

### Largest absolute session residuals

| activity | name | cohort | actual min | pred min | error min | error % | α | κ |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 15018253166 | Revanche aux 7 laux | hardRunOrTrailRun | 516.38 | 392.64 | -123.75 | -23.96 | 1.00 | 0.60 |
| 15018253166 | Revanche aux 7 laux | hardTrailRun | 516.38 | 412.84 | -103.54 | -20.05 | 1.00 | 0.80 |
| 15179557231 | La croix et les lacs | hardRunOrTrailRun | 559.05 | 479.66 | -79.39 | -14.20 | 1.00 | 0.60 |
| 15114097922 | Petite sortie reco de l’EB pour finir le week-en | hardRunOrTrailRun | 563.65 | 499.57 | -64.08 | -11.37 | 1.00 | 0.60 |
| 17481444994 | Trail du Grésivaudan 2026 : Le Grand V | hardTrailRun | 212.98 | 271.29 | 58.31 | 27.38 | 1.00 | 0.80 |
| 14953411411 | Moyen duc 2025 | selectedDateRaces | 551.45 | 498.59 | -52.86 | -9.58 | 1.00 | 0.40 |
| 17481444994 | Trail du Grésivaudan 2026 : Le Grand V | hardRunOrTrailRun | 212.98 | 264.14 | 51.16 | 24.02 | 1.00 | 0.60 |
| 16043688740 | GTV J1 : Saint-Nizier - Abri de Carette | hardRunOrTrailRun | 470.17 | 421.60 | -48.57 | -10.33 | 1.00 | 0.60 |
| 15250276952 | Trail locunolois 2025 : mission accomplie | hardTrailRun | 132.98 | 178.95 | 45.96 | 34.56 | 1.00 | 0.80 |
| 15114097922 | Petite sortie reco de l’EB pour finir le week-en | hardTrailRun | 563.65 | 518.44 | -45.21 | -8.02 | 1.00 | 0.80 |
| 15563904138 | Echappee Belle 2025 : Parcours des crêtes | hardTrailRun | 704.27 | 749.08 | 44.81 | 6.36 | 1.00 | 0.80 |
| 15563904138 | Echappee Belle 2025 : Parcours des crêtes | hardRunOrTrailRun | 704.27 | 743.78 | 39.52 | 5.61 | 1.00 | 0.60 |

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

## Interpretation Guide

- **Positive error** (pred > actual): model too slow / athlete faster than twin.
- **Negative error** (pred < actual): model too fast / athlete slower — often
  includes aid stops, walking, or device-open idle segments.
- Prefer diagnosing large residuals with segment exclusion enabled: if fit-eligible
  MAE improves while full-race MAE stays high, the gap is non-model time.

## Next Experiments

1. Run `configs/trail_digital_twin_benchmark_segment_exclusion.yaml` when
   `data/timeseries` is available.
2. Compare baseline vs exclusion using full-race Stage 3 LOO MAE and per-session
   `excludedTimeSec` from `segment_qc` / LOO columns.
3. Re-generate this report after the exclusion sweep with
   `--session-source trail_digital_twin_segment_exclusion`.

## Regenerator

```bash
uv run python scripts/synthesize_trail_digital_twin_sessions.py
```
