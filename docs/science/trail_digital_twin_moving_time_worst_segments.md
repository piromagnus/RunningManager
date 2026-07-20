# Moving-time Stage 3 fit — worst segments & full-race LOO
Generated: 2026-07-20
## Method
- Fit target: `actualMovingTimeSec` = segment clock − stationary dwell
- Segment objective used to diagnose terrain fit quality
- Full-race LOO still scores observed activity `movingSec` (all segments predicted)
- Physiology: hrr=0.88, factors 0.30–1.00, λ=0.20, floor=0.60

## Full-race Stage 3 LOO (activity objective)
| cohort | baseline MAE min | moving-fit MAE min | baseline MAPE % | moving-fit MAPE % |
|---|---:|---:|---:|---:|
| hardRunOrTrailRun | 10.76 | 9.82 | 9.06 | 7.50 |
| hardTrailRun | 16.98 | 17.01 | 10.00 | 8.71 |
| selectedDateRaces | 10.94 | 10.94 | 6.18 | 6.18 |
| top10HardTrailByHRR | 4.81 | 5.60 | 6.84 | 7.29 |

## Worst terrain families (hardTrailRun, segment objective, moving MAE)
| terrain | n | MAE moving min | bias moving min | MAPE moving % |
|---|---:|---:|---:|---:|
| steep_descent | 67 | 3.518 | -3.505 | 37.1 |
| steep_climb | 59 | 2.679 | 2.262 | 17.0 |
| descent | 191 | 1.584 | -1.475 | 19.4 |
| mixed_climb_descent | 110 | 1.428 | -0.341 | 13.4 |
| climb | 191 | 1.410 | 0.611 | 12.7 |
| flat | 218 | 0.882 | -0.061 | 12.4 |

## Example worst-fitted segments (hardTrailRun)
| Run | km | terrain | seg residual (moving) min | full race min | race err min |
|---|---:|---|---:|---:|---:|
| La croix et les lacs | 13.0–14.0 | steep_climb | +12.95 | 559.0 | -77.0 |
| Bon y aura pas le kom aujourd'hui | 4.0–5.0 | steep_descent | -10.61 | 74.5 | +20.8 |
| Bon y aura pas le kom aujourd'hui | 5.0–6.0 | steep_descent | -10.32 | 74.5 | +20.8 |
| La croix et les lacs | 10.0–11.0 | steep_climb | +9.25 | 559.0 | -77.0 |
| Verticale du Grand Serre | 1.0–2.0 | steep_climb | +8.97 | 46.4 | +8.8 |
| Moyen duc 2025 | 7.0–8.0 | steep_descent | -8.75 | 551.5 | -33.4 |
| Petite sortie reco de l’EB pour finir le week-end chooooooc | 10.0–11.0 | steep_climb | +7.56 | 563.6 | -74.2 |
| Echappee Belle 2025 : Parcours des crêtes | 28.0–29.0 | steep_climb | +7.51 | 704.3 | +9.1 |
| Lunch Trail Run | 7.0–8.0 | descent | -7.21 | 168.2 | -16.2 |
| La croix et les lacs | 24.0–25.0 | steep_descent | -7.06 | 559.0 | -77.0 |
| Revanche aux 7 laux | 7.0–8.0 | climb | +6.78 | 516.4 | -111.0 |
| Lunch Trail Run | 0.0–1.0 | climb | +6.77 | 256.7 | -35.0 |
