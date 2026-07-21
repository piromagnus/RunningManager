**Table 8. Segment-level terrain optimisation via asymmetric trail GAP scales.**

| Terrain | n segments | MAE before (min) | Bias before (min) | MAE after (min) | Bias after (min) | ΔMAE (min) |
| --- | --- | --- | --- | --- | --- | --- |
| Flat | 218 | 0.91 | -0.32 | 0.86 | -0.03 | -0.05 |
| Climb | 191 | 1.37 | 0.21 | 1.33 | 0.05 | -0.04 |
| Steep climb | 59 | 2.44 | 1.69 | 1.92 | -0.08 | -0.53 |
| Descent | 191 | 1.72 | -1.63 | 1.11 | 0.05 | -0.61 |
| Steep descent | 67 | 3.65 | -3.65 | 1.36 | -0.07 | -2.29 |
| Mixed | 110 | 1.47 | -0.6 | 1.35 | -0.23 | -0.12 |

*Before/after soft-ramped climb scale 0.85 and descent scale 1.60 (hardTrailRun, segment objective, moving-time residuals). Positive bias = model too slow.*
