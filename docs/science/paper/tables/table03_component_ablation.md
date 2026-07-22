**Table 3. Component ablation with re-optimized (α, κ) per removal (LOO, activity objective).**

| Cohort | Variant | MAE (min) | MAPE (%) | Bias (min) | R² | ΔMAE vs full (min) | Protocol |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hard trail runs | Full model (M3 + trail GAP scales) | 17.4 | 9.67 | -9.82 | 0.969 | 0.0 | ladder_loo_baseline |
| Hard trail runs | Without grade-adjusted pace | 34.48 | 16.65 | -18.47 | 0.865 | 17.08 | reoptimize_loo |
| Hard trail runs | Without altitude correction | 19.91 | 10.45 | -12.04 | 0.958 | 2.51 | reoptimize_loo |
| Hard trail runs | Without REDI readiness | 19.17 | 10.77 | -14.07 | 0.956 | 1.77 | reoptimize_loo |
| Hard trail runs | Without HRR effort term | 46.17 | 17.1 | -40.05 | 0.725 | 28.77 | reoptimize_loo |
| Hard trail runs | Without acute TRIMP fatigue | 47.12 | 20.9 | -46.36 | 0.776 | 29.72 | reoptimize_loo |
| Hard trail runs | Acute TRIMP → linear progress fatigue | 23.78 | 12.18 | -6.68 | 0.942 | 6.38 | reoptimize_loo |
| Hard trail runs | Without asymmetric trail GAP scales | 20.93 | 11.12 | -12.53 | 0.958 | 3.53 | reoptimize_loo |
| Hard run or trail runs | Full model (M3 + trail GAP scales) | 8.25 | 7.33 | -4.2 | 0.986 | 0.0 | ladder_loo_baseline |
| Hard run or trail runs | Without grade-adjusted pace | 18.7 | 13.47 | -8.52 | 0.901 | 10.45 | reoptimize_loo |
| Hard run or trail runs | Without altitude correction | 9.23 | 7.81 | -0.27 | 0.984 | 0.98 | reoptimize_loo |
| Hard run or trail runs | Without REDI readiness | 8.81 | 7.55 | -2.58 | 0.983 | 0.56 | reoptimize_loo |
| Hard run or trail runs | Without HRR effort term | 22.61 | 13.14 | -19.23 | 0.844 | 14.35 | reoptimize_loo |
| Hard run or trail runs | Without acute TRIMP fatigue | 24.66 | 15.42 | -23.63 | 0.835 | 16.41 | reoptimize_loo |
| Hard run or trail runs | Acute TRIMP → linear progress fatigue | 16.61 | 16.4 | 4.02 | 0.953 | 8.36 | reoptimize_loo |
| Hard run or trail runs | Without asymmetric trail GAP scales | 8.87 | 7.89 | -0.72 | 0.984 | 0.62 | reoptimize_loo |
| Run/trail > 20 min | Full model (M3 + trail GAP scales) | 5.91 | 8.57 | -2.05 | 0.983 | 0.0 | ladder_loo_baseline |
| Run/trail > 20 min | Without grade-adjusted pace | 12.35 | 13.92 | -4.89 | 0.894 | 6.44 | reoptimize_loo |
| Run/trail > 20 min | Without altitude correction | 6.23 | 8.69 | -2.45 | 0.981 | 0.32 | reoptimize_loo |
| Run/trail > 20 min | Without REDI readiness | 5.17 | 7.78 | -3.97 | 0.986 | -0.74 | reoptimize_loo |
| Run/trail > 20 min | Without HRR effort term | 11.99 | 14.35 | -8.59 | 0.896 | 6.08 | reoptimize_loo |
| Run/trail > 20 min | Without acute TRIMP fatigue | 12.55 | 11.87 | -11.13 | 0.867 | 6.63 | reoptimize_loo |
| Run/trail > 20 min | Acute TRIMP → linear progress fatigue | 13.55 | 23.07 | 5.84 | 0.946 | 7.64 | reoptimize_loo |
| Run/trail > 20 min | Without asymmetric trail GAP scales | 6.36 | 8.51 | -2.7 | 0.98 | 0.45 | reoptimize_loo |
| Top-10 hard trail by HRR | Full model (M3 + trail GAP scales) | 3.78 | 4.69 | -0.65 | 0.993 | 0.0 | reoptimize_in_sample |
| Top-10 hard trail by HRR | Without grade-adjusted pace | 11.6 | 19.19 | -8.9 | 0.932 | 7.82 | reoptimize_in_sample |
| Top-10 hard trail by HRR | Without altitude correction | 3.96 | 5.28 | -1.53 | 0.992 | 0.18 | reoptimize_in_sample |
| Top-10 hard trail by HRR | Without REDI readiness | 5.08 | 6.8 | -2.53 | 0.99 | 1.29 | reoptimize_in_sample |
| Top-10 hard trail by HRR | Without HRR effort term | 4.43 | 6.77 | -1.01 | 0.992 | 0.65 | reoptimize_in_sample |
| Top-10 hard trail by HRR | Without acute TRIMP fatigue | 21.0 | 19.47 | -21.0 | 0.807 | 17.22 | reoptimize_in_sample |
| Top-10 hard trail by HRR | Acute TRIMP → linear progress fatigue | 5.42 | 6.08 | 0.41 | 0.984 | 1.64 | reoptimize_in_sample |
| Top-10 hard trail by HRR | Without asymmetric trail GAP scales | 4.93 | 6.86 | -1.95 | 0.99 | 1.15 | reoptimize_in_sample |
| Selected race dates | Full model (M3 + trail GAP scales) | 11.82 | 5.89 | 4.07 | 0.987 | 0.0 | ladder_loo_baseline |
| Selected race dates | Without grade-adjusted pace | 29.72 | 14.83 | -11.75 | 0.89 | 17.89 | reoptimize_loo |
| Selected race dates | Without altitude correction | 12.68 | 6.06 | 1.3 | 0.984 | 0.85 | reoptimize_loo |
| Selected race dates | Without REDI readiness | 9.18 | 4.26 | 2.83 | 0.992 | -2.64 | reoptimize_loo |
| Selected race dates | Without HRR effort term | 26.48 | 10.81 | -10.74 | 0.91 | 14.66 | reoptimize_loo |
| Selected race dates | Without acute TRIMP fatigue | 45.52 | 18.47 | -45.52 | 0.792 | 33.69 | reoptimize_loo |
| Selected race dates | Acute TRIMP → linear progress fatigue | 20.94 | 15.91 | 8.06 | 0.976 | 9.11 | reoptimize_loo |
| Selected race dates | Without asymmetric trail GAP scales | 10.96 | 5.76 | 1.68 | 0.986 | -0.86 | reoptimize_loo |

*Positive ΔMAE indicates degraded accuracy after removing the component and re-fitting. Protocol column reports reoptimize_loo (preferred) vs legacy frozen. Trail GAP scales refer to asymmetric soft-ramped climb/descent corrections.*
