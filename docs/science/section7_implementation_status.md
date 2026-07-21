# Paper §7 status — shipped vs remaining

Artifacts: `data/exp_perf_predictions/trail_digital_twin_paper_section7/`.
Robustness report: [`robustness_experiments_report.md`](robustness_experiments_report.md).
Remaining checklist: [`remaining_experiments.md`](remaining_experiments.md).
Paper draft: [`trail_digital_twin_hr_performance_paper_draft.md`](trail_digital_twin_hr_performance_paper_draft.md) (v0.6).

## Headline findings (canonical)

| Metric | Value |
|--------|-------|
| Hard run/trail M0 → M3 LOO MAE | **30.2 → 9.09 min** (MAPE 6.5%) |
| Hard trail M3 LOO MAE | **17.39 min** |
| Ablation ΔMAE (−HRR / −TRIMP / −GAP), R1-aligned | **+13.5 / +15.6 / +9.6 min** |
| Steep terrain MAE (before → after GAP scales) | **3.05 → 1.64 min (−46%)** |
| Prospective Δ (trail): obs. mean / feasible / HRR_ref | **~+13–19 / mixed / systematically fast**; Échappée obs. **+16.8** vs feasible **−73** |
| Rome (road) | out of scope |
| Grésivaudan with 12 min aid budget | **≈ −5 min** |
| §7 slight rejects | **4 segments (0.17%, 84 min)** |

## Robustness suite (R1–R11)

| Status | IDs |
|--------|-----|
| Pass | R1–R11 (incl. R3: 4 trail + Rome) |

## Still blocked (data)

Multi-athlete (B1), strata (B2), DEM (B3), weather term (B4), structured aid logs (B5).

## Ethics / seeds

- Single-athlete case study.
- Seeds: bootstrap `20260623`, LOO cap `20260721` + SHA-256 cohort offset.
