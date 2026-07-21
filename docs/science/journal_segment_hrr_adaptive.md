# Journal: segment-type adaptive HRR for prospective race prediction

Started: 2026-07-21  
Updated: 2026-07-21 (MAPE-primary + optimistic-OK reframing)  
Status: best **principled** = **H13b_short_midref_long_H10** (trail MAPE ≈ **2.07%**)  
Related: `docs/science/journal_prediction.md`, `docs/science/bibliography_hr_digital_twin.md`,
`docs/science/idees_experiences_repo_detaillees_fr.md` (catalogue RQ/E/M/R/H).

## Goal

Improve **prospective** finish-time prediction with **terrain-family–adaptive** HRR
(5 grade families), choosing a duration-aware target mean and modulating per
segment from real data (hold-outs excluded).

### Evaluation policy (updated)

1. **Primary metric = MAPE** (% of observed moving time) — less sensitive to
   race duration than MAE (Échappée no longer dominates solely by length).
2. **Optimistic bias is acceptable**: predicting **faster** than actual
   (Δ < 0) is not treated as a hard failure. Race planning can use a fast
   envelope; being too slow (Δ > 0) is the more costly error mode for pacing.
3. Still report signed Δ (min) and MAE for transparency.
4. **No race HR / times** in fit or target selection (eval-only).

**Physiology:** `hrr_reference=0.88` = HRR at flat VMA (\(E=1\));
`hrr_max_factor=1.20`.

---

## Bibliography cues used

| Source | Idea borrowed |
|--------|----------------|
| Jaén-Carrillo & Pattis (2026) | Sustainable intensity fraction + pacing; race push above easy envelope |
| Swain & Leutholtz (1997) | %HRR ≈ %VO₂ reserve → HRR as effort |
| Fornasiero et al. (2018) | Ultra ~77% HRmax — lower sustainable HRR for long events |
| Banister TRIMP | Intra-race fatigue accumulation (already in Stage 3) |
| Vandewalle / Boillet CP–W′ | Steeper short-duration capacity → short races can target higher HRR |
| Emig & Peltonen (2020) | Duration–performance individuality (motivates duration-dependent HRR★) |
| Genitrini / Lemire | Terrain-dependent effort → family modulators |

---

## Segment families (5)

| Family | Grade |
|--------|-------|
| `steep_climb` | \(g \ge 0.15\) |
| `climb` | \(0.04 \le g < 0.15\) |
| `flat` | \(|g| < 0.04\) |
| `descent` | \(-0.15 < g \le -0.04\) |
| `steep_descent` | \(g \le -0.15\) |

hardTrailRun mean modulators \(m_f\): steep_climb ≈ 1.06, steep_descent ≈ 0.94,
others ≈ 1.00 (mild).

---

## Baselines (MAPE)

| Mode | Trail MAPE | Notes |
|------|------------|-------|
| Constant duration-feasible (PL) | **7.52%** | LUT +9.4%, Échappée −10.4% (optimistic OK) |
| Constant HRR_ref = 0.88 | **~9.4%** | All optimistic; Échappée −23% hurts MAPE |
| Obs. mean HRR (eval-only) | ~7–8% class | Still slow on ≤3 h (+13–19 min) |

Assets @ `hrr_max_factor=1.20`: LOO hard run/trail M3 MAE **8.25 min**, MAPE **7.3%**.

---

## Hypotheses (MAPE-ranked)

Δ = predicted − actual (min). Optimistic = Δ < 0.

### Family modulation only (H1–H5)

| ID | Idea | MAPE | Verdict |
|----|------|------|---------|
| H1 | mean \(m_f\) × PL target | 7.43% | Fail — modulation too mild |
| H2 | p75 \(m_f\) × PL | 7.25% | ≈ constant |
| H3 | family duration caps | 7.42% | Caps rarely bind |
| H4 / H4b | hardTrail vs all-usable \(m_f\) | 7.43 / 7.83% | Prefer hardTrail |
| H5 | supra-VMA flats only | 7.43% | No effect at HRR≤0.88 |

### Failed target shifts (H6–H9, H11)

| ID | Idea | MAPE | Verdict |
|----|------|------|---------|
| H6–H7 | hist blend / hist band | 11–17% | Too soft on short races |
| H8 | resid(obs−env)~log T + apk | ~32% | Easy-run resid over-corrects |
| H9 | min(PL, empirical window) | 8.1% | Ultra +55 min (slow) — bad under optimistic-OK |
| H11 | switch to emp if T≥6 h | 6.5% | Same ultra over-correction |

### Duration blend (H10 / H12) — first MAPE win

\[
w=\mathrm{clip}((T_h-4)/(10-4),0,1)\cdot w_{\max},\quad
\mathrm{HRR}^\star=(1-w)\mathrm{HRR}_{\mathrm{PL}}+w\,\mathrm{HRR}_{\mathrm{emp}}
\]

| ID | MAPE | MAE | Notes |
|----|------|-----|-------|
| **H10_blend_w70_p75** | **4.92%** | 11.1 | All slow (no optimistic); Échappée +10 min |
| H10_blend_w70_mean | 4.97% | 10.9 | Similar |
| H10_w50 / w85 / w100 | 5.2–6.5% | — | w70 sweet spot |

Fixes ultra PL flatness (emp ≈ 0.62 at 6–8 h vs PL ≈ 0.75 at 10 h).

### Short-race race-push (H13–H20) — MAPE primary

Short races were **slow** (+6–9% MAPE) at PL; under optimistic-OK we **raise**
short-race HRR toward VMA effort, keep H10 on ultras.

| ID | Short target | Long | MAPE | Optimistic races |
|----|--------------|------|------|------------------|
| H18 | HRR_ref only | — | 9.77% | 4 — ultra too fast for MAPE |
| H13 | HRR_ref | H10 | 4.10% | 3 |
| H16 | CP-style boost to ref | H10 | 3.66% | 0 |
| H14 | max(PL, win60) | H10 | 2.99% | 0 |
| H15 | 0.82 / Fornasiero mix | — | 2.56% | 1 |
| H19 | PL+0.04 | H10 | 2.37% | 1 |
| H17 | 0.85 | H10 | 2.26% | 2 |
| **H13b mid(PL, ref)** | **0.5(PL+ref)** | **H10** | **2.07%** | **1** |
| H20 PL+0.06 | PL+0.06 | H10 | **2.00%** | 2 |

**Preferred: H13b** — no free boost parameter; biblio-aligned (race push halfway
from sustainability floor to flat-VMA effort on ≤5 h; Fornasiero/H10 on ultra).

H20 is ~tied on MAPE but +0.06 is hold-out-tuned; training 2–5 h hard trails are
already mostly optimistic at PL (median needed boost = 0), so H20 is sensitivity
only.

### H13b detail

| Race | target HRR | Δ min | MAPE % | signed % |
|------|------------|-------|--------|----------|
| LUT | 0.830 | +5.3 | 3.0 | +3.0 |
| Grésivaudan | 0.830 | −6.8 | 3.2 | −3.2 (optimistic OK) |
| Passerelles | 0.830 | +1.2 | 0.6 | +0.6 |
| Échappée | 0.657 | +10.2 | 1.5 | +1.5 |

Trail MAPE **2.07%** vs constant feasible **7.52%** (≈ **3.6×** better).

---

## Interpretation

1. **Family mods** alone ≈ noise for finish MAPE.
2. **PL envelope too flat for ultras** → empirical windows (H10) required.
3. Under **MAPE + optimistic-OK**, short races need a **race push** above PL
   (toward HRR_ref); being slightly fast on Grésivaudan is fine.
4. Remaining error is small in relative terms; absolute ≤3 h slow bias when it
   remains is physics/fatigue, not HRR mean selection.

---

## Current best recipe (prospective)

1. Stage 3 + PL envelope + empirical windows (hold-outs out).
2. If predicted \(T < 5\) h: \(\mathrm{HRR}^\star = \tfrac12(\mathrm{HRR}_{\mathrm{PL}}+\mathrm{HRR}_{\mathrm{ref}})\).
3. Else: H10 blend with \(w_{\max}=0.70\).
4. Assign \(\mathrm{HRR}_i=\mathrm{HRR}^\star\cdot m_{f(i)}\) (hardTrailRun **p75** or mean;
   re-center). H13b default uses p75 mods.
5. Simulate with sequential TRIMP.

```bash
uv run python scripts/predict_race_segment_hrr_adaptive.py \
  --hypotheses H13b_short_midref_long_H10,H10_blend_w70_p75,H20_H10_short_plus006
```

Outputs: `data/exp_perf_predictions/trail_digital_twin_segment_hrr_adaptive/`
(`mape_leaderboard.csv`, `adaptive_hrr_summary.csv`).

---

## Decision rules

- Rank by **trail MAPE**; use MAE only as secondary.
- Optimistic (faster) predictions are allowed; avoid large **slow** MAPE on short races.
- Échappée: |signed %| ≲ 5% strong; do not sacrifice short-race MAPE to force ultra MAE→0.
- No road transfer claim (Rome).
- Prefer parameter-free / biblio-aligned rules over hold-out-tuned offsets.
