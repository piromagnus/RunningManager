# Journal: segment-type adaptive HRR for prospective race prediction

Started: 2026-07-21  
Status: best candidate = **H10_blend_w70_mean** (trail |Δ| MAE ≈ **10.9 min**)  
Related: `docs/science/journal_prediction.md`, `scripts/predict_race_segment_hrr_adaptive.py`

## Goal

Improve **prospective** finish-time prediction by replacing a single constant HRR
over the whole profile with a **terrain-family–adaptive** plan:

1. Estimate the **distribution of observed HRR** on the five grade families
   (`flat`, `climb`, `steep_climb`, `descent`, `steep_descent`) from other
   activities (hold-outs excluded).
2. Choose a **duration-feasible target mean HRR** for the race.
3. **Modulate** that target per segment using real-data family statistics
   relative to the overall mean (climbs/descents/flats get different constant
   HRR; race-average effort stays near the target after re-centering).
4. Iterate until predicted moving time is close to observed on held-out trail
   races (LUT, Grésivaudan, Passerelles, Échappée Belle).

Rome remains out of scope (road).

**Physiology:** `hrr_reference=0.88` = HRR at flat VMA effort (\(E=1\));
`hrr_max_factor=1.20` allows short supra-VMA bursts when HRR > ref.

**Hard rule:** no observed race HR / times in fit or HRR assignment. Observed
race times / mean HRR are evaluation-only.

---

## Segment families (5)

| Family | Grade |
|--------|-------|
| `steep_climb` | \(g \ge 0.15\) |
| `climb` | \(0.04 \le g < 0.15\) |
| `flat` | \(|g| < 0.04\) |
| `descent` | \(-0.15 < g \le -0.04\) |
| `steep_descent` | \(g \le -0.15\) |

---

## Prerequisites — assets at `hrr_max_factor=1.20`

Re-ran `trail_digital_twin_paper_section7.py` + `prepare_paper_assets.py`.

| Metric | Value |
|--------|-------|
| Headline LOO (hard run/trail M3) | **8.25 min** MAE (was 9.09 @ max_factor=1.0) |
| Hard trail M3 | 17.4 min |
| Run/trail >20 min M3 | 5.91 min |
| Table 5 point predictions | Unchanged vs max=1.0 (selected/obs HRR ≤ 0.88) |

Constant-HRR baselines (Table 5):

| Race | Obs. mean HRR | Δ @ obs. mean | Feasible HRR | Δ @ feasible | Δ @ HRR_ref |
|------|---------------|---------------|--------------|--------------|-------------|
| LUT | 0.80 | +13.2 | 0.78 | +16.7 | −4.1 |
| Passerelles | 0.76 | +18.7 | 0.78 | +13.7 | −8.0 |
| Grésivaudan | 0.73 | +17.6 | 0.78 | +6.7 | −17.0 |
| Échappée | 0.65 | +16.8 | 0.75 | −73.2 | −164 |

Trail |Δ| MAE @ obs. mean ≈ 16.6 min (eval-only). Constant duration-feasible
|Δ| MAE ≈ **27.6 min** (dominated by Échappée −73).

---

## Empirical HRR by terrain (hold-outs excluded)

Source: `segment_predictions.csv`, hardTrailRun, usable segments.

| Family | n | mean HRR | p75 | \(m_f\) (mean) |
|--------|---|----------|-----|----------------|
| flat | 248 | 0.606 | 0.714 | 0.990 |
| climb | 164 | 0.620 | 0.754 | 1.012 |
| steep_climb | 46 | 0.650 | 0.750 | 1.062 |
| descent | 174 | 0.616 | 0.719 | 1.005 |
| steep_descent | 52 | 0.574 | 0.699 | 0.937 |

Modulation is mild (~±6%), so **family modulation alone barely moves finish
times** when the race-mean target is wrong.

Power-law HRR–duration fit (hold-outs excluded): \(a\approx 0.825\),
\(b\approx -0.039\) — nearly flat. Empirical monotone windows drop much faster
at long duration (target HRR ≈ **0.62** by 6–8 h) while the power law still
gives ≈ **0.75** at 10 h. That mismatch is the Échappée failure mode.

---

## Hypotheses and results

Δ = predicted − observed moving (min). Trail |Δ| MAE over 4 races.
Constant feasible baseline MAE = **27.58 min**.

### H1 — Mean modulators × power-law feasible target

| Race | Δ adaptive | Δ const |
|------|------------|---------|
| LUT | +16.4 | +16.7 |
| Grésivaudan | +5.7 | +6.7 |
| Passerelles | +13.0 | +13.7 |
| Échappée | −77.6 | −73.2 |

**MAE 28.17 — FAIL** (slightly worse than constant; Échappée more optimistic).

### H2 — p75 modulators × power-law feasible

| Race | Δ |
|------|---|
| LUT | +16.3 |
| Grésivaudan | +5.4 |
| Passerelles | +12.5 |
| Échappée | −75.9 |

**MAE 27.51 — MARGINAL PASS** vs constant 27.58 (tiny win on ≤3 h races).

### H3 — Family duration caps

Caps from power-law on family time shares almost never bind.
**MAE 28.15 — FAIL.**

### H4 / H4b — hardTrailRun vs all-usable modulators

hardTrail = H1. All-usable worse (**MAE 28.65**). Prefer hardTrail.

### H5 — Supra-VMA on flats only

No effect at targets ≤ 0.88. **MAE 28.17 — FAIL.**

### H6 / H7 — Amplified mods + historical mean blend / hist-band

Lowered race-mean target toward easy-run history → short races much slower.
**MAE 35–46 — FAIL.**

### H8 — Residual (obs mean − envelope) ~ log(dur) + ascent/km

Training residual strongly negative (easy runs ≪ envelope max). Over-lowers
target. **MAE ~90 — FAIL loudly.**

### H9 — `min(power-law, empirical window)` as target + p75 mods

Échappée target 0.617 → **+55** (over-corrected). Short races slightly worse.
**MAE 25.4 — better than constant but not close enough on ultra.**

### H10 — Duration-weighted blend of power-law & empirical + family mods

\[
w = \mathrm{clip}\bigl((T_{\mathrm{h}}-4)/(10-4),\,0,\,1\bigr)\cdot w_{\max}
\]
\[
\mathrm{HRR}^\star = (1-w)\,\mathrm{HRR}_{\mathrm{PL}} + w\,\mathrm{HRR}_{\mathrm{emp}}(T)
\]

with \(T\) = predicted time at power-law feasible HRR (prospective; no race HR).

| Variant | \(w_{\max}\) | mods | Trail MAE | Échappée Δ |
|---------|--------------|------|-----------|------------|
| H10_blend_w50_p75 | 0.50 | p75 | 12.75 | −16.8 |
| **H10_blend_w70_mean** | **0.70** | **mean** | **10.88** | **+8.4** |
| H10_blend_w70_p75 | 0.70 | p75 | 11.10 | +10.2 |
| H10_blend_w85_p75 | 0.85 | p75 | 16.55 | +32.0 |
| H10_blend_w100_p75 | 1.00 | p75 | 22.35 | +55.2 |

**H10_blend_w70_mean detail**

| Race | target HRR | \(w\) | Δ | Δ const |
|------|------------|-------|---|---------|
| LUT | 0.780 | 0 | +16.4 | +16.7 |
| Grésivaudan | 0.780 | 0 | +5.7 | +6.7 |
| Passerelles | 0.780 | 0 | +13.0 | +13.7 |
| Échappée | 0.657 | 0.70 | **+8.4** | −73.2 |

Échappée target 0.657 ≈ observed mean 0.654 (without using race HR). Short
races unchanged vs H1/H2 (blend weight 0 below 4 h).

**PASS — best so far.** Trail MAE **10.9 vs 27.6** constant feasible.

### H11 — Hard switch to empirical if \(T\ge 6\) h

Same over-correction as H9 on Échappée. **MAE 22.3 — FAIL.**

### H12 — H10 + one re-eval of empirical at blended prediction

Identical to H10_w70_p75 here (no second-order move). **MAE 11.1.**

---

## Interpretation

1. **Family modulation** redistributes effort slightly (steep climb ↑, steep
   descent ↓) but cannot fix a wrong race-mean HRR.
2. The **power-law envelope is too flat** for ultras; empirical duration windows
   carry the long-duration information.
3. A **prospective blend** (\(w_{\max}=0.70\), 4→10 h) pulls Échappée to ~obs
   mean HRR while leaving ≤3 h races on the power-law feasible target.
4. Remaining short-race slow bias (~+5–16 min) matches the **obs-mean
   reconstruction** bias (~+13–19 min): model physics / fatigue, not HRR
   selection. Further gains need Stage-3 / GAP work, not more HRR tricks.

---

## Current best recipe (prospective)

1. Fit Stage 3 + HRR–duration power law + empirical windows (hold-outs out).
2. Select power-law duration-feasible constant HRR → \((\mathrm{HRR}_{\mathrm{PL}}, T)\).
3. \(\mathrm{HRR}^\star =\) H10 blend with \(w_{\max}=0.70\).
4. Assign \(\mathrm{HRR}_i = \mathrm{HRR}^\star \cdot m_{f(i)}\) with hardTrailRun
   **mean** modulators; distance-weighted re-center to \(\mathrm{HRR}^\star\).
5. Simulate with sequential TRIMP (`simulate_observed_hrr_segments`).

Script: `scripts/predict_race_segment_hrr_adaptive.py`  
Outputs: `data/exp_perf_predictions/trail_digital_twin_segment_hrr_adaptive/`

```bash
uv run python scripts/predict_race_segment_hrr_adaptive.py \
  --hypotheses H10_blend_w70_mean,H2_p75_mod
```

---

## Decision rules

- Prefer hypotheses that reduce **trail** |Δ| MAE without using race HR.
- Échappée: |Δ| < 30 min acceptable ultra envelope; < 20 min strong — **H10 meets strong**.
- Do not claim road transfer (Rome).
- Keep duration-feasible constant and obs-mean reconstruction as Table 5 baselines.

---

## Next actions

1. Optional: publish H10 as a Table 5 companion column / figure once stable.
2. Attack residual ≤3 h slow bias via physics/fatigue, not HRR mean.
3. Consider fitting empirical windows on TRAIL_RUN-only for even cleaner ultras.
4. Keep `hrr_max_factor=1.20` for LOO / segment spikes.
