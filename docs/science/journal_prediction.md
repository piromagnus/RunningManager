# Journal: prospective constant-HRR race prediction

Started: 2026-07-20  
Goal: predict finish times at **constant \(\mathrm{HRR}=\mathrm{HRR}_{\mathrm{ref}}\)**
(\(E=1\) reference effort under the paper effort law) for LUT 30k and Grésivaudan
**before** the race (profile only).  
Note: \(\mathrm{HRR}_{\mathrm{ref}}\) is the normalization/ceiling of \(E(\mathrm{HRR})\),
**not** “HRR at VMA” (VMA is `vma_flat_kmh`; flat fresh speed at \(E=1\) is \(v_{\mathrm{VMA}}\cdot\alpha\)).  
Hard rule: **no observed race HR / times / streams in fit or prediction**. Actuals used only for evaluation.

Model summary (athlete vs general): `docs/science/trail_digital_twin_model_summary.md`

---

## Hold-outs

| Race | activityId | Actual moving | Observed avg HRR (eval only) |
|------|------------|---------------|------------------------------|
| Lyon Urban Trail By Night 2025 | `16325125849` | **2h57m18s** (177.3 min) | 0.795 |
| Trail du Grésivaudan 2026 Le Grand V | `17481444994` | **3h32m59s** (213.0 min) | 0.735 |

Athlete expectation (qualitative):
- LUT: near max; maybe **a few minutes** left on the table.
- Grésivaudan: almost max; maybe **slightly** faster.
- Therefore prediction at best HRR should be **≤ actual**, closer for Grésivaudan than a large gap.

---

## Method (v1)

1. Train Stage 3 on `hardRunOrTrailRun` segments **excluding** both hold-out IDs.
2. Physiology (existing, no new knobs): `vma=18`, `hrr_ref=0.88`, `gap_climb=0.85`, `gap_descent=1.60`, `λ=0.2`, fatigue floor `0.6`.
3. Course = planned `race_pacing` segments (D+ close to race) + mean altitude from GPX (not race GPS).
4. Constant \(\mathrm{HRR}=\mathrm{HRR}_{\mathrm{ref}}=0.88\) → effort multiplier \(E=1\) (higher HRR cannot raise \(E\) because `hrr_max_factor=1.0`; it only adds TRIMP fatigue). This is reference effort, not lab VMA heart rate.
5. Sequential `cumTrimpBefore` fatigue along the course.
6. Script: `scripts/predict_race_constant_hrr.py`

Leak check: train activities ∩ hold-outs = ∅.

---

## Hypotheses tested

### H1 — GPX-only profile is enough
**Why:** Official GPX is pre-race.  
**Result:** GPX understates D+ (LUT 516 vs Strava ~1008; Grésivaudan 1238 vs ~1616). Predictions too sensitive to wrong climb cost.  
**Verdict:** **Reject as primary.** Use planned `race_pacing` D+ instead (LUT 980, Grésivaudan 1645). GPX kept for altitude only.

### H2 — Segment-objective fit (α,κ) transfers to races
**Fit:** α=**0.90**, κ=**0.30**, exponential.  
**Pred (race_pacing+alt, HRR=0.88):**

| Race | Pred | Δ vs actual |
|------|------|-------------|
| LUT | 3h01m06s | **+3.8 min** (slower than actual) |
| Grésivaudan | 3h24m22s | **−8.6 min** |

**Verdict:** Grésivaudan direction OK (faster); LUT **fails** “faster than actual”. Segment calibration is conservative on runnable urban trails.

### H3 — Race-objective fit better for “best race time”
**Fit:** α=**0.95**, κ=**0.40**, exponential (same grids, no new params).  
**Pred:**

| Race | Pred | Δ vs actual |
|------|------|-------------|
| LUT | **2h53m10s** | **−4.1 min** |
| Grésivaudan | **3h16m00s** | **−17.0 min** |

**Verdict:** LUT matches athlete story (few minutes headroom). Grésivaudan **too optimistic** vs “slightly faster” (~17 min).

### H4 — Higher constant HRR than 0.88 helps
**Why:** “Max effort”.  
**Result:** With `hrr_max_factor=1.0`, HRR>0.88 does **not** increase speed; it increases TRIMP and can **slow** the prediction. Best constant hard under current effort law is exactly `hrr_reference`.  
**Verdict:** Confirmed; do not sweep upward without changing the effort ceiling (that would be a new modeling choice).

### H5 — Decayed vs cumulative TRIMP for prospective fatigue
Decayed is slightly faster (~2–4 min). Cumulative is the monotone “work done” story for a planned race.  
**Verdict:** Keep **`cumTrimpBefore`** for prospective; decayed as sensitivity only.

### H6 — Altitude from GPX on pacing segments
Grésivaudan GPX max ~918 m → only ~1 min slower than sea-level. Real Chartreuse altitudes may be higher; GPX may be incomplete.  
**Verdict:** Partial fix; remaining optimism on Grésivaudan not explained by altitude alone.

---

## Critical synthesis (avoid overfit)

| Choice | LUT | Grésivaudan | Risk |
|--------|-----|-------------|------|
| Segment fit (H2) | too slow (+3.8) | plausible (−8.6) | Underestimates runnable races |
| Race fit (H3) | good (−4.1) | too fast (−17) | Overestimates mountain races |

**No new parameters** were added to “fix” either race. Cherry-picking α between 0.90 and 0.95 would be evaluation-set overfit.

### Recommended reporting (primary = race-objective)

For **expected time if constant hard HRR = 0.88** (best effort under current effort law):

| Race | Predicted | Actual moving | Δ | Interpretation |
|------|-----------|---------------|---|----------------|
| **LUT 30k** | **2h53m10s** | 2h57m18s | −4.1 min | Aligns with “few minutes left” |
| **Grésivaudan** | **3h16m00s** | 3h32m59s | −17.0 min | Direction OK but **optimistic**; treat as upper-bound form, not a tight forecast |

Conservative companion (segment fit): LUT 3h01 / Grésivaudan 3h24 — use when preferring not to over-promise on mountain races.

---

## What worked / what did not

**Worked**
- Strict hold-out of both races from fit.
- Planned pacing profile for D+ (not GPX D+).
- Constant hard = `hrr_reference` (effort 1.0) with sequential TRIMP.
- Existing GAP trail scales + physiology; no new knobs.
- LUT race-objective prediction matches qualitative athlete feedback.

**Did not work**
- GPX-only D+.
- Segment-objective for LUT (predicts slower than a near-max race).
- Race-objective magnitude on Grésivaudan (17 min gap > “slightly faster”).
- Raising HRR above reference under `hrr_max_factor=1.0`.

---

## How to improve next (still no random params)

1. **Better altitude profile** for Grésivaudan (DEM / official course elev) — not athlete race data.
2. **Aid / slowdown budget** as a *known race-plan input* (minutes), not a fitted parameter — would move Grésivaudan prediction toward actual without touching α.
3. Revisit whether race- vs segment-objective should be chosen by validation on *older* races only (nested hold-out), not these two.
4. Only if needed later: allow `hrr_max_factor>1` for short races (explicit hypothesis, not silent overfit).

---

## Artifacts

```bash
PYTHONPATH=/workspace uv run python scripts/predict_race_constant_hrr.py --fit-objective race
PYTHONPATH=/workspace uv run python scripts/predict_race_constant_hrr.py --fit-objective segment \
  --output-dir data/exp_perf_predictions/trail_digital_twin_race_prediction_segment
```

- `data/exp_perf_predictions/trail_digital_twin_race_prediction/` — primary (race objective)
- `…/trail_digital_twin_race_prediction_segment/` — conservative companion
- Sensitivities: `hrr_sensitivity_race_pacing.csv`, `cohort_sensitivity.csv`, `altitude_sensitivity.csv`


---

## Addendum — duration-feasible HRR (2026-07-21)

Primary prospective mode is now **duration-feasible**: sweep constant HRR, keep
only values where predicted finish ≤ power-law max maintainable duration at that
HRR (hold-outs excluded from envelope), pick fastest feasible.

| Race | Feasible HRR | Δ min | Reference E=1 Δ (HRR_ref=0.88) |
|------|--------------|-------|-------------------------------|
| LUT | 0.78 | +16.7 | −4.1 |
| Passerelles | 0.78 | +13.7 | −8.0 |
| Grésivaudan | 0.78 | +6.7 | −17.0 |
| Échappée Belle | 0.75 | −73.2 | −164 |
| Rome | 0.79 | −62.4 | −79 (out of scope) |

`--hrr-mode reference` restores the previous E=1 ceiling scenario.
