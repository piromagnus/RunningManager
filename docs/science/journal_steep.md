# Journal: steep climb / descent Stage 3 residuals

Started: 2026-07-20  
Baseline run: `trail_digital_twin_moving_time_fit` / `fit_moving_time_only`  
Cohort for segment diagnosis: `hardTrailRun`, `fitObjective=segment`, moving-time residual.

## Baseline (before steep GAP work)

| terrain | n | MAE moving min | bias min | note |
|---|---:|---:|---:|---|
| steep_descent | 67 | 3.518 | −3.505 | model too fast |
| steep_climb | 59 | 2.679 | +2.262 | model too slow |
| descent | 191 | 1.584 | −1.475 | bleeds from steep |
| climb | 191 | 1.410 | +0.611 | |
| mixed_climb_descent | 110 | 1.428 | −0.341 | |
| flat | 218 | 0.882 | −0.061 | guardrail |

Full-race Stage 3 LOO MAE (activity): hardTrailRun 17.01 min, hardRunOrTrailRun 9.82 min.

Guardrail: flat MAE moving must stay ≤ **0.97 min** (+10% vs 0.88). Prefer not to worsen climb/flat/mixed by >10%.

## Physics snapshot

\[
t = \frac{d\cdot 3600\cdot f_\mathrm{GAP}}{v_\mathrm{VMA}\,\alpha\,f_\mathrm{alt}\,f_\mathrm{REDI}\,E(\mathrm{HRR})\,F}
\]

Empirical speed ratios (pred / act moving): steep_climb **0.90**, steep_descent **1.64**, flat **1.00**.  
Implied GAP correction: climb scale ≈ **0.87**, descent scale ≈ **1.58** on `|grade|≥0.15`.

---

## Hypotheses

### H1 — Minetti running descent GAP too optimistic on trail (HIGH) — CONFIRMED
Trail descents include braking / technical footing; treadmill downhill cost understates time.  
**Test:** multiply GAP by `gap_descent_scale > 1` with soft ramp from `|grade|=0.04` to `0.15`.  
**Result:** steep_descent MAE 3.65 → **1.36**, bias −3.65 → **−0.07**.

### H2 — Minetti climb GAP too high for steep trail (HIGH) — CONFIRMED (mean)
Athletes hike / shorten stride; run-cost overstates effort.  
**Test:** `gap_climb_scale < 1` with same soft ramp.  
**Result:** steep_climb MAE 2.44 → **1.92**, bias +1.69 → **−0.08**. Remaining MAE is mostly scatter, not bias.

### H3 — Asymmetric scales needed (HIGH) — CONFIRMED
Opposite-signed biases; flat already good. Joint fit of climb+descent scales beats either alone.  
**Winner:** `gap_climb_scale=0.85`, `gap_descent_scale=1.60`, `gap_soft_start=0.04`, `gap_steep_threshold=0.15`.

### H4 — HRR linear effort on steep climbs (MEDIUM) — DEFERRED
Climb mean bias already ~0 after H2; leftover MAE looks heteroscedastic / activity-specific. Revisit if race LOO needs more.

### H5 — Technicality secondary (LOW) — DEFERRED

---

## Experiments

### Exp 0 — harness
- Added `trail_gap_multiplier` / `apply_trail_gap_multipliers` and wired through Stage 3 predict/grid via physiology.
- Focused grid: `scripts/steep_gap_scale_grid.py` on 836 hardTrailRun moving-time segments.

### Exp 1 — hard threshold grid
Climb ∈ {1.0…0.80}, descent ∈ {1.0…1.80}, threshold=0.15.  
Best eligible: **0.85 / 1.60**, steep MAE **1.72**, flat **0.88**.

### Exp 2 — refine + threshold 0.12
Finer climb grid; threshold 0.12 slightly helps steep but nudges moderate climb MAE up. Keep threshold **0.15**.

### Exp 3 — soft ramp (0.04 → 0.15)
Ramped scales improve moderate descent without hurting flat.

| terrain | baseline MAE | winner MAE | baseline bias | winner bias |
|---|---:|---:|---:|---:|
| steep_descent | 3.653 | **1.361** | −3.648 | **−0.066** |
| steep_climb | 2.443 | **1.918** | +1.691 | **−0.076** |
| descent | 1.719 | **1.107** | −1.635 | **+0.045** |
| climb | 1.373 | **1.329** | +0.214 | **+0.048** |
| mixed | 1.468 | **1.348** | −0.602 | −0.226 |
| flat | 0.910 | **0.860** | −0.322 | **−0.032** |

Steep combined MAE: **3.05 → 1.64** (−46%). Flat improved (within guardrail).

### Full-race LOO (moving-time fit, activity objective)

| cohort | baseline MAE | winner MAE | MAPE |
|---|---:|---:|---|
| hardTrailRun | 17.73 | 17.74 | 8.99 → **8.72** |
| hardRunOrTrailRun | 9.82 | **9.57** | 7.50 → **7.31** |

Race MAE on hardTrailRun is almost unchanged (climb/descent errors previously cancelled). Segment-level physics is much healthier; hardRunOrTrailRun race MAE improves ~0.25 min.

---

## Decision / defaults

Ship physiology defaults:

```yaml
gap_steep_threshold: 0.15
gap_soft_start: 0.04
gap_climb_scale: 0.85
gap_descent_scale: 1.60
```

Code: `services/trail_performance_model.py` (`trail_gap_multiplier`)  
Config: `configs/trail_digital_twin_extensions.yaml`  
Artifacts: `data/exp_perf_predictions/trail_digital_twin_steep_gap/`

## Open questions / next

1. Remaining steep_climb scatter (MAE ~1.9 with ~0 bias) — inspect outlier activities (La croix steep pins, Verticale).
2. Optional H4: grade×HRR interaction only if race LOO needs another cut.
3. Re-run full benchmark with new defaults when convenient for session review refresh.
