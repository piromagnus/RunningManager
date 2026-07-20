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

### H1 — Minetti running descent GAP too optimistic on trail (HIGH)
Trail descents include braking / technical footing; treadmill downhill cost understates time.  
**Test:** multiply GAP by `gap_descent_scale > 1` when `avgGrade ≤ −0.15`.  
**Success:** steep_descent MAE < 2.0 min; flat ≤ 0.97.

### H2 — Minetti climb GAP too high for steep trail (HIGH)
Athletes hike / shorten stride; run-cost overstates effort.  
**Test:** multiply GAP by `gap_climb_scale < 1` when `avgGrade ≥ 0.15`.  
**Success:** steep_climb MAE < 1.8 min; flat ≤ 0.97.

### H3 — Asymmetric scales needed (single α cannot fix both tails) (HIGH)
Opposite-signed biases; flat already good.  
**Test:** fit both scales jointly; keep global α.  
**Success:** combined steep MAE ↓ ≥30% without flat ↑ >10%.

### H4 — HRR linear effort wrong on steep climbs (MEDIUM)
Deferred unless H1–H3 leave climb bias >1 min.

### H5 — Technicality / late-race fatigue secondary (LOW)
Deferred.

---

## Experiments

### Exp 0 — harness
- Add `trail_gap_multiplier(grade)` + wire `gap_climb_scale` / `gap_descent_scale` / `gap_steep_threshold` through physiology → Stage 3 predict/grid.
- Focused grid on hardTrailRun moving-time segments (reuse built features).

*(results filled below as we iterate)*
