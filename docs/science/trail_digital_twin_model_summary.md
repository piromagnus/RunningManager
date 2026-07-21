# Trail digital twin — model summary

Athlete-specific vs general parameters, and what they mean for race prediction.

## Core equation (Stage 3)

For each course segment:

\[
t_\mathrm{seg}
=
\frac{d\cdot 3600\cdot f_\mathrm{GAP}(i)}
{v_\mathrm{VMA}\cdot\alpha\cdot f_\mathrm{alt}\cdot f_\mathrm{REDI}\cdot E(\mathrm{HRR})\cdot F(\mathrm{load})}
\]

| Symbol | Meaning |
|--------|---------|
| \(d\) | Segment distance (km) |
| \(f_\mathrm{GAP}(i)\) | Grade cost (Minetti running) × trail climb/descent soft-ramp scales |
| \(v_\mathrm{VMA}\) | Flat speed anchor (km/h) |
| \(\alpha\) | Free speed factor (fitness / how close to VMA the athlete races) |
| \(f_\mathrm{alt}\) | Thin-air VO₂ penalty from mean altitude |
| \(f_\mathrm{REDI}\) | Day-of readiness from chronic load balance |
| \(E(\mathrm{HRR})\) | Effort from heart-rate reserve vs a reference |
| \(F(\mathrm{load})\) | In-race fatigue from cumulative / decayed TRIMP |

Race time = sum of segment times (prospective: sequential predicted TRIMP).

---

## Athlete-specific (must be personal)

| Parameter | Source | Intuition |
|-----------|--------|-----------|
| **hrRest / hrMax** | `athlete.csv` | Define HRR = (HR − rest)/(max − rest). Wrong max/rest shifts all efforts. |
| **\(v_\mathrm{VMA}\)** | thresholds / config (`vma_flat_kmh`) | Flat “engine size”. Higher → faster everywhere at same effort. |
| **\(\alpha\)** | Fitted on *this* athlete’s segments | How hard they typically run relative to VMA. Race form / efficiency. |
| **\(\kappa\) / fatigue model** | Fitted | How fast TRIMP eats speed mid-race. Sensitive athlete → higher \(\kappa\). |
| **REDI / CTL readiness** | Daily metrics | Freshness on race day. Often ~1.0 if not modeling taper. |
| **Observed HRR on past races** | activities | For *retrospective* replay only — **not** for prospective “best HRR” planning. |

These change when the athlete changes (fitness, aging, device HR).

---

## General / shared structure (not athlete-tuned)

| Parameter | Typical value | Intuition |
|-----------|---------------|-----------|
| **Minetti GAP polynomial** | literature | Physics of grade cost for *running*. |
| **`gap_climb_scale` / `gap_descent_scale`** | 0.85 / 1.60 | Trail correction: steep climbs slightly easier than pure run-cost; descents cost more (braking). Soft-ramp 4%→15%. Tuned on this athlete’s trails; treat as **semi-general** until revalidated. |
| **Altitude VO₂ formula** | Sensors paper | Shared physiology. |
| **HRR effort law** | \(E=\mathrm{clip}(\mathrm{HRR}/h_\mathrm{ref},0.3,1.0)\) | Shape is general; \(h_\mathrm{ref}\) is a modeling choice (often ~0.85–0.88). |
| **Segment length** | 1 km | Aggregation choice. |
| **Stationary strip** | `actualMovingTimeSec` | Fit hygiene: remove device-open dwell from fit targets. |
| **TRIMP formula** | Banister-style | Standard acute load from duration × HRR. |

---

## Prospective race prediction (duration-feasible constant HRR)

**Idea:** Before the race, sweep constant HRR on the planned profile. For each candidate \(x\), predict finish time \(T(x)\) and keep \(x\) only if \(T(x)\) ≤ the athlete’s historically maintainable duration at HRR \(x\) (power-law envelope from other activities). Select the **fastest feasible** HRR. \(\mathrm{HRR}_{\mathrm{ref}}\) remains the \(E=1\) normalization/ceiling—not the default race HRR and not “HRR at VMA.”

| Allowed as input | Forbidden for estimation |
|------------------|--------------------------|
| GPX / planned race profile (distance, elevation) | Observed race HR stream |
| Athlete HR limits, VMA, \(\alpha,\kappa\) fitted on **other** activities | Observed race segment times / moving time |
| HRR–duration envelope from **other** activities | Hold-out races in the envelope fit |
| Readiness if known a priori | Fitting \(\alpha\) on the race itself |

**Modes:** `duration-feasible` (default prospective) vs `reference` (hold HRR = `hrr_reference` for \(E=1\) ceiling comparison). Optional: `--hard-hrr` fixed override.

Prediction should usually be **slightly faster** than a near-max executed race (small pacing/aid inefficiencies remain in reality).
