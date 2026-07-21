# Remaining experiments for robust / publishable results

Companion to `trail_digital_twin_hr_performance_paper_draft.md` §7 and
`section7_implementation_status.md`.

**Canonical headline (single athlete, §7 pipeline):** hard run/trail LOO MAE
**9.09 min** (MAPE 6.5%) vs physics baseline **30.2 min**; steep terrain MAE
**−46%**; prospective LUT **−4.1 min**, Grésivaudan **−17.0 min**, Rome **−79.2 min**.

---

## A. Runnable now (no new athletes / DEM required)

| ID | Experiment | Purpose | Method | Success criterion |
|----|------------|---------|--------|-------------------|
| **R1** | Reconcile Table 2 vs Table 3 “full” MAE | Internal consistency | Single LOO entry point for Stage-3 full and ablation baseline; re-run §7 ladder + ablation under stable SHA-256 LOO caps | Hard run/trail full MAE matches M3 within rounding (today: Table 2 **9.09** vs Table 3 **8.24**) |
| **R2** | Nested CV for race vs segment objective | Avoid prospective overfit | Hold out older races; choose activity vs segment objective on validation only; freeze for prospective | Objective choice justified without peeking at LUT/Grésivaudan |
| **R3** | Expand prospective trail hold-outs | Strengthen prediction claim | Preregister ≥2–3 additional races with race_pacing profiles; constant HRR = 0.88 | Qualitative band: pred ≤ observed when race HRR < 0.88 |
| **R4** | Non-fitted aid-time budget | Realistic race plans | Add planned aid minutes as post-hoc additive; do not refit α,κ | Grésivaudan Δ moves toward 0 without new physiology knobs |
| **R5** | Road / flat recalibration or exclusion | Transfer limits | Fit GAP scales (or disable trail scales) on road cohort; re-evaluate Rome | Rome Δ not catastrophic **or** road explicitly out of scope |
| **R6** | Blocked LOO / by-race-date folds | Honest uncertainty | Outer folds by race date / blocked activities; bootstrap α,κ | CIs stable; no same-race leakage |
| **R7** | HR QC sensitivity | Strap dropout | Exclude activities below `hrValidShare` thresholds; recompute LOO | MAE change within agreed ε (e.g. ≤1 min) |
| **R8** | Steep-climb residual diagnostics | Residual MAE ~1.9 min | Inspect Verticale / La Croix-style outliers; optional Winsorize | Mechanism identified or scatter accepted in Discussion |
| **R9** | Prospective uncertainty bands fix | Coaching utility | Proper resampling of LOO (α,κ) → finish times (fix degenerate P05=point) | P05 < P50 < P95 with meaningful spread |
| **R10** | Full benchmark refresh with shipped GAP | Align ops defaults | Re-run extensions benchmark with climb 0.85 / descent 1.60 | Leaderboard ~9 min on primary cohort under paper defaults |
| **R11** | κ grid boundary check | Hard-trail κ pinned at 0.2 | Extend κ grid below/above 0.2; report sensitivity | Document whether floor is binding |

---

## B. Blocked on data / infrastructure

| ID | Experiment | Blocker | Minimal unlock |
|----|------------|---------|----------------|
| **B1** | Multi-athlete replication (≥8–15) | Single-athlete repo | Consent + anonymized GPS/HR release |
| **B2** | Sex / age / level strata | Single athlete | Multi-athlete cohort |
| **B3** | DEM-corrected elevation | No DEM pipeline | DEM attach on segments / Grésivaudan altitude QA |
| **B4** | Weather / heat multiplier | Sparse temp (≈0–9% coverage) | Systematic weather logging |
| **B5** | Structured aid / nutrition logs | Not in data | Aid-station timestamps beyond moving-time proxy |

---

## C. Deferred modeling (optional, not v1 gate)

| ID | Idea | Note |
|----|------|------|
| **D1** | Grade × HRR interaction (H4) | Steep-climb heteroscedasticity; LOO gain ≥ ~1 min without flat guardrail break |
| **D2** | Long-term Banister fitness–fatigue | Between-activity readiness beyond REDI/CTL |
| **D3** | Descent-specific eccentric cost | Beyond GAP descent scale |
| **D4** | Heat / RPE fusion when HR saturates | Needs weather + RPE |

---

## D. Reporting gates (submission)

- Explicit **single-athlete case study** framing (or B1).
- Ethics / data-availability statement; cite `software_versions.json`.
- STROBE-like observational reporting checklist.
- Complete bibliography (`bibliography_hr_digital_twin.md` §H).
- Banner or archive stale `trail_digital_twin_hrr_trimp_paper.md` numbers.
