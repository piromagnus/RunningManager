# Remaining experiments for robust / publishable results

Companion to `trail_digital_twin_hr_performance_paper_draft.md` §7 and
`section7_implementation_status.md`.

**Canonical headline (single athlete, §7 pipeline):** hard run/trail LOO MAE
**9.09 min** (MAPE 6.5%) vs physics baseline **30.2 min**; steep terrain MAE
**−46%**; prospective LUT **−4.1 min**, Grésivaudan **−17.0 min** (≈−5 with aid budget), Rome **out of scope**.

**Robustness suite:** `docs/science/robustness_experiments_report.md` (R1–R11 run; R3 partial).

---

## A. Runnable experiments — status after robustness pass

| ID | Status | Finding |
|----|--------|---------|
| **R1** | **Done** | Table 2/3 full MAE aligned at 9.09 min |
| **R2** | **Done** | Activity objective frozen without hold-out peeking |
| **R3** | **Open** | Need ≥2–3 more trail prospective races |
| **R4** | **Done** | Aid budget: Grésivaudan −17 → ≈−5 min |
| **R5** | **Done** | Rome road out of scope |
| **R6** | **Done** | Race-date LOO 11.8 min (90% CI 6.5–18.1) |
| **R7** | **Done** | HR QC all = 1.0; no sensitivity |
| **R8** | **Done** | Steep outliers named |
| **R9** | **Done** | Non-degenerate finish bands |
| **R10** | **Done** | §7 headline ≈9.09 under shipped GAP |
| **R11** | **Done** | κ floor 0.20 binding but protective |

---

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
