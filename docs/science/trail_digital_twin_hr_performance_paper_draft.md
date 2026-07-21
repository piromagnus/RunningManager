# Heart-Rate Digital Twin for Trail-Running Performance: Modeling, Cross-Activity Evaluation, and Measured Prediction Errors

**Working draft (v0.5)** — findings recap + robustness / remaining-experiments checklist  
**Status:** single-athlete proof-of-concept; multi-athlete replication planned  
**Companion code:** Running Manager (`services/trail_performance_model.py`, `scripts/predict_race_constant_hrr.py`)  
**Publication assets:** `docs/science/paper/` (figures PNG + tables)  
**Related docs:** `trail_digital_twin_model_summary.md`, `journal_steep.md`, `journal_prediction.md`, `section7_implementation_status.md`, `remaining_experiments.md`  
**Annotated bibliography:** `bibliography_hr_digital_twin.md`  
**Note:** Older draft `trail_digital_twin_hrr_trimp_paper.md` is superseded for headline numbers.

---

## Abstract (draft)

Accurate prediction of trail-running finish times remains difficult because grade, altitude, and fatigue interact nonlinearly with athlete physiology. Building on the physics-informed digital-twin framework of Jaén-Carrillo and Pattis (2026), we evaluate a **single athlete-specific** model that uses continuous heart-rate reserve (HRR) as instantaneous effort and Banister-style TRIMP as intra-activity fatigue, with soft-ramped trail-grade cost corrections. Using leave-one-out (LOO) validation across hard-run, hard-trail, and run/trail (>20 min) cohorts—with moving-time fitting and slight near-flat immobile segment rejection—a physics baseline (M0) yielded substantially larger errors than the full HRR+TRIMP specification (M3): for mixed hard run/trail activities, MAE decreased from **30.2 to 9.1 min** (MAPE 26.5% → 6.5%). Re-optimized component ablations attribute most of this gain to HRR and acute TRIMP. Soft-ramped trail GAP scales reduced steep-terrain segment MAE by ≈46%. Prospective constant-HRR simulations for held-out races produced finish times at or faster than observed performances when race HRR was submaximal (LUT −4.1 min; Grésivaudan −17.0 min), while a road marathon (Rome) failed to transfer (−79 min). Publication assets: `docs/science/paper/`.

**Keywords:** trail running; digital twin; heart-rate reserve; TRIMP; performance prediction; leave-one-out; grade-adjusted pace

---

## 1. Introduction

### 1.1 Motivation

Trail races combine steep grades, altitude, aid stops, and pacing decisions. Physics-only cost-of-transport models (Minetti grade cost, grade-adjusted pace, altitude VO₂ correction) explain much of the variance in finish time but systematically miss athlete-specific **effort regulation** and **fatigue accumulation**. Wearable HR provides a continuous, field-ready proxy for metabolic intensity when laboratory VO₂ is unavailable.

### 1.2 Positioning relative to the 2026 digital twin

Jaén-Carrillo and Pattis (2026) proposed a physics-based digital twin for trail racing with LOO validation on 13 races (**MAE 18.2 min, MAPE 11.1%, R² = 0.864**). Their stack uses GAP/Minetti, altitude, Banister TRIMP load, and pacing decay—but **not** continuous HR as the primary instantaneous effort signal. Individual parameters are a sustainable VT2 fraction and a pacing-decay slope.

**This paper’s focus (intentionally narrow):**

1. **One modeling family** — HRR-scaled equivalent speed + TRIMP fatigue, with optional trail GAP soft-ramp corrections.  
2. **Diverse datasets for the same athlete** — road/trail/race activities for fitting; races held out for LOO / prospective tests.  
3. **Measured errors** — MAE, MAPE, signed bias, R² at race and segment levels.  
4. **HR as better effort estimation** — effort-aware predictions vs route-only / fixed-effort baselines where available.

We deliberately defer multi-athlete transfer, sensor fusion, and product UX to a later study (§7).

### 1.3 Research questions

- **RQ1.** Does an athlete-specific HRR+TRIMP digital twin reduce LOO race-time error relative to physics-only references on diverse trail/road hard efforts?  
- **RQ2.** How do errors distribute across terrain classes (flat, climb, steep climb/descent)?  
- **RQ3.** Can the same calibrated twin produce **prospective** constant-HRR race predictions that are plausibly faster than realized race times (upper-bound hard pacing)?

---

## 2. Related work

Full citations: **§9** and `bibliography_hr_digital_twin.md`.

### 2.1 Digital twins and physics-informed race models

| Work | Contribution | Relevance |
|------|---------------|-----------|
| Jaén-Carrillo & Pattis, *Sensors* 2026 | Trail digital twin; GAP/Minetti + altitude + Banister + pacing; LOO n=13 | Direct baseline structure and metrics |
| Boillet et al., *Sci. Rep.* 2024 | Margaria–Morton digital twin (cycling) | Digital-twin + physiological cost outside running |
| Minetti et al., *J. Appl. Physiol.* 2002 | Metabolic cost vs grade (−45%…+45%) | Grade cost backbone |
| Tobler / Scarf hiking-speed literature | Grade vs walking speed | Steep locomotion priors |
| Daniels VDOT family | Road performance curves | Flat-speed reference |
| West; Péronnet–Thibault | Altitude VO₂ correction | Hypoxia multiplier |

### 2.2 Heart rate, HRR, and training load

| Work | Contribution | Relevance |
|------|---------------|-----------|
| Banister et al. | Fitness–fatigue; TRIMP | Fatigue accumulation prior |
| Edwards zones; Lucia; Manzi iTRIMP (2009) | Session / individualized TRIMP | Continuous HR weighting |
| Impellizzeri / Foster session-RPE | Subjective load vs HR load | Fallback when HR missing |
| Borresen & Lambert; IJSPP load comparisons | Objective vs subjective load | Supports HR-based internal load |
| CISS field trail thresholds (2025/2026) | Outdoor trail HR ↔ lab thresholds | Ecological HR zones for trail |
| Free-living wearable HR → race prediction | HR as prediction feature | Applied precedent |

### 2.3 Trail pacing, GAP, and field performance

| Work | Contribution | Relevance |
|------|---------------|-----------|
| Operational GAP / Strava-style practice | Equivalent flat distance | Operational GAP |
| Genitrini et al., *Front. Sports* 2024 | Trail kinematics / fatigue across stages | Why UH/DH costs differ mid-race |
| Ultra pacing / durability literature | Pacing decay, DNF risk | Fatigue context |

**Gap addressed here:** continuous **HRR as instantaneous effort**, combined with **intra-activity TRIMP**, evaluated with **LOO-style error reporting** similar to the 2026 twin, on a **heterogeneous activity corpus** (not races alone).

---

## 3. Methods

### 3.1 Data

**Athlete.** One recreational/competitive trail runner (male); HR max and resting HR from athlete profile / thresholds.

**Activities.** GPS+HR streams (Strava/Garmin). Corpus spans road/track hard runs, trail long runs, and races with near-flat to steep alpine grades.

**Evaluation cohorts (as used in experiments).**

| Cohort | Role |
|--------|------|
| `hardRunOrTrailRun` (*n* = 103; LOO capped at 80) | Mixed hard efforts; **primary** aggregate LOO |
| `hardTrailRun` (*n* = 45) | Trail-only races/hard sessions; stress test |
| `runTrailOver20Min` (*n* = 208; LOO capped at 80) | Broader intensity mix |
| `selectedDateRaces` (*n* = 18) | Race-date LOO / objective comparison |
| LUT 30k, Grésivaudan, Rome marathon | Prospective constant-HRR hold-outs (preregistered) |

**Preprocessing.**

- Segment activities (~1 km equivalent or route-based bins).  
- Apply a **slight near-flat immobile rejection** for physiology fitting (§3.1.1).  
- Fit on **moving time** (`actualTime − stationaryTime`) so aid-station dwell does not inflate residuals; report full clock time separately when needed.

#### 3.1.1 Slight segment rejection

Aid stops, traffic lights, and device-open dwell contaminate grade–speed physiology if treated as locomotion. We therefore mark a segment as unfit for parameter estimation when it is simultaneously:

1. **Near-flat in altitude–time**, defined as gross elevation change rate `|Δelev|/hour ≤ 120 m·h⁻¹` (not distance grade, so steep hikes remain eligible); and  
2. **Immobile**, defined as grade-adjusted speed `meanSpeedEqKmh < 3 km·h⁻¹` **or** stationary time share `> 0.40`.

Rejected segments retain their observed times for full-race evaluation but are withheld from the Stage-3 fit mask (`isFitEligible = false`). The paper pipeline pairs this **slight** gate with moving-time fitting, so only a few residual near-flat dwells remain after stationary scrubbing. A stricter moderate gate (`speedEq < 4`, share `> 0.30`) is reported as a sensitivity analysis (Table 7).

### 3.2 Model (single family)

Stage-3 athlete model (aligned with codebase notation):

\[
v_{\mathrm{eq}}^{\mathrm{eff}}(t)
=
v_{\mathrm{eq,base}}(\mathrm{grade},\,\mathrm{altitude})
\cdot
E\bigl(\mathrm{HRR}(t)\bigr)
\cdot
F\bigl(\mathrm{TRIMP}_{\mathrm{cum}}(t)\bigr)
\]

with

\[
\mathrm{HRR}=\frac{\mathrm{HR}-\mathrm{HR}_{\mathrm{rest}}}{\mathrm{HR}_{\mathrm{max}}-\mathrm{HR}_{\mathrm{rest}}},
\quad
E = \mathrm{clip}\bigl(1+\alpha\,(\mathrm{HRR}-\mathrm{HRR}_{\mathrm{ref}})\bigr),
\]

\[
\mathrm{TRIMP}\propto \Delta t\cdot\mathrm{HRR}\cdot e^{b\cdot\mathrm{HRR}},
\quad
F = \max\bigl(F_{\min},\, 1-\kappa\cdot\mathrm{TRIMP}_{\mathrm{cum}}/\mathrm{TRIMP}_{\mathrm{norm}}\bigr).
\]

**Base equivalent speed** combines flat reference (athlete VMA / `v_flat`), Minetti/GAP grade multiplier, optional **soft-ramped trail GAP scales** on steep climb/descent (`gap_climb_scale≈0.85`, `gap_descent_scale≈1.60`; soft start ~4% → full at ~15%), and altitude VO₂ factor.

**Fitted per athlete / LOO fold:** primarily \(\alpha\) (HRR gain) and \(\kappa\) (TRIMP fatigue). Defaults used in prospective runs: `hrr_reference=0.88`, `hrr_max_factor=1.0`, `decay_lambda=0.20`, `min_fatigue_factor=0.60`.

**Prediction modes.**

1. **Reconstruction** with observed HR (diagnostic fit).  
2. **Prospective race** with **constant HRR** on a pre-race profile (race_pacing D+ + GPX altitude)—no target-race HR/times in the fit.

### 3.3 Experimental protocol

| ID | Design | Metrics |
|----|--------|---------|
| **E1** | LOO on activity cohorts | MAE, MAPE, bias, R² (minutes) |
| **E2** | Segment residuals by terrain class | MAE, bias (min) |
| **E3** | Prospective constant-HRR (LUT, Grésivaudan, Rome) | Pred vs actual moving time; Δ min |
| **E4** | Component ablations with **re-optimized** (α, κ) per removal; LOO eval | ΔMAE vs full |
| **E5** | Slight segment rejection A/B | Rejected share; LOO MAE |
| **E6** | Segment optimisation: trail GAP scales; segment vs race objective | Terrain MAE/bias; LOO Δ |

**“Diverse datasets”** in this draft = diverse **activities and terrains for one athlete**. Multi-athlete extension is planned (§7).

### 3.4 Implementation

Python/pandas CSV storage; YAML configs (`configs/trail_digital_twin_*.yaml`); grid search / LOO for \((\alpha,\kappa)\); scripts for steep GAP grid and constant-HRR prediction.

---

## 4. Results

Publication tables and figures are assembled under `docs/science/paper/`  
(regenerate with `uv run python scripts/prepare_paper_assets.py`).  
Unless stated otherwise, leave-one-out (LOO) metrics use the **activity** fit objective on moving-time–eligible segments.

### 4.1 Cohorts

Five evaluation cohorts were defined from the same athlete archive (Table 1 / `table01_cohort_descriptives`): hard trail runs (*n* = 45), hard run or trail runs (*n* = 103), all run/trail activities longer than 20 min (*n* = 208), a top-10 hard-trail subset by mean HRR, and selected race dates (*n* = 18). The >20 min cohort broadens the intensity distribution beyond race-like efforts while retaining usable GPS+HR streams.

### 4.2 Baseline physics twin and successive additions (E1)

We report LOO finish-time error for a nested model ladder (Table 2 / `table02_incremental_model_loo`; Fig. `fig_incremental_model_mae.png`):

| ID | Specification |
|----|---------------|
| **M0** | Physics baseline: Minetti/GAP grade cost, altitude correction, CTL readiness, progress fatigue |
| **M1** | M0 with acute TRIMP replacing progress fatigue |
| **M2** | M1 with REDI readiness replacing CTL |
| **M3** | M2 with continuous HRR effort (full Stage-3 twin) |

On the hard run/trail cohort, M0 yielded MAE = 30.2 min (MAPE = 26.5%). Adding acute TRIMP (M1) reduced MAE to 23.9 min, whereas replacing CTL by REDI (M2) left error essentially unchanged (24.2 min). Introducing continuous HRR effort (M3) produced the dominant improvement: MAE = 9.1 min, MAPE = 6.5%, *R*² = 0.982 (ΔMAE vs M0 = −21.1 min). M3 includes the paper physiology defaults (trail GAP soft-ramp scales 0.85 / 1.60).

The same M0→M3 ordering held on selected race dates (36.4 → 11.8 min) and on run/trail >20 min (9.4 → 5.1 min). On **hard trail alone**, M1/M2 temporarily *worsened* error relative to M0 (44.5 → 47.2–47.7 min) before M3 recovered to 17.4 min: acute TRIMP without continuous HRR mis-fits heterogeneous submaximal trail efforts. Agreement plots: Figs. `fig_pred_vs_actual_*.png`; Bland–Altman: Figs. `fig_bland_altman_*.png`.

For reference, Jaén-Carrillo and Pattis (2026) reported LOO MAE = 18.2 min and MAPE = 11.1% (*n* = 13 races) without continuous HR effort. Our hard-trail M3 MAE (17.4 min) is of similar absolute magnitude on a different athlete and race set; mixed hard-run/trail and >20 min cohorts achieve substantially lower absolute error. Direct numerical comparison remains indicative.

Bootstrap percentile intervals on M3 LOO folds (Table 4) place hard run/trail MAE at 9.1 min (90% CI 6.6–11.9) with fold-mean α ≈ 0.95 and κ ≈ 0.39. On hard trail, κ sits at the grid floor (0.20)—a sensitivity item for robustness work (§7).

### 4.3 Component ablation of the full model

For leave-one-component ablations (Table 3 / `table03_component_ablation`; Fig. `fig_component_ablation_delta_mae.png`), we **re-optimize** (α, κ) after each removal under the same grid and LOO protocol as Stage 3, rather than freezing parameters from the full model. This estimates *recoverable* contribution once remaining parameters adapt—the recommended ablation design when fitting is cheap relative to model complexity—whereas a frozen-parameter removal would measure only inference-time dependence of one fitted solution.

On hard run/trail activities, removing the HRR effort term and re-fitting increased LOO MAE by +14.4 min; removing acute TRIMP increased MAE by +16.4 min. Both channels therefore remain necessary after compensation (frozen-parameter ablations had inflated these deltas to about +25–27 min). Removing GAP entirely remained highly detrimental (+10.5 min). Asymmetric trail GAP soft-ramp scales contributed a smaller improvement when removed and re-optimized (+0.6 min), while altitude and REDI readiness had modest effects in this athlete.

**Note on baselines.** Table 2 reports Stage-3 ladder LOO (hard run/trail M3 = 9.09 min). Table 3’s “full” row is an independently re-optimized LOO reference for ablation deltas (8.24 min on the same cohort under the current stable LOO-cap seed). Absolute levels may differ slightly until a single shared LOO entry point is used (remaining experiment R1); **ΔMAE columns in Table 3 remain the primary ablation evidence**.

### 4.4 Terrain-resolved residuals and trail GAP scales

Prior to asymmetric trail GAP correction, hard-trail segment residuals showed opposing biases on steep terrain: steep climbs were under-sped (bias ≈ +2.3 min) and steep descents over-sped (bias ≈ −3.5 min), whereas flat segments remained well calibrated (MAE ≈ 0.9 min). Soft-ramped scales (`gap_climb_scale` = 0.85, `gap_descent_scale` = 1.60) reduced combined steep-terrain MAE from 3.05 to 1.64 min (−46%) without degrading flat residuals, supporting an athlete-specific correction to treadmill Minetti priors on technical trail grades.

### 4.5 Prospective constant-HRR predictions (E3)

Holding out target races from estimation and simulating planned profiles at constant HRR = 0.88 produced predictions at or faster than observed moving times (Table 5 / `table05_prospective_predictions`): LUT By Night **−4.1 min**; Trail du Grésivaudan **−17.0 min**. Observed mean race HRR was submaximal relative to the hard reference (≈0.80 and ≈0.74), so these forecasts are best read as **upper-bound sustained-effort** scenarios rather than the athlete’s realized pacing. A road marathon hold-out (Rome) was markedly optimistic (**−79.2 min**), indicating **no transfer** of trail-calibrated GAP scales to flat road racing without recalibration.

Finish-time bands from resampling LOO (α, κ) are reported in Table 5 but currently show limited spread (P05 often equals the point prediction)—improving band construction is remaining experiment **R9**. Race- vs segment-objective (α, κ) choice also shifts prospective sign/magnitude (`journal_prediction.md`); nested selection without peeking at hold-outs is **R2**.

### 4.6 Model-implied speed–HRR response

On a synthetic 1 km flat segment under fresh conditions (TRIMP = 0), predicted ground speed increased approximately linearly with HRR until the effort ceiling at HRR_ref = 0.88 (Table 6; Fig. `fig_speed_vs_hrr.png`). Parallel curves at ±10% grade illustrate the interaction of HRR effort with grade cost, providing an interpretable physiological transfer function for coaching “what-if” simulations.

### 4.7 Slight segment rejection

Near-flat immobile rejection is intentionally **slight**. Table 7 reports two designs that must not be conflated: (i) a dedicated **clock-time** A/B on 2364 segments (none / slight / moderate), and (ii) the **paper §7 pipeline** row that pairs slight exclusion with **moving-time** fitting.

In the clock-time A/B (Fig. `fig_segment_rejection_policies.png`), disabling exclusion left all segments in the fit set (mean Stage-3 LOO MAE 10.87 min). Slight exclusion alone removed ~2% of segments but raised mean LOO MAE to 11.85 min; a moderate policy rejected 4.3% and lowered MAE to 10.30 min. Thus slight exclusion **without** moving-time scrubbing is not uniformly helpful.

When slight exclusion is combined with moving-time fitting in the paper §7 pipeline, only **4 segments (0.17%, 84 min)** remain rejected—stationary scrubbing already removes most aid-station dwell from the fit target, and the slight gate catches residual near-flat immobility (headline LOO MAE **9.09 min** on hard run/trail). Illustrative rejects include long dwells on Echappée Belle (Table 7b). Flatness is judged on altitude-over-time so that slow steep climbs are not mistaken for idle flats.

### 4.8 Segment optimisation

Two segment-level optimisations refine the twin beyond activity-level (α, κ) search.

**Trail GAP scales (terrain objective).** Soft-ramped asymmetric scales (`gap_climb_scale = 0.85`, `gap_descent_scale = 1.60`) were calibrated on hard-trail moving-time segment residuals (Table 8; Figs. `fig_segment_gap_optimisation_mae.png`, `fig_segment_gap_optimisation_bias.png`). Combined steep-terrain MAE fell from 3.05 to 1.64 min (−46%), with steep-descent bias corrected from −3.65 to ≈0 and steep-climb bias from +1.69 to ≈0, while flat MAE remained ≤ 0.9 min. Race-level LOO MAE changed only modestly (hard run/trail 9.82 → 9.57 min), indicating that segment optimisation primarily restores physically consistent terrain behaviour rather than chasing finish-time alone.

**Segment versus activity fit objective.** Optimising (α, κ) under a segment residual objective versus an activity finish-time objective yields nearly identical LOO MAE on mixed hard efforts (≈9.1 vs 9.3 min) but diverges on selected race dates (11.8 vs 7.7 min; Table 9; Fig. `fig_segment_vs_race_objective.png`). Segment-objective calibration is therefore preferable for local physiology diagnostics, whereas race-objective calibration better supports prospective finish-time envelopes (cf. LUT / Grésivaudan constant-HRR forecasts).

### 4.9 Full recap of quantitative findings

| Finding | Evidence |
|---------|----------|
| **Physics → HRR+TRIMP is the main gain** | Hard run/trail LOO MAE **30.2 → 9.1 min** (MAPE 26.5% → 6.5%; *R*² 0.982). Hard trail **44.5 → 17.4**; races **36.4 → 11.8**; >20 min **9.4 → 5.1**. |
| **HRR and acute TRIMP are both necessary** | Re-optimized ablations (Table 3): −HRR **+14.4 min**, −TRIMP **+16.4 min** on hard run/trail; −GAP **+10.5 min**; trail GAP scales **+0.6 min**. |
| **REDI / altitude are secondary here** | Ablation ΔMAE ≈ 0–1 min on mixed hard efforts; REDI can even improve some race-date LOO after re-fit. |
| **M1/M2 without HRR can hurt trail-only** | Hard trail M1/M2 MAE rises above M0 until M3 restores accuracy. |
| **Moving time + slight rejection cleans dwell** | §7 pipeline: **4** unfit segments (0.17%); slight alone on clock time can worsen LOO. |
| **Trail GAP scales fix steep physics** | Combined steep MAE **3.05 → 1.64 min (−46%)**; race LOO only ~0.25 min better on mixed hard—terrain consistency, not finish-time chasing. |
| **Fit objective matters on race dates** | Activity vs segment LOO MAE **11.8 vs 7.7 min** on selected races; nearly tied on mixed hard (~9.1 vs 9.3). |
| **Prospective = upper bound when HRR submaximal** | LUT **−4.1 min** (obs HRR ≈0.80); Grésivaudan **−17.0 min** (≈0.74); Rome road **−79.2 min** (no transfer). |
| **Uncertainty reporting is partial** | Bootstrap MAE CIs exist (Table 4); prospective finish bands need stronger resampling (R9). |

**Interpretation.** On this athlete, continuous HRR is the principal incremental predictor beyond a matched physics twin; acute TRIMP is the complementary fatigue channel; asymmetric trail GAP scales restore local climb/descent behaviour; prospective constant-HRR forecasts are coherent as hard-effort envelopes on trail courses but must not be over-claimed as expected finish times or road-ready predictions.

---

## 5. Discussion

### 5.1 Contribution relative to the 2026 digital twin

We retain the digital-twin structure (route physics + athlete state → time) while making **heart-rate reserve the primary instantaneous effort channel** and **acute TRIMP the intra-activity fatigue channel**. The experimental design emphasizes one modeling family, matched LOO reporting against a physics baseline (M0), and prospective hold-outs with uncertainty bands.

### 5.2 Limitations

Evidence remains a **single-athlete case study**. Additional limits: HR artifacts; barometric elevation (no DEM); sparse temperature coverage; constant-HRR abstraction of race tactics; incomplete road transfer (Rome); LOO activity caps on large cohorts; hard-trail κ at the grid floor; Table 2 vs Table 3 absolute-level mismatch pending R1; immature prospective uncertainty bands. Multi-athlete replication is required before population claims.

### 5.3 Practical implication

Within coaching software, the twin supports (i) retrospective effort-normalized race explanation, (ii) prospective finish-time *envelopes* at a target HRR (not point forecasts of realized pacing), and (iii) identification of terrain regimes where grade-cost assumptions fail.

---

## 6. Conclusion

For one recreational/competitive trail runner, a HRR+TRIMP digital twin reduced leave-one-out finish-time error from a physics baseline of ~30 min MAE to ~9 min MAE on mixed hard run/trail activities (MAPE ≈ 6.5%), with parallel gains on broader >20 min and race-date cohorts. Component ablation attributes most of that gain to continuous HRR and acute TRIMP. Soft-ramped trail GAP scales improve steep-terrain residuals, and prospective constant-HRR simulations provide interpretable upper-bound race times. Publication assets are collected in `docs/science/paper/`. Multi-athlete validation remains the primary next step for journal submission.

---

## 7. What remains for robust / publishable results

Full checklist: `docs/science/remaining_experiments.md`.  
Shipped artifacts: `docs/science/section7_implementation_status.md`.

### 7.1 Already shipped in this repository

| Item | Artifact / note |
|------|-----------------|
| Preregistered race IDs & splits | `preregistered_race_protocol.json` |
| Matched physics baseline (M0) vs M3 | Tables 2 / `table_frozen_physics_vs_hrr.csv` |
| Bootstrap MAE + α,κ CIs | Table 4 |
| Re-optimized component ablation | Table 3 (`reoptimize_loo`) |
| Slight rejection + moving-time fit | Tables 7 / 7b |
| Trail GAP scale optimisation | Table 8; `journal_steep.md` |
| Segment vs race objective | Table 9 |
| Prospective LUT / Grésivaudan / Rome | Table 5 |
| HR QC, weather coverage, elevation QA | §7 CSVs (weather sparse; DEM absent) |
| Speed–HRR curve; LOO figures | `docs/science/paper/figures/` |
| Software versions / seeds | `software_versions.json` (LOO cap seed `20260721`) |

### 7.2 Runnable next (robustness — no new athletes required)

| ID | Experiment | Why |
|----|------------|-----|
| **R1** | Reconcile Table 2 vs Table 3 full MAE (shared LOO entry point) | Internal consistency |
| **R2** | Nested CV for race vs segment objective (no hold-out peeking) | Prospective integrity |
| **R3** | Expand preregistered trail prospective set (≥2–3 more races) | Prediction claim |
| **R4** | Non-fitted planned aid-time budget on race profiles | Grésivaudan realism |
| **R5** | Road/flat recalibration **or** explicit road exclusion | Rome −79 min |
| **R6** | Blocked LOO / by-race-date outer folds | Honest uncertainty |
| **R7** | HR QC threshold sensitivity | Strap dropout |
| **R8** | Steep-climb residual / outlier diagnostics | Residual MAE ~1.9 min |
| **R9** | Fix prospective finish-time bands (non-degenerate P05–P95) | Coaching utility |
| **R10** | Refresh ops benchmark under shipped GAP defaults | Align leaderboard |
| **R11** | κ grid boundary sensitivity (hard trail κ = 0.2) | Parameter floor |

### 7.3 Blocked on data / infrastructure

| ID | Item | Blocker |
|----|------|---------|
| **B1** | Multi-athlete (≥8–15) replication | Single-athlete data |
| **B2** | Sex / age / level strata | Single athlete |
| **B3** | DEM-corrected elevation | No DEM pipeline |
| **B4** | Weather / heat model term | Sparse temperature coverage |
| **B5** | Structured aid / nutrition logs | Moving-time proxy only |

### 7.4 Reporting / ethics (submission gates)

Ethics/consent for multi-athlete GPS; data-availability statement; code + configs as supplement; STROBE-like reporting; cite software versions/seeds; complete bibliography (`bibliography_hr_digital_twin.md`).

### 7.5 Optional modeling (not v1 gate)

Grade×HRR interaction (H4); long-term Banister fitness–fatigue; descent-specific eccentric cost; heat/RPE fusion when HR saturates.
---

## 8. Target conferences and journals (**avoid Sensors**)

Focus: **exercise physiology, sports performance, sports analytics**—not wearable hardware.

### 8.1 Journals

| Venue | Fit | Notes |
|-------|-----|------|
| **International Journal of Sports Physiology and Performance (IJSPP)** | Strong | Field performance models, practical metrics |
| **Journal of Sports Sciences** | Strong | Applied modeling; multi-athlete preferred |
| **European Journal of Sport Science (EJSS)** | Strong | ECSS-linked |
| **Journal of Sports Analytics** | Good | Prediction-error / analytics framing |
| **Frontiers in Sports and Active Living** | Good | Performance computing / digital athletes |
| **Scientific Reports** | Possible | If multi-athlete + clear novelty (cf. Boillet 2024) |
| **International Journal of Computer Science in Sport** | Niche | Computational methods |
| **Current Issues in Sport Science (CISS)** | Open | Trail/HR field studies already appear |

**Do not target for this manuscript:** *Sensors* (MDPI), *Biosensors*, IEEE sensor-hardware tracks—wrong audience if the contribution is modeling/prediction, not a new device.

### 8.2 Conferences

| Venue | Fit |
|-------|-----|
| **ECSS** | Strong (abstract → later EJSS) |
| **ACSM Annual Meeting** | Strong |
| **ISBS** | Medium (locomotion/grade emphasis) |
| **MathSport / OR in Sport** | Medium (if pacing formalized) |
| **icSPORTS** | Medium |
| **MLSA @ ECML** | Only if adding strong ML baselines |

### 8.3 Suggested path

1. **Abstract** → ECSS or ACSM (methods + LOO table + one prospective figure).  
2. **Full paper** → IJSPP or *Journal of Sports Sciences* once **≥1 additional athlete** (or explicit single-athlete case study with matched physics baseline).  
3. Optional methods note → *Journal of Sports Analytics* for reproducible error benchmarking.

---

## 9. Bibliography (core)

1. Jaén-Carrillo D, Pattis D. A Physics-Based Digital Twin for Trail Running Race Performance Prediction: A Proof-of-Concept Study. *Sensors*. 2026;26(12):3731. https://doi.org/10.3390/s26123731  
2. Boillet A, et al. Margaria–Morton digital twin for cycling performance. *Sci Rep*. 2024. https://doi.org/10.1038/s41598-024-71772-x  
3. Minetti AE, Moia C, Roi GS, Susta D, Ferretti G. Energy cost of walking and running at extreme uphill and downhill slopes. *J Appl Physiol*. 2002;93:1039–1046.  
4. Banister EW, Calvert TW, Savage MV, Bach T. A systems model of training for athletic performance. *Aust J Sports Med*. 1975;7:57–61.  
5. Banister EW. Modeling elite athletic performance. In: MacDougall JD, Wenger HA, Green HJ, eds. *Physiological Testing of the High-Performance Athlete*. 1991.  
6. Edwards S. *The Heart Rate Monitor Book*. Fleet Feet Press; 1993.  
7. Manzi V, Iellamo F, Impellizzeri F, D’Ottavio S, Castagna C. Relation between individualized training impulses and performance in distance runners. *Med Sci Sports Exerc*. 2009;41(11):2090–2096. https://doi.org/10.1249/MSS.0b013e3181a6a959  
8. Foster C, et al. A new approach to monitoring exercise training. *J Strength Cond Res*. 2001.  
9. Impellizzeri FM, Rampinini E, Coutts AJ, Sassi A, Marcora SM. Use of RPE-based training load in soccer. *Med Sci Sports Exerc*. 2004.  
10. Borresen J, Lambert MI. The quantification of training load, the training response and the effect on performance. *Sports Med*. 2009.  
11. Herman L, Foster C, et al. / IJSPP comparisons of session-RPE vs TRIMP/SHRZ (e.g. *Int J Sports Physiol Perform*. 2008;3:16–xxx).  
12. Daniels J. *Daniels’ Running Formula* (VDOT system). Human Kinetics.  
13. Péronnet F, Thibault G; West JB. Altitude and VO₂max correction classics.  
14. Tobler W. Three presentations on geography / hiking-speed function; Scarf P. related grade–speed analyses.  
15. Bridging lab and field: predicting laboratory thresholds from outdoor trail-running data. *Curr Issues Sport Sci (CISS)*. https://doi.org/10.36950/2026.2ciss025  
16. Genitrini M, et al. Spatiotemporal parameters and kinematics differ between race stages in trail running—a field study. *Front Sports Act Living*. 2024. https://doi.org/10.3389/fspor.2024.1406824  
17. Internal: `docs/science/journal_steep.md`; `journal_prediction.md`; `trail_digital_twin_model_summary.md`.

See `bibliography_hr_digital_twin.md` for extended notes and venue links.

---

## 10. Camera-ready outline

1. Introduction (RQ1–3; link to 2026 twin)  
2. Related work  
3. Methods (data; equations; LOO; prospective protocol)  
4. Results (E1–E4)  
5. Discussion  
6. Conclusion  
7. Data/code/ethics  

**Target length:** ~4–6k words + 4–6 figures (IJSPP / JSS); or ~3k for an analytics methods note.

---

## Appendix A — Parameter roles

| Param | Role | Athlete-specific? |
|-------|------|-------------------|
| \(v_{\mathrm{flat}}\) / VMA | Flat reference speed | Yes |
| \(\alpha\) | HRR gain | Yes (fitted) |
| \(\kappa\) | TRIMP fatigue | Yes (fitted) |
| HR rest / max | HRR denominator | Yes |
| Minetti / GAP base | Grade cost | Shared prior |
| `gap_climb_scale`, `gap_descent_scale` | Trail steep correction | Shared or lightly tuned |
| Altitude factors | Hypoxia | Shared formula |
| Soft-ramp (4%→15%) | Grade blending | Shared |

## Appendix B — Results postcard

- **Model:** HRR + acute TRIMP + soft-ramped trail GAP (0.85 / 1.60); moving-time fit + slight near-flat immobile rejection.  
- **Primary LOO:** hard run/trail **MAE 9.09 min**, MAPE **6.5%**, *R*² **0.982** (physics M0 **30.2 min**).  
- **Ablations (reoptimize):** −HRR **+14.4 min**; −TRIMP **+16.4 min**; −GAP **+10.5 min**.  
- **Steep terrain:** combined MAE **3.05 → 1.64 min (−46%)**.  
- **Prospective (HRR = 0.88):** LUT **−4.1 min**; Grésivaudan **−17.0 min**; Rome **−79.2 min** (road fail).  
- **Still needed for robust claims:** R1–R11 runnable checks; B1 multi-athlete (or explicit case-study framing); DEM/weather/aid logs blocked.  
- **Submit toward:** IJSPP / JSS / EJSS / ECSS — **not Sensors**.  
- **Checklist:** `docs/science/remaining_experiments.md`.