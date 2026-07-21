# Heart-Rate Digital Twin for Trail-Running Performance: Modeling, Cross-Activity Evaluation, and Measured Prediction Errors

**Working draft (v0.4)** — modeling / methods / results focus  
**Status:** single-athlete proof-of-concept; multi-athlete replication planned  
**Companion code:** Running Manager (`services/trail_performance_model.py`, `scripts/predict_race_constant_hrr.py`)  
**Publication assets:** `docs/science/paper/` (figures PNG + tables)  
**Related docs:** `trail_digital_twin_hrr_trimp_paper.md`, `trail_digital_twin_model_summary.md`, `journal_steep.md`, `journal_prediction.md`  
**Annotated bibliography:** `bibliography_hr_digital_twin.md`

---

## Abstract (draft)

Accurate prediction of trail-running finish times remains difficult because grade, altitude, and fatigue interact nonlinearly with athlete physiology. Building on the physics-informed digital-twin framework of Jaén-Carrillo and Pattis (2026), we evaluate a single athlete-specific model that uses continuous heart-rate reserve (HRR) as instantaneous effort and Banister-style TRIMP as intra-activity fatigue, optionally combined with soft-ramped trail-grade cost corrections. Using leave-one-out validation across hard-run, hard-trail, and run/trail (>20 min) cohorts, a physics baseline (M0) yielded substantially larger errors than the full HRR+TRIMP specification (M3): for mixed hard run/trail activities, MAE decreased from 30.2 to 9.1 min (MAPE 26.5% → 6.5%). Component ablation attributes most of this gain to the HRR and acute-TRIMP terms. Soft-ramped trail GAP scales reduced steep-terrain segment MAE by ≈46%. Prospective constant-HRR simulations for held-out races produced finish times at or faster than observed performances, consistent with an upper-bound sustained-effort scenario. Publication figures and tables are collected in `docs/science/paper/`.

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
| `hardRunOrTrailRun` | Mixed hard efforts; primary aggregate LOO |
| `hardTrailRun` | Trail-only races/hard sessions; stress test |
| LUT 30k (`16325125849`), Grésivaudan (`17481444994`) | Prospective constant-HRR hold-outs |

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

On the hard run/trail cohort, M0 yielded MAE = 30.2 min (MAPE = 26.5%). Adding acute TRIMP (M1) reduced MAE to 23.9 min, whereas replacing CTL by REDI (M2) left error essentially unchanged (24.2 min). Introducing continuous HRR effort (M3) produced the dominant improvement: MAE = 9.1 min, MAPE = 6.5%, *R*² = 0.982 (ΔMAE vs M0 = −21.1 min).

The same ordering held on hard trail runs (M0 44.5 → M3 17.4 min) and on selected race dates (M0 36.4 → M3 11.8 min). On the broader run/trail >20 min cohort, absolute errors were smaller at baseline (M0 9.4 min) and M3 further reduced MAE to 5.1 min (MAPE = 8.1%). Agreement plots for M3 LOO predictions are shown in Figs. `fig_pred_vs_actual_*.png` and Bland–Altman analyses in Figs. `fig_bland_altman_*.png`.

For reference, Jaén-Carrillo and Pattis (2026) reported LOO MAE = 18.2 min and MAPE = 11.1% (*n* = 13 races) for a physics-oriented twin without continuous HR effort. Our hard-trail M3 MAE (17.4 min) is of similar absolute magnitude on a different athlete and race set, while mixed hard-run/trail and >20 min cohorts achieve substantially lower absolute error. Direct numerical comparison remains indicative rather than a matched re-analysis of their data.

Bootstrap percentile intervals on M3 LOO folds (Table 4) place hard run/trail MAE at 9.1 min (90% CI 6.6–11.9) with fold-mean α ≈ 0.95 and κ ≈ 0.39.

### 4.3 Component ablation of the full model

For leave-one-component ablations (Table 3 / `table03_component_ablation`; Fig. `fig_component_ablation_delta_mae.png`), we **re-optimize** (α, κ) after each removal under the same grid and LOO protocol as Stage 3, rather than freezing parameters from the full model. This estimates *recoverable* contribution once remaining parameters adapt—the recommended ablation design when fitting is cheap relative to model complexity—whereas a frozen-parameter removal would measure only inference-time dependence of one fitted solution.

On hard run/trail activities, removing the HRR effort term and re-fitting increased LOO MAE by +14.4 min; removing acute TRIMP increased MAE by +16.4 min. Both channels therefore remain necessary after compensation (frozen-parameter ablations had inflated these deltas to about +25–27 min). Removing GAP entirely remained highly detrimental (+10.5 min). Asymmetric trail GAP soft-ramp scales contributed a smaller improvement when removed and re-optimized (+0.6 min), while altitude and REDI readiness had modest effects in this athlete.

### 4.4 Terrain-resolved residuals and trail GAP scales

Prior to asymmetric trail GAP correction, hard-trail segment residuals showed opposing biases on steep terrain: steep climbs were under-sped (bias ≈ +2.3 min) and steep descents over-sped (bias ≈ −3.5 min), whereas flat segments remained well calibrated (MAE ≈ 0.9 min). Soft-ramped scales (`gap_climb_scale` = 0.85, `gap_descent_scale` = 1.60) reduced combined steep-terrain MAE from 3.05 to 1.64 min (−46%) without degrading flat residuals, supporting an athlete-specific correction to treadmill Minetti priors on technical trail grades.

### 4.5 Prospective constant-HRR predictions (E3)

Holding out target races from estimation and simulating planned profiles at constant HRR = 0.88 produced predictions at or faster than observed moving times (Table 5 / `table05_prospective_predictions`): LUT By Night −4.1 min; Trail du Grésivaudan −17.0 min. Resampling LOO (α, κ) pairs yields finish-time bands (P05–P95) that encompass the point prediction and extend several minutes slower. Observed mean race HRR was submaximal relative to the hard reference (≈0.80 and ≈0.74), consistent with predictions that represent an upper-bound sustained-effort scenario rather than the athlete’s realized pacing. A road marathon hold-out (Rome) was markedly optimistic, indicating limited transfer of trail-calibrated GAP scales to flat road racing.

### 4.6 Model-implied speed–HRR response

On a synthetic 1 km flat segment under fresh conditions (TRIMP = 0), predicted ground speed increased approximately linearly with HRR until the effort ceiling at HRR_ref = 0.88 (Table 6; Fig. `fig_speed_vs_hrr.png`). Parallel curves at ±10% grade illustrate the interaction of HRR effort with grade cost, providing an interpretable physiological transfer function for coaching “what-if” simulations.

### 4.7 Slight segment rejection

Near-flat immobile rejection is intentionally **slight**. In the dedicated A/B experiment (Table 7; Fig. `fig_segment_rejection_policies.png`), disabling exclusion left all 2364 segments in the fit set (mean Stage-3 LOO MAE 10.87 min). The slight policy (`speedEq < 3 km·h⁻¹` or stationary share `> 0.40`, with altitude–time flatness) removed on the order of 2% of segments. A moderate policy (`speedEq < 4`, share `> 0.30`) rejected 101 segments (4.3%, ≈32 h of dwell) and lowered mean LOO MAE to 10.30 min.

When slight exclusion is combined with moving-time fitting in the paper §7 pipeline, only **4 segments (0.17%, 84 min)** remain rejected—stationary scrubbing already removes most aid-station dwell from the fit target, and the slight gate catches residual near-flat immobility. Illustrative rejects include long dwells on Echappée Belle and related near-flat stops (Table 7b). Importantly, flatness is judged on altitude-over-time so that slow steep climbs are not mistaken for idle flats.

### 4.8 Segment optimisation

Two segment-level optimisations refine the twin beyond activity-level (α, κ) search.

**Trail GAP scales (terrain objective).** Soft-ramped asymmetric scales (`gap_climb_scale = 0.85`, `gap_descent_scale = 1.60`) were calibrated on hard-trail moving-time segment residuals (Table 8; Figs. `fig_segment_gap_optimisation_mae.png`, `fig_segment_gap_optimisation_bias.png`). Combined steep-terrain MAE fell from 3.05 to 1.64 min (−46%), with steep-descent bias corrected from −3.65 to ≈0 and steep-climb bias from +1.69 to ≈0, while flat MAE remained ≤ 0.9 min. Race-level LOO MAE changed only modestly (hard run/trail 9.82 → 9.57 min), indicating that segment optimisation primarily restores physically consistent terrain behaviour rather than chasing finish-time alone.

**Segment versus activity fit objective.** Optimising (α, κ) under a segment residual objective versus an activity finish-time objective yields nearly identical LOO MAE on mixed hard efforts (≈9.1 vs 9.3 min) but diverges on selected race dates (11.8 vs 7.7 min; Table 9; Fig. `fig_segment_vs_race_objective.png`). Segment-objective calibration is therefore preferable for local physiology diagnostics, whereas race-objective calibration better supports prospective finish-time envelopes (cf. LUT / Grésivaudan constant-HRR forecasts).

### 4.9 Summary of quantitative findings

1. Continuous HRR is the principal incremental predictor beyond the physics baseline on this athlete.  
2. Acute TRIMP improves mixed hard-effort cohorts; REDI readiness yields limited additional LOO gain here.  
3. Slight near-flat immobile rejection plus moving-time fitting removes aid-station dwell from the physiology fit with negligible segment loss in the paper pipeline.  
4. Asymmetric trail GAP scales correct steep climb/descent bias at the segment level.  
5. Segment versus race objectives agree on mixed hard efforts but can diverge on race-date cohorts.  
6. Prospective constant-HRR forecasts are coherent upper bounds when race HRR is submaximal; road transfer remains an open limitation.

---

## 5. Discussion

### 5.1 Contribution relative to the 2026 digital twin

We retain the digital-twin structure (route physics + athlete state → time) while making **heart-rate reserve the primary instantaneous effort channel** and **acute TRIMP the intra-activity fatigue channel**. The experimental design emphasizes one modeling family, matched LOO reporting against a physics baseline (M0), and prospective hold-outs with uncertainty bands.

### 5.2 Limitations

Evidence remains a single-athlete case study. Additional limits include HR artifacts, barometric elevation noise (no DEM), sparse temperature coverage, constant-HRR abstraction of race tactics, and incomplete transfer to flat road racing. Multi-athlete replication is required before population claims.

### 5.3 Practical implication

Within coaching software, the twin supports (i) retrospective effort-normalized race explanation, (ii) prospective finish-time envelopes at a target HRR, and (iii) identification of terrain regimes where grade-cost assumptions fail.

---

## 6. Conclusion

For one recreational/competitive trail runner, a HRR+TRIMP digital twin reduced leave-one-out finish-time error from a physics baseline of ~30 min MAE to ~9 min MAE on mixed hard run/trail activities (MAPE ≈ 6.5%), with parallel gains on broader >20 min and race-date cohorts. Component ablation attributes most of that gain to continuous HRR and acute TRIMP. Soft-ramped trail GAP scales improve steep-terrain residuals, and prospective constant-HRR simulations provide interpretable upper-bound race times. Publication assets are collected in `docs/science/paper/`. Multi-athlete validation remains the primary next step for journal submission.

---

## 7. Missing elements for a publishable paper in this domain

### 7.1 Scientific / experimental

| Missing item | Why it matters | Minimal fix | Status (this repo) |
|--------------|----------------|-------------|--------------------|
| **≥8–15 athletes** (or explicit case-study framing) | External validity | Same pipeline; per-athlete + pooled MAE | Blocked (single athlete) |
| **Pre-registered race set & splits** | Avoid selective reporting | Freeze race IDs, dates, inclusion rules | Done → `preregistered_race_protocol.json` |
| **Nested CV / repeated LOO** | Honest uncertainty on α,κ | Outer LOO + inner grid; bootstrap CIs | Done → bootstrap MAE + α/κ tables |
| **Matched physics-only baseline on same data** | Fair vs 2026 twin | Re-implement no-HR twin on identical races | Done → `table_frozen_physics_vs_hrr.csv` |
| **Broader prospective set** | Strengthen “prediction” claim | Expand E3 beyond two races | Done (LUT, Grésivaudan, Rome) |
| **Uncertainty bands on finish time** | Coaching utility | Bootstrap / posterior on (α,κ) | Done → prospective bands CSV |
| **Sex, level, age strata** | Population heterogeneity | Recruit diversity; subgroup errors | Blocked |
| **Weather / heat / mud** | Environmental variance | Log conditions; optional multipliers | Coverage logged (sparse); no fitted term |
| **Structured aid / nutrition logs** | Stationary ≠ fatigue | Beyond moving-time scrubbing | Blocked (moving-time proxy only) |
| **HR QC (% valid samples)** | Strap dropouts bias HRR | Artifact filters; coverage threshold | Done → `table_hr_qc.csv` |
| **DEM / barometric elevation QA** | Grade noise | Prefer DEM-corrected elevation | Barometric QA only (no DEM) |
| **Reoptimized ablation table** | Paper clarity | Re-fit (α, κ) after ±HRR / ±TRIMP / ±GAP; LOO | Done |
| **Segment vs race objective analysis** | Why objectives disagree | Short bias–variance note | Done |
| **Cohort run/trail >20 min** | Broader activity mix | `runTrailOver20Min` | Done (n=208; LOO capped at 80) |
| **Speed vs HRR (1 km)** | Effort–speed response curve | `speed_vs_hrr_1km.csv` | Done |

See `docs/science/section7_implementation_status.md` and  
`data/exp_perf_predictions/trail_digital_twin_paper_section7/`.

### 7.2 Reporting / ethics / reproducibility

| Missing item | Notes |
|--------------|-------|
| Ethics / consent / anonymization | Required for multi-athlete GPS |
| Data availability | Synthetic profiles + params if raw GPS restricted |
| Code availability | Pipeline configs + scripts as supplement |
| Reporting guideline | STROBE-like for observational sports data |
| Software versions / seeds | Python, libs, grid seeds |
| Figures | Pred vs actual scatter; Bland–Altman; terrain boxplots |
| Zotero-complete bibliography | Replace any remaining placeholders (§9) |

### 7.3 Optional modeling extensions (not required for v1)

Long-term Banister fitness–fatigue; descent-specific eccentric cost; heat/RPE fusion when HR saturates; multi-objective pacing (not only constant HRR).

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

- **Model:** HRR + TRIMP (+ trail GAP soft-ramp).  
- **Tests:** LOO + 2 prospective constant-HRR races.  
- **Headline:** ~10 min MAE / ~8% MAPE mixed LOO; steep segment MAE −46%; LUT −4 min; Grésivaudan −17 min (pred faster).  
- **Missing:** more athletes, matched physics baseline, CIs, ethics, figures.  
- **Submit toward:** IJSPP / JSS / EJSS / ECSS — **not Sensors**.
