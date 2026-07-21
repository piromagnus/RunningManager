# Heart-Rate Digital Twin for Trail-Running Performance: Modeling, Cross-Activity Evaluation, and Measured Prediction Errors

**Working draft (v0.3)** — modeling / methods / results focus  
**Status:** single-athlete proof-of-concept; multi-athlete replication planned  
**Companion code:** Running Manager (`services/trail_performance_model.py`, `scripts/predict_race_constant_hrr.py`)  
**Related docs:** `trail_digital_twin_hrr_trimp_paper.md`, `trail_digital_twin_model_summary.md`, `journal_steep.md`, `journal_prediction.md`  
**Annotated bibliography:** `bibliography_hr_digital_twin.md`

---

## Abstract (draft)

Accurate prediction of trail-running finish times remains difficult because grade, altitude, and fatigue interact nonlinearly with athlete physiology. Building on the physics-informed digital-twin framework of Jaén-Carrillo and Pattis (2026), we evaluate a **single** athlete-specific model that uses continuous **heart-rate reserve (HRR)** as instantaneous effort and **Banister-style TRIMP** as intra-activity fatigue, optionally combined with soft-ramped trail-grade cost corrections. The model is fitted and tested on a diverse corpus of GPS+HR activities (road, trail, races) for one recreational/competitive trail runner. Leave-one-out (LOO) evaluation and prospective constant-HRR race predictions report **MAE, MAPE, bias, and R²**. On a mixed hard-run/trail cohort, Stage-3 LOO MAE is about **9.6 min** (MAPE ≈ **7.3%**); on hard-trail races alone, MAE remains higher (~17–19 min) but MAPE ≈ **9%**. Soft-ramped trail GAP scales cut combined steep-terrain segment MAE by ≈46% (3.05 → 1.64 min) without hurting flat residuals. Prospective predictions for two held-out races (LUT 30k, Trail du Grésivaudan) at constant hard HRR are faster than realized performances (−4.1 and −17.0 min), consistent with imperfect race-day pacing. We list missing elements for a multi-athlete paper and recommend sports-performance venues (**not** sensor-hardware journals).

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
- Exclude near-flat **immobile** segments using altitude-over-time flatness (`|Δelev|/hour`) plus low moving speed / high stationary share—not grade-from-distance alone.  
- Fit on **moving time** (`actualTime − stationaryTime`) so aid stops do not inflate physiology residuals; report full clock time separately when needed.

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
| **E2** | Segment residuals by terrain class | MAE, bias (min / km/h) |
| **E3** | Prospective constant-HRR (LUT, Grésivaudan) | Pred vs actual moving time; Δ min |
| **E4** | Ablations: ± moving-time fit; ± trail GAP soft-ramp; segment vs race objective | ΔMAE |

**“Diverse datasets”** in this draft = diverse **activities and terrains for one athlete**. Multi-athlete extension is planned (§7).

### 3.4 Implementation

Python/pandas CSV storage; YAML configs (`configs/trail_digital_twin_*.yaml`); grid search / LOO for \((\alpha,\kappa)\); scripts for steep GAP grid and constant-HRR prediction.

---

## 4. Results

> Numbers below are from repository experiment journals (2026-07). Freeze exports and regenerate tables before journal submission.

### 4.1 LOO race / activity performance (E1)

**Aggregate Stage-3 LOO (hyperparameter-refined high-reference family):**

| Metric | Value |
|--------|------:|
| Mean Stage-3 LOO MAE | **10.85 min** |
| Mean MAPE | **7.91%** |
| Mean R² | **0.982** |

**Moving-time fit + trail GAP soft-ramp (activity LOO):**

| Cohort | MAE (min) | MAPE (%) |
|--------|----------:|---------:|
| `hardRunOrTrailRun` | **9.57** | **7.31** |
| `hardTrailRun` | ~17.7 | ~8.7–9.0 |

**Reference (Jaén-Carrillo & Pattis 2026, LOO n=13 races):** MAE 18.2 min, MAPE 11.1%, R² 0.864.

**Interpretation.** On mixed hard efforts, HRR+TRIMP LOO errors are substantially lower than the published physics twin’s race-only LOO. On **hard-trail-only** races, absolute MAE remains comparable in magnitude to that baseline (~18 min), while relative error (MAPE) is somewhat lower. Athlete, race set, and preprocessing differ—comparison is **indicative**, not a formal head-to-head on identical data.

### 4.2 Terrain-level errors (E2)

**Baseline (moving-time fit, before steep GAP correction), hardTrailRun segments:**

| Terrain | MAE (min) | Bias (min) | Pattern |
|---------|----------:|-----------:|---------|
| Flat | 0.88 | −0.06 | healthy |
| Steep climb | 2.68 | +2.26 | model **too slow** |
| Steep descent | 3.52 | −3.51 | model **too fast** |

**After soft-ramped trail GAP scales (0.85 climb / 1.60 descent):**

| Terrain | MAE before → after | Bias after |
|---------|--------------------:|-----------:|
| Steep descent | 3.65 → **1.36** | ≈ 0 |
| Steep climb | 2.44 → **1.92** | ≈ 0 |
| Flat | 0.91 → **0.86** | ≈ 0 |

Steep combined MAE: **3.05 → 1.64** (−46%). Race-level `hardRunOrTrailRun` MAE: **9.82 → 9.57** min.

### 4.3 Prospective constant-HRR predictions (E3)

Fit **without** the target race; profile = race_pacing D+ + GPX altitudes; constant hard HRR = **0.88**.

| Race | Actual moving | Pred (race-obj α=0.95, κ=0.40) | Δ |
|------|---------------|--------------------------------:|---:|
| LUT 30k | 2h57m18s | **2h53m10s** | **−4.1 min** |
| Trail du Grésivaudan | 3h32m59s | **3h16m00s** | **−17.0 min** |

Segment-objective calibration (α=0.90, κ=0.30): LUT **+3.8 min** (too slow); Grésivaudan **−8.6 min**.

**Notes.** GPX-only D+ understates climb (rejected as primary profile). Higher constant HRR than 0.88 does not speed the current effort law (`hrr_max_factor=1.0`); it only adds TRIMP. Observed mean race HRR (eval only): LUT ≈0.80, Grésivaudan ≈0.74—below the hard reference, partly explaining why constant-0.88 predictions are faster.

### 4.4 What the errors teach

1. **HRR carries pacing information** that pure grade×distance misses.  
2. **Steep trail locomotion** needs asymmetric climb/descent GAP scales; treadmill Minetti priors mis-price trail descents especially.  
3. **Stationary time** must be stripped for physiology fitting.  
4. **Constant-HRR prospective mode** gives a clean “best hard effort” bound, not a full race strategy (nutrition, heat, pack, tactics).

---

## 5. Discussion

### 5.1 Contribution relative to the 2026 digital twin

We keep the digital-twin philosophy (route physics + athlete state → time) but make **HR the primary instantaneous effort channel** and **TRIMP the fatigue channel**, then quantify errors on LOO and prospective tasks. The modeling story is intentionally **one family**.

### 5.2 Limitations (brief; expanded in §7)

Single athlete; HR artifacts; GPX altitude noise; constant-HRR ignores race tactics; Banister/TRIMP parameters partly fixed; no nested CV yet; hard-trail absolute MAE still large on long/high-TRIMP races.

### 5.3 Practical implication

For coaching software: (i) post-hoc race explanation from HR; (ii) “what-if” finish times at a target HRR on a course profile; (iii) detection of terrain regimes where the twin is unreliable before correction.

---

## 6. Conclusion

A single HRR+TRIMP trail digital twin, evaluated with LOO and prospective constant-HRR tests on diverse activities of one athlete, achieves mixed-cohort LOO errors near **~10 min MAE / ~8% MAPE**, improves steep-terrain segment residuals by ~46% with soft-ramped GAP scales, and produces held-out race predictions that are faster than actual finishes under a hard sustained HRR. The next scientific step is **multi-athlete replication with the same protocol and error tables** (§7), aimed at sports-performance venues rather than sensor journals.

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
| **Frozen ablation table** | Paper clarity | Physics / +HRR / +TRIMP / +GAP scales | Done |
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
