# Bibliography — HR / digital-twin trail performance prediction

Supporting material for `trail_digital_twin_hr_performance_paper_draft.md`.  
Focus: modeling, LOO/error reporting, HR–TRIMP effort, trail grade cost.  
**Out of scope for venue targeting:** sensor-hardware journals (*Sensors*, *Biosensors*, IEEE sensors tracks).

---

## A. Core digital-twin / trail prediction

1. **Jaén-Carrillo D, Pattis D.** A Physics-Based Digital Twin for Trail Running Race Performance Prediction: A Proof-of-Concept Study. *Sensors*. 2026;26(12):3731.  
   https://doi.org/10.3390/s26123731  
   - LOO n=13: MAE 18.2 min, MAPE 11.1%, R² 0.864; bias +2.0 min.  
   - GAP/Minetti + altitude + Banister TRIMP + pacing decay; VT2 fraction α, decay μ.  
   - **Use as structural baseline; do not submit our HR paper to Sensors.**

2. **Boillet A, et al.** (Margaria–Morton / cycling digital twin). *Scientific Reports*. 2024.  
   https://doi.org/10.1038/s41598-024-71772-x  
   - Precedent for mechanistic digital twin + performance outside pure ML.

3. **Genitrini M, et al.** Spatiotemporal parameters and kinematics differ between race stages in trail running—a field study. *Frontiers in Sports and Active Living*. 2024.  
   https://doi.org/10.3389/fspor.2024.1406824  
   - UH/DH kinematics and fatigue across race stages; motivates asymmetric climb/descent costs.

---

## B. Grade cost, GAP, altitude, flat performance

4. **Minetti AE, Moia C, Roi GS, Susta D, Ferretti G.** Energy cost of walking and running at extreme uphill and downhill slopes. *Journal of Applied Physiology*. 2002;93:1039–1046.

5. **Tobler W.** Three presentations on geography (hiking function). Related: Scarf P. analyses of grade vs speed.

6. **Daniels J.** *Daniels’ Running Formula* (VDOT). Human Kinetics. (flat performance reference family)

7. **Péronnet F, Thibault G.** Mathematical analysis of running performance and world running records. *J Appl Physiol*. 1989.  
8. **West JB.** *High Life* / altitude physiology classics — VO₂max decline with altitude.

9. Operational **grade-adjusted pace (GAP)** practice (Strava community / coaching software) — cite as operational definition when formal peer-reviewed GAP paper is thin.

---

## C. HR, HRR, TRIMP, training load

10. **Banister EW, Calvert TW, Savage MV, Bach T.** A systems model of training for athletic performance. *Australian Journal of Sports Medicine*. 1975;7:57–61.

11. **Banister EW.** Modeling elite athletic performance. In: MacDougall et al., *Physiological Testing of the High-Performance Athlete*. 1991.

12. **Edwards S.** *The Heart Rate Monitor Book*. Fleet Feet Press; 1993. (zone TRIMP)

13. **Manzi V, Iellamo F, Impellizzeri F, D’Ottavio S, Castagna C.** Relation between individualized training impulses and performance in distance runners. *Medicine & Science in Sports & Exercise*. 2009;41(11):2090–2096.  
    https://doi.org/10.1249/MSS.0b013e3181a6a959

14. **Lucia A, Hoyos J, Santalla A, Earnest C, Chicharro JL.** Tour de France versus Vuelta a España: which is harder? *Med Sci Sports Exerc*. 2003. (Lucia TRIMP zones — cite appropriate Lucia TRIMP source)

15. **Foster C, et al.** A new approach to monitoring exercise training. *Journal of Strength and Conditioning Research*. 2001. (session-RPE)

16. **Impellizzeri FM, Rampinini E, Coutts AJ, Sassi A, Marcora SM.** Use of RPE-based training load in soccer. *Med Sci Sports Exerc*. 2004.

17. **Borresen J, Lambert MI.** The quantification of training load, the training response and the effect on performance. *Sports Medicine*. 2009.

18. **Herman L, Foster C, Maher MA, Mikat RP, Porcari JP.** Validity and reliability of the session RPE method for monitoring exercise training intensity. Related IJSPP load-comparison papers, e.g. session-RPE vs TRIMP/SHRZ in *Int J Sports Physiol Perform*. 2008;3:16–xxx.

19. **Wallace LK, Slattery KM, Coutts AJ.** The ecological validity and application of the session-RPE method for quantifying training loads in swimming. *J Strength Cond Res*. 2009. (session-RPE ↔ Banister/Edwards TRIMP)

---

## D. Trail-specific HR / field diagnostics

20. **Bridging Lab and Field: Predicting Laboratory Thresholds from Outdoor Trail-Running Data for Field-Based Performance Diagnostics.** *Current Issues in Sport Science (CISS)*.  
    https://doi.org/10.36950/2026.2ciss025  
    - Outdoor trail sectors HR ↔ lab anaerobic threshold; LOOCV RMSE ≈ 4.3 bpm.

21. **IncremenTrail / outdoor uphill testing** papers (e.g. *Sports* / *Human Kinetics* trail incremental tests) — sport-specific aerobic testing for trail runners.

---

## E. Pacing, ultra, durability (contextual)

22. Ultra-endurance pacing and durability reviews (select 2–3 recent *Sports Med* / *IJSPP* reviews on pacing decay and fatigue resistance).  
23. Ehrström / Björklund-type short trail split analyses (UH time dominates finish)—as cited inside Genitrini 2024.

---

## F. Internal technical sources (not peer-reviewed)

24. `docs/science/trail_digital_twin_model_summary.md` — athlete vs general parameters.  
25. `docs/science/journal_steep.md` — steep GAP soft-ramp experiments.  
26. `docs/science/journal_prediction.md` — prospective LUT / Grésivaudan constant-HRR.  
27. `docs/science/trail_digital_twin_benchmark_hyperparameter_report.md` — LOO MAE/MAPE/R² tables.

---

## G. Venue shortlist (submission targets)

| Type | Venue | Why |
|------|-------|-----|
| Journal | IJSPP | Performance modeling, field metrics |
| Journal | Journal of Sports Sciences | Applied methods + multi-athlete |
| Journal | European Journal of Sport Science | ECSS pathway |
| Journal | Journal of Sports Analytics | Error tables / prediction framing |
| Journal | Frontiers in Sports and Active Living | Digital / computational sport |
| Journal | Scientific Reports | Multi-athlete mechanistic twin |
| Journal | CISS | Open; trail HR already present |
| Conference | ECSS | Abstract first |
| Conference | ACSM | Applied physiology audience |
| Conference | MathSport / icSPORTS | Optional methods angle |

**Avoid:** MDPI *Sensors*, *Biosensors*, IEEE Sensors / wearable-hardware tracks (wrong reader for a modeling paper).

---

## H. Zotero checklist before submission

- [ ] Replace any “et al.” / “related” stubs with full records  
- [ ] Confirm Boillet 2024 exact title/authors from DOI landing page  
- [ ] Add page numbers for Banister 1975 / Foster 2001  
- [ ] Pick one canonical altitude formula citation (Péronnet–Thibault or West)  
- [ ] Add 1–2 ultra pacing reviews with DOIs  
- [ ] Decide whether to cite Jaén-Carrillo 2026 only as related work (yes) while submitting elsewhere (yes)
