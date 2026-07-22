# Validation plan for the intensity-conditioned trail-running digital twin

**Status:** analysis plan, not completed evidence  
**Date:** 2026-07-22  
**Scope:** longitudinal N-of-1 study using the currently available athlete archive  
**Narrative source:** `docs/science/ideas.md` as updated in commit `ff7d61f`; current remote
science state `d6130f9`

## 1. Decision and recommended claim

The paper should not attempt to prove one large claim at once. It should build a four-level
chain of evidence and stop at the highest level supported by leakage-safe tests.

### Primary claim to test

> For one trail runner, conditioning a physics-based, segment-level course model on a
> pre-specified mean heart-rate reserve (HRR) improves out-of-activity finish-time estimation
> over an otherwise matched fixed-intensity model, within the duration and terrain range
> represented in the athlete's Strava history.

This wording separates three quantities that the current draft sometimes conflates:

- **capacity:** the athlete-specific speed scale after course physics (`alpha`, or preferably
  the identifiable product `v_anchor * alpha`);
- **chosen intensity:** an HRR supplied before simulation;
- **durability:** the reduction in speed as a causal, sequential load state accumulates.

### Secondary claims, in descending order of evidential strength

1. **History-only intensity selection:** previous activities can estimate an archive-supported
   target HRR for a new event without using that event's HR or time.
2. **Longitudinal state:** a slowly varying effective capacity parameter improves prediction of
   future activities over one static parameter.
3. **Within-race updating:** observations from completed segments improve the remaining-time
   forecast beyond simple terrain-weighted pace extrapolation.
4. **Reusable implementation:** the method can be run from a Strava archive without a new
   laboratory visit.

The following formulations must remain out of scope unless their specific gates pass:

- Do not call observed HRR **intended intensity**. It is realised cardiac response. Use
  *prescribed HRR* only when the value is fixed before the simulation.
- Do not call the fastest simulation a **biological maximum** or **best possible time**. Call it
  the *fastest archive-supported scenario*.
- Do not call `alpha` a direct biological fitness marker while VMA is fixed: VMA and `alpha`
  are multiplicatively confounded.
- Do not claim that the method is valid for **any athlete** from an N-of-1 analysis. The
  software may be Strava-compatible even though scientific generalisation is untested.

## 2. Evidence and data currently available

### 2.1 Archive inventory

As of the current data freeze, the repository contains:

| Asset | Available support | Consequence for the study |
|---|---:|---|
| Run/trail activities | 253 from 2024-04-27 to 2026-06-05 | Dense repeated N-of-1 archive over about 26 months |
| Run/trail distance | 2,232.5 km | Broad training exposure, but not population replication |
| Activities marked with time series and average HR | 253/253 | HR-conditioned analyses are feasible |
| Versioned anonymised activity features | 248 rows | Sufficient for activity-level and duration-frontier analyses |
| Versioned anonymised segment features | 10,454 rows | Sufficient for segment diagnostics; activities remain the inferential units |
| Hard trail activities | 45 | Primary target-domain cohort |
| Hard run or trail activities | 103 | Secondary, more heterogeneous cohort |
| Run/trail activities longer than 20 min | 208 | Sensitivity cohort across a wider intensity range |
| Selected high-effort dates | 18 | Candidate rolling-origin benchmarks, not all verified races |
| Strict trail hold-outs with pre-race-style course profiles | 2 | LUT and Grésivaudan |
| Trail replays using executed activity geometry | 2 | Échappée and Passerelles are pseudo-prospective, not strict pre-race tests |
| Planned sessions / linked activities / RPE records | 49 / 25 / 9 | Too sparse to use RPE or planned target as the primary intensity label |
| Linked planned races | 3 | Useful as a small external check only |

Duration support is uneven: 71 activities last at least 1 h, 31 at least 2 h, 16 at least
3 h, 10 at least 4 h, 6 at least 6 h, 5 at least 8 h, and only 1 at least 10 h. Only seven
activities have mean HRR between 0.80 and 0.90, and none has mean HRR at or above 0.90.
Consequently, an HRR-duration frontier is reasonably testable in the common range but is
weakly identified in the ultra-duration and very-high-HRR tails.

The activities are marked as time-series-bearing, and derived segment assets are versioned.
Analyses requiring raw HR lag, route overlap, or checkpoint replay must additionally verify
that the local raw time-series cache is available; the current checkout does not contain that
cache.

### 2.2 Current results that motivate, but do not prove, the claim

- On hard run/trail activities, the current activity-level LOO ladder improves from 25.65 min
  MAE for the physics baseline to 8.25 min for the full observed-HRR model (MAPE 7.33%).
- On the target-domain hard-trail cohort, the corresponding full-model MAE is 17.40 min
  (MAPE 9.67%) with a -9.82 min bias.
- Re-optimised ablations on hard run/trail increase MAE by 14.35 min without HRR and by
  16.41 min without acute TRIMP. Replacing TRIMP with linear progress raises MAE by 8.36 min.
- On four trail hold-outs, supplying the observed activity-mean HRR still leaves about
  16.6 min absolute error. Therefore the current segment-HRR result cannot be assumed to
  transfer directly to a single prescribed mean HRR.
- History-selected duration-feasible HRR is unstable in the ultra tail: the Échappée estimate
  is about 73 min too fast. The existing H13b/H20 adaptive recipes were explored on the same
  small hold-out set and must not be treated as confirmatory results.

These results support **retrospective HR-informed reconstruction**. They do not yet validate
the main prescribed-intensity claim, automatic target-HRR selection, or temporal evolution of
the fitted parameters.

## 3. Main validity threats to address

1. **Target leakage.** Full segment HRR and preceding observed segment durations are taken
   from the target activity in the reconstruction model.
2. **Endogeneity.** HRR is a response to speed, grade, heat, hydration, drift, and fatigue; it is
   not an externally randomised dose.
3. **Coupled predictors.** HRR and Banister TRIMP come from the same HR signal, while TRIMP also
   uses duration. Their separate ablation gains do not by themselves establish two independent
   physiological mechanisms.
4. **Pseudo-replication.** Thousands of segments do not create thousands of independent
   observations. The outer unit is an activity, or an event date when two activities share a day.
5. **Model-selection leakage.** Terrain scales, HRR rules, fatigue form, grids, and H13b/H20
   choices have been examined using the available outcomes. They need nesting or explicit
   exploratory labels.
6. **Sparse frontier tails.** The archive has little support beyond 4-6 h and above mean
   HRR 0.80.
7. **Parameter non-identifiability.** `vma_flat_kmh * alpha` is the effective capacity scale;
   `alpha` alone is not identifiable when VMA is fixed. `alpha` and `kappa` may also compensate
   for each other.
8. **Outcome definition.** Moving time omits aid and stopping behaviour; elapsed time mixes
   locomotor performance with race logistics. Both should be reported with different labels.
9. **Course-profile leakage.** Executed GPS geometry is not a true pre-race input when the
   official/planned course was unavailable.
10. **Selection bias.** The 18 `selected_race_dates` include high-effort sessions as well as
    races. Their labels must be frozen independently of model error.

## 4. Frozen evaluation protocol

Before adding or comparing models:

1. Freeze the input file hashes, analysis date, software versions, HR-rest/HR-max assumptions,
   and the exact activity eligibility table.
2. Classify every candidate event as `race`, `benchmark workout`, `training`, or `unknown`
   without looking at prediction errors. Preserve both the label source and date.
3. Use **hard trail activities (`n=45`) as the primary scientific cohort**. Use hard run/trail
   and >20 min cohorts as secondary sensitivity analyses.
4. Evaluate every eligible activity rather than a favourable 80-fold sample. If runtime forces
   a cap, pre-specify the sample and repeat it across at least ten fixed seeds.
5. Use nested, chronological validation for every prospective claim:
   - outer fold: one future activity/event;
   - training set: only earlier activities;
   - inner folds: choose all hyperparameters and structural variants using earlier data only.
6. Use activity-level paired errors. Bootstrap whole activities (or whole dates), never
   individual segments, for confidence intervals.
7. Make moving-time prediction primary for locomotor modelling and elapsed-time prediction a
   separately reported operational outcome. Do not add estimated aid minutes to the primary
   result.
8. Use symmetric loss for scientific forecasting. An optimistic prediction is not automatically
   acceptable. If a coaching *fast envelope* is desired, report it separately as a calibrated
   lower finish-time quantile.

## 5. Experiments

### E0 - Data quality and support map

**Question:** Where can the model be evaluated without extrapolation?

Produce an activity-level audit covering HR coverage, HR-rest/max sensitivity, device/sensor
provenance where recoverable, duration, mean HRR, distance, ascent per kilometre, altitude,
terrain composition, moving/elapsed gap, and date. Define an in-domain hull using duration and
course characteristics before evaluating forecasts.

Required outputs:

- counts and distributions by duration and HRR;
- strict versus pseudo-prospective course-profile status;
- support flags: `interpolation`, `weak support`, `extrapolation`;
- sensitivity to HR rest +/-5 bpm and HR max +/-5 bpm;
- confirmation that `hrValidShare=1.0` means non-missing samples, not artifact-free HR.

### E1 - Test the premise that fixed intensity is inadequate

**Question:** Does realised intensity vary materially between activities after duration and
course demands are accounted for?

1. Regress activity-mean HRR on log duration, terrain composition, ascent per kilometre,
   altitude, date/season, and independently frozen activity purpose.
2. Cluster repeated routes using GPS/polyline overlap. If raw geometry is unavailable, use a
   pre-declared coarse match on distance, ascent, altitude, and terrain-family shares.
3. Within matched route/duration groups, compare HRR and physics-baseline residuals between
   race/benchmark and training efforts.
4. Test whether M0 residuals retain an association with activity-mean HRR after course and
   duration adjustment.

This experiment supports the **motivation** for an intensity moderator. It cannot prove that
changing HRR would causally change time.

### E2 - Primary test: prediction conditional on one prescribed mean HRR

**Question:** If a mean HRR is supplied as an external scenario input, does it improve
out-of-activity time estimation over fixed-intensity physics?

For every outer activity, exclude all of its segments from fitting. Supply its observed
activity-mean HRR only as an **oracle proxy for a pre-specified target**, never its segment HR,
segment times, or final duration. Accumulate fatigue using predicted segment durations and the
prescribed HRR.

Compare matched models with the same course physics and fitting budget:

| ID | Model | Information from target activity |
|---|---|---|
| B0 | Prior fixed-%VT2 physics twin | Course profile only |
| B1 | Fixed HRR reference | Course profile only |
| B2 | Constant prescribed mean HRR + time/progress fatigue | Course + one HRR scalar |
| B3 | Constant prescribed mean HRR + causal predicted-TRIMP state | Course + one HRR scalar |
| B4 | Past-data terrain-family HRR allocation, re-centred to prescribed mean | Course + one HRR scalar |
| U1 | Full observed segment HRR reconstruction | Retrospective upper bound; not a forecast |

Report activity-level MAE, MAPE, median absolute percentage error, signed bias, error-versus-
duration slope, and paired differences versus B0/B1. Show hard trail first. U1 must be visually
and verbally separated from B0-B4.

**Primary success gate:** on hard trail, B3 must beat the best fixed-intensity baseline by at
least 10% relative MAPE, with the activity-block-bootstrap 95% interval for the paired error
difference excluding zero. It must also reduce residual dependence on mean HRR without creating
a worse long-duration bias.

If B3 fails but U1 wins, the defensible story is HR-informed retrospective reconstruction, not
prescribed-intensity forecasting.

### E3 - Validate the time-HRR curve with repeated-course contrasts

**Question:** Does the model predict the direction and magnitude of time differences when the
same or a closely matched route is completed at different intensities?

For each independently defined route cluster with at least two activities:

1. fit the model without the entire route cluster;
2. simulate every activity using its activity-mean HRR;
3. compare observed and predicted within-route time differences;
4. compute pairwise ordering accuracy, concordance, and difference MAE.

This is the closest available observational test of the counterfactual curve. A useful result
requires concordance above chance with a cluster-bootstrap interval and no domination by one
route. If too few repeated routes exist, the curve must be described as model-implied rather
than empirically validated.

### E4 - History-only selection of sustainable HRR

**Question:** Can earlier Strava activities choose the HRR scenario for a later event?

Run rolling-origin forecasts on independently labelled race and benchmark efforts. At each
date, estimate the HRR-duration relationship from earlier activities only and solve the
time-HRR fixed point on the held-out course.

Compare:

- fixed `hrr_reference`;
- median HRR of previous hard efforts;
- HRR of the nearest previous activity by duration and course difficulty;
- empirical duration-bin quantile;
- monotone quantile regression or isotonic frontier on log duration;
- the existing maximum-window power law;
- H13b/H20 only as exploratory candidates inside the nested inner loop.

Do not select the frontier by the maximum observation alone. Bootstrap the frontier by
activity and expose uncertainty. Restrict the confirmatory claim to duration/course regions
with adequate historical support; mark forecasts beyond 4-6 h as weak-support or exploratory
unless the bootstrap frontier is stable.

**Success gate:** the history-only rule must improve paired event-level error over the best
simple baseline in rolling-origin evaluation, with a confidence interval that excludes no
gain, and must not depend on Échappée or another single event. Report symmetric MAE/MAPE first.

For a separate fastest-supported scenario, estimate a lower finish-time quantile and test its
empirical coverage. Do not reward arbitrary optimism.

### E5 - Causal fatigue-state and leakage ablation

**Question:** Is TRIMP adding information beyond HRR and elapsed progress when used in a way
available at prediction time?

Under the same nested outer folds, compare:

1. no within-activity fatigue;
2. linear route progress;
3. accumulated predicted time only;
4. accumulated distance-equivalent/mechanical load;
5. TRIMP computed from prescribed HRR and **predicted** segment time;
6. TRIMP computed from observed segment HRR/time, labelled reconstruction-only.

Also vary segment length (0.5/1/2 km), HR lag where raw streams exist, fatigue form, and the
`kappa` grid. This experiment determines whether the current +8.36 min advantage over progress
survives removal of observed-time leakage. If it does not, TRIMP should be presented as a
reconstruction feature rather than a prospective fatigue state.

### E6 - Parameter identifiability and longitudinal evolution

**Question:** Can an interpretable capacity state be recovered and does allowing it to evolve
improve future prediction?

First test identifiability by simulating outcomes on the observed course/HRR designs with known
parameters, adding block-resampled residuals, and refitting. Examine recovery bias, interval
coverage, and the joint `alpha`-`kappa` surface.

Then compare chronological models:

- one global effective capacity and one global `kappa`;
- rolling 90- and 180-day capacity;
- last 20 and last 40 eligible activities;
- exponentially weighted capacity with pre-specified half-lives;
- a regularised random-walk/state-space capacity.

Keep `kappa` global initially. Only allow time-varying `kappa` if simulation shows that both
parameters are recoverable. Use the effective speed scale `v_anchor * alpha`, not `alpha` alone,
as the longitudinal quantity. Include shuffled-date, season, route-mix, and sensor-change
negative controls.

**Success gate:** the dynamic state must improve strictly future, rolling-origin prediction
over the static model and its temporal changes must exceed parameter uncertainty. Otherwise
report parameter stability/non-identifiability and remove the progression claim. The present
26-month archive supports a longitudinal or seasonal analysis, not a strong multi-year trend.

### E7 - Optional causal within-race updating

At 10%, 25%, 50%, and 75% course completion, update a shrinkage estimate of race-day capacity
using completed segments only, then predict the remaining course. Compare against:

- no update from the pre-race model;
- naive current-pace extrapolation;
- terrain-weighted pace extrapolation.

Future HR, future segment times, and final time must be hidden. The relevant result is the
change in final-time error at each checkpoint, not fit to completed segments. This remains
secondary until there are enough independently labelled events.

### E8 - Test the lab-free and Strava-compatible software claim

The current configuration fixes VMA at 18 km/h and contains athlete/date-specific choices.
Compare three anchors:

1. configured VMA;
2. profile threshold value;
3. archive-derived flat capacity or critical-speed estimate from previous activities only.

If the archive-derived anchor preserves performance, the paper can claim **no laboratory
visit required**. Otherwise say **no new laboratory testing, given a user-supplied speed and HR
profile**.

For software reuse, perform a clean replay from an exported archive into an empty data
directory and require:

- explicit athlete selection rather than "first athlete";
- no hard-coded activity IDs or race dates;
- automatic eligibility and support reports;
- a complete config, seed, input hash, and software manifest;
- the same outputs within numerical tolerance.

This validates implementation portability for another archive. It does not establish
scientific accuracy for another athlete.

## 6. Statistical reporting

- **Unit of inference:** activity; event date when multiple records share one effort.
- **Primary cohort:** hard trail. Secondary cohorts must not replace it because they have a
  better headline error.
- **Primary endpoint:** paired change in activity-level MAPE for B3 versus the best matched
  fixed-intensity baseline.
- **Uncertainty:** 10,000 activity/date block-bootstrap replicates, with the full fit repeated
  inside each replicate when computationally feasible.
- **Secondary endpoints:** MAE in minutes, median APE, signed bias, duration-dependent bias,
  route-pair concordance, and predictive-interval coverage.
- **Segments:** descriptive diagnostics or cluster-bootstrap inference only.
- **Multiplicity:** one primary contrast; label all other model variants exploratory or apply a
  Holm correction to a short, frozen family.
- **Missingness:** report exclusions and reasons; do not silently remove failed or difficult
  events.
- **Negative controls:** shuffled HRR between activities of similar duration, shuffled dates for
  longitudinal models, and Rome as an explicit out-of-domain road control rather than evidence
  about trail accuracy.

## 7. Decision table for the final paper story

| Result pattern | Defensible conclusion |
|---|---|
| B3 beats fixed intensity; E4 also wins | Strava-history, intensity-conditioned prospective trail model within the supported domain |
| B3 wins; E4 fails | Useful prescribed-intensity simulator, but the athlete/coach must choose HRR |
| Only full segment-HRR U1 wins | Retrospective HR-informed reconstruction; no prospective target-HRR claim |
| E6 dynamic state wins | Effective capacity can be tracked longitudinally for this athlete |
| E6 does not win | Static athlete model; omit progression language |
| E8 archive anchor wins | Lab-free calibration from an ordinary activity archive |
| E8 archive anchor fails | Strava-compatible pipeline requiring a physiological speed anchor |

## 8. Priority and implementation artifacts

### P0 - Required before using the new story

1. E0 data/support freeze and independent event labels.
2. E2 constant-mean-HRR causal LOO/rolling-origin comparison.
3. E5 predicted-time TRIMP leakage ablation.
4. E4 fully nested history-only HRR selection.
5. Rewrite the manuscript around whichever decision-table row passes.

### P1 - Strong additions with the current archive

6. E3 repeated-route contrast.
7. E6 identifiability and time-varying capacity.
8. E8 archive-derived anchor and clean Strava replay.

### P2 - Exploratory

9. E7 within-race updating.
10. Terrain-family HRR allocation after the mean-HRR result is established.

Suggested implementation targets:

- `configs/trail_digital_twin_claim_validation.yaml`
- `scripts/validate_intensity_conditioned_claim.py`
- `data/exp_perf_predictions/trail_digital_twin_claim_validation/`
- `table_data_support.csv`
- `table_prescribed_hrr_model_comparison.csv`
- `table_route_pair_validation.csv`
- `table_rolling_origin_hrr_selection.csv`
- `table_causal_fatigue_ablation.csv`
- `table_parameter_recovery.csv`
- `table_longitudinal_state_validation.csv`
- `table_inrace_update.csv`

The run should end with one machine-readable claim-gate table containing `pass`, `fail`, or
`not testable` for every claim. That prevents a strong retrospective result from being used
later as evidence for a different prospective claim.

