# Trail Digital Twin Research Documentation

Technical documentation for the two notebook workflow:

- `notebooks/trail_digital_twin_reproduction.ipynb`
- `notebooks/trail_digital_twin_extensions.ipynb`

Repository audit date: 2026-06-16.

## Abstract

The local trail digital twin work is now split into two analyses. The reproduction notebook
tries to follow the Sensors 2026 paper as closely as possible on repository data. It uses
race-level calibration after summing 1 km segment predictions, matching the paper's
validation objective. The extension notebook deliberately goes beyond the paper with
heart-rate effort, REDI, VMA 18 km/h, acute in-activity TRIMP, richer segment regression,
and a separate segment-level grid search.

The paper-style reproduction is evaluated on three cohorts: all hard TrailRuns, the top
10 hard TrailRuns by average HR reserve ratio, and a configurable selected-date race
subset. CTL/ATL/TSB and REDI are computed from all activities, not only TrailRuns.
Weather remains limited to first-party Strava fields or reviewed provider caches. Activity
descriptions and Enduraw-derived text are not used.

## 1. Sensors Paper Context

The Sensors paper proposes a physics-based trail-running digital twin for one athlete
across 13 trail races. Its core claim is that trail race time is better predicted by a
course-segment model than by distance alone because grade, altitude, training status,
and pacing decay change sustainable speed.

The original study used:

| Element | Paper design |
| --- | --- |
| Athlete | One highly trained male trail runner |
| Dataset | 13 trail races |
| Course grain | 1 km segments from 1 Hz barometric data |
| Physiology | Laboratory VT2 at 20 percent grade |
| Terrain model | Minetti cost polynomial and GAP |
| Readiness | Banister-style CTL/TSB term |
| Calibration | Race-level grid search over `alpha` and `mu` |
| Validation | Sequential stages and leave-one-out CV |

The paper reports approximately:

| Paper stage | R2 | MAE min | MAPE pct |
| --- | ---: | ---: | ---: |
| Stage 1 distance-only | 0.679 | 28.3 | 17.6 |
| Stage 2 GAP + altitude | 0.871 | 21.1 | 14.0 |
| Stage 3 calibrated | 0.905 | 14.6 | 8.7 |
| LOO Stage 3 | 0.864 | 18.2 | 11.1 |

## 2. Local Data and Cohorts

The notebooks use local repository files under `data/`:

| Source | Use |
| --- | --- |
| `activities.csv` | Activity metadata, moving time, source timestamps |
| `activities_metrics.csv` | Category, TRIMP, distance-equivalent load, HR summaries |
| `daily_metrics.csv` | All-activity daily load for CTL/ATL/TSB and REDI |
| `athlete.csv` | HR rest and HR max |
| `thresholds.csv` | `Threshold 30` flat-speed proxy for VT2 |
| `data/timeseries/` | GPS, elevation, HR, speed streams |
| `data/metrics_ts/` | Fallback processed streams |
| `data/raw/strava/` | First-party `average_temp` when present |

The three paper-reproduction cohorts are:

| Cohort | Definition |
| --- | --- |
| `hardTrailRun` | `TRAIL_RUN`, usable segment stream, moving time at least 1800 s, and hard by distance, ascent, or HRR |
| `top10HardTrailByHRR` | Top 10 hard TrailRuns sorted by highest average HR reserve ratio |
| `selectedDateRaces` | One best usable activity per editable ISO date list |

The selected-date cohort is configured directly in the notebook:

```python
SELECTED_RACE_DATE_STRINGS = [
    "2025-10-18",
    "2026-04-12",
    ...
]
```

For each date, the selector chooses the best activity by:

1. Prefer `TRAIL_RUN`.
2. Then prefer `RUN`.
3. Then choose the largest `distanceKm + 0.01 * ascentM`.

This makes the race-date subset easy to change without editing model code.

## 3. Paper Reproduction Methodology

### 3.1 Segment Extraction

The notebooks reconstruct 1 km segments from local streams. Each segment stores
distance, duration, D+/D-, mean altitude, grade, progress, mean HR reserve, speed, terrain
family, and GPS technicality.

The GPS technicality proxy is:

$$
T_s = \mathrm{clip}\left(\frac{\sigma(d_{\perp})}{50}, 0, 1\right)
$$

where \(d_{\perp}\) is perpendicular GPS deviation from the segment chord. This is a
geometry proxy, not a validated trail-surface measure.

### 3.2 Minetti GAP

The Minetti running cost polynomial is:

$$
C(i)=155.4i^5-30.4i^4-43.3i^3+46.3i^2+19.5i+3.6
$$

with grade \(i\) clamped to \([-0.75,0.75]\) in the current benchmark code. Raw GPS
grade samples with absolute grade above 100 percent are treated as elevation spikes
and linearly interpolated before this clamp. The grade adjustment is:

$$
f_{GAP}(i)=\frac{C(i)}{C(0)}
$$

so \(f_{GAP}(0)=1\).

### 3.3 Altitude

The altitude factor is:

$$
f_{alt}(a)=1-11.7\times10^{-9}a^2-4.01\times10^{-6}a
$$

where \(a\) is mean segment altitude in meters.

### 3.4 Pacing Decay

The paper-style linear decay term is:

$$
f_{pad}(s)=1+\mu s
$$

where \(s\in[0,1]\) is race progress. Negative \(\mu\) reduces sustainable speed later in
the race and therefore increases segment time. The exponential sensitivity model is:

$$
f_{exp}(s)=e^{\lambda s}
$$

The reproduction notebook now uses a wider local calibration grid than the first
implementation. The previous grid was too narrow and forced optima onto its lower
boundary, so it could not distinguish a real optimum from a search-range artifact.

| Parameter | Grid |
| --- | --- |
| `alpha` | 0.40 to 1.00, step 0.02 |
| `mu` or `lambda` | -0.50 to 0.00, step 0.02 |

The notebook also includes fixed-alpha curves and LOO fold distributions to explain why
the in-sample optimum can be \(\mu=0\) in the paper while fold-level optima may be
negative. In the wider local sweep, race-level CTL calibration generally selected small
negative \(\mu\) values or zero, while segment-level diagnostics selected stronger
negative \(\mu\). The race-summed objective mostly corrected global speed scale through
\(\alpha\); the segment objective was more sensitive to within-race fatigue shape.

### 3.5 Readiness

Source audit finding, 2026-06-16: the Sensors paper states that pre-race readiness is
scaled by a Banister-type \(f_{CTL}\) term using 42-day CTL and race-day TSB, but it does
not publish the actual scaling formula or numeric coefficients. The Banister-related
sources support the impulse-response structure, not the current bounded multiplier.

| Source | Finding for readiness |
| --- | --- |
| Sensors 2026 trail paper | Mentions \(f_{CTL}\), 42-day CTL, and race-day TSB, but does not publish the factor formula |
| Calvert/Banister systems model | Models performance as fitness from training minus weighted fatigue from training |
| Fitz-Clarke/Morton/Banister 1991 | Expresses the model as an influence curve and gives example defaults near \(\tau_1=45\), \(\tau_2=15\), \(k_1=1\), \(k_2=2\) |
| Mujika et al. 1996 | Uses positive and negative training influences; taper gains are attributed mainly to reduced negative influence |

The source-backed Banister/Fitz-Clarke/Mujika structure is:

$$
PI_d = k_1 \sum_{i \le d} w_i e^{-(d-i)/\tau_{fitness}}
$$

$$
NI_d = k_2 \sum_{i \le d} w_i e^{-(d-i)/\tau_{fatigue}}
$$

$$
R_d = PI_d - NI_d
$$

where \(w_i\) is the daily training impulse, \(PI\) is the positive influence or fitness
component, \(NI\) is the negative influence or fatigue component, and \(R_d\) is the raw
readiness/performance state. Mujika's taper result is consistent with this form:
performance improved mainly because \(NI\) decreased, while \(PI\) was not meaningfully
increased by tapering.

For implementation, each exponential state can be computed recursively:

$$
X_{\tau,d} = w_d + e^{-1/\tau}X_{\tau,d-1}
$$

If using normalized CTL/ATL style EWMAs instead of unnormalized Banister convolutions:

$$
CTL_d = \alpha_{42}w_d + (1-\alpha_{42})CTL_{d-1}
$$

$$
ATL_d = \alpha_{\tau_f}w_d + (1-\alpha_{\tau_f})ATL_{d-1}
$$

$$
TSB_d = CTL_d - ATL_d
$$

then the Banister form becomes:

$$
R_d = k_1CTL_d-k_2ATL_d = (k_1-k_2)CTL_d + k_2TSB_d
$$

This is the paper-derived CTL/TSB readiness state. Because CTL and ATL are normalized
EWMAs, \(k_1\) and \(k_2\) should be fitted or explicitly chosen for this normalization;
they should not be copied blindly from unnormalized impulse-response examples.

The current local implementation is a pragmatic speed multiplier:

$$
f_{CTL}=\mathrm{clip}\left(
1+w_{CTL}\frac{CTL-CTL_{ref}}{CTL_{ref}}+
w_{TSB}\frac{TSB}{CTL_{ref}},
0.90,1.08
\right)
$$

with \(w_{CTL}=0.05\) and \(w_{TSB}=0.10\). REDI is compared with the same bounded form,
using slow REDI as the chronic term and slow-minus-fast REDI as balance.

This formula is intentionally bounded because the race model needs a positive,
dimensionless multiplier near 1.0. It should be treated as a heuristic baseline, not as a
direct derivation from the Banister-related sources.

Recommended paper-derived factor for future tests:

$$
f_{readiness,d} = \exp\left(\beta\frac{R_d-R_{ref}}{R_{scale}}\right)
$$

where \(R_{ref}\), \(R_{scale}\), and \(\beta\) are estimated on the training fold only.
The exponential keeps the factor positive and makes \(\beta=0\) equivalent to no
readiness effect.

Future readiness tests should cover:

| Test | Expected result |
| --- | --- |
| Single impulse decay | \(X_{\tau,d+1}=e^{-1/\tau}X_{\tau,d}\) on days without load |
| No race-day leakage | Race readiness uses the latest strictly previous daily state |
| CTL/TSB algebra | \(k_1CTL-k_2ATL\) equals \((k_1-k_2)CTL+k_2TSB\) |
| Taper behavior | After training stops, fatigue decays faster than fitness and \(R_d\) rises initially |
| Fold-local scaling | \(R_{ref}\), \(R_{scale}\), and \(\beta\) are learned without held-out races |
| Model comparison | Compare no readiness, current heuristic, and Banister-derived readiness under the same LOO split |

Acceptance criterion for replacing the heuristic: the Banister-derived factor should
improve held-out race-level error versus no readiness without merely absorbing the global
speed scale already handled by \(\alpha\). If fitted \(\beta\) collapses to zero or worsens
LOO error, keep readiness as a diagnostic feature rather than a pace multiplier.

### 3.6 Segment Pace Equation

The paper writes a pace equation. The implementation uses the equivalent segment-time
form:

$$
\hat{t}_s =
\frac{d_s\,3600\,f_{GAP,s}}
{v_{VT2}\alpha f_{alt,s}f_{heat,s}f_{CTL}f_{pad}(s)}
$$

where \(d_s\) is segment distance in kilometers. \(f_{heat}=1\) in the current data because
no allowed TrailRun weather observations exist.

Calibration is paper-faithful: segment predictions are summed by race first, then
`alpha` and `mu` are chosen by highest race-level \(R^2\), with MAE as the tie-breaker.

## 4. Paper Reproduction Results

The executed reproduction notebook with the wider `alpha` and `mu` sweep produced:

| Cohort | Best paper-style model | R2 | MAE min | MAPE pct | Best alpha | Best mu |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| hardTrailRun | Stage 3 CTL | 0.822 | 50.7 | 35.4 | 0.42 | -0.04 |
| top10HardTrailByHRR | Stage 3 CTL | 0.968 | 8.7 | 11.2 | 0.64 | -0.02 |
| selectedDateRaces | Stage 3 CTL | 0.909 | 45.0 | 36.8 | 0.48 | 0.00 |

The top-10 HRR subset is much closer to the paper than all hard TrailRuns. This is
consistent with the paper using true race efforts from a controlled race dataset, whereas
the local hard set mixes races and hard training runs.

The wider sweep changes the interpretation of the earlier calibration. The previous
`alpha = 0.80`, `mu = -0.20` values were boundary solutions caused by the restricted grid.
Once `alpha` is allowed to fall to 0.40, Stage 3 improves strongly on all cohorts and the
best race-level \(\mu\) moves back near zero. The dominant correction is therefore a lower
effective speed scale than the local flat `Threshold 30` proxy implies, not a very large
race-level pacing-decay term.

Stage 2 remains systematically too fast on the full hard set:

| Cohort | Stage 2 variant | R2 | MAE min | Bias min |
| --- | --- | ---: | ---: | ---: |
| hardTrailRun | GAP + altitude | 0.132 | 100.0 | -100.0 |
| hardTrailRun | GAP + altitude + CTL | 0.115 | 101.3 | -101.3 |
| top10HardTrailByHRR | GAP + altitude | 0.434 | 36.4 | -36.4 |
| selectedDateRaces | GAP + altitude | 0.420 | 80.3 | -80.3 |

The reason is scale: the local flat `Threshold 30` speed is not the same physiological
quantity as the paper's laboratory uphill VT2. Distance-only can outperform uncalibrated
Stage 2 because its fitted coefficient absorbs athlete speed scale directly, while Stage 2
applies GAP and altitude to a speed anchor that is too fast for the local trail context.
Readiness can moderate predictions but cannot repair that speed-scale mismatch.

The segment-level diagnostic in the reproduction notebook is labelled non-paper
equivalent. It directly fits segment time and reports race-summed metrics as a diagnostic:

| Cohort | Segment diagnostic alpha | Segment diagnostic mu | Segment MAE min | Race MAE min |
| --- | ---: | ---: | ---: | ---: |
| hardTrailRun | 0.56 | -0.32 | 3.72 | 52.41 |
| top10HardTrailByHRR | 0.80 | -0.30 | 1.14 | 13.46 |
| selectedDateRaces | 0.64 | -0.26 | 2.30 | 42.30 |

This diagnostic confirms that segment-level fitting wants stronger late-race slowing than
the paper-faithful race-level objective. It should not replace the race-level reproduction,
but it helps reveal segment residuals hidden by finish-time-only calibration.

## 5. Extension Methodology

The extension notebook is not a paper replication. It starts from the Sensors speed
equation and then tests a smaller number of explicit physiological hypotheses with
\(VMA=18.0\) km/h as the flat speed anchor for extension-only stages.

### 5.1 Extension Hypotheses and Notation

The reproduction estimates pre-race performance from route, altitude, readiness, and a
progress-based pacing decay. The extension asks a different question: can observed HR and
acute in-activity load explain submaximal or variable-effort trail runs without moving to
a high-dimensional black-box model?

The main hypotheses are:

| Hypothesis | Implementation |
| --- | --- |
| Terrain cost remains the physical baseline | Keep \(f_{GAP}\) from Minetti and \(f_{alt}\) from the Sensors paper |
| \(p_{rs}\) is a progress proxy, not a physiological load | Use \(p_{rs}\) only before acute TRIMP enters the model |
| In-race fatigue should depend on accumulated work | Compare decayed \(D_{rs}\) and cumulative \(C_{rs}\) TRIMP before each segment |
| REDI may represent readiness better than CTL/ATL averages | Replace the CTL factor with a REDI slow/balance factor in Stage 2 |
| HR reserve should scale speed approximately linearly at submaximal effort | Add \(E(HRR)=HRR/0.70\) only in Stage 3 |
| Small cohorts cannot support many free coefficients | Remove the previous high-dimensional segment model from the staged method and keep it out of the conclusions |

For activity \(r\) and segment \(s\):

| Symbol | Meaning |
| --- | --- |
| \(t_{rs}\) | observed segment moving time |
| \(T_r\) | observed activity moving time |
| \(d_{rs}\) | segment distance in km |
| \(i_{rs}\) | segment grade |
| \(a_{rs}\) | mean segment altitude |
| \(p_{rs}\) | race progress in \([0,1]\) |
| \(HRR_{rs}\) | mean segment HR reserve ratio |
| \(C_{rs}\) | cumulative TRIMP before the segment |
| \(D_{rs}\) | exponentially decayed cumulative TRIMP before the segment |
| \(f_{CTL,r}\) | bounded CTL/TSB readiness factor |
| \(f_{REDI,r}\) | bounded REDI slow/balance readiness factor |

### 5.2 Linear Regression Diagnostic: E1 to E6bis

The all-linear segment regression is kept as a diagnostic only. It is useful for
inspecting coefficient signs and standardized effect sizes, but it is not the main
extension model because it abandons the paper's speed-equation structure.

The diagnostic equation is:

$$
\log(t_{rs})=\beta_0+\beta_d\log(d_{rs})+\beta_g\log(f_{GAP,rs})
+\beta_a(1-f_{alt,rs})+\beta_p p_{rs}+\beta_h HRR_{rs}
+\beta_lL_r+\beta_uU_{rs}+\epsilon_{rs}
$$

Predicted segment times are exponentiated and summed:

$$
\hat t_{rs}=\exp(x_{rs}^{\top}\hat\beta), \qquad
\hat T_r=\sum_s \hat t_{rs}
$$

The staged diagnostic feature sets are:

| Stage | Formula terms in \(x_{rs}\) |
| --- | --- |
| E1 distance | \(1,\log(d_{rs})\) |
| E2 terrain physics + progress | E1 + \(\log(f_{GAP,rs})+(1-f_{alt,rs})+p_{rs}+\mathrm{terrainFamily}_{rs}\) |
| E3 CTL readiness | E2 + \(CTL_r+TSB_r\) |
| E4 REDI readiness | E2 + \(REDI^{slow}_r+REDI^{balance}_r\) |
| E5 HR effort | E2 + \(HRR_{rs}+CTL_r+TSB_r\) |
| E6 decayed acute TRIMP | E5 without \(p_{rs}\), plus \(D_{rs}\) only |
| E6bis cumulative acute TRIMP | E5 without \(p_{rs}\), plus \(C_{rs}\) only |

The removal of \(p_{rs}\) in E6/E6bis is intentional. Once acute TRIMP is added, fatigue
is represented by accumulated work rather than by race progress alone.

The notebook displays the complete coefficient table for E6 and E6bis, including raw
coefficients and standardized coefficients:

$$
\beta_j^{std} = \beta_j \frac{\sigma(x_j)}{\sigma(\log(t))}
$$

This gives a variable-importance diagnostic on the same scale. It is not interpreted as a
causal effect because terrain dummies, grade cost, HR, and TRIMP are correlated.

### 5.3 Acute In-Activity TRIMP

Segment TRIMP is:

$$
TRIMP_{rs} = \frac{t_{rs}}{3600}HRR_{rs}\,0.64\,e^{1.92HRR_{rs}}
$$

The leakage-free cumulative term is:

$$
C_{rs}=\sum_{j<s}TRIMP_{rj}
$$

The leakage-free decayed term is:

$$
D_{rs}=\sum_{j<s}e^{-\lambda(s-j)}TRIMP_{rj}
$$

The notebooks use `cumTrimpBefore` for \(C_{rs}\) and `decayedTrimpBefore` for
\(D_{rs}\). Segment \(s\) receives only previous work \(j<s\), preventing the model from
using the segment's own observed duration and HR to predict itself.

For pre-race constant-HRR simulation and completed-activity visualization, the sequential
simulators use `cumTrimpBefore` by default as a one-way acute performance reserve. This is
a local modelling choice: it makes fixed-HRR performance decrease over elapsed duration.
Use `decayedTrimpBefore` only when the goal is to model recent in-activity load that can
partly recover after easier segments.

### 5.4 HR Global and HR Segment Baselines

The HR global model is an activity-level LOO model trained on all usable TrailRuns except
the held-out activity:

$$
\log(T_r)=
\beta_0+\beta_D\log(DE_r)+\beta_A\frac{ascent_r}{distance_r}
+\beta_H HRR_r+\beta_C CTL_r+\beta_B TSB_r
+\beta_R REDI^{slow}_r+\beta_Q REDI^{balance}_r
+\beta_G T^{gps}_r+\beta_Z altitude_r+\beta_W temperature_r+\epsilon_r
$$

where \(DE_r\) is the repository distance-equivalent metric, with fallback:

$$
DE_r = distance_r + 0.01\,ascent_r
$$

The HR segment model is also all-trail LOO, but it uses the E6 decayed-TRIMP feature set
at segment level:

$$
\log(t_{rs}) = x^{E6}_{rs}\hat\beta+\epsilon_{rs}
$$

For held-out activity \(r\), all segments from \(r\) are removed from training, each
held-out segment is predicted, and predictions are summed:

$$
\hat T_r=\sum_s \exp(x^{E6}_{rs}\hat\beta_{-r})
$$

The global model has fewer rows and fewer segment-shape assumptions. The segment model
has many more rows but correlated observations inside each activity. Both are
retrospective because they use observed HR.

### 5.5 Paper-Based Extension Stages

The primary extension now uses a paper-like speed equation at every stage. There is no
linear regression of time in these stages. Parameters are fitted by grid search against
race/activity moving time, and predictions are sums of segment times.

**Stage 0: reproduction Stage 3.** This is the Sensors-style calibrated model with CTL
readiness and progress decay:

$$
\hat t_{rs} =
\frac{d_{rs}\,3600\,f_{GAP,rs}}
{v_{VT2}\,\alpha\,f_{alt,rs}\,f_{CTL,r}\,f_{pad}(p_{rs})}
$$

with:

$$
f_{pad}(p_{rs})=\mathrm{clip}(1+\mu p_{rs},0.1,\infty)
$$

**Stage 1: TRIMP fatigue with CTL readiness.** The progress decay is removed and replaced
with acute decayed TRIMP:

$$
\hat t_{rs} =
\frac{d_{rs}\,3600\,f_{GAP,rs}}
{VMA\,\alpha\,f_{alt,rs}\,f_{CTL,r}\,F(D_{rs})}
$$

Two fatigue shapes are compared:

$$
F_{lin}(D_{rs})=\mathrm{clip}\left(1-\kappa\frac{D_{rs}}{U_{scale}},F_{min},1\right)
$$

$$
F_{exp}(D_{rs})=\exp\left(-\kappa\frac{D_{rs}}{U_{scale}}\right)
$$

The fatigue factor multiplies speed. A smaller \(F\) therefore increases predicted time.

**Stage 2: REDI readiness.** Stage 2 keeps the Stage 1 TRIMP fatigue structure but
replaces the CTL readiness factor with the REDI readiness factor:

$$
\hat t_{rs} =
\frac{d_{rs}\,3600\,f_{GAP,rs}}
{VMA\,\alpha\,f_{alt,rs}\,f_{REDI,r}\,F(D_{rs})}
$$

**Stage 3: HRR speed ratio.** Stage 3 keeps REDI and TRIMP fatigue, adds a fixed
linear HR reserve speed multiplier, and compares decayed versus cumulative acute TRIMP
states:

$$
\hat t_{rs} =
\frac{d_{rs}\,3600\,f_{GAP,rs}}
{VMA\,\alpha\,f_{alt,rs}\,f_{REDI,r}\,E(HRR_{rs})\,F(U_{rs})},
\quad U_{rs}\in\{D_{rs}, C_{rs}\}
$$

with:

$$
E(HRR_{rs})=
\mathrm{clip}\left(\frac{HRR_{rs}}{0.70},E_{min},E_{max}\right)
$$

This implements the hypothesis that, within the submaximal range, speed scales roughly
linearly with HR reserve and the remaining fatigue effect should be carried by acute
TRIMP. HRR has no fitted coefficient in Stage 3; only \(\alpha\), \(\kappa\), and the
fatigue shape are selected. Stage 3 additionally selects whether \(U\) is the decayed
state \(D\) or cumulative state \(C\), each tested with linear and exponential fatigue.

The grids are intentionally small:

| Parameter | Stage | Grid |
| --- | --- | --- |
| \(\alpha\) | Stage 0 | 0.40 to 1.00, step 0.05 |
| \(\mu\) | Stage 0 | -0.50 to 0.00, step 0.05 |
| \(\alpha\) | Stages 1-3 | 0.35 to 0.85, step 0.05 |
| \(\kappa\) | Stages 1-3 | 0.00, 0.10, 0.20, 0.30, 0.40 |
| Fatigue shape | Stages 1-3 | linear or exponential |
| Fatigue state | Stage 3 | decayed \(D\) or cumulative \(C\) |

### 5.6 Statistical Evaluation

Each paper-based stage is evaluated on the same three cohorts:
`hardTrailRun`, `top10HardTrailByHRR`, and `selectedDateRaces`.

Two validations are reported:

| Validation | Training set | Evaluation target |
| --- | --- | --- |
| In-sample | Fit the grid on the evaluated cohort | Same cohort moving times |
| Cohort LOO | For each held-out activity, fit on the same cohort excluding that activity | Held-out activity moving time |

The LOO split is activity-level: all segments from the held-out activity are removed
from training. However, Stage 3 still uses observed HR in the held-out activity, so it is
a retrospective effort-adjusted model, not a pre-race predictor.

Variable importance is evaluated in two ways:

1. The OLS diagnostic reports raw and standardized coefficients for every E6/E6bis
   variable.
2. The paper-based Stage 3 model uses one-variable ablation: keep the fitted Stage 3
   parameters fixed, neutralize one component, and measure the change in race MAE.

### 5.7 Exploratory Segment-Level Grid Search

The previous segment grid search is retained only as a stress test. It directly optimizes
segment time with more degrees of freedom than the paper-based stages:

$$
\hat{t}_s =
\frac{d_s\,3600\,f_{GAP,s}}
{VMA\,\alpha\,f_{alt,s}\,f_{fatigue}(s)\,M_{terrain}\,
\exp(\gamma_{HR}(HRR_s-0.70)-\gamma_{tech}T_s-\gamma_{acute}D_s)}
$$

It reports both segment-level and race-summed metrics so segment fit cannot be interpreted
without finish-time consequences. It is not used as a main conclusion model.

## 6. Extension Results

### 6.1 Linear Diagnostic

The E6 decayed acute-TRIMP diagnostic generalized better than E6bis on the full hard and
selected-date cohorts:

| Cohort | Linear diagnostic | R2 | MAE min | MAPE pct |
| --- | --- | ---: | ---: | ---: |
| hardTrailRun | E6 decayed in-sample | 0.975 | 15.0 | 9.1 |
| hardTrailRun | E6 decayed LOO | 0.967 | 16.5 | 9.8 |
| hardTrailRun | E6bis cumulative LOO | 0.930 | 19.1 | 10.0 |
| top10HardTrailByHRR | E6 decayed LOO | 0.977 | 7.8 | 8.9 |
| top10HardTrailByHRR | E6bis cumulative LOO | 0.982 | 7.2 | 8.9 |
| selectedDateRaces | E6 decayed LOO | 0.920 | 18.9 | 9.5 |
| selectedDateRaces | E6bis cumulative LOO | 0.764 | 26.7 | 11.1 |

The coefficient table in the notebook shows every E6/E6bis variable. The largest
standardized coefficients are usually terrain-family indicators, \(\log(d)\), and
\(\log(f_{GAP})\). The acute TRIMP coefficients are positive in the inspected final
stages, meaning higher prior in-activity load increases predicted segment time after
conditioning on distance, grade, altitude, HR, and readiness. The cumulative term is less
stable than the decayed term in LOO, especially on selected-date races.

### 6.2 HR Baselines

The HR LOO models produced:

| Cohort | HR model | R2 | MAE min | MAPE pct |
| --- | --- | ---: | ---: | ---: |
| hardTrailRun | HR global LOO | 0.974 | 16.9 | 10.5 |
| hardTrailRun | HR E6 segment LOO | 0.967 | 16.1 | 9.4 |
| top10HardTrailByHRR | HR global LOO | 0.952 | 10.0 | 13.0 |
| top10HardTrailByHRR | HR E6 segment LOO | 0.982 | 6.4 | 7.4 |
| selectedDateRaces | HR global LOO | 0.995 | 10.3 | 6.9 |
| selectedDateRaces | HR E6 segment LOO | 0.957 | 18.2 | 7.4 |

The HR global model remains the strongest selected-date retrospective predictor, while
the HR segment model is strongest on the top-10 HRR cohort.

### 6.3 Paper-Based Stage Results

In-sample stage results:

| Cohort | Stage | R2 | MAE min | MAPE pct |
| --- | --- | ---: | ---: | ---: |
| hardTrailRun | Stage 0 reproduction Stage 3 | 0.858 | 47.5 | 35.1 |
| hardTrailRun | Stage 1 TRIMP fatigue CTL | 0.858 | 45.7 | 32.7 |
| hardTrailRun | Stage 2 TRIMP fatigue REDI | 0.855 | 46.1 | 32.7 |
| hardTrailRun | Stage 3 HRR speed ratio | 0.962 | 25.6 | 19.3 |
| top10HardTrailByHRR | Stage 0 reproduction Stage 3 | 0.975 | 7.7 | 9.8 |
| top10HardTrailByHRR | Stage 1 TRIMP fatigue CTL | 0.978 | 7.1 | 9.6 |
| top10HardTrailByHRR | Stage 2 TRIMP fatigue REDI | 0.980 | 6.9 | 8.9 |
| top10HardTrailByHRR | Stage 3 HRR speed ratio | 0.986 | 4.9 | 5.5 |
| selectedDateRaces | Stage 0 reproduction Stage 3 | 0.933 | 37.5 | 30.2 |
| selectedDateRaces | Stage 1 TRIMP fatigue CTL | 0.933 | 39.9 | 33.8 |
| selectedDateRaces | Stage 2 TRIMP fatigue REDI | 0.935 | 39.0 | 32.9 |
| selectedDateRaces | Stage 3 HRR speed ratio | 0.989 | 11.4 | 7.6 |

Cohort-level LOO stage results:

| Cohort | Stage | R2 | MAE min | MAPE pct |
| --- | --- | ---: | ---: | ---: |
| hardTrailRun | Stage 0 reproduction Stage 3 LOO | 0.843 | 48.9 | 35.4 |
| hardTrailRun | Stage 1 TRIMP fatigue CTL LOO | 0.853 | 46.7 | 32.9 |
| hardTrailRun | Stage 2 TRIMP fatigue REDI LOO | 0.848 | 47.2 | 33.0 |
| hardTrailRun | Stage 3 HRR speed ratio LOO | 0.961 | 26.3 | 19.8 |
| top10HardTrailByHRR | Stage 0 reproduction Stage 3 LOO | 0.967 | 8.4 | 10.3 |
| top10HardTrailByHRR | Stage 1 TRIMP fatigue CTL LOO | 0.973 | 7.6 | 9.9 |
| top10HardTrailByHRR | Stage 2 TRIMP fatigue REDI LOO | 0.977 | 7.2 | 9.1 |
| top10HardTrailByHRR | Stage 3 HRR speed ratio LOO | 0.982 | 5.5 | 5.9 |
| selectedDateRaces | Stage 0 reproduction Stage 3 LOO | 0.909 | 41.6 | 31.4 |
| selectedDateRaces | Stage 1 TRIMP fatigue CTL LOO | 0.904 | 44.9 | 34.8 |
| selectedDateRaces | Stage 2 TRIMP fatigue REDI LOO | 0.905 | 44.3 | 34.1 |
| selectedDateRaces | Stage 3 HRR speed ratio LOO | 0.979 | 15.3 | 8.5 |

Interpretation:

- Replacing progress fatigue by decayed TRIMP improves Stage 0 modestly on the hard and
  top-10 HRR cohorts, but not on selected-date races.
- REDI improves Stage 1 on top-10 HRR in LOO, but not on the hard or selected-date
  cohorts. REDI is therefore not yet a universal replacement for CTL in this dataset.
- Adding the HRR speed ratio is the large step change. It reduces hardTrailRun LOO MAE
  from 48.9 min at Stage 0 to 26.3 min and selected-date LOO MAE from 41.6 min to
  15.3 min.
- Stage 3 is retrospective because it uses observed HR. It should be interpreted as an
  effort-adjusted performance model, not a race forecast available before the start.

### 6.4 Stage 3 Fatigue State Comparison

Stage 3 now compares the new integrated GAP model with both decayed and cumulative acute
TRIMP states, each fitted with linear and exponential fatigue. LOO selects:

| Cohort | Best Stage 3 fatigue state | Shape | LOO R2 | LOO MAE min | LOO MAPE pct |
| --- | --- | --- | ---: | ---: | ---: |
| hardTrailRun | decayed TRIMP | exponential | 0.961 | 26.3 | 19.8 |
| top10HardTrailByHRR | cumulative TRIMP | linear | 0.982 | 5.5 | 5.9 |
| selectedDateRaces | cumulative TRIMP | exponential | 0.979 | 15.3 | 8.5 |

The acute fatigue horizon is cohort-dependent. Heterogeneous hard trail runs still prefer
recent decayed load, while high-HRR and selected race-like runs prefer cumulative load.
This is the expected direction if fatigue is meant to model the decreasing speed that can
be sustained at a fixed HRR over time.

### 6.5 Stage 3 Ablation

The Stage 3 one-variable ablation reports the MAE increase relative to the full fitted
Stage 3 model:

| Cohort | Ablated component | Delta MAE min |
| --- | --- | ---: |
| hardTrailRun | no HRR speed ratio | +17.0 |
| hardTrailRun | no GAP | +12.8 |
| hardTrailRun | no REDI readiness | +2.9 |
| hardTrailRun | no altitude | +1.8 |
| hardTrailRun | no TRIMP fatigue | +0.0 |
| top10HardTrailByHRR | no GAP | +15.7 |
| top10HardTrailByHRR | no TRIMP fatigue | +7.0 |
| top10HardTrailByHRR | no HRR speed ratio | +5.0 |
| top10HardTrailByHRR | no REDI readiness | +2.1 |
| top10HardTrailByHRR | no altitude | +0.7 |
| selectedDateRaces | no GAP | +25.7 |
| selectedDateRaces | no HRR speed ratio | +12.4 |
| selectedDateRaces | no TRIMP fatigue | +10.3 |
| selectedDateRaces | no REDI readiness | +2.7 |
| selectedDateRaces | no altitude | +1.9 |

The dominant variables in the paper-based Stage 3 model are therefore integrated GAP and
HRR. TRIMP fatigue is neutral on the full hard set because the selected in-sample fatigue
coefficient is zero, but it is important on top10HardTrailByHRR and selectedDateRaces
when cumulative TRIMP is selected. Better calibration of \(U_{scale}\), segment length,
and HR lag may improve this.

### 6.6 Exploratory Segment Grid

The extension segment grid search produced:

| Cohort | Alpha | Mu | Segment MAE sec | Race MAE sec | Terrain profile | HR coef | Acute coef |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hardTrailRun | 0.50 | -0.30 | 207.7 | 2845.4 | 0 | 0.35 | 0.04 |
| top10HardTrailByHRR | 0.75 | -0.15 | 63.4 | 646.2 | 0 | 0.35 | 0.04 |
| selectedDateRaces | 0.60 | -0.30 | 126.2 | 2266.8 | 0 | 0.35 | 0.04 |

The grid selected non-zero acute TRIMP and HR coefficients in every cohort, which
supports modelling acute in-race effort/fatigue. However, the selected terrain profile was
the neutral profile in this first grid, suggesting either the tested terrain multipliers were
too coarse or the paper GAP term already absorbed most grade-family variation.

## 7. Limitations

- The repository does not contain the paper athlete's lab uphill VT2. The local threshold
  proxy is not physiologically equivalent.
- Moving time is used as the target. Race elapsed time effects such as aid stations and
  stops are not fully modelled.
- GPS-derived grade and technicality are weaker than the paper's barometric pipeline.
- CTL and REDI are pragmatic load approximations, not direct physiological fatigue
  measurements.
- The best acute TRIMP state is cohort-dependent; do not treat decayed or cumulative
  load as a general physiological conclusion from this single-athlete dataset.
- Segment regressions can overfit badly on small cohorts.
- HR-informed models use observed HR, so they are retrospective effort-adjusted models,
  not purely pre-race simulations.
- HRR is treated as a linear effort proxy in the constrained model. This is supported by
  HRR-to-VO2 reserve literature and submaximal running oxygen-cost studies, but cardiac
  drift means HR and speed are not interchangeable during prolonged hard efforts.

## 8. Verification

The helper tests cover:

- Minetti clamp and `f_GAP(0)=1`.
- Altitude monotonicity.
- Pacing-decay sign and speed-form equivalence.
- Cohort selection for top-10 HRR and selected dates.
- REDI feature computation and no future leakage in daily joins.
- In-activity cumulative and decayed TRIMP.
- Stage 3 fatigue-state comparison between decayed/cumulative and linear/exponential
  forms.
- Segment grid search synthetic recovery and race-summed metrics.
- Paper-based HRR-TRIMP speed equation, optional HRR multiplier, observed activity
  targets, and LOO reporting.

Executed commands:

```bash
uv run pytest tests/test_trail_performance_model.py
uv run ruff check services/trail_performance_model.py tests/test_trail_performance_model.py
uv run jupyter execute --inplace --timeout=1800 notebooks/trail_digital_twin_reproduction.ipynb
uv run jupyter execute --inplace --timeout=1800 notebooks/trail_digital_twin_extensions.ipynb
```



# Angle of analysis
- original papers use data from a single pro athlete on rather short distance (< 5h) so the athlete is at a consistent high percent of HRR and with a unique \alpha and mu it can be relvant. 

## References

1. Jaen-Carrillo, D.; Pattis, D. "A Physics-Based Digital Twin for Trail Running Race
   Performance Prediction: A Proof-of-Concept Study." Sensors 2026, 26, 3731.
   DOI: <https://doi.org/10.3390/s26123731>.
2. Local paper copy: `docs/science/sensors-26-03731.pdf`.
3. Minetti et al. walking/running slope-cost paper:
   `docs/science/minetti-et-al-2002-energy-cost-of-walking-and-running-at-extreme-uphill-and-downhill-slopes.pdf`.
4. Weather and technicality methodology:
   `docs/science/trail_weather_technicality_methodology.md`.
5. Swain, D.P.; Leutholtz, B.C. "Heart rate reserve is equivalent to %VO2 reserve, not
   to %VO2max." PubMed: <https://pubmed.ncbi.nlm.nih.gov/9139182/>.
6. Daniels, J.; Daniels, N. "Oxygen cost of running in trained and untrained men and
   women." PubMed: <https://pubmed.ncbi.nlm.nih.gov/870783/>.
7. Millet, G.Y. et al. "Heart rate running speed relationships-during exhaustive bouts
   in the laboratory." PubMed: <https://pubmed.ncbi.nlm.nih.gov/15630146/>.
