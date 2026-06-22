| dataset | column | description |
| --- | --- | --- |
| anonymized_activity_features.csv | activityIndex | Anonymous activity index with no link to source platform identifiers. |
| anonymized_activity_features.csv | cohort_hardTrailRun | Derived manuscript feature. |
| anonymized_activity_features.csv | cohort_top10HardTrailByHRR | Derived manuscript feature. |
| anonymized_activity_features.csv | cohort_selectedDateRaces | Derived manuscript feature. |
| anonymized_activity_features.csv | distanceKm | Distance in kilometers. |
| anonymized_activity_features.csv | ascentM | Total positive elevation gain in meters. |
| anonymized_activity_features.csv | actualTimeSec | Observed moving-time target in seconds. |
| anonymized_activity_features.csv | elapsedSec | Elapsed-time sensitivity target in seconds when available. |
| anonymized_activity_features.csv | hrReserveRatio | Activity-level heart-rate reserve ratio. |
| anonymized_activity_features.csv | ctl | Previous-day TRIMP chronic training load. |
| anonymized_activity_features.csv | tsb | Previous-day CTL minus ATL balance. |
| anonymized_activity_features.csv | trimpRediSlow | Slow REDI load state from all activities. |
| anonymized_activity_features.csv | trimpRediBalance | Slow minus fast REDI balance. |
| anonymized_activity_features.csv | ctlReadinessFactor | Bounded CTL/TSB readiness multiplier. |
| anonymized_activity_features.csv | rediReadinessFactor | Bounded REDI readiness multiplier. |
| anonymized_activity_features.csv | meanAltitudeM | Mean altitude in meters. |
| anonymized_activity_features.csv | technicalityGps | Activity-level GPS technicality proxy. |
| anonymized_activity_features.csv | temperatureC | First-party Strava average temperature when available. |
| anonymized_segment_features.csv | cohort | Evaluation cohort label; rows may be repeated across cohorts. |
| anonymized_segment_features.csv | activityIndex | Anonymous activity index with no link to source platform identifiers. |
| anonymized_segment_features.csv | segmentIndex | Distance-order segment number inside an anonymized activity. |
| anonymized_segment_features.csv | startKm | Segment start distance in kilometers. |
| anonymized_segment_features.csv | endKm | Segment end distance in kilometers. |
| anonymized_segment_features.csv | distanceKm | Distance in kilometers. |
| anonymized_segment_features.csv | avgGrade | Segment grade ratio. |
| anonymized_segment_features.csv | meanAltitudeM | Mean altitude in meters. |
| anonymized_segment_features.csv | elevGainM | Segment elevation gain in meters. |
| anonymized_segment_features.csv | elevLossM | Segment elevation loss in meters. |
| anonymized_segment_features.csv | progress | Segment race-progress fraction from 0 to 1. |
| anonymized_segment_features.csv | gapFactor | Minetti grade-adjustment factor. |
| anonymized_segment_features.csv | altitudePenalty | One minus the altitude correction factor. |
| anonymized_segment_features.csv | meanHrReserve | Segment-level heart-rate reserve ratio. |
| anonymized_segment_features.csv | segmentTrimp | Segment TRIMP computed from segment duration and HRR. |
| anonymized_segment_features.csv | cumTrimpBefore | Lagged cumulative TRIMP before the segment. |
| anonymized_segment_features.csv | decayedTrimpBefore | Lagged exponentially decayed TRIMP before the segment. |
| anonymized_segment_features.csv | terrainFamily | Coarse grade family. |
| anonymized_segment_features.csv | technicalityCombined | Segment-level technicality feature used by regression diagnostics. |
| anonymized_segment_features.csv | actualTimeSec | Observed moving-time target in seconds. |
| anonymized_segment_features.csv | stage3PredictedTimeSec | Stage 3 segment prediction from the cohort-specific fit. |
| anonymized_segment_features.csv | stage3ResidualSec | Stage 3 segment prediction minus observed segment moving time. |
| anonymized_segment_features.csv | stage3FatigueState | Selected Stage 3 acute fatigue state for this cohort. |
| anonymized_segment_features.csv | stage3AcuteTrimpCol | Source column used as the selected Stage 3 acute fatigue input. |
| anonymized_segment_features.csv | stage3FatigueModel | Selected Stage 3 fatigue shape for this cohort. |
