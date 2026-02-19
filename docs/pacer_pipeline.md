# Pacer and Activity Preprocessing Pipeline

## Overview

This document explains the preprocessing pipelines used for pacing (route preparation)
and for activity analytics, including how grade is classified and how outliers are
removed from elevation data.

## Inputs

- **Pacer (route preparation)**: GPX route without timestamps.
  - Source: `utils/gpx_parser.parse_gpx_to_timeseries`
  - Expected columns: `lat`, `lon`, `elevationM`
- **Activity preprocessing**: Activity timeseries with timestamps.
  - Source: activity metrics/timeseries storage
  - Expected columns: `lat`, `lon`, `elevationM`, `timestamp`

## Preprocessing Pipelines

### Pacer preprocessing (distance-referenced)

Location: `services/pacer_service.PacerService.preprocess_timeseries_for_pacing`

Key characteristics:

- Reference unit: **distance**
- Timestamps: **not required**
- Outputs: `cumulated_distance`, `elevationM_ma_5`, `grade_ma_10`

Steps (simplified):

1. Compute GPS distance on original lat/lon.
2. Build `cumulated_distance`.
3. Filter elevation spikes using distance-based window.
4. Smooth elevation and compute elevation differences.
5. Smooth distance, compute grade, smooth grade.

### Activity preprocessing (time-referenced)

Location: `services/speed_profile_service.SpeedProfileService.preprocess_timeseries`

Key characteristics:

- Reference unit: **time**
- Timestamps: **required** for duration/speed
- Outputs include: `cumulated_distance`, `duration_seconds`, `speed_km_h`, `grade`

Steps (simplified):

1. Filter elevation spikes using time-based window.
2. Smooth lat/lon and elevation.
3. Compute GPS distance and cumulative distance.
4. Convert timestamps to datetime and compute durations.
5. Compute speed and grade.

## Outlier Filtering

Location: `utils/series_filters.filter_series_outliers`

This uses a Hampel-style filter:

- Compute rolling median.
- Compute MAD (median absolute deviation).
- Flag outliers when `|x - median| > sigma * 1.4826 * MAD`.
- Replace outliers with mean of immediate neighbors (fallback to median).

Window behavior:

- **Datetime reference**: `window` is interpreted in seconds.
- **Numeric reference**: `window` is in the same units (e.g., km).
- **No reference**: `window` is treated as sample count.

Current usage:

- Activity preprocessing uses `reference_col="timestamp"`.
- Pacer preprocessing uses `reference_col="cumulated_distance"`.

## Grade Classification

Location: `utils/grade_classification`

- `classify_grade_pacer_5cat`: Used by the pacer segmentation pipeline.
  - Categories: `steep_up`, `run_up`, `flat`, `down`, `steep_down`
- `classify_grade_elevation_8cat`: Used for elevation chart visualization.
  - Categories: `grade_lt_neg_0_5`, `grade_lt_neg_0_25`, `grade_lt_neg_0_05`,
    `grade_neutral`, `grade_lt_0_1`, `grade_lt_0_25`, `grade_lt_0_5`, `grade_ge_0_5`
