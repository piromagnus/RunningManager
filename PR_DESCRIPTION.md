# Add HR vs Speed Analysis with Cluster-Based Visualization

## Overview

This PR implements a comprehensive speed profile analysis feature that analyzes the relationship between Heart Rate (HR) and Speed across activities. The implementation is based on the analysis methodology from `notebooks/record_speed_profil.ipynb` and integrates it into the Streamlit dashboard.

## Features

### 1. Speed Profile Service (`services/speed_profile_service.py`)
- **Preprocessing pipeline**: Distance computation from GPS coordinates, speed calculation, moving averages, and HR-speed alignment
- **HR-Speed shift computation**: Automatically finds optimal time offset between HR and Speed signals for maximum correlation
- **Cluster-based analysis**: KMeans clustering on HR/Speed data with configurable number of clusters
- **Data persistence**: Caches processed metrics in `data/metrics_ts/` directory for performance

### 2. Dashboard Visualization (`pages/Dashboard.py`)
- **New tab**: "FC vs Vitesse" added to the Dashboard
- **Cluster centers visualization**: Displays cluster centers from all activities with uncertainty bars (based on standard deviation)
- **Weighted regression**: Implements weighted least squares regression where clusters with lower speed uncertainty have higher weight
- **Outlier filtering**: Automatically filters out clusters that:
  - Contain less than 5% of total points per activity (or minimum 20 points)
  - Have mean speed less than 6 km/h (walking/stopped periods)
- **Fixed axis ranges**: 
  - HR: 80-210 bpm (Y-axis)
  - Speed: 4-24 km/h (X-axis)
- **Rich tooltips**: Shows activity name, date, cluster ID, HR/Speed values and uncertainties

### 3. Activity Metrics Integration (`services/metrics_service.py`)
- **HR-Speed shift storage**: Automatically computes and stores `hrSpeedShift` in `activities_metrics.csv` during metrics recomputation
- **Robust GPS handling**: Falls back to `paceKmh` column when GPS data (lat/lon) is missing

### 4. Cache Reconstruction (`services/strava_service.py`)
- **Metrics recomputation**: "Reconstruire les activités depuis le cache Strava" now also recomputes `metrics_ts` files
- **Progress tracking**: Added progress bar with activity name display during reconstruction

## Technical Implementation

### Data Flow
1. Timeseries data is preprocessed (GPS-based distance → speed calculation)
2. HR and Speed signals are smoothed using moving averages (window size: 10)
3. Optimal HR-Speed offset is computed to maximize correlation
4. KMeans clustering is applied per activity (default: 5 clusters, configurable via `N_CLUSTER` env var)
5. Cluster centers are computed with mean and standard deviation
6. Clusters are filtered based on size and speed thresholds
7. Weighted regression is computed on all cluster centers

### Weighted Regression
- Uses inverse of speed standard deviation as weights: `weight = 1 / (speed_std + ε)`
- Clusters with lower speed uncertainty contribute more to the regression fit
- R² and standard error are calculated using weighted residuals

### Configuration
- Added `metrics_ts_dir` to `Config` (default: `data/metrics_ts`)
- Added `n_cluster` to `Config` (default: 5, configurable via `N_CLUSTER` env var)

### Data Storage
- `activities_metrics.csv`: Added `hrSpeedShift` column
- `data/metrics_ts/{activityId}.csv`: Stores precomputed HR/Speed data, smoothed signals, and cluster assignments

## Breaking Changes

None - this is a purely additive feature.

## Migration Notes

- Existing `activities_metrics.csv` files will be automatically migrated to include the `hrSpeedShift` column
- `metrics_ts` files will be generated on-demand when viewing the HR vs Speed graph
- Full reconstruction available via Settings page → "Reconstruire les activités depuis le cache Strava"

## Testing

- Handles missing GPS data gracefully (falls back to `paceKmh`)
- Handles activities without timeseries data
- Validates cluster size thresholds before including in visualization
- Cache invalidation works correctly with Streamlit's `@st.cache_data`

## Performance Considerations

- Precomputed `metrics_ts` files avoid reprocessing on every dashboard load
- Streamlit caching (`ttl=3600`) reduces computation for repeated views
- Clustering is performed per activity, then aggregated for visualization

## Future Enhancements

- Profile-based analysis method (currently only cluster-based is implemented)
- Interactive cluster selection/removal
- Time-based filtering (e.g., show only recent activities)
- Export functionality for regression parameters

