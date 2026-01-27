# Graph

Visualization components using Altair and Matplotlib.

## Files

| File | Purpose |
|------|---------|
| `training_load.py` | Acute/chronic training load charts |
| `analytics.py` | Planned vs actual bar charts |
| `elevation.py` | Elevation profile with grade coloring |
| `timeseries.py` | Activity timeseries (HR, speed, elevation, SpeedEq) |
| `hr_speed.py` | HR vs Speed scatter with regression |
| `speed_scatter.py` | SpeedEq scatter visualization |
| `speed_profile.py` | Speed profile charts |
| `pacer_comparison.py` | Race pacing comparison charts |
| `pacer_segments.py` | Race segment visualization |

## Key Visualizations

### training_load.py
- `render_training_load_chart(df)`: Acute/chronic load time series
- Altair-based with tooltips

### analytics.py
- `render_planned_vs_actual_chart(df, metric)`: Stacked bars
- Metrics: Distance, DistEq, Time, TRIMP
- Weekly/daily granularity

### elevation.py
- `render_elevation_profile(df)`: Colored by grade category
- `render_grade_histogram(df)`: Grade distribution
- Grade categories: steep descent → steep ascent

Grade color mapping:
```python
GRADE_COLOR_MAPPING = {
    "grade_lt_neg_0_5": "#001f3f",   # steep descent
    "grade_lt_neg_0_25": "#004d26",
    "grade_lt_neg_0_05": "#22c55e",
    "grade_neutral": "#d1d5db",
    "grade_lt_0_1": "#eab308",
    "grade_lt_0_25": "#f97316",
    "grade_lt_0_5": "#dc2626",
    "grade_ge_0_5": "#000000",       # steep ascent
}
```

### timeseries.py
- `render_timeseries_charts(ts_service, activity_id, speed_profile_service=None)`: Dict of charts
- Charts: HR, speed, elevation, SpeedEq (10s smoothing)

### hr_speed.py
- `render_hr_speed_scatter(df)`: Weighted regression scatter
- Shows HR vs Speed relationship

## Chart Patterns

Common structure:
```python
def render_chart(df: pd.DataFrame) -> alt.Chart:
    # Validate data
    if df.empty:
        return None
    # Build chart
    chart = alt.Chart(df).mark_*().encode(...)
    return chart
```

Display in Streamlit:
```python
chart = render_chart(df)
if chart:
    st.altair_chart(chart, use_container_width=True)
```

## Related Files

- `pages/Dashboard.py`: Uses training_load, speed_scatter
- `pages/Analytics.py`: Uses analytics charts
- `pages/Activity.py`: Uses elevation, timeseries
- `utils/segments.py`: Segment merging utilities
- `utils/elevation_preprocessing.py`: Data preparation

## Maintaining This File

Update when:
- Adding new chart types
- Changing color schemes
- Adding new visualization patterns
- Modifying chart APIs
