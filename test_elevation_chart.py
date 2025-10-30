#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, '.')

import streamlit as st
import pandas as pd
from services.speed_profile_service import SpeedProfileService
import altair as alt

st.set_page_config(page_title="Test Elevation Chart")

@st.cache_data
def load_and_process():
    timeseries_folder = "data/timeseries"
    files = os.listdir(timeseries_folder)
    first_file = os.path.join(timeseries_folder, files[0])
    df = pd.read_csv(first_file)
    
    sp = SpeedProfileService(config=None)
    metrics_df = sp.preprocess_timeseries(df)
    metrics_df = sp.moving_average(metrics_df, window_size=10, col="grade")
    return metrics_df

st.title("Test Elevation Profile Visualization")

metrics_df = load_and_process()

# Prepare data
plot_df = metrics_df[['cumulated_distance', 'elevationM_ma_5', 'grade_ma_10', 'speed_km_h', 'hr']].copy()
plot_df = plot_df.dropna(subset=['cumulated_distance', 'elevationM_ma_5'])
if 'grade_ma_10' in plot_df.columns:
    plot_df['grade_ma_10'] = plot_df['grade_ma_10'].fillna(0)
plot_df = plot_df.reset_index(drop=True)

st.write(f"Data shape: {plot_df.shape}")
st.write(f"Grade range: {plot_df['grade_ma_10'].min():.4f} to {plot_df['grade_ma_10'].max():.4f}")

# Format values
plot_df['speed_display'] = plot_df['speed_km_h'].fillna(0).round(1).astype(str) + ' km/h'
plot_df['hr_display'] = plot_df['hr'].fillna(0).round(0).astype(str) + ' bpm'
plot_df['grade_display'] = (plot_df['grade_ma_10'] * 100).round(1).astype(str) + '%'
plot_df['distance_display'] = plot_df['cumulated_distance'].round(2).astype(str) + ' km'
plot_df['elevation_display'] = plot_df['elevationM_ma_5'].round(0).astype(str) + ' m'

# Create color scale
color_scale = alt.Scale(
    domain=[-0.5, -0.25, -0.05, 0.05, 0.1, 0.25, 0.5],
    range=['#001f3f', '#004d26', '#22c55e', '#d1d5db', '#eab308', '#f97316', '#dc2626'],
    clamp=True
)

# Create area chart
area = alt.Chart(plot_df).mark_area(opacity=0.4, interpolate='monotone').encode(
    x=alt.X('cumulated_distance:Q', title='Distance (km)', scale=alt.Scale(nice=True)),
    y=alt.Y('elevationM_ma_5:Q', title='Altitude (m)', scale=alt.Scale(nice=True)),
    color=alt.Color('grade_ma_10:Q', scale=color_scale, title='Pente')
)

# Create line chart
line = alt.Chart(plot_df).mark_line(point=True, interpolate='monotone').encode(
    x=alt.X('cumulated_distance:Q', title='Distance (km)', scale=alt.Scale(nice=True)),
    y=alt.Y('elevationM_ma_5:Q', title='Altitude (m)', scale=alt.Scale(nice=True)),
    color=alt.Color('grade_ma_10:Q', scale=color_scale, title='Pente', legend=None),
    tooltip=[
        alt.Tooltip('distance_display:N', title='Distance'),
        alt.Tooltip('elevation_display:N', title='Altitude'),
        alt.Tooltip('grade_display:N', title='Pente'),
        alt.Tooltip('speed_display:N', title='Vitesse'),
        alt.Tooltip('hr_display:N', title='FC'),
    ]
).properties(width=800, height=400).interactive()

combined = (area + line).properties(
    title='Elevation Profile with Grade-Based Coloring'
).resolve_scale(color='shared')

st.altair_chart(combined, use_container_width=True)

st.success("Chart rendered successfully!")
