"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later

Trail running digital-twin helpers used by the research notebook.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from utils.redi import compute_redi

logger = logging.getLogger(__name__)

MINETTI_RUNNING_FLAT_COST = 3.6
MINETTI_GRADE_CLAMP = 0.75
GRADE_OUTLIER_ABS_THRESHOLD = 1.0
EARTH_RADIUS_M = 6_371_000.0
SEGMENT_GRADE_SHARE_THRESHOLD = 0.03
MIXED_CLIMB_DESCENT_MIN_SHARE = 0.25
DEFAULT_STATIONARY_SPEED_KMH = 1.0
DEFAULT_MIN_MEAN_SPEED_EQ_KMH = 3.0
DEFAULT_MAX_STATIONARY_TIME_SHARE = 0.40
# Near-flat on the altitude–time profile: gross |Δelev| per clock hour.
# (Not distance-grade: slow climbs stay non-flat even when |avgGrade| is modest.)
DEFAULT_MAX_ABS_ALTITUDE_RATE_MPH = 120.0
# Deprecated distance-grade gate (kept only for backward-compatible kwargs).
DEFAULT_MAX_ABS_GRADE_FOR_EXCLUSION = 0.05
DEFAULT_FORBIDDEN_ANONYMIZED_COLUMNS = frozenset(
    {
        "activityid",
        "stravaid",
        "id",
        "name",
        "starttime",
        "startdate",
        "date",
        "timestamp",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "coordinates",
        "polyline",
    }
)


@dataclass(frozen=True)
class RegressionModel:
    """Small linear model container for notebook experiments."""

    feature_cols: tuple[str, ...]
    coefficients: np.ndarray


def _to_float(value: object, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def minetti_running_cost(grade: float) -> float:
    """Minetti running energy-cost polynomial clamped to the benchmark grade range."""
    i = max(-MINETTI_GRADE_CLAMP, min(MINETTI_GRADE_CLAMP, _to_float(grade, 0.0)))
    cost = 155.4 * i**5 - 30.4 * i**4 - 43.3 * i**3 + 46.3 * i**2 + 19.5 * i + 3.6
    return max(0.1 * MINETTI_RUNNING_FLAT_COST, cost)


def gap_factor(grade: float) -> float:
    """Grade-adjustment cost factor with f_gap(0) = 1."""
    return minetti_running_cost(grade) / MINETTI_RUNNING_FLAT_COST


DEFAULT_GAP_STEEP_THRESHOLD = 0.15
DEFAULT_GAP_SOFT_START = 0.04
DEFAULT_GAP_CLIMB_SCALE = 1.0
DEFAULT_GAP_DESCENT_SCALE = 1.0


def trail_gap_multiplier(
    grade: float,
    *,
    steep_threshold: float = DEFAULT_GAP_STEEP_THRESHOLD,
    soft_start: float = DEFAULT_GAP_SOFT_START,
    climb_scale: float = DEFAULT_GAP_CLIMB_SCALE,
    descent_scale: float = DEFAULT_GAP_DESCENT_SCALE,
) -> float:
    """Asymmetric trail correction on top of Minetti running GAP.

    Scales ramp from 1.0 at ``±soft_start`` to the full climb/descent scale at
    ``±steep_threshold``, then stay constant beyond that. Defaults keep Minetti
    unchanged (scales = 1.0).
    """
    g = _to_float(grade, 0.0)
    thresh = abs(float(steep_threshold))
    soft = max(0.0, min(abs(float(soft_start)), thresh - 1e-6))
    climb = max(0.1, float(climb_scale))
    descent = max(0.1, float(descent_scale))
    if g >= soft:
        if climb == 1.0:
            return 1.0
        span = max(thresh - soft, 1e-9)
        t = min(1.0, max(0.0, (g - soft) / span))
        return 1.0 + t * (climb - 1.0)
    if g <= -soft:
        if descent == 1.0:
            return 1.0
        span = max(thresh - soft, 1e-9)
        t = min(1.0, max(0.0, (-g - soft) / span))
        return 1.0 + t * (descent - 1.0)
    return 1.0


def apply_trail_gap_multipliers(
    gap: np.ndarray | pd.Series,
    grades: np.ndarray | pd.Series,
    *,
    steep_threshold: float = DEFAULT_GAP_STEEP_THRESHOLD,
    soft_start: float = DEFAULT_GAP_SOFT_START,
    climb_scale: float = DEFAULT_GAP_CLIMB_SCALE,
    descent_scale: float = DEFAULT_GAP_DESCENT_SCALE,
) -> np.ndarray:
    """Element-wise trail GAP scaling for segment arrays."""
    gap_arr = np.asarray(gap, dtype=float)
    grade_arr = np.asarray(pd.to_numeric(pd.Series(grades), errors="coerce").fillna(0.0), dtype=float)
    if gap_arr.shape != grade_arr.shape:
        raise ValueError("gap and grades must have the same shape")
    multipliers = np.array(
        [
            trail_gap_multiplier(
                float(g),
                steep_threshold=steep_threshold,
                soft_start=soft_start,
                climb_scale=climb_scale,
                descent_scale=descent_scale,
            )
            for g in grade_arr
        ],
        dtype=float,
    )
    return np.clip(gap_arr * multipliers, 0.1, None)


def altitude_factor(altitude_m: float) -> float:
    """Altitude-VO2max correction from the Sensors trail digital-twin paper."""
    altitude = max(0.0, _to_float(altitude_m, 0.0))
    return max(0.1, 1.0 - 11.7e-9 * altitude**2 - 4.01e-6 * altitude)


def linear_decay_factor(progress: float, mu: float) -> float:
    """Within-race linear pacing-decay factor."""
    s = max(0.0, min(1.0, _to_float(progress, 0.0)))
    return max(0.1, 1.0 + float(mu) * s)


def exponential_decay_factor(progress: float, decay_lambda: float) -> float:
    """Within-race exponential pacing-decay factor."""
    s = max(0.0, min(1.0, _to_float(progress, 0.0)))
    return max(0.1, math.exp(float(decay_lambda) * s))


def ewma_from_tau(values: Sequence[float], tau_days: float) -> np.ndarray:
    """Compute an EWMA using a time constant expressed in days."""
    if tau_days <= 0:
        raise ValueError("tau_days must be positive")
    alpha = 1.0 - math.exp(-1.0 / tau_days)
    array = np.asarray(values, dtype=float)
    result = np.full_like(array, np.nan, dtype=float)
    valid = np.flatnonzero(np.isfinite(array))
    if valid.size == 0:
        return result
    start = int(valid[0])
    result[start] = float(array[start])
    for idx in range(start + 1, array.size):
        value = result[idx - 1] if not np.isfinite(array[idx]) else float(array[idx])
        result[idx] = alpha * value + (1.0 - alpha) * result[idx - 1]
    return result


def compute_ctl_atl_tsb(
    daily_df: pd.DataFrame,
    load_col: str = "trimp",
    ctl_tau_days: float = 42.0,
    atl_tau_days: float = 7.0,
) -> pd.DataFrame:
    """Return daily CTL, ATL, and TSB from a load column."""
    if daily_df.empty:
        return pd.DataFrame(columns=["date", "load", "ctl", "atl", "tsb"])
    working = daily_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    values = pd.to_numeric(working.get(load_col), errors="coerce").fillna(0.0).to_numpy()
    working["load"] = values
    working["ctl"] = ewma_from_tau(values, ctl_tau_days)
    working["atl"] = ewma_from_tau(values, atl_tau_days)
    working["tsb"] = working["ctl"] - working["atl"]
    return working[["date", "load", "ctl", "atl", "tsb"]]


def ctl_readiness_factor(
    ctl: float,
    tsb: float,
    ctl_reference: float,
    ctl_weight: float = 0.05,
    tsb_weight: float = 0.10,
    min_factor: float = 0.90,
    max_factor: float = 1.08,
) -> float:
    """Bounded paper-style readiness multiplier inferred from CTL and TSB.

    The Sensors paper uses an f_CTL term based on 42-day CTL and race-day TSB, but it
    does not publish a numeric formula. This helper keeps that assumption explicit.
    """
    reference = _to_float(ctl_reference, np.nan)
    if not math.isfinite(reference) or reference <= 0.0:
        return 1.0

    ctl_value = _to_float(ctl, reference)
    tsb_value = _to_float(tsb, 0.0)
    relative_ctl = (ctl_value - reference) / reference
    relative_tsb = tsb_value / reference
    factor = 1.0 + float(ctl_weight) * relative_ctl + float(tsb_weight) * relative_tsb
    return max(float(min_factor), min(float(max_factor), factor))


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid = numeric.notna() & (numeric_weights > 0)
    if not valid.any():
        return float(numeric.mean()) if numeric.notna().any() else np.nan
    return float(np.average(numeric[valid], weights=numeric_weights[valid]))


def _weighted_std(values: pd.Series, weights: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0.0)
    valid = numeric.notna() & (numeric_weights > 0)
    if not valid.any():
        return float(numeric.std(ddof=0)) if numeric.notna().any() else np.nan
    mean = float(np.average(numeric[valid], weights=numeric_weights[valid]))
    variance = float(np.average((numeric[valid] - mean) ** 2, weights=numeric_weights[valid]))
    return math.sqrt(max(0.0, variance))


def _gap_factors_for_segments(
    segments_df: pd.DataFrame,
    *,
    steep_threshold: float = DEFAULT_GAP_STEEP_THRESHOLD,
    soft_start: float = DEFAULT_GAP_SOFT_START,
    climb_scale: float = DEFAULT_GAP_CLIMB_SCALE,
    descent_scale: float = DEFAULT_GAP_DESCENT_SCALE,
) -> pd.Series:
    grades = pd.to_numeric(
        segments_df.get("avgGrade", pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)
    fallback = grades.map(gap_factor)
    for col in ("gapFactorIntegrated", "gapFactor"):
        if col in segments_df.columns:
            gap_values = pd.to_numeric(segments_df[col], errors="coerce")
            base = gap_values.where(gap_values > 0.0).fillna(fallback).clip(lower=0.1)
            break
    else:
        base = fallback.clip(lower=0.1)
    if climb_scale == 1.0 and descent_scale == 1.0:
        return base
    scaled = apply_trail_gap_multipliers(
        base.to_numpy(dtype=float),
        grades.to_numpy(dtype=float),
        steep_threshold=steep_threshold,
        soft_start=soft_start,
        climb_scale=climb_scale,
        descent_scale=descent_scale,
    )
    return pd.Series(scaled, index=segments_df.index)


def gps_technicality_index(latitudes: Sequence[float], longitudes: Sequence[float]) -> float:
    """Estimate route technicality from perpendicular GPS variance."""
    coords = pd.DataFrame({"lat": latitudes, "lon": longitudes}).dropna()
    if len(coords) < 3:
        return 0.0

    lat = np.radians(coords["lat"].to_numpy(dtype=float))
    lon = np.radians(coords["lon"].to_numpy(dtype=float))
    lat0 = float(lat[0])
    lon0 = float(lon[0])
    mean_lat = float(np.nanmean(lat))

    x = EARTH_RADIUS_M * np.cos(mean_lat) * (lon - lon0)
    y = EARTH_RADIUS_M * (lat - lat0)
    points = np.column_stack([x, y])
    direction = points[-1] - points[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1.0:
        return 0.0

    rel = points - points[0]
    residuals = np.abs(direction[0] * rel[:, 1] - direction[1] * rel[:, 0]) / norm
    return float(max(0.0, min(1.0, np.nanstd(residuals) / 50.0)))


def prepare_raw_timeseries_for_segments(
    raw_df: pd.DataFrame,
    max_speed_kmh: float = 40.0,
    elevation_window: int = 30,
    grade_window: int = 10,
) -> pd.DataFrame:
    """Prepare raw Strava/Garmin streams for distance-based segment aggregation.

    This is intentionally narrower and faster than the app's full speed-profile pipeline.
    It keeps the paper-relevant fields: cumulative distance/time, 30-s median elevation,
    grade, HR, speed, and GPS coordinates.
    """
    if raw_df.empty or "lat" not in raw_df.columns or "lon" not in raw_df.columns:
        return pd.DataFrame()

    df = raw_df.copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).reset_index(drop=True)
    if len(df) < 2:
        return pd.DataFrame()

    has_real_timestamps = False
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df = df.sort_values("timestamp", na_position="last").reset_index(drop=True)
        duration = df["timestamp"].diff().dt.total_seconds()
        duration = duration.fillna(0.0).clip(lower=0.0, upper=120.0)
        has_real_timestamps = bool(df["timestamp"].notna().sum() > 1)
    else:
        duration = pd.Series(np.r_[0.0, np.ones(len(df) - 1)], index=df.index)

    lat_rad = np.radians(df["lat"].to_numpy(dtype=float))
    lon_rad = np.radians(df["lon"].to_numpy(dtype=float))
    dlat = np.diff(lat_rad, prepend=lat_rad[0])
    dlon = np.diff(lon_rad, prepend=lon_rad[0])
    haversine_a = np.sin(dlat / 2.0) ** 2 + np.cos(lat_rad) * np.cos(
        np.r_[lat_rad[0], lat_rad[:-1]]
    ) * np.sin(dlon / 2.0) ** 2
    step_distance_km = (
        2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(haversine_a, 0.0, 1.0))) / 1000.0
    )
    step_distance_km[0] = 0.0

    duration_values = duration.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        step_speed_kmh = step_distance_km / duration_values * 3600.0
    step_speed_kmh[~np.isfinite(step_speed_kmh)] = 0.0
    if has_real_timestamps:
        valid_speed = (step_speed_kmh <= max_speed_kmh) | (step_distance_km == 0.0)
    else:
        valid_speed = np.ones(len(step_distance_km), dtype=bool)
    step_distance_km = np.where(valid_speed, step_distance_km, 0.0)
    step_speed_kmh = np.where(valid_speed, step_speed_kmh, 0.0)

    if "elevationM" in df.columns:
        elevation = pd.to_numeric(df["elevationM"], errors="coerce").interpolate(
            limit_direction="both"
        )
    else:
        elevation = pd.Series(np.nan, index=df.index)
    smoothed_elevation = elevation.rolling(
        window=max(1, int(elevation_window)),
        min_periods=1,
        center=True,
    ).median()
    elevation_difference = smoothed_elevation.diff().fillna(0.0)

    with np.errstate(divide="ignore", invalid="ignore"):
        grade = elevation_difference.to_numpy(dtype=float) / (step_distance_km * 1000.0)
    grade[~np.isfinite(grade)] = np.nan
    grade[np.abs(grade) > GRADE_OUTLIER_ABS_THRESHOLD] = np.nan
    grade = (
        pd.Series(grade)
        .interpolate(method="linear", limit_direction="both")
        .fillna(0.0)
        .to_numpy(dtype=float)
    )
    grade = np.clip(grade, -MINETTI_GRADE_CLAMP, MINETTI_GRADE_CLAMP)
    grade[~np.isfinite(grade)] = 0.0
    grade_smooth = (
        pd.Series(grade)
        .rolling(window=max(1, int(grade_window)), min_periods=1, center=True)
        .median()
        .to_numpy(dtype=float)
    )

    prepared = pd.DataFrame(
        {
            "lat": df["lat"].to_numpy(dtype=float),
            "lon": df["lon"].to_numpy(dtype=float),
            "distance": step_distance_km,
            "cumulated_distance": np.cumsum(step_distance_km),
            "cumulated_duration_seconds": np.cumsum(duration_values),
            "elevationM_ma_5": smoothed_elevation.to_numpy(dtype=float),
            "elevation_difference": elevation_difference.to_numpy(dtype=float),
            "grade_ma_10": grade_smooth,
            "speed_km_h": step_speed_kmh,
        }
    )
    if "hr" in df.columns:
        prepared["hr"] = pd.to_numeric(df["hr"], errors="coerce")
    elif "hr_smooth" in df.columns:
        prepared["hr_smooth"] = pd.to_numeric(df["hr_smooth"], errors="coerce")
    if "paceKmh" in df.columns:
        prepared["paceKmh"] = pd.to_numeric(df["paceKmh"], errors="coerce")

    return prepared[prepared["cumulated_distance"] > 0].reset_index(drop=True)


def segment_timeseries(
    ts_df: pd.DataFrame,
    segment_km: float = 1.0,
    hr_rest: Optional[float] = None,
    hr_max: Optional[float] = None,
    min_distance_km: float = 0.05,
) -> pd.DataFrame:
    """Aggregate a processed activity timeseries into distance-based segments."""
    if ts_df.empty or "cumulated_distance" not in ts_df.columns:
        return pd.DataFrame()

    df = ts_df.copy()
    df["cumulated_distance"] = pd.to_numeric(df["cumulated_distance"], errors="coerce")
    df = df.dropna(subset=["cumulated_distance"]).sort_values("cumulated_distance")
    df = df.reset_index(drop=True)
    if df.empty:
        return pd.DataFrame()

    if "cumulated_duration_seconds" in df.columns:
        df["cumulated_duration_seconds"] = pd.to_numeric(
            df["cumulated_duration_seconds"], errors="coerce"
        )
    else:
        df["cumulated_duration_seconds"] = np.arange(len(df), dtype=float)

    df["prev_distance"] = df["cumulated_distance"].shift(fill_value=0.0)
    df["prev_duration"] = df["cumulated_duration_seconds"].shift(fill_value=0.0)
    df["delta_km"] = (df["cumulated_distance"] - df["prev_distance"]).clip(lower=0.0)
    df["delta_time_sec"] = (df["cumulated_duration_seconds"] - df["prev_duration"]).clip(
        lower=0.0
    )
    if "elevationM_ma_5" in df.columns:
        elevation_col = "elevationM_ma_5"
    elif "elevationM" in df.columns:
        elevation_col = "elevationM"
    else:
        elevation_col = ""
        df["elevation_difference"] = 0.0

    if elevation_col:
        df[elevation_col] = pd.to_numeric(df[elevation_col], errors="coerce")
        df["elevation_difference"] = df[elevation_col].diff().fillna(0.0)

    # Keep zero-distance rows so device-open / aid-station dwell time is attributed to
    # the current distance segment instead of being dropped before aggregation.
    if df["delta_km"].gt(0).sum() == 0:
        return pd.DataFrame()

    total_distance = float(df["cumulated_distance"].max())
    df["segmentIndex"] = np.floor(df["prev_distance"] / max(float(segment_km), 1e-9)).astype(int)

    rows: list[dict[str, object]] = []
    for seg_idx, seg_df in df.groupby("segmentIndex", sort=True):
        moving_df = seg_df[seg_df["delta_km"] > 0]
        distance_km = float(moving_df["delta_km"].sum()) if not moving_df.empty else 0.0
        if distance_km < min_distance_km:
            continue
        time_sec = float(seg_df["delta_time_sec"].sum())
        elev_source = moving_df if not moving_df.empty else seg_df
        elev_diff = pd.to_numeric(elev_source["elevation_difference"], errors="coerce").fillna(0.0)
        gain_m = float(elev_diff.clip(lower=0.0).sum())
        loss_m = float((-elev_diff.clip(upper=0.0)).sum())
        start_km = float(seg_df["prev_distance"].min())
        end_km = float(seg_df["cumulated_distance"].max())
        progress = (start_km + end_km) / max(2.0 * total_distance, 1e-9)
        grade_col = "grade_ma_10" if "grade_ma_10" in elev_source.columns else None
        net_grade = (gain_m - loss_m) / (distance_km * 1000.0) if distance_km > 0 else np.nan
        grade_weights = pd.to_numeric(elev_source["delta_km"], errors="coerce").fillna(0.0)
        if grade_col:
            local_grade = pd.to_numeric(elev_source[grade_col], errors="coerce")
            local_grade = local_grade.mask(local_grade.abs() > GRADE_OUTLIER_ABS_THRESHOLD)
            local_grade = local_grade.interpolate(method="linear", limit_direction="both")
            avg_grade = _weighted_mean(local_grade, grade_weights)
        else:
            local_grade = pd.Series(np.nan, index=elev_source.index)
            avg_grade = np.nan
        fallback_grade = avg_grade if pd.notna(avg_grade) else net_grade
        local_grade = local_grade.fillna(fallback_grade if pd.notna(fallback_grade) else 0.0)
        local_grade = local_grade.clip(-MINETTI_GRADE_CLAMP, MINETTI_GRADE_CLAMP)
        gap_integrated = _weighted_mean(local_grade.map(gap_factor), grade_weights)
        gap_avg_grade = gap_factor(fallback_grade if pd.notna(fallback_grade) else 0.0)
        abs_grade_mean = _weighted_mean(local_grade.abs(), grade_weights)
        grade_std = _weighted_std(local_grade, grade_weights)
        climb_mask = local_grade > SEGMENT_GRADE_SHARE_THRESHOLD
        descent_mask = local_grade < -SEGMENT_GRADE_SHARE_THRESHOLD
        climb_distance = float(grade_weights[climb_mask].sum())
        descent_distance = float(grade_weights[descent_mask].sum())
        flat_distance = max(0.0, distance_km - climb_distance - descent_distance)
        climb_share = climb_distance / distance_km if distance_km > 0 else np.nan
        descent_share = descent_distance / distance_km if distance_km > 0 else np.nan
        flat_share = flat_distance / distance_km if distance_km > 0 else np.nan
        grade_sign = np.where(climb_mask, 1, np.where(descent_mask, -1, 0))
        grade_sign = grade_sign[grade_sign != 0]
        grade_switch_count = int(np.sum(grade_sign[1:] != grade_sign[:-1])) if len(grade_sign) > 1 else 0
        is_mixed_climb_descent = bool(
            climb_share >= MIXED_CLIMB_DESCENT_MIN_SHARE
            and descent_share >= MIXED_CLIMB_DESCENT_MIN_SHARE
        )
        segment_terrain = (
            "mixed_climb_descent"
            if is_mixed_climb_descent
            else terrain_family(fallback_grade if pd.notna(fallback_grade) else 0.0)
        )

        mean_altitude = (
            _weighted_mean(elev_source[elevation_col], elev_source["delta_km"]) if elevation_col else np.nan
        )
        mean_hr = _weighted_mean(seg_df["hr"], seg_df["delta_time_sec"]) if "hr" in seg_df else np.nan
        if pd.isna(mean_hr) and "hr_smooth" in seg_df:
            mean_hr = _weighted_mean(seg_df["hr_smooth"], seg_df["delta_time_sec"])

        mean_hr_reserve = np.nan
        if hr_rest is not None and hr_max is not None and hr_max > hr_rest and pd.notna(mean_hr):
            mean_hr_reserve = max(0.0, min(1.2, (mean_hr - hr_rest) / (hr_max - hr_rest)))

        if "speed_km_h" in seg_df.columns:
            mean_speed = _weighted_mean(seg_df["speed_km_h"], seg_df["delta_time_sec"])
        elif "paceKmh" in seg_df.columns:
            mean_speed = _weighted_mean(seg_df["paceKmh"], seg_df["delta_time_sec"])
        else:
            mean_speed = distance_km / time_sec * 3600.0 if time_sec > 0 else np.nan

        if "speedeq_smooth" in seg_df.columns:
            mean_speed_eq = _weighted_mean(seg_df["speedeq_smooth"], seg_df["delta_time_sec"])
        elif "speed_eq_km_h" in seg_df.columns:
            mean_speed_eq = _weighted_mean(seg_df["speed_eq_km_h"], seg_df["delta_time_sec"])
        else:
            mean_speed_eq = mean_speed

        delta_time = pd.to_numeric(seg_df["delta_time_sec"], errors="coerce").fillna(0.0)
        delta_km = pd.to_numeric(seg_df["delta_km"], errors="coerce").fillna(0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            instant_speed = np.where(delta_time > 0.0, delta_km / delta_time * 3600.0, 0.0)
        stationary_mask = (delta_km <= 0.0) | (instant_speed < DEFAULT_STATIONARY_SPEED_KMH)
        stationary_time_sec = float(delta_time.to_numpy(dtype=float)[stationary_mask].sum())
        stationary_time_share = stationary_time_sec / time_sec if time_sec > 0 else 0.0
        # Moving clock for Stage 3 fit: strip device-open / aid-station dwell.
        actual_moving_time_sec = max(0.0, time_sec - stationary_time_sec)

        # Altitude-over-time flatness (full segment clock, including dwell).
        # Distinct from avgGrade / netGrade which are altitude-over-distance.
        full_elev_diff = pd.to_numeric(seg_df["elevation_difference"], errors="coerce").fillna(0.0)
        full_gain_m = float(full_elev_diff.clip(lower=0.0).sum())
        full_loss_m = float((-full_elev_diff.clip(upper=0.0)).sum())
        abs_altitude_rate_mph = (
            (full_gain_m + full_loss_m) / time_sec * 3600.0 if time_sec > 0 else np.nan
        )
        net_altitude_rate_mph = (
            (full_gain_m - full_loss_m) / time_sec * 3600.0 if time_sec > 0 else np.nan
        )

        technicality = 0.0
        if "lat" in elev_source.columns and "lon" in elev_source.columns:
            technicality = gps_technicality_index(elev_source["lat"], elev_source["lon"])

        rows.append(
            {
                "segmentIndex": int(seg_idx),
                "startKm": start_km,
                "endKm": end_km,
                "distanceKm": distance_km,
                "actualTimeSec": time_sec,
                "elevGainM": gain_m,
                "elevLossM": loss_m,
                "netElevM": gain_m - loss_m,
                "meanAltitudeM": mean_altitude,
                "avgGrade": avg_grade if pd.notna(avg_grade) else net_grade,
                "netGrade": net_grade,
                "gapFactorIntegrated": gap_integrated,
                "gapFactorAvgGrade": gap_avg_grade,
                "absGradeMean": abs_grade_mean,
                "gradeStd": grade_std,
                "climbDistanceKm": climb_distance,
                "descentDistanceKm": descent_distance,
                "flatDistanceKm": flat_distance,
                "climbShare": climb_share,
                "descentShare": descent_share,
                "flatShare": flat_share,
                "gradeSwitchCount": grade_switch_count,
                "isMixedClimbDescent": is_mixed_climb_descent,
                "terrainFamily": segment_terrain,
                "technicalityGps": technicality,
                "meanHr": mean_hr,
                "meanHrReserve": mean_hr_reserve,
                "meanSpeedKmh": mean_speed,
                "meanSpeedEqKmh": mean_speed_eq,
                "stationaryTimeShare": stationary_time_share,
                "stationaryTimeSec": stationary_time_sec,
                "actualMovingTimeSec": actual_moving_time_sec,
                "absAltitudeRateMph": abs_altitude_rate_mph,
                "netAltitudeRateMph": net_altitude_rate_mph,
                "progress": progress,
            }
        )

    return pd.DataFrame(rows)


def apply_segment_exclusion(
    segments_df: pd.DataFrame,
    *,
    enabled: bool = True,
    min_mean_speed_eq_kmh: float = DEFAULT_MIN_MEAN_SPEED_EQ_KMH,
    max_stationary_time_share: float = DEFAULT_MAX_STATIONARY_TIME_SHARE,
    stationary_speed_kmh: float = DEFAULT_STATIONARY_SPEED_KMH,
    max_abs_altitude_rate_mph: float = DEFAULT_MAX_ABS_ALTITUDE_RATE_MPH,
    max_abs_grade: Optional[float] = None,
    min_mean_speed_kmh: Optional[float] = None,
) -> pd.DataFrame:
    """Flag immobile segments that are flat on the altitude–time profile.

    A segment is excluded only when **all** of the following hold:
    - altitude-over-time is flat / almost flat
      (``absAltitudeRateMph <= max_abs_altitude_rate_mph``, gross |Δelev|/hour)
    - AND there is immobility evidence: low grade-adjusted ``meanSpeedEqKmh``
      and/or high ``stationaryTimeShare`` (device-open / aid-station dwell)

    Distance-based grade (``avgGrade``) is intentionally not used: a slow climb can
    look mild per km while rising clearly on the altitude–time chart. Excluded
    segments remain in the frame for full-race evaluation.
    """
    if segments_df.empty:
        return segments_df.copy()

    # Backward-compatible alias from earlier raw-speed configs.
    if min_mean_speed_kmh is not None:
        logger.warning(
            "apply_segment_exclusion: min_mean_speed_kmh is deprecated; "
            "using it as min_mean_speed_eq_kmh (grade-adjusted)"
        )
        min_mean_speed_eq_kmh = float(min_mean_speed_kmh)

    if max_abs_grade is not None:
        logger.warning(
            "apply_segment_exclusion: max_abs_grade is deprecated; "
            "flatness uses absAltitudeRateMph (altitude over time), not avgGrade. "
            "Ignoring max_abs_grade=%s; using max_abs_altitude_rate_mph=%s",
            max_abs_grade,
            max_abs_altitude_rate_mph,
        )

    out = segments_df.copy()
    speed_eq = pd.to_numeric(out.get("meanSpeedEqKmh"), errors="coerce")
    if speed_eq.isna().all():
        raw_speed = pd.to_numeric(out.get("meanSpeedKmh"), errors="coerce")
        if not raw_speed.isna().all():
            logger.warning(
                "apply_segment_exclusion: meanSpeedEqKmh missing; "
                "falling back to meanSpeedKmh (steep climbs may be over-excluded)"
            )
            speed_eq = raw_speed
            out["meanSpeedEqKmh"] = speed_eq
        elif {"distanceKm", "actualTimeSec"}.issubset(out.columns):
            logger.warning(
                "apply_segment_exclusion: speed columns missing; "
                "falling back to distance/time speed"
            )
            distance = pd.to_numeric(out["distanceKm"], errors="coerce")
            time_sec = pd.to_numeric(out["actualTimeSec"], errors="coerce")
            speed_eq = pd.Series(
                np.where(
                    (time_sec > 0) & np.isfinite(distance) & np.isfinite(time_sec),
                    distance / time_sec * 3600.0,
                    np.nan,
                ),
                index=out.index,
            )
            out["meanSpeedEqKmh"] = speed_eq
        else:
            speed_eq = pd.Series(np.nan, index=out.index)

    if "stationaryTimeShare" in out.columns:
        stationary_share = pd.to_numeric(out["stationaryTimeShare"], errors="coerce")
    else:
        logger.warning(
            "apply_segment_exclusion: stationaryTimeShare missing; "
            "falling back to speed-eq only"
        )
        stationary_share = pd.Series(0.0, index=out.index)
        out["stationaryTimeShare"] = stationary_share

    if "absAltitudeRateMph" in out.columns:
        altitude_rate = pd.to_numeric(out["absAltitudeRateMph"], errors="coerce")
    elif {"elevGainM", "elevLossM", "actualTimeSec"}.issubset(out.columns):
        logger.warning(
            "apply_segment_exclusion: absAltitudeRateMph missing; "
            "falling back to (elevGainM+elevLossM)/actualTimeSec"
        )
        gain = pd.to_numeric(out["elevGainM"], errors="coerce").fillna(0.0)
        loss = pd.to_numeric(out["elevLossM"], errors="coerce").fillna(0.0)
        time_sec = pd.to_numeric(out["actualTimeSec"], errors="coerce")
        altitude_rate = pd.Series(
            np.where(
                (time_sec > 0) & np.isfinite(time_sec),
                (gain + loss) / time_sec * 3600.0,
                np.nan,
            ),
            index=out.index,
        )
        out["absAltitudeRateMph"] = altitude_rate
    else:
        logger.warning(
            "apply_segment_exclusion: altitude-rate columns missing; "
            "falling back to treating all segments as flat on altitude–time"
        )
        altitude_rate = pd.Series(0.0, index=out.index)
        out["absAltitudeRateMph"] = altitude_rate

    reasons: list[str] = []
    eligible: list[bool] = []
    for idx in out.index:
        reason_parts: list[str] = []
        speed_val = speed_eq.loc[idx]
        share_val = stationary_share.loc[idx]
        rate_val = altitude_rate.loc[idx]
        if enabled:
            near_flat_time = pd.notna(rate_val) and float(rate_val) <= float(
                max_abs_altitude_rate_mph
            )
            low_speed = pd.notna(speed_val) and float(speed_val) < float(min_mean_speed_eq_kmh)
            high_share = pd.notna(share_val) and float(share_val) > float(max_stationary_time_share)
            # Immobile only when altitude-over-time profile is flat / almost flat.
            if near_flat_time and (low_speed or high_share):
                reason_parts.append("near_flat_altitude_time")
                if low_speed:
                    reason_parts.append("low_mean_speed_eq")
                if high_share:
                    reason_parts.append("high_stationary_share")
        eligible.append(not reason_parts)
        reasons.append("|".join(reason_parts))

    out["isFitEligible"] = eligible if enabled else [True] * len(out)
    out["exclusionReason"] = reasons if enabled else [""] * len(out)
    out.attrs["segment_exclusion"] = {
        "enabled": bool(enabled),
        "min_mean_speed_eq_kmh": float(min_mean_speed_eq_kmh),
        "max_stationary_time_share": float(max_stationary_time_share),
        "stationary_speed_kmh": float(stationary_speed_kmh),
        "max_abs_altitude_rate_mph": float(max_abs_altitude_rate_mph),
        "excluded_count": int((~pd.Series(out["isFitEligible"])).sum()) if enabled else 0,
    }
    return out


def _resolve_fit_mask(
    segments_df: pd.DataFrame,
    fit_mask_col: Optional[str],
    *,
    context: str,
) -> pd.Series:
    """Return a boolean fit mask; fall back to all-True with a warning when missing."""
    if not fit_mask_col:
        return pd.Series(True, index=segments_df.index)
    if fit_mask_col not in segments_df.columns:
        logger.warning(
            "%s: fit_mask_col=%s missing; falling back to fitting on all segments",
            context,
            fit_mask_col,
        )
        return pd.Series(True, index=segments_df.index)
    mask = segments_df[fit_mask_col].fillna(False).astype(bool)
    if not bool(mask.any()):
        logger.warning(
            "%s: fit mask excluded every segment; falling back to fitting on all segments",
            context,
        )
        return pd.Series(True, index=segments_df.index)
    return mask


def route_segments_from_points(
    route_df: pd.DataFrame,
    segment_km: float = 1.0,
    min_distance_km: float = 0.05,
) -> pd.DataFrame:
    """Build distance/elevation route segments from GPX-like points.

    The input is a planned route, not an observed activity. Any pseudo duration generated
    during preprocessing is therefore removed from the returned segment table.
    """
    route_points = route_df.drop(columns=["timestamp"], errors="ignore")
    prepared = prepare_raw_timeseries_for_segments(route_points)
    segments = segment_timeseries(
        prepared,
        segment_km=segment_km,
        min_distance_km=min_distance_km,
    )
    if segments.empty:
        return segments
    result = segments.copy()
    result["activityId"] = "route"
    result["actualTimeSec"] = np.nan
    result["meanHrReserve"] = np.nan
    result["meanHr"] = np.nan
    result["meanSpeedKmh"] = np.nan
    result["meanSpeedEqKmh"] = np.nan
    return result


def hard_trailrun_mask(
    df: pd.DataFrame,
    min_moving_sec: float = 1800.0,
    min_distance_km: float = 10.0,
    min_ascent_m: float = 500.0,
    min_hr_reserve: float = 0.70,
) -> pd.Series:
    """Return the hard/race-like TrailRun mask used by the notebook."""
    category = df.get("category", pd.Series("", index=df.index)).astype(str).str.upper()
    moving = pd.to_numeric(
        df.get("movingSec", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    distance = pd.to_numeric(
        df.get("distanceKm", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    ascent = pd.to_numeric(
        df.get("ascentM", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    reserve = pd.to_numeric(
        df.get("hrReserveRatio", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    has_ts = df.get("hasTimeseries", pd.Series(True, index=df.index)).astype(str).str.lower()
    usable_ts = has_ts.isin({"true", "1", "yes"})
    hard = (distance >= min_distance_km) | (ascent >= min_ascent_m) | (reserve >= min_hr_reserve)
    return category.eq("TRAIL_RUN") & usable_ts & (moving >= min_moving_sec) & hard


def add_hr_reserve(
    df: pd.DataFrame,
    avg_hr_col: str = "avgHr",
    hr_rest: float = 60.0,
    hr_max: float = 190.0,
) -> pd.DataFrame:
    """Add an HR reserve ratio column without dropping submaximal observations."""
    result = df.copy()
    avg_hr = pd.to_numeric(
        result.get(avg_hr_col, pd.Series(np.nan, index=result.index)), errors="coerce"
    )
    if hr_max <= hr_rest:
        result["hrReserveRatio"] = np.nan
        return result
    result["hrReserveRatio"] = ((avg_hr - hr_rest) / (hr_max - hr_rest)).clip(lower=0.0, upper=1.2)
    return result


def estimate_hrr_duration_envelope(
    activities_df: pd.DataFrame,
    hrr_col: str = "hrReserveRatio",
    avg_hr_col: str = "avgHr",
    duration_col: Optional[str] = None,
    category_col: str = "category",
    categories: Optional[Sequence[str]] = ("RUN", "TRAIL_RUN"),
    hr_rest: Optional[float] = None,
    hr_max: Optional[float] = None,
    hrr_min: float = 0.30,
    hrr_max: float = 0.95,
    bin_width: float = 0.05,
    min_duration_sec: float = 300.0,
) -> pd.DataFrame:
    """Estimate the observed sustainable-duration envelope by activity HRR range.

    This is an empirical guardrail for constant-HRR race planning. It reports the maximum
    observed duration in each average-HRR band, not a physiological time-to-exhaustion
    estimate.
    """
    columns = [
        "hrrLower",
        "hrrUpper",
        "hrrMid",
        "hrrRange",
        "activityCount",
        "maxDurationSec",
        "p90DurationSec",
        "medianDurationSec",
        "maxDurationMin",
    ]
    if activities_df.empty:
        return pd.DataFrame(columns=columns)

    working = activities_df.copy()
    if duration_col is None:
        duration_col = "timeSec" if "timeSec" in working.columns else "movingSec"
    if duration_col not in working.columns:
        return pd.DataFrame(columns=columns)

    if hrr_col not in working.columns:
        if avg_hr_col not in working.columns or hr_rest is None or hr_max is None:
            return pd.DataFrame(columns=columns)
        working = add_hr_reserve(
            working,
            avg_hr_col=avg_hr_col,
            hr_rest=float(hr_rest),
            hr_max=float(hr_max),
        )

    if categories is not None and category_col in working.columns:
        allowed = {str(value).upper() for value in categories}
        category = working[category_col].astype(str).str.upper()
        working = working[category.isin(allowed)].copy()

    working["_hrr"] = pd.to_numeric(working[hrr_col], errors="coerce")
    working["_duration"] = pd.to_numeric(working[duration_col], errors="coerce")
    working = working[
        working["_hrr"].notna()
        & working["_duration"].notna()
        & (working["_duration"] >= float(min_duration_sec))
    ].copy()

    if bin_width <= 0:
        raise ValueError("bin_width must be positive")
    lower = float(hrr_min)
    upper = float(hrr_max)
    if upper <= lower:
        raise ValueError("hrr_max must be greater than hrr_min")

    edges = np.arange(lower, upper + bin_width * 0.5, float(bin_width))
    if edges[-1] < upper:
        edges = np.r_[edges, upper]
    intervals = pd.IntervalIndex.from_breaks(edges, closed="left")

    rows: list[dict[str, object]] = []
    for interval in intervals:
        subset = working[(working["_hrr"] >= interval.left) & (working["_hrr"] < interval.right)]
        durations = subset["_duration"]
        count = int(len(subset))
        max_duration = float(durations.max()) if count else np.nan
        p90_duration = float(durations.quantile(0.90)) if count else np.nan
        median_duration = float(durations.median()) if count else np.nan
        rows.append(
            {
                "hrrLower": float(interval.left),
                "hrrUpper": float(interval.right),
                "hrrMid": float((interval.left + interval.right) / 2.0),
                "hrrRange": f"{interval.left:.2f}-{interval.right:.2f}",
                "activityCount": count,
                "maxDurationSec": max_duration,
                "p90DurationSec": p90_duration,
                "medianDurationSec": median_duration,
                "maxDurationMin": max_duration / 60.0 if math.isfinite(max_duration) else np.nan,
            }
        )
    return pd.DataFrame(rows, columns=columns)


def max_duration_for_hrr(
    hrr: float,
    envelope_df: pd.DataFrame,
    duration_col: str = "maxDurationSec",
) -> float:
    """Return the observed max duration for the HRR band containing ``hrr``."""
    if envelope_df.empty or duration_col not in envelope_df.columns:
        return np.nan
    value = _to_float(hrr, np.nan)
    if not math.isfinite(value):
        return np.nan
    required = {"hrrLower", "hrrUpper"}
    if not required.issubset(envelope_df.columns):
        return np.nan
    lower = pd.to_numeric(envelope_df["hrrLower"], errors="coerce")
    upper = pd.to_numeric(envelope_df["hrrUpper"], errors="coerce")
    duration = pd.to_numeric(envelope_df[duration_col], errors="coerce")
    mask = (lower <= value) & (value < upper) & duration.notna()
    if not mask.any():
        return np.nan
    return float(duration[mask].iloc[0])


def estimate_hrr_duration_power_law(
    activities_df: pd.DataFrame,
    duration_windows_min: Sequence[float],
    hrr_col: str = "hrReserveRatio",
    avg_hr_col: str = "avgHr",
    duration_col: Optional[str] = None,
    category_col: str = "category",
    categories: Optional[Sequence[str]] = ("RUN", "TRAIL_RUN"),
    hr_rest: Optional[float] = None,
    hr_max: Optional[float] = None,
    hrr_min: float = 0.30,
    hrr_max: float = 0.98,
    min_duration_sec: float = 300.0,
    target_stat: str = "max",
    target_quantile: float = 0.90,
    min_activity_count: int = 2,
    fit_weight_mode: str = "performance",
    fit_weight_power: float = 4.0,
) -> tuple[Mapping[str, object], pd.DataFrame]:
    """Fit a monotone power law for sustainable average HRR over duration.

    The fitted frontier is based on representative duration windows. For a window
    ``T``, the target HRR is computed from activities lasting at least ``T``;
    targets are made non-increasing before fitting ``HRR = a * hours**b``.
    """
    columns = [
        "durationMin",
        "durationHours",
        "activityCount",
        "maxObservedHrr",
        "quantileObservedHrr",
        "targetHrrRaw",
        "targetHrr",
        "fittedHrr",
        "residualHrr",
        "fitWeight",
        "usedForFit",
        "isExtrapolated",
    ]
    empty_windows = pd.DataFrame(columns=columns)
    target_mode = str(target_stat).strip().lower()
    if target_mode not in {"max", "quantile"}:
        raise ValueError("target_stat must be 'max' or 'quantile'")
    quantile = float(target_quantile)
    if not 0.0 < quantile <= 1.0:
        raise ValueError("target_quantile must be in (0, 1]")
    if not duration_windows_min:
        raise ValueError("duration_windows_min must not be empty")
    windows = sorted({float(value) for value in duration_windows_min if float(value) > 0.0})
    if not windows:
        raise ValueError("duration_windows_min must contain positive values")
    if hrr_max <= hrr_min:
        raise ValueError("hrr_max must be greater than hrr_min")
    weight_mode = str(fit_weight_mode).strip().lower()
    if weight_mode not in {"uniform", "performance"}:
        raise ValueError("fit_weight_mode must be 'uniform' or 'performance'")
    weight_power = max(0.0, float(fit_weight_power))

    base_params: dict[str, object] = {
        "model": "hrr_duration_power_law",
        "formula": "HRR(T_hours) = coefficient * T_hours ** exponent",
        "coefficient": np.nan,
        "exponent": np.nan,
        "r2Log": np.nan,
        "weightedR2Log": np.nan,
        "maeHrr": np.nan,
        "weightedMaeHrr": np.nan,
        "targetStatistic": target_mode,
        "targetQuantile": quantile if target_mode == "quantile" else np.nan,
        "minActivityCount": int(min_activity_count),
        "fitWeightMode": weight_mode,
        "fitWeightPower": weight_power,
        "fitWindowCount": 0,
        "hrrMin": float(hrr_min),
        "hrrMax": float(hrr_max),
        "minWindowSec": float(min(windows) * 60.0),
        "maxWindowSec": float(max(windows) * 60.0),
        "observedMinDurationSec": np.nan,
        "observedMaxDurationSec": np.nan,
    }
    if activities_df.empty:
        return base_params, empty_windows

    working = activities_df.copy()
    if duration_col is None:
        duration_col = "timeSec" if "timeSec" in working.columns else "movingSec"
    if duration_col not in working.columns:
        return base_params, empty_windows

    if hrr_col not in working.columns:
        if avg_hr_col not in working.columns or hr_rest is None or hr_max is None:
            return base_params, empty_windows
        working = add_hr_reserve(
            working,
            avg_hr_col=avg_hr_col,
            hr_rest=float(hr_rest),
            hr_max=float(hr_max),
        )

    if categories is not None and category_col in working.columns:
        allowed = {str(value).upper() for value in categories}
        category = working[category_col].astype(str).str.upper()
        working = working[category.isin(allowed)].copy()

    working["_hrr"] = pd.to_numeric(working[hrr_col], errors="coerce")
    working["_duration"] = pd.to_numeric(working[duration_col], errors="coerce")
    working = working[
        working["_hrr"].notna()
        & working["_duration"].notna()
        & working["_hrr"].between(float(hrr_min), float(hrr_max), inclusive="both")
        & (working["_duration"] >= float(min_duration_sec))
    ].copy()
    if working.empty:
        return base_params, empty_windows

    rows: list[dict[str, object]] = []
    for duration_min in windows:
        duration_sec = duration_min * 60.0
        subset = working[working["_duration"] >= duration_sec]
        count = int(len(subset))
        hrr_values = subset["_hrr"]
        max_hrr = float(hrr_values.max()) if count else np.nan
        quantile_hrr = float(hrr_values.quantile(quantile)) if count else np.nan
        target_raw = max_hrr if target_mode == "max" else quantile_hrr
        rows.append(
            {
                "durationMin": float(duration_min),
                "durationHours": float(duration_min / 60.0),
                "activityCount": count,
                "maxObservedHrr": max_hrr,
                "quantileObservedHrr": quantile_hrr,
                "targetHrrRaw": target_raw,
            }
        )

    windows_df = pd.DataFrame(rows)
    finite_target = pd.to_numeric(windows_df["targetHrrRaw"], errors="coerce")
    monotone_targets: list[float] = []
    running_min = np.inf
    for value in finite_target:
        if math.isfinite(float(value)):
            running_min = min(running_min, float(value))
            monotone_targets.append(running_min)
        else:
            monotone_targets.append(np.nan)
    windows_df["targetHrr"] = monotone_targets

    enough_count = pd.to_numeric(windows_df["activityCount"], errors="coerce").fillna(0)
    fit_mask = (
        windows_df["targetHrr"].notna()
        & windows_df["durationHours"].gt(0)
        & (enough_count >= max(1, int(min_activity_count)))
    )
    if int(fit_mask.sum()) < 2:
        fit_mask = windows_df["targetHrr"].notna() & windows_df["durationHours"].gt(0)
    target_for_weights = pd.to_numeric(windows_df["targetHrr"], errors="coerce")
    fit_weights = pd.Series(np.nan, index=windows_df.index, dtype=float)
    if fit_mask.any():
        if weight_mode == "performance":
            max_target = float(target_for_weights[fit_mask].max())
            if math.isfinite(max_target) and max_target > 0.0:
                raw_weights = np.power(
                    np.clip(target_for_weights / max_target, 1e-9, None),
                    weight_power,
                )
            else:
                raw_weights = pd.Series(1.0, index=windows_df.index)
        else:
            raw_weights = pd.Series(1.0, index=windows_df.index)
        fit_weights.loc[target_for_weights.notna()] = raw_weights[target_for_weights.notna()]
    windows_df["fitWeight"] = fit_weights
    fit_df = windows_df[fit_mask].copy()

    coefficient = np.nan
    exponent = np.nan
    r2_log = np.nan
    weighted_r2_log = np.nan
    mae_hrr = np.nan
    weighted_mae_hrr = np.nan
    if len(fit_df) >= 2:
        x = np.log(fit_df["durationHours"].to_numpy(dtype=float))
        y = np.log(fit_df["targetHrr"].to_numpy(dtype=float))
        weights = fit_df["fitWeight"].to_numpy(dtype=float)
        weights = np.where(np.isfinite(weights) & (weights > 0.0), weights, 1.0)
        exponent, log_coefficient = np.polyfit(x, y, deg=1, w=np.sqrt(weights))
        exponent = min(0.0, float(exponent))
        coefficient = float(math.exp(log_coefficient))
        fitted = coefficient * np.power(fit_df["durationHours"].to_numpy(dtype=float), exponent)
        fitted = np.clip(fitted, float(hrr_min), float(hrr_max))
        residual = fit_df["targetHrr"].to_numpy(dtype=float) - fitted
        mae_hrr = float(np.mean(np.abs(residual)))
        weighted_mae_hrr = float(np.average(np.abs(residual), weights=weights))
        total = float(np.sum((y - y.mean()) ** 2))
        predicted_y = np.log(np.clip(fitted, 1e-9, None))
        r2_log = float(1.0 - np.sum((y - predicted_y) ** 2) / total) if total > 0 else np.nan
        weighted_mean = float(np.average(y, weights=weights))
        weighted_total = float(np.sum(weights * (y - weighted_mean) ** 2))
        weighted_residual = float(np.sum(weights * (y - predicted_y) ** 2))
        weighted_r2_log = (
            float(1.0 - weighted_residual / weighted_total) if weighted_total > 0 else np.nan
        )

    if math.isfinite(coefficient) and math.isfinite(exponent):
        duration_hours = windows_df["durationHours"].to_numpy(dtype=float)
        fitted_all = coefficient * np.power(duration_hours, exponent)
        windows_df["fittedHrr"] = np.clip(fitted_all, float(hrr_min), float(hrr_max))
        windows_df["residualHrr"] = windows_df["targetHrr"] - windows_df["fittedHrr"]
    else:
        windows_df["fittedHrr"] = np.nan
        windows_df["residualHrr"] = np.nan
    windows_df["usedForFit"] = fit_mask
    observed_max = float(working["_duration"].max())
    windows_df["isExtrapolated"] = windows_df["durationMin"].mul(60.0).gt(observed_max)

    params = dict(base_params)
    params.update(
        {
            "coefficient": coefficient,
            "exponent": exponent,
            "r2Log": r2_log,
            "weightedR2Log": weighted_r2_log,
            "maeHrr": mae_hrr,
            "weightedMaeHrr": weighted_mae_hrr,
            "fitWindowCount": int(fit_mask.sum()),
            "observedMinDurationSec": float(working["_duration"].min()),
            "observedMaxDurationSec": observed_max,
        }
    )
    return params, windows_df.reindex(columns=columns)


def hrr_for_duration_power_law(duration_sec: float, params: Mapping[str, object]) -> float:
    """Evaluate fitted sustainable HRR for ``duration_sec``."""
    coefficient = _to_float(params.get("coefficient"), np.nan)
    exponent = _to_float(params.get("exponent"), np.nan)
    if not math.isfinite(coefficient) or not math.isfinite(exponent):
        return np.nan
    duration_hours = max(_to_float(duration_sec, np.nan) / 3600.0, 1e-9)
    hrr = coefficient * duration_hours**exponent
    hrr_min = _to_float(params.get("hrrMin"), 0.0)
    hrr_max = _to_float(params.get("hrrMax"), 1.2)
    return float(np.clip(hrr, hrr_min, hrr_max))


def max_duration_for_hrr_power_law(
    hrr: float,
    params: Mapping[str, object],
    clip_to_window: bool = True,
) -> float:
    """Invert a fitted HRR-duration power law into max sustainable duration."""
    coefficient = _to_float(params.get("coefficient"), np.nan)
    exponent = _to_float(params.get("exponent"), np.nan)
    reserve = _to_float(hrr, np.nan)
    if not all(math.isfinite(value) for value in (coefficient, exponent, reserve)):
        return np.nan
    min_window = _to_float(params.get("minWindowSec"), np.nan)
    max_window = _to_float(params.get("maxWindowSec"), np.nan)
    if abs(exponent) < 1e-12:
        duration_sec = max_window if reserve <= coefficient and math.isfinite(max_window) else min_window
    else:
        duration_hours = (max(reserve, 1e-9) / coefficient) ** (1.0 / exponent)
        duration_sec = float(duration_hours * 3600.0)
    if clip_to_window:
        if math.isfinite(min_window):
            duration_sec = max(duration_sec, min_window)
        if math.isfinite(max_window):
            duration_sec = min(duration_sec, max_window)
    return float(duration_sec)


def top_hrr_hard_trailrun_ids(
    df: pd.DataFrame,
    n: int = 10,
    hard_col: str = "hardTrailRun",
    usable_col: str = "usableTrailRun",
) -> list[str]:
    """Return hard TrailRun IDs sorted by highest average HR reserve."""
    if n <= 0 or df.empty:
        return []
    category = df.get("category", pd.Series("", index=df.index)).astype(str).str.upper()
    hard = df.get(hard_col, pd.Series(False, index=df.index)).astype(bool)
    usable = df.get(usable_col, pd.Series(True, index=df.index)).astype(bool)
    working = df[category.eq("TRAIL_RUN") & hard & usable].copy()
    if working.empty:
        return []
    working["hrReserveSort"] = pd.to_numeric(working.get("hrReserveRatio"), errors="coerce")
    working["distanceSort"] = pd.to_numeric(working.get("distanceKm"), errors="coerce").fillna(0.0)
    working = working.sort_values(
        ["hrReserveSort", "distanceSort", "activityId"],
        ascending=[False, False, True],
        na_position="last",
    )
    return working["activityId"].astype(str).head(n).tolist()


def select_best_activity_by_dates(
    df: pd.DataFrame,
    date_strings: Sequence[str],
    require_timeseries: bool = True,
) -> pd.DataFrame:
    """Select the best usable activity for each requested date.

    Selection is intentionally deterministic: prefer TrailRun, then Run, then largest
    distance-equivalent score. The returned frame includes one row per matched date.
    """
    if df.empty or not date_strings:
        return df.iloc[0:0].copy()

    requested_dates = [pd.Timestamp(value).date() for value in date_strings]
    working = df.copy()
    if "startDate" in working.columns:
        working["selectionDate"] = pd.to_datetime(working["startDate"], errors="coerce").dt.date
    else:
        working["selectionDate"] = pd.to_datetime(working["startTime"], errors="coerce").dt.date
    working = working[working["selectionDate"].isin(requested_dates)].copy()
    if require_timeseries and "hasTimeseries" in working.columns:
        has_ts = working["hasTimeseries"].astype(str).str.lower().isin({"true", "1", "yes"})
        working = working[has_ts].copy()
    if working.empty:
        return working

    category_priority = {"TRAIL_RUN": 4.0, "RUN": 3.0, "HIKE": 2.0, "RIDE": 1.0}
    category = working.get("category", pd.Series("", index=working.index)).astype(str).str.upper()
    distance = pd.to_numeric(working.get("distanceKm"), errors="coerce").fillna(0.0)
    ascent = pd.to_numeric(working.get("ascentM"), errors="coerce").fillna(0.0)
    actual_time = pd.to_numeric(
        working.get("actualTimeSec", working.get("movingSec")),
        errors="coerce",
    ).fillna(0.0)
    working["categoryPriority"] = category.map(category_priority).fillna(0.0)
    working["distanceEqSelection"] = distance + 0.01 * ascent
    working["selectionScore"] = (
        working["categoryPriority"] * 1_000_000.0
        + working["distanceEqSelection"] * 1_000.0
        + actual_time / 3_600.0
    )
    selected = (
        working.sort_values(["selectionDate", "selectionScore"], ascending=[True, False])
        .groupby("selectionDate", as_index=False)
        .head(1)
        .copy()
    )
    selected["requestedDate"] = selected["selectionDate"]
    return selected


def compute_redi_load_features(
    daily_df: pd.DataFrame,
    load_col: str = "trimp",
    slow_lam: float = 0.03,
    fast_lam: float = 0.14,
    prefix: str = "trimp",
) -> pd.DataFrame:
    """Compute slow/fast REDI readiness features from daily workload."""
    if daily_df.empty:
        return pd.DataFrame(columns=["date", f"{prefix}RediSlow", f"{prefix}RediFast"])
    working = daily_df.copy()
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    values = pd.to_numeric(working.get(load_col), errors="coerce").fillna(0.0).to_numpy()
    slow_col = f"{prefix}RediSlow"
    fast_col = f"{prefix}RediFast"
    working[slow_col] = compute_redi(values, slow_lam)
    working[fast_col] = compute_redi(values, fast_lam)
    working[f"{prefix}RediBalance"] = working[slow_col] - working[fast_col]
    return working[["date", slow_col, fast_col, f"{prefix}RediBalance"]]


def attach_previous_daily_features(
    activity_df: pd.DataFrame,
    daily_features_df: pd.DataFrame,
    activity_date_col: str = "startDate",
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Attach latest strictly previous daily features to each activity."""
    result = activity_df.copy()
    if result.empty or daily_features_df.empty:
        return result

    daily = daily_features_df.copy()
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")
    daily = daily.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if feature_cols is None:
        feature_cols = [col for col in daily.columns if col != "date"]
    date_key = daily["date"].dt.date

    feature_rows: list[dict[str, object]] = []
    for _, row in result.iterrows():
        activity_date = pd.to_datetime(row.get(activity_date_col), errors="coerce")
        payload: dict[str, object] = {"activityId": str(row.get("activityId", ""))}
        if pd.isna(activity_date):
            for col in feature_cols:
                payload[col] = np.nan
            feature_rows.append(payload)
            continue
        previous = daily[date_key < activity_date.date()]
        if previous.empty:
            for col in feature_cols:
                payload[col] = np.nan
        else:
            payload.update(previous.iloc[-1][list(feature_cols)].to_dict())
        feature_rows.append(payload)

    features = pd.DataFrame(feature_rows)
    return result.merge(features, on="activityId", how="left", suffixes=("", "_daily"))


def add_in_activity_trimp_features(
    segments_df: pd.DataFrame,
    decay_lambda: float = 0.30,
    activity_col: str = "activityId",
) -> pd.DataFrame:
    """Add segment TRIMP and leakage-free cumulative in-activity TRIMP features."""
    if segments_df.empty:
        return segments_df.copy()
    result = segments_df.copy()
    duration_hours = pd.to_numeric(result["actualTimeSec"], errors="coerce").fillna(0.0) / 3_600.0
    hrr = pd.to_numeric(result.get("meanHrReserve"), errors="coerce").fillna(0.0).clip(0.0, 1.2)
    result["segmentTrimp"] = duration_hours * hrr * 0.64 * np.exp(1.92 * hrr)
    result["cumTrimpBefore"] = 0.0
    result["cumTrimp"] = 0.0
    result["decayedTrimpBefore"] = 0.0
    result["decayedTrimp"] = 0.0

    sort_cols = [activity_col]
    if "segmentIndex" in result.columns:
        sort_cols.append("segmentIndex")
    elif "startKm" in result.columns:
        sort_cols.append("startKm")
    result = result.sort_values(sort_cols).copy()
    decay = float(math.exp(-max(0.0, decay_lambda)))

    for _, group in result.groupby(activity_col, sort=False):
        cumulative = 0.0
        decayed = 0.0
        for idx in group.index:
            trimp = _to_float(result.at[idx, "segmentTrimp"], 0.0)
            result.at[idx, "cumTrimpBefore"] = cumulative
            result.at[idx, "decayedTrimpBefore"] = decayed
            cumulative += trimp
            decayed = trimp + decay * decayed
            result.at[idx, "cumTrimp"] = cumulative
            result.at[idx, "decayedTrimp"] = decayed
    return result


def terrain_family(grade: float) -> str:
    """Coarse grade family for model diagnostics."""
    g = _to_float(grade, 0.0)
    if g >= 0.15:
        return "steep_climb"
    if g >= 0.04:
        return "climb"
    if g <= -0.15:
        return "steep_descent"
    if g <= -0.04:
        return "descent"
    return "flat"


def predict_segment_times(
    segments_df: pd.DataFrame,
    v_vt2_kmh: float,
    alpha: float,
    mu: float = 0.0,
    fatigue_model: str = "linear",
    heat_factor: float = 1.0,
    ctl_factor: float = 1.0,
) -> pd.Series:
    """Predict segment times using the paper-style GAP model."""
    if segments_df.empty:
        return pd.Series(dtype=float)

    distance = pd.to_numeric(segments_df["distanceKm"], errors="coerce").fillna(0.0)
    altitude = pd.to_numeric(
        segments_df.get("meanAltitudeM", pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)
    progress = pd.to_numeric(
        segments_df.get("progress", pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)

    gap = _gap_factors_for_segments(segments_df)
    altitude_values = altitude.map(altitude_factor)
    if fatigue_model == "exponential":
        fatigue = progress.map(lambda value: exponential_decay_factor(value, mu))
    else:
        fatigue = progress.map(lambda value: linear_decay_factor(value, mu))

    base_speed = max(float(v_vt2_kmh) * float(alpha), 1e-9)
    speed = base_speed * altitude_values * float(heat_factor) * float(ctl_factor) * fatigue / gap
    speed = speed.clip(lower=0.1)
    return distance / speed * 3600.0


def predict_activity_time(
    segments_df: pd.DataFrame,
    v_vt2_kmh: float,
    alpha: float,
    mu: float = 0.0,
    fatigue_model: str = "linear",
    heat_factor: float = 1.0,
    ctl_factor: float = 1.0,
) -> float:
    """Predict total activity time from segment-level predictions."""
    predicted = predict_segment_times(
        segments_df,
        v_vt2_kmh=v_vt2_kmh,
        alpha=alpha,
        mu=mu,
        fatigue_model=fatigue_model,
        heat_factor=heat_factor,
        ctl_factor=ctl_factor,
    )
    return float(predicted.sum())


def predict_many(
    segments_by_activity: Mapping[str, pd.DataFrame],
    v_vt2_kmh: float,
    alpha: float,
    mu: float = 0.0,
    fatigue_model: str = "linear",
    ctl_factors: Optional[Mapping[str, float]] = None,
) -> pd.Series:
    """Predict times for several activities."""
    rows = {
        str(activity_id): predict_activity_time(
            segments,
            v_vt2_kmh=v_vt2_kmh,
            alpha=alpha,
            mu=mu,
            fatigue_model=fatigue_model,
            ctl_factor=_to_float(
                ctl_factors.get(str(activity_id), 1.0) if ctl_factors is not None else 1.0,
                1.0,
            ),
        )
        for activity_id, segments in segments_by_activity.items()
    }
    return pd.Series(rows, dtype=float)


def _regression_metrics_arrays(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    y = np.asarray(actual, dtype=float)
    y_hat = np.asarray(predicted, dtype=float)
    valid = np.isfinite(y) & np.isfinite(y_hat) & (y > 0)
    if valid.sum() == 0:
        return {"r2": np.nan, "maeSec": np.nan, "mapePct": np.nan, "biasSec": np.nan}
    y = y[valid]
    y_hat = y_hat[valid]
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {
        "r2": r2,
        "maeSec": float(np.mean(np.abs(y_hat - y))),
        "mapePct": float(np.mean(np.abs((y_hat - y) / y)) * 100.0),
        "biasSec": float(np.mean(y_hat - y)),
    }


def regression_metrics(actual: Sequence[float], predicted: Sequence[float]) -> dict[str, float]:
    """Compute R2, MAE, MAPE, and bias in seconds."""
    return _regression_metrics_arrays(actual, predicted)


def _grid_activity_terms(
    segments_df: pd.DataFrame,
    v_vt2_kmh: float,
    ctl_factor: float,
) -> tuple[np.ndarray, np.ndarray]:
    distance = pd.to_numeric(segments_df["distanceKm"], errors="coerce").fillna(0.0).to_numpy()
    altitude = pd.to_numeric(
        segments_df.get("meanAltitudeM", pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)
    progress = pd.to_numeric(
        segments_df.get("progress", pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)

    gap = _gap_factors_for_segments(segments_df).to_numpy(dtype=float)
    altitude_values = np.asarray([altitude_factor(value) for value in altitude], dtype=float)
    denominator = max(float(v_vt2_kmh) * float(ctl_factor), 1e-9) * altitude_values
    static_time_terms = distance * 3600.0 * gap / np.clip(denominator, 1e-9, None)
    progress_values = progress.clip(lower=0.0, upper=1.0).to_numpy(dtype=float)
    return static_time_terms, progress_values


def _fatigue_array(progress: np.ndarray, mu: float, fatigue_model: str) -> np.ndarray:
    if fatigue_model == "exponential":
        fatigue = np.exp(float(mu) * progress)
    else:
        fatigue = 1.0 + float(mu) * progress
    return np.clip(fatigue, 0.1, None)


def grid_search_model(
    segments_by_activity: Mapping[str, pd.DataFrame],
    observed_times_sec: Mapping[str, float],
    v_vt2_kmh: float,
    alpha_grid: Iterable[float],
    mu_grid: Iterable[float],
    fatigue_model: str = "linear",
    ctl_factors: Optional[Mapping[str, float]] = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Grid-search alpha and fatigue/decay parameters."""
    rows: list[dict[str, float]] = []
    ids = [str(activity_id) for activity_id in segments_by_activity.keys()]
    actual = np.asarray([float(observed_times_sec[activity_id]) for activity_id in ids], dtype=float)
    alpha_values = [float(value) for value in alpha_grid]
    mu_values = [float(value) for value in mu_grid]
    precomputed = {}
    for activity_id, segments in segments_by_activity.items():
        normalized_id = str(activity_id)
        ctl_factor = _to_float(
            ctl_factors.get(normalized_id, 1.0) if ctl_factors is not None else 1.0,
            1.0,
        )
        precomputed[normalized_id] = _grid_activity_terms(
            segments,
            v_vt2_kmh=v_vt2_kmh,
            ctl_factor=ctl_factor,
        )

    mu_predictions = {}
    for mu in mu_values:
        predicted_base = []
        for activity_id in ids:
            static_terms, progress = precomputed[activity_id]
            fatigue = _fatigue_array(progress, mu, fatigue_model)
            predicted_base.append(float(np.sum(static_terms / fatigue)))
        mu_predictions[mu] = np.asarray(predicted_base, dtype=float)

    best: Optional[dict[str, float]] = None
    for alpha in alpha_values:
        for mu in mu_values:
            predicted = mu_predictions[mu] / max(alpha, 1e-9)
            metrics = regression_metrics(actual, predicted)
            row = {"alpha": alpha, "mu": mu, **metrics}
            rows.append(row)
            if best is None:
                best = row
            else:
                best_r2 = -np.inf if pd.isna(best["r2"]) else best["r2"]
                row_r2 = -np.inf if pd.isna(row["r2"]) else row["r2"]
                if (row_r2, -row["maeSec"]) > (best_r2, -best["maeSec"]):
                    best = row

    result_df = pd.DataFrame(rows)
    assert best is not None
    return best, result_df


def leave_one_out_grid_search(
    segments_by_activity: Mapping[str, pd.DataFrame],
    observed_times_sec: Mapping[str, float],
    v_vt2_kmh: float,
    alpha_grid: Iterable[float],
    mu_grid: Iterable[float],
    fatigue_model: str = "linear",
    ctl_factors: Optional[Mapping[str, float]] = None,
) -> pd.DataFrame:
    """Leave-one-activity-out grid-search validation."""
    ids = [str(activity_id) for activity_id in segments_by_activity.keys()]
    folds: list[dict[str, float | str]] = []
    for held_out in ids:
        train_segments = {
            activity_id: segments
            for activity_id, segments in segments_by_activity.items()
            if str(activity_id) != held_out
        }
        train_observed = {
            activity_id: observed_times_sec[activity_id]
            for activity_id in train_segments.keys()
        }
        train_ctl_factors = None
        if ctl_factors is not None:
            train_ctl_factors = {
                str(activity_id): float(ctl_factors[str(activity_id)])
                for activity_id in train_segments.keys()
                if str(activity_id) in ctl_factors
            }
        best, _grid = grid_search_model(
            train_segments,
            train_observed,
            v_vt2_kmh=v_vt2_kmh,
            alpha_grid=alpha_grid,
            mu_grid=mu_grid,
            fatigue_model=fatigue_model,
            ctl_factors=train_ctl_factors,
        )
        predicted = predict_activity_time(
            segments_by_activity[held_out],
            v_vt2_kmh=v_vt2_kmh,
            alpha=best["alpha"],
            mu=best["mu"],
            fatigue_model=fatigue_model,
            ctl_factor=_to_float(
                ctl_factors.get(held_out, 1.0) if ctl_factors is not None else 1.0,
                1.0,
            ),
        )
        actual = float(observed_times_sec[held_out])
        folds.append(
            {
                "activityId": held_out,
                "alpha": best["alpha"],
                "mu": best["mu"],
                "actualTimeSec": actual,
                "predictedTimeSec": predicted,
                "errorSec": predicted - actual,
                "errorPct": (predicted - actual) / actual * 100.0,
            }
        )
    return pd.DataFrame(folds)


def predict_extension_segment_times(
    segments_df: pd.DataFrame,
    v_anchor_kmh: float,
    alpha: float,
    mu: float = 0.0,
    fatigue_model: str = "linear",
    terrain_multipliers: Optional[Mapping[str, float]] = None,
    technicality_coef: float = 0.0,
    hr_coef: float = 0.0,
    acute_trimp_coef: float = 0.0,
    hr_center: float = 0.70,
    acute_trimp_col: str = "decayedTrimpBefore",
) -> pd.Series:
    """Predict segment times for extension models with direct segment modifiers."""
    if segments_df.empty:
        return pd.Series(dtype=float)
    base_time = predict_segment_times(
        segments_df,
        v_vt2_kmh=v_anchor_kmh,
        alpha=alpha,
        mu=mu,
        fatigue_model=fatigue_model,
    )
    terrain = segments_df.get("terrainFamily", pd.Series("flat", index=segments_df.index))
    terrain_factors = terrain.astype(str).map(terrain_multipliers or {}).fillna(1.0)
    technicality = pd.to_numeric(
        segments_df.get("technicalityGps", pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)
    hrr = pd.to_numeric(
        segments_df.get("meanHrReserve", pd.Series(hr_center, index=segments_df.index)),
        errors="coerce",
    ).fillna(hr_center)
    acute = pd.to_numeric(
        segments_df.get(acute_trimp_col, pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)
    modifier = terrain_factors.to_numpy(dtype=float) * np.exp(
        float(hr_coef) * (hrr.to_numpy(dtype=float) - float(hr_center))
        - float(technicality_coef) * technicality.to_numpy(dtype=float)
        - float(acute_trimp_coef) * acute.to_numpy(dtype=float)
    )
    modifier = np.clip(modifier, 0.1, 10.0)
    return pd.Series(base_time.to_numpy(dtype=float) / modifier, index=segments_df.index)


def segment_grid_search_model(
    segments_df: pd.DataFrame,
    v_anchor_kmh: float,
    alpha_grid: Iterable[float],
    mu_grid: Iterable[float],
    terrain_multiplier_grid: Optional[Sequence[Mapping[str, float]]] = None,
    technicality_coef_grid: Optional[Iterable[float]] = None,
    hr_coef_grid: Optional[Iterable[float]] = None,
    acute_trimp_coef_grid: Optional[Iterable[float]] = None,
    fatigue_model: str = "linear",
    activity_col: str = "activityId",
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Grid-search extension parameters against segment times directly."""
    if segments_df.empty:
        empty = pd.DataFrame()
        return {}, empty, empty

    terrain_grid = list(terrain_multiplier_grid or [{}])
    technicality_grid = [float(value) for value in (technicality_coef_grid or [0.0])]
    hr_grid = [float(value) for value in (hr_coef_grid or [0.0])]
    acute_grid = [float(value) for value in (acute_trimp_coef_grid or [0.0])]
    rows: list[dict[str, object]] = []
    actual = pd.to_numeric(segments_df["actualTimeSec"], errors="coerce").to_numpy(dtype=float)

    best: Optional[dict[str, object]] = None
    best_prediction = pd.Series(dtype=float)
    for alpha in [float(value) for value in alpha_grid]:
        for mu in [float(value) for value in mu_grid]:
            for terrain_idx, terrain_multipliers in enumerate(terrain_grid):
                for technicality_coef in technicality_grid:
                    for hr_coef in hr_grid:
                        for acute_trimp_coef in acute_grid:
                            predicted = predict_extension_segment_times(
                                segments_df,
                                v_anchor_kmh=v_anchor_kmh,
                                alpha=alpha,
                                mu=mu,
                                fatigue_model=fatigue_model,
                                terrain_multipliers=terrain_multipliers,
                                technicality_coef=technicality_coef,
                                hr_coef=hr_coef,
                                acute_trimp_coef=acute_trimp_coef,
                            )
                            segment_metrics = regression_metrics(actual, predicted)
                            race_metrics = {
                                "r2": np.nan,
                                "maeSec": np.nan,
                                "mapePct": np.nan,
                                "biasSec": np.nan,
                            }
                            if activity_col in segments_df.columns:
                                race_frame = pd.DataFrame(
                                    {
                                        activity_col: segments_df[activity_col].astype(str),
                                        "actual": actual,
                                        "predicted": predicted.to_numpy(dtype=float),
                                    }
                                )
                                race_actual = race_frame.groupby(activity_col)["actual"].sum()
                                race_predicted = race_frame.groupby(activity_col)["predicted"].sum()
                                race_metrics = regression_metrics(race_actual, race_predicted)
                            row = {
                                "alpha": alpha,
                                "mu": mu,
                                "terrainProfile": terrain_idx,
                                "terrainMultipliers": dict(terrain_multipliers),
                                "technicalityCoef": technicality_coef,
                                "hrCoef": hr_coef,
                                "acuteTrimpCoef": acute_trimp_coef,
                                "segmentR2": segment_metrics["r2"],
                                "segmentMaeSec": segment_metrics["maeSec"],
                                "segmentMapePct": segment_metrics["mapePct"],
                                "segmentBiasSec": segment_metrics["biasSec"],
                                "raceR2": race_metrics["r2"],
                                "raceMaeSec": race_metrics["maeSec"],
                                "raceMapePct": race_metrics["mapePct"],
                                "raceBiasSec": race_metrics["biasSec"],
                            }
                            rows.append(row)
                            if best is None:
                                best = row
                                best_prediction = predicted
                            else:
                                best_r2 = -np.inf if pd.isna(best["segmentR2"]) else best["segmentR2"]
                                row_r2 = -np.inf if pd.isna(row["segmentR2"]) else row["segmentR2"]
                                if (row_r2, -row["segmentMaeSec"]) > (
                                    best_r2,
                                    -best["segmentMaeSec"],
                                ):
                                    best = row
                                    best_prediction = predicted

    prediction_df = segments_df.copy()
    prediction_df["predictedTimeSec"] = best_prediction.to_numpy(dtype=float)
    prediction_df["errorSec"] = prediction_df["predictedTimeSec"] - pd.to_numeric(
        prediction_df["actualTimeSec"],
        errors="coerce",
    )
    assert best is not None
    return best, pd.DataFrame(rows), prediction_df


def _hrr_effort_values(
    segments_df: pd.DataFrame,
    hrr_reference: float,
    min_factor: float,
    max_factor: float,
    hrr_col: str,
) -> np.ndarray:
    reference = max(float(hrr_reference), 1e-6)
    hrr = pd.to_numeric(
        segments_df.get(hrr_col, pd.Series(reference, index=segments_df.index)),
        errors="coerce",
    ).fillna(reference)
    return np.clip(hrr.to_numpy(dtype=float) / reference, float(min_factor), float(max_factor))


def _trimp_fatigue_values(
    segments_df: pd.DataFrame,
    fatigue_coef: float,
    fatigue_model: str,
    trimp_scale: float,
    min_factor: float,
    acute_trimp_col: str,
    secondary_fatigue_coef: float = 0.0,
    secondary_acute_trimp_col: Optional[str] = None,
    secondary_fatigue_model: Optional[str] = None,
) -> np.ndarray:
    _ = trimp_scale  # Deprecated compatibility argument; fatigue now uses raw load.
    acute = pd.to_numeric(
        segments_df.get(acute_trimp_col, pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)
    load = acute.to_numpy(dtype=float)
    primary = _trimp_fatigue_from_load(
        load,
        fatigue_coef=fatigue_coef,
        fatigue_model=fatigue_model,
        min_factor=min_factor,
    )
    if not secondary_acute_trimp_col:
        return primary
    secondary = pd.to_numeric(
        segments_df.get(secondary_acute_trimp_col, pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)
    secondary_load = secondary.to_numpy(dtype=float)
    secondary_fatigue = _trimp_fatigue_from_load(
        secondary_load,
        fatigue_coef=secondary_fatigue_coef,
        fatigue_model=secondary_fatigue_model or fatigue_model,
        min_factor=min_factor,
    )
    return np.clip(primary * secondary_fatigue, float(min_factor), 1.0)


def _trimp_fatigue_from_load(
    load: np.ndarray,
    fatigue_coef: float,
    fatigue_model: str,
    min_factor: float,
) -> np.ndarray:
    normalized_load = np.clip(np.asarray(load, dtype=float), 0.0, None)
    if fatigue_model == "exponential":
        fatigue = np.exp(-float(fatigue_coef) * normalized_load)
    else:
        fatigue = 1.0 - float(fatigue_coef) * normalized_load
    return np.clip(fatigue, float(min_factor), 1.0)


def predict_hrr_trimp_segment_times(
    segments_df: pd.DataFrame,
    v_anchor_kmh: float,
    alpha: float,
    fatigue_coef: float = 0.0,
    fatigue_model: str = "linear",
    hrr_reference: float = 0.70,
    hrr_min_factor: float = 0.55,
    hrr_max_factor: float = 1.30,
    trimp_scale: float = 10.0,
    min_fatigue_factor: float = 0.50,
    hrr_col: str = "meanHrReserve",
    acute_trimp_col: str = "decayedTrimpBefore",
    secondary_fatigue_coef: float = 0.0,
    secondary_acute_trimp_col: Optional[str] = None,
    secondary_fatigue_model: Optional[str] = None,
    load_factor_col: Optional[str] = None,
    use_hrr_effort: bool = True,
    gap_steep_threshold: float = DEFAULT_GAP_STEEP_THRESHOLD,
    gap_soft_start: float = DEFAULT_GAP_SOFT_START,
    gap_climb_scale: float = DEFAULT_GAP_CLIMB_SCALE,
    gap_descent_scale: float = DEFAULT_GAP_DESCENT_SCALE,
) -> pd.Series:
    """Predict segment times with a constrained HRR and acute-load speed equation.

    HRR is a fixed linear effort multiplier, while the configured fatigue column is used
    directly as the acute load. Typical fatigue columns are decayed in-activity TRIMP,
    cumulative in-activity TRIMP, or normalized route progress. A secondary fatigue
    column can be multiplied in with its own coefficient for short-term plus muscular
    fatigue variants. ``trimp_scale`` is kept only for compatibility with older scripts
    and is ignored.

    ``gap_climb_scale`` / ``gap_descent_scale`` apply an asymmetric trail correction on
    Minetti GAP for steep grades (``|avgGrade| >= gap_steep_threshold``).
    """
    if segments_df.empty:
        return pd.Series(dtype=float)

    distance = pd.to_numeric(segments_df["distanceKm"], errors="coerce").fillna(0.0)
    altitude = pd.to_numeric(
        segments_df.get("meanAltitudeM", pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)
    gap = _gap_factors_for_segments(
        segments_df,
        steep_threshold=gap_steep_threshold,
        soft_start=gap_soft_start,
        climb_scale=gap_climb_scale,
        descent_scale=gap_descent_scale,
    ).to_numpy(dtype=float)
    altitude_values = altitude.map(altitude_factor).to_numpy(dtype=float)
    if use_hrr_effort:
        hrr_effort = _hrr_effort_values(
            segments_df,
            hrr_reference=hrr_reference,
            min_factor=hrr_min_factor,
            max_factor=hrr_max_factor,
            hrr_col=hrr_col,
        )
    else:
        hrr_effort = np.ones(len(segments_df), dtype=float)
    fatigue = _trimp_fatigue_values(
        segments_df,
        fatigue_coef=fatigue_coef,
        fatigue_model=fatigue_model,
        trimp_scale=trimp_scale,
        min_factor=min_fatigue_factor,
        acute_trimp_col=acute_trimp_col,
        secondary_fatigue_coef=secondary_fatigue_coef,
        secondary_acute_trimp_col=secondary_acute_trimp_col,
        secondary_fatigue_model=secondary_fatigue_model,
    )
    if load_factor_col:
        load_factor = pd.to_numeric(
            segments_df.get(load_factor_col, pd.Series(1.0, index=segments_df.index)),
            errors="coerce",
        ).fillna(1.0)
        load_values = np.clip(load_factor.to_numpy(dtype=float), 0.1, 10.0)
    else:
        load_values = np.ones(len(segments_df), dtype=float)

    speed = (
        max(float(v_anchor_kmh) * float(alpha), 1e-9)
        * altitude_values
        * load_values
        * hrr_effort
        * fatigue
        / np.clip(gap, 1e-9, None)
    )
    speed = np.clip(speed, 0.1, None)
    predicted = distance.to_numpy(dtype=float) / speed * 3600.0
    return pd.Series(predicted, index=segments_df.index)


def _segment_trimp_from_prediction(time_sec: float, hrr: float) -> float:
    duration_hours = max(0.0, _to_float(time_sec, 0.0)) / 3600.0
    reserve = max(0.0, min(1.2, _to_float(hrr, 0.0)))
    return float(duration_hours * reserve * 0.64 * math.exp(1.92 * reserve))


def _cumulative_trimp_fatigue_state(
    cumulative_trimp: float,
    fatigue_coef: float,
    fatigue_model: str,
    trimp_scale: float,
    min_factor: float,
) -> float:
    _ = trimp_scale  # Deprecated compatibility argument; fatigue now uses raw load.
    load = np.array([max(0.0, _to_float(cumulative_trimp, 0.0))])
    return float(
        _trimp_fatigue_from_load(
            load,
            fatigue_coef=fatigue_coef,
            fatigue_model=fatigue_model,
            min_factor=min_factor,
        )[0]
    )


def _hrr_trimp_grid_terms(
    segments_df: pd.DataFrame,
    v_anchor_kmh: float,
    prediction_kwargs: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distance = pd.to_numeric(segments_df["distanceKm"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    altitude = pd.to_numeric(
        segments_df.get("meanAltitudeM", pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)
    altitude_values = altitude.map(altitude_factor).to_numpy(dtype=float)
    gap = _gap_factors_for_segments(
        segments_df,
        steep_threshold=float(prediction_kwargs.get("gap_steep_threshold", DEFAULT_GAP_STEEP_THRESHOLD)),
        soft_start=float(prediction_kwargs.get("gap_soft_start", DEFAULT_GAP_SOFT_START)),
        climb_scale=float(prediction_kwargs.get("gap_climb_scale", DEFAULT_GAP_CLIMB_SCALE)),
        descent_scale=float(prediction_kwargs.get("gap_descent_scale", DEFAULT_GAP_DESCENT_SCALE)),
    ).to_numpy(dtype=float)
    if bool(prediction_kwargs.get("use_hrr_effort", True)):
        hrr_effort = _hrr_effort_values(
            segments_df,
            hrr_reference=float(prediction_kwargs.get("hrr_reference", 0.70)),
            min_factor=float(prediction_kwargs.get("hrr_min_factor", 0.55)),
            max_factor=float(prediction_kwargs.get("hrr_max_factor", 1.30)),
            hrr_col=str(prediction_kwargs.get("hrr_col", "meanHrReserve")),
        )
    else:
        hrr_effort = np.ones(len(segments_df), dtype=float)
    load_factor_col = prediction_kwargs.get("load_factor_col")
    if load_factor_col:
        load_factor = pd.to_numeric(
            segments_df.get(str(load_factor_col), pd.Series(1.0, index=segments_df.index)),
            errors="coerce",
        ).fillna(1.0)
        load_values = np.clip(load_factor.to_numpy(dtype=float), 0.1, 10.0)
    else:
        load_values = np.ones(len(segments_df), dtype=float)
    denominator = (
        max(float(v_anchor_kmh), 1e-9)
        * altitude_values
        * load_values
        * hrr_effort
        / np.clip(gap, 1e-9, None)
    )
    base_time_terms = distance * 3600.0 / np.clip(denominator, 1e-9, None)
    acute_col = str(prediction_kwargs.get("acute_trimp_col", "decayedTrimpBefore"))
    primary_load = pd.to_numeric(
        segments_df.get(acute_col, pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)
    secondary_col = prediction_kwargs.get("secondary_acute_trimp_col")
    secondary_load = pd.to_numeric(
        segments_df.get(str(secondary_col), pd.Series(0.0, index=segments_df.index)),
        errors="coerce",
    ).fillna(0.0)
    return (
        base_time_terms,
        primary_load.to_numpy(dtype=float),
        secondary_load.to_numpy(dtype=float),
    )


def _hrr_trimp_prediction_from_terms(
    base_time_terms: np.ndarray,
    primary_load: np.ndarray,
    secondary_load: np.ndarray,
    *,
    alpha: float,
    fatigue_coef: float,
    fatigue_model: str,
    min_fatigue_factor: float,
    secondary_fatigue_coef: float,
    secondary_fatigue_model: str,
    has_secondary_fatigue: bool,
) -> np.ndarray:
    primary = _trimp_fatigue_from_load(
        primary_load,
        fatigue_coef=fatigue_coef,
        fatigue_model=fatigue_model,
        min_factor=min_fatigue_factor,
    )
    if has_secondary_fatigue:
        secondary = _trimp_fatigue_from_load(
            secondary_load,
            fatigue_coef=secondary_fatigue_coef,
            fatigue_model=secondary_fatigue_model or fatigue_model,
            min_factor=min_fatigue_factor,
        )
        fatigue = np.clip(primary * secondary, float(min_fatigue_factor), 1.0)
    else:
        fatigue = primary
    speed_factor = np.clip(float(alpha) * fatigue, 1e-9, None)
    return base_time_terms / speed_factor


def simulate_constant_hrr_route(
    segments_df: pd.DataFrame,
    hrr: float,
    v_anchor_kmh: float,
    alpha: float,
    fatigue_coef: float = 0.0,
    fatigue_model: str = "linear",
    hrr_reference: float = 0.70,
    hrr_min_factor: float = 0.55,
    hrr_max_factor: float = 1.30,
    trimp_scale: float = 10.0,
    decay_lambda: float = 0.30,
    min_fatigue_factor: float = 0.50,
    load_factor: float = 1.0,
    use_hrr_effort: bool = True,
    fatigue_input_col: str = "cumTrimpBefore",
    gap_steep_threshold: float = DEFAULT_GAP_STEEP_THRESHOLD,
    gap_soft_start: float = DEFAULT_GAP_SOFT_START,
    gap_climb_scale: float = DEFAULT_GAP_CLIMB_SCALE,
    gap_descent_scale: float = DEFAULT_GAP_DESCENT_SCALE,
) -> pd.DataFrame:
    """Predict a planned route at constant HRR with sequential predicted TRIMP fatigue.

    By default, the speed penalty uses cumulative predicted TRIMP before each segment.
    That makes fixed-HRR performance decline monotonically as acute work is consumed.
    Pass ``fatigue_input_col="decayedTrimpBefore"`` to reproduce the recent-load state.
    """
    if segments_df.empty:
        return segments_df.copy()

    route = segments_df.sort_values(["segmentIndex", "startKm"], na_position="last").copy()
    predicted_rows: list[pd.Series] = []
    cumulative_trimp = 0.0
    decayed_trimp = 0.0
    decay = math.exp(-max(0.0, float(decay_lambda)))

    for _, segment in route.iterrows():
        one_segment = pd.DataFrame([segment]).copy()
        one_segment["meanHrReserve"] = float(hrr)
        one_segment["cumTrimpBefore"] = cumulative_trimp
        one_segment["decayedTrimpBefore"] = decayed_trimp
        one_segment["_preRaceLoadFactor"] = float(load_factor)
        fatigue_state_before = _cumulative_trimp_fatigue_state(
            cumulative_trimp,
            fatigue_coef=fatigue_coef,
            fatigue_model=fatigue_model,
            trimp_scale=trimp_scale,
            min_factor=min_fatigue_factor,
        )
        predicted_time = float(
            predict_hrr_trimp_segment_times(
                one_segment,
                v_anchor_kmh=v_anchor_kmh,
                alpha=alpha,
                fatigue_coef=fatigue_coef,
                fatigue_model=fatigue_model,
                hrr_reference=hrr_reference,
                hrr_min_factor=hrr_min_factor,
                hrr_max_factor=hrr_max_factor,
                trimp_scale=trimp_scale,
                min_fatigue_factor=min_fatigue_factor,
                hrr_col="meanHrReserve",
                acute_trimp_col=fatigue_input_col,
                load_factor_col="_preRaceLoadFactor",
                use_hrr_effort=use_hrr_effort,
                gap_steep_threshold=gap_steep_threshold,
                gap_soft_start=gap_soft_start,
                gap_climb_scale=gap_climb_scale,
                gap_descent_scale=gap_descent_scale,
            ).iloc[0]
        )
        segment_trimp = _segment_trimp_from_prediction(predicted_time, float(hrr))
        output = one_segment.iloc[0].copy()
        output["predictedTimeSec"] = predicted_time
        output["predictedPaceMinKm"] = (
            predicted_time / 60.0 / max(_to_float(output.get("distanceKm"), 0.0), 1e-9)
        )
        output["segmentTrimp"] = segment_trimp
        output["cumTrimp"] = cumulative_trimp + segment_trimp
        output["decayedTrimp"] = segment_trimp + decay * decayed_trimp
        output["predictedFatigueStateBefore"] = fatigue_state_before
        output["predictedFatigueState"] = _cumulative_trimp_fatigue_state(
            cumulative_trimp + segment_trimp,
            fatigue_coef=fatigue_coef,
            fatigue_model=fatigue_model,
            trimp_scale=trimp_scale,
            min_factor=min_fatigue_factor,
        )
        predicted_rows.append(output)
        cumulative_trimp += segment_trimp
        decayed_trimp = segment_trimp + decay * decayed_trimp

    return pd.DataFrame(predicted_rows).reset_index(drop=True)


def simulate_observed_hrr_segments(
    segments_df: pd.DataFrame,
    v_anchor_kmh: float,
    alpha: float,
    fatigue_coef: float = 0.0,
    fatigue_model: str = "linear",
    hrr_reference: float = 0.70,
    hrr_min_factor: float = 0.55,
    hrr_max_factor: float = 1.30,
    trimp_scale: float = 10.0,
    decay_lambda: float = 0.30,
    min_fatigue_factor: float = 0.50,
    load_factor: float = 1.0,
    fallback_hrr: float = 0.70,
    hrr_col: str = "meanHrReserve",
    fatigue_input_col: str = "cumTrimpBefore",
) -> pd.DataFrame:
    """Predict segments sequentially using observed per-segment HRR.

    Segment HRR is observed, but acute TRIMP is accumulated from predicted segment time
    to avoid using actual segment duration inside the prediction. By default, the speed
    penalty uses cumulative predicted TRIMP so that a fixed HRR maps to a decreasing
    acute performance state over time.
    """
    if segments_df.empty:
        return segments_df.copy()

    route = segments_df.sort_values(["segmentIndex", "startKm"], na_position="last").copy()
    predicted_rows: list[pd.Series] = []
    cumulative_trimp = 0.0
    decayed_trimp = 0.0
    cumulative_predicted = 0.0
    cumulative_actual = 0.0
    decay = math.exp(-max(0.0, float(decay_lambda)))

    for _, segment in route.iterrows():
        hrr = _to_float(segment.get(hrr_col), np.nan)
        if not math.isfinite(hrr):
            hrr = float(fallback_hrr)
        hrr = max(0.0, min(1.2, hrr))
        one_segment = pd.DataFrame([segment]).copy()
        one_segment[hrr_col] = hrr
        one_segment["cumTrimpBefore"] = cumulative_trimp
        one_segment["decayedTrimpBefore"] = decayed_trimp
        one_segment["_activityLoadFactor"] = float(load_factor)
        fatigue_state_before = _cumulative_trimp_fatigue_state(
            cumulative_trimp,
            fatigue_coef=fatigue_coef,
            fatigue_model=fatigue_model,
            trimp_scale=trimp_scale,
            min_factor=min_fatigue_factor,
        )
        predicted_time = float(
            predict_hrr_trimp_segment_times(
                one_segment,
                v_anchor_kmh=v_anchor_kmh,
                alpha=alpha,
                fatigue_coef=fatigue_coef,
                fatigue_model=fatigue_model,
                hrr_reference=hrr_reference,
                hrr_min_factor=hrr_min_factor,
                hrr_max_factor=hrr_max_factor,
                trimp_scale=trimp_scale,
                min_fatigue_factor=min_fatigue_factor,
                hrr_col=hrr_col,
                acute_trimp_col=fatigue_input_col,
                load_factor_col="_activityLoadFactor",
                use_hrr_effort=True,
            ).iloc[0]
        )
        actual_time = _to_float(segment.get("actualTimeSec"), np.nan)
        segment_trimp = _segment_trimp_from_prediction(predicted_time, hrr)
        cumulative_trimp += segment_trimp
        decayed_trimp = segment_trimp + decay * decayed_trimp
        cumulative_predicted += predicted_time
        if math.isfinite(actual_time):
            cumulative_actual += actual_time

        output = one_segment.iloc[0].copy()
        output["predictedTimeSec"] = predicted_time
        output["predictedPaceMinKm"] = (
            predicted_time / 60.0 / max(_to_float(output.get("distanceKm"), 0.0), 1e-9)
        )
        output["segmentTrimp"] = segment_trimp
        output["cumTrimp"] = cumulative_trimp
        output["decayedTrimp"] = decayed_trimp
        output["predictedFatigueStateBefore"] = fatigue_state_before
        output["predictedFatigueState"] = _cumulative_trimp_fatigue_state(
            cumulative_trimp,
            fatigue_coef=fatigue_coef,
            fatigue_model=fatigue_model,
            trimp_scale=trimp_scale,
            min_factor=min_fatigue_factor,
        )
        output["cumulativePredictedTimeSec"] = cumulative_predicted
        output["cumulativeActualTimeSec"] = cumulative_actual if math.isfinite(actual_time) else np.nan
        if math.isfinite(actual_time):
            output["errorSec"] = predicted_time - actual_time
            output["errorPct"] = (predicted_time - actual_time) / actual_time * 100.0
            output["cumulativeErrorSec"] = cumulative_predicted - cumulative_actual
        else:
            output["errorSec"] = np.nan
            output["errorPct"] = np.nan
            output["cumulativeErrorSec"] = np.nan
        predicted_rows.append(output)

    return pd.DataFrame(predicted_rows).reset_index(drop=True)


def sweep_constant_hrr_route(
    segments_df: pd.DataFrame,
    hrr_values: Iterable[float],
    v_anchor_kmh: float,
    alpha: float,
    fatigue_coef: float = 0.0,
    fatigue_model: str = "linear",
    hrr_reference: float = 0.70,
    hrr_min_factor: float = 0.55,
    hrr_max_factor: float = 1.30,
    trimp_scale: float = 10.0,
    decay_lambda: float = 0.30,
    min_fatigue_factor: float = 0.50,
    load_factor: float = 1.0,
    envelope_df: Optional[pd.DataFrame] = None,
    hr_rest: Optional[float] = None,
    hr_max: Optional[float] = None,
    fatigue_input_col: str = "cumTrimpBefore",
    use_hrr_effort: bool = True,
    gap_steep_threshold: float = DEFAULT_GAP_STEEP_THRESHOLD,
    gap_soft_start: float = DEFAULT_GAP_SOFT_START,
    gap_climb_scale: float = DEFAULT_GAP_CLIMB_SCALE,
    gap_descent_scale: float = DEFAULT_GAP_DESCENT_SCALE,
) -> pd.DataFrame:
    """Sweep constant-HRR race estimates and mark historically feasible choices."""
    route_distance = float(
        pd.to_numeric(
            segments_df.get("distanceKm", pd.Series(dtype=float)),
            errors="coerce",
        ).sum()
    )
    has_envelope = envelope_df is not None and not envelope_df.empty
    rows: list[dict[str, object]] = []
    for hrr in sorted({float(value) for value in hrr_values}):
        prediction = simulate_constant_hrr_route(
            segments_df,
            hrr=hrr,
            v_anchor_kmh=v_anchor_kmh,
            alpha=alpha,
            fatigue_coef=fatigue_coef,
            fatigue_model=fatigue_model,
            hrr_reference=hrr_reference,
            hrr_min_factor=hrr_min_factor,
            hrr_max_factor=hrr_max_factor,
            trimp_scale=trimp_scale,
            decay_lambda=decay_lambda,
            min_fatigue_factor=min_fatigue_factor,
            load_factor=load_factor,
            fatigue_input_col=fatigue_input_col,
            use_hrr_effort=use_hrr_effort,
            gap_steep_threshold=gap_steep_threshold,
            gap_soft_start=gap_soft_start,
            gap_climb_scale=gap_climb_scale,
            gap_descent_scale=gap_descent_scale,
        )
        total_time = float(pd.to_numeric(prediction.get("predictedTimeSec"), errors="coerce").sum())
        max_duration = max_duration_for_hrr(hrr, envelope_df) if has_envelope else np.nan
        has_limit = math.isfinite(max_duration)
        feasible = bool(total_time <= max_duration) if has_envelope and has_limit else not has_envelope
        heart_rate_bpm = (
            float(hr_rest) + hrr * (float(hr_max) - float(hr_rest))
            if hr_rest is not None and hr_max is not None and hr_max > hr_rest
            else np.nan
        )
        rows.append(
            {
                "hrr": hrr,
                "heartRateBpm": heart_rate_bpm,
                "predictedTimeSec": total_time,
                "predictedTimeMin": total_time / 60.0,
                "predictedTimeHours": total_time / 3600.0,
                "distanceKm": route_distance,
                "avgSpeedKmh": route_distance / (total_time / 3600.0)
                if total_time > 0 and route_distance > 0
                else np.nan,
                "paceMinPerKm": (total_time / 60.0) / route_distance
                if route_distance > 0
                else np.nan,
                "maxSustainableSec": max_duration,
                "maxSustainableHours": max_duration / 3600.0 if has_limit else np.nan,
                "sustainabilityMarginMin": (max_duration - total_time) / 60.0
                if has_limit
                else np.nan,
                "feasible": feasible,
            }
        )
    return pd.DataFrame(rows)


def select_best_constant_hrr(sweep_df: pd.DataFrame) -> pd.Series:
    """Select the fastest feasible HRR row, or least-infeasible row if none is feasible."""
    if sweep_df.empty:
        return pd.Series(dtype=object)
    working = sweep_df.copy()
    feasible_col = working.get("feasible", pd.Series(False, index=working.index))
    feasible = working[feasible_col.astype(bool)]
    if not feasible.empty:
        return feasible.sort_values(["predictedTimeSec", "hrr"], ascending=[True, True]).iloc[0]
    if "sustainabilityMarginMin" in working.columns:
        return working.sort_values(["sustainabilityMarginMin", "predictedTimeSec"], ascending=[False, True]).iloc[0]
    return working.sort_values("predictedTimeSec").iloc[0]


def hrr_trimp_grid_search_model(
    segments_df: pd.DataFrame,
    v_anchor_kmh: float,
    alpha_grid: Iterable[float],
    fatigue_coef_grid: Iterable[float],
    secondary_fatigue_coef_grid: Optional[Iterable[float]] = None,
    fatigue_models: Sequence[str] = ("linear", "exponential"),
    activity_col: str = "activityId",
    objective: str = "race",
    observed_activity_times_sec: Optional[Mapping[str, float]] = None,
    fit_mask_col: Optional[str] = None,
    actual_time_col: str = "actualTimeSec",
    **prediction_kwargs: object,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Grid-search the constrained HRR+TRIMP model with segment and race metrics.

    When ``fit_mask_col`` is set, hyperparameters are selected using only fit-eligible
    segments. Predictions are still produced for every segment so full-race evaluation
    remains available after optimization.

    ``actual_time_col`` selects the segment clock used for fit metrics (e.g.
    ``actualMovingTimeSec`` to strip stationary dwell). Full-race metrics still use
    ``observed_activity_times_sec`` when provided; per-segment ``errorSec`` remains
    versus full ``actualTimeSec``.
    """
    if segments_df.empty:
        empty = pd.DataFrame()
        return {}, empty, empty

    fit_mask = _resolve_fit_mask(segments_df, fit_mask_col, context="hrr_trimp_grid_search_model")
    if actual_time_col not in segments_df.columns:
        logger.warning(
            "hrr_trimp_grid_search_model: actual_time_col=%s missing; "
            "falling back to actualTimeSec",
            actual_time_col,
        )
        actual_time_col = "actualTimeSec"
    fit_actual_segment = pd.to_numeric(segments_df[actual_time_col], errors="coerce")
    full_actual_segment = pd.to_numeric(segments_df["actualTimeSec"], errors="coerce")
    fit_actual_values = fit_actual_segment.to_numpy(dtype=float)
    full_actual_values = full_actual_segment.to_numpy(dtype=float)
    fit_mask_values = fit_mask.to_numpy(dtype=bool)
    # Drop zero-moving segments from the fit set when using moving time.
    if actual_time_col != "actualTimeSec":
        positive_moving = np.isfinite(fit_actual_values) & (fit_actual_values > 1.0)
        if not bool(positive_moving[fit_mask_values].any()):
            logger.warning(
                "hrr_trimp_grid_search_model: no positive %s rows under fit mask; "
                "keeping original mask",
                actual_time_col,
            )
        else:
            fit_mask_values = fit_mask_values & positive_moving
    activity_codes: Optional[np.ndarray] = None
    race_actual_values = np.array([], dtype=float)
    race_actual_fit_values = np.array([], dtype=float)
    if activity_col in segments_df.columns:
        activity_labels, activity_uniques = pd.factorize(segments_df[activity_col].astype(str), sort=False)
        activity_codes = activity_labels.astype(int)
        race_actual_values = np.bincount(
            activity_codes,
            weights=np.nan_to_num(full_actual_values, nan=0.0),
            minlength=len(activity_uniques),
        ).astype(float)
        race_actual_fit_values = np.bincount(
            activity_codes,
            weights=np.nan_to_num(fit_actual_values * fit_mask_values, nan=0.0),
            minlength=len(activity_uniques),
        ).astype(float)
        if observed_activity_times_sec is not None:
            race_actual_values = np.asarray(
                [
                    _to_float(observed_activity_times_sec.get(str(activity_id)), np.nan)
                    for activity_id in activity_uniques.astype(str)
                ],
                dtype=float,
            )
    rows: list[dict[str, object]] = []
    best: Optional[dict[str, object]] = None
    best_prediction_values = np.array([], dtype=float)
    secondary_col = prediction_kwargs.get("secondary_acute_trimp_col")
    secondary_model = str(prediction_kwargs.get("secondary_fatigue_model") or "")
    has_secondary_fatigue = bool(secondary_col)
    min_fatigue_factor = float(prediction_kwargs.get("min_fatigue_factor", 0.50))
    base_time_terms, primary_load, secondary_load = _hrr_trimp_grid_terms(
        segments_df,
        v_anchor_kmh=v_anchor_kmh,
        prediction_kwargs=prediction_kwargs,
    )
    secondary_values = (
        [float(value) for value in secondary_fatigue_coef_grid]
        if secondary_col and secondary_fatigue_coef_grid is not None
        else [0.0]
    )
    for fatigue_model in fatigue_models:
        for alpha in [float(value) for value in alpha_grid]:
            for fatigue_coef in [float(value) for value in fatigue_coef_grid]:
                for secondary_fatigue_coef in secondary_values:
                    predicted_values = _hrr_trimp_prediction_from_terms(
                        base_time_terms,
                        primary_load,
                        secondary_load,
                        alpha=alpha,
                        fatigue_coef=fatigue_coef,
                        fatigue_model=str(fatigue_model),
                        min_fatigue_factor=min_fatigue_factor,
                        secondary_fatigue_coef=secondary_fatigue_coef,
                        secondary_fatigue_model=secondary_model,
                        has_secondary_fatigue=has_secondary_fatigue,
                    )
                    segment_metrics = _regression_metrics_arrays(
                        fit_actual_values[fit_mask_values],
                        predicted_values[fit_mask_values],
                    )
                    segment_metrics_full = _regression_metrics_arrays(
                        full_actual_values,
                        predicted_values,
                    )
                    race_metrics = {
                        "r2": np.nan,
                        "maeSec": np.nan,
                        "mapePct": np.nan,
                        "biasSec": np.nan,
                    }
                    race_metrics_full = {
                        "r2": np.nan,
                        "maeSec": np.nan,
                        "mapePct": np.nan,
                        "biasSec": np.nan,
                    }
                    if activity_codes is not None:
                        race_predicted_values = np.bincount(
                            activity_codes,
                            weights=np.nan_to_num(predicted_values, nan=0.0),
                            minlength=len(race_actual_values),
                        ).astype(float)
                        race_predicted_fit_values = np.bincount(
                            activity_codes,
                            weights=np.nan_to_num(predicted_values * fit_mask_values, nan=0.0),
                            minlength=len(race_actual_fit_values),
                        ).astype(float)
                        # Prefer cleaned / moving-time totals for selection when active.
                        use_fit_only = bool(fit_mask_col) and (not bool(fit_mask_values.all()))
                        use_moving_fit = actual_time_col != "actualTimeSec"
                        if use_fit_only or use_moving_fit:
                            race_metrics = _regression_metrics_arrays(
                                race_actual_fit_values,
                                race_predicted_fit_values,
                            )
                        else:
                            race_metrics = _regression_metrics_arrays(
                                race_actual_values,
                                race_predicted_values,
                            )
                        race_metrics_full = _regression_metrics_arrays(
                            race_actual_values,
                            race_predicted_values,
                        )
                    row = {
                        "alpha": alpha,
                        "fatigueCoef": fatigue_coef,
                        "fatigueModel": str(fatigue_model),
                        "secondaryFatigueCoef": secondary_fatigue_coef,
                        "secondaryFatigueModel": secondary_model,
                        "secondaryAcuteTrimpCol": str(secondary_col or ""),
                        "actualTimeCol": actual_time_col,
                        "segmentR2": segment_metrics["r2"],
                        "segmentMaeSec": segment_metrics["maeSec"],
                        "segmentMapePct": segment_metrics["mapePct"],
                        "segmentBiasSec": segment_metrics["biasSec"],
                        "segmentR2Full": segment_metrics_full["r2"],
                        "segmentMaeSecFull": segment_metrics_full["maeSec"],
                        "segmentMapePctFull": segment_metrics_full["mapePct"],
                        "segmentBiasSecFull": segment_metrics_full["biasSec"],
                        "raceR2": race_metrics["r2"],
                        "raceMaeSec": race_metrics["maeSec"],
                        "raceMapePct": race_metrics["mapePct"],
                        "raceBiasSec": race_metrics["biasSec"],
                        "raceR2Full": race_metrics_full["r2"],
                        "raceMaeSecFull": race_metrics_full["maeSec"],
                        "raceMapePctFull": race_metrics_full["mapePct"],
                        "raceBiasSecFull": race_metrics_full["biasSec"],
                        "fitSegmentCount": int(fit_mask_values.sum()),
                        "fullSegmentCount": int(len(fit_mask_values)),
                    }
                    rows.append(row)
                    if best is None:
                        best = row
                        best_prediction_values = predicted_values
                        continue
                    if objective == "segment":
                        best_r2 = -np.inf if pd.isna(best["segmentR2"]) else best["segmentR2"]
                        row_r2 = -np.inf if pd.isna(row["segmentR2"]) else row["segmentR2"]
                        best_score = (best_r2, -best["segmentMaeSec"])
                        row_score = (row_r2, -row["segmentMaeSec"])
                    else:
                        best_r2 = -np.inf if pd.isna(best["raceR2"]) else best["raceR2"]
                        row_r2 = -np.inf if pd.isna(row["raceR2"]) else row["raceR2"]
                        best_score = (best_r2, -best["raceMaeSec"])
                        row_score = (row_r2, -row["raceMaeSec"])
                    if row_score > best_score:
                        best = row
                        best_prediction_values = predicted_values

    prediction_df = segments_df.copy()
    prediction_df["predictedTimeSec"] = best_prediction_values.astype(float)
    prediction_df["errorSec"] = prediction_df["predictedTimeSec"] - full_actual_segment
    prediction_df["fitActualTimeSec"] = fit_actual_segment
    prediction_df["fitErrorSec"] = prediction_df["predictedTimeSec"] - fit_actual_segment
    prediction_df["isFitEligible"] = fit_mask_values
    assert best is not None
    return best, pd.DataFrame(rows), prediction_df


def leave_one_out_hrr_trimp_grid_search(
    segments_df: pd.DataFrame,
    v_anchor_kmh: float,
    alpha_grid: Iterable[float],
    fatigue_coef_grid: Iterable[float],
    secondary_fatigue_coef_grid: Optional[Iterable[float]] = None,
    fatigue_models: Sequence[str] = ("linear", "exponential"),
    activity_col: str = "activityId",
    observed_activity_times_sec: Optional[Mapping[str, float]] = None,
    objective: str = "race",
    fit_mask_col: Optional[str] = None,
    actual_time_col: str = "actualTimeSec",
    **prediction_kwargs: object,
) -> pd.DataFrame:
    """Leave-one-activity-out validation for the constrained HRR+TRIMP model.

    Hyperparameters are fit on cleaned / moving-time segments of the training
    activities. Each fold still scores the held-out activity on the full race
    (all segments vs observed activity time).
    """
    if segments_df.empty or activity_col not in segments_df.columns:
        return pd.DataFrame()

    folds: list[dict[str, object]] = []
    working = segments_df.copy()
    working[activity_col] = working[activity_col].astype(str)
    fit_mask = _resolve_fit_mask(working, fit_mask_col, context="leave_one_out_hrr_trimp_grid_search")
    working["_fitMask"] = fit_mask.to_numpy(dtype=bool)
    if actual_time_col not in working.columns:
        logger.warning(
            "leave_one_out_hrr_trimp_grid_search: actual_time_col=%s missing; "
            "falling back to actualTimeSec",
            actual_time_col,
        )
        actual_time_col = "actualTimeSec"
    for held_out in working[activity_col].dropna().unique().tolist():
        train = working[working[activity_col].ne(held_out)]
        test = working[working[activity_col].eq(held_out)]
        if train.empty or test.empty:
            continue
        best, _grid, _prediction = hrr_trimp_grid_search_model(
            train,
            v_anchor_kmh=v_anchor_kmh,
            alpha_grid=alpha_grid,
            fatigue_coef_grid=fatigue_coef_grid,
            secondary_fatigue_coef_grid=secondary_fatigue_coef_grid,
            fatigue_models=fatigue_models,
            activity_col=activity_col,
            objective=objective,
            observed_activity_times_sec=observed_activity_times_sec,
            fit_mask_col="_fitMask",
            actual_time_col=actual_time_col,
            **prediction_kwargs,
        )
        predicted_segments = predict_hrr_trimp_segment_times(
            test,
            v_anchor_kmh=v_anchor_kmh,
            alpha=float(best["alpha"]),
            fatigue_coef=float(best["fatigueCoef"]),
            fatigue_model=str(best["fatigueModel"]),
            secondary_fatigue_coef=float(best.get("secondaryFatigueCoef", 0.0)),
            **prediction_kwargs,
        )
        actual = _to_float(
            observed_activity_times_sec.get(str(held_out)) if observed_activity_times_sec else np.nan,
            np.nan,
        )
        if not math.isfinite(actual):
            actual = pd.to_numeric(test["actualTimeSec"], errors="coerce").sum()
        predicted = float(predicted_segments.sum())
        test_mask = test["_fitMask"].to_numpy(dtype=bool)
        fit_actual_arr = pd.to_numeric(test[actual_time_col], errors="coerce").to_numpy(dtype=float)
        if actual_time_col != "actualTimeSec":
            test_mask = test_mask & np.isfinite(fit_actual_arr) & (fit_actual_arr > 1.0)
        actual_fit = float(fit_actual_arr[test_mask].sum())
        predicted_fit = float(np.asarray(predicted_segments, dtype=float)[test_mask].sum())
        folds.append(
            {
                "activityId": held_out,
                "alpha": best["alpha"],
                "fatigueCoef": best["fatigueCoef"],
                "fatigueModel": best["fatigueModel"],
                "secondaryFatigueCoef": best.get("secondaryFatigueCoef", 0.0),
                "secondaryFatigueModel": best.get("secondaryFatigueModel", ""),
                "secondaryAcuteTrimpCol": best.get("secondaryAcuteTrimpCol", ""),
                "actualTimeSec": float(actual),
                "predictedTimeSec": predicted,
                "errorSec": float(predicted - actual),
                "errorPct": float((predicted - actual) / actual * 100.0)
                if actual > 0
                else np.nan,
                "actualFitEligibleSec": actual_fit,
                "predictedFitEligibleSec": predicted_fit,
                "fitEligibleErrorSec": float(predicted_fit - actual_fit),
                "fitEligibleSegmentCount": int(test_mask.sum()),
                "excludedSegmentCount": int((~test_mask).sum()),
                "excludedTimeSec": float(
                    pd.to_numeric(test["actualTimeSec"], errors="coerce").to_numpy()[~test_mask].sum()
                ),
                "actualTimeCol": actual_time_col,
            }
        )
    return pd.DataFrame(folds)



def grouped_segment_metrics(
    prediction_df: pd.DataFrame,
    group_col: str = "terrainFamily",
) -> pd.DataFrame:
    """Compute segment prediction metrics by a grouping column."""
    if prediction_df.empty or group_col not in prediction_df.columns:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for group_name, group_df in prediction_df.groupby(group_col, sort=True):
        metrics = regression_metrics(group_df["actualTimeSec"], group_df["predictedTimeSec"])
        rows.append(
            {
                group_col: group_name,
                "count": int(len(group_df)),
                "r2": metrics["r2"],
                "maeSec": metrics["maeSec"],
                "mapePct": metrics["mapePct"],
                "biasSec": metrics["biasSec"],
            }
        )
    return pd.DataFrame(rows)


def fit_linear_regression(df: pd.DataFrame, feature_cols: Sequence[str], target_col: str) -> RegressionModel:
    """Fit a small least-squares model with median imputation."""
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    working = df.copy()
    x_values = []
    for col in feature_cols:
        series = pd.to_numeric(working[col], errors="coerce")
        fill_value = float(series.median()) if series.notna().any() else 0.0
        x_values.append(series.fillna(fill_value).to_numpy(dtype=float))
    x = np.column_stack([np.ones(len(working)), *x_values])
    y = pd.to_numeric(working[target_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    if valid.sum() < len(feature_cols) + 1:
        raise ValueError("not enough valid rows for regression")
    coefficients, *_ = np.linalg.lstsq(x[valid], y[valid], rcond=None)
    return RegressionModel(tuple(feature_cols), coefficients)


def predict_linear_regression(df: pd.DataFrame, model: RegressionModel) -> np.ndarray:
    """Predict with a RegressionModel."""
    x_values = []
    for col in model.feature_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        fill_value = float(series.median()) if series.notna().any() else 0.0
        x_values.append(series.fillna(fill_value).to_numpy(dtype=float))
    x = np.column_stack([np.ones(len(df)), *x_values])
    return x @ model.coefficients


def forbidden_anonymized_columns(
    columns: Iterable[object],
    forbidden_names: Iterable[str] = DEFAULT_FORBIDDEN_ANONYMIZED_COLUMNS,
) -> list[str]:
    """Return columns that should not appear in anonymized paper exports."""
    forbidden = {str(name).lower() for name in forbidden_names}
    return [str(column) for column in columns if str(column).lower() in forbidden]
