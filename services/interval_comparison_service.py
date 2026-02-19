"""Copyright (C) 2025 Pierre Marrec
SPDX-License-Identifier: GPL-3.0-or-later
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from persistence.csv_storage import CsvStorage
from services.interval_utils import normalize_steps
from services.lap_metrics_service import LapMetricsService
from services.timeseries_service import TimeseriesService
from utils.coercion import safe_float_optional, safe_int_optional
from utils.config import Config


@dataclass(frozen=True)
class PlannedSegment:
    index: int
    section_label: str
    kind: str
    sec: int
    target_type: Optional[str]
    target_label: Optional[str]
    ascend_m: float
    descend_m: float


@dataclass(frozen=True)
class LapSegment:
    lap_index: int
    kind: str
    start_time: Optional[pd.Timestamp]
    time_sec: float
    distance_km: Optional[float]
    avg_speed_kmh: Optional[float]
    distance_eq_km: Optional[float]
    avg_hr: Optional[float]
    ascent_m: float
    descent_m: float


@dataclass(frozen=True)
class AggregatedMetrics:
    time_sec: float
    distance_m: Optional[float]
    avg_speed_kmh: Optional[float]
    speed_eq_kmh: Optional[float]
    pace_min_km: Optional[float]
    avg_hr: Optional[float]
    ascent_m: float
    descent_m: float


@dataclass(frozen=True)
class MatchedSegment:
    planned: Optional[PlannedSegment]
    laps: Tuple[LapSegment, ...]
    aggregated_metrics: AggregatedMetrics
    match_status: str  # matched | planned_only | actual_only


@dataclass(frozen=True)
class StructuralGroup:
    kind: str  # pre | loop | between | post | buffer
    segments: Tuple[PlannedSegment, ...]
    expected_total_sec: int
    expected_count: int
    min_laps: int
    max_laps: int


@dataclass
class IntervalComparisonService:
    storage: CsvStorage
    config: Config
    lap_metrics_service: Optional[LapMetricsService] = None
    timeseries_service: Optional[TimeseriesService] = None

    def __post_init__(self) -> None:
        self.lap_metrics_service = self.lap_metrics_service or LapMetricsService(
            storage=self.storage,
            config=self.config,
        )
        self.timeseries_service = self.timeseries_service or TimeseriesService(self.config)

    def compare(self, activity_id: str, planned_session: Dict[str, object]) -> List[MatchedSegment]:
        steps_payload = self._steps_payload(planned_session.get("stepsJson"))
        planned_segments = self.flatten_planned_segments(steps_payload)

        laps_df = self.lap_metrics_service.load(str(activity_id))
        if laps_df is None:
            laps_df = pd.DataFrame()
        lap_segments = self._build_lap_segments(str(activity_id), laps_df)

        matches = self.match_segments_to_laps(planned_segments, lap_segments)
        matches = self._absorb_trailing_actual_into_last_cooldown(matches)
        matches = self._move_actual_unmatched_to_end(matches)
        return self._fuse_consecutive_actual_unmatched(matches)

    @staticmethod
    def _steps_payload(value: object) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return {}
            try:
                parsed = json.loads(stripped)
            except Exception:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def flatten_planned_segments(self, raw_steps: Dict[str, Any]) -> List[PlannedSegment]:
        steps = normalize_steps(raw_steps)
        segments: List[PlannedSegment] = []
        idx = 1

        def _append(section_label: str, action: Dict[str, Any]) -> None:
            nonlocal idx
            kind = str(action.get("kind") or "run").strip().lower()
            sec = max(int(float(action.get("sec") or 0)), 0)
            if sec <= 0:
                return
            segments.append(
                PlannedSegment(
                    index=idx,
                    section_label=section_label,
                    kind=kind if kind in {"run", "recovery"} else "run",
                    sec=sec,
                    target_type=self._clean_opt_str(action.get("targetType")),
                    target_label=self._clean_opt_str(action.get("targetLabel")),
                    ascend_m=float(max(int(float(action.get("ascendM") or 0)), 0)),
                    descend_m=float(max(int(float(action.get("descendM") or 0)), 0)),
                )
            )
            idx += 1

        for block_index, block in enumerate(steps.get("preBlocks") or [], start=1):
            _append(f"Avant {block_index}", block)

        loops = steps.get("loops") or []
        between_block = steps.get("betweenBlock")
        post_blocks = steps.get("postBlocks") or []
        has_between_block = (
            between_block is not None and max(int(float(between_block.get("sec") or 0)), 0) > 0
        )
        use_post_blocks_between_loops = len(loops) > 1 and not has_between_block and bool(post_blocks)

        for loop_index, loop in enumerate(loops, start=1):
            repeats = max(int(float(loop.get("repeats") or 1)), 1)
            actions = loop.get("actions") or []
            for repeat_index in range(1, repeats + 1):
                for action_index, action in enumerate(actions, start=1):
                    _append(f"Boucle {loop_index} R{repeat_index}/{repeats} A{action_index}", action)
            if loop_index < len(loops):
                if has_between_block and between_block:
                    _append(f"Entre blocs {loop_index}", between_block)
                elif use_post_blocks_between_loops:
                    for block_index, block in enumerate(post_blocks, start=1):
                        _append(f"Entre blocs {loop_index}.{block_index}", block)

        if not use_post_blocks_between_loops:
            for block_index, block in enumerate(post_blocks, start=1):
                _append(f"Apres {block_index}", block)

        return segments

    def compute_lap_descent(self, activity_id: str, laps_df: pd.DataFrame) -> List[float]:
        if laps_df.empty:
            return []
        ts_df = self.timeseries_service.load(str(activity_id))
        if ts_df is None or ts_df.empty:
            return [0.0] * len(laps_df)
        if "timestamp" not in ts_df.columns or "elevationM" not in ts_df.columns:
            return [0.0] * len(laps_df)

        working = ts_df[["timestamp", "elevationM"]].copy()
        working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce", utc=True)
        working["elevationM"] = pd.to_numeric(working["elevationM"], errors="coerce")
        working = working.dropna(subset=["timestamp", "elevationM"]).sort_values("timestamp")
        if working.empty:
            return [0.0] * len(laps_df)

        descents: List[float] = []
        for _, lap_row in laps_df.iterrows():
            start = pd.to_datetime(lap_row.get("startTime"), errors="coerce", utc=True)
            time_sec = safe_float_optional(lap_row.get("timeSec")) or 0.0
            if pd.isna(start) or time_sec <= 0:
                descents.append(0.0)
                continue
            end = start + pd.to_timedelta(time_sec, unit="s")
            mask = (working["timestamp"] >= start) & (working["timestamp"] <= end)
            lap_slice = working.loc[mask, "elevationM"]
            if lap_slice.size < 2:
                descents.append(0.0)
                continue
            diffs = lap_slice.diff().dropna()
            descent = float((-diffs[diffs < 0]).sum())
            descents.append(round(max(descent, 0.0), 1))
        return descents

    def match_segments_to_laps(
        self,
        planned_segments: Sequence[PlannedSegment],
        lap_segments: Sequence[LapSegment],
    ) -> List[MatchedSegment]:
        if not planned_segments and not lap_segments:
            return []
        if not planned_segments:
            return [
                MatchedSegment(
                    planned=None,
                    laps=(lap,),
                    aggregated_metrics=self.aggregate_metrics((lap,)),
                    match_status="actual_only",
                )
                for lap in lap_segments
            ]
        if not lap_segments:
            return [
                MatchedSegment(
                    planned=segment,
                    laps=(),
                    aggregated_metrics=self.aggregate_metrics(()),
                    match_status="planned_only",
                )
                for segment in planned_segments
            ]

        results: List[MatchedSegment] = []
        groups = self._build_structural_groups(planned_segments, len(lap_segments))
        group_regions = self._dp_region_split(groups, lap_segments)
        for group, region_laps in group_regions:
            results.extend(self._assign_region_to_segments(group, region_laps))
        return results

    def _build_structural_groups(
        self,
        planned_segments: Sequence[PlannedSegment],
        lap_count: int,
    ) -> List[StructuralGroup]:
        groups: List[StructuralGroup] = [
            StructuralGroup(
                kind="buffer",
                segments=(),
                expected_total_sec=0,
                expected_count=0,
                min_laps=0,
                max_laps=lap_count,
            )
        ]
        current_kind: Optional[str] = None
        current_key: Optional[str] = None
        current_segments: List[PlannedSegment] = []

        for segment in planned_segments:
            kind, group_key = self._structural_kind_and_key(segment.section_label)
            if current_segments and (kind != current_kind or group_key != current_key):
                groups.append(self._make_structural_group(current_kind or "other", current_segments, lap_count))
                current_segments = []
            if not current_segments:
                current_kind = kind
                current_key = group_key
            current_segments.append(segment)

        if current_segments:
            groups.append(self._make_structural_group(current_kind or "other", current_segments, lap_count))

        groups.append(
            StructuralGroup(
                kind="buffer",
                segments=(),
                expected_total_sec=0,
                expected_count=0,
                min_laps=0,
                max_laps=lap_count,
            )
        )
        return groups

    @staticmethod
    def _structural_kind_and_key(section_label: str) -> Tuple[str, str]:
        text = section_label.strip()
        if text.startswith("Boucle "):
            tail = text[len("Boucle ") :]
            return "loop", IntervalComparisonService._leading_digits(tail) or text
        if text.startswith("Entre blocs "):
            tail = text[len("Entre blocs ") :]
            return "between", IntervalComparisonService._leading_digits(tail) or text
        if text.startswith("Avant "):
            return "pre", text
        if text.startswith("Apres "):
            return "post", text
        return "other", text

    @staticmethod
    def _leading_digits(text: str) -> str:
        out = []
        for char in text:
            if char.isdigit():
                out.append(char)
                continue
            if out:
                break
        return "".join(out)

    @staticmethod
    def _make_structural_group(
        kind: str,
        segments: Sequence[PlannedSegment],
        lap_count: int,
    ) -> StructuralGroup:
        expected_total_sec = sum(max(int(segment.sec), 0) for segment in segments)
        expected_count = len(segments)

        if kind == "loop":
            min_laps = max(0, expected_count - max(expected_count // 3, 1))
            max_laps = min(lap_count, expected_count + 3)
        elif kind == "between":
            min_laps = 0
            max_laps = min(lap_count, max(2, expected_count + 2))
        elif kind in {"pre", "post"}:
            min_laps = 0
            max_laps = min(lap_count, max(3, expected_count + 4))
        else:
            min_laps = 0
            max_laps = min(lap_count, max(2, expected_count + 2))

        max_laps = max(max_laps, min_laps)
        return StructuralGroup(
            kind=kind,
            segments=tuple(segments),
            expected_total_sec=expected_total_sec,
            expected_count=expected_count,
            min_laps=min_laps,
            max_laps=max_laps,
        )

    def _dp_region_split(
        self,
        groups: Sequence[StructuralGroup],
        laps: Sequence[LapSegment],
    ) -> List[Tuple[StructuralGroup, Tuple[LapSegment, ...]]]:
        if not groups:
            return []

        group_count = len(groups)
        lap_count = len(laps)
        inf = float("inf")
        dp = [[inf] * (lap_count + 1) for _ in range(group_count + 1)]
        prev: List[List[Optional[Tuple[int, int]]]] = [
            [None] * (lap_count + 1) for _ in range(group_count + 1)
        ]
        dp[0][0] = 0.0

        for group_idx, group in enumerate(groups):
            for used_laps in range(lap_count + 1):
                base_cost = dp[group_idx][used_laps]
                if base_cost == inf:
                    continue
                min_take = max(0, group.min_laps)
                max_take = min(group.max_laps, lap_count - used_laps)
                for take in range(min_take, max_take + 1):
                    region = laps[used_laps : used_laps + take]
                    candidate_cost = base_cost + self._group_region_cost(group, region)
                    if candidate_cost + 1e-9 < dp[group_idx + 1][used_laps + take]:
                        dp[group_idx + 1][used_laps + take] = candidate_cost
                        prev[group_idx + 1][used_laps + take] = (used_laps, take)

        if dp[group_count][lap_count] == inf:
            # Keep output stable even if constraints become inconsistent.
            fallback: List[Tuple[StructuralGroup, Tuple[LapSegment, ...]]] = []
            for idx, group in enumerate(groups):
                region = tuple(laps) if idx == (len(groups) - 1) else tuple()
                fallback.append((group, region))
            return fallback

        assignments: List[Tuple[StructuralGroup, Tuple[LapSegment, ...]]] = []
        used_laps = lap_count
        for group_idx in range(group_count, 0, -1):
            step = prev[group_idx][used_laps]
            if step is None:
                break
            start, take = step
            group = groups[group_idx - 1]
            assignments.append((group, tuple(laps[start : start + take])))
            used_laps = start
        assignments.reverse()
        return assignments

    def _group_region_cost(
        self,
        group: StructuralGroup,
        laps: Sequence[LapSegment],
    ) -> float:
        lap_count = len(laps)
        if group.kind == "buffer":
            return 0.05 * lap_count

        expected_total = max(float(group.expected_total_sec), 1.0)
        actual_total = sum(max(float(lap.time_sec), 0.0) for lap in laps)
        duration_error = abs(actual_total - float(group.expected_total_sec)) / expected_total

        expected_count = max(group.expected_count, 1)
        count_error = abs(lap_count - group.expected_count) / float(expected_count)

        if group.kind == "loop":
            base = (0.4 * duration_error) + (0.6 * count_error)
            if lap_count == 0:
                base += 0.5
            return base

        if group.kind in {"pre", "post"}:
            return duration_error + (0.10 * count_error)
        if group.kind == "between":
            return duration_error + (0.25 * count_error)
        return duration_error + (0.20 * count_error)

    def _assign_region_to_segments(
        self,
        group: StructuralGroup,
        region_laps: Sequence[LapSegment],
    ) -> List[MatchedSegment]:
        if group.kind == "buffer" or not group.segments:
            return [
                MatchedSegment(
                    planned=None,
                    laps=(lap,),
                    aggregated_metrics=self.aggregate_metrics((lap,)),
                    match_status="actual_only",
                )
                for lap in region_laps
            ]

        rows: List[MatchedSegment] = []
        lap_cursor = 0
        segments = group.segments
        for seg_idx, segment in enumerate(segments):
            remaining_laps = region_laps[lap_cursor:]
            remaining_segments = len(segments) - seg_idx
            if not remaining_laps:
                rows.append(
                    MatchedSegment(
                        planned=segment,
                        laps=(),
                        aggregated_metrics=self.aggregate_metrics(()),
                        match_status="planned_only",
                    )
                )
                continue

            take = self._laps_for_segment(segment, group.kind, remaining_laps, remaining_segments)
            matched_laps = tuple(remaining_laps[:take]) if take > 0 else tuple()
            if matched_laps:
                rows.append(
                    MatchedSegment(
                        planned=segment,
                        laps=matched_laps,
                        aggregated_metrics=self.aggregate_metrics(matched_laps),
                        match_status="matched",
                    )
                )
            else:
                rows.append(
                    MatchedSegment(
                        planned=segment,
                        laps=(),
                        aggregated_metrics=self.aggregate_metrics(()),
                        match_status="planned_only",
                    )
                )
            lap_cursor += len(matched_laps)

        while lap_cursor < len(region_laps):
            lap = region_laps[lap_cursor]
            rows.append(
                MatchedSegment(
                    planned=None,
                    laps=(lap,),
                    aggregated_metrics=self.aggregate_metrics((lap,)),
                    match_status="actual_only",
                )
            )
            lap_cursor += 1

        return rows

    def _laps_for_segment(
        self,
        segment: PlannedSegment,
        group_kind: str,
        laps: Sequence[LapSegment],
        remaining_segments: int,
    ) -> int:
        if not laps:
            return 0

        if self._is_short_segment(segment, group_kind):
            return self._short_segment_lap_take(segment, laps, remaining_segments)
        return self._long_segment_lap_take(segment, laps, remaining_segments)

    @staticmethod
    def _is_short_segment(segment: PlannedSegment, group_kind: str) -> bool:
        if group_kind == "loop" and segment.sec <= 150:
            return True
        return segment.sec <= 90

    @staticmethod
    def _short_segment_lap_take(
        segment: PlannedSegment,
        laps: Sequence[LapSegment],
        remaining_segments: int,
    ) -> int:
        if len(laps) < 2:
            return 1
        # Merge only when needed: when a single lap is clearly too short for this segment.
        can_take_two = (len(laps) - 2) >= max(remaining_segments - 1, 0)
        if not can_take_two:
            return 1

        first = max(float(laps[0].time_sec), 0.0)
        second = max(float(laps[1].time_sec), 0.0)
        single_error = abs(first - float(segment.sec))
        merged_error = abs((first + second) - float(segment.sec))
        if first < (float(segment.sec) * 0.65) and merged_error + 1e-6 < single_error:
            return 2
        return 1

    @staticmethod
    def _long_segment_lap_take(
        segment: PlannedSegment,
        laps: Sequence[LapSegment],
        remaining_segments: int,
    ) -> int:
        min_keep = max(remaining_segments - 1, 0)
        max_take = max(1, len(laps) - min_keep)
        target_time = max(float(segment.sec) * 0.60, 1.0)
        totals: List[float] = []
        running_total = 0.0

        for idx in range(max_take):
            running_total += max(float(laps[idx].time_sec), 0.0)
            totals.append(running_total)

        candidates = [idx + 1 for idx, total in enumerate(totals) if total >= target_time]
        if not candidates:
            candidates = [max_take]

        best_take = min(
            candidates,
            key=lambda take: (abs(totals[take - 1] - float(segment.sec)), take),
        )
        return max(1, best_take)

    @staticmethod
    def aggregate_metrics(laps: Sequence[LapSegment]) -> AggregatedMetrics:
        if not laps:
            return AggregatedMetrics(
                time_sec=0.0,
                distance_m=None,
                avg_speed_kmh=None,
                speed_eq_kmh=None,
                pace_min_km=None,
                avg_hr=None,
                ascent_m=0.0,
                descent_m=0.0,
            )

        total_time = sum(max(float(lap.time_sec or 0.0), 0.0) for lap in laps)
        total_distance_km = sum(float(lap.distance_km or 0.0) for lap in laps)
        total_distance_eq_km = sum(float(lap.distance_eq_km or 0.0) for lap in laps)
        total_ascent_m = sum(float(lap.ascent_m or 0.0) for lap in laps)
        total_descent_m = sum(float(lap.descent_m or 0.0) for lap in laps)

        avg_speed_kmh: Optional[float] = None
        speed_eq_kmh: Optional[float] = None
        pace_min_km: Optional[float] = None
        if total_time > 0:
            avg_speed_kmh = (total_distance_km * 3600.0) / total_time if total_distance_km > 0 else None
            speed_eq_kmh = (
                (total_distance_eq_km * 3600.0) / total_time if total_distance_eq_km > 0 else None
            )
            if avg_speed_kmh and avg_speed_kmh > 0:
                pace_min_km = 60.0 / avg_speed_kmh

        weighted_hr = 0.0
        weighted_time = 0.0
        for lap in laps:
            if lap.avg_hr is None or lap.time_sec <= 0:
                continue
            weighted_hr += lap.avg_hr * lap.time_sec
            weighted_time += lap.time_sec
        avg_hr = (weighted_hr / weighted_time) if weighted_time > 0 else None

        return AggregatedMetrics(
            time_sec=round(total_time, 1),
            distance_m=round(total_distance_km * 1000.0, 1) if total_distance_km > 0 else None,
            avg_speed_kmh=round(avg_speed_kmh, 2) if avg_speed_kmh is not None else None,
            speed_eq_kmh=round(speed_eq_kmh, 2) if speed_eq_kmh is not None else None,
            pace_min_km=round(pace_min_km, 2) if pace_min_km is not None else None,
            avg_hr=round(avg_hr, 1) if avg_hr is not None else None,
            ascent_m=round(total_ascent_m, 1),
            descent_m=round(total_descent_m, 1),
        )

    def _build_lap_segments(self, activity_id: str, laps_df: pd.DataFrame) -> List[LapSegment]:
        if laps_df.empty:
            return []
        descents = self.compute_lap_descent(activity_id, laps_df)
        out: List[LapSegment] = []
        for index, row in laps_df.reset_index(drop=True).iterrows():
            out.append(
                LapSegment(
                    lap_index=safe_int_optional(row.get("lapIndex")) or (index + 1),
                    kind=self._lap_kind(row),
                    start_time=self._as_timestamp(row.get("startTime")),
                    time_sec=safe_float_optional(row.get("timeSec")) or 0.0,
                    distance_km=safe_float_optional(row.get("distanceKm")),
                    avg_speed_kmh=safe_float_optional(row.get("avgSpeedKmh")),
                    distance_eq_km=safe_float_optional(row.get("distanceEqKm")),
                    avg_hr=safe_float_optional(row.get("avgHr")),
                    ascent_m=safe_float_optional(row.get("ascentM")) or 0.0,
                    descent_m=descents[index] if index < len(descents) else 0.0,
                )
            )
        return out

    @staticmethod
    def _lap_kind(row: pd.Series) -> str:
        label = str(row.get("label") or "").strip().lower()
        if label.startswith("run"):
            return "run"
        if label.startswith("recovery"):
            return "recovery"
        return "unknown"

    @staticmethod
    def _clean_opt_str(value: object) -> Optional[str]:
        if value in (None, ""):
            return None
        return str(value).strip() or None

    @staticmethod
    def _as_timestamp(value: object) -> Optional[pd.Timestamp]:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        return None if pd.isna(ts) else ts

    def _fuse_consecutive_actual_unmatched(
        self, rows: Sequence[MatchedSegment]
    ) -> List[MatchedSegment]:
        fused: List[MatchedSegment] = []
        for row in rows:
            if (
                fused
                and row.match_status == "actual_only"
                and fused[-1].match_status == "actual_only"
            ):
                merged_laps = fused[-1].laps + row.laps
                fused[-1] = MatchedSegment(
                    planned=None,
                    laps=merged_laps,
                    aggregated_metrics=self.aggregate_metrics(merged_laps),
                    match_status="actual_only",
                )
            else:
                fused.append(row)
        return fused

    def _absorb_trailing_actual_into_last_cooldown(
        self, rows: Sequence[MatchedSegment]
    ) -> List[MatchedSegment]:
        if not rows:
            return []

        trailing_actual_start = len(rows)
        for idx in range(len(rows) - 1, -1, -1):
            if rows[idx].match_status == "actual_only":
                trailing_actual_start = idx
                continue
            break

        if trailing_actual_start >= len(rows):
            return list(rows)

        trailing_actual_rows = list(rows[trailing_actual_start:])
        if not trailing_actual_rows:
            return list(rows)

        target_idx: Optional[int] = None
        for idx in range(trailing_actual_start - 1, -1, -1):
            planned = rows[idx].planned
            if planned is None:
                continue
            if planned.section_label.startswith("Apres"):
                target_idx = idx
                break

        if target_idx is None:
            return list(rows)

        trailing_laps: Tuple[LapSegment, ...] = tuple(
            lap for row in trailing_actual_rows for lap in row.laps
        )
        if not trailing_laps:
            return list(rows)

        target_row = rows[target_idx]
        merged_laps = target_row.laps + trailing_laps
        merged_status = "matched" if merged_laps else target_row.match_status
        merged_row = MatchedSegment(
            planned=target_row.planned,
            laps=merged_laps,
            aggregated_metrics=self.aggregate_metrics(merged_laps),
            match_status=merged_status,
        )

        out = list(rows[:trailing_actual_start])
        out[target_idx] = merged_row
        return out

    @staticmethod
    def _move_actual_unmatched_to_end(rows: Sequence[MatchedSegment]) -> List[MatchedSegment]:
        matched_rows = [row for row in rows if row.match_status != "actual_only"]
        actual_only_rows = [row for row in rows if row.match_status == "actual_only"]
        return matched_rows + actual_only_rows
