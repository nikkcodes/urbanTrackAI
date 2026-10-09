"""
UrbanTrack AI — Timestamp Parsing, Synchronization & Chronological Ordering.

Implements authoritative clock offset loading from official scenario metadata
and enforces direction-independent temporal association logic.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

from layer2.ingestion.canonical_models import CanonicalTracklet, ChronologicalRelationship

TIME_PATTERN = re.compile(r"^(\d{2}):(\d{2}):(\d{2})\.(\d{3})$")


def parse_raw_timestamp(ts_str: str) -> float:
    """
    Parses a raw observation timestamp 'HH:MM:SS.mmm' into float seconds.

    Raises:
        ValueError: If ts_str does not strictly match 'HH:MM:SS.mmm'.
    """
    if not isinstance(ts_str, str):
        raise ValueError(f"Timestamp must be string, got {type(ts_str)}: {ts_str}")
    match = TIME_PATTERN.match(ts_str.strip())
    if not match:
        raise ValueError(f"Invalid timestamp format: '{ts_str}'. Expected 'HH:MM:SS.mmm'")
    h, m, s, ms = match.groups()
    return int(h) * 3600.0 + int(m) * 60.0 + int(s) + int(ms) / 1000.0


def format_timestamp_seconds(sec: float) -> str:
    """
    Converts float seconds into a canonical 'HH:MM:SS.mmm' timestamp string.
    """
    if sec < 0.0 or math.isnan(sec) or math.isinf(sec):
        raise ValueError(f"Cannot format non-positive/non-finite seconds: {sec}")
    total_ms = int(round(sec * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


class TimestampSynchronizer:
    """
    Loads official camera start offsets from CityFlowV2 / AI City Challenge metadata.
    """

    def __init__(
        self,
        primary_dir: Union[str, Path] = "data/cityflowv2/cam_timestamp",
        fallback_dir: Union[str, Path] = "/Users/yanalavivekreddy/Downloads/AICity22_Track1_MTMC_Tracking/cam_timestamp",
    ) -> None:
        self.primary_dir = Path(primary_dir)
        self.fallback_dir = Path(fallback_dir)
        self._offsets: Dict[Tuple[str, str], float] = {}
        self._load_all_scenarios()

    def _normalize_cam_num(self, camera_id: str) -> str:
        """Normalizes 'CAM_S01_C001' or 'C001' or 'c001' to lowercase 'c001'."""
        clean = camera_id.strip().lower()
        if "_" in clean:
            clean = clean.split("_")[-1]
        return clean

    def _load_scenario(self, scenario_id: str) -> None:
        fname = f"{scenario_id}.txt"
        path = self.primary_dir / fname
        if not path.is_file():
            path = self.fallback_dir / fname
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing camera timestamp file for scenario {scenario_id} in {self.primary_dir} and {self.fallback_dir}"
            )

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    cnum = parts[0].strip().lower()
                    offset = float(parts[1].strip())
                    self._offsets[(scenario_id, cnum)] = offset

    def _load_all_scenarios(self) -> None:
        for sc in ["S01", "S02", "S03", "S04", "S05", "S06"]:
            self._load_scenario(sc)

    def get_offset(self, scenario_id: str, camera_id: str) -> float:
        """
        Retrieves the official clock offset in seconds for the given camera.
        """
        cnum = self._normalize_cam_num(camera_id)
        key = (scenario_id, cnum)
        if key not in self._offsets:
            raise KeyError(f"No official timestamp offset found for scenario {scenario_id}, camera {camera_id} ({cnum})")
        return self._offsets[key]

    def synchronize_raw_seconds(self, raw_sec: float, scenario_id: str, camera_id: str) -> float:
        """
        Computes synchronized time: t_sync = t_raw + offset.
        """
        offset = self.get_offset(scenario_id, camera_id)
        return raw_sec + offset

    def synchronize_raw_timestamp(self, ts_str: str, scenario_id: str, camera_id: str) -> Tuple[float, str]:
        """
        Returns (sync_seconds, sync_timestamp_str).
        """
        raw_sec = parse_raw_timestamp(ts_str)
        sync_sec = self.synchronize_raw_seconds(raw_sec, scenario_id, camera_id)
        return sync_sec, format_timestamp_seconds(sync_sec)


def compare_chronological_order(
    tracklet_a: CanonicalTracklet,
    tracklet_b: CanonicalTracklet,
) -> ChronologicalRelationship:
    """
    Determines the chronological relationship between two tracklets strictly using
    synchronized timestamps, independent of camera identifier string ordering.

    Invariant:
        If start_sync_seconds(A) <= start_sync_seconds(B):
            origin = A, destination = B, direction = 'A_TO_B'
        Else:
            origin = B, destination = A, direction = 'B_TO_A'

        delta_t_seconds = destination.start_sync_seconds - origin.end_sync_seconds
    """
    t_a_start = tracklet_a.temporal.start_sync_seconds
    t_a_end = tracklet_a.temporal.end_sync_seconds

    t_b_start = tracklet_b.temporal.start_sync_seconds
    t_b_end = tracklet_b.temporal.end_sync_seconds

    if t_a_start <= t_b_start:
        origin = tracklet_a
        dest = tracklet_b
        direction = "A_TO_B"
        t_orig_dep = t_a_end
        t_dest_arr = t_b_start
    else:
        origin = tracklet_b
        dest = tracklet_a
        direction = "B_TO_A"
        t_orig_dep = t_b_end
        t_dest_arr = t_a_start

    # Physical transit gap between departure from origin and arrival at destination
    delta_t = t_dest_arr - t_orig_dep

    # Overlap occurs if destination arrived before origin departed
    overlap = max(0.0, t_orig_dep - t_dest_arr)

    return ChronologicalRelationship(
        origin_tracklet_id=origin.canonical_id,
        destination_tracklet_id=dest.canonical_id,
        chronological_direction=direction,
        delta_t_seconds=round(delta_t, 4),
        is_chronologically_valid=(dest.temporal.start_sync_seconds >= origin.temporal.start_sync_seconds),
        t_origin_departure_sync=round(t_orig_dep, 4),
        t_dest_arrival_sync=round(t_dest_arr, 4),
        overlap_seconds=round(overlap, 4),
    )
