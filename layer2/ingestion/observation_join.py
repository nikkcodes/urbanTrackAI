"""
UrbanTrack AI — Observation to Tracklet Join Implementation.

Performs deterministic joining of frame-level observations with ByteTrack trajectory tracklets,
enforcing frame-boundary invariants and extracting aggregated tracklet attributes.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from layer2.ingestion.canonical_models import ANPRData
from layer2.ingestion.ocr_aggregation import aggregate_tracklet_ocr


class JoinInvariantError(Exception):
    """Raised when frame boundaries between trajectories.json and observations.json diverge."""

    def __init__(
        self,
        camera_id: str,
        track_id: int,
        expected_start: int,
        actual_start: int,
        expected_end: int,
        actual_end: int,
    ) -> None:
        self.camera_id = camera_id
        self.track_id = track_id
        self.expected_start = expected_start
        self.actual_start = actual_start
        self.expected_end = expected_end
        self.actual_end = actual_end
        super().__init__(
            f"Join Invariant Failure for {camera_id} track {track_id}: "
            f"Start frame expected={expected_start}, actual_min={actual_start}; "
            f"End frame expected={expected_end}, actual_max={actual_end}"
        )


def index_camera_observations(
    observations_data: Dict[str, Any]
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, str]]:
    """
    Indexes frame-level observations by vehicle track_id in a single pass.

    Returns:
        track_observations: Map of track_id -> list of vehicle observation dicts with frame metadata.
        frame_timestamps: Map of frame_number -> raw timestamp string.
    """
    frames = observations_data.get("frames", [])
    track_observations: Dict[int, List[Dict[str, Any]]] = {}
    frame_timestamps: Dict[int, str] = {}

    for frame in frames:
        f_num = frame["frame_number"]
        ts_str = frame["timestamp"]
        frame_timestamps[f_num] = ts_str

        for vehicle in frame.get("vehicles", []):
            tid = vehicle["track_id"]
            if tid not in track_observations:
                track_observations[tid] = []

            obs_record = dict(vehicle)
            obs_record["frame_number"] = f_num
            obs_record["frame_timestamp"] = ts_str
            obs_record["fps"] = frame.get("fps", 10.0)
            track_observations[tid].append(obs_record)

    return track_observations, frame_timestamps


def validate_and_extract_join(
    trajectory_record: Dict[str, Any],
    vehicle_observations: List[Dict[str, Any]],
    camera_id: str,
    frame_timestamps: Dict[int, str],
) -> Dict[str, Any]:
    """
    Validates boundary invariants and computes aggregated observation metrics.

    Raises:
        JoinInvariantError: If min/max observed frames diverge from trajectory start/end frames.
    """
    tid = trajectory_record["track_id"]
    expected_start = trajectory_record["start_frame"]
    expected_end = trajectory_record["end_frame"]

    if not vehicle_observations:
        raise JoinInvariantError(camera_id, tid, expected_start, -1, expected_end, -1)

    observed_frames = [obs["frame_number"] for obs in vehicle_observations]
    actual_min = min(observed_frames)
    actual_max = max(observed_frames)

    # Strict invariant validation: min frame == start_frame and max frame == end_frame
    if actual_min != expected_start or actual_max != expected_end:
        raise JoinInvariantError(
            camera_id, tid, expected_start, actual_min, expected_end, actual_max
        )

    # Extract start and end raw timestamps
    start_raw_ts = frame_timestamps.get(expected_start)
    end_raw_ts = frame_timestamps.get(expected_end)
    if not start_raw_ts or not end_raw_ts:
        raise ValueError(
            f"Missing frame timestamp for {camera_id} track {tid}: "
            f"start_frame {expected_start} ({start_raw_ts}), end_frame {expected_end} ({end_raw_ts})"
        )

    # Average detection confidence
    confidences = [
        float(obs["confidence"])
        for obs in vehicle_observations
        if obs.get("confidence") is not None
    ]
    avg_det_conf = sum(confidences) / len(confidences) if confidences else None

    # Direction consensus
    directions = [
        obs["direction"].strip().lower()
        for obs in vehicle_observations
        if obs.get("direction") and obs["direction"].strip().lower() != "stationary"
    ]
    if directions:
        dir_counts = Counter(directions)
        consensus_direction = dir_counts.most_common(1)[0][0]
    else:
        # Check trajectory displacement if all are stationary or null
        pts = trajectory_record.get("trajectory", [])
        if len(pts) >= 2:
            dx = pts[-1][0] - pts[0][0]
            dy = pts[-1][1] - pts[0][1]
            if abs(dx) > abs(dy):
                consensus_direction = "eastbound" if dx > 0 else "westbound"
            else:
                consensus_direction = "southbound" if dy > 0 else "northbound"
        else:
            consensus_direction = "stationary"

    # Aggregate ANPR / OCR
    anpr_data = aggregate_tracklet_ocr(vehicle_observations)

    return {
        "start_raw_timestamp": start_raw_ts,
        "end_raw_timestamp": end_raw_ts,
        "average_detector_confidence": round(avg_det_conf, 4) if avg_det_conf is not None else None,
        "direction": consensus_direction,
        "anpr": anpr_data,
        "observation_count": len(vehicle_observations),
    }
