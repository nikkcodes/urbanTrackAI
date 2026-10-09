"""
UrbanTrack AI — Layer 2 Candidate Generation: Temporal Gate & Chronology.

Enforces direction-independent chronological ordering and temporal gating
strictly using synchronized timestamps, preventing reverse-direction bugs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from layer2.ingestion.canonical_models import CanonicalTracklet


@dataclass
class ChronologyEvaluation:
    """Chronological relationship and temporal feasibility between origin and destination."""
    origin_start_sync: float
    origin_end_sync: float
    destination_start_sync: float
    destination_end_sync: float
    delta_t_seconds: float
    direction: str
    is_temporally_feasible: bool
    is_overlapping: bool = False
    overlap_duration_seconds: float = 0.0
    rejection_reason_code: Optional[str] = None
    rejection_reason_detail: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "origin_start_sync": round(self.origin_start_sync, 4),
            "origin_end_sync": round(self.origin_end_sync, 4),
            "destination_start_sync": round(self.destination_start_sync, 4),
            "destination_end_sync": round(self.destination_end_sync, 4),
            "delta_t_seconds": round(self.delta_t_seconds, 4),
            "direction": self.direction,
        }
        if self.is_overlapping:
            d["is_overlapping"] = True
            d["overlap_duration_seconds"] = self.overlap_duration_seconds
        return d


class TemporalGate:
    """
    Evaluates temporal feasibility between pairs of canonical tracklets.

    Key Invariants:
    1. Chronological order is determined strictly by synchronized timestamps:
       origin = tracklet with earlier start_sync_seconds.
       destination = tracklet with later start_sync_seconds.
       Never alphabetical by camera ID.
    2. delta_t = destination.start_sync_seconds - origin.end_sync_seconds.
    3. Overlap-aware gating (delta_t <= 0):
       Admitted as concurrent observation if cameras are within max_overlap_distance_m
       or have directed topology edges; rejected if separated by distant corridors.
    4. Sequential gating (delta_t > 0):
       Feasible for further spatial and speed evaluation up to max_time_seconds.
    """

    def __init__(
        self,
        max_time_seconds: Optional[float] = 120.0,
        max_overlap_distance_m: float = 150.0,
    ) -> None:
        self.max_time_seconds = float(max_time_seconds) if max_time_seconds is not None else None
        self.max_overlap_distance_m = float(max_overlap_distance_m)

    @staticmethod
    def determine_origin_destination(
        tracklet_a: CanonicalTracklet,
        tracklet_b: CanonicalTracklet,
    ) -> Tuple[CanonicalTracklet, CanonicalTracklet, str]:
        """
        Determines origin and destination strictly based on synchronized start timestamps.
        Returns: (origin, destination, direction) where direction is 'A_TO_B' or 'B_TO_A'.
        """
        t_a_start = tracklet_a.temporal.start_sync_seconds
        t_b_start = tracklet_b.temporal.start_sync_seconds

        if t_a_start < t_b_start:
            return tracklet_a, tracklet_b, "A_TO_B"
        elif t_b_start < t_a_start:
            return tracklet_b, tracklet_a, "B_TO_A"
        else:
            # Deterministic tie-breaking on end_sync then canonical_id
            t_a_end = tracklet_a.temporal.end_sync_seconds
            t_b_end = tracklet_b.temporal.end_sync_seconds
            if t_a_end < t_b_end:
                return tracklet_a, tracklet_b, "A_TO_B"
            elif t_b_end < t_a_end:
                return tracklet_b, tracklet_a, "B_TO_A"
            else:
                if tracklet_a.canonical_id <= tracklet_b.canonical_id:
                    return tracklet_a, tracklet_b, "A_TO_B"
                else:
                    return tracklet_b, tracklet_a, "B_TO_A"

    def evaluate_temporal_feasibility(
        self,
        origin: CanonicalTracklet,
        destination: CanonicalTracklet,
        direction: str,
        camera_distance_m: Optional[float] = None,
        has_topology_edge: bool = False,
    ) -> ChronologyEvaluation:
        """
        Evaluates temporal transit feasibility between origin and destination tracklets.
        """
        t_orig_start = origin.temporal.start_sync_seconds
        t_orig_end = origin.temporal.end_sync_seconds
        t_dest_start = destination.temporal.start_sync_seconds
        t_dest_end = destination.temporal.end_sync_seconds

        delta_t = t_dest_start - t_orig_end

        # Regime A: Overlapping / Concurrent visibility (delta_t <= 0)
        if delta_t <= 0.0:
            overlap_dur = min(t_orig_end, t_dest_end) - t_dest_start
            can_overlap = (
                (camera_distance_m is not None and camera_distance_m <= self.max_overlap_distance_m)
                or has_topology_edge
            )
            if can_overlap:
                return ChronologyEvaluation(
                    origin_start_sync=t_orig_start,
                    origin_end_sync=t_orig_end,
                    destination_start_sync=t_dest_start,
                    destination_end_sync=t_dest_end,
                    delta_t_seconds=delta_t,
                    direction=direction,
                    is_temporally_feasible=True,
                    is_overlapping=True,
                    overlap_duration_seconds=round(max(0.0, overlap_dur), 4),
                    rejection_reason_code=None,
                    rejection_reason_detail=None,
                )
            else:
                dist_str = f"{camera_distance_m:.1f}m" if camera_distance_m is not None else "unknown distance"
                return ChronologyEvaluation(
                    origin_start_sync=t_orig_start,
                    origin_end_sync=t_orig_end,
                    destination_start_sync=t_dest_start,
                    destination_end_sync=t_dest_end,
                    delta_t_seconds=delta_t,
                    direction=direction,
                    is_temporally_feasible=False,
                    is_overlapping=False,
                    overlap_duration_seconds=0.0,
                    rejection_reason_code="REJECT_NEGATIVE_TIME",
                    rejection_reason_detail=(
                        f"Simultaneous observation physically impossible across separated cameras "
                        f"(distance = {dist_str} > {self.max_overlap_distance_m:.1f}m, delta_t = {delta_t:.4f}s)"
                    ),
                )

        # Regime B: Sequential transit (delta_t > 0)
        if self.max_time_seconds is not None and delta_t > self.max_time_seconds:
            return ChronologyEvaluation(
                origin_start_sync=t_orig_start,
                origin_end_sync=t_orig_end,
                destination_start_sync=t_dest_start,
                destination_end_sync=t_dest_end,
                delta_t_seconds=delta_t,
                direction=direction,
                is_temporally_feasible=False,
                is_overlapping=False,
                overlap_duration_seconds=0.0,
                rejection_reason_code="REJECT_EXCEED_MAX_TIME",
                rejection_reason_detail=(
                    f"Transit time delta_t = {delta_t:.2f}s exceeds maximum temporal limit {self.max_time_seconds:.2f}s"
                ),
            )

        return ChronologyEvaluation(
            origin_start_sync=t_orig_start,
            origin_end_sync=t_orig_end,
            destination_start_sync=t_dest_start,
            destination_end_sync=t_dest_end,
            delta_t_seconds=delta_t,
            direction=direction,
            is_temporally_feasible=True,
            is_overlapping=False,
            overlap_duration_seconds=0.0,
            rejection_reason_code=None,
            rejection_reason_detail=None,
        )
