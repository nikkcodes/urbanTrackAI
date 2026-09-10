"""
Temporal feasibility calculation and temporal comparability evaluation
for vehicle observation pairs in UrbanTrack AI.
"""

from datetime import datetime
import math
from typing import Any, Dict, Optional, Tuple, Union

from schemas.observation_schema import Observation
from .similarity import time_difference


def _extract_obs_temporal(obs: Union[Observation, Dict[str, Any]]) -> Tuple[str, float, Optional[int], Optional[str], Optional[str], Optional[float]]:
    """Extract (camera_id, timestamp_seconds, frame_id, semantics, ref_id, clock_offset)."""
    if isinstance(obs, Observation):
        return (
            str(obs.camera_id),
            float(obs.timestamp_seconds),
            obs.frame_id,
            getattr(obs, "timestamp_semantics", None),
            getattr(obs, "time_reference_id", None),
            getattr(obs, "clock_offset_seconds", None),
        )
    if isinstance(obs, dict):
        cam = str(obs.get("camera_id", ""))
        t = obs.get("timestamp_seconds")
        if t is None:
            t = obs.get("timestamp", 0.0)
            if isinstance(t, str):
                try:
                    t = datetime.fromisoformat(t).timestamp()
                except Exception:
                    t = 0.0
        frame = obs.get("frame_id")
        sem = obs.get("timestamp_semantics")
        ref = obs.get("time_reference_id")
        offset = obs.get("clock_offset_seconds")
        return (cam, float(t), frame, sem, ref, float(offset) if offset is not None else None)
    raise TypeError(f"Unsupported observation type: {type(obs).__name__}")


def _get_camera_meta(cam_id: str, camera_metadata: Optional[Any]) -> Optional[Dict[str, Any]]:
    """Lookup camera metadata with network-level fallback."""
    if camera_metadata is None:
        return None
    res = {}
    if isinstance(camera_metadata, dict):
        # Network-level defaults if present in camera_metadata
        for k in ("timestamp_semantics", "time_reference_id", "clock_offset_seconds", "synchronization_status"):
            if k in camera_metadata:
                res[k] = camera_metadata[k]
        val = camera_metadata.get(cam_id)
        if val is not None:
            if hasattr(val, "to_dict"):
                res.update(val.to_dict())
            elif isinstance(val, dict):
                res.update(val)
        return res if res else None
    if hasattr(camera_metadata, "metadata") and isinstance(camera_metadata.metadata, dict):
        return _get_camera_meta(cam_id, camera_metadata.metadata)
    if hasattr(camera_metadata, "get_camera"):
        val = camera_metadata.get_camera(cam_id)
        if val is not None:
            if hasattr(val, "to_dict"):
                return val.to_dict()
            if isinstance(val, dict):
                return val
    return None


def check_temporal_comparability(
    obs_a: Union[Observation, Dict[str, Any]],
    obs_b: Union[Observation, Dict[str, Any]],
    camera_metadata: Optional[Union[Dict[str, Any], Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate whether two vehicle observations have comparable timestamps.

    Rules:
        SAME CAMERA:
            Video-relative timestamps within the same camera timeline are comparable.
            If delta_t = t_b - t_a >= 0: comparable = True, status = "available".
            If delta_t < 0: comparable = False, status = "invalid_negative_time".

        DIFFERENT CAMERAS:
            Timestamps can ONLY be compared if an explicit temporal relationship exists:
            1. Both cameras share the same documented time_reference_id.
            2. Both explicitly declare synchronized timestamps with a common reference.
            3. A known clock offset is explicitly documented relative to a shared reference.
            Otherwise:
                comparable = False, status = "unavailable", delta_seconds = None.
                Reason indicates independent video-relative streams or missing synchronization metadata.

    Returns:
        Dict containing: comparable, status, reason, delta_seconds, timestamp_semantics,
        time_reference_id, used_in_route_scoring.
    """
    cam_a, t_a, frame_a, sem_a, ref_a, offset_a = _extract_obs_temporal(obs_a)
    cam_b, t_b, frame_b, sem_b, ref_b, offset_b = _extract_obs_temporal(obs_b)

    meta_a = _get_camera_meta(cam_a, camera_metadata)
    meta_b = _get_camera_meta(cam_b, camera_metadata)

    if meta_a:
        sem_a = sem_a or meta_a.get("timestamp_semantics")
        ref_a = ref_a or meta_a.get("time_reference_id")
        if offset_a is None and "clock_offset_seconds" in meta_a:
            offset_a = float(meta_a["clock_offset_seconds"]) if meta_a["clock_offset_seconds"] is not None else None
    if meta_b:
        sem_b = sem_b or meta_b.get("timestamp_semantics")
        ref_b = ref_b or meta_b.get("time_reference_id")
        if offset_b is None and "clock_offset_seconds" in meta_b:
            offset_b = float(meta_b["clock_offset_seconds"]) if meta_b["clock_offset_seconds"] is not None else None

    # SAME CAMERA
    if cam_a == cam_b:
        delta_t = t_b - t_a
        if delta_t < 0:
            return {
                "comparable": False,
                "status": "invalid_negative_time",
                "reason": "negative_elapsed_time_same_camera",
                "delta_seconds": None,
                "timestamp_semantics": sem_a or "video_relative",
                "time_reference_id": ref_a or f"camera_{cam_a}_local",
                "used_in_route_scoring": False,
            }
        return {
            "comparable": True,
            "status": "available",
            "reason": "same_camera_video_relative_valid",
            "delta_seconds": delta_t,
            "timestamp_semantics": sem_a or "video_relative",
            "time_reference_id": ref_a or f"camera_{cam_a}_local",
            "used_in_route_scoring": True,
        }

    # DIFFERENT CAMERAS
    # Check for explicit shared time reference or documented synchronization
    sync_status_a = meta_a.get("synchronization_status") if meta_a else None
    sync_status_b = meta_b.get("synchronization_status") if meta_b else None

    # Case 1: Independent video-relative streams without shared temporal reference
    if (sem_a == "video_relative" or sem_b == "video_relative") and (ref_a is None or ref_b is None or ref_a != ref_b):
        return {
            "comparable": False,
            "status": "unavailable",
            "reason": "independent_camera_video_relative_timestamps_without_shared_time_reference",
            "delta_seconds": None,
            "timestamp_semantics": "video_relative",
            "time_reference_id": None,
            "used_in_route_scoring": False,
        }

    # Case 2: Absolute timestamps without shared temporal reference
    if sem_a == "absolute" and sem_b == "absolute":
        if ref_a is None or ref_b is None or ref_a != ref_b:
            return {
                "comparable": False,
                "status": "unavailable",
                "reason": "absolute_timestamps_without_shared_temporal_reference",
                "delta_seconds": None,
                "timestamp_semantics": "absolute",
                "time_reference_id": None,
                "used_in_route_scoring": False,
            }

    # Case 3: Clock offsets exist but without a meaningful shared reference
    has_offsets = (offset_a is not None or offset_b is not None)
    if has_offsets and (ref_a is None or ref_b is None or ref_a != ref_b):
        return {
            "comparable": False,
            "status": "unavailable",
            "reason": "clock_offset_without_meaningful_shared_reference",
            "delta_seconds": None,
            "timestamp_semantics": sem_a or "unknown",
            "time_reference_id": None,
            "used_in_route_scoring": False,
        }

    # Case 4: Explicit synchronization or shared reference
    is_sync = (
        (sem_a == "synchronized" and sem_b == "synchronized")
        or (sync_status_a == "synchronized" and sync_status_b == "synchronized")
        or (ref_a is not None and ref_b is not None and ref_a == ref_b)
    )

    if not is_sync:
        return {
            "comparable": False,
            "status": "unavailable",
            "reason": "missing_synchronization_metadata",
            "delta_seconds": None,
            "timestamp_semantics": sem_a or "unknown",
            "time_reference_id": None,
            "used_in_route_scoring": False,
        }

    # Verify if different references are declared (or one declared while the other omitted)
    if (ref_a is not None or ref_b is not None) and ref_a != ref_b:
        return {
            "comparable": False,
            "status": "unavailable",
            "reason": "mismatched_time_reference_ids",
            "delta_seconds": None,
            "timestamp_semantics": sem_a,
            "time_reference_id": f"{ref_a}_vs_{ref_b}",
            "used_in_route_scoring": False,
        }

    # Compute synchronized delta_t with clock offset adjustment if documented
    adj_a = offset_a if offset_a is not None else 0.0
    adj_b = offset_b if offset_b is not None else 0.0
    delta_t = (t_b + adj_b) - (t_a + adj_a)

    if delta_t < 0:
        return {
            "comparable": False,
            "status": "invalid_negative_time",
            "reason": "negative_elapsed_time_cross_camera",
            "delta_seconds": None,
            "timestamp_semantics": sem_a or "synchronized",
            "time_reference_id": ref_a or "shared_sync",
            "used_in_route_scoring": False,
        }

    return {
        "comparable": True,
        "status": "available",
        "reason": f"cross_camera_synchronized_{ref_a or 'network'}",
        "delta_seconds": delta_t,
        "timestamp_semantics": sem_a or "synchronized",
        "time_reference_id": ref_a or "shared_sync",
        "used_in_route_scoring": True,
    }


def temporal_feasibility(
    obs_a: Union[Observation, Dict[str, Any]],
    obs_b: Union[Observation, Dict[str, Any]],
    max_reasonable_gap_seconds: float = 3600.0,
    camera_metadata: Optional[Union[Dict[str, Any], Any]] = None,
) -> Dict[str, Any]:
    """
    Evaluate chronological temporal feasibility between two vehicle observations.

    Rules:
        - Evaluates temporal comparability via check_temporal_comparability.
        - When temporal evidence is unavailable:
            feasibility_score = None, delta_t_seconds = None, status = "unavailable".
        - When temporal evidence is available:
            evaluates physical feasibility, simultaneous detections, and exponential decay.

    Args:
        obs_a: First observation (chronological origin).
        obs_b: Second observation (chronological destination).
        max_reasonable_gap_seconds: Maximum plausible temporal gap (default 3600s).
        camera_metadata: Optional camera metadata dict or RoadGraph.

    Returns:
        Dict containing: feasibility_score, delta_t_seconds, status, explanation,
        temporal_evidence, timestamp_semantics, time_reference_id, reason, used_in_route_scoring.
    """
    comp = check_temporal_comparability(obs_a, obs_b, camera_metadata=camera_metadata)

    if not comp["comparable"]:
        if comp["status"] == "invalid_negative_time":
            return {
                "feasibility_score": 0.0,
                "delta_t_seconds": None,
                "status": "impossible_negative_time",
                "explanation": f"Chronologically inverted: {comp['reason']}.",
                "temporal_evidence": comp,
                "timestamp_semantics": comp.get("timestamp_semantics"),
                "time_reference_id": comp.get("time_reference_id"),
                "reason": comp.get("reason"),
                "delta_seconds": None,
                "used_in_route_scoring": False,
            }
        # Temporal evidence unavailable: DO NOT return 0.5 or fake probability
        return {
            "feasibility_score": None,
            "delta_t_seconds": None,
            "status": "unavailable",
            "explanation": f"Temporal feasibility unavailable: {comp['reason']}.",
            "temporal_evidence": comp,
            "timestamp_semantics": comp.get("timestamp_semantics", "video_relative"),
            "time_reference_id": comp.get("time_reference_id"),
            "reason": comp.get("reason"),
            "delta_seconds": None,
            "used_in_route_scoring": False,
        }

    delta_t = comp["delta_seconds"]
    cam_a, _, frame_a, _, _, _ = _extract_obs_temporal(obs_a)
    cam_b, _, frame_b, _, _, _ = _extract_obs_temporal(obs_b)

    id_a = getattr(obs_a, "observation_id", obs_a.get("observation_id") if isinstance(obs_a, dict) else None)
    id_b = getattr(obs_b, "observation_id", obs_b.get("observation_id") if isinstance(obs_b, dict) else None)

    if delta_t == 0.0:
        if cam_a != cam_b:
            return {
                "feasibility_score": 0.0,
                "delta_t_seconds": 0.0,
                "status": "impossible_simultaneous_different_cameras",
                "explanation": f"Simultaneous detection at different cameras ({cam_a} vs {cam_b}).",
                "temporal_evidence": comp,
                "timestamp_semantics": comp.get("timestamp_semantics"),
                "time_reference_id": comp.get("time_reference_id"),
                "reason": comp.get("reason"),
                "delta_seconds": 0.0,
                "used_in_route_scoring": True,
            }
        else:
            trk_a = getattr(obs_a, "track_id", obs_a.get("track_id") if isinstance(obs_a, dict) else None)
            trk_b = getattr(obs_b, "track_id", obs_b.get("track_id") if isinstance(obs_b, dict) else None)
            if (
                (id_a is not None and id_b is not None and id_a != id_b and frame_a is not None and frame_a == frame_b)
                or (trk_a is not None and trk_b is not None and trk_a != trk_b)
            ):
                return {
                    "feasibility_score": 0.0,
                    "delta_t_seconds": 0.0,
                    "status": "impossible_simultaneous_same_camera_distinct_bbox",
                    "explanation": f"Simultaneous distinct detections/tracks ({trk_a or id_a} vs {trk_b or id_b}) at camera {cam_a}.",
                    "temporal_evidence": comp,
                    "timestamp_semantics": comp.get("timestamp_semantics"),
                    "time_reference_id": comp.get("time_reference_id"),
                    "reason": comp.get("reason"),
                    "delta_seconds": 0.0,
                    "used_in_route_scoring": True,
                }
            return {
                "feasibility_score": 1.0,
                "delta_t_seconds": 0.0,
                "status": "same_camera_same_instant",
                "explanation": "Simultaneous detection at the same camera.",
                "temporal_evidence": comp,
                "timestamp_semantics": comp.get("timestamp_semantics"),
                "time_reference_id": comp.get("time_reference_id"),
                "reason": comp.get("reason"),
                "delta_seconds": 0.0,
                "used_in_route_scoring": True,
            }

    # Positive time gap delta_t > 0
    if delta_t > max_reasonable_gap_seconds:
        decay = math_decay(delta_t, max_reasonable_gap_seconds)
        return {
            "feasibility_score": decay,
            "delta_t_seconds": delta_t,
            "status": "large_time_gap",
            "explanation": f"Time gap of {delta_t:.1f}s exceeds typical window ({max_reasonable_gap_seconds:.0f}s).",
            "temporal_evidence": comp,
            "timestamp_semantics": comp.get("timestamp_semantics"),
            "time_reference_id": comp.get("time_reference_id"),
            "reason": comp.get("reason"),
            "delta_seconds": delta_t,
            "used_in_route_scoring": True,
        }

    return {
        "feasibility_score": 1.0,
        "delta_t_seconds": delta_t,
        "status": "plausible_time_gap",
        "explanation": f"Plausible chronological time gap of {delta_t:.1f}s.",
        "temporal_evidence": comp,
        "timestamp_semantics": comp.get("timestamp_semantics"),
        "time_reference_id": comp.get("time_reference_id"),
        "reason": comp.get("reason"),
        "delta_seconds": delta_t,
        "used_in_route_scoring": True,
    }


def math_decay(val: float, max_val: float) -> float:
    """Exponential decay calculation for values exceeding threshold."""
    excess = val - max_val
    return max(0.0, math.exp(-excess / max_val))

