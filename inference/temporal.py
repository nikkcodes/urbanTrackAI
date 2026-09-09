"""
Temporal feasibility calculation for vehicle observation pairs.
"""

from typing import Dict, Tuple, Union

from schemas.observation_schema import Observation
from .similarity import time_difference


def temporal_feasibility(
    obs_a: Observation,
    obs_b: Observation,
    max_reasonable_gap_seconds: float = 3600.0,
) -> Dict[str, Union[float, str]]:
    """
    Evaluate the chronological temporal feasibility between two vehicle observations.

    Rules:
        - delta_t = timestamp_b - timestamp_a
        - If delta_t < 0: Chronologically inverted (B before A). Feasibility = 0.0 ("impossible_negative_time").
        - If delta_t == 0 and camera_id_a != camera_id_b: Impossible simultaneous detection at 2 different cameras. Feasibility = 0.0 ("impossible_simultaneous_different_cameras").
        - Otherwise returns a feasibility score in [0.0, 1.0].

    Args:
        obs_a: First observation (chronological origin).
        obs_b: Second observation (chronological destination).
        max_reasonable_gap_seconds: Maximum plausible temporal gap before feasibility decays (default 3600s = 1 hr).

    Returns:
        Dict containing 'feasibility_score', 'delta_t_seconds', and 'status'.
    """
    delta_t = time_difference(obs_a, obs_b)

    if delta_t < 0:
        return {
            "feasibility_score": 0.0,
            "delta_t_seconds": delta_t,
            "status": "impossible_negative_time",
            "explanation": f"Observation B occurred {abs(delta_t):.1f}s before Observation A."
        }

    if delta_t == 0:
        if obs_a.camera_id != obs_b.camera_id:
            return {
                "feasibility_score": 0.0,
                "delta_t_seconds": 0.0,
                "status": "impossible_simultaneous_different_cameras",
                "explanation": f"Simultaneous detection at different cameras ({obs_a.camera_id} vs {obs_b.camera_id})."
            }
        else:
            if obs_a.observation_id != obs_b.observation_id and obs_a.frame_id == obs_b.frame_id and obs_a.frame_id is not None:
                return {
                    "feasibility_score": 0.0,
                    "delta_t_seconds": 0.0,
                    "status": "impossible_simultaneous_same_camera_distinct_bbox",
                    "explanation": f"Simultaneous distinct detections in frame {obs_a.frame_id} at camera {obs_a.camera_id}."
                }
            return {
                "feasibility_score": 1.0,
                "delta_t_seconds": 0.0,
                "status": "same_camera_same_instant",
                "explanation": "Simultaneous detection at the same camera."
            }

    # For positive time gaps delta_t > 0
    if delta_t > max_reasonable_gap_seconds:
        # Smooth exponential decay for large gaps
        decay = math_decay(delta_t, max_reasonable_gap_seconds)
        return {
            "feasibility_score": decay,
            "delta_t_seconds": delta_t,
            "status": "large_time_gap",
            "explanation": f"Time gap of {delta_t:.1f}s exceeds typical window ({max_reasonable_gap_seconds:.0f}s)."
        }

    return {
        "feasibility_score": 1.0,
        "delta_t_seconds": delta_t,
        "status": "plausible_time_gap",
        "explanation": f"Plausible chronological time gap of {delta_t:.1f}s."
    }


def math_decay(val: float, max_val: float) -> float:
    """Exponential decay calculation for values exceeding threshold."""
    import math
    excess = val - max_val
    return max(0.0, math.exp(-excess / max_val))
