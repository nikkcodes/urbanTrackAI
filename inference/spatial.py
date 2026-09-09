"""
Spatial feasibility and required travel speed calculation for vehicle observation pairs.
"""

from typing import Dict, Optional, Union

from schemas.observation_schema import Observation
from .similarity import geographic_distance, time_difference


def spatial_feasibility(
    obs_a: Observation,
    obs_b: Observation,
    max_plausible_speed_kmh: float = 120.0,
    min_plausible_speed_kmh: float = 0.0,
) -> Dict[str, Union[float, str, bool]]:
    """
    Calculate geographic distance and required travel speed feasibility between two observations.

    required_speed_kmh = (distance_meters / time_difference_seconds) * 3.6

    Rules:
        - If coordinates are missing on either observation: returns neutral score (0.5, "coordinates_unavailable").
        - If obs_a and obs_b are at the same camera: distance = 0.0m, required_speed = 0.0 km/h, feasibility = 1.0.
        - If delta_t <= 0 and distance > 0: required speed is infinite/impossible -> feasibility = 0.0 ("impossible_speed").
        - If required_speed_kmh > max_plausible_speed_kmh: feasibility = 0.0 ("physically_impossible_speed").
        - Otherwise returns a normalized feasibility score in [0.0, 1.0].

    Args:
        obs_a: First observation.
        obs_b: Second observation.
        max_plausible_speed_kmh: Maximum plausible speed limit in km/h (default 120.0 km/h).
        min_plausible_speed_kmh: Minimum plausible speed limit in km/h (default 0.0 km/h).

    Returns:
        Dict containing 'feasibility_score', 'distance_meters', 'required_speed_kmh', and 'status'.
    """
    # Check if geographic coordinates exist
    if (
        obs_a.latitude is None
        or obs_a.longitude is None
        or obs_b.latitude is None
        or obs_b.longitude is None
    ):
        return {
            "feasibility_score": 0.5,
            "distance_meters": 0.0,
            "required_speed_kmh": 0.0,
            "status": "coordinates_unavailable",
            "explanation": "Geographic coordinates unavailable for spatial evaluation.",
        }

    # Same camera shortcut
    if obs_a.camera_id == obs_b.camera_id:
        dist = geographic_distance(obs_a, obs_b)
        return {
            "feasibility_score": 1.0,
            "distance_meters": dist,
            "required_speed_kmh": 0.0,
            "status": "same_camera",
            "explanation": f"Observations at same camera ({obs_a.camera_id}). Distance: {dist:.1f}m.",
        }

    dist_meters = geographic_distance(obs_a, obs_b)
    delta_t = time_difference(obs_a, obs_b)

    if delta_t <= 0:
        return {
            "feasibility_score": 0.0,
            "distance_meters": dist_meters,
            "required_speed_kmh": float("inf"),
            "status": "impossible_speed",
            "explanation": f"Distance of {dist_meters:.1f}m cannot be traversed in {delta_t:.1f}s (infinite speed required).",
        }

    # Speed calculation in m/s then km/h
    speed_ms = dist_meters / delta_t
    speed_kmh = speed_ms * 3.6

    if speed_kmh > max_plausible_speed_kmh:
        return {
            "feasibility_score": 0.0,
            "distance_meters": dist_meters,
            "required_speed_kmh": speed_kmh,
            "status": "physically_impossible_speed",
            "explanation": (
                f"Required travel speed of {speed_kmh:.1f} km/h exceeds maximum plausible limit "
                f"({max_plausible_speed_kmh:.1f} km/h)."
            ),
        }

    # High feasibility for realistic urban speeds
    # Linear scale down if close to max speed
    if speed_kmh <= max_plausible_speed_kmh * 0.75:
        score = 1.0
    else:
        # Smooth scaling between 75% max speed and max speed
        excess_ratio = (speed_kmh - max_plausible_speed_kmh * 0.75) / (max_plausible_speed_kmh * 0.25)
        score = max(0.1, 1.0 - 0.9 * excess_ratio)

    return {
        "feasibility_score": score,
        "distance_meters": dist_meters,
        "required_speed_kmh": speed_kmh,
        "status": "plausible_speed",
        "explanation": f"Plausible required speed of {speed_kmh:.1f} km/h over {dist_meters:.1f}m in {delta_t:.1f}s.",
    }
