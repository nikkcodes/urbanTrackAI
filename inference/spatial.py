"""
Spatial feasibility and required travel speed calculation for vehicle observation pairs.
"""

import math
from typing import Any, Dict, Optional, Union

from schemas.observation_schema import Observation
from .similarity import geographic_distance
from .temporal import check_temporal_comparability


def spatial_feasibility(
    obs_a: Union[Observation, Dict[str, Any]],
    obs_b: Union[Observation, Dict[str, Any]],
    max_plausible_speed_kmh: float = 120.0,
    min_plausible_speed_kmh: float = 0.0,
    camera_metadata: Optional[Union[Dict[str, Any], Any]] = None,
) -> Dict[str, Any]:
    """
    Calculate geographic distance and required travel speed feasibility between two observations.

    required_speed_kmh = (distance_meters / time_difference_seconds) * 3.6

    Rules:
        - If coordinates are missing on either observation: returns neutral score (0.5, "coordinates_unavailable").
        - If obs_a and obs_b are at the same camera: distance = 0.0m, required_speed = 0.0 km/h, feasibility = 1.0.
        - Checks temporal comparability via check_temporal_comparability.
        - If temporal evidence is unavailable:
            distance_meters = valid distance
            required_speed_kmh = None
            status = "temporal_evidence_unavailable"
            (Does NOT divide by fake delta_t or mark impossible_speed).
        - If delta_t <= 0 and distance > 0: required speed is infinite/impossible -> feasibility = 0.0 ("impossible_speed").
        - If required_speed_kmh > max_plausible_speed_kmh: feasibility = 0.0 ("physically_impossible_speed").
        - Otherwise returns a normalized feasibility score in [0.0, 1.0].

    Args:
        obs_a: First observation.
        obs_b: Second observation.
        max_plausible_speed_kmh: Maximum plausible speed limit in km/h (default 120.0 km/h).
        min_plausible_speed_kmh: Minimum plausible speed limit in km/h (default 0.0 km/h).
        camera_metadata: Optional camera metadata dict or RoadGraph.

    Returns:
        Dict containing 'feasibility_score', 'distance_meters', 'required_speed_kmh', and 'status'.
    """
    lat_a = getattr(obs_a, "latitude", obs_a.get("latitude") if isinstance(obs_a, dict) else None)
    lon_a = getattr(obs_a, "longitude", obs_a.get("longitude") if isinstance(obs_a, dict) else None)
    lat_b = getattr(obs_b, "latitude", obs_b.get("latitude") if isinstance(obs_b, dict) else None)
    lon_b = getattr(obs_b, "longitude", obs_b.get("longitude") if isinstance(obs_b, dict) else None)
    cam_a = str(getattr(obs_a, "camera_id", obs_a.get("camera_id") if isinstance(obs_a, dict) else ""))
    cam_b = str(getattr(obs_b, "camera_id", obs_b.get("camera_id") if isinstance(obs_b, dict) else ""))

    world_a = getattr(obs_a, "world_position", obs_a.get("world_position") if isinstance(obs_a, dict) else None)
    world_b = getattr(obs_b, "world_position", obs_b.get("world_position") if isinstance(obs_b, dict) else None)
    world_sys_a = getattr(obs_a, "world_coordinate_system", obs_a.get("world_coordinate_system") if isinstance(obs_a, dict) else None)
    world_sys_b = getattr(obs_b, "world_coordinate_system", obs_b.get("world_coordinate_system") if isinstance(obs_b, dict) else None)
    has_native_world = (
        isinstance(world_a, (list, tuple)) and len(world_a) == 2
        and isinstance(world_b, (list, tuple)) and len(world_b) == 2
        and world_sys_a is not None and world_sys_a == world_sys_b
    )

    if has_native_world:
        distance_fn = lambda: math.hypot(float(world_a[0]) - float(world_b[0]), float(world_a[1]) - float(world_b[1]))
        distance_source = f"native_world_coordinates:{world_sys_a}"
    else:
        distance_fn = lambda: geographic_distance(float(lat_a), float(lon_a), float(lat_b), float(lon_b))
        distance_source = "geographic_coordinates"

    # Check if geographic coordinates exist
    if not has_native_world and (lat_a is None or lon_a is None or lat_b is None or lon_b is None):
        return {
            "feasibility_score": 0.5,
            "distance_meters": 0.0,
            "required_speed_kmh": None,
            "status": "coordinates_unavailable",
            "explanation": "No common native world coordinates or geographic coordinates are available for spatial evaluation.",
        }

    # Same camera shortcut
    if cam_a and cam_b and cam_a == cam_b:
        dist = distance_fn()
        return {
            "feasibility_score": 1.0,
            "distance_meters": dist,
            "required_speed_kmh": 0.0,
            "status": "same_camera",
            "coordinate_source": distance_source,
            "explanation": f"Observations at same camera ({cam_a}). Distance: {dist:.1f}m from {distance_source}.",
        }

    dist_meters = distance_fn()

    # Check temporal comparability
    comp = check_temporal_comparability(obs_a, obs_b, camera_metadata=camera_metadata)

    if not comp["comparable"]:
        if comp["status"] == "invalid_negative_time":
            return {
                "feasibility_score": 0.0,
                "distance_meters": dist_meters,
                "required_speed_kmh": float("inf"),
            "status": "impossible_speed",
            "explanation": f"Distance of {dist_meters:.1f}m cannot be traversed in negative time ({comp['reason']}).",
            "coordinate_source": distance_source,
                "temporal_evidence": comp,
            }
        # Temporal evidence unavailable: DO NOT divide by fake delta_t or mark impossible_speed
        return {
            "feasibility_score": 0.5,
            "distance_meters": dist_meters,
            "required_speed_kmh": None,
            "status": "temporal_evidence_unavailable",
            "explanation": f"Distance of {dist_meters:.1f}m measured, but cross-camera temporal evidence is unavailable ({comp['reason']}).",
            "coordinate_source": distance_source,
            "temporal_evidence": comp,
        }

    delta_t = comp["delta_seconds"]

    if delta_t == 0.0:
        if dist_meters > 0.0:
            return {
                "feasibility_score": 0.0,
                "distance_meters": dist_meters,
                "required_speed_kmh": float("inf"),
                "status": "impossible_speed",
                "explanation": f"Distance of {dist_meters:.1f}m cannot be traversed in 0.0s (infinite speed required).",
                "coordinate_source": distance_source,
                "temporal_evidence": comp,
            }
        return {
            "feasibility_score": 1.0,
            "distance_meters": 0.0,
            "required_speed_kmh": 0.0,
            "status": "same_location_same_instant",
            "explanation": "Simultaneous observation at identical coordinates.",
            "coordinate_source": distance_source,
            "temporal_evidence": comp,
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
            "coordinate_source": distance_source,
            "temporal_evidence": comp,
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
        "coordinate_source": distance_source,
        "temporal_evidence": comp,
    }
