"""
Identity Fusion Engine for UrbanTrack AI.
Calculates pairwise estimated match probabilities and explainable evidence
between vehicle observations across cameras.
"""

from typing import Any, Dict, Optional

from schemas.observation_schema import Observation
from .similarity import appearance_similarity, plate_similarity, vehicle_type_compatibility
from .spatial import spatial_feasibility
from .temporal import temporal_feasibility


def match_observations(
    obs_a: Observation,
    obs_b: Observation,
    camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute pairwise identity fusion score and explainable evidence between two observations.

    Answers: "Given two vehicle observations from different cameras, how likely
    is it that they represent the same physical vehicle?"

    Args:
        obs_a: First vehicle observation.
        obs_b: Second vehicle observation.
        camera_metadata: Optional camera metadata dict with geographic coordinates.
        config: Optional configuration dictionary (e.g. max_plausible_speed_kmh, weights).

    Returns:
        Dict[str, Any]: Structured output containing observation IDs, detailed evidence,
                       estimated match probability, and human-readable explanation.
    """
    config = config or {}
    max_speed_kmh = float(config.get("max_plausible_speed_kmh", 120.0))

    # Populate coordinates from camera metadata if missing on observations
    if camera_metadata:
        if obs_a.latitude is None and obs_a.camera_id in camera_metadata:
            obs_a.latitude = camera_metadata[obs_a.camera_id].get("latitude")
            obs_a.longitude = camera_metadata[obs_a.camera_id].get("longitude")
        if obs_b.latitude is None and obs_b.camera_id in camera_metadata:
            obs_b.latitude = camera_metadata[obs_b.camera_id].get("latitude")
            obs_b.longitude = camera_metadata[obs_b.camera_id].get("longitude")

    # 1. Vehicle Type Compatibility Evidence
    type_score, type_status = vehicle_type_compatibility(obs_a.vehicle_type, obs_b.vehicle_type)
    type_match = type_status == "compatible"

    # 2. Temporal Feasibility Evidence
    t_res = temporal_feasibility(obs_a, obs_b)
    t_score = float(t_res["feasibility_score"])
    t_status = str(t_res["status"])
    t_delta_sec = float(t_res["delta_t_seconds"])

    # 3. Spatial Feasibility Evidence
    s_res = spatial_feasibility(obs_a, obs_b, max_plausible_speed_kmh=max_speed_kmh)
    s_score = float(s_res["feasibility_score"])
    s_status = str(s_res["status"])
    s_dist_m = float(s_res["distance_meters"])
    s_speed_kmh = float(s_res["required_speed_kmh"])

    # 4. Appearance Similarity Evidence
    app_score = None
    app_status = "unavailable"
    if obs_a.appearance_embedding is not None and obs_b.appearance_embedding is not None:
        if (
            isinstance(obs_a.appearance_embedding, (list, tuple))
            and isinstance(obs_b.appearance_embedding, (list, tuple))
            and len(obs_a.appearance_embedding) > 0
            and len(obs_a.appearance_embedding) == len(obs_b.appearance_embedding)
        ):
            app_score = appearance_similarity(obs_a.appearance_embedding, obs_b.appearance_embedding)
            app_status = "available" if app_score is not None else "invalid"
        else:
            app_score = None
            app_status = "invalid_mismatched"
    else:
        app_score = None
        app_status = "missing"

    # 5. License Plate Evidence (if available)
    plate_score = None
    if obs_a.plate is not None and obs_b.plate is not None:
        plate_score = plate_similarity(obs_a.plate, obs_b.plate)

    # --- HARD REJECTION / PHYSICAL IMPOSSIBILITY RULES ---
    is_rejected = False
    rejection_reason = ""

    if type_status == "incompatible":
        is_rejected = True
        rejection_reason = f"Incompatible vehicle types ({obs_a.vehicle_type} vs {obs_b.vehicle_type})."
    elif t_status in ("impossible_negative_time", "impossible_simultaneous_different_cameras", "impossible_simultaneous_same_camera_distinct_bbox"):
        is_rejected = True
        rejection_reason = f"Physically impossible temporal alignment ({t_res['explanation']})."
    elif s_status in ("impossible_speed", "physically_impossible_speed"):
        is_rejected = True
        rejection_reason = f"Physically impossible travel speed ({s_res['explanation']})."

    has_identity_evidence = (app_score is not None) or (plate_score is not None)

    # --- ESTIMATED MATCH PROBABILITY CALCULATION ---
    if is_rejected:
        estimated_prob = 0.0
        explanation = f"Match rejected (0.0): {rejection_reason}"
    else:
        # Baseline spatio-temporal and vehicle type feasibility (gating factor)
        st_composite = (t_score + s_score) / 2.0
        feasibility_score = 0.70 * st_composite + 0.30 * type_score

        if has_identity_evidence:
            # Combine available identity features (appearance and/or plate)
            if app_score is not None and plate_score is not None:
                id_score = 0.55 * app_score + 0.45 * plate_score
            elif app_score is not None:
                id_score = app_score
            else:
                id_score = plate_score

            # Match probability: spatio-temporal feasibility gating * identity similarity score
            estimated_prob = feasibility_score * id_score
        else:
            # Identity evidence unavailable (missing/invalid appearance and no plate):
            # Spatio-temporal feasibility alone gives unconfirmed candidate score (max 0.50)
            estimated_prob = feasibility_score * 0.50

        estimated_prob = round(max(0.0, min(1.0, estimated_prob)), 4)

        # Generate human-readable explanation
        reasons = []
        if app_status == "available" and app_score is not None:
            qual = "high" if app_score >= 0.8 else ("moderate" if app_score >= 0.5 else "low")
            reasons.append(f"Appearance similarity is {app_score:.2f} ({qual})")
        elif app_status == "invalid_mismatched":
            reasons.append("Appearance evidence invalid (mismatched vector dimensions)")
        else:
            reasons.append("Appearance evidence unavailable")

        if type_status != "unknown":
            reasons.append(f"vehicle types are {type_status} ({obs_a.vehicle_type} vs {obs_b.vehicle_type})")

        if s_dist_m > 0 and t_delta_sec > 0:
            reasons.append(f"required travel speed is {s_speed_kmh:.1f} km/h over {s_dist_m:.1f}m in {t_delta_sec:.1f}s")

        explanation = f"Estimated match probability is {estimated_prob:.2f}: " + ", ".join(reasons) + "."

    # Structured Output Payload
    return {
        "observation_a": obs_a.observation_id,
        "observation_b": obs_b.observation_id,
        "evidence": {
            "appearance_similarity": round(app_score, 4) if app_score is not None else None,
            "appearance_status": app_status,
            "identity_evidence_available": has_identity_evidence,
            "vehicle_type_match": type_match,
            "vehicle_type_status": type_status,
            "temporal_feasibility": round(t_score, 4),
            "spatial_feasibility": round(s_score, 4),
            "plate_similarity": round(plate_score, 4) if plate_score is not None else None,
            "time_difference_seconds": round(t_delta_sec, 2),
            "geographic_distance_meters": round(s_dist_m, 2),
            "required_speed_kmh": round(s_speed_kmh, 2) if s_speed_kmh != float("inf") else "infinite",
        },
        "same_vehicle_probability": estimated_prob,
        "explanation": explanation,
    }
