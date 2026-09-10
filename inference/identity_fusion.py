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

    if isinstance(obs_a, dict):
        if "timestamp" not in obs_a:
            t_sec = obs_a.get("timestamp_seconds", 0.0)
            from datetime import datetime, timezone
            obs_a = dict(obs_a)
            obs_a["timestamp"] = datetime.fromtimestamp(t_sec, timezone.utc).isoformat()
        obs_a = Observation.from_dict(obs_a)
    if isinstance(obs_b, dict):
        if "timestamp" not in obs_b:
            t_sec = obs_b.get("timestamp_seconds", 0.0)
            from datetime import datetime, timezone
            obs_b = dict(obs_b)
            obs_b["timestamp"] = datetime.fromtimestamp(t_sec, timezone.utc).isoformat()
        obs_b = Observation.from_dict(obs_b)

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
    t_res = temporal_feasibility(obs_a, obs_b, camera_metadata=camera_metadata)
    t_score = t_res.get("feasibility_score")
    t_status = str(t_res.get("status", "unavailable"))
    t_delta_sec = t_res.get("delta_t_seconds")

    # 3. Spatial Feasibility Evidence
    s_res = spatial_feasibility(obs_a, obs_b, max_plausible_speed_kmh=max_speed_kmh, camera_metadata=camera_metadata)
    s_score = float(s_res["feasibility_score"])
    s_status = str(s_res["status"])
    s_dist_m = float(s_res["distance_meters"])
    s_speed_kmh = s_res.get("required_speed_kmh")

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
    plate_confidence = None
    if obs_a.plate is not None and obs_b.plate is not None:
        plate_score = plate_similarity(obs_a.plate, obs_b.plate)
        if obs_a.plate_confidence is not None and obs_b.plate_confidence is not None:
            try:
                plate_confidence = round(min(float(obs_a.plate_confidence), float(obs_b.plate_confidence)), 4)
            except (ValueError, TypeError):
                plate_confidence = None

    # --- HARD REJECTION / PHYSICAL IMPOSSIBILITY RULES ---
    is_rejected = False
    rejection_reason = ""

    if type_status == "incompatible":
        is_rejected = True
        rejection_reason = f"Incompatible vehicle types ({obs_a.vehicle_type} vs {obs_b.vehicle_type})."
    elif t_status in ("impossible_negative_time", "impossible_simultaneous_different_cameras", "impossible_simultaneous_same_camera_distinct_bbox"):
        is_rejected = True
        rejection_reason = f"Physically impossible temporal alignment ({t_res.get('explanation')})."
    elif s_status in ("impossible_speed", "physically_impossible_speed"):
        is_rejected = True
        rejection_reason = f"Physically impossible travel speed ({s_res.get('explanation')})."

    has_identity_evidence = (app_score is not None) or (plate_score is not None)

    # --- ESTIMATED MATCH PROBABILITY CALCULATION ---
    if is_rejected:
        estimated_prob = 0.0
        explanation = f"Match rejected (0.0): {rejection_reason}"
    else:
        # Baseline spatio-temporal and vehicle type feasibility (gating factor)
        if t_score is not None:
            st_composite = (float(t_score) + s_score) / 2.0
        else:
            # Temporal evidence unavailable: do not penalize or fabricate 0.5, use available spatial signal
            st_composite = s_score
        feasibility_score = 0.70 * st_composite + 0.30 * type_score

        if has_identity_evidence:
            # Combine available identity features (appearance and/or plate)
            if app_score is not None and plate_score is not None:
                # Phase 3: Adaptive weighting if actual OCR confidence is provided
                if plate_confidence is not None:
                    # Scale plate contribution by verified OCR confidence [0.30, 0.60]
                    w_plate = 0.30 + 0.30 * plate_confidence
                    w_app = 1.0 - w_plate
                    id_score = w_app * app_score + w_plate * plate_score
                else:
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

        if s_dist_m > 0 and t_delta_sec is not None and t_delta_sec > 0 and s_speed_kmh is not None:
            reasons.append(f"required travel speed is {s_speed_kmh:.1f} km/h over {s_dist_m:.1f}m in {t_delta_sec:.1f}s")
        elif t_status == "unavailable":
            reasons.append(f"cross-camera temporal evidence is unavailable ({t_res.get('reason')})")

        explanation = f"Estimated match probability is {estimated_prob:.2f}: " + ", ".join(reasons) + "."

    # Day 5: Evaluate observation reliabilities and identity uncertainty
    from .reliability_engine import evaluate_identity_uncertainty, evaluate_observation_reliability

    rel_a = evaluate_observation_reliability(obs_a, camera_metadata=camera_metadata, config=config)
    rel_b = evaluate_observation_reliability(obs_b, camera_metadata=camera_metadata, config=config)
    id_rel = evaluate_identity_uncertainty(
        {
            "same_vehicle_probability": estimated_prob,
            "evidence": {
                "appearance_similarity": app_score,
                "plate_similarity": plate_score,
            },
        },
        rel_a,
        rel_b,
        config=config,
    )

    # --- EVIDENCE LEDGER (Phases 2, 3, 4) ---
    # Explicit status: missing | available | contradictory | invalid | unavailable
    # Explicit detailed_status: available_supportive | available_contradictory | missing | unavailable | invalid

    # Plate ledger
    if obs_a.plate is None or obs_b.plate is None:
        plate_ledger_status = "missing"
        plate_detailed_status = "missing"
        plate_ledger_value = None
        plate_ledger_contribution = None
    elif plate_score is not None:
        # Contradictory rule: both present, length >= 4, and normalized similarity < 0.30 (outside OCR variation)
        clean_a = "".join(c for c in str(obs_a.plate).upper() if c.isalnum())
        clean_b = "".join(c for c in str(obs_b.plate).upper() if c.isalnum())
        if len(clean_a) >= 4 and len(clean_b) >= 4 and plate_score < 0.30:
            plate_ledger_status = "contradictory"
            plate_detailed_status = "available_contradictory"
        elif plate_score >= 0.85:
            plate_ledger_status = "available"
            plate_detailed_status = "available_supportive"
        else:
            plate_ledger_status = "available"
            plate_detailed_status = "available_weak"
        plate_ledger_value = round(plate_score, 4)
        plate_ledger_contribution = round(0.45 * plate_score, 4) if has_identity_evidence and app_score is not None else (round(plate_score, 4) if has_identity_evidence else None)
    else:
        plate_ledger_status = "invalid"
        plate_detailed_status = "invalid"
        plate_ledger_value = None
        plate_ledger_contribution = None

    # Appearance ledger
    if app_status == "missing":
        app_ledger_status = "missing"
        app_detailed_status = "missing"
        app_ledger_value = None
        app_ledger_contribution = None
    elif app_status in ("invalid", "invalid_mismatched"):
        app_ledger_status = "invalid"
        app_detailed_status = "invalid"
        app_ledger_value = None
        app_ledger_contribution = None
    elif app_score is not None:
        if app_score >= 0.75:
            app_ledger_status = "available"
            app_detailed_status = "available_supportive"
        else:
            app_ledger_status = "available"
            app_detailed_status = "available_weak"
        app_ledger_value = round(app_score, 4)
        app_ledger_contribution = round(0.55 * app_score, 4) if (has_identity_evidence and plate_score is not None) else round(app_score, 4)
    else:
        app_ledger_status = "unavailable"
        app_detailed_status = "unavailable"
        app_ledger_value = None
        app_ledger_contribution = None

    # Vehicle type ledger
    if type_status == "compatible":
        type_detailed_status = "available_supportive"
    elif type_status == "incompatible":
        type_detailed_status = "available_contradictory"
    else:
        type_detailed_status = "unavailable"
    type_ledger_status = type_status
    type_ledger_value = type_score
    type_ledger_contribution = round(0.30 * type_score, 4)

    # Temporal ledger — map from t_status codes
    _temporal_impossible_statuses = {
        "impossible_negative_time", "impossible_simultaneous_different_cameras",
        "impossible_simultaneous_same_camera_distinct_bbox",
    }
    _temporal_unavailable_statuses = {"unavailable", "different_cameras_no_shared_reference"}
    if t_score is None and t_status in _temporal_unavailable_statuses:
        temporal_ledger_status = "unavailable"
        temporal_detailed_status = "unavailable"
        temporal_ledger_value = None
    elif t_status in _temporal_impossible_statuses:
        temporal_ledger_status = "impossible"
        temporal_detailed_status = "available_contradictory"
        temporal_ledger_value = 0.0
    elif t_score is not None and t_status in ("feasible", "same_camera", "same_camera_same_frame"):
        temporal_ledger_status = "feasible"
        temporal_detailed_status = "available_supportive"
        temporal_ledger_value = round(float(t_score), 4)
    elif t_score is not None:
        temporal_ledger_status = "unverified"
        temporal_detailed_status = "unverified"
        temporal_ledger_value = round(float(t_score), 4)
    else:
        temporal_ledger_status = "unavailable"
        temporal_detailed_status = "unavailable"
        temporal_ledger_value = None
    temporal_ledger_contribution = round(0.70 * 0.5 * temporal_ledger_value, 4) if temporal_ledger_value is not None else None

    # Spatial ledger
    _spatial_impossible_statuses = {"impossible_speed", "physically_impossible_speed"}
    if s_status in _spatial_impossible_statuses:
        spatial_ledger_status = "impossible"
        spatial_detailed_status = "available_contradictory"
    elif s_score > 0.0:
        spatial_ledger_status = "feasible"
        spatial_detailed_status = "available_supportive"
    else:
        spatial_ledger_status = "unavailable"
        spatial_detailed_status = "unavailable"
    spatial_ledger_value = round(s_score, 4)
    spatial_ledger_contribution = round(0.70 * 0.5 * s_score, 4)

    evidence_ledger = {
        "plate": {
            "status": plate_ledger_status,
            "detailed_status": plate_detailed_status,
            "value": plate_ledger_value,
            "score": plate_ledger_value,
            "confidence": plate_confidence,
            "contribution": plate_ledger_contribution,
            "source": "actual_observation" if (obs_a.plate is not None or obs_b.plate is not None) else "missing",
        },
        "appearance": {
            "status": app_ledger_status,
            "detailed_status": app_detailed_status,
            "value": app_ledger_value,
            "score": app_ledger_value,
            "contribution": app_ledger_contribution,
            "source": "actual_observation" if (obs_a.appearance_embedding is not None or obs_b.appearance_embedding is not None) else "missing",
        },
        "vehicle_type": {
            "status": type_ledger_status,
            "detailed_status": type_detailed_status,
            "value": type_ledger_value,
            "score": type_ledger_value,
            "contribution": type_ledger_contribution,
            "source": "actual_observation" if (obs_a.vehicle_type or obs_b.vehicle_type) else "missing",
        },
        "temporal": {
            "status": temporal_ledger_status,
            "detailed_status": temporal_detailed_status,
            "value": temporal_ledger_value,
            "score": temporal_ledger_value,
            "contribution": temporal_ledger_contribution,
            "delta_t_seconds": round(float(t_delta_sec), 2) if t_delta_sec is not None else None,
            "source": "actual_observation",
        },
        "spatial": {
            "status": spatial_ledger_status,
            "detailed_status": spatial_detailed_status,
            "value": spatial_ledger_value,
            "score": spatial_ledger_value,
            "contribution": spatial_ledger_contribution,
            "distance_meters": round(s_dist_m, 2),
            "source": "actual_observation" if (obs_a.latitude is not None and obs_b.latitude is not None) else "unavailable",
        },
    }

    # Structured Output Payload (100% backward compatible with Day 2)
    return {
        "observation_a": obs_a.observation_id,
        "observation_b": obs_b.observation_id,
        "evidence": {
            "appearance_similarity": round(app_score, 4) if app_score is not None else None,
            "appearance_status": app_status,
            "identity_evidence_available": has_identity_evidence,
            "vehicle_type_match": type_match,
            "vehicle_type_status": type_status,
            "temporal_feasibility": round(float(t_score), 4) if t_score is not None else None,
            "temporal_status": t_status,
            "temporal_evidence": t_res.get("temporal_evidence") or t_res,
            "spatial_feasibility": round(s_score, 4),
            "plate_similarity": round(plate_score, 4) if plate_score is not None else None,
            "time_difference_seconds": round(float(t_delta_sec), 2) if t_delta_sec is not None else None,
            "geographic_distance_meters": round(s_dist_m, 2),
            "required_speed_kmh": round(float(s_speed_kmh), 2) if (s_speed_kmh is not None and s_speed_kmh != float("inf")) else (None if s_speed_kmh is None else "infinite"),
        },
        "evidence_ledger": evidence_ledger,
        "same_vehicle_probability": estimated_prob,
        "explanation": explanation,
        "observation_reliability": {
            "observation_a": rel_a.to_dict(),
            "observation_b": rel_b.to_dict(),
            "endpoint_reliability": round(id_rel.endpoint_reliability, 4),
        },
        "reliability": id_rel.to_dict(),
        "uncertainty": round(id_rel.uncertainty, 4),
        "uncertainty_details": {
            "score": round(id_rel.uncertainty, 4),
            "level": "low" if id_rel.uncertainty <= 0.25 else ("moderate" if id_rel.uncertainty <= 0.50 else ("high" if id_rel.uncertainty <= 0.75 else "critical")),
            "explanation": id_rel.explanation,
        },
    }
