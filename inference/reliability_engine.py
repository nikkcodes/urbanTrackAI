"""
Reliability and Uncertainty Propagation Engine for UrbanTrack AI (Day 5).

Calculates deterministic and explainable:
- CameraReliability: Operational trust in physical camera/sensor feeds.
- ObservationReliability: Detection-level evidence quality and completeness.
- IdentityMatchReliability: Trust in pairwise identity fusion associations.
- TrajectoryReliability: Propagated uncertainty across reconstructed vehicle paths and unobserved gaps.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from schemas.observation_schema import Observation
from schemas.reliability_schema import (
    CameraReliability,
    IdentityMatchReliability,
    ObservationReliability,
    TrajectoryReliability,
)


# =============================================================================
# DETERMINISTIC HEURISTIC CONFIGURATION CONSTANTS
# =============================================================================
# NOTE: All formulas produce deterministic heuristic reliability scores for operational
# ranking and risk mitigation. They are NOT statistically calibrated Bayesian probabilities
# and do NOT represent mathematically proven true probabilities of correctness.

DEFAULT_CAMERA_RELIABILITY: float = 0.85

# Observation-level composite weighting
OBS_WEIGHT_PHYSICAL: float = 0.60
OBS_WEIGHT_IDENTITY: float = 0.40

# Physical observation component weights (sum = 1.0)
WEIGHT_CAM_RELIABILITY: float = 0.50
WEIGHT_DETECTION_CONF: float = 0.35
WEIGHT_COMPLETENESS: float = 0.15

# Cross-camera identity evidence component weights (sum = 1.0)
WEIGHT_REID_EMBEDDING: float = 0.50
WEIGHT_PLATE_CONFIDENCE: float = 0.50

# Trajectory and unobserved gap uncertainty scaling parameters
GAP_NODE_PENALTY_RATE: float = 0.06
GAP_MAX_NODE_PENALTY: float = 0.25
GAP_DURATION_SCALE_SEC: float = 600.0
GAP_DURATION_PENALTY_RATE: float = 0.10
GAP_MAX_DURATION_PENALTY: float = 0.15
GAP_MAX_TOTAL_PENALTY: float = 0.40
ROUTE_ENTROPY_THRESHOLD: float = 0.40
ROUTE_ENTROPY_PENALTY_RATE: float = 0.06
ROUTE_MAX_ENTROPY_PENALTY: float = 0.15
ROUTE_AMBIGUITY_THRESHOLD: float = 0.18
ROUTE_AMBIGUITY_PENALTY: float = 0.08


def evaluate_camera_reliability(
    camera_id: str,
    camera_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    uptime: Optional[float] = None,
    calibration_score: Optional[float] = None,
    weather_degradation: Optional[float] = None,
    **kwargs: Any,
) -> CameraReliability:
    """
    Evaluate the operational reliability of a physical camera or sensor source.

    Determines reliability using configured camera metadata, explicit configuration,
    or direct sensor quality parameters. If historical or explicit metrics are absent,
    assigns a deterministic default value without manufacturing fake historical data.

    Camera reliability represents strictly the operational trustworthiness of the sensor,
    completely independent of vehicle identity, route probability, or detection confidence.

    Args:
        camera_id: Unique camera identifier.
        camera_metadata: Optional camera metadata registry.
        config: Optional configuration dictionary.
        uptime: Optional historical uptime score [0.0, 1.0].
        calibration_score: Optional calibration alignment score [0.0, 1.0].
        weather_degradation: Optional weather/environmental impairment score [0.0, 1.0].

    Returns:
        CameraReliability: Structured reliability assessment.
    """
    config = config or {}
    cam_meta = (camera_metadata or {}).get(camera_id, {})

    # Check for directly provided sensor metrics
    if uptime is not None or calibration_score is not None or weather_degradation is not None:
        u = float(uptime if uptime is not None else 0.90)
        c = float(calibration_score if calibration_score is not None else 0.90)
        w = float(weather_degradation if weather_degradation is not None else 0.0)
        rel = max(0.05, min(1.0, 0.40 * u + 0.40 * c + 0.20 * (1.0 - w)))
        status = "degraded" if rel <= 0.45 or w > 0.50 else "known"
        factors = {"uptime": round(u, 4), "calibration": round(c, 4), "weather_degradation": round(w, 4)}
        flags = []
        if w > 0.50:
            flags.append("severe_environmental_degradation")
        factors["flags"] = flags
        explanation = (
            f"Camera '{camera_id}' evaluated from operational metrics: "
            f"uptime={u:.2f}, calibration={c:.2f}, weather={w:.2f} -> reliability={rel:.2f} (status: {status})."
        )
        return CameraReliability(
            camera_id=camera_id,
            reliability=rel,
            status=status,
            factors=factors,
            explanation=explanation,
        )

    # Check for explicitly configured reliability in metadata or config
    configured_rel = None
    if "reliability" in cam_meta:
        configured_rel = float(cam_meta["reliability"])
    elif "camera_reliability" in cam_meta:
        configured_rel = float(cam_meta["camera_reliability"])
    elif "camera_reliabilities" in config and camera_id in config["camera_reliabilities"]:
        configured_rel = float(config["camera_reliabilities"][camera_id])

    default_reliability = float(config.get("default_camera_reliability", 0.85))

    if configured_rel is not None:
        rel = max(0.0, min(1.0, configured_rel))
        status = "degraded" if rel <= 0.40 else "known"
        explanation = f"Camera '{camera_id}' has configured reliability {rel:.2f} (status: {status})."
        factors = {"configured_reliability": rel, "source": "camera_metadata_or_config"}
    else:
        rel = default_reliability
        status = "default"
        explanation = f"Camera '{camera_id}' reliability not specified; using deterministic default ({rel:.2f})."
        factors = {"configured_reliability": rel, "source": "default_fallback"}

    # Factor in explicit degraded flag if present
    if cam_meta.get("is_degraded") or cam_meta.get("status") == "degraded":
        rel = min(rel, 0.35)
        status = "degraded"
        factors["is_degraded"] = True
        explanation = f"Camera '{camera_id}' is marked as degraded/impaired (reliability: {rel:.2f})."

    return CameraReliability(
        camera_id=camera_id,
        reliability=rel,
        status=status,
        factors=factors,
        explanation=explanation,
    )


def evaluate_observation_reliability(
    obs: Union[Observation, Dict[str, Any]],
    camera_reliability: Optional[CameraReliability] = None,
    camera_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> ObservationReliability:
    """
    Evaluate the quality, completeness, and reliability of an individual vehicle observation.

    Missing evidence is NOT treated as positive evidence. Absence of Re-ID embeddings,
    license plates, or OCR confidence results in lower composite reliability and
    increased uncertainty.

    Args:
        obs: Vehicle observation instance or dictionary.
        camera_reliability: Optional pre-evaluated CameraReliability object.
        camera_metadata: Optional camera metadata dictionary.
        config: Optional configuration dictionary.

    Returns:
        ObservationReliability: Structured quality and uncertainty metrics.
    """
    config = config or {}
    degradation_factors: List[str] = []

    # Helper accessors
    if isinstance(obs, Observation):
        obs_id = obs.observation_id
        cam_id = obs.camera_id
        plate = obs.plate
        plate_conf = obs.plate_confidence
        app_emb = obs.appearance_embedding
        det_conf = obs.detection_confidence
        lat = obs.latitude
        lon = obs.longitude
        v_type = obs.vehicle_type
        bbox = obs.bbox
        obs_cam_rel = obs.camera_reliability
    else:
        obs_id = str(obs.get("observation_id", f"obs_{id(obs)}"))
        cam_id = str(obs.get("camera_id", "unknown_cam"))
        plate = obs.get("plate") or obs.get("plate_text")
        plate_conf = obs.get("plate_confidence") if obs.get("plate_confidence") is not None else obs.get("ocr_confidence")
        app_emb = obs.get("appearance_embedding") if obs.get("appearance_embedding") is not None else obs.get("reid_embedding")
        det_conf = obs.get("detection_confidence") if obs.get("detection_confidence") is not None else obs.get("confidence")
        lat = obs.get("latitude")
        lon = obs.get("longitude")
        v_type = obs.get("vehicle_type")
        bbox = obs.get("bbox")
        obs_cam_rel = obs.get("camera_reliability")

    # 1. Camera Reliability Factor
    if camera_reliability is not None:
        cam_rel_val = camera_reliability.reliability
    elif obs_cam_rel is not None:
        cam_rel_val = max(0.0, min(1.0, float(obs_cam_rel)))
    else:
        cam_eval = evaluate_camera_reliability(cam_id, camera_metadata=camera_metadata, config=config)
        cam_rel_val = cam_eval.reliability
        if "camera_reliability" not in (camera_metadata or {}).get(cam_id, {}) and cam_id not in (config.get("camera_reliabilities") or {}):
            degradation_factors.append("missing_camera_reliability")

    if cam_rel_val <= 0.45:
        degradation_factors.append("degraded_camera_sensor")

    available_evidence: List[str] = ["camera_reliability"]
    missing_evidence: List[str] = []
    factors: Dict[str, float] = {"camera_reliability": cam_rel_val}

    # 2. License Plate / OCR Evidence Factor
    if plate is not None and str(plate).strip():
        if plate_conf is not None:
            p_conf = max(0.0, min(1.0, float(plate_conf)))
            factors["plate_confidence"] = p_conf
            available_evidence.append("plate_confidence")
            if p_conf < 0.60 or "?" in str(plate):
                degradation_factors.append("low_ocr_confidence")
        else:
            p_conf = 0.70  # Neutral presence without confidence score
            factors["plate_confidence"] = p_conf
            available_evidence.append("plate_number")
    else:
        p_conf = 0.0
        missing_evidence.append("plate_evidence")

    # 3. Appearance / Re-ID Embedding Evidence Factor
    if app_emb is not None and isinstance(app_emb, (list, tuple)) and len(app_emb) > 0:
        # Check numerical validity and non-zero magnitude
        try:
            norm_sq = sum(float(x) ** 2 for x in app_emb)
            if norm_sq > 1e-6 and not math.isnan(norm_sq) and not math.isinf(norm_sq):
                app_score = 1.0
                available_evidence.append("appearance_embedding")
            else:
                app_score = 0.0
                missing_evidence.append("appearance_embedding_degenerate")
                degradation_factors.append("corrupted_appearance_embedding")
        except (ValueError, TypeError):
            app_score = 0.0
            missing_evidence.append("appearance_embedding_invalid")
            degradation_factors.append("corrupted_appearance_embedding")
    else:
        app_score = 0.0
        missing_evidence.append("appearance_embedding")

    factors["appearance_quality"] = app_score

    # 4. Object Detection Quality Factor
    if det_conf is not None:
        d_conf = max(0.0, min(1.0, float(det_conf)))
        factors["detection_confidence"] = d_conf
        available_evidence.append("detection_confidence")
        if d_conf < 0.25:
            degradation_factors.append("severe_detection_degradation")
    else:
        d_conf = 0.0
        missing_evidence.append("detection_confidence")
        degradation_factors.append("severe_detection_degradation")

    # 5. Observation Completeness Factor
    comp_fields = [
        lat is not None and lon is not None,
        v_type is not None,
        bbox is not None and len(bbox) == 4,
    ]
    comp_score = sum(1.0 for present in comp_fields if present) / len(comp_fields)
    factors["metadata_completeness"] = comp_score
    if comp_score >= 0.66:
        available_evidence.append("metadata_completeness")
    else:
        missing_evidence.append("incomplete_metadata")

    # --- SEPARATE PHYSICAL OBSERVATION QUALITY FROM IDENTITY EVIDENCE ---
    # A. Physical Observation Quality: Camera sensor trust, detection clarity, and spatial metadata
    raw_obs_quality = (
        WEIGHT_CAM_RELIABILITY * cam_rel_val
        + WEIGHT_DETECTION_CONF * d_conf
        + WEIGHT_COMPLETENESS * comp_score
    )
    sensor_envelope = 0.50 + 0.50 * cam_rel_val
    observation_quality = max(0.05, min(1.0, raw_obs_quality * sensor_envelope))

    # B. Identity Evidence Quality: Cross-camera re-identification features (Re-ID vector, plate OCR)
    identity_evidence_quality = max(
        0.0,
        min(1.0, WEIGHT_REID_EMBEDDING * app_score + WEIGHT_PLATE_CONFIDENCE * p_conf),
    )

    # C. Composite Tracking Reliability: Deterministic heuristic combination
    composite_reliability = max(
        0.05,
        min(
            1.0,
            OBS_WEIGHT_PHYSICAL * observation_quality
            + OBS_WEIGHT_IDENTITY * identity_evidence_quality,
        ),
    )

    factors["observation_quality"] = observation_quality
    factors["identity_evidence_quality"] = identity_evidence_quality
    factors["composite_reliability"] = composite_reliability

    # Concise, explanatory rationale distinguishing physical quality from identity evidence
    obs_desc = []
    if cam_rel_val >= 0.80 and d_conf >= 0.70:
        obs_desc.append(
            f"physical observation quality is strong ({observation_quality:.2f}) "
            f"supported by reliable camera ({cam_rel_val:.2f}) and detection ({d_conf:.2f})"
        )
    elif cam_rel_val <= 0.45:
        obs_desc.append(
            f"physical observation quality is degraded ({observation_quality:.2f}) "
            f"due to impaired camera reliability ({cam_rel_val:.2f})"
        )
    elif d_conf < 0.50:
        obs_desc.append(
            f"physical observation quality is reduced ({observation_quality:.2f}) "
            f"due to low detection confidence ({d_conf:.2f})"
        )
    else:
        obs_desc.append(f"physical observation quality is moderate ({observation_quality:.2f})")

    id_desc = []
    if app_score > 0 and p_conf > 0:
        id_desc.append(f"complete identity evidence available (Re-ID vector and plate OCR {p_conf:.2f})")
    elif app_score > 0:
        id_desc.append("identity evidence has Re-ID embedding but missing plate")
    elif p_conf > 0:
        id_desc.append(f"identity evidence has plate OCR ({p_conf:.2f}) but missing Re-ID embedding")
    else:
        id_desc.append("identity evidence is incomplete because no Re-ID embedding or license plate is available")

    explanation = (
        f"Observation '{obs_id}' on camera '{cam_id}' (composite trust: {composite_reliability:.2f}): "
        + "; ".join(obs_desc + id_desc) + "."
    )

    return ObservationReliability(
        observation_id=obs_id,
        camera_id=cam_id,
        observation_quality=observation_quality,
        identity_evidence_quality=identity_evidence_quality,
        reliability=composite_reliability,
        overall_reliability=composite_reliability,
        camera_reliability=cam_rel_val,
        detection_confidence=d_conf,
        ocr_reliability=p_conf if p_conf > 0 else None,
        reid_reliability=app_score if app_score > 0 else None,
        available_evidence=available_evidence,
        missing_evidence=missing_evidence,
        degradation_factors=degradation_factors,
        factors=factors,
        explanation=explanation,
    )


def evaluate_identity_uncertainty(
    match_result: Dict[str, Any],
    rel_a: ObservationReliability,
    rel_b: ObservationReliability,
    config: Optional[Dict[str, Any]] = None,
) -> IdentityMatchReliability:
    """
    Evaluate the propagated trustworthiness and uncertainty of an identity match.

    Combines endpoint observation reliabilities with the quality and depth of identity evidence.
    NOTE: same_vehicle_probability is an uncalibrated heuristic similarity score from Day 2,
    not a Bayesian posterior probability.

    Args:
        match_result: Output dictionary from Day 2 match_observations().
        rel_a: Evaluated reliability of start observation.
        rel_b: Evaluated reliability of end observation.
        config: Optional configuration dictionary.

    Returns:
        IdentityMatchReliability: Structured identity trust assessment.
    """
    match_prob = float(match_result.get("same_vehicle_probability", 0.0))
    evidence = match_result.get("evidence", {})

    # Geometric mean of endpoint observation reliabilities
    endpoint_rel = math.sqrt(max(0.01, rel_a.reliability) * max(0.01, rel_b.reliability))

    # Identity evidence depth factor
    app_sim = evidence.get("appearance_similarity")
    plate_sim = evidence.get("plate_similarity")

    if app_sim is not None and plate_sim is not None:
        ev_quality = 1.0
        ev_desc = "multi-modal appearance and license plate evidence"
    elif app_sim is not None:
        ev_quality = 0.85
        ev_desc = "appearance embedding similarity only"
    elif plate_sim is not None:
        ev_quality = 0.80
        ev_desc = "license plate similarity only"
    else:
        # Spatio-temporal feasibility alone (like Kanishka feed without ReID/plate)
        ev_quality = 0.40
        ev_desc = "spatio-temporal feasibility only (no direct visual/plate identity evidence)"

    # Combined identity reliability: geometric mean of endpoints * evidence depth * match probability
    if match_prob > 0:
        combined_rel = endpoint_rel * ev_quality * (0.50 + 0.50 * match_prob)
    else:
        combined_rel = 0.0

    combined_rel = max(0.0, min(1.0, combined_rel))
    uncertainty = round(1.0 - combined_rel, 4)

    explanation = (
        f"Identity match between '{rel_a.observation_id}' and '{rel_b.observation_id}' has reliability "
        f"{combined_rel:.2f} (uncertainty: {uncertainty:.2f}): endpoint reliability {endpoint_rel:.2f}, "
        f"supported by {ev_desc}."
    )

    return IdentityMatchReliability(
        observation_a=rel_a.observation_id,
        observation_b=rel_b.observation_id,
        match_probability=match_prob,
        endpoint_reliability=endpoint_rel,
        evidence_quality=ev_quality,
        combined_reliability=combined_rel,
        explanation=explanation,
    )


def propagate_trajectory_uncertainty(
    trajectory_id: str,
    obs_rel_a: ObservationReliability,
    obs_rel_b: ObservationReliability,
    candidate_routes: List[Any],
    is_gap: bool = False,
    unobserved_intermediate_nodes: Optional[List[str]] = None,
    gap_duration_seconds: Optional[float] = None,
    config: Optional[Dict[str, Any]] = None,
) -> TrajectoryReliability:
    """
    Propagate reliability and uncertainty into trajectory and candidate route inference.

    Flows from:
        Observation Reliability (A & B)
                ↓
        Endpoint Evidence Reliability
                ↓
        Unobserved Gap / Distance Penalty (if missing intermediate observations)
                ↓
        Route Ambiguity / Competition Entropy
                ↓
        Trajectory & Route Uncertainty

    Args:
        trajectory_id: Identifier of trajectory segment or gap.
        obs_rel_a: Reliability of origin sighting.
        obs_rel_b: Reliability of destination sighting.
        candidate_routes: List of candidate route objects.
        is_gap: Flag indicating if this interval contains unobserved intermediate transit.
        unobserved_intermediate_nodes: List of unobserved intermediate nodes (if gap).
        gap_duration_seconds: Elapsed duration of gap (if gap).
        config: Optional configuration dictionary.

    Returns:
        TrajectoryReliability: Comprehensive trajectory uncertainty model.
    """
    config = config or {}
    unobserved_nodes = unobserved_intermediate_nodes or []
    gap_dur = gap_duration_seconds or 0.0

    endpoint_rel = math.sqrt(max(0.01, obs_rel_a.reliability) * max(0.01, obs_rel_b.reliability))
    sources_of_uncertainty: List[str] = []

    if obs_rel_a.reliability < 0.70:
        sources_of_uncertainty.append(
            f"Origin observation '{obs_rel_a.observation_id}' has lower reliability ({obs_rel_a.reliability:.2f})"
        )
    if obs_rel_b.reliability < 0.70:
        sources_of_uncertainty.append(
            f"Destination observation '{obs_rel_b.observation_id}' has lower reliability ({obs_rel_b.reliability:.2f})"
        )

    # 1. Unobserved Gap Penalty
    gap_penalty = 0.0
    if is_gap:
        n_nodes = len(unobserved_nodes)
        dur_penalty = min(GAP_MAX_DURATION_PENALTY, (gap_dur / GAP_DURATION_SCALE_SEC) * GAP_DURATION_PENALTY_RATE)
        node_penalty = min(GAP_MAX_NODE_PENALTY, n_nodes * GAP_NODE_PENALTY_RATE)
        gap_penalty = min(GAP_MAX_TOTAL_PENALTY, node_penalty + dur_penalty)
        if gap_penalty > 0:
            sources_of_uncertainty.append(
                f"Missing intermediate observations across {n_nodes} junction(s) over {gap_dur:.1f}s gap"
            )

    # 2. Alternative Route Ambiguity & Entropy Check
    feasible_routes = [r for r in candidate_routes if getattr(r, "feasible", True)]
    route_entropy = 0.0
    if len(feasible_routes) > 1:
        for r in feasible_routes:
            p = getattr(r, "estimated_likelihood", 0.0)
            if p > 0:
                route_entropy -= p * math.log2(p)
        if route_entropy > ROUTE_ENTROPY_THRESHOLD:
            sources_of_uncertainty.append(f"Multiple alternative corridors with route entropy ({route_entropy:.2f})")

        p1 = getattr(feasible_routes[0], "estimated_likelihood", 0.5)
        p2 = getattr(feasible_routes[1], "estimated_likelihood", 0.5)
        diff = abs(p1 - p2)
        if diff < ROUTE_AMBIGUITY_THRESHOLD:
            ambiguity_penalty = ROUTE_AMBIGUITY_PENALTY
            sources_of_uncertainty.append(
                f"High route competition between top alternatives (P1={p1:.2f} vs P2={p2:.2f})"
            )
        else:
            ambiguity_penalty = 0.0
    else:
        ambiguity_penalty = 0.0

    # 3. Overall Trajectory Reliability
    entropy_penalty = min(ROUTE_MAX_ENTROPY_PENALTY, route_entropy * ROUTE_ENTROPY_PENALTY_RATE) if route_entropy > 0 else 0.0
    overall_rel = endpoint_rel * (1.0 - gap_penalty) * (1.0 - ambiguity_penalty) * (1.0 - entropy_penalty)
    overall_rel = max(0.05, min(1.0, overall_rel))
    overall_uncertainty = round(1.0 - overall_rel, 4)

    # 4. Attach Route-Level Reliability and Uncertainty to Candidate Routes
    for r in candidate_routes:
        is_feas = getattr(r, "feasible", True)
        if is_feas:
            raw_sc = getattr(r, "raw_score", 1.0)
            r_rel = overall_rel * min(1.0, max(0.2, raw_sc))
            r_unc = round(1.0 - r_rel, 4)
        else:
            r_rel = 0.0
            r_unc = 1.0

        if hasattr(r, "reliability"):
            r.reliability = round(r_rel, 4)
        if hasattr(r, "uncertainty"):
            r.uncertainty = round(r_unc, 4)
        if hasattr(r, "metadata"):
            if getattr(r, "metadata", None) is None:
                r.metadata = {}
            r.metadata["reliability"] = round(r_rel, 4)
            r.metadata["uncertainty"] = round(r_unc, 4)
            r.metadata["probability_semantics"] = "relative_estimated_likelihood_among_feasible_routes"
            r.metadata["reliability_semantics"] = "evidence_trustworthiness_supporting_corridor_inference"
            r.metadata["uncertainty_semantics"] = "evidence_uncertainty (1.0 - reliability)"

    endpoint_details = [
        {"observation_id": obs_rel_a.observation_id, "camera_id": obs_rel_a.camera_id, "reliability": round(obs_rel_a.reliability, 4)},
        {"observation_id": obs_rel_b.observation_id, "camera_id": obs_rel_b.camera_id, "reliability": round(obs_rel_b.reliability, 4)},
    ]

    reasons_str = "; ".join(sources_of_uncertainty) if sources_of_uncertainty else "High-quality endpoint observations and clear route corridor."
    explanation = (
        f"Trajectory '{trajectory_id}' overall reliability is {overall_rel:.2f} "
        f"(uncertainty: {overall_uncertainty:.2f}): {reasons_str}. "
        f"Route probabilities represent relative estimated likelihoods among feasible alternatives (summing to 1.0); "
        f"route reliability represents evidence trustworthiness supporting each corridor."
    )

    return TrajectoryReliability(
        trajectory_id=trajectory_id,
        overall_reliability=overall_rel,
        overall_uncertainty=overall_uncertainty,
        endpoint_reliabilities=endpoint_details,
        gap_penalty=gap_penalty,
        route_entropy=route_entropy,
        sources_of_uncertainty=sources_of_uncertainty,
        explanation=explanation,
    )
