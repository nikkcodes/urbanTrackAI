"""
Reliability and Uncertainty Propagation Engine for UrbanTrack AI (Day 5).

Calculates deterministic and explainable:
- CameraReliability: Operational trust in physical camera/sensor feeds.
- ObservationReliability: Detection-level evidence quality and completeness.
- IdentityMatchReliability: Trust in pairwise identity fusion associations.
- TrajectoryReliability: Propagated uncertainty across reconstructed vehicle paths and unobserved gaps.
"""

from __future__ import annotations
from dataclasses import dataclass, field

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



def profile_camera_reliability_from_observations(
    camera_id: str,
    observations: List[Union[Observation, Dict[str, Any]]],
    min_sample_threshold: int = 3,
    config: Optional[Dict[str, Any]] = None,
) -> CameraReliability:
    """
    Empirically evaluate physical camera reliability directly from observable stream metrics.

    Strict Zero-Magic-Number Invariant:
    If observations are absent or sample count < min_sample_threshold, returns status='unknown'
    or 'insufficient_evidence' with reliability=None. Never invents an arbitrary 0.85 default.

    Measurable signals:
    1. Detection confidence distribution (YOLO mean detection score).
    2. ReID embedding completeness and finite vector validity.
    3. OCR success rate and plate confidence.
    4. Tracklet continuity and tracking stability.
    """
    config = config or {}
    # Filter observations matching camera_id if mixed
    matching = []
    for o in observations:
        cid = getattr(o, "camera_id", o.get("camera_id") if isinstance(o, dict) else None)
        if cid == camera_id or not cid:
            matching.append(o)

    sample_count = len(matching)
    if sample_count == 0:
        return CameraReliability(
            camera_id=camera_id,
            reliability=None,
            status="unknown",
            factors={"sample_count": 0, "source": "no_observations_available"},
            explanation=f"Camera '{camera_id}' has zero observations; operational reliability is unknown.",
        )

    if sample_count < min_sample_threshold:
        return CameraReliability(
            camera_id=camera_id,
            reliability=None,
            status="insufficient_evidence",
            factors={"sample_count": sample_count, "min_required": min_sample_threshold, "source": "insufficient_samples"},
            explanation=f"Camera '{camera_id}' has insufficient sample size ({sample_count} < {min_sample_threshold}); reliable metric cannot be derived.",
        )

    # 1. Detection confidence
    det_confs = []
    for o in matching:
        c = getattr(o, "detection_confidence", None)
        if c is None and isinstance(o, dict):
            c = o.get("detection_confidence", o.get("confidence"))
        if c is not None:
            try:
                det_confs.append(float(c))
            except (ValueError, TypeError):
                pass
    mean_det_conf = (sum(det_confs) / len(det_confs)) if det_confs else 0.80

    # 2. ReID embedding validity
    reid_valid_count = 0
    for o in matching:
        emb = getattr(o, "appearance_embedding", None)
        if emb is None and isinstance(o, dict):
            emb = o.get("appearance_embedding")
        if emb is not None and isinstance(emb, (list, tuple)) and len(emb) > 0:
            if not any(math.isnan(x) or math.isinf(x) for x in emb):
                reid_valid_count += 1
    reid_valid_rate = reid_valid_count / sample_count

    # 3. OCR success rate and plate confidence
    plate_reads = 0
    plate_confs = []
    for o in matching:
        p = getattr(o, "plate", None)
        if p is None and isinstance(o, dict):
            p = o.get("plate")
        if p and len(str(p).strip()) >= 2:
            plate_reads += 1
            pc = getattr(o, "plate_confidence", None)
            if pc is None and isinstance(o, dict):
                pc = o.get("plate_confidence")
            if pc is not None:
                try:
                    plate_confs.append(float(pc))
                except (ValueError, TypeError):
                    pass
    ocr_rate = plate_reads / sample_count
    mean_plate_conf = (sum(plate_confs) / len(plate_confs)) if plate_confs else None

    # 4. Tracking stability
    track_ids = set()
    for o in matching:
        tid = getattr(o, "track_id", None)
        if tid is None and isinstance(o, dict):
            tid = o.get("track_id")
        if tid is not None:
            track_ids.add(str(tid))
    unique_tracks = len(track_ids)
    # Ratio of observations per track: stable tracking has multiple observations per track
    obs_per_track = sample_count / max(1, unique_tracks)
    tracking_stability = min(1.0, max(0.3, obs_per_track / 5.0))

    # Composite empirical score
    if plate_reads > 0:
        ocr_contrib = 0.5 * ocr_rate + 0.5 * (mean_plate_conf if mean_plate_conf is not None else 0.8)
        rel = 0.35 * mean_det_conf + 0.35 * reid_valid_rate + 0.15 * ocr_contrib + 0.15 * tracking_stability
    else:
        # CCTV feed without ANPR expectation
        rel = 0.45 * mean_det_conf + 0.45 * reid_valid_rate + 0.10 * tracking_stability

    rel = round(max(0.05, min(1.0, rel)), 4)
    status = "degraded" if rel < 0.45 else "computed_data_driven"

    factors = {
        "sample_count": sample_count,
        "mean_detection_confidence": round(mean_det_conf, 4),
        "reid_validity_rate": round(reid_valid_rate, 4),
        "ocr_success_rate": round(ocr_rate, 4),
        "mean_plate_confidence": round(mean_plate_conf, 4) if mean_plate_conf is not None else None,
        "unique_tracks": unique_tracks,
        "tracking_stability": round(tracking_stability, 4),
        "source": "empirical_stream_profiler",
    }

    explanation = (
        f"Camera '{camera_id}' empirically profiled from {sample_count} observations: "
        f"det_conf={mean_det_conf:.2f}, reid_valid={reid_valid_rate:.2f}, "
        f"ocr_rate={ocr_rate:.2f} -> data-driven reliability={rel:.2f} ({status})."
    )

    return CameraReliability(
        camera_id=camera_id,
        reliability=rel,
        status=status,
        factors=factors,
        explanation=explanation,
    )

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

    obs_list = kwargs.get("observations") or cam_meta.get("observations")

    default_reliability = float(config.get("default_camera_reliability", 0.85))

    if configured_rel is not None:
        rel = max(0.0, min(1.0, configured_rel))
        status = "degraded" if rel <= 0.40 else "known"
        explanation = f"Camera '{camera_id}' has configured reliability {rel:.2f} (status: {status})."
        factors = {"configured_reliability": rel, "source": "camera_metadata_or_config"}
    elif obs_list:
        return profile_camera_reliability_from_observations(camera_id, obs_list, config=config)
    elif config.get("require_data_driven", False):
        return CameraReliability(
            camera_id=camera_id,
            reliability=None,
            status="insufficient_evidence",
            factors={"source": "insufficient_evidence", "sample_count": 0},
            explanation=f"Camera '{camera_id}' has insufficient evidence to derive reliability.",
        )
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
    match_prob = float(match_result.get("same_vehicle_score", match_result.get("same_vehicle_probability", 0.0)))
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


# =============================================================================
# DYNAMIC TIME-DEPENDENT CAMERA RELIABILITY TRACKER R(c, t)
# =============================================================================

@dataclass
class DynamicCameraReliabilitySnapshot:
    """Point-in-time reliability evaluation for camera c at timestamp t."""
    camera_id: str
    timestamp_seconds: float
    reliability: Optional[float]
    status: str
    detection_quality: Optional[float]
    ocr_quality: Optional[float]
    tracking_stability: Optional[float]
    observation_density: Optional[float]
    sample_count: int
    is_smoothed: bool
    explanation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "timestamp_seconds": self.timestamp_seconds,
            "reliability": round(self.reliability, 4) if self.reliability is not None else None,
            "status": self.status,
            "detection_quality": round(self.detection_quality, 4) if self.detection_quality is not None else None,
            "ocr_quality": round(self.ocr_quality, 4) if self.ocr_quality is not None else None,
            "tracking_stability": round(self.tracking_stability, 4) if self.tracking_stability is not None else None,
            "observation_density": round(self.observation_density, 4) if self.observation_density is not None else None,
            "sample_count": self.sample_count,
            "is_smoothed": self.is_smoothed,
            "explanation": self.explanation,
        }


class DynamicCameraReliabilityTracker:
    """
    Dynamic, time-dependent camera reliability tracker R(c, t).

    Evaluates camera reliability over temporal sliding windows with exponential smoothing
    to prevent abrupt frame-to-frame fluctuations while accurately reflecting sensor degradation.
    """

    def __init__(
        self,
        window_size_seconds: float = 30.0,
        smoothing_alpha: float = 0.35,
        min_samples_for_evaluation: int = 3,
    ) -> None:
        self.window_size_seconds = window_size_seconds
        self.smoothing_alpha = smoothing_alpha
        self.min_samples = min_samples_for_evaluation
        self._history: Dict[str, List[Observation]] = {}
        self._last_smoothed_reliability: Dict[str, float] = {}

    def add_observations(self, observations: List[Observation]) -> None:
        """Ingest observations into the temporal history buffer."""
        for obs in observations:
            cid = obs.camera_id or "UNKNOWN"
            if cid not in self._history:
                self._history[cid] = []
            self._history[cid].append(obs)
        # Ensure chronological ordering
        for cid in self._history:
            self._history[cid].sort(key=lambda o: o.timestamp_seconds)

    def evaluate_reliability(
        self,
        camera_id: str,
        timestamp_seconds: float,
    ) -> DynamicCameraReliabilitySnapshot:
        """
        Compute R(c, t) in the window [t - window_size, t].
        """
        if camera_id not in self._history or not self._history[camera_id]:
            return DynamicCameraReliabilitySnapshot(
                camera_id=camera_id,
                timestamp_seconds=timestamp_seconds,
                reliability=None,
                status="no_history",
                detection_quality=None,
                ocr_quality=None,
                tracking_stability=None,
                observation_density=None,
                sample_count=0,
                is_smoothed=False,
                explanation=f"Camera '{camera_id}' has no recorded observations.",
            )

        t_start = max(0.0, timestamp_seconds - self.window_size_seconds)
        t_end = timestamp_seconds

        window_obs = [
            o for o in self._history[camera_id]
            if t_start <= o.timestamp_seconds <= t_end
        ]

        n_samples = len(window_obs)
        if n_samples < self.min_samples:
            return DynamicCameraReliabilitySnapshot(
                camera_id=camera_id,
                timestamp_seconds=timestamp_seconds,
                reliability=None,
                status="insufficient_evidence",
                detection_quality=None,
                ocr_quality=None,
                tracking_stability=None,
                observation_density=None,
                sample_count=n_samples,
                is_smoothed=False,
                explanation=(
                    f"Camera '{camera_id}' at t={timestamp_seconds:.1f}s has only {n_samples} "
                    f"observation(s) in window [{t_start:.1f}s, {t_end:.1f}s] (< min {self.min_samples})."
                ),
            )

        # 1. Detection Quality
        det_confs = [o.detection_confidence for o in window_obs if o.detection_confidence is not None]
        det_q = sum(det_confs) / len(det_confs) if det_confs else 0.50

        # 2. OCR Quality
        ocr_valid = [o for o in window_obs if o.plate_text and len(o.plate_text) >= 4]
        ocr_rate = len(ocr_valid) / n_samples
        ocr_confs = [o.ocr_confidence for o in ocr_valid if o.ocr_confidence is not None]
        mean_ocr_conf = sum(ocr_confs) / len(ocr_confs) if ocr_confs else (0.80 if ocr_valid else 0.0)
        ocr_q = (0.5 * ocr_rate + 0.5 * mean_ocr_conf) if ocr_valid else 0.50

        # 3. Tracking Stability
        stable_tracks = [o for o in window_obs if o.local_track_history and len(o.local_track_history) >= 2]
        has_tracking_info = any(o.local_track_history is not None for o in window_obs)
        track_q = (len(stable_tracks) / n_samples) if has_tracking_info else 0.80

        # 4. Observation Density (scaled relative to active rate)
        expected_density = max(1.0, (t_end - t_start) / 10.0)
        density_q = min(1.0, n_samples / expected_density)

        # Raw window score: transparent weighted combination of empirical components
        raw_r = 0.40 * det_q + 0.30 * ocr_q + 0.20 * track_q + 0.10 * density_q
        raw_r = max(0.0, min(1.0, raw_r))

        # Temporal Exponential Smoothing
        prev_r = self._last_smoothed_reliability.get(camera_id)
        if prev_r is not None:
            smoothed_r = self.smoothing_alpha * raw_r + (1.0 - self.smoothing_alpha) * prev_r
            is_smoothed = True
        else:
            smoothed_r = raw_r
            is_smoothed = False

        self._last_smoothed_reliability[camera_id] = smoothed_r

        explanation = (
            f"Camera '{camera_id}' reliability at t={timestamp_seconds:.1f}s is {smoothed_r:.2f} "
            f"(detection: {det_q:.2f}, OCR: {ocr_q:.2f}, tracking: {track_q:.2f}, density: {density_q:.2f}, "
            f"{n_samples} samples in {self.window_size_seconds:.0f}s window, "
            f"smoothed: {is_smoothed})."
        )

        return DynamicCameraReliabilitySnapshot(
            camera_id=camera_id,
            timestamp_seconds=timestamp_seconds,
            reliability=smoothed_r,
            status="healthy" if smoothed_r >= 0.70 else ("degraded" if smoothed_r >= 0.40 else "critical"),
            detection_quality=det_q,
            ocr_quality=ocr_q,
            tracking_stability=track_q,
            observation_density=density_q,
            sample_count=n_samples,
            is_smoothed=is_smoothed,
            explanation=explanation,
        )
