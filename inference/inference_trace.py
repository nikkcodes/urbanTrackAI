"""
Inference Trace module for UrbanTrack AI (Phase G).

Provides a structured, serializable explanation trace for every inference decision
produced by the multi-stage mobility pipeline.

Key distinctions:
  - `generated_at` records when the trace was generated (pipeline timestamp).
  - Observation timestamps are stored inside evidence_items only — never conflated
    with the trace generation timestamp.
  - Evidence items carry status, value, and contribution exactly as computed
    by the identity fusion evidence ledger.
  - No evidence is fabricated. If evidence is missing, the status is 'missing' or 'unavailable'.
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# InferenceTrace dataclass
# ---------------------------------------------------------------------------

@dataclass
class InferenceTrace:
    """
    A structured, serializable record of an individual inference decision.

    Fields:
        trace_id:        Unique identifier for this trace record.
        entity_type:     Category of the entity being traced:
                         'identity_pair' | 'cluster' | 'trajectory_segment' |
                         'trajectory_hypothesis' | 'sparse_gap'
        entity_id:       Identifier of the traced entity (e.g. observation pair, cluster ID).
        stage:           Pipeline stage that produced this inference:
                         'identity_fusion' | 'clustering' | 'trajectory' | 'sparse_inference'
        evidence_items:  List of per-modality evidence dicts containing
                         {signal_type, status, value, contribution, note}.
        decision:        Outcome label: 'matched' | 'rejected' | 'ambiguous' |
                         'insufficient_evidence' | 'candidate' | 'singleton' |
                         'feasible' | 'infeasible' | 'unverified'
        decision_score:  The numeric evidence score driving the decision (heuristic, not probability).
        explanation:     Human-readable rationale derived from actual computed values.
                         Never claims evidence that does not exist.
        generated_at:    UNIX timestamp (float) recording when this trace was produced.
                         NOT a vehicle observation timestamp.
    """

    trace_id: str
    entity_type: str
    entity_id: str
    stage: str
    evidence_items: List[Dict[str, Any]] = field(default_factory=list)
    decision: str = "unverified"
    decision_score: float = 0.0
    explanation: str = ""
    generated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary for JSON output or storage."""
        return {
            "trace_id": self.trace_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "stage": self.stage,
            "evidence_items": list(self.evidence_items),
            "decision": self.decision,
            "decision_score": round(self.decision_score, 6),
            "explanation": self.explanation,
            "generated_at": round(self.generated_at, 6),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InferenceTrace":
        """Deserialize from a plain dictionary. Preserves all fields."""
        return cls(
            trace_id=str(data["trace_id"]),
            entity_type=str(data["entity_type"]),
            entity_id=str(data["entity_id"]),
            stage=str(data["stage"]),
            evidence_items=list(data.get("evidence_items", [])),
            decision=str(data.get("decision", "unverified")),
            decision_score=float(data.get("decision_score", 0.0)),
            explanation=str(data.get("explanation", "")),
            generated_at=float(data.get("generated_at", 0.0)),
        )


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------

def build_identity_pair_trace(
    match_result: Dict[str, Any],
    trace_id: Optional[str] = None,
) -> InferenceTrace:
    """
    Construct an InferenceTrace from a match_observations() output dict.

    The trace is derived entirely from the computed match_result.
    No evidence is fabricated or inferred beyond what the fusion engine produced.

    Args:
        match_result: Output dict from identity_fusion.match_observations().
        trace_id: Optional explicit trace ID. Auto-generated if None.

    Returns:
        InferenceTrace for the identity pair decision.
    """
    obs_a = str(match_result.get("observation_a", "unknown_a"))
    obs_b = str(match_result.get("observation_b", "unknown_b"))
    entity_id = f"{obs_a}:{obs_b}"
    tid = trace_id or f"trace_pair_{obs_a}_{obs_b}"

    score = float(match_result.get("same_vehicle_probability", 0.0))
    explanation_raw = str(match_result.get("explanation", ""))

    # Decision label based on score threshold (matches graph threshold convention)
    if score >= 0.70:
        decision = "matched"
    elif score >= 0.40:
        decision = "ambiguous"
    elif score == 0.0 and "rejected" in explanation_raw.lower():
        decision = "rejected"
    else:
        decision = "insufficient_evidence"

    # Build evidence items from the evidence_ledger (Phase B output)
    ledger = match_result.get("evidence_ledger", {})
    evidence = match_result.get("evidence", {})
    evidence_items: List[Dict[str, Any]] = []

    for signal_type in ("plate", "appearance", "vehicle_type", "temporal", "spatial"):
        entry = ledger.get(signal_type, {})
        item: Dict[str, Any] = {
            "signal_type": signal_type,
            "status": entry.get("status", "unavailable"),
            "value": entry.get("value"),
            "contribution": entry.get("contribution"),
        }
        # Add modality-specific notes
        if signal_type == "temporal":
            item["delta_t_seconds"] = entry.get("delta_t_seconds")
        elif signal_type == "spatial":
            item["distance_meters"] = entry.get("distance_meters")
        evidence_items.append(item)

    # If no ledger available (pre-Phase B output), fall back to raw evidence dict
    if not ledger and evidence:
        evidence_items = [
            {"signal_type": "appearance", "status": evidence.get("appearance_status", "unavailable"),
             "value": evidence.get("appearance_similarity"), "contribution": None},
            {"signal_type": "vehicle_type", "status": evidence.get("vehicle_type_status", "unknown"),
             "value": evidence.get("vehicle_type_match"), "contribution": None},
            {"signal_type": "temporal", "status": evidence.get("temporal_status", "unavailable"),
             "value": evidence.get("temporal_feasibility"), "contribution": None,
             "delta_t_seconds": evidence.get("time_difference_seconds")},
            {"signal_type": "spatial", "status": "feasible" if evidence.get("spatial_feasibility", 0) > 0 else "unavailable",
             "value": evidence.get("spatial_feasibility"), "contribution": None,
             "distance_meters": evidence.get("geographic_distance_meters")},
            {"signal_type": "plate", "status": "available" if evidence.get("plate_similarity") is not None else "missing",
             "value": evidence.get("plate_similarity"), "contribution": None},
        ]

    # Uncertainty sources
    uncertainty_details = match_result.get("uncertainty_details", {})

    return InferenceTrace(
        trace_id=tid,
        entity_type="identity_pair",
        entity_id=entity_id,
        stage="identity_fusion",
        evidence_items=evidence_items,
        decision=decision,
        decision_score=score,
        explanation=explanation_raw or f"Evidence score={score:.4f}, decision={decision}.",
        generated_at=time.time(),
    )


def build_cluster_trace(
    cluster: Dict[str, Any],
    trace_id: Optional[str] = None,
) -> InferenceTrace:
    """
    Construct an InferenceTrace from a get_candidate_identities() cluster dict.

    Args:
        cluster: A candidate cluster dict as returned by IdentityGraph.get_candidate_identities().
        trace_id: Optional explicit trace ID. Auto-generated if None.

    Returns:
        InferenceTrace for the cluster decision.
    """
    identity_id = str(cluster.get("identity_id", "unknown_cluster"))
    tid = trace_id or f"trace_cluster_{identity_id}"

    identity_status = str(cluster.get("identity_status", "unknown"))
    identity_confidence = cluster.get("identity_confidence")
    member_count = cluster.get("member_observations_count", 0)
    consistency = cluster.get("cluster_consistency", {})

    # Decision based on identity_status + consistency
    if identity_status == "unconfirmed_singleton":
        decision = "singleton"
        score = 0.0
    elif consistency.get("is_consistent", True) and identity_confidence is not None:
        decision = "candidate"
        score = float(identity_confidence)
    elif not consistency.get("is_consistent", True):
        decision = "ambiguous"
        score = float(identity_confidence or 0.0)
    else:
        decision = "insufficient_evidence"
        score = 0.0

    # Evidence summary items
    ev_summary = cluster.get("identity_evidence_summary", {})
    evidence_items: List[Dict[str, Any]] = []
    for signal_type in ("appearance", "temporal", "spatial", "vehicle_type"):
        evidence_items.append({
            "signal_type": signal_type,
            "status": ev_summary.get(signal_type, "unknown"),
            "value": None,
            "contribution": None,
        })

    # Consistency info
    violations = consistency.get("violations", [])
    explanation_parts = [
        f"Cluster {identity_id}: {member_count} member observation(s).",
        f"Identity status: {identity_status}.",
        f"Consistency: {consistency.get('status', 'unknown')}",
    ]
    if violations:
        explanation_parts.append(f"Violations: {'; '.join(violations[:3])}.")
    if identity_confidence is not None:
        explanation_parts.append(f"Evidence score: {identity_confidence:.4f}.")
    else:
        explanation_parts.append("No cross-camera identity evidence available (singleton or no matches).")

    return InferenceTrace(
        trace_id=tid,
        entity_type="cluster",
        entity_id=identity_id,
        stage="clustering",
        evidence_items=evidence_items,
        decision=decision,
        decision_score=score,
        explanation=" ".join(explanation_parts),
        generated_at=time.time(),
    )


def build_trajectory_segment_trace(
    segment: Any,
    trace_id: Optional[str] = None,
) -> InferenceTrace:
    """
    Construct an InferenceTrace from a TrajectorySegment object or dict.

    Args:
        segment: TrajectorySegment (or dict with same keys).
        trace_id: Optional explicit trace ID.

    Returns:
        InferenceTrace for the trajectory segment decision.
    """
    def _g(obj: Any, key: str, default: Any = None) -> Any:
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    seg_id = str(_g(segment, "segment_id", "unknown_seg"))
    tid = trace_id or f"trace_seg_{seg_id}"

    status = str(_g(segment, "status", "unknown"))
    confidence = float(_g(segment, "confidence", 0.0))
    ambiguous = bool(_g(segment, "is_ambiguous", False))
    is_gap = bool(_g(segment, "is_gap", False))

    if status == "success" and not ambiguous:
        decision = "feasible"
    elif status == "success" and ambiguous:
        decision = "ambiguous"
    elif status in ("no_path", "unassociated_camera", "infeasible"):
        decision = "infeasible"
    else:
        decision = "unverified"

    t_ev = _g(segment, "temporal_evidence", {}) or {}
    evidence_items = [
        {
            "signal_type": "temporal",
            "status": "feasible" if t_ev.get("comparable") else "unavailable",
            "value": _g(segment, "time_difference_seconds"),
            "contribution": None,
            "delta_t_seconds": _g(segment, "time_difference_seconds"),
        },
        {
            "signal_type": "road_connectivity",
            "status": "feasible" if _g(segment, "candidate_routes") else "unavailable",
            "value": len(_g(segment, "candidate_routes", []) or []),
            "contribution": None,
        },
    ]

    explanation = str(_g(segment, "ambiguity_reason", "") or _g(segment, "status", ""))

    return InferenceTrace(
        trace_id=tid,
        entity_type="trajectory_segment",
        entity_id=seg_id,
        stage="trajectory",
        evidence_items=evidence_items,
        decision=decision,
        decision_score=confidence,
        explanation=explanation or f"Segment {seg_id}: {decision} (confidence={confidence:.4f}).",
        generated_at=time.time(),
    )


def build_sparse_gap_trace(
    gap: Any,
    trace_id: Optional[str] = None,
) -> InferenceTrace:
    """
    Construct an InferenceTrace from a SparseObservationGap object or dict.

    Args:
        gap: SparseObservationGap (or dict with same keys).
        trace_id: Optional explicit trace ID.

    Returns:
        InferenceTrace for the sparse gap inference decision.
    """
    def _g(obj: Any, key: str, default: Any = None) -> Any:
        if hasattr(obj, key):
            return getattr(obj, key)
        if isinstance(obj, dict):
            return obj.get(key, default)
        return default

    gap_id = str(_g(gap, "gap_id", "unknown_gap"))
    tid = trace_id or f"trace_gap_{gap_id}"

    status = str(_g(gap, "status", "unknown"))
    confidence = float(_g(gap, "confidence", 0.0))
    gap_state = str(_g(gap, "gap_state", "unknown"))
    ambiguous = bool(_g(gap, "is_ambiguous", False))
    candidate_routes = _g(gap, "candidate_routes", []) or []
    feasible_routes = [r for r in candidate_routes if getattr(r, "feasible", False)]

    if status == "success" and feasible_routes:
        decision = "feasible"
    elif status == "no_path":
        decision = "infeasible"
    elif status in ("temporal_inversion", "infeasible"):
        decision = "rejected"
    elif status == "unassociated_camera":
        decision = "insufficient_evidence"
    else:
        decision = "unverified"

    meta = _g(gap, "metadata", {}) or {}
    delta_t = meta.get("delta_t_seconds")
    t_ev = meta.get("temporal_evidence", {}) or {}

    evidence_items = [
        {
            "signal_type": "temporal",
            "status": "feasible" if t_ev.get("comparable") else ("unavailable" if status != "temporal_inversion" else "impossible"),
            "value": delta_t,
            "contribution": None,
            "delta_t_seconds": delta_t,
        },
        {
            "signal_type": "road_routes",
            "status": "available" if candidate_routes else "unavailable",
            "value": len(candidate_routes),
            "contribution": None,
            "feasible_route_count": len(feasible_routes),
        },
    ]

    reason = str(_g(gap, "ambiguity_reason", "") or "")
    explanation = reason or (
        f"Sparse gap {gap_id}: {decision}. Gap state={gap_state}. "
        f"Candidate routes: {len(candidate_routes)}, feasible: {len(feasible_routes)}. "
        f"Confidence: {confidence:.4f}."
    )

    return InferenceTrace(
        trace_id=tid,
        entity_type="sparse_gap",
        entity_id=gap_id,
        stage="sparse_inference",
        evidence_items=evidence_items,
        decision=decision,
        decision_score=confidence,
        explanation=explanation,
        generated_at=time.time(),
    )
