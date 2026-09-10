"""
Adapter Layer: Day 3 Trajectory Output to Member 3 NormalizedTrajectory Contract.

Converts Day 3 internal trajectory representations (VehicleTrajectory, TrajectorySegment)
into the standardized NormalizedTrajectory model expected by Member 3's mobility graph,
traffic analytics, OD flow estimation, and bottleneck detection engines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from schemas.gap_schema import SparseObservationGap
from schemas.normalized_trajectory_schema import (
    NormalizedCandidateRoute,
    NormalizedTrajectory,
)
from schemas.trajectory_schema import TrajectorySegment, VehicleTrajectory


def _format_metric(val: Any) -> Any:
    if isinstance(val, (int, float)):
        return round(float(val), 4)
    if isinstance(val, dict):
        return dict(val)
    return val


def adapt_trajectory_segment_to_normalized(
    segment: TrajectorySegment,
    vehicle_weight: float = 1.0,
    vehicle_class: Optional[str] = None,
) -> NormalizedTrajectory:
    """
    Convert a pairwise Day 3 TrajectorySegment into Member 3's NormalizedTrajectory.

    Args:
        segment: Day 3 TrajectorySegment instance.
        vehicle_weight: Demand weighting / vehicle count (default 1.0).
        vehicle_class: Optional vehicle classification (e.g. 'car', 'bus', 'truck').

    Returns:
        NormalizedTrajectory: Standardized model conforming to Member 3 contract.
    """
    if segment.start_node_id is None or segment.end_node_id is None:
        raise ValueError(
            f"Segment '{segment.segment_id}' cannot be adapted to NormalizedTrajectory: "
            f"Unassociated start or end camera ({segment.ambiguity_reason or segment.status})."
        )

    origin = segment.start_node_id
    destination = segment.end_node_id

    # Extract feasible candidate routes
    feasible_routes = [r for r in segment.candidate_routes if r.feasible]
    if not feasible_routes:
        raise ValueError(
            f"Segment '{segment.segment_id}' has no feasible candidate routes to adapt."
        )

    # Normalize probabilities to ensure exact sum == 1.0
    total_prob = sum(r.estimated_likelihood for r in feasible_routes)
    norm_candidate_routes: List[NormalizedCandidateRoute] = []

    for r in feasible_routes:
        prob = (r.estimated_likelihood / total_prob) if total_prob > 0 else (1.0 / len(feasible_routes))

        # Node sequence validation: must have at least 2 nodes
        nodes = list(r.nodes) if r.nodes else [origin, destination]
        if len(nodes) < 2:
            if origin != destination:
                nodes = [origin, destination]
            else:
                nodes = [origin, origin]

        # Verify endpoints match origin/destination
        if nodes[0] != origin:
            nodes.insert(0, origin)
        if nodes[-1] != destination:
            nodes.append(destination)

        meta = {
            "route_id": r.route_id,
            "edges": list(r.edges),
            "distance_m": r.distance_meters,
            "required_speed_kmh": r.required_speed_kmh,
            "speed_limit_kmh": r.speed_limit_kmh,
            "estimated_travel_time_s": r.estimated_travel_time_seconds,
            "explanation": r.explanation,
            "unobserved_intermediate_nodes": list(nodes[1:-1]) if len(nodes) > 2 else [],
        }
        if getattr(r, "reliability", None) is not None:
            meta["reliability"] = _format_metric(r.reliability)
        if getattr(r, "uncertainty", None) is not None:
            meta["uncertainty"] = _format_metric(r.uncertainty)
        if isinstance(getattr(r, "metadata", None), dict):
            for sem_k in ("probability_semantics", "reliability_semantics", "uncertainty_semantics"):
                if sem_k in r.metadata:
                    meta[sem_k] = r.metadata[sem_k]

        norm_candidate_routes.append(
            NormalizedCandidateRoute(
                nodes=nodes,
                probability=round(prob, 6),
                metadata=meta,
            )
        )

    # Adjust rounding discrepancy on highest probability route
    prob_sum = sum(cr.probability for cr in norm_candidate_routes)
    diff = round(1.0 - prob_sum, 6)
    if diff != 0.0 and norm_candidate_routes:
        norm_candidate_routes[0].probability = round(norm_candidate_routes[0].probability + diff, 6)

    metadata = {
        "confidence": segment.confidence,
        "is_ambiguous": segment.is_ambiguous,
        "ambiguity_reason": segment.ambiguity_reason,
        "status": segment.status,
    }
    if getattr(segment, "reliability", None) is not None:
        metadata["reliability"] = _format_metric(segment.reliability)
    if getattr(segment, "uncertainty", None) is not None:
        metadata["uncertainty"] = _format_metric(segment.uncertainty)
    if getattr(segment, "is_gap", False):
        metadata["gap_detected"] = True
        metadata["gap_duration_seconds"] = segment.gap_duration_seconds
        metadata["observed_endpoints"] = [segment.start_camera_id, segment.end_camera_id]
        metadata["inferred_segment"] = True
        metadata["inference_reason"] = segment.evidence_explanation
        metadata["gap_state"] = getattr(segment, "gap_state", None) or "observed_endpoints_with_unobserved_interval"
        metadata["observations_used"] = [segment.start_observation_id, segment.end_observation_id]
        metadata["unobserved_intermediate_nodes"] = list(segment.unobserved_intermediate_nodes)
        metadata["most_likely_route_intermediate_nodes"] = list(segment.most_likely_intermediate_nodes)
        metadata["common_intermediate_nodes"] = list(segment.common_intermediate_nodes)
        metadata["intermediate_node_semantics"] = "union_of_feasible_candidate_routes"
    if vehicle_class:
        metadata["vehicle_class"] = vehicle_class

    if segment.time_difference_seconds is None:
        t_win_start = None
        t_win_end = None
        metadata["observation_time_a"] = segment.start_timestamp
        metadata["observation_time_b"] = segment.end_timestamp
        if segment.temporal_evidence is not None:
            metadata["temporal_evidence"] = segment.temporal_evidence
    else:
        t_win_start = segment.start_timestamp
        t_win_end = segment.end_timestamp

    return NormalizedTrajectory(
        track_id=segment.identity_id,
        origin_node=origin,
        destination_node=destination,
        candidate_routes=norm_candidate_routes,
        vehicle_weight=float(vehicle_weight),
        timestamp=segment.start_timestamp,
        time_window_start=t_win_start,
        time_window_end=t_win_end,
        metadata=metadata,
    )


def adapt_vehicle_trajectory_to_normalized(
    trajectory: VehicleTrajectory,
    vehicle_weight: float = 1.0,
    vehicle_class: Optional[str] = None,
) -> NormalizedTrajectory:
    """
    Convert a complete multi-segment VehicleTrajectory into Member 3's NormalizedTrajectory.

    For multi-observation vehicle journeys (A -> B -> C -> D), projects the full
    journey from overall origin to destination, preserving alternative corridors and
    their combined path likelihoods.

    Args:
        trajectory: Day 3 VehicleTrajectory instance.
        vehicle_weight: Demand weighting / vehicle count (default 1.0).
        vehicle_class: Optional vehicle classification (e.g. 'car', 'bus', 'truck').

    Returns:
        NormalizedTrajectory: Member 3 integration object.
    """
    if not trajectory.segments:
        raise ValueError(
            f"Trajectory '{trajectory.identity_id}' contains no segments to adapt."
        )

    # Check for unassociated endpoints
    first_seg = trajectory.segments[0]
    last_seg = trajectory.segments[-1]

    if first_seg.start_node_id is None or last_seg.end_node_id is None:
        raise ValueError(
            f"Trajectory '{trajectory.identity_id}' has unassociated origin or destination camera."
        )

    origin = first_seg.start_node_id
    destination = last_seg.end_node_id

    # Handle Single-Segment Trajectory
    if len(trajectory.segments) == 1:
        return adapt_trajectory_segment_to_normalized(
            first_seg,
            vehicle_weight=vehicle_weight,
            vehicle_class=vehicle_class,
        )

    # Multi-Segment Trajectory: Compose Complete Path Corridors
    # Candidate routes across segments are combined by Cartesian concatenation of node paths
    # with joint probability P(path) = Product(P(seg_r)).
    current_corridors: List[Dict[str, Any]] = [
        {"nodes": [origin], "edges": [], "probability": 1.0, "distance_m": 0.0}
    ]

    for seg in trajectory.segments:
        feasible_r = [r for r in seg.candidate_routes if r.feasible]
        if not feasible_r:
            # Fallback to single top candidate if marked infeasible
            feasible_r = seg.candidate_routes[:1] if seg.candidate_routes else []

        if not feasible_r:
            continue

        seg_total_prob = sum(r.estimated_likelihood for r in feasible_r)
        new_corridors: List[Dict[str, Any]] = []

        for curr in current_corridors:
            for r in feasible_r:
                r_prob = (r.estimated_likelihood / seg_total_prob) if seg_total_prob > 0 else (1.0 / len(feasible_r))
                r_nodes = list(r.nodes) if r.nodes else [seg.start_node_id, seg.end_node_id]

                # Merge node lists avoiding duplication at segment junctions
                merged_nodes = list(curr["nodes"])
                if merged_nodes and r_nodes and merged_nodes[-1] == r_nodes[0]:
                    merged_nodes.extend(r_nodes[1:])
                else:
                    merged_nodes.extend(r_nodes)

                new_corridors.append({
                    "nodes": merged_nodes,
                    "edges": curr["edges"] + list(r.edges),
                    "probability": curr["probability"] * r_prob,
                    "distance_m": curr["distance_m"] + r.distance_meters,
                })

        # Cap corridor explosion to top 5 most likely full paths
        new_corridors.sort(key=lambda c: c["probability"], reverse=True)
        current_corridors = new_corridors[:5]

    # Re-normalize corridor probabilities to sum to 1.0
    total_p = sum(c["probability"] for c in current_corridors)
    norm_candidate_routes: List[NormalizedCandidateRoute] = []

    for c in current_corridors:
        prob = (c["probability"] / total_p) if total_p > 0 else (1.0 / len(current_corridors))
        nodes = list(c["nodes"])

        if len(nodes) < 2:
            nodes = [origin, destination]

        if nodes[0] != origin:
            nodes.insert(0, origin)
        if nodes[-1] != destination:
            nodes.append(destination)

        meta = {
            "edges": c["edges"],
            "distance_m": round(c["distance_m"], 1),
            "unobserved_intermediate_nodes": list(nodes[1:-1]) if len(nodes) > 2 else [],
        }

        norm_candidate_routes.append(
            NormalizedCandidateRoute(
                nodes=nodes,
                probability=round(prob, 6),
                metadata=meta,
            )
        )

    # Adjust rounding discrepancy on highest probability route
    prob_sum = sum(cr.probability for cr in norm_candidate_routes)
    diff = round(1.0 - prob_sum, 6)
    if diff != 0.0 and norm_candidate_routes:
        norm_candidate_routes[0].probability = round(norm_candidate_routes[0].probability + diff, 6)

    metadata = {
        "observations_count": trajectory.observations_count,
        "cameras_visited": list(trajectory.cameras_visited),
        "overall_confidence": trajectory.overall_confidence,
        "is_ambiguous": trajectory.is_ambiguous,
        "total_distance_m": trajectory.total_distance_meters,
        "gaps_count": getattr(trajectory, "gaps_count", 0),
        "gap_detected": getattr(trajectory, "gaps_count", 0) > 0,
    }
    if getattr(trajectory, "reliability", None) is not None:
        metadata["reliability"] = _format_metric(trajectory.reliability)
    if getattr(trajectory, "uncertainty", None) is not None:
        metadata["uncertainty"] = _format_metric(trajectory.uncertainty)
    if getattr(trajectory, "gaps_count", 0) > 0:
        metadata["inferred_segment"] = True
        metadata["gap_state"] = "sparse_observations"
    if vehicle_class:
        metadata["vehicle_class"] = vehicle_class

    if trajectory.total_time_seconds is None or any(s.time_difference_seconds is None for s in trajectory.segments):
        t_win_start = None
        t_win_end = None
        metadata["start_timestamp"] = trajectory.start_timestamp
        metadata["end_timestamp"] = trajectory.end_timestamp
    else:
        t_win_start = trajectory.start_timestamp
        t_win_end = trajectory.end_timestamp

    return NormalizedTrajectory(
        track_id=trajectory.identity_id,
        origin_node=origin,
        destination_node=destination,
        candidate_routes=norm_candidate_routes,
        vehicle_weight=float(vehicle_weight),
        timestamp=trajectory.start_timestamp,
        time_window_start=t_win_start,
        time_window_end=t_win_end,
        metadata=metadata,
    )


def adapt_sparse_gap_to_normalized(
    gap: SparseObservationGap,
    vehicle_weight: float = 1.0,
    vehicle_class: Optional[str] = None,
) -> NormalizedTrajectory:
    """
    Convert a Day 4 SparseObservationGap into Member 3's NormalizedTrajectory contract.

    Args:
        gap: Day 4 SparseObservationGap instance.
        vehicle_weight: Demand weighting / vehicle count (default 1.0).
        vehicle_class: Optional vehicle classification (e.g. 'car', 'bus', 'truck').

    Returns:
        NormalizedTrajectory: Standardized model conforming to Member 3 contract with gap metadata.
    """
    if gap.start_node_id is None or gap.end_node_id is None:
        raise ValueError(
            f"Gap '{gap.gap_id}' cannot be adapted to NormalizedTrajectory: "
            f"Unassociated start or end camera ({gap.ambiguity_reason or gap.status})."
        )

    origin = gap.start_node_id
    destination = gap.end_node_id

    feasible_routes = [r for r in gap.candidate_routes if r.feasible]
    if not feasible_routes:
        raise ValueError(
            f"Gap '{gap.gap_id}' has no feasible candidate routes to adapt."
        )

    total_prob = sum(r.estimated_likelihood for r in feasible_routes)
    norm_candidate_routes: List[NormalizedCandidateRoute] = []

    for r in feasible_routes:
        prob = (r.estimated_likelihood / total_prob) if total_prob > 0 else (1.0 / len(feasible_routes))
        nodes = list(r.nodes) if r.nodes else [origin, destination]
        if len(nodes) < 2:
            nodes = [origin, destination] if origin != destination else [origin, origin]
        if nodes[0] != origin:
            nodes.insert(0, origin)
        if nodes[-1] != destination:
            nodes.append(destination)

        meta = {
            "route_id": r.route_id,
            "edges": list(r.edges),
            "distance_m": r.distance_meters,
            "required_speed_kmh": r.required_speed_kmh,
            "speed_limit_kmh": r.speed_limit_kmh,
            "estimated_travel_time_s": r.estimated_travel_time_seconds,
            "explanation": r.explanation,
            "unobserved_intermediate_nodes": list(nodes[1:-1]) if len(nodes) > 2 else [],
        }
        rel = getattr(r, "reliability", None)
        if rel is None and isinstance(getattr(r, "metadata", None), dict):
            rel = r.metadata.get("reliability")
        if rel is not None:
            meta["reliability"] = _format_metric(rel)

        unc = getattr(r, "uncertainty", None)
        if unc is None and isinstance(getattr(r, "metadata", None), dict):
            unc = r.metadata.get("uncertainty")
        if unc is not None:
            meta["uncertainty"] = _format_metric(unc)

        if isinstance(getattr(r, "metadata", None), dict):
            for sem_k in ("probability_semantics", "reliability_semantics", "uncertainty_semantics"):
                if sem_k in r.metadata:
                    meta[sem_k] = r.metadata[sem_k]

        norm_candidate_routes.append(
            NormalizedCandidateRoute(
                nodes=nodes,
                probability=round(prob, 6),
                metadata=meta,
            )
        )

    prob_sum = sum(cr.probability for cr in norm_candidate_routes)
    diff = round(1.0 - prob_sum, 6)
    if diff != 0.0 and norm_candidate_routes:
        norm_candidate_routes[0].probability = round(norm_candidate_routes[0].probability + diff, 6)

    metadata = {
        "confidence": gap.confidence,
        "is_ambiguous": gap.is_ambiguous,
        "ambiguity_reason": gap.ambiguity_reason,
        "status": gap.status,
        "gap_detected": True,
        "gap_duration_seconds": gap.gap_duration_seconds,
        "observed_endpoints": [
            gap.start_observation.get("camera_id", ""),
            gap.end_observation.get("camera_id", ""),
        ],
        "inferred_segment": True,
        "inference_reason": gap.candidate_routes[0].explanation if gap.candidate_routes else gap.ambiguity_reason,
        "gap_state": gap.gap_state,
        "unobserved_intermediate_nodes": list(gap.unobserved_intermediate_nodes),
        "most_likely_route_intermediate_nodes": list(gap.most_likely_intermediate_nodes),
        "common_intermediate_nodes": list(gap.common_intermediate_nodes),
        "intermediate_node_semantics": "union_of_feasible_candidate_routes",
        "observations_used": [
            gap.start_observation.get("observation_id", ""),
            gap.end_observation.get("observation_id", ""),
        ],
    }
    if getattr(gap, "reliability", None) is not None:
        metadata["reliability"] = _format_metric(gap.reliability)
    if getattr(gap, "uncertainty", None) is not None:
        metadata["uncertainty"] = _format_metric(gap.uncertainty)
    if vehicle_class:
        metadata["vehicle_class"] = vehicle_class

    t_start = gap.start_observation.get("timestamp_seconds")
    t_end = gap.end_observation.get("timestamp_seconds")

    if gap.gap_duration_seconds is None:
        t_win_start = None
        t_win_end = None
        metadata["observation_time_a"] = t_start
        metadata["observation_time_b"] = t_end
        if "temporal_evidence" in gap.metadata:
            metadata["temporal_evidence"] = gap.metadata["temporal_evidence"]
    else:
        t_win_start = t_start
        t_win_end = t_end

    return NormalizedTrajectory(
        track_id=gap.identity_id,
        origin_node=origin,
        destination_node=destination,
        candidate_routes=norm_candidate_routes,
        vehicle_weight=float(vehicle_weight),
        timestamp=t_start,
        time_window_start=t_win_start,
        time_window_end=t_win_end,
        metadata=metadata,
    )


def adapt_trajectories_to_batch_payload(
    trajectories: List[Union[VehicleTrajectory, TrajectorySegment, SparseObservationGap]],
    default_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    Convert a batch of Day 3 / Day 4 trajectory objects into a standardized JSON payload
    compatible with Member 3's flow aggregation adapter (MockTrajectoryAdapter).

    Args:
        trajectories: List of VehicleTrajectory, TrajectorySegment, or SparseObservationGap instances.
        default_weight: Default vehicle demand weight.

    Returns:
        Dict[str, Any]: Payload dictionary with 'trajectories' array.
    """
    normalized_list: List[NormalizedTrajectory] = []

    for t in trajectories:
        if isinstance(t, VehicleTrajectory):
            try:
                norm = adapt_vehicle_trajectory_to_normalized(t, vehicle_weight=default_weight)
                normalized_list.append(norm)
            except ValueError:
                continue
        elif isinstance(t, TrajectorySegment):
            try:
                norm = adapt_trajectory_segment_to_normalized(t, vehicle_weight=default_weight)
                normalized_list.append(norm)
            except ValueError:
                continue
        elif isinstance(t, SparseObservationGap):
            try:
                norm = adapt_sparse_gap_to_normalized(t, vehicle_weight=default_weight)
                normalized_list.append(norm)
            except ValueError:
                continue

    return {
        "source": "UrbanTrackAI_Day3_TrajectoryReconstruction",
        "description": "Probabilistically reconstructed vehicle trajectories from Member 2 mobility inference.",
        "trajectories_count": len(normalized_list),
        "trajectories": [t.to_dict() for t in normalized_list],
    }
