"""
Adapter Layer: Day 3 Trajectory Output to Member 3 NormalizedTrajectory Contract.

Converts Day 3 internal trajectory representations (VehicleTrajectory, TrajectorySegment)
into the standardized NormalizedTrajectory model expected by Member 3's mobility graph,
traffic analytics, OD flow estimation, and bottleneck detection engines.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from schemas.normalized_trajectory_schema import (
    NormalizedCandidateRoute,
    NormalizedTrajectory,
)
from schemas.trajectory_schema import TrajectorySegment, VehicleTrajectory


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
        }

        norm_candidate_routes.append(
            NormalizedCandidateRoute(
                nodes=nodes,
                probability=round(prob, 4),
                metadata=meta,
            )
        )

    # Adjust rounding discrepancy on highest probability route
    prob_sum = sum(cr.probability for cr in norm_candidate_routes)
    diff = round(1.0 - prob_sum, 4)
    if abs(diff) > 1e-6 and norm_candidate_routes:
        norm_candidate_routes[0].probability = round(norm_candidate_routes[0].probability + diff, 4)

    metadata = {
        "confidence": segment.confidence,
        "is_ambiguous": segment.is_ambiguous,
        "ambiguity_reason": segment.ambiguity_reason,
        "status": segment.status,
    }
    if vehicle_class:
        metadata["vehicle_class"] = vehicle_class

    return NormalizedTrajectory(
        track_id=segment.identity_id,
        origin_node=origin,
        destination_node=destination,
        candidate_routes=norm_candidate_routes,
        vehicle_weight=float(vehicle_weight),
        timestamp=segment.start_timestamp,
        time_window_start=segment.start_timestamp,
        time_window_end=segment.end_timestamp,
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
        }

        norm_candidate_routes.append(
            NormalizedCandidateRoute(
                nodes=nodes,
                probability=round(prob, 4),
                metadata=meta,
            )
        )

    # Adjust rounding discrepancy on highest probability route
    prob_sum = sum(cr.probability for cr in norm_candidate_routes)
    diff = round(1.0 - prob_sum, 4)
    if abs(diff) > 1e-6 and norm_candidate_routes:
        norm_candidate_routes[0].probability = round(norm_candidate_routes[0].probability + diff, 4)

    metadata = {
        "observations_count": trajectory.observations_count,
        "cameras_visited": list(trajectory.cameras_visited),
        "overall_confidence": trajectory.overall_confidence,
        "is_ambiguous": trajectory.is_ambiguous,
        "total_distance_m": trajectory.total_distance_meters,
    }
    if vehicle_class:
        metadata["vehicle_class"] = vehicle_class

    return NormalizedTrajectory(
        track_id=trajectory.identity_id,
        origin_node=origin,
        destination_node=destination,
        candidate_routes=norm_candidate_routes,
        vehicle_weight=float(vehicle_weight),
        timestamp=trajectory.start_timestamp,
        time_window_start=trajectory.start_timestamp,
        time_window_end=trajectory.end_timestamp,
        metadata=metadata,
    )


def adapt_trajectories_to_batch_payload(
    trajectories: List[Union[VehicleTrajectory, TrajectorySegment]],
    default_weight: float = 1.0,
) -> Dict[str, Any]:
    """
    Convert a batch of Day 3 trajectory objects into a standardized JSON payload
    compatible with Member 3's flow aggregation adapter (MockTrajectoryAdapter).

    Args:
        trajectories: List of VehicleTrajectory or TrajectorySegment instances.
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

    return {
        "source": "UrbanTrackAI_Day3_TrajectoryReconstruction",
        "description": "Probabilistically reconstructed vehicle trajectories from Member 2 mobility inference.",
        "trajectories_count": len(normalized_list),
        "trajectories": [t.to_dict() for t in normalized_list],
    }
