"""
Mobility Flow Aggregation Engine for UrbanTrack AI (Day 6).

Converts inferred vehicle trajectories into road-level flow, capacity utilization,
and Origin-Destination (OD) demand matrices.

Critical Semantics Preserved & Audited:
1. vehicle_weight: Physical traffic demand (PCU multiplier).
2. route_probability: Relative estimated likelihood of candidate route hypothesis (never overwritten).
3. flow_allocation_probability: Normalized allocation among feasible/open routes under network state.
4. route_demand: Expected vehicle demand = vehicle_weight * flow_allocation_probability.
5. Demand conservation invariant:
   - For unconstrained trajectories: sum(route_demand) == vehicle_weight.
   - When network constraints filter closed routes: sum(route_demand across open routes) == vehicle_weight.
   - When all candidate routes are closed: allocated demand is 0.0 and reported explicitly as an infeasible state.
6. Reliability & Uncertainty: Preserved as evidence quality indicators, NEVER multiplied into physical demand.
7. OD Demand: Accumulates vehicle_weight directly per (origin, destination), independent of route probabilities.
8. Time Units & Hourly Rate:
   - expected_demand_in_window: Dimensionless weighted demand in observation window.
   - expected_demand_vph: Converted to hourly rate using synthetic elapsed seconds duration (end - start).
   - utilization_ratio: expected_demand_vph / capacity_vph (only computed when duration > 0).
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from inference.road_graph import RoadGraph, RoadEdge
from schemas.normalized_trajectory_schema import NormalizedTrajectory, NormalizedCandidateRoute
from schemas.mobility_schema import (
    FlowAllocatedRoute,
    ODMatrix,
    ODPairDemand,
    RoadFlowMetric,
    RoadStatus,
    ValidationIssue,
)


class MobilityFlowEngine:
    """
    Aggregates NormalizedTrajectory instances onto a physical RoadGraph to compute
    road-level demand, dimensional utilization ratios, and OD demand matrices.
    """

    def __init__(
        self,
        road_graph: RoadGraph,
        probability_tolerance: float = 1e-4,
        allow_normalization: bool = True,
    ) -> None:
        """
        Initialize the flow engine with the shared city road graph.

        Args:
            road_graph: The RoadGraph instance (e.g. loaded from city_network.json).
            probability_tolerance: Allowable tolerance for probability sums deviating from 1.0.
            allow_normalization: Whether near-1.0 probability sums are normalized for flow allocation.
        """
        if not isinstance(road_graph, RoadGraph):
            raise TypeError(f"Expected RoadGraph, got: {type(road_graph)}")
        self.graph = road_graph
        self.probability_tolerance = probability_tolerance
        self.allow_normalization = allow_normalization

    def validate_trajectory(
        self, trajectory: NormalizedTrajectory
    ) -> Tuple[bool, List[ValidationIssue], List[FlowAllocatedRoute]]:
        """Backward-compatible validation wrapper around validate_and_allocate_routes."""
        return self.validate_and_allocate_routes(trajectory)

    def map_route_to_roads(self, route: NormalizedCandidateRoute) -> Tuple[List[str], List[str]]:
        """
        Map candidate route nodes to directed road segment IDs.

        Returns:
            Tuple of (valid_road_ids, errors)
        """
        meta_edges = route.metadata.get("edges")
        expected_edge_count = len(route.nodes) - 1

        if meta_edges and isinstance(meta_edges, list) and len(meta_edges) == expected_edge_count:
            road_ids = []
            for rid in meta_edges:
                if rid not in self.graph.edges:
                    return [], [f"Edge ID '{rid}' in route metadata not found in road graph."]
                road_ids.append(rid)
            return road_ids, []

        derived_road_ids: List[str] = []
        for u, v in zip(route.nodes[:-1], route.nodes[1:]):
            edge = self.graph.get_edge_by_nodes(u, v)
            if edge is None:
                return [], [f"No directed road segment connects '{u}' -> '{v}'."]
            derived_road_ids.append(edge.road_id)

        return derived_road_ids, []

    def validate_and_allocate_routes(
        self, trajectory: NormalizedTrajectory
    ) -> Tuple[bool, List[ValidationIssue], List[FlowAllocatedRoute]]:
        """
        Validate trajectory structure and allocate flow probabilities among open routes.

        Semantic Rules:
        1. Original candidate route probabilities ('original_route_probability') are preserved intact.
        2. Routes traversing closed roads ('is_closed=True') are excluded from flow allocation.
        3. If some routes are closed while others remain feasible, the feasible routes are
           re-normalized for flow allocation as 'flow_allocation_probability' and flagged
           with 'network_constraint_applied=True'.
        4. If all routes are closed, flow allocation fails gracefully with 0.0 valid demand,
           reporting 'all_candidate_routes_closed'.

        Returns:
            Tuple of (is_valid, validation_issues, flow_allocated_routes)
        """
        issues: List[ValidationIssue] = []
        tid = getattr(trajectory, "track_id", "UNKNOWN")

        # 1. Validate track_id
        if not tid or not isinstance(tid, str) or not tid.strip():
            issues.append(ValidationIssue(
                track_id=str(tid),
                issue_type="invalid_track_id",
                message="Trajectory track_id must be a non-empty string.",
            ))
            return False, issues, []

        # 2. Validate vehicle_weight
        weight = getattr(trajectory, "vehicle_weight", None)
        if weight is None or not isinstance(weight, (int, float)) or math.isnan(weight) or math.isinf(weight) or weight < 0:
            issues.append(ValidationIssue(
                track_id=tid,
                issue_type="invalid_vehicle_weight",
                message=f"Trajectory '{tid}' vehicle_weight must be a finite non-negative float, got: {weight}",
            ))
            return False, issues, []

        # 3. Validate origin and destination in graph
        origin = getattr(trajectory, "origin_node", "")
        dest = getattr(trajectory, "destination_node", "")
        if origin not in self.graph.nodes:
            issues.append(ValidationIssue(
                track_id=tid,
                issue_type="unknown_origin_node",
                message=f"Trajectory '{tid}' origin '{origin}' not found in RoadGraph.",
            ))
            return False, issues, []

        if dest not in self.graph.nodes:
            issues.append(ValidationIssue(
                track_id=tid,
                issue_type="unknown_destination_node",
                message=f"Trajectory '{tid}' destination '{dest}' not found in RoadGraph.",
            ))
            return False, issues, []

        # 4. Validate candidate routes existence
        candidate_routes = getattr(trajectory, "candidate_routes", None)
        if not candidate_routes or not isinstance(candidate_routes, list) or len(candidate_routes) == 0:
            issues.append(ValidationIssue(
                track_id=tid,
                issue_type="empty_candidate_routes",
                message=f"Trajectory '{tid}' must contain at least one candidate route.",
            ))
            return False, issues, []

        # 5. Process candidate routes
        valid_open_routes: List[Tuple[NormalizedCandidateRoute, List[str], float]] = []
        has_closed_route = False

        for idx, route in enumerate(candidate_routes):
            if not isinstance(route, NormalizedCandidateRoute):
                if isinstance(route, dict):
                    try:
                        route = NormalizedCandidateRoute.from_dict(route)
                    except Exception as e:
                        issues.append(ValidationIssue(
                            track_id=tid,
                            issue_type="malformed_candidate_route",
                            message=f"Trajectory '{tid}' route #{idx} could not be parsed: {e}",
                        ))
                        continue
                else:
                    issues.append(ValidationIssue(
                        track_id=tid,
                        issue_type="invalid_route_type",
                        message=f"Trajectory '{tid}' route #{idx} is not a NormalizedCandidateRoute.",
                    ))
                    continue

            # Validate endpoints
            if len(route.nodes) < 2:
                issues.append(ValidationIssue(
                    track_id=tid,
                    issue_type="route_too_short",
                    message=f"Trajectory '{tid}' route #{idx} has fewer than 2 nodes: {route.nodes}",
                ))
                continue

            if route.nodes[0] != origin:
                issues.append(ValidationIssue(
                    track_id=tid,
                    issue_type="route_origin_mismatch",
                    message=f"Trajectory '{tid}' route #{idx} starts at '{route.nodes[0]}', expected origin '{origin}'.",
                ))
                continue

            if route.nodes[-1] != dest:
                issues.append(ValidationIssue(
                    track_id=tid,
                    issue_type="route_destination_mismatch",
                    message=f"Trajectory '{tid}' route #{idx} ends at '{route.nodes[-1]}', expected destination '{dest}'.",
                ))
                continue

            # Validate probability value range
            if not isinstance(route.probability, (int, float)) or math.isnan(route.probability) or route.probability < 0.0 or route.probability > 1.0:
                issues.append(ValidationIssue(
                    track_id=tid,
                    issue_type="invalid_route_probability",
                    message=f"Trajectory '{tid}' route #{idx} probability {route.probability} outside [0.0, 1.0].",
                ))
                continue

            orig_prob = float(route.probability)

            # Map route to road edges
            road_ids, edge_errors = self.map_route_to_roads(route)
            if edge_errors:
                for err in edge_errors:
                    issues.append(ValidationIssue(
                        track_id=tid,
                        issue_type="disconnected_route",
                        message=f"Trajectory '{tid}' route #{idx}: {err}",
                    ))
                continue

            # Check for closed roads along the route
            is_route_closed = False
            for rid in road_ids:
                edge = self.graph.edges.get(rid)
                if edge is not None and edge.is_closed:
                    is_route_closed = True
                    has_closed_route = True
                    issues.append(ValidationIssue(
                        track_id=tid,
                        issue_type="closed_road_traversed",
                        message=f"Trajectory '{tid}' route #{idx} traverses closed road '{rid}'; excluded from flow allocation.",
                    ))
                    break

            if is_route_closed:
                continue

            # Preserve road_ids in metadata if not present
            if "edges" not in route.metadata:
                route.metadata["edges"] = road_ids

            valid_open_routes.append((route, road_ids, orig_prob))

        # Check if all candidate routes were closed
        if not valid_open_routes:
            issues.append(ValidationIssue(
                track_id=tid,
                issue_type="all_candidate_routes_closed" if has_closed_route else "no_feasible_routes",
                message=f"Trajectory '{tid}' has no open, feasible routes on the network; zero flow allocated.",
            ))
            return False, issues, []

        # 6. Compute Flow Allocation Probabilities
        open_prob_sum = sum(prob for _, _, prob in valid_open_routes)
        network_constraint_applied = has_closed_route

        allocated_routes: List[FlowAllocatedRoute] = []

        if network_constraint_applied:
            # Re-normalize flow allocation probabilities among remaining open routes
            if open_prob_sum > 0.0:
                for r, rids, orig_p in valid_open_routes:
                    flow_p = orig_p / open_prob_sum
                    allocated_routes.append(FlowAllocatedRoute(
                        route=r,
                        road_ids=rids,
                        original_route_probability=orig_p,
                        flow_allocation_probability=flow_p,
                        network_constraint_applied=True,
                    ))
                issues.append(ValidationIssue(
                    track_id=tid,
                    issue_type="flow_probabilities_renormalized_for_closed_roads",
                    message=(
                        f"Trajectory '{tid}' had closed routes filtered; flow allocation probabilities "
                        f"re-normalized from open sum {open_prob_sum:.4f} to 1.0 (original probabilities retained)."
                    ),
                    recovered=True,
                ))
            else:
                issues.append(ValidationIssue(
                    track_id=tid,
                    issue_type="zero_open_probability",
                    message=f"Trajectory '{tid}' open routes have zero cumulative probability.",
                ))
                return False, issues, []
        else:
            # Normal route probability validation
            if abs(open_prob_sum - 1.0) > self.probability_tolerance:
                if self.allow_normalization and 0.01 <= open_prob_sum <= 2.0:
                    for r, rids, orig_p in valid_open_routes:
                        flow_p = orig_p / open_prob_sum
                        allocated_routes.append(FlowAllocatedRoute(
                            route=r,
                            road_ids=rids,
                            original_route_probability=orig_p,
                            flow_allocation_probability=flow_p,
                            network_constraint_applied=False,
                        ))
                    issues.append(ValidationIssue(
                        track_id=tid,
                        issue_type="probability_normalized",
                        message=f"Trajectory '{tid}' route probabilities summed to {open_prob_sum:.6f}; normalized to 1.0.",
                        recovered=True,
                    ))
                else:
                    issues.append(ValidationIssue(
                        track_id=tid,
                        issue_type="invalid_probability_sum",
                        message=f"Trajectory '{tid}' candidate route probabilities sum to {open_prob_sum:.6f} != 1.0.",
                    ))
                    return False, issues, []
            else:
                # Perfectly normalized candidate routes
                for r, rids, orig_p in valid_open_routes:
                    allocated_routes.append(FlowAllocatedRoute(
                        route=r,
                        road_ids=rids,
                        original_route_probability=orig_p,
                        flow_allocation_probability=orig_p,
                        network_constraint_applied=False,
                    ))

        return True, issues, allocated_routes

    def aggregate_flows(
        self,
        trajectories: Iterable[NormalizedTrajectory],
        time_window: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[RoadFlowMetric], ODMatrix, List[ValidationIssue], List[Dict[str, Any]]]:
        """
        Aggregate expected road flow and OD demand across a batch of trajectories.

        Audit Invariants:
        1. expected_demand_in_window represents weighted vehicle demand inside observation window.
        2. expected_demand_vph represents hourly rate only when duration_seconds > 0.
        3. utilization_ratio = expected_demand_vph / capacity_vph.
        4. OD demand accumulates total physical vehicle_weight directly without multiplying by route probability.
        """
        all_issues: List[ValidationIssue] = []
        conservation_records: List[Dict[str, Any]] = []

        road_expected_demands: Dict[str, float] = defaultdict(float)
        road_contributing_trajectories: Dict[str, Set[str]] = defaultdict(set)
        road_reliabilities: Dict[str, List[float]] = defaultdict(list)
        road_uncertainties: Dict[str, List[float]] = defaultdict(list)

        od_demands: Dict[Tuple[str, str], float] = defaultdict(float)
        od_traj_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        traj_list = list(trajectories)

        # ---------------------------------------------------------------------
        # TIME WINDOW AUDIT: Parse synthetic elapsed seconds
        # ---------------------------------------------------------------------
        window_start = time_window.get("start") if time_window else None
        window_end = time_window.get("end") if time_window else None

        if traj_list and window_start is None and window_end is None:
            starts = [
                float(t.time_window_start) for t in traj_list
                if t.time_window_start is not None and isinstance(t.time_window_start, (int, float))
            ]
            ends = [
                float(t.time_window_end) for t in traj_list
                if t.time_window_end is not None and isinstance(t.time_window_end, (int, float))
            ]
            if starts:
                window_start = min(starts)
            if ends:
                window_end = max(ends)

        # Validate time duration
        duration_seconds: Optional[float] = None
        is_hourly_rate_valid = False

        if window_start is not None and window_end is not None:
            try:
                start_val = float(window_start)
                end_val = float(window_end)
                delta_t = end_val - start_val
                if delta_t > 0.0:
                    duration_seconds = delta_t
                    is_hourly_rate_valid = True
                elif delta_t == 0.0:
                    all_issues.append(ValidationIssue(
                        track_id="BATCH_TIME_WINDOW",
                        issue_type="zero_duration_window",
                        message="Time window duration is 0.0s; cannot calculate hourly demand rate (vph).",
                    ))
                else:
                    all_issues.append(ValidationIssue(
                        track_id="BATCH_TIME_WINDOW",
                        issue_type="negative_duration_window",
                        message=f"Reversed timestamps in time window ({start_val} > {end_val}); cannot calculate hourly rate.",
                    ))
            except (ValueError, TypeError) as e:
                all_issues.append(ValidationIssue(
                    track_id="BATCH_TIME_WINDOW",
                    issue_type="non_numeric_time_window",
                    message=f"Could not parse numerical time window: {e}",
                ))

        for traj in traj_list:
            is_valid, issues, allocated_routes = self.validate_and_allocate_routes(traj)
            all_issues.extend(issues)

            if not is_valid or not allocated_routes:
                conservation_records.append({
                    "track_id": getattr(traj, "track_id", "UNKNOWN"),
                    "vehicle_weight": getattr(traj, "vehicle_weight", 0.0),
                    "sum_route_demand": 0.0,
                    "is_conserved": False,
                    "allocated_routes_count": 0,
                    "status": "infeasible_or_all_routes_closed",
                })
                continue

            weight = float(traj.vehicle_weight)

            # -----------------------------------------------------------------
            # DEMAND CONSERVATION CHECK: sum(route_demand) == vehicle_weight
            # -----------------------------------------------------------------
            route_demands_sum = sum(weight * ar.flow_allocation_probability for ar in allocated_routes)
            conserved = math.isclose(route_demands_sum, weight, rel_tol=1e-5, abs_tol=1e-5)
            conservation_records.append({
                "track_id": traj.track_id,
                "vehicle_weight": weight,
                "sum_route_demand": round(route_demands_sum, 6),
                "is_conserved": conserved,
                "allocated_routes_count": len(allocated_routes),
                "network_constraint_applied": any(ar.network_constraint_applied for ar in allocated_routes),
            })

            # -----------------------------------------------------------------
            # OD DEMAND ACCUMULATION: Add vehicle_weight directly to OD matrix
            # (Invariant: Independent of route probabilities or reliability)
            # -----------------------------------------------------------------
            od_key = (traj.origin_node, traj.destination_node)
            od_demands[od_key] += weight
            od_traj_counts[od_key] += 1

            meta = getattr(traj, "metadata", {}) or {}
            reliability = meta.get("trajectory_reliability") or meta.get("overall_confidence")
            uncertainty = meta.get("trajectory_uncertainty") or meta.get("uncertainty")

            # -----------------------------------------------------------------
            # ROAD FLOW ACCUMULATION:
            # route_demand = vehicle_weight * flow_allocation_probability
            # -----------------------------------------------------------------
            for ar in allocated_routes:
                route_demand = weight * ar.flow_allocation_probability
                if route_demand <= 0.0:
                    continue

                # Deduplicate edges within candidate route hypothesis to model spatial presence
                unique_roads = list(dict.fromkeys(ar.road_ids))

                for rid in unique_roads:
                    road_expected_demands[rid] += route_demand
                    road_contributing_trajectories[rid].add(traj.track_id)
                    if reliability is not None and isinstance(reliability, (int, float)):
                        road_reliabilities[rid].append(float(reliability))
                    if uncertainty is not None and isinstance(uncertainty, (int, float)):
                        road_uncertainties[rid].append(float(uncertainty))

        # ---------------------------------------------------------------------
        # BUILD ROAD FLOW METRICS WITH DIMENSIONAL COMPATIBILITY
        # ---------------------------------------------------------------------
        duration_hours = (duration_seconds / 3600.0) if duration_seconds is not None else None

        road_metrics: List[RoadFlowMetric] = []
        for rid in sorted(self.graph.edges.keys()):
            edge = self.graph.edges[rid]
            demand_in_window = road_expected_demands.get(rid, 0.0)
            
            # Sanitize capacity
            raw_capacity = edge.capacity_vph
            if raw_capacity is not None and isinstance(raw_capacity, (int, float)) and raw_capacity > 0:
                capacity_vph = float(raw_capacity)
            else:
                capacity_vph = 0.0

            # Calculate hourly rate and utilization ratio only if duration is valid
            expected_demand_vph: Optional[float] = None
            utilization_ratio: Optional[float] = None

            if is_hourly_rate_valid and duration_hours is not None:
                expected_demand_vph = demand_in_window / duration_hours
                if capacity_vph > 0.0:
                    utilization_ratio = expected_demand_vph / capacity_vph
                else:
                    utilization_ratio = 0.0
                status = RoadStatus.classify(utilization_ratio)
            else:
                # No valid time window available; do NOT fabricate an hourly rate
                status = RoadStatus.UNCALIBRATED

            contrib_trajs = len(road_contributing_trajectories.get(rid, set()))
            contrib_ids = sorted(list(road_contributing_trajectories.get(rid, set())))

            rel_summary = None
            if road_reliabilities.get(rid):
                rels = road_reliabilities[rid]
                rel_summary = {
                    "mean_reliability": round(sum(rels) / len(rels), 4),
                    "min_reliability": round(min(rels), 4),
                    "max_reliability": round(max(rels), 4),
                    "count": len(rels),
                }

            unc_summary = None
            if road_uncertainties.get(rid):
                uncs = road_uncertainties[rid]
                unc_summary = {
                    "mean_uncertainty": round(sum(uncs) / len(uncs), 4),
                    "min_uncertainty": round(min(uncs), 4),
                    "max_uncertainty": round(max(uncs), 4),
                    "count": len(uncs),
                }

            metric = RoadFlowMetric(
                road_id=rid,
                from_node=edge.from_node,
                to_node=edge.to_node,
                expected_demand_in_window=demand_in_window,
                capacity_vph=capacity_vph,
                expected_demand_vph=expected_demand_vph,
                utilization_ratio=utilization_ratio,
                is_hourly_rate_valid=is_hourly_rate_valid,
                duration_seconds=duration_seconds,
                status=status,
                contributing_trajectories_count=contrib_trajs,
                contributing_identities=contrib_ids,
                reliability_summary=rel_summary,
                uncertainty_summary=unc_summary,
                time_window_start=window_start,
                time_window_end=window_end,
                metadata={
                    "is_closed": edge.is_closed,
                    "distance_km": edge.distance_km,
                    "speed_limit_kmph": edge.speed_limit_kmh,
                },
            )
            road_metrics.append(metric)

        # ---------------------------------------------------------------------
        # BUILD OD MATRIX
        # ---------------------------------------------------------------------
        od_pair_list: List[ODPairDemand] = []
        for (orig, dest), demand in sorted(od_demands.items(), key=lambda x: (x[0][0], x[0][1])):
            cnt = od_traj_counts[(orig, dest)]
            od_pair_list.append(ODPairDemand(
                origin=orig,
                destination=dest,
                demand=demand,
                contributing_trajectories_count=cnt,
                time_window_start=window_start,
                time_window_end=window_end,
            ))
        od_matrix = ODMatrix(pairs=od_pair_list)

        return road_metrics, od_matrix, all_issues, conservation_records
