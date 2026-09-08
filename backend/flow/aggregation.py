"""Expected road flow aggregation engine for UrbanTrackAI.

Consumes normalized probabilistic trajectories and maps them across the directed
MobilityGraph to compute expected traffic volumes per road segment.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from backend.flow.models import (
    CandidateRoute,
    FlowAggregationResult,
    InvalidRouteError,
    NormalizedTrajectory,
    ProbabilityValidationError,
    RoadFlow,
)
from backend.mobility.graph import MobilityGraph


class ExpectedFlowAggregator:
    """Aggregates probabilistic vehicle trajectories into expected road-level traffic flow.

    Attributes:
        graph: The directed MobilityGraph defining the road network topology.
        probability_tolerance: Allowable tolerance when validating route probability sums (default: 1e-6).
    """

    def __init__(self, graph: MobilityGraph, probability_tolerance: float = 1e-6) -> None:
        """Initialize aggregator with a MobilityGraph.

        Args:
            graph: MobilityGraph instance.
            probability_tolerance: Tolerance for candidate route probability sum validation.
        """
        if not isinstance(graph, MobilityGraph):
            raise TypeError(f"Expected MobilityGraph instance, got: {type(graph)}")
        if probability_tolerance < 0:
            raise ValueError(f"probability_tolerance must be non-negative, got: {probability_tolerance}")

        self.graph: MobilityGraph = graph
        self.probability_tolerance: float = probability_tolerance

    def validate_trajectory_routes(self, trajectory: NormalizedTrajectory) -> None:
        """Validate that all candidate routes in a trajectory exist and connect in the graph.

        Args:
            trajectory: NormalizedTrajectory to validate.

        Raises:
            ProbabilityValidationError: If route probabilities do not sum to ~1.0 within tolerance.
            InvalidRouteError: If nodes are missing or edges do not exist in the MobilityGraph.
        """
        # Validate probability sum
        trajectory.validate_probabilities(tolerance=self.probability_tolerance)

        for route_idx, route in enumerate(trajectory.candidate_routes):
            if len(route.nodes) < 2:
                raise InvalidRouteError(
                    f"Trajectory '{trajectory.track_id}' route #{route_idx} has fewer than 2 nodes: {route.nodes}"
                )

            # Validate nodes exist in graph
            for node_id in route.nodes:
                if not self.graph.has_node(node_id):
                    raise InvalidRouteError(
                        f"Trajectory '{trajectory.track_id}' references unknown node '{node_id}' not found in MobilityGraph."
                    )

            # Validate consecutive connected directed edges exist in graph
            for u, v in zip(route.nodes[:-1], route.nodes[1:]):
                road = self.graph.get_road_by_nodes(u, v)
                if road is None:
                    raise InvalidRouteError(
                        f"Trajectory '{trajectory.track_id}' route contains invalid edge ({u} -> {v}): "
                        f"no directed road segment connects these nodes."
                    )

    def map_route_to_roads(self, candidate_route: CandidateRoute) -> List[str]:
        """Convert an ordered node sequence into a sequence of directed road IDs.

        Args:
            candidate_route: CandidateRoute instance.

        Returns:
            List of road segment IDs corresponding to the consecutive node pairs.

        Raises:
            InvalidRouteError: If an edge does not correspond to a registered road.
        """
        road_ids: List[str] = []
        for u, v in zip(candidate_route.nodes[:-1], candidate_route.nodes[1:]):
            road = self.graph.get_road_by_nodes(u, v)
            if road is None:
                raise InvalidRouteError(f"No directed road segment connects '{u}' -> '{v}'.")
            road_ids.append(road.road_id)
        return road_ids

    def aggregate(
        self,
        trajectories: Iterable[NormalizedTrajectory],
        include_zero_flow_roads: bool = False,
        time_window: Optional[Tuple[Optional[str], Optional[str]]] = None,
    ) -> FlowAggregationResult:
        """Aggregate expected flow from normalized trajectories onto road segments.

        DUPLICATE ROAD TRAVERSAL RULE:
        If a candidate route contains cyclic movements that traverse the same road
        more than once, the road is counted ONCE per candidate route. The expected flow
        models the expected vehicle presence on that road segment from this route
        (weight * probability) rather than visit count.

        Flow contribution:
            expected_flow += trajectory.vehicle_weight * candidate_route.probability

        Args:
            trajectories: Iterable of NormalizedTrajectory instances.
            include_zero_flow_roads: If True, include all network roads in output even if expected flow is 0.0.
            time_window: Optional tuple of (start, end) time window labels.

        Returns:
            FlowAggregationResult containing road flows, contributing trajectory counts,
            and the retained trajectory collection.
        """
        trajectory_list = list(trajectories)
        road_expected_flows: Dict[str, float] = {}
        road_contributing_trajectories: Dict[str, Set[str]] = {}
        road_endpoints: Dict[str, Tuple[str, str]] = {}

        # If include_zero_flow_roads is True, initialize all registered roads
        if include_zero_flow_roads:
            for road in self.graph.all_roads(include_closed=True):
                road_expected_flows[road.road_id] = 0.0
                road_contributing_trajectories[road.road_id] = set()
                road_endpoints[road.road_id] = (road.from_node, road.to_node)

        # Process each trajectory
        for trajectory in trajectory_list:
            self.validate_trajectory_routes(trajectory)

            weight = trajectory.vehicle_weight

            for route in trajectory.candidate_routes:
                prob = route.probability
                if prob == 0.0:
                    continue

                route_road_ids = self.map_route_to_roads(route)

                # Deduplicate roads within the route per the documented traversal rule
                unique_route_roads = list(dict.fromkeys(route_road_ids))

                flow_contribution = weight * prob

                for r_id in unique_route_roads:
                    if r_id not in road_expected_flows:
                        road_obj = self.graph.get_road(r_id)
                        if road_obj is None:
                            raise InvalidRouteError(f"Road '{r_id}' not found in MobilityGraph.")
                        road_expected_flows[r_id] = 0.0
                        road_contributing_trajectories[r_id] = set()
                        road_endpoints[r_id] = (road_obj.from_node, road_obj.to_node)

                    road_expected_flows[r_id] += flow_contribution
                    road_contributing_trajectories[r_id].add(trajectory.track_id)

        # Determine time window labels
        window_start: Optional[str] = None
        window_end: Optional[str] = None
        if time_window is not None:
            window_start, window_end = time_window
        elif trajectory_list:
            # If all trajectories have the same time window, propagate it
            starts = {t.time_window_start for t in trajectory_list if t.time_window_start}
            ends = {t.time_window_end for t in trajectory_list if t.time_window_end}
            if len(starts) == 1:
                window_start = next(iter(starts))
            if len(ends) == 1:
                window_end = next(iter(ends))

        # Build RoadFlow instances sorted deterministically by road_id
        road_flow_objects: Dict[str, RoadFlow] = {}
        for r_id in sorted(road_expected_flows.keys()):
            u, v = road_endpoints[r_id]
            expected_flow = round(road_expected_flows[r_id], 6)
            contrib_count = len(road_contributing_trajectories.get(r_id, set()))

            road_flow_objects[r_id] = RoadFlow(
                road_id=r_id,
                from_node=u,
                to_node=v,
                expected_flow=expected_flow,
                time_window_start=window_start,
                time_window_end=window_end,
                contributing_trajectories_count=contrib_count,
            )

        return FlowAggregationResult(
            flows=road_flow_objects,
            trajectories=trajectory_list,
            time_window_start=window_start,
            time_window_end=window_end,
        )

    def aggregate_by_time_window(
        self,
        trajectories: Iterable[NormalizedTrajectory],
        include_zero_flow_roads: bool = False,
    ) -> Dict[Tuple[Optional[str], Optional[str]], FlowAggregationResult]:
        """Group trajectories by their time window and aggregate flow for each window.

        Args:
            trajectories: Iterable of NormalizedTrajectory instances.
            include_zero_flow_roads: If True, include all network roads in each window result.

        Returns:
            Dictionary mapping (time_window_start, time_window_end) tuples to FlowAggregationResult.
        """
        grouped: Dict[Tuple[Optional[str], Optional[str]], List[NormalizedTrajectory]] = {}
        for t in trajectories:
            key = (t.time_window_start, t.time_window_end)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(t)

        results: Dict[Tuple[Optional[str], Optional[str]], FlowAggregationResult] = {}
        for key, traj_group in sorted(grouped.items(), key=lambda x: (x[0][0] or "", x[0][1] or "")):
            results[key] = self.aggregate(
                traj_group,
                include_zero_flow_roads=include_zero_flow_roads,
                time_window=key,
            )

        return results
