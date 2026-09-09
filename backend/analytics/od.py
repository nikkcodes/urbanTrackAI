"""Origin-destination and probabilistic route demand analytics."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

from backend.flow.aggregation import ExpectedFlowAggregator
from backend.flow.models import FlowAggregationResult, NormalizedTrajectory
from backend.mobility.graph import MobilityGraph
from backend.analytics.models import ODAnalysisResult, ODMatrix, ODPairDemand, RouteDemand


class ODAnalyzer:
    """Analyze retained Phase 2 trajectories without reconstructing OD from road flow."""

    def analyze(self, flow_result: FlowAggregationResult) -> ODAnalysisResult:
        """Build accumulated OD demand from trajectory origins, destinations, and weights."""
        if not isinstance(flow_result, FlowAggregationResult):
            raise TypeError(f"Expected FlowAggregationResult, got: {type(flow_result)}")

        grouped: Dict[Tuple[str, str, Optional[str], Optional[str]], float] = defaultdict(float)
        for trajectory in flow_result.trajectories:
            self._validate_trajectory(trajectory)
            key = (
                trajectory.origin_node,
                trajectory.destination_node,
                trajectory.time_window_start,
                trajectory.time_window_end,
            )
            grouped[key] += trajectory.vehicle_weight

        records = tuple(
            ODPairDemand(
                origin=origin,
                destination=destination,
                demand=demand,
                time_window_start=start,
                time_window_end=end,
            )
            for (origin, destination, start, end), demand in sorted(
                grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2] or "", item[0][3] or "")
            )
        )
        matrix = ODMatrix(pairs=records)
        return ODAnalysisResult(
            matrix=matrix,
            total_demand=matrix.total_demand,
            top_od_pairs=matrix.top_pairs(),
            trajectory_count=len(flow_result.trajectories),
        )

    def analyze_trajectories(self, trajectories: Iterable[NormalizedTrajectory]) -> ODAnalysisResult:
        """Analyze a trajectory iterable through the standard FlowAggregationResult contract."""
        trajectory_list = list(trajectories)
        return self.analyze(FlowAggregationResult(flows={}, trajectories=trajectory_list))

    def analyze_route_demand(
        self,
        flow_result: FlowAggregationResult,
        graph: MobilityGraph,
    ) -> Tuple[RouteDemand, ...]:
        """Aggregate candidate-route demand as weight times route probability.

        This ranks probabilistic routes and corridors; it does not replace the
        Phase 2 road-flow aggregation or collapse candidate routes.
        """
        if not isinstance(flow_result, FlowAggregationResult):
            raise TypeError(f"Expected FlowAggregationResult, got: {type(flow_result)}")
        if not isinstance(graph, MobilityGraph):
            raise TypeError(f"Expected MobilityGraph, got: {type(graph)}")

        aggregator = ExpectedFlowAggregator(graph)
        grouped: Dict[Tuple[Tuple[str, ...], Tuple[str, ...], Optional[str], Optional[str]], float] = defaultdict(float)
        for trajectory in flow_result.trajectories:
            self._validate_trajectory(trajectory)
            aggregator.validate_trajectory_routes(trajectory)
            for route in trajectory.candidate_routes:
                road_ids = tuple(aggregator.map_route_to_roads(route))
                key = (
                    tuple(route.nodes),
                    road_ids,
                    trajectory.time_window_start,
                    trajectory.time_window_end,
                )
                grouped[key] += trajectory.vehicle_weight * route.probability

        return tuple(
            RouteDemand(
                route_nodes=route_nodes,
                road_ids=road_ids,
                demand=demand,
                time_window_start=start,
                time_window_end=end,
            )
            for (route_nodes, road_ids, start, end), demand in sorted(
                grouped.items(), key=lambda item: (item[0][0], item[0][2] or "", item[0][3] or "")
            )
        )

    @staticmethod
    def top_route_demands(route_demands: Iterable[RouteDemand], limit: Optional[int] = None) -> Tuple[RouteDemand, ...]:
        """Return deterministic route-demand ranking."""
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise ValueError("limit must be a non-negative integer or None")
        records = list(route_demands)
        if any(not isinstance(item, RouteDemand) for item in records):
            raise TypeError("route_demands must contain only RouteDemand instances")
        records.sort(key=lambda item: (-item.demand, item.route_nodes, item.time_window_start or "", item.time_window_end or ""))
        return tuple(records if limit is None else records[:limit])

    @staticmethod
    def _validate_trajectory(trajectory: NormalizedTrajectory) -> None:
        if not isinstance(trajectory, NormalizedTrajectory):
            raise TypeError(f"Expected NormalizedTrajectory, got: {type(trajectory)}")
        trajectory.validate_probabilities()
