"""Counterfactual simulation orchestration."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Tuple

from backend.flow.models import FlowAggregationResult, NormalizedTrajectory, RoadFlow
from backend.mobility.graph import MobilityGraph
from backend.simulation.comparison import ComparisonConfig, ScenarioComparator
from backend.simulation.models import Scenario, ScenarioResult
from backend.simulation.routing import DeterministicRouter
from backend.simulation.scenarios import ScenarioGraphBuilder
from backend.traffic.metrics import TrafficMetricsCalculator
from backend.traffic.models import NetworkTrafficSummary, TrafficMetric


class CounterfactualSimulationEngine:
    """Run deterministic route redistribution and reuse Phase 3 traffic metrics."""

    def __init__(
        self,
        aggregation_duration_hours: float = 1.0,
        time_window_start: Optional[str] = None,
        time_window_end: Optional[str] = None,
        max_candidate_routes: int = 20,
        comparison_config: Optional[ComparisonConfig] = None,
    ) -> None:
        if not isinstance(aggregation_duration_hours, (int, float)) or not math.isfinite(aggregation_duration_hours) or aggregation_duration_hours <= 0.0:
            raise ValueError("aggregation_duration_hours must be a finite positive number")
        if (time_window_start is None) != (time_window_end is None):
            raise ValueError("time_window_start and time_window_end must be provided together")
        self.aggregation_duration_hours = float(aggregation_duration_hours)
        self.time_window_start = time_window_start
        self.time_window_end = time_window_end
        self.router = DeterministicRouter(max_candidate_routes=max_candidate_routes)
        self.graph_builder = ScenarioGraphBuilder()
        self.comparator = ScenarioComparator(comparison_config)

    def simulate(
        self,
        baseline_graph: MobilityGraph,
        trajectories: Iterable[NormalizedTrajectory],
        scenario: Scenario,
    ) -> ScenarioResult:
        """Simulate one scenario against an unchanged baseline graph."""
        if not isinstance(baseline_graph, MobilityGraph):
            raise TypeError(f"Expected MobilityGraph, got: {type(baseline_graph)}")
        if not isinstance(scenario, Scenario):
            raise TypeError(f"Expected Scenario, got: {type(scenario)}")
        trajectory_list = list(trajectories)
        baseline_assignments = self.router.assign_baseline(trajectory_list, baseline_graph)
        scenario_graph = self.graph_builder.build(baseline_graph, scenario)
        scenario_outcome = self.router.assign_counterfactual(trajectory_list, baseline_assignments.assignments, scenario_graph)

        baseline_flows = self._flows_from_assignments(baseline_assignments.assignments)
        scenario_flows = self._flows_from_assignments(scenario_outcome.assignments)
        baseline_metrics = self._calculate_metrics(baseline_graph, baseline_flows)
        scenario_metrics = self._calculate_metrics(scenario_graph, scenario_flows)
        baseline_summary = TrafficMetricsCalculator(graph=baseline_graph, default_aggregation_duration_hours=self.aggregation_duration_hours).summarize_network(baseline_metrics, self.time_window_start, self.time_window_end)
        scenario_summary = TrafficMetricsCalculator(graph=scenario_graph, default_aggregation_duration_hours=self.aggregation_duration_hours).summarize_network(scenario_metrics, self.time_window_start, self.time_window_end)
        impacts = self.comparator.road_impacts(baseline_metrics, scenario_metrics)
        od_impacts = self.comparator.od_impacts(baseline_assignments.assignments, scenario_outcome.assignments, scenario_outcome.unroutable)
        decision = self.comparator.decision(scenario, impacts, od_impacts, baseline_summary, scenario_summary, scenario_outcome.unroutable)
        return ScenarioResult(
            scenario=scenario,
            baseline_metrics=tuple(baseline_metrics),
            scenario_metrics=tuple(scenario_metrics),
            baseline_summary=baseline_summary,
            scenario_summary=scenario_summary,
            road_impacts=impacts,
            od_impacts=od_impacts,
            baseline_assignments=baseline_assignments.assignments,
            scenario_assignments=scenario_outcome.assignments,
            unroutable=scenario_outcome.unroutable,
            decision=decision,
            scenario_graph=scenario_graph,
        )

    def simulate_many(
        self,
        baseline_graph: MobilityGraph,
        trajectories: Iterable[NormalizedTrajectory],
        scenarios: Iterable[Scenario],
    ) -> Tuple[ScenarioResult, ...]:
        """Run independent scenarios and rank results deterministically."""
        results = [self.simulate(baseline_graph, trajectories, scenario) for scenario in scenarios]
        results.sort(key=lambda result: (
            not result.decision.feasible,
            result.decision.recommendation_class.value,
            result.decision.total_travel_time_delta_minutes,
            result.decision.unroutable_demand,
            result.scenario.scenario_id,
        ))
        return tuple(results)

    def _flows_from_assignments(self, assignments: Iterable) -> Dict[str, float]:
        flows: Dict[str, float] = {}
        for assignment in assignments:
            hourly_demand = assignment.demand / self.aggregation_duration_hours
            for road_id in assignment.road_ids:
                flows[road_id] = flows.get(road_id, 0.0) + hourly_demand
        return flows

    def _calculate_metrics(self, graph: MobilityGraph, hourly_flows: Dict[str, float]) -> List[TrafficMetric]:
        road_flows = {
            road.road_id: RoadFlow(
                road_id=road.road_id,
                from_node=road.from_node,
                to_node=road.to_node,
                expected_flow=hourly_flows.get(road.road_id, 0.0) * self.aggregation_duration_hours,
                time_window_start=self.time_window_start,
                time_window_end=self.time_window_end,
            )
            for road in graph.all_roads(include_closed=True)
        }
        result = FlowAggregationResult(
            flows=road_flows,
            time_window_start=self.time_window_start,
            time_window_end=self.time_window_end,
        )
        calculator = TrafficMetricsCalculator(graph=graph, default_aggregation_duration_hours=self.aggregation_duration_hours)
        return calculator.calculate_metrics(result)
