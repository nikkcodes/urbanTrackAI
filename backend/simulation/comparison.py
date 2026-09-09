"""Baseline-versus-counterfactual impact comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from backend.simulation.models import (
    DecisionSummary,
    ODImpact,
    RecommendationClass,
    RoadImpact,
    RouteAssignment,
    Scenario,
    UnroutableDemand,
)
from backend.traffic.models import CongestionLevel, NetworkTrafficSummary, TrafficMetric


@dataclass(frozen=True, slots=True)
class ComparisonConfig:
    """Transparent decision thresholds; units are minutes, ratios, and vehicles/hour."""

    favorable_travel_time_delta_minutes: float = -0.01
    unfavorable_travel_time_delta_minutes: float = 0.01
    favorable_congestion_delta: float = -0.001
    unfavorable_congestion_delta: float = 0.001
    max_favorable_unroutable_demand: float = 0.0


class ScenarioComparator:
    """Build road, OD, and decision comparisons from Phase 3 metrics and assignments."""

    _LEVEL_RANK = {"FREE": 0, "MODERATE": 1, "HEAVY": 2, "SEVERE": 3}

    def __init__(self, config: Optional[ComparisonConfig] = None) -> None:
        self.config = config or ComparisonConfig()
        if self.config.max_favorable_unroutable_demand < 0.0:
            raise ValueError("max_favorable_unroutable_demand must be non-negative")

    def road_impacts(
        self,
        baseline_metrics: Iterable[TrafficMetric],
        scenario_metrics: Iterable[TrafficMetric],
    ) -> Tuple[RoadImpact, ...]:
        baseline = {metric.road_id: metric for metric in baseline_metrics}
        scenario = {metric.road_id: metric for metric in scenario_metrics}
        impacts: List[RoadImpact] = []
        for road_id in sorted(set(baseline) | set(scenario)):
            before = baseline.get(road_id)
            after = scenario.get(road_id)
            if before is None or after is None:
                continue
            baseline_level = before.congestion_level.value
            scenario_level = after.congestion_level.value
            became_new = self._LEVEL_RANK[scenario_level] >= self._LEVEL_RANK["HEAVY"] and self._LEVEL_RANK[baseline_level] < self._LEVEL_RANK["HEAVY"]
            impacts.append(RoadImpact(
                road_id=road_id,
                baseline_hourly_flow=before.hourly_flow or 0.0,
                scenario_hourly_flow=after.hourly_flow or 0.0,
                flow_delta=(after.hourly_flow or 0.0) - (before.hourly_flow or 0.0),
                baseline_utilization=before.utilization_ratio,
                scenario_utilization=after.utilization_ratio,
                utilization_delta=after.utilization_ratio - before.utilization_ratio,
                baseline_congestion_level=baseline_level,
                scenario_congestion_level=scenario_level,
                travel_time_delta_minutes=after.estimated_travel_time_minutes - before.estimated_travel_time_minutes,
                additional_flow=max(0.0, (after.hourly_flow or 0.0) - (before.hourly_flow or 0.0)),
                became_new_bottleneck=became_new,
            ))
        return tuple(impacts)

    def od_impacts(
        self,
        baseline_assignments: Iterable[RouteAssignment],
        scenario_assignments: Iterable[RouteAssignment],
        unroutable: Iterable[UnroutableDemand],
    ) -> Tuple[ODImpact, ...]:
        before = {assignment.track_id: assignment for assignment in baseline_assignments}
        after = {assignment.track_id: assignment for assignment in scenario_assignments}
        impacts: List[ODImpact] = []
        for track_id in sorted(before):
            baseline = before[track_id]
            scenario = after.get(track_id)
            if scenario is None:
                impacts.append(ODImpact(
                    baseline.origin, baseline.destination, baseline.demand,
                    baseline.route_nodes, (), True, baseline.route_travel_time_minutes,
                    None, None, "UNROUTABLE",
                ))
                continue
            impacts.append(ODImpact(
                origin=baseline.origin,
                destination=baseline.destination,
                demand=baseline.demand,
                baseline_route=baseline.route_nodes,
                scenario_route=scenario.route_nodes,
                route_changed=baseline.route_nodes != scenario.route_nodes,
                baseline_travel_time_minutes=baseline.route_travel_time_minutes,
                scenario_travel_time_minutes=scenario.route_travel_time_minutes,
                travel_time_delta_minutes=scenario.route_travel_time_minutes - baseline.route_travel_time_minutes,
            ))
        for item in unroutable:
            if not any(impact.origin == item.origin and impact.destination == item.destination and impact.status == "UNROUTABLE" for impact in impacts):
                impacts.append(ODImpact(item.origin, item.destination, item.demand, (), (), False, None, None, None, "UNROUTABLE"))
        impacts.sort(key=lambda item: (item.origin, item.destination, item.baseline_route, item.status))
        return tuple(impacts)

    def decision(
        self,
        scenario: Scenario,
        road_impacts: Tuple[RoadImpact, ...],
        od_impacts: Tuple[ODImpact, ...],
        baseline_summary: NetworkTrafficSummary,
        scenario_summary: NetworkTrafficSummary,
        unroutable: Tuple[UnroutableDemand, ...],
    ) -> DecisionSummary:
        rerouted_demand = sum(item.demand for item in od_impacts if item.route_changed)
        newly_congested = tuple(sorted(item.road_id for item in road_impacts if item.became_new_bottleneck))
        relieved = tuple(sorted(
            item.road_id for item in road_impacts
            if self._LEVEL_RANK[item.scenario_congestion_level or "FREE"] < self._LEVEL_RANK[item.baseline_congestion_level or "FREE"]
        ))
        travel_delta = scenario_summary.flow_weighted_travel_time_minutes - baseline_summary.flow_weighted_travel_time_minutes
        congestion_delta = scenario_summary.flow_weighted_utilization - baseline_summary.flow_weighted_utilization
        unroutable_total = sum(item.demand for item in unroutable)
        feasible = not unroutable
        if unroutable_total > self.config.max_favorable_unroutable_demand:
            recommendation = RecommendationClass.UNFAVORABLE
            explanation = "UNFAVORABLE because some demand is unroutable after the intervention."
        elif travel_delta >= self.config.unfavorable_travel_time_delta_minutes or congestion_delta >= self.config.unfavorable_congestion_delta or newly_congested:
            recommendation = RecommendationClass.UNFAVORABLE
            explanation = "UNFAVORABLE because estimated travel time, congestion, or bottleneck count increased."
        elif travel_delta <= self.config.favorable_travel_time_delta_minutes and congestion_delta <= self.config.favorable_congestion_delta:
            recommendation = RecommendationClass.FAVORABLE
            explanation = "FAVORABLE because both estimated travel time and weighted utilization decreased."
        else:
            recommendation = RecommendationClass.NEUTRAL
            explanation = "NEUTRAL because measured network impacts remain within configured decision thresholds."
        return DecisionSummary(
            scenario_id=scenario.scenario_id,
            feasible=feasible,
            affected_road_count=sum(1 for item in road_impacts if abs(item.flow_delta) > 0.0),
            rerouted_demand=rerouted_demand,
            newly_congested_roads=newly_congested,
            relieved_roads=relieved,
            total_travel_time_delta_minutes=travel_delta,
            network_congestion_delta=congestion_delta,
            unroutable_demand=unroutable_total,
            recommendation_class=recommendation,
            explanation=explanation,
        )
