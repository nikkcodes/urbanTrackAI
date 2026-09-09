"""Serialization-friendly models for Phase 6 counterfactual simulation."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from backend.traffic.models import CongestionLevel, NetworkTrafficSummary, TrafficMetric


def _label(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _number(value: float, name: str, non_negative: bool = True) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if non_negative and value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return value


class RecommendationClass(str, Enum):
    FAVORABLE = "FAVORABLE"
    NEUTRAL = "NEUTRAL"
    UNFAVORABLE = "UNFAVORABLE"


@dataclass(frozen=True, slots=True)
class Scenario:
    """A deterministic network intervention definition."""

    scenario_id: str
    name: str
    description: str = ""
    closed_road_ids: Tuple[str, ...] = ()
    capacity_modifications_vph: Mapping[str, float] = field(default_factory=dict)
    speed_modifications_kmph: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_id", _label(self.scenario_id, "scenario_id"))
        object.__setattr__(self, "name", _label(self.name, "name"))
        if not isinstance(self.description, str):
            raise ValueError("description must be a string")
        if len(set(self.closed_road_ids)) != len(self.closed_road_ids):
            raise ValueError("closed_road_ids must not contain duplicates")
        if any(not isinstance(road_id, str) or not road_id.strip() for road_id in self.closed_road_ids):
            raise ValueError("closed_road_ids must contain non-empty strings")
        for name, modifications in (("capacity_modifications_vph", self.capacity_modifications_vph), ("speed_modifications_kmph", self.speed_modifications_kmph)):
            if not isinstance(modifications, Mapping):
                raise TypeError(f"{name} must be a mapping")
            for road_id, value in modifications.items():
                if not isinstance(road_id, str) or not road_id.strip():
                    raise ValueError(f"{name} keys must be non-empty road IDs")
                value = _number(value, f"{name}[{road_id!r}]")
                if value <= 0.0:
                    raise ValueError(f"{name}[{road_id!r}] must be positive")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "name": self.name,
            "description": self.description,
            "closed_road_ids": list(self.closed_road_ids),
            "capacity_modifications_vph": dict(self.capacity_modifications_vph),
            "speed_modifications_kmph": dict(self.speed_modifications_kmph),
        }


@dataclass(frozen=True, slots=True)
class RouteAssignment:
    """One deterministic demand assignment before or after intervention."""

    track_id: str
    origin: str
    destination: str
    demand: float
    road_ids: Tuple[str, ...]
    route_nodes: Tuple[str, ...]
    route_travel_time_minutes: float
    route_changed: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "track_id", _label(self.track_id, "track_id"))
        object.__setattr__(self, "origin", _label(self.origin, "origin"))
        object.__setattr__(self, "destination", _label(self.destination, "destination"))
        object.__setattr__(self, "demand", _number(self.demand, "demand"))
        if len(self.route_nodes) < 2 or len(self.road_ids) != len(self.route_nodes) - 1:
            raise ValueError("route_nodes and road_ids must describe the same route")
        object.__setattr__(self, "route_travel_time_minutes", _number(self.route_travel_time_minutes, "route_travel_time_minutes"))


@dataclass(frozen=True, slots=True)
class UnroutableDemand:
    origin: str
    destination: str
    demand: float
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _label(self.origin, "origin"))
        object.__setattr__(self, "destination", _label(self.destination, "destination"))
        object.__setattr__(self, "demand", _number(self.demand, "demand"))
        object.__setattr__(self, "reason", _label(self.reason, "reason"))


@dataclass(frozen=True, slots=True)
class RoadImpact:
    road_id: str
    baseline_hourly_flow: float
    scenario_hourly_flow: float
    flow_delta: float
    baseline_utilization: float
    scenario_utilization: float
    utilization_delta: float
    baseline_congestion_level: Optional[str]
    scenario_congestion_level: Optional[str]
    travel_time_delta_minutes: float
    additional_flow: float
    became_new_bottleneck: bool


@dataclass(frozen=True, slots=True)
class ODImpact:
    origin: str
    destination: str
    demand: float
    baseline_route: Tuple[str, ...]
    scenario_route: Tuple[str, ...]
    route_changed: bool
    baseline_travel_time_minutes: Optional[float]
    scenario_travel_time_minutes: Optional[float]
    travel_time_delta_minutes: Optional[float]
    status: str = "ROUTABLE"


@dataclass(frozen=True, slots=True)
class DecisionSummary:
    scenario_id: str
    feasible: bool
    affected_road_count: int
    rerouted_demand: float
    newly_congested_roads: Tuple[str, ...]
    relieved_roads: Tuple[str, ...]
    total_travel_time_delta_minutes: float
    network_congestion_delta: float
    unroutable_demand: float
    recommendation_class: RecommendationClass
    explanation: str


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    scenario: Scenario
    baseline_metrics: Tuple[TrafficMetric, ...]
    scenario_metrics: Tuple[TrafficMetric, ...]
    baseline_summary: NetworkTrafficSummary
    scenario_summary: NetworkTrafficSummary
    road_impacts: Tuple[RoadImpact, ...]
    od_impacts: Tuple[ODImpact, ...]
    baseline_assignments: Tuple[RouteAssignment, ...]
    scenario_assignments: Tuple[RouteAssignment, ...]
    unroutable: Tuple[UnroutableDemand, ...]
    decision: DecisionSummary
    scenario_graph: Any = field(repr=False, compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scenario": self.scenario.to_dict(),
            "baseline_summary": self.baseline_summary.to_dict(),
            "scenario_summary": self.scenario_summary.to_dict(),
            "road_impacts": [impact.__dict__ if hasattr(impact, "__dict__") else {
                "road_id": impact.road_id,
                "baseline_hourly_flow": impact.baseline_hourly_flow,
                "scenario_hourly_flow": impact.scenario_hourly_flow,
                "flow_delta": impact.flow_delta,
                "baseline_utilization": impact.baseline_utilization,
                "scenario_utilization": impact.scenario_utilization,
                "utilization_delta": impact.utilization_delta,
                "baseline_congestion_level": impact.baseline_congestion_level,
                "scenario_congestion_level": impact.scenario_congestion_level,
                "travel_time_delta_minutes": impact.travel_time_delta_minutes,
                "additional_flow": impact.additional_flow,
                "became_new_bottleneck": impact.became_new_bottleneck,
            } for impact in self.road_impacts],
            "od_impacts": [impact.__dict__ if hasattr(impact, "__dict__") else {
                "origin": impact.origin,
                "destination": impact.destination,
                "demand": impact.demand,
                "baseline_route": list(impact.baseline_route),
                "scenario_route": list(impact.scenario_route),
                "route_changed": impact.route_changed,
                "baseline_travel_time_minutes": impact.baseline_travel_time_minutes,
                "scenario_travel_time_minutes": impact.scenario_travel_time_minutes,
                "travel_time_delta_minutes": impact.travel_time_delta_minutes,
                "status": impact.status,
            } for impact in self.od_impacts],
            "unroutable": [item.__dict__ if hasattr(item, "__dict__") else {
                "origin": item.origin, "destination": item.destination,
                "demand": item.demand, "reason": item.reason,
            } for item in self.unroutable],
            "decision": {
                "scenario_id": self.decision.scenario_id,
                "feasible": self.decision.feasible,
                "affected_road_count": self.decision.affected_road_count,
                "rerouted_demand": self.decision.rerouted_demand,
                "newly_congested_roads": list(self.decision.newly_congested_roads),
                "relieved_roads": list(self.decision.relieved_roads),
                "total_travel_time_delta_minutes": self.decision.total_travel_time_delta_minutes,
                "network_congestion_delta": self.decision.network_congestion_delta,
                "unroutable_demand": self.decision.unroutable_demand,
                "recommendation_class": self.decision.recommendation_class.value,
                "explanation": self.decision.explanation,
            },
        }
