"""
Scenario Schema and Data Models for UrbanTrack AI (Day 8).

Defines data structures for counterfactual / what-if traffic simulation:
1. ScenarioType: Supported hypothetical network interventions.
2. ScenarioStatus: Execution and validation status codes.
3. ScenarioDefinition: Specification of a counterfactual scenario with strict validation.
4. RoadImpactMetric: Road-level comparison between baseline and counterfactual states.
5. ODImpactMetric: Origin-Destination demand comparison.
6. CounterfactualReport: Consolidated explainable impact report.

Non-Negotiable Semantics Preserved:
- Baseline immutability: baseline state is never mutated.
- Probability semantics: relative estimated likelihood vs counterfactual allocation probability.
- Demand semantics: vehicle_weight * route_allocation (never multiplied by reliability).
- Reliability and uncertainty: preserved as evidence quality indicators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from typing import Any, Dict, List, Optional, Tuple, Union

from inference.road_graph import RoadGraph
from schemas.mobility_schema import RoadFlowMetric, RoadStatus, ODPairDemand


class ScenarioType(str, Enum):
    """Supported hypothetical network interventions."""
    ROAD_CLOSURE = "ROAD_CLOSURE"
    CAPACITY_REDUCTION = "CAPACITY_REDUCTION"
    CAPACITY_INCREASE = "CAPACITY_INCREASE"
    DEMAND_INCREASE = "DEMAND_INCREASE"
    DEMAND_DECREASE = "DEMAND_DECREASE"
    MULTI_ROAD_CLOSURE = "MULTI_ROAD_CLOSURE"
    ROAD_RECOVERY = "ROAD_RECOVERY"
    INVALID_SCENARIO = "INVALID_SCENARIO"  # For testing invalid scenario rejection


class ScenarioStatus(str, Enum):
    """Operational and validation status of a counterfactual simulation."""
    SUCCESS = "success"
    INVALID_INPUT = "invalid_input"
    NO_FEASIBLE_ROUTE = "no_feasible_route"
    PARTIALLY_ROUTED = "partially_routed"
    EXECUTION_ERROR = "execution_error"


@dataclass
class ScenarioDefinition:
    """
    Specification of a counterfactual scenario.

    Attributes:
        scenario_id: Unique identifier for the scenario (e.g., 'SCN_001').
        scenario_type: Category of intervention (ScenarioType).
        description: Human-readable explanation of the hypothetical scenario.
        affected_roads: List of road segment IDs directly targeted by intervention.
        capacity_changes: Mapping of road_id to new scenario capacity (in vph).
        demand_multiplier: Multiplier applied to baseline vehicle demand (default 1.0).
        assumptions: List of explicit hypothetical assumptions made for this scenario.
        metadata: Additional context, configuration, or experimental flags.
    """
    scenario_id: str
    scenario_type: Union[ScenarioType, str]
    description: str
    affected_roads: List[str] = field(default_factory=list)
    capacity_changes: Dict[str, float] = field(default_factory=dict)
    demand_multiplier: float = 1.0
    assumptions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Coerce string scenario type if needed."""
        if isinstance(self.scenario_type, str):
            try:
                self.scenario_type = ScenarioType(self.scenario_type)
            except ValueError:
                # Keep as string for validation failure handling
                pass

    def validate(self, road_graph: RoadGraph) -> Tuple[bool, Optional[str]]:
        """
        Validate scenario parameters against road graph and physical bounds.

        Validation Rules:
        1. Scenario type must be recognized and supported (not INVALID_SCENARIO).
        2. All affected roads must exist in the provided road graph.
        3. Capacity changes must reference existing roads and have non-negative values.
        4. Demand multiplier must be a positive finite float (> 0.0).
        5. Closures require at least one affected road.
        6. Capacity changes require at least one valid entry.
        7. Recovery must reference roads that can be meaningfully restored.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # 1. Type validation
        if not isinstance(self.scenario_type, ScenarioType) or self.scenario_type == ScenarioType.INVALID_SCENARIO:
            return False, f"Unsupported or invalid scenario type: '{self.scenario_type}'"

        # 2. Demand multiplier validation
        if not isinstance(self.demand_multiplier, (int, float)) or math.isnan(self.demand_multiplier) or math.isinf(self.demand_multiplier):
            return False, f"demand_multiplier must be a finite float, got: {self.demand_multiplier}"
        if self.demand_multiplier <= 0.0:
            return False, f"demand_multiplier must be strictly positive (> 0.0), got: {self.demand_multiplier}"

        # 3. Affected roads existence
        if not isinstance(self.affected_roads, list):
            return False, f"affected_roads must be a list, got: {type(self.affected_roads)}"

        for rid in self.affected_roads:
            if not isinstance(rid, str) or not rid.strip():
                return False, f"Invalid road ID in affected_roads: '{rid}'"
            if rid not in road_graph.edges:
                return False, f"Road ID '{rid}' in affected_roads does not exist in the road graph."

        # 4. Capacity changes validation
        if not isinstance(self.capacity_changes, dict):
            return False, f"capacity_changes must be a dictionary, got: {type(self.capacity_changes)}"

        for rid, cap in self.capacity_changes.items():
            if rid not in road_graph.edges:
                return False, f"Road ID '{rid}' in capacity_changes does not exist in the road graph."
            if not isinstance(cap, (int, float)) or math.isnan(cap) or math.isinf(cap):
                return False, f"Capacity for road '{rid}' must be a finite float, got: {cap}"
            if cap < 0.0:
                return False, f"Capacity for road '{rid}' must be non-negative (>= 0.0), got: {cap}"

        # 5. Consistency per scenario type
        if self.scenario_type in (ScenarioType.ROAD_CLOSURE, ScenarioType.MULTI_ROAD_CLOSURE):
            if not self.affected_roads:
                return False, f"Scenario '{self.scenario_id}' of type '{self.scenario_type.value}' requires non-empty affected_roads."

        if self.scenario_type in (ScenarioType.CAPACITY_REDUCTION, ScenarioType.CAPACITY_INCREASE):
            if not self.capacity_changes and not self.affected_roads:
                return False, f"Scenario '{self.scenario_id}' requires capacity_changes or affected_roads."

        if self.scenario_type == ScenarioType.ROAD_RECOVERY:
            if not self.affected_roads:
                return False, f"Scenario '{self.scenario_id}' of type ROAD_RECOVERY requires non-empty affected_roads."
            # Check if roads can meaningfully be restored
            all_open = all(not road_graph.is_road_closed(rid) for rid in self.affected_roads)
            is_hypothetical_initial_closure = self.metadata.get("hypothetical_initial_closure", False)
            if all_open and not is_hypothetical_initial_closure:
                return False, (
                    f"Roads {self.affected_roads} are already open in the baseline graph. "
                    f"Recovery cannot be evaluated without an initial closure state."
                )

        return True, None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize scenario definition to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type.value if isinstance(self.scenario_type, ScenarioType) else str(self.scenario_type),
            "description": self.description,
            "affected_roads": list(self.affected_roads),
            "capacity_changes": dict(self.capacity_changes),
            "demand_multiplier": self.demand_multiplier,
            "assumptions": list(self.assumptions),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ScenarioDefinition:
        """Construct ScenarioDefinition from dictionary."""
        return cls(
            scenario_id=str(data["scenario_id"]),
            scenario_type=data.get("scenario_type", ScenarioType.ROAD_CLOSURE),
            description=str(data.get("description", "")),
            affected_roads=list(data.get("affected_roads", [])),
            capacity_changes=dict(data.get("capacity_changes", {})),
            demand_multiplier=float(data.get("demand_multiplier", 1.0)),
            assumptions=list(data.get("assumptions", [])),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class RoadImpactMetric:
    """
    Road-level impact comparison between baseline and counterfactual states.

    Attributes:
        road_id: Road identifier.
        from_node: Upstream junction node.
        to_node: Downstream junction node.
        baseline_demand: Expected demand in baseline (PCU in window).
        counterfactual_demand: Expected demand in counterfactual (PCU in window).
        demand_delta: counterfactual_demand - baseline_demand.
        demand_delta_percent: Percentage demand change relative to baseline (None if baseline is 0.0).
        baseline_utilization: Baseline capacity utilization ratio.
        counterfactual_utilization: Counterfactual capacity utilization ratio.
        utilization_delta: counterfactual_utilization - baseline_utilization.
        baseline_status: Operational RoadStatus in baseline.
        counterfactual_status: Operational RoadStatus in counterfactual.
        is_new_bottleneck: True if road transitioned from normal/moderate to high/over_capacity.
        is_relieved: True if demand or utilization decreased substantially.
        is_alternate_corridor: True if road absorbed displaced demand from rerouted traffic.
        is_bottleneck_spillover: True if new bottleneck topologically connected to an affected road.
    """
    road_id: str
    from_node: str
    to_node: str
    baseline_demand: float
    counterfactual_demand: float
    demand_delta: float
    demand_delta_percent: Optional[float] = None
    baseline_utilization: Optional[float] = None
    counterfactual_utilization: Optional[float] = None
    utilization_delta: Optional[float] = None
    baseline_status: str = "normal"
    counterfactual_status: str = "normal"
    is_new_bottleneck: bool = False
    is_relieved: bool = False
    is_alternate_corridor: bool = False
    is_bottleneck_spillover: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize road impact metric to dictionary."""
        return {
            "road_id": self.road_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "baseline_demand": round(self.baseline_demand, 4),
            "counterfactual_demand": round(self.counterfactual_demand, 4),
            "demand_delta": round(self.demand_delta, 4),
            "demand_delta_percent": round(self.demand_delta_percent, 2) if self.demand_delta_percent is not None else None,
            "baseline_utilization": round(self.baseline_utilization, 6) if self.baseline_utilization is not None else None,
            "counterfactual_utilization": round(self.counterfactual_utilization, 6) if self.counterfactual_utilization is not None else None,
            "utilization_delta": round(self.utilization_delta, 6) if self.utilization_delta is not None else None,
            "baseline_status": self.baseline_status,
            "counterfactual_status": self.counterfactual_status,
            "is_new_bottleneck": self.is_new_bottleneck,
            "is_relieved": self.is_relieved,
            "is_alternate_corridor": self.is_alternate_corridor,
            "is_bottleneck_spillover": self.is_bottleneck_spillover,
        }


@dataclass
class ODImpactMetric:
    """
    Origin-Destination demand impact comparison.

    Attributes:
        origin: Starting junction node ID.
        destination: Ending junction node ID.
        baseline_demand: Total vehicle demand in baseline state.
        counterfactual_demand: Total routed vehicle demand in counterfactual state.
        demand_delta: counterfactual_demand - baseline_demand.
        unroutable_demand: Unroutable demand for this OD pair.
    """
    origin: str
    destination: str
    baseline_demand: float
    counterfactual_demand: float
    demand_delta: float
    unroutable_demand: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize OD impact metric to dictionary."""
        return {
            "origin": self.origin,
            "destination": self.destination,
            "baseline_demand": round(self.baseline_demand, 4),
            "counterfactual_demand": round(self.counterfactual_demand, 4),
            "demand_delta": round(self.demand_delta, 4),
            "unroutable_demand": round(self.unroutable_demand, 4),
        }


@dataclass
class CounterfactualReport:
    """
    Consolidated Day 8 Counterfactual Traffic Simulation Report.

    Compares baseline network state against counterfactual scenario state
    and produces an explainable, auditable impact report.
    """
    scenario_id: str
    scenario_type: str
    status: ScenarioStatus
    baseline: Dict[str, Any] = field(default_factory=dict)
    counterfactual: Dict[str, Any] = field(default_factory=dict)
    impact: Dict[str, Any] = field(default_factory=dict)
    assumptions: List[str] = field(default_factory=list)
    uncertainties: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize report to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "scenario_type": self.scenario_type,
            "status": self.status.value if isinstance(self.status, ScenarioStatus) else str(self.status),
            "baseline": dict(self.baseline),
            "counterfactual": dict(self.counterfactual),
            "impact": dict(self.impact),
            "assumptions": list(self.assumptions),
            "uncertainties": list(self.uncertainties),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CounterfactualReport:
        """Construct CounterfactualReport from dictionary."""
        st = data.get("status", ScenarioStatus.SUCCESS)
        if isinstance(st, str):
            try:
                st = ScenarioStatus(st)
            except ValueError:
                st = ScenarioStatus.SUCCESS

        return cls(
            scenario_id=str(data["scenario_id"]),
            scenario_type=str(data["scenario_type"]),
            status=st,
            baseline=dict(data.get("baseline", {})),
            counterfactual=dict(data.get("counterfactual", {})),
            impact=dict(data.get("impact", {})),
            assumptions=list(data.get("assumptions", [])),
            uncertainties=list(data.get("uncertainties", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self, indent: int = 2) -> str:
        """Convert report to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
