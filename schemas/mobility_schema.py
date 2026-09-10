"""
Mobility Analytics Schemas for UrbanTrack AI (Day 6).

Defines data models for road-level flow, capacity utilization, OD matrices,
network structural centrality, priority road rankings, and the consolidated
CityMobilityReport.

Audit Hardening:
- Clearly separates expected_demand_in_window from expected_demand_vph.
- Dimensionally validates utilization_ratio against hourly capacity.
- Distinguishes original inference probabilities from network-constrained flow allocation probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from schemas.normalized_trajectory_schema import NormalizedCandidateRoute


class RoadStatus(str, Enum):
    """Operational road capacity utilization category."""
    NORMAL = "normal"              # utilization < 0.50
    MODERATE = "moderate"          # 0.50 <= utilization < 0.80
    HIGH = "high"                  # 0.80 <= utilization <= 1.00
    OVER_CAPACITY = "over_capacity"# utilization > 1.00
    UNCALIBRATED = "uncalibrated"  # observation window duration not available

    @classmethod
    def classify(cls, utilization_ratio: Optional[float]) -> RoadStatus:
        """Classify utilization ratio into operational status category."""
        if utilization_ratio is None:
            return cls.UNCALIBRATED
        if utilization_ratio < 0.50:
            return cls.NORMAL
        elif utilization_ratio < 0.80:
            return cls.MODERATE
        elif utilization_ratio <= 1.00:
            return cls.HIGH
        else:
            return cls.OVER_CAPACITY


@dataclass
class FlowAllocatedRoute:
    """
    Candidate route with explicitly separated inference and flow allocation probabilities.

    Critical Semantic Note:
    'original_route_probability' is the relative estimated likelihood from trajectory inference.
    'flow_allocation_probability' is the normalized allocation weight across open/feasible
    roads on the network, used strictly for traffic flow assignment.
    """
    route: NormalizedCandidateRoute
    road_ids: List[str]
    original_route_probability: float
    flow_allocation_probability: float
    network_constraint_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": list(self.route.nodes),
            "road_ids": list(self.road_ids),
            "original_route_probability": round(self.original_route_probability, 6),
            "flow_allocation_probability": round(self.flow_allocation_probability, 6),
            "network_constraint_applied": self.network_constraint_applied,
        }


@dataclass
class RoadFlowMetric:
    """
    Road segment flow and capacity utilization metric.

    Critical Dimensional Semantics:
    1. 'expected_demand_in_window': Total weighted demand (PCU) accumulated within
       the observation window.
    2. 'expected_demand_vph': Rate converted to vehicles per hour:
       expected_demand_in_window / (duration_seconds / 3600.0).
       Only calculable when duration_seconds > 0.
    3. 'capacity_vph': Physical road design capacity in vehicles per hour.
    4. 'utilization_ratio': expected_demand_vph / capacity_vph.
       Only dimensionally valid when duration_seconds > 0 and capacity_vph > 0.
    5. 'expected_demand': Alias field matching expected_demand_in_window.
    """
    road_id: str
    from_node: str
    to_node: str
    expected_demand_in_window: float = 0.0
    capacity_vph: float = 0.0
    expected_demand_vph: Optional[float] = None
    utilization_ratio: Optional[float] = None
    is_hourly_rate_valid: bool = False
    duration_seconds: Optional[float] = None
    status: RoadStatus = RoadStatus.NORMAL
    contributing_trajectories_count: int = 0
    contributing_identities: List[str] = field(default_factory=list)
    reliability_summary: Optional[Dict[str, Any]] = None
    uncertainty_summary: Optional[Dict[str, Any]] = None
    time_window_start: Optional[Any] = None
    time_window_end: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    expected_demand: float = 0.0

    def __post_init__(self) -> None:
        if self.expected_demand_in_window == 0.0 and self.expected_demand != 0.0:
            self.expected_demand_in_window = self.expected_demand
        elif self.expected_demand_in_window != 0.0 and self.expected_demand == 0.0:
            self.expected_demand = self.expected_demand_in_window

    def to_dict(self) -> Dict[str, Any]:
        return {
            "road_id": self.road_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "expected_demand_in_window": round(self.expected_demand_in_window, 4),
            "expected_demand": round(self.expected_demand_in_window, 4),
            "expected_demand_vph": round(self.expected_demand_vph, 4) if self.expected_demand_vph is not None else None,
            "capacity_vph": self.capacity_vph,
            "utilization_ratio": round(self.utilization_ratio, 6) if self.utilization_ratio is not None else None,
            "is_hourly_rate_valid": self.is_hourly_rate_valid,
            "duration_seconds": self.duration_seconds,
            "status": self.status.value if isinstance(self.status, RoadStatus) else str(self.status),
            "contributing_trajectories": self.contributing_trajectories_count,
            "contributing_identities": list(self.contributing_identities),
            "reliability_summary": dict(self.reliability_summary) if self.reliability_summary else None,
            "uncertainty_summary": dict(self.uncertainty_summary) if self.uncertainty_summary else None,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "metadata": dict(self.metadata),
        }


@dataclass
class ODPairDemand:
    """
    Origin-Destination demand for a junction pair.

    Critical Semantic Note:
    OD demand accumulates total physical vehicle_weight without multiplying by
    candidate route probabilities (the vehicle travels between origin and destination
    regardless of which route hypothesis was chosen).
    """
    origin: str
    destination: str
    demand: float
    contributing_trajectories_count: int = 0
    time_window_start: Optional[Any] = None
    time_window_end: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin,
            "destination": self.destination,
            "demand": round(self.demand, 4),
            "contributing_trajectories": self.contributing_trajectories_count,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
        }


@dataclass
class ODMatrix:
    """Consolidated Origin-Destination demand matrix."""
    pairs: List[ODPairDemand] = field(default_factory=list)

    @property
    def total_demand(self) -> float:
        """Total accumulated physical vehicle demand across all OD pairs."""
        return sum(p.demand for p in self.pairs)

    def get_demand(self, origin: str, destination: str) -> float:
        """Retrieve demand for a specific origin-destination pair."""
        for p in self.pairs:
            if p.origin == origin and p.destination == destination:
                return p.demand
        return 0.0

    def top_pairs(self, limit: Optional[int] = None) -> List[ODPairDemand]:
        """Return OD pairs ranked by demand descending."""
        sorted_pairs = sorted(self.pairs, key=lambda p: (-p.demand, p.origin, p.destination))
        return sorted_pairs if limit is None else sorted_pairs[:limit]

    def to_dict(self) -> List[Dict[str, Any]]:
        return [p.to_dict() for p in self.pairs]


@dataclass
class RoadCentralityMetric:
    """
    Network structural importance metrics for a road segment.

    Critical Semantic Note:
    Degree and betweenness centrality reflect topology and network connectivity,
    labeled as 'network structural importance' rather than 'traffic importance'.
    """
    road_id: str
    from_node: str
    to_node: str
    in_degree: int
    out_degree: int
    total_degree: int
    betweenness_centrality: float
    metric_type: str = "network_structural_importance"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "road_id": self.road_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "in_degree": self.in_degree,
            "out_degree": self.out_degree,
            "total_degree": self.total_degree,
            "betweenness_centrality": round(self.betweenness_centrality, 6),
            "metric_type": self.metric_type,
        }


@dataclass
class RoadPriority:
    """
    Ranked priority road segment based on operational utilization and expected demand.
    """
    road_id: str
    priority_score: float
    rank: int
    expected_demand_in_window: float = 0.0
    capacity_vph: float = 0.0
    expected_demand_vph: Optional[float] = None
    utilization_ratio: Optional[float] = None
    is_hourly_rate_valid: bool = False
    formula: str = "utilization_descending_then_expected_demand_descending"
    expected_demand: float = 0.0

    def __post_init__(self) -> None:
        if self.expected_demand_in_window == 0.0 and self.expected_demand != 0.0:
            self.expected_demand_in_window = self.expected_demand
        elif self.expected_demand_in_window != 0.0 and self.expected_demand == 0.0:
            self.expected_demand = self.expected_demand_in_window

    def to_dict(self) -> Dict[str, Any]:
        return {
            "road_id": self.road_id,
            "priority_score": round(self.priority_score, 6),
            "rank": self.rank,
            "expected_demand_in_window": round(self.expected_demand_in_window, 4),
            "expected_demand": round(self.expected_demand_in_window, 4),
            "expected_demand_vph": round(self.expected_demand_vph, 4) if self.expected_demand_vph is not None else None,
            "capacity_vph": self.capacity_vph,
            "utilization_ratio": round(self.utilization_ratio, 6) if self.utilization_ratio is not None else None,
            "is_hourly_rate_valid": self.is_hourly_rate_valid,
            "formula": self.formula,
        }


@dataclass
class ValidationIssue:
    """Validation issue or warning encountered while processing trajectories."""
    track_id: str
    issue_type: str
    message: str
    recovered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "issue_type": self.issue_type,
            "message": self.message,
            "recovered": self.recovered,
        }


@dataclass
class CityMobilityReport:
    """
    Consolidated Day 6 City Mobility Analytics Report.

    Provides a clean machine-readable integration payload containing road-level flow,
    OD matrix, network centrality, priority roads, and execution metadata.
    """
    time_window: Dict[str, Any]
    road_metrics: List[RoadFlowMetric]
    od_demand: List[ODPairDemand]
    priority_roads: List[RoadPriority]
    network_centrality: Dict[str, RoadCentralityMetric] = field(default_factory=dict)
    validation_issues: List[ValidationIssue] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_window": dict(self.time_window),
            "road_metrics": [m.to_dict() for m in self.road_metrics],
            "od_demand": [od.to_dict() for od in self.od_demand],
            "priority_roads": [p.road_id for p in self.priority_roads],
            "priority_road_details": [p.to_dict() for p in self.priority_roads],
            "network_centrality": {
                rid: c.to_dict() for rid, c in self.network_centrality.items()
            },
            "validation_issues": [v.to_dict() for v in self.validation_issues],
            "summary": dict(self.summary),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
