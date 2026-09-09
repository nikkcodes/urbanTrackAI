"""Validated models for Phase 5 mobility-state comparison."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from backend.analytics.models import Bottleneck, RouteDemand, UrbanMobilityAnalysisResult, validate_time_window
from backend.mobility.graph import MobilityGraph


DistributionKey = Tuple[str, ...]


def _finite_non_negative(value: float, name: str) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return float(value)


def _topology_identity(graph: MobilityGraph) -> str:
    nodes = tuple(sorted(node.node_id for node in graph.all_nodes()))
    edges = tuple(sorted((road.from_node, road.to_node, road.road_id) for road in graph.all_roads(include_closed=True)))
    payload = repr((nodes, edges)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class RoadObservationStatus(str, Enum):
    COMPARABLE = "COMPARABLE"
    MISSING_CURRENT = "MISSING_CURRENT"
    MISSING_BASELINE = "MISSING_BASELINE"


@dataclass(frozen=True, slots=True)
class MobilitySnapshot:
    """Minimal normalized mobility state consumed by the anomaly detector.

    `road_hourly_flows`, OD demand, and route demand are window-specific state
    values. Accumulated OD/route demand is retained as demand; road values are
    hourly rates. Missing road keys mean unobserved, not zero flow.
    """

    snapshot_id: str
    network_identity: str
    aggregation_duration_hours: float
    road_hourly_flows: Mapping[str, float]
    od_demand: Mapping[DistributionKey, float]
    route_demand: Mapping[DistributionKey, float]
    hhi: float
    bottlenecks: Tuple[Bottleneck, ...] = ()
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot_id, str) or not self.snapshot_id.strip():
            raise ValueError("snapshot_id must be a non-empty string")
        if not isinstance(self.network_identity, str) or not self.network_identity.strip():
            raise ValueError("network_identity must be a non-empty string")
        object.__setattr__(self, "snapshot_id", self.snapshot_id.strip())
        object.__setattr__(self, "network_identity", self.network_identity.strip())
        duration = _finite_non_negative(self.aggregation_duration_hours, "aggregation_duration_hours")
        if duration <= 0.0:
            raise ValueError("aggregation_duration_hours must be positive")
        object.__setattr__(self, "aggregation_duration_hours", duration)
        object.__setattr__(self, "hhi", _finite_non_negative(self.hhi, "hhi"))
        if self.hhi > 1.0 + 1e-9:
            raise ValueError("hhi must be between 0.0 and 1.0")
        for name, values in (("road_hourly_flows", self.road_hourly_flows), ("od_demand", self.od_demand), ("route_demand", self.route_demand)):
            if not isinstance(values, Mapping):
                raise TypeError(f"{name} must be a mapping")
            for key, value in values.items():
                if name == "road_hourly_flows" and (not isinstance(key, str) or not key.strip()):
                    raise ValueError("road_hourly_flows keys must be non-empty road IDs")
                if name != "road_hourly_flows" and (not isinstance(key, tuple) or not key or any(not isinstance(part, str) or not part for part in key)):
                    raise ValueError(f"{name} keys must be non-empty string tuples")
                _finite_non_negative(value, f"{name}[{key!r}]")
        if any(not isinstance(item, Bottleneck) for item in self.bottlenecks):
            raise TypeError("bottlenecks must contain only Bottleneck instances")
        start, end = validate_time_window(self.time_window_start, self.time_window_end)
        object.__setattr__(self, "time_window_start", start)
        object.__setattr__(self, "time_window_end", end)

    @classmethod
    def from_phase4_result(
        cls,
        snapshot_id: str,
        result: UrbanMobilityAnalysisResult,
        graph: MobilityGraph,
        aggregation_duration_hours: float,
        time_window_start: Optional[str] = None,
        time_window_end: Optional[str] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "MobilitySnapshot":
        """Adapt an existing Phase 4 result without recalculating its analytics."""
        if not isinstance(result, UrbanMobilityAnalysisResult):
            raise TypeError(f"Expected UrbanMobilityAnalysisResult, got: {type(result)}")
        if not isinstance(graph, MobilityGraph):
            raise TypeError(f"Expected MobilityGraph, got: {type(graph)}")
        start, end = validate_time_window(time_window_start, time_window_end)
        route_records = result.route_demands
        if start is not None:
            route_records = tuple(
                route for route in route_records
                if (route.time_window_start, route.time_window_end) == (start, end)
            )
        od_records = result.od_analysis.matrix.pairs
        if start is not None:
            od_records = tuple(
                pair for pair in od_records
                if (pair.time_window_start, pair.time_window_end) == (start, end)
            )
        od_demand: Dict[DistributionKey, float] = {}
        for pair in od_records:
            key = (pair.origin, pair.destination)
            od_demand[key] = od_demand.get(key, 0.0) + pair.demand
        route_demand: Dict[DistributionKey, float] = {}
        for route in route_records:
            key = tuple(route.road_ids)
            route_demand[key] = route_demand.get(key, 0.0) + route.demand
        network = result.network
        road_hourly_flows = {
            road_id: network.total_hourly_flow * share
            for road_id, share in network.road_flow_shares.items()
            if share > 0.0
        }
        return cls(
            snapshot_id=snapshot_id,
            network_identity=_topology_identity(graph),
            aggregation_duration_hours=aggregation_duration_hours,
            road_hourly_flows=road_hourly_flows,
            od_demand=od_demand,
            route_demand=route_demand,
            hhi=network.flow_concentration_hhi,
            bottlenecks=result.bottlenecks,
            time_window_start=start,
            time_window_end=end,
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the snapshot with tuple distribution keys made readable."""
        return {
            "snapshot_id": self.snapshot_id,
            "network_identity": self.network_identity,
            "aggregation_duration_hours": self.aggregation_duration_hours,
            "road_hourly_flows": dict(self.road_hourly_flows),
            "od_demand": {" -> ".join(key): value for key, value in self.od_demand.items()},
            "route_demand": {" -> ".join(key): value for key, value in self.route_demand.items()},
            "hhi": self.hhi,
            "bottlenecks": [item.road_id for item in self.bottlenecks],
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class RoadAnomalyEvidence:
    """Comparison evidence for one road, including missing-observation state."""

    road_id: str
    status: RoadObservationStatus
    baseline_hourly_flow: Optional[float]
    current_hourly_flow: Optional[float]
    absolute_change: Optional[float]
    relative_change: Optional[float]
    flow_signal: Optional[str] = None
    bottleneck_signal: Optional[str] = None
    evidence_count: int = 0


@dataclass(frozen=True, slots=True)
class DistributionShift:
    """Jensen-Shannon divergence comparison for OD or route distributions."""

    divergence: float
    baseline_distribution: Mapping[DistributionKey, float]
    current_distribution: Mapping[DistributionKey, float]
    threshold_exceeded: bool


@dataclass(frozen=True, slots=True)
class BottleneckChange:
    """Before/after bottleneck state change for one road."""

    road_id: str
    baseline_severity: Optional[str]
    current_severity: Optional[str]
    change_type: str


@dataclass(frozen=True, slots=True)
class SpatialAnomalyRegion:
    """Deterministic connected component of anomalous roads."""

    region_id: str
    road_ids: Tuple[str, ...]
    evidence_count: int
    severity: str
    explanation: str


@dataclass(frozen=True, slots=True)
class AnomalyEvent:
    """Explainable anomaly event; severity is evidence classification, not probability."""

    anomaly_id: str
    anomaly_type: str
    severity: str
    evidence_count: int
    triggered_signals: Tuple[str, ...]
    affected_roads: Tuple[str, ...]
    affected_od_pairs: Tuple[DistributionKey, ...]
    explanation: str
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "evidence_count": self.evidence_count,
            "triggered_signals": list(self.triggered_signals),
            "affected_roads": list(self.affected_roads),
            "affected_od_pairs": [list(pair) for pair in self.affected_od_pairs],
            "explanation": self.explanation,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class AnomalyAnalysisResult:
    """Complete deterministic baseline/current mobility-state comparison."""

    baseline_snapshot_id: str
    current_snapshot_id: str
    road_evidence: Tuple[RoadAnomalyEvidence, ...]
    od_divergence: DistributionShift
    route_divergence: DistributionShift
    hhi_delta: float
    bottleneck_changes: Tuple[BottleneckChange, ...]
    anomaly_events: Tuple[AnomalyEvent, ...]
    spatial_regions: Tuple[SpatialAnomalyRegion, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_snapshot_id": self.baseline_snapshot_id,
            "current_snapshot_id": self.current_snapshot_id,
            "road_evidence": [e.__dict__ if hasattr(e, "__dict__") else {
                "road_id": e.road_id,
                "status": e.status.value,
                "baseline_hourly_flow": e.baseline_hourly_flow,
                "current_hourly_flow": e.current_hourly_flow,
                "absolute_change": e.absolute_change,
                "relative_change": e.relative_change,
                "flow_signal": e.flow_signal,
                "bottleneck_signal": e.bottleneck_signal,
                "evidence_count": e.evidence_count,
            } for e in self.road_evidence],
            "od_divergence": self.od_divergence.divergence,
            "route_divergence": self.route_divergence.divergence,
            "hhi_delta": self.hhi_delta,
            "bottleneck_changes": [change.__dict__ if hasattr(change, "__dict__") else {
                "road_id": change.road_id,
                "baseline_severity": change.baseline_severity,
                "current_severity": change.current_severity,
                "change_type": change.change_type,
            } for change in self.bottleneck_changes],
            "anomaly_events": [event.to_dict() for event in self.anomaly_events],
            "spatial_regions": [region.__dict__ if hasattr(region, "__dict__") else {
                "region_id": region.region_id,
                "road_ids": list(region.road_ids),
                "evidence_count": region.evidence_count,
                "severity": region.severity,
                "explanation": region.explanation,
            } for region in self.spatial_regions],
        }
