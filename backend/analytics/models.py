"""Validated result models for UrbanTrackAI Phase 4 analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime, time
from typing import Any, Dict, Iterable, List, Optional, Tuple


TimeWindow = Tuple[Optional[str], Optional[str]]


def _validate_label(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_number(value: float, field_name: str, non_negative: bool = True) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{field_name} must be a finite number, got: {value}")
    result = float(value)
    if non_negative and result < 0.0:
        raise ValueError(f"{field_name} must be non-negative, got: {result}")
    return result


def validate_time_window(start: Optional[str], end: Optional[str]) -> TimeWindow:
    """Validate the existing Phase 2/3 ISO time-only or ISO datetime convention."""
    if start is None and end is None:
        return None, None
    if start is None or end is None:
        raise ValueError("time_window_start and time_window_end must be provided together")
    if not isinstance(start, str) or not isinstance(end, str) or not start.strip() or not end.strip():
        raise ValueError("time-window boundaries must be non-empty strings")

    def parse(value: str, field_name: str) -> datetime | time:
        try:
            return datetime.fromisoformat(value.strip())
        except ValueError:
            try:
                return time.fromisoformat(value.strip())
            except ValueError as exc:
                raise ValueError(f"{field_name} is not a valid ISO time value: {value!r}") from exc

    parsed_start = parse(start, "time_window_start")
    parsed_end = parse(end, "time_window_end")
    if type(parsed_start) is not type(parsed_end):
        raise ValueError("time-window boundaries must use the same date-time format")
    if isinstance(parsed_start, time):
        parsed_start = datetime.combine(date(2000, 1, 1), parsed_start)
        parsed_end = datetime.combine(date(2000, 1, 1), parsed_end)  # type: ignore[arg-type]
    try:
        duration_seconds = (parsed_end - parsed_start).total_seconds()  # type: ignore[operator]
    except TypeError as exc:
        raise ValueError("time-window boundaries must have compatible timezone information") from exc
    if duration_seconds <= 0.0:
        raise ValueError("time_window_end must be after time_window_start")
    return start.strip(), end.strip()


@dataclass(frozen=True, slots=True)
class ODPairDemand:
    """Accumulated expected demand for one origin-destination pair and window."""

    origin: str
    destination: str
    demand: float
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin", _validate_label(self.origin, "origin"))
        object.__setattr__(self, "destination", _validate_label(self.destination, "destination"))
        object.__setattr__(self, "demand", _validate_number(self.demand, "demand"))
        start, end = validate_time_window(self.time_window_start, self.time_window_end)
        object.__setattr__(self, "time_window_start", start)
        object.__setattr__(self, "time_window_end", end)


@dataclass(frozen=True, slots=True)
class RouteDemand:
    """Expected demand assigned to one probabilistic candidate route."""

    route_nodes: Tuple[str, ...]
    road_ids: Tuple[str, ...]
    demand: float
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None

    def __post_init__(self) -> None:
        if len(self.route_nodes) < 2 or any(not isinstance(node, str) or not node.strip() for node in self.route_nodes):
            raise ValueError("route_nodes must contain at least two non-empty node IDs")
        if len(self.road_ids) != len(self.route_nodes) - 1:
            raise ValueError("road_ids must contain one road ID per route edge")
        if any(not isinstance(road_id, str) or not road_id.strip() for road_id in self.road_ids):
            raise ValueError("road_ids must contain non-empty road IDs")
        object.__setattr__(self, "route_nodes", tuple(node.strip() for node in self.route_nodes))
        object.__setattr__(self, "road_ids", tuple(road_id.strip() for road_id in self.road_ids))
        object.__setattr__(self, "demand", _validate_number(self.demand, "demand"))
        start, end = validate_time_window(self.time_window_start, self.time_window_end)
        object.__setattr__(self, "time_window_start", start)
        object.__setattr__(self, "time_window_end", end)


@dataclass(frozen=True, slots=True)
class ODMatrix:
    """A collection of window-preserving accumulated OD demand records."""

    pairs: Tuple[ODPairDemand, ...] = ()

    def __post_init__(self) -> None:
        if any(not isinstance(pair, ODPairDemand) for pair in self.pairs):
            raise TypeError("pairs must contain only ODPairDemand instances")

    @property
    def total_demand(self) -> float:
        """Return accumulated demand across all retained windows."""
        return sum(pair.demand for pair in self.pairs)

    @property
    def origins(self) -> Tuple[str, ...]:
        """Return sorted unique origin IDs."""
        return tuple(sorted({pair.origin for pair in self.pairs}))

    @property
    def destinations(self) -> Tuple[str, ...]:
        """Return sorted unique destination IDs."""
        return tuple(sorted({pair.destination for pair in self.pairs}))

    @property
    def windows(self) -> Tuple[TimeWindow, ...]:
        """Return sorted unique time windows represented by the matrix."""
        return tuple(sorted({(pair.time_window_start, pair.time_window_end) for pair in self.pairs}, key=lambda item: (item[0] or "", item[1] or "")))

    def get_demand(
        self,
        origin: str,
        destination: str,
        time_window: Optional[TimeWindow] = None,
    ) -> float:
        """Return demand, requiring a window when the matrix contains multiple windows."""
        origin = _validate_label(origin, "origin")
        destination = _validate_label(destination, "destination")
        if time_window is None and len(self.windows) > 1:
            raise ValueError("time_window is required when ODMatrix contains multiple windows")
        selected_window = None if time_window is None else validate_time_window(*time_window)
        return sum(
            pair.demand
            for pair in self.pairs
            if pair.origin == origin
            and pair.destination == destination
            and (selected_window is None or (pair.time_window_start, pair.time_window_end) == selected_window)
        )

    def top_pairs(self, limit: Optional[int] = None, time_window: Optional[TimeWindow] = None) -> Tuple[ODPairDemand, ...]:
        """Return demand records sorted by descending demand, then stable IDs."""
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise ValueError("limit must be a non-negative integer or None")
        selected_window = None if time_window is None else validate_time_window(*time_window)
        records = [
            pair for pair in self.pairs
            if selected_window is None or (pair.time_window_start, pair.time_window_end) == selected_window
        ]
        records.sort(key=lambda pair: (-pair.demand, pair.origin, pair.destination, pair.time_window_start or "", pair.time_window_end or ""))
        return tuple(records if limit is None else records[:limit])


@dataclass(frozen=True, slots=True)
class ODAnalysisResult:
    """OD analysis output retaining matrix, rankings, and trajectory count."""

    matrix: ODMatrix
    total_demand: float
    top_od_pairs: Tuple[ODPairDemand, ...]
    trajectory_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, ODMatrix):
            raise TypeError("matrix must be an ODMatrix")
        object.__setattr__(self, "total_demand", _validate_number(self.total_demand, "total_demand"))
        if not isinstance(self.trajectory_count, int) or self.trajectory_count < 0:
            raise ValueError("trajectory_count must be a non-negative integer")
        if any(not isinstance(pair, ODPairDemand) for pair in self.top_od_pairs):
            raise TypeError("top_od_pairs must contain only ODPairDemand instances")


@dataclass(frozen=True, slots=True)
class Bottleneck:
    """Deterministic bottleneck ranking record derived from a TrafficMetric."""

    road_id: str
    severity: str
    utilization_ratio: float
    congestion_score: float
    hourly_flow: float
    capacity_vph: float
    delay_minutes: float
    bottleneck_score: float
    signals: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "road_id", _validate_label(self.road_id, "road_id"))
        object.__setattr__(self, "severity", _validate_label(self.severity, "severity"))
        for name in ("utilization_ratio", "congestion_score", "hourly_flow", "capacity_vph", "delay_minutes", "bottleneck_score"):
            object.__setattr__(self, name, _validate_number(getattr(self, name), name))
        if self.capacity_vph <= 0.0:
            raise ValueError("capacity_vph must be positive")
        if any(not isinstance(signal, str) or not signal.strip() for signal in self.signals):
            raise ValueError("signals must contain non-empty strings")
        object.__setattr__(self, "signals", tuple(signal.strip() for signal in self.signals))


@dataclass(frozen=True, slots=True)
class RoadPriority:
    """Road ranking record exposing every component of the priority score."""

    road_id: str
    priority_score: float
    hourly_flow: float
    flow_share: float
    utilization_ratio: float
    congestion_score: float
    delay_ratio: float
    structural_importance: float


@dataclass(frozen=True, slots=True)
class NetworkIntelligence:
    """Descriptive hourly-flow and topology analytics for a mobility graph."""

    total_hourly_flow: float
    flow_concentration_hhi: float
    road_flow_shares: Dict[str, float] = field(default_factory=dict)
    structural_importance: Dict[str, float] = field(default_factory=dict)
    road_count: int = 0
    active_roads_with_flow: int = 0
    evaluated_roads_count: int = 0


@dataclass(frozen=True, slots=True)
class UrbanMobilityAnalysisResult:
    """Reusable aggregate of Phase 4 OD, route, bottleneck, and network results."""

    od_analysis: ODAnalysisResult
    route_demands: Tuple[RouteDemand, ...]
    bottlenecks: Tuple[Bottleneck, ...]
    network: NetworkIntelligence
    top_priority_roads: Tuple[RoadPriority, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.od_analysis, ODAnalysisResult):
            raise TypeError("od_analysis must be an ODAnalysisResult")
        if not isinstance(self.network, NetworkIntelligence):
            raise TypeError("network must be a NetworkIntelligence")
        if any(not isinstance(item, RouteDemand) for item in self.route_demands):
            raise TypeError("route_demands must contain only RouteDemand instances")
        if any(not isinstance(item, Bottleneck) for item in self.bottlenecks):
            raise TypeError("bottlenecks must contain only Bottleneck instances")
        if any(not isinstance(item, RoadPriority) for item in self.top_priority_roads):
            raise TypeError("top_priority_roads must contain only RoadPriority instances")
