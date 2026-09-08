"""Data models for UrbanTrackAI probabilistic trajectory and flow aggregation.

Provides normalized internal representations for probabilistic trajectories,
candidate routes, road flow aggregation results, and validation logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ProbabilityValidationError(ValueError):
    """Raised when candidate route probabilities fail validation (range or sum)."""
    pass


class InvalidRouteError(ValueError):
    """Raised when a candidate route has invalid structure or nodes."""
    pass


@dataclass(slots=True)
class CandidateRoute:
    """Represents a single candidate route traversed with an associated probability.

    Attributes:
        nodes: Ordered sequence of node IDs (at least 2 nodes).
        probability: Float in [0.0, 1.0] representing likelihood of this route.
        metadata: Optional dictionary for tracking source or confidence metadata.
    """

    nodes: List[str]
    probability: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate candidate route structure and probability value."""
        if not isinstance(self.nodes, list) or len(self.nodes) < 2:
            raise InvalidRouteError(
                f"Candidate route must contain at least 2 nodes, got: {self.nodes}"
            )

        for idx, node in enumerate(self.nodes):
            if not isinstance(node, str) or not node.strip():
                raise InvalidRouteError(
                    f"Candidate route node at index {idx} must be a non-empty string, got: {node!r}"
                )

        if not isinstance(self.probability, (int, float)) or math.isnan(self.probability) or math.isinf(self.probability):
            raise ProbabilityValidationError(
                f"Probability must be a finite float, got: {self.probability}"
            )

        self.probability = float(self.probability)

        if self.probability < 0.0 or self.probability > 1.0:
            raise ProbabilityValidationError(
                f"Candidate route probability must be between 0.0 and 1.0, got: {self.probability}"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize candidate route to dictionary."""
        return {
            "nodes": list(self.nodes),
            "probability": self.probability,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CandidateRoute:
        """Construct CandidateRoute from dictionary."""
        return cls(
            nodes=list(data["nodes"]),
            probability=float(data["probability"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class NormalizedTrajectory:
    """Internal normalized representation of a probabilistic vehicle trajectory.

    This representation decouples downstream traffic analytics from upstream
    detection/tracking implementation details (OCR, Re-ID, camera topology).

    Attributes:
        track_id: Unique identifier for the vehicle track or trip.
        origin_node: Starting junction node ID.
        destination_node: Ending junction node ID.
        candidate_routes: List of candidate routes with probabilistic weighting.
        vehicle_weight: Count or weight of vehicles represented (default: 1.0).
        timestamp: Optional ISO 8601 or reference timestamp.
        time_window_start: Optional start of time window.
        time_window_end: Optional end of time window.
        metadata: Additional metadata (e.g., vehicle class, confidence).
    """

    track_id: str
    origin_node: str
    destination_node: str
    candidate_routes: List[CandidateRoute]
    vehicle_weight: float = 1.0
    timestamp: Optional[str] = None
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate normalized trajectory attributes and candidate route distributions."""
        if not isinstance(self.track_id, str) or not self.track_id.strip():
            raise ValueError("track_id must be a non-empty string.")
        self.track_id = self.track_id.strip()

        if not isinstance(self.origin_node, str) or not self.origin_node.strip():
            raise ValueError("origin_node must be a non-empty string.")
        self.origin_node = self.origin_node.strip()

        if not isinstance(self.destination_node, str) or not self.destination_node.strip():
            raise ValueError("destination_node must be a non-empty string.")
        self.destination_node = self.destination_node.strip()

        if not isinstance(self.vehicle_weight, (int, float)) or math.isnan(self.vehicle_weight) or math.isinf(self.vehicle_weight):
            raise ValueError(f"vehicle_weight must be a finite float, got: {self.vehicle_weight}")
        self.vehicle_weight = float(self.vehicle_weight)
        if self.vehicle_weight <= 0.0:
            raise ValueError(f"vehicle_weight must be strictly positive (> 0.0), got: {self.vehicle_weight}")

        if not isinstance(self.candidate_routes, list) or len(self.candidate_routes) == 0:
            raise ValueError(f"Trajectory '{self.track_id}' must contain at least one candidate route.")

        # Ensure candidate route instances and validate endpoint consistency
        for idx, route in enumerate(self.candidate_routes):
            if not isinstance(route, CandidateRoute):
                raise TypeError(f"Candidate route at index {idx} must be a CandidateRoute instance, got {type(route)}")

            if route.nodes[0] != self.origin_node:
                raise InvalidRouteError(
                    f"Candidate route #{idx} start node '{route.nodes[0]}' does not match origin '{self.origin_node}'."
                )
            if route.nodes[-1] != self.destination_node:
                raise InvalidRouteError(
                    f"Candidate route #{idx} end node '{route.nodes[-1]}' does not match destination '{self.destination_node}'."
                )

        # Validate probability distribution sum
        self.validate_probabilities(tolerance=1e-6)

    def validate_probabilities(self, tolerance: float = 1e-6) -> None:
        """Validate that candidate route probabilities form a valid distribution summing to 1.0.

        Args:
            tolerance: Configurable numerical tolerance (default 1e-6).

        Raises:
            ProbabilityValidationError: If total probability deviates from 1.0 beyond tolerance.
        """
        total_prob = sum(route.probability for route in self.candidate_routes)
        if abs(total_prob - 1.0) > tolerance:
            raise ProbabilityValidationError(
                f"Trajectory '{self.track_id}' candidate route probabilities sum to {total_prob:.8f}, "
                f"expected approximately 1.0 (tolerance: {tolerance})."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalized trajectory to dictionary."""
        return {
            "track_id": self.track_id,
            "origin_node": self.origin_node,
            "destination_node": self.destination_node,
            "candidate_routes": [r.to_dict() for r in self.candidate_routes],
            "vehicle_weight": self.vehicle_weight,
            "timestamp": self.timestamp,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedTrajectory:
        """Construct NormalizedTrajectory from dictionary."""
        routes = [
            CandidateRoute.from_dict(r) if isinstance(r, dict) else r
            for r in data["candidate_routes"]
        ]
        return cls(
            track_id=data["track_id"],
            origin_node=data["origin_node"],
            destination_node=data["destination_node"],
            candidate_routes=routes,
            vehicle_weight=float(data.get("vehicle_weight", 1.0)),
            timestamp=data.get("timestamp"),
            time_window_start=data.get("time_window_start"),
            time_window_end=data.get("time_window_end"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class RoadFlow:
    """Represents aggregated expected traffic flow on a specific road segment.

    Attributes:
        road_id: Unique identifier for the road segment.
        from_node: Starting node of the directed road.
        to_node: Ending node of the directed road.
        expected_flow: Aggregated expected vehicle count.
        time_window_start: Optional start of analysis time window.
        time_window_end: Optional end of analysis time window.
        contributing_trajectories_count: Number of distinct trajectories that contributed flow.
        metadata: Optional additional metadata.
    """

    road_id: str
    from_node: str
    to_node: str
    expected_flow: float
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    contributing_trajectories_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate road flow attributes."""
        if not isinstance(self.road_id, str) or not self.road_id.strip():
            raise ValueError("road_id must be a non-empty string.")
        self.road_id = self.road_id.strip()

        if not isinstance(self.from_node, str) or not self.from_node.strip():
            raise ValueError("from_node must be a non-empty string.")
        self.from_node = self.from_node.strip()

        if not isinstance(self.to_node, str) or not self.to_node.strip():
            raise ValueError("to_node must be a non-empty string.")
        self.to_node = self.to_node.strip()

        if not isinstance(self.expected_flow, (int, float)) or math.isnan(self.expected_flow) or math.isinf(self.expected_flow):
            raise ValueError(f"expected_flow must be a finite float, got: {self.expected_flow}")
        self.expected_flow = float(self.expected_flow)
        if self.expected_flow < 0.0:
            raise ValueError(f"expected_flow must be non-negative (>= 0.0), got: {self.expected_flow}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize road flow to dictionary."""
        return {
            "road_id": self.road_id,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "expected_flow": round(self.expected_flow, 6),
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "contributing_trajectories_count": self.contributing_trajectories_count,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> RoadFlow:
        """Construct RoadFlow from dictionary."""
        return cls(
            road_id=data["road_id"],
            from_node=data["from_node"],
            to_node=data["to_node"],
            expected_flow=float(data["expected_flow"]),
            time_window_start=data.get("time_window_start"),
            time_window_end=data.get("time_window_end"),
            contributing_trajectories_count=int(data.get("contributing_trajectories_count", 0)),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(slots=True)
class FlowAggregationResult:
    """Container for the output of expected flow aggregation across network roads.

    Keeps references to both the aggregated road flows and the underlying
    normalized trajectories so Phase 3 (traffic volume, congestion metrics)
    and Phase 4 (OD analysis, bottlenecks) can directly consume what they need.
    """

    flows: Dict[str, RoadFlow]
    trajectories: List[NormalizedTrajectory] = field(default_factory=list)
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_flow(self, road_id: str) -> float:
        """Return expected flow for a given road ID, defaulting to 0.0 if not present."""
        road_flow = self.flows.get(road_id)
        return road_flow.expected_flow if road_flow else 0.0

    def get_road_flow(self, road_id: str) -> Optional[RoadFlow]:
        """Return RoadFlow object for a given road ID if present."""
        return self.flows.get(road_id)

    def all_flows(self, sorted_by_id: bool = True) -> List[RoadFlow]:
        """Return list of all road flow objects, optionally sorted by road ID."""
        flow_list = list(self.flows.values())
        if sorted_by_id:
            flow_list.sort(key=lambda rf: rf.road_id)
        return flow_list

    def to_dict(self) -> Dict[str, Any]:
        """Serialize aggregation result to dictionary."""
        return {
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "total_roads_with_flow": len([rf for rf in self.flows.values() if rf.expected_flow > 0]),
            "flows": {r_id: rf.to_dict() for r_id, rf in sorted(self.flows.items())},
            "trajectories_count": len(self.trajectories),
            "metadata": dict(self.metadata),
        }
