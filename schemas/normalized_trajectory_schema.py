"""
NormalizedTrajectory Schema for Member 3 Integration.

Defines the integration contract between Member 2 (Mobility Inference Engine)
and Member 3 (Mobility / Decision Intelligence & Flow Aggregation).

Interface Contract:
NormalizedTrajectory
├── track_id            (str: unique vehicle identity or track identifier)
├── origin_node         (str: starting junction node ID)
├── destination_node    (str: ending junction node ID)
├── vehicle_weight      (float: vehicle count / demand multiplier, default: 1.0)
├── candidate_routes[]
│   ├── nodes           (list[str]: ordered junction node IDs, length >= 2)
│   ├── probability     (float: normalized relative likelihood in [0.0, 1.0])
│   └── metadata        (dict: route details such as distance, speed limit)
├── time_window_start   (str | float: start timestamp)
├── time_window_end     (str | float: end timestamp)
├── timestamp           (str | float: reference timestamp)
└── metadata            (dict: vehicle type, confidence, ambiguity flag)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


class ProbabilityValidationError(ValueError):
    """Raised when candidate route probabilities fail validation (range or sum)."""
    pass


class InvalidRouteError(ValueError):
    """Raised when a candidate route has invalid structure or node sequence."""
    pass


@dataclass
class NormalizedCandidateRoute:
    """
    Represents a single candidate route traversed with an associated probability.

    Attributes:
        nodes: Ordered sequence of junction node IDs (at least 2 nodes).
        probability: Float in [0.0, 1.0] representing normalized relative route likelihood.
        metadata: Optional dictionary containing road distances, speeds, edge IDs, etc.
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
            "probability": round(self.probability, 4),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedCandidateRoute:
        """Construct NormalizedCandidateRoute from dictionary."""
        nodes = data.get("nodes") or data.get("path")
        if nodes is None:
            raise KeyError("Candidate route missing required 'nodes' or 'path' field.")
        return cls(
            nodes=list(nodes),
            probability=float(data["probability"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class NormalizedTrajectory:
    """
    Normalized representation of a probabilistic vehicle trajectory for Member 3.

    This representation provides the standardized integration contract required by
    Member 3's downstream mobility graph, flow aggregation (W * P), traffic analytics,
    and decision intelligence layer.

    Attributes:
        track_id: Unique vehicle identity or track identifier (e.g., 'VEHICLE_CANDIDATE_001').
        origin_node: Starting junction node ID in the road network.
        destination_node: Ending junction node ID in the road network.
        candidate_routes: List of candidate routes with probabilistic weighting.
        vehicle_weight: Demand multiplier / vehicle count represented (default: 1.0).
        timestamp: Optional reference timestamp (float seconds or ISO string).
        time_window_start: Start of observation time window.
        time_window_end: End of observation time window.
        metadata: Trajectory metadata (e.g., vehicle_type, overall_confidence, is_ambiguous).
    """
    track_id: str
    origin_node: str
    destination_node: str
    candidate_routes: List[NormalizedCandidateRoute]
    vehicle_weight: float = 1.0
    timestamp: Optional[Any] = None
    time_window_start: Optional[Any] = None
    time_window_end: Optional[Any] = None
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
        if self.vehicle_weight < 0.0:
            raise ValueError(f"vehicle_weight must be non-negative (>= 0.0), got: {self.vehicle_weight}")

        if not isinstance(self.candidate_routes, list) or len(self.candidate_routes) == 0:
            raise ValueError(f"Trajectory '{self.track_id}' must contain at least one candidate route.")

        # Ensure candidate route instances and validate endpoint consistency
        for idx, route in enumerate(self.candidate_routes):
            if not isinstance(route, NormalizedCandidateRoute):
                raise TypeError(
                    f"Candidate route at index {idx} must be a NormalizedCandidateRoute instance, got {type(route)}"
                )

            if route.nodes[0] != self.origin_node:
                raise InvalidRouteError(
                    f"Candidate route #{idx} start node '{route.nodes[0]}' does not match origin '{self.origin_node}'."
                )
            if route.nodes[-1] != self.destination_node:
                raise InvalidRouteError(
                    f"Candidate route #{idx} end node '{route.nodes[-1]}' does not match destination '{self.destination_node}'."
                )

        # Validate probability distribution sum
        self.validate_probabilities(tolerance=1e-4)

    def validate_probabilities(self, tolerance: float = 1e-4) -> None:
        """
        Validate that candidate route probabilities sum to approximately 1.0.

        Args:
            tolerance: Numerical tolerance for floating-point sum (default 1e-4).

        Raises:
            ProbabilityValidationError: If total probability deviates from 1.0 beyond tolerance.
        """
        total_prob = sum(route.probability for route in self.candidate_routes)
        if abs(total_prob - 1.0) > tolerance:
            raise ProbabilityValidationError(
                f"Trajectory '{self.track_id}' candidate route probabilities sum to {total_prob:.6f}, "
                f"expected approximately 1.0 (tolerance: {tolerance})."
            )

    def calculate_route_demands(self) -> List[Dict[str, Any]]:
        """
        Compute expected demand contributions for each candidate route.
        Formula: route demand = vehicle_weight * route_probability.

        Returns:
            List[Dict[str, Any]]: Candidate routes with calculated route_demand.
        """
        demands = []
        for r in self.candidate_routes:
            demands.append({
                "nodes": list(r.nodes),
                "probability": r.probability,
                "vehicle_weight": self.vehicle_weight,
                "route_demand": round(self.vehicle_weight * r.probability, 4),
            })
        return demands

    def to_dict(self) -> Dict[str, Any]:
        """Serialize normalized trajectory to Member 3 integration dictionary format."""
        return {
            "track_id": self.track_id,
            "origin_node": self.origin_node,
            "destination_node": self.destination_node,
            "vehicle_weight": round(self.vehicle_weight, 2),
            "candidate_routes": [r.to_dict() for r in self.candidate_routes],
            "time_window": {
                "start": self.time_window_start,
                "end": self.time_window_end,
            },
            "timestamp": self.timestamp,
            "time_window_start": self.time_window_start,
            "time_window_end": self.time_window_end,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> NormalizedTrajectory:
        """Construct NormalizedTrajectory from dictionary."""
        routes = [
            NormalizedCandidateRoute.from_dict(r) if isinstance(r, dict) else r
            for r in data["candidate_routes"]
        ]
        time_win = data.get("time_window", {})
        t_start = data.get("time_window_start", time_win.get("start"))
        t_end = data.get("time_window_end", time_win.get("end"))

        return cls(
            track_id=str(data["track_id"]),
            origin_node=str(data["origin_node"]),
            destination_node=str(data["destination_node"]),
            candidate_routes=routes,
            vehicle_weight=float(data.get("vehicle_weight", 1.0)),
            timestamp=data.get("timestamp"),
            time_window_start=t_start,
            time_window_end=t_end,
            metadata=dict(data.get("metadata", {})),
        )
