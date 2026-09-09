"""
Data models and schemas for Day 3 Probabilistic Trajectory Reconstruction.
Defines CandidateRoute, TrajectorySegment, and VehicleTrajectory.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CandidateRoute:
    """
    A single candidate route hypothesis between two observation locations.
    """
    route_id: str
    edges: List[str]                  # Road IDs traversed, e.g. ["road_12", "road_18"]
    nodes: List[str]                  # Junction IDs traversed, e.g. ["junc_01", "junc_02"]
    distance_meters: float            # Total route road distance in meters
    estimated_travel_time_seconds: float # Free-flow travel time at expected road speeds
    min_travel_time_seconds: float    # Minimum physical travel time at maximum speed limits
    required_speed_kmh: float         # Speed needed to traverse this distance in observed delta_t
    speed_limit_kmh: float            # Bottleneck / representative speed limit on route
    feasible: bool                    # Feasibility flag based on physical & road constraints
    feasibility_status: str           # "feasible", "speed_exceeded", "temporal_inversion", etc.
    estimated_likelihood: float       # Relative estimated likelihood among candidates [0.0, 1.0]
    raw_score: float                  # Unnormalized composite feasibility/efficiency score
    explanation: str                  # Human-readable rationale

    @property
    def route(self) -> List[str]:
        return self.edges

    @property
    def distance_m(self) -> float:
        return self.distance_meters

    def to_dict(self) -> Dict[str, Any]:
        return {
            "route_id": self.route_id,
            "route": self.edges,
            "nodes": self.nodes,
            "distance_m": round(self.distance_meters, 1),
            "estimated_travel_time_s": round(self.estimated_travel_time_seconds, 1),
            "min_travel_time_s": round(self.min_travel_time_seconds, 1),
            "required_speed_kmh": round(self.required_speed_kmh, 1),
            "speed_limit_kmh": round(self.speed_limit_kmh, 1),
            "feasible": self.feasible,
            "feasibility_status": self.feasibility_status,
            "estimated_likelihood": round(self.estimated_likelihood, 4),
            "raw_score": round(self.raw_score, 4),
            "explanation": self.explanation,
        }


@dataclass
class TrajectorySegment:
    """
    Trajectory hypothesis between two consecutive observations of a vehicle identity.
    """
    segment_id: str
    identity_id: str
    start_observation_id: str
    start_camera_id: str
    start_timestamp: float
    end_observation_id: str
    end_camera_id: str
    end_timestamp: float
    time_difference_seconds: float
    start_node_id: Optional[str]
    end_node_id: Optional[str]
    candidate_routes: List[CandidateRoute] = field(default_factory=list)
    most_likely_route: Optional[List[str]] = None
    confidence: float = 0.0
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    status: str = "success"           # "success", "no_path", "unassociated_camera", "infeasible"

    @property
    def feasible(self) -> bool:
        return self.status == "success"

    @property
    def travel_distance_m(self) -> float:
        if self.candidate_routes and self.status == "success":
            return self.candidate_routes[0].distance_meters
        return 0.0

    @property
    def evidence_explanation(self) -> str:
        if self.ambiguity_reason:
            return self.ambiguity_reason
        if self.candidate_routes:
            return self.candidate_routes[0].explanation
        return f"Segment status: {self.status}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "identity_id": self.identity_id,
            "start_observation": {
                "observation_id": self.start_observation_id,
                "camera_id": self.start_camera_id,
                "timestamp_seconds": self.start_timestamp,
                "node_id": self.start_node_id,
            },
            "end_observation": {
                "observation_id": self.end_observation_id,
                "camera_id": self.end_camera_id,
                "timestamp_seconds": self.end_timestamp,
                "node_id": self.end_node_id,
            },
            "time_difference_seconds": round(self.time_difference_seconds, 2),
            "candidate_routes": [r.to_dict() for r in self.candidate_routes],
            "most_likely_route": self.most_likely_route,
            "confidence": round(self.confidence, 4),
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "status": self.status,
        }


@dataclass
class VehicleTrajectory:
    """
    Full multi-segment trajectory hypothesis across an entire sequence of vehicle observations.
    Follows Day 3 Output Contract schema.
    """
    identity_id: str
    observations_count: int
    cameras_visited: List[str]
    start_timestamp: float
    end_timestamp: float
    total_time_seconds: float
    total_distance_meters: float
    overall_confidence: float
    is_ambiguous: bool
    segments: List[TrajectorySegment] = field(default_factory=list)
    complete_route_edges: List[str] = field(default_factory=list)
    complete_route_nodes: List[str] = field(default_factory=list)

    @property
    def feasible(self) -> bool:
        if not self.segments:
            return True
        return all(s.feasible for s in self.segments)

    @property
    def full_route(self) -> List[str]:
        return self.complete_route_edges

    @property
    def total_distance_m(self) -> float:
        return self.total_distance_meters

    @property
    def total_duration_s(self) -> float:
        return self.total_time_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "observations_count": self.observations_count,
            "cameras_visited": self.cameras_visited,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "total_time_seconds": round(self.total_time_seconds, 1),
            "total_distance_m": round(self.total_distance_meters, 1),
            "overall_confidence": round(self.overall_confidence, 4),
            "is_ambiguous": self.is_ambiguous,
            "most_likely_route": self.complete_route_edges,
            "complete_route_nodes": self.complete_route_nodes,
            "segments": [s.to_dict() for s in self.segments],
        }
