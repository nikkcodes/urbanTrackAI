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
    required_speed_kmh: Optional[float] = None # Speed needed to traverse this distance in observed delta_t
    speed_limit_kmh: float = 50.0            # Bottleneck / representative speed limit on route
    feasible: bool = True                    # Feasibility flag based on physical & road constraints
    feasibility_status: str = "feasible"     # "feasible", "speed_exceeded", "temporal_inversion", etc.
    estimated_likelihood: float = 0.0        # Relative estimated likelihood among candidates [0.0, 1.0]
    raw_score: float = 0.0                   # Unnormalized composite feasibility/efficiency score
    explanation: str = ""                    # Human-readable rationale
    reliability: Optional[float] = None      # Day 5: Route reliability score in [0.0, 1.0]
    uncertainty: Optional[float] = None      # Day 5: Route uncertainty score in [0.0, 1.0]
    temporal_evidence: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def route(self) -> List[str]:
        return self.edges

    @property
    def distance_m(self) -> float:
        return self.distance_meters

    @property
    def unobserved_intermediate_nodes(self) -> List[str]:
        """Intermediate junctions traversed between origin (nodes[0]) and destination (nodes[-1])."""
        if len(self.nodes) > 2:
            return list(self.nodes[1:-1])
        return []

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "route_id": self.route_id,
            "route": self.edges,
            "nodes": self.nodes,
            "unobserved_intermediate_nodes": self.unobserved_intermediate_nodes,
            "distance_m": round(self.distance_meters, 1),
            "estimated_travel_time_s": round(self.estimated_travel_time_seconds, 1),
            "min_travel_time_s": round(self.min_travel_time_seconds, 1),
            "required_speed_kmh": round(self.required_speed_kmh, 1) if self.required_speed_kmh is not None else None,
            "speed_limit_kmh": round(self.speed_limit_kmh, 1),
            "feasible": self.feasible,
            "feasibility_status": self.feasibility_status,
            "estimated_likelihood": round(self.estimated_likelihood, 4),
            "raw_score": round(self.raw_score, 4),
            "explanation": self.explanation,
        }
        if self.reliability is not None:
            d["reliability"] = round(self.reliability, 4)
        if self.uncertainty is not None:
            d["uncertainty"] = round(self.uncertainty, 4)
        if self.temporal_evidence is not None:
            d["temporal_evidence"] = self.temporal_evidence
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d



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
    time_difference_seconds: Optional[float] = None
    start_node_id: Optional[str] = None
    end_node_id: Optional[str] = None
    candidate_routes: List[CandidateRoute] = field(default_factory=list)
    most_likely_route: Optional[List[str]] = None
    confidence: float = 0.0
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    status: str = "success"           # "success", "no_path", "unassociated_camera", "infeasible"
    is_gap: bool = False
    gap_duration_seconds: Optional[float] = None
    gap_state: Optional[str] = None
    reliability: Optional[Dict[str, Any]] = None # Day 5: Trajectory reliability evaluation
    uncertainty: Optional[Dict[str, Any]] = None # Day 5: Trajectory uncertainty evaluation
    temporal_evidence: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        return self.status == "success"

    @property
    def travel_distance_m(self) -> float:
        if self.candidate_routes and self.status == "success":
            return self.candidate_routes[0].distance_meters
        return 0.0

    @property
    def most_likely_route_edges(self) -> Optional[List[str]]:
        return self.most_likely_route

    @property
    def most_likely_route_nodes(self) -> List[str]:
        if self.candidate_routes and self.candidate_routes[0].nodes:
            return list(self.candidate_routes[0].nodes)
        return []

    @property
    def unobserved_intermediate_nodes(self) -> List[str]:
        """Union of intermediate nodes across all feasible candidate routes."""
        nodes_set = set()
        for r in self.candidate_routes:
            if r.feasible and len(r.nodes) > 2:
                nodes_set.update(r.nodes[1:-1])
        return sorted(list(nodes_set))

    @property
    def most_likely_intermediate_nodes(self) -> List[str]:
        """Intermediate junctions traversed along the top/most-likely candidate route."""
        if self.candidate_routes and self.candidate_routes[0].feasible and len(self.candidate_routes[0].nodes) > 2:
            return list(self.candidate_routes[0].nodes[1:-1])
        return []

    @property
    def common_intermediate_nodes(self) -> List[str]:
        """Intermediate junctions shared across all feasible candidate routes."""
        feasible = [r for r in self.candidate_routes if r.feasible]
        if not feasible:
            return []
        common = set(feasible[0].nodes[1:-1])
        for r in feasible[1:]:
            common.intersection_update(r.nodes[1:-1])
        return sorted(list(common))

    @property
    def evidence_explanation(self) -> str:
        if self.ambiguity_reason:
            return self.ambiguity_reason
        if self.candidate_routes:
            return self.candidate_routes[0].explanation
        return f"Segment status: {self.status}"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
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
            "time_difference_seconds": round(self.time_difference_seconds, 2) if self.time_difference_seconds is not None else None,
            "candidate_routes": [r.to_dict() for r in self.candidate_routes],
            "most_likely_route": self.most_likely_route,
            "confidence": round(self.confidence, 4),
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "status": self.status,
            "is_gap": self.is_gap,
            "gap_duration_seconds": round(self.gap_duration_seconds, 2) if self.gap_duration_seconds is not None else None,
            "gap_state": self.gap_state,
        }
        if self.is_gap:
            d["unobserved_intermediate_nodes"] = self.unobserved_intermediate_nodes
            d["most_likely_intermediate_nodes"] = self.most_likely_intermediate_nodes
            d["common_intermediate_nodes"] = self.common_intermediate_nodes
            d["intermediate_node_semantics"] = "union_of_feasible_candidate_routes"
        if self.reliability is not None:
            d["reliability"] = self.reliability
        if self.uncertainty is not None:
            d["uncertainty"] = self.uncertainty
        if self.temporal_evidence is not None:
            d["temporal_evidence"] = self.temporal_evidence
        if self.metadata:
            d["metadata"] = dict(self.metadata)
        return d


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
    total_time_seconds: Optional[float] = None
    total_distance_meters: float = 0.0
    overall_confidence: float = 0.0
    is_ambiguous: bool = False
    segments: List[TrajectorySegment] = field(default_factory=list)
    complete_route_edges: List[str] = field(default_factory=list)
    complete_route_nodes: List[str] = field(default_factory=list)
    gaps_count: int = 0
    reliability: Optional[Dict[str, Any]] = None # Day 5: Trajectory reliability evaluation
    uncertainty: Optional[Dict[str, Any]] = None # Day 5: Trajectory uncertainty evaluation

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
    def total_duration_s(self) -> Optional[float]:
        return self.total_time_seconds

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "identity_id": self.identity_id,
            "observations_count": self.observations_count,
            "cameras_visited": self.cameras_visited,
            "start_timestamp": self.start_timestamp,
            "end_timestamp": self.end_timestamp,
            "total_time_seconds": round(self.total_time_seconds, 1) if self.total_time_seconds is not None else None,
            "total_distance_m": round(self.total_distance_meters, 1),
            "overall_confidence": round(self.overall_confidence, 4),
            "is_ambiguous": self.is_ambiguous,
            "most_likely_route": self.complete_route_edges,
            "complete_route_nodes": self.complete_route_nodes,
            "segments": [s.to_dict() for s in self.segments],
            "gaps_count": self.gaps_count,
        }
        if self.reliability is not None:
            d["reliability"] = self.reliability
        if self.uncertainty is not None:
            d["uncertainty"] = self.uncertainty
        return d

