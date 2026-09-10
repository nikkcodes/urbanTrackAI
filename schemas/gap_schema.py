"""
Data model for Day 4 Sparse / Missing-Camera Trajectory Inference.
Represents an unobserved interval (gap) between two vehicle sightings
belonging to the same cross-camera vehicle identity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from schemas.trajectory_schema import CandidateRoute


@dataclass
class SparseObservationGap:
    """
    Explicit representation of an unobserved movement interval (gap) between two sightings.

    Represents:
    - Observed endpoints (Observation A and Observation B)
    - Gap duration (elapsed time)
    - Candidate hidden routes through the road network
    - Route feasibility & relative estimated likelihoods
    - Ambiguity & explainability metadata

    CRITICAL: Does NOT fabricate synthetic observation records for unobserved cameras.
    """
    gap_id: str
    identity_id: str
    start_observation: Dict[str, Any]
    end_observation: Dict[str, Any]
    gap_duration_seconds: Optional[float] = None
    start_node_id: Optional[str] = None
    end_node_id: Optional[str] = None
    candidate_routes: List[CandidateRoute] = field(default_factory=list)
    confidence: float = 0.0
    is_ambiguous: bool = False
    ambiguity_reason: Optional[str] = None
    status: str = "success"  # "success", "infeasible", "no_path", "unassociated_camera", "temporal_inversion"
    gap_state: str = "observed_endpoints_with_unobserved_interval"
    unobserved_intermediate_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    reliability: Optional[Dict[str, Any]] = None # Day 5: Gap reliability evaluation
    uncertainty: Optional[Dict[str, Any]] = None # Day 5: Gap uncertainty evaluation

    @property
    def feasible(self) -> bool:
        """A gap transition is feasible if at least one candidate hidden route is feasible."""
        return self.status == "success" and any(r.feasible for r in self.candidate_routes)

    @property
    def most_likely_route(self) -> Optional[List[str]]:
        """Road edge sequence of the top candidate hidden route, if feasible."""
        if self.candidate_routes and self.candidate_routes[0].feasible:
            return self.candidate_routes[0].edges
        return None

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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize gap representation to a dictionary."""
        d: Dict[str, Any] = {
            "gap_id": self.gap_id,
            "identity_id": self.identity_id,
            "start_observation": dict(self.start_observation),
            "end_observation": dict(self.end_observation),
            "gap_duration_seconds": round(self.gap_duration_seconds, 2) if self.gap_duration_seconds is not None else None,
            "start_node_id": self.start_node_id,
            "end_node_id": self.end_node_id,
            "gap_state": self.gap_state,
            "status": self.status,
            "feasible": self.feasible,
            "confidence": round(self.confidence, 4),
            "is_ambiguous": self.is_ambiguous,
            "ambiguity_reason": self.ambiguity_reason,
            "unobserved_intermediate_nodes": list(self.unobserved_intermediate_nodes),
            "candidate_routes": [r.to_dict() for r in self.candidate_routes],
            "metadata": dict(self.metadata),
        }
        if self.reliability is not None:
            d["reliability"] = self.reliability
        if self.uncertainty is not None:
            d["uncertainty"] = self.uncertainty
        return d

