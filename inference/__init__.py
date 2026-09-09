"""
UrbanTrack AI Mobility Inference Engine module.
"""

from schemas.normalized_trajectory_schema import (
    InvalidRouteError,
    NormalizedCandidateRoute,
    NormalizedTrajectory,
    ProbabilityValidationError,
)
from schemas.observation_schema import Observation
from schemas.trajectory_schema import CandidateRoute, TrajectorySegment, VehicleTrajectory
from .identity_fusion import match_observations
from .identity_graph import IdentityGraph
from .member3_adapter import (
    adapt_trajectories_to_batch_payload,
    adapt_trajectory_segment_to_normalized,
    adapt_vehicle_trajectory_to_normalized,
)
from .observation_loader import load_camera_metadata, load_observations_from_json
from .road_graph import RoadEdge, RoadGraph, RoadNode
from .similarity import (
    appearance_similarity,
    geographic_distance,
    plate_similarity,
    time_difference,
    vehicle_type_compatibility,
)
from .spatial import spatial_feasibility
from .temporal import temporal_feasibility
from .trajectory_engine import (
    evaluate_route_feasibility_and_score,
    reconstruct_identity_trajectory,
    reconstruct_trajectory_segment,
)

__all__ = [
    "Observation",
    "CandidateRoute",
    "TrajectorySegment",
    "VehicleTrajectory",
    "NormalizedCandidateRoute",
    "NormalizedTrajectory",
    "ProbabilityValidationError",
    "InvalidRouteError",
    "adapt_trajectory_segment_to_normalized",
    "adapt_vehicle_trajectory_to_normalized",
    "adapt_trajectories_to_batch_payload",
    "load_camera_metadata",
    "load_observations_from_json",
    "plate_similarity",
    "appearance_similarity",
    "vehicle_type_compatibility",
    "time_difference",
    "geographic_distance",
    "temporal_feasibility",
    "spatial_feasibility",
    "match_observations",
    "IdentityGraph",
    "RoadNode",
    "RoadEdge",
    "RoadGraph",
    "evaluate_route_feasibility_and_score",
    "reconstruct_trajectory_segment",
    "reconstruct_identity_trajectory",
]

