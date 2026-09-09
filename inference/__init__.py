"""
UrbanTrack AI Mobility Inference Engine module.
"""

from schemas.observation_schema import Observation
from schemas.trajectory_schema import CandidateRoute, TrajectorySegment, VehicleTrajectory
from .identity_fusion import match_observations
from .identity_graph import IdentityGraph
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

