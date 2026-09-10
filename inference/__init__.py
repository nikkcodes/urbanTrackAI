"""
UrbanTrack AI Mobility Inference Engine module.
"""

from schemas.gap_schema import SparseObservationGap
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
    adapt_sparse_gap_to_normalized,
    adapt_trajectories_to_batch_payload,
    adapt_trajectory_segment_to_normalized,
    adapt_vehicle_trajectory_to_normalized,
)
from .observation_loader import load_camera_metadata, load_observations_from_json
from schemas.reliability_schema import (
    CameraReliability,
    IdentityMatchReliability,
    ObservationReliability,
    TrajectoryReliability,
)
from .reliability_engine import (
    evaluate_camera_reliability,
    evaluate_identity_uncertainty,
    evaluate_observation_reliability,
    propagate_trajectory_uncertainty,
)
from .road_graph import RoadEdge, RoadGraph, RoadNode
from .similarity import (
    appearance_similarity,
    geographic_distance,
    plate_similarity,
    time_difference,
    vehicle_type_compatibility,
)
from .spatial import spatial_feasibility
from .sparse_engine import (
    detect_observation_gaps,
    infer_sparse_gap,
    infer_sparse_identity_trajectory,
)
from .temporal import temporal_feasibility
from .trajectory_engine import (
    evaluate_route_feasibility_and_score,
    reconstruct_identity_trajectory,
    reconstruct_trajectory_segment,
)

from .inference_trace import (
    InferenceTrace,
    build_cluster_trace,
    build_global_trajectory_trace,
    build_identity_pair_trace,
    build_sparse_gap_trace,
    build_trajectory_segment_trace,
)

__all__ = [
    "Observation",
    "CandidateRoute",
    "TrajectorySegment",
    "VehicleTrajectory",
    "SparseObservationGap",
    "NormalizedCandidateRoute",
    "NormalizedTrajectory",
    "ProbabilityValidationError",
    "InvalidRouteError",
    "CameraReliability",
    "ObservationReliability",
    "IdentityMatchReliability",
    "TrajectoryReliability",
    "evaluate_camera_reliability",
    "evaluate_observation_reliability",
    "evaluate_identity_uncertainty",
    "propagate_trajectory_uncertainty",
    "adapt_trajectory_segment_to_normalized",
    "adapt_vehicle_trajectory_to_normalized",
    "adapt_sparse_gap_to_normalized",
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
    "detect_observation_gaps",
    "infer_sparse_gap",
    "infer_sparse_identity_trajectory",
    "InferenceTrace",
    "build_identity_pair_trace",
    "build_cluster_trace",
    "build_trajectory_segment_trace",
    "build_sparse_gap_trace",
    "build_global_trajectory_trace",
]




