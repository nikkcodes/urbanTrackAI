"""UrbanTrackAI Flow Aggregation Package (Member 3 Phase 2).

Provides normalized probabilistic trajectory representations, adapters,
and the expected road flow aggregation engine.
"""

from backend.flow.adapters import BaseTrajectoryAdapter, MockTrajectoryAdapter
from backend.flow.aggregation import ExpectedFlowAggregator
from backend.flow.models import (
    CandidateRoute,
    FlowAggregationResult,
    InvalidRouteError,
    NormalizedTrajectory,
    ProbabilityValidationError,
    RoadFlow,
)

__all__ = [
    "CandidateRoute",
    "NormalizedTrajectory",
    "RoadFlow",
    "FlowAggregationResult",
    "ProbabilityValidationError",
    "InvalidRouteError",
    "BaseTrajectoryAdapter",
    "MockTrajectoryAdapter",
    "ExpectedFlowAggregator",
]
