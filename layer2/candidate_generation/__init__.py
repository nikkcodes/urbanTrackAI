"""
UrbanTrack AI — Layer 2 Candidate Generation Package.

Provides modular candidate generation and gating components for multi-camera tracking:
- CandidateGenerator
- TemporalGate
- SpatialGate
- TopologyGate
"""

from layer2.candidate_generation.candidate_generator import (
    CandidateGenerator,
    CandidatePair,
    RejectionRecord,
)
from layer2.candidate_generation.spatial_gate import (
    SpatialFeasibilityResult,
    SpatialGate,
    haversine_distance_m,
)
from layer2.candidate_generation.temporal_gate import (
    ChronologyEvaluation,
    TemporalGate,
)
from layer2.candidate_generation.topology_gate import (
    TopologyEvidence,
    TopologyGate,
)

__all__ = [
    "CandidateGenerator",
    "CandidatePair",
    "RejectionRecord",
    "TemporalGate",
    "ChronologyEvaluation",
    "SpatialGate",
    "SpatialFeasibilityResult",
    "haversine_distance_m",
    "TopologyGate",
    "TopologyEvidence",
]
