"""
UrbanTrack AI — Layer 2 Multimodal Identity Association and Evidence Fusion.
"""

from layer2.association.association_engine import AssociationEngine
from layer2.association.decision_policy import DecisionPolicy, PolicyThresholds
from layer2.association.evidence_fusion import EvidenceFusionScorer, cosine_similarity, levenshtein_distance
from layer2.association.evidence_ledger import (
    AppearanceEvidence,
    AssociationDecision,
    CameraQualityEvidence,
    CandidateEvidenceLedger,
    MotionEvidence,
    OCREvidence,
    TemporalEvidence,
    TopologyEvidenceSummary,
    VehicleTypeEvidence,
)
from layer2.association.validation import AssociationValidator

__all__ = [
    "AppearanceEvidence",
    "AssociationDecision",
    "AssociationEngine",
    "AssociationValidator",
    "CameraQualityEvidence",
    "CandidateEvidenceLedger",
    "DecisionPolicy",
    "EvidenceFusionScorer",
    "MotionEvidence",
    "OCREvidence",
    "PolicyThresholds",
    "TemporalEvidence",
    "TopologyEvidenceSummary",
    "VehicleTypeEvidence",
    "cosine_similarity",
    "levenshtein_distance",
]
