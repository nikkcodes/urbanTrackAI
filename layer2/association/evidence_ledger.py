"""
UrbanTrack AI — Layer 2 Identity Association: Evidence Ledger.

Defines canonical evidence structures, modality records, and machine-readable
ledger schemas for cross-camera vehicle identity association.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AssociationDecision(str, Enum):
    CONFIRMED = "CONFIRMED"
    AMBIGUOUS = "AMBIGUOUS"
    REJECTED = "REJECTED"


@dataclass
class AppearanceEvidence:
    """Appearance / Re-ID evidence evaluation."""
    origin_model: Optional[str]
    destination_model: Optional[str]
    models_compatible: bool
    origin_has_embedding: bool
    destination_has_embedding: bool
    cosine_similarity: Optional[float]
    normalized_score: Optional[float]
    status: str  # "COMPATIBLE", "INCOMPATIBLE_SPACES", "MISSING_ORIGIN", "MISSING_DESTINATION", "MISSING_BOTH"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_model": self.origin_model,
            "destination_model": self.destination_model,
            "models_compatible": self.models_compatible,
            "origin_has_embedding": self.origin_has_embedding,
            "destination_has_embedding": self.destination_has_embedding,
            "cosine_similarity": round(self.cosine_similarity, 4) if self.cosine_similarity is not None else None,
            "normalized_score": round(self.normalized_score, 4) if self.normalized_score is not None else None,
            "status": self.status,
        }


@dataclass
class TemporalEvidence:
    """Temporal transit consistency evaluation."""
    origin_start_sync: float
    origin_end_sync: float
    destination_start_sync: float
    destination_end_sync: float
    delta_t_seconds: float
    is_overlapping: bool
    overlap_duration_seconds: float
    regime: str  # "OVERLAPPING", "BOUNDARY_HANDOFF", "SEQUENTIAL_TRANSIT"
    normalized_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_start_sync": round(self.origin_start_sync, 4),
            "origin_end_sync": round(self.origin_end_sync, 4),
            "destination_start_sync": round(self.destination_start_sync, 4),
            "destination_end_sync": round(self.destination_end_sync, 4),
            "delta_t_seconds": round(self.delta_t_seconds, 4),
            "is_overlapping": self.is_overlapping,
            "overlap_duration_seconds": round(self.overlap_duration_seconds, 4),
            "regime": self.regime,
            "normalized_score": round(self.normalized_score, 4),
        }


@dataclass
class MotionEvidence:
    """Trajectory and motion direction consistency evaluation."""
    origin_direction: Optional[str]
    destination_direction: Optional[str]
    relative_bearing_deg: Optional[float]
    is_aligned: bool
    normalized_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_direction": self.origin_direction,
            "destination_direction": self.destination_direction,
            "relative_bearing_deg": round(self.relative_bearing_deg, 2) if self.relative_bearing_deg is not None else None,
            "is_aligned": self.is_aligned,
            "normalized_score": round(self.normalized_score, 4),
        }


@dataclass
class VehicleTypeEvidence:
    """Vehicle type classification consistency evaluation (soft evidence)."""
    origin_type: str
    destination_type: str
    is_exact_match: bool
    normalized_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_type": self.origin_type,
            "destination_type": self.destination_type,
            "is_exact_match": self.is_exact_match,
            "normalized_score": round(self.normalized_score, 4),
        }


@dataclass
class OCREvidence:
    """ANPR / OCR license plate consistency evaluation."""
    origin_plate: Optional[str]
    destination_plate: Optional[str]
    origin_has_plate: bool
    destination_has_plate: bool
    is_exact_match: bool
    edit_distance: Optional[int]
    levenshtein_similarity: Optional[float]
    is_contradiction: bool
    status: str  # "EXACT_MATCH", "PARTIAL_MATCH", "CONTRADICTION", "MISSING_ONE_SIDE", "MISSING_BOTH"
    normalized_score: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_plate": self.origin_plate,
            "destination_plate": self.destination_plate,
            "origin_has_plate": self.origin_has_plate,
            "destination_has_plate": self.destination_has_plate,
            "is_exact_match": self.is_exact_match,
            "edit_distance": self.edit_distance,
            "levenshtein_similarity": round(self.levenshtein_similarity, 4) if self.levenshtein_similarity is not None else None,
            "is_contradiction": self.is_contradiction,
            "status": self.status,
            "normalized_score": round(self.normalized_score, 4) if self.normalized_score is not None else None,
        }


@dataclass
class TopologyEvidenceSummary:
    """Camera graph road topology prior evaluation."""
    has_directed_edge: bool
    relationship: Optional[str]
    edge_distance_m: Optional[float]
    normalized_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_directed_edge": self.has_directed_edge,
            "relationship": self.relationship,
            "edge_distance_m": round(self.edge_distance_m, 2) if self.edge_distance_m is not None else None,
            "normalized_score": round(self.normalized_score, 4),
        }


@dataclass
class CameraQualityEvidence:
    """Camera reliability and observation quality factors."""
    origin_reliability: float
    destination_reliability: float
    origin_embedding_quality: Optional[float]
    destination_embedding_quality: Optional[float]
    combined_quality_factor: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin_reliability": round(self.origin_reliability, 3),
            "destination_reliability": round(self.destination_reliability, 3),
            "origin_embedding_quality": round(self.origin_embedding_quality, 3) if self.origin_embedding_quality is not None else None,
            "destination_embedding_quality": round(self.destination_embedding_quality, 3) if self.destination_embedding_quality is not None else None,
            "combined_quality_factor": round(self.combined_quality_factor, 4),
        }


@dataclass
class CandidateEvidenceLedger:
    """
    Comprehensive, auditable evidence ledger record for an evaluated candidate pair.
    """
    candidate_id: str
    scenario_id: str
    origin_tracklet: str
    destination_tracklet: str
    decision: str  # "CONFIRMED", "AMBIGUOUS", "REJECTED"
    association_score: float
    evidence: Dict[str, Any]
    active_weights: Dict[str, float]
    available_modalities: List[str]
    missing_modalities: List[str]
    incompatible_modalities: List[str]
    reason_codes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "scenario_id": self.scenario_id,
            "origin_tracklet": self.origin_tracklet,
            "destination_tracklet": self.destination_tracklet,
            "decision": self.decision,
            "association_score": round(self.association_score, 4),
            "evidence": self.evidence,
            "active_weights": {k: round(v, 4) for k, v in self.active_weights.items()},
            "available_modalities": self.available_modalities,
            "missing_modalities": self.missing_modalities,
            "incompatible_modalities": self.incompatible_modalities,
            "reason_codes": self.reason_codes,
        }
