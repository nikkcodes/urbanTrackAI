"""
Data models and schemas for Day 5 Camera Reliability & Uncertainty Propagation.

Defines:
- CameraReliability: Trustworthiness and operational quality of a camera/sensor.
- ObservationReliability: Quality and completeness of an individual observation detection.
- IdentityMatchReliability: Trustworthiness of pairwise identity fusion evidence.
- TrajectoryReliability: Propagated uncertainty and reliability across a reconstructed trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class CameraReliability:
    """
    Operational reliability model for a physical camera or sensor source.

    Attributes:
        camera_id: Unique camera identifier.
        reliability: Scalar reliability score in [0.0, 1.0].
        overall_reliability: Optional alias for reliability.
        status: Status classification ('known', 'configured', 'default', 'degraded').
        factors: Breakdown of contributing quality factors.
        metadata: Detailed operational flags and metrics.
        explanation: Human-readable rationale for camera reliability score.
    """
    camera_id: str
    reliability: float = 0.85
    overall_reliability: Optional[float] = None
    status: str = "configured"
    factors: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    uptime_score: Optional[float] = None
    calibration_score: Optional[float] = None
    environmental_score: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    explanation: str = ""

    def __post_init__(self) -> None:
        if self.overall_reliability is not None:
            self.reliability = float(self.overall_reliability)
        self.reliability = max(0.0, min(1.0, float(self.reliability)))
        self.overall_reliability = self.reliability
        if not self.metadata and self.factors:
            self.metadata = dict(self.factors)
        elif not self.factors and self.metadata:
            self.factors = dict(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "camera_id": self.camera_id,
            "reliability": round(self.reliability, 4),
            "overall_reliability": round(self.reliability, 4),
            "status": self.status,
            "factors": dict(self.factors),
            "metadata": dict(self.metadata),
            "explanation": self.explanation,
        }
        if self.uptime_score is not None:
            d["uptime_score"] = round(self.uptime_score, 4)
        if self.calibration_score is not None:
            d["calibration_score"] = round(self.calibration_score, 4)
        if self.environmental_score is not None:
            d["environmental_score"] = round(self.environmental_score, 4)
        if self.confidence_interval is not None:
            d["confidence_interval"] = self.confidence_interval
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CameraReliability:
        rel = data.get("reliability", data.get("overall_reliability", 0.85))
        return cls(
            camera_id=str(data["camera_id"]),
            reliability=float(rel),
            overall_reliability=float(rel),
            status=str(data.get("status", "configured")),
            factors=dict(data.get("factors", {})),
            metadata=dict(data.get("metadata", {})),
            uptime_score=data.get("uptime_score"),
            calibration_score=data.get("calibration_score"),
            environmental_score=data.get("environmental_score"),
            confidence_interval=tuple(data["confidence_interval"]) if data.get("confidence_interval") else None,
            explanation=str(data.get("explanation", "")),
        )


@dataclass
class ObservationReliability:
    """
    Observation-level quality assessment separating physical sighting quality from
    cross-camera identity evidence completeness.

    NOTE: All reliability values are deterministic heuristic estimates for operational
    ranking and uncertainty propagation. They are NOT statistically calibrated Bayesian
    probabilities and do not represent mathematical certainty.

    Attributes:
        observation_id: Unique observation identifier.
        camera_id: Camera identifier where detection occurred.
        observation_quality: Physical observation quality in [0.0, 1.0] (camera reliability,
            detection confidence, bounding box / spatio-temporal completeness).
        identity_evidence_quality: Quality and completeness of cross-camera identity evidence
            in [0.0, 1.0] (appearance Re-ID embeddings, license plate OCR).
        reliability: Composite deterministic heuristic tracking reliability in [0.0, 1.0].
        overall_reliability: Optional alias for reliability.
        camera_reliability: Optional camera reliability score.
        detection_confidence: Optional object detection confidence.
        ocr_reliability: Optional license plate OCR confidence.
        reid_reliability: Optional appearance embedding quality score.
        uncertainty: Tracking uncertainty score (1.0 - reliability).
        uncertainty_level: Categorical level ('low', 'moderate', 'high', 'critical').
        available_evidence: List of confirmed, usable evidence fields.
        missing_evidence: List of unavailable evidence fields.
        degradation_factors: List of identified sensor or detection degradation causes.
        factors: Dictionary of individual feature reliability scores.
        explanation: Human-readable explanation distinguishing physical quality from identity completeness.
    """
    observation_id: str
    camera_id: str
    observation_quality: float = 0.85
    identity_evidence_quality: float = 0.0
    reliability: float = 0.85
    overall_reliability: Optional[float] = None
    camera_reliability: Optional[float] = None
    detection_confidence: Optional[float] = None
    ocr_reliability: Optional[float] = None
    reid_reliability: Optional[float] = None
    uncertainty: float = 0.0
    uncertainty_level: str = "low"
    available_evidence: List[str] = field(default_factory=list)
    missing_evidence: List[str] = field(default_factory=list)
    degradation_factors: List[str] = field(default_factory=list)
    factors: Dict[str, float] = field(default_factory=dict)
    explanation: str = ""

    def __post_init__(self) -> None:
        if self.overall_reliability is not None:
            self.reliability = float(self.overall_reliability)
        self.reliability = max(0.0, min(1.0, float(self.reliability)))
        self.overall_reliability = self.reliability
        self.observation_quality = max(0.0, min(1.0, float(self.observation_quality)))
        self.identity_evidence_quality = max(0.0, min(1.0, float(self.identity_evidence_quality)))
        if self.camera_reliability is None and "camera_reliability" in self.factors:
            self.camera_reliability = self.factors["camera_reliability"]
        self.uncertainty = round(1.0 - self.reliability, 4)
        if self.uncertainty <= 0.25:
            self.uncertainty_level = "low"
        elif self.uncertainty <= 0.50:
            self.uncertainty_level = "moderate"
        elif self.uncertainty <= 0.75:
            self.uncertainty_level = "high"
        else:
            self.uncertainty_level = "critical"

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "observation_id": self.observation_id,
            "camera_id": self.camera_id,
            "observation_quality": round(self.observation_quality, 4),
            "identity_evidence_quality": round(self.identity_evidence_quality, 4),
            "reliability": round(self.reliability, 4),
            "overall_reliability": round(self.reliability, 4),
            "uncertainty": round(self.uncertainty, 4),
            "uncertainty_level": self.uncertainty_level,
            "available_evidence": list(self.available_evidence),
            "missing_evidence": list(self.missing_evidence),
            "degradation_factors": list(self.degradation_factors),
            "factors": {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in self.factors.items()},
            "explanation": self.explanation,
        }
        if self.camera_reliability is not None:
            d["camera_reliability"] = round(self.camera_reliability, 4)
        if self.detection_confidence is not None:
            d["detection_confidence"] = round(self.detection_confidence, 4)
        if self.ocr_reliability is not None:
            d["ocr_reliability"] = round(self.ocr_reliability, 4)
        if self.reid_reliability is not None:
            d["reid_reliability"] = round(self.reid_reliability, 4)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ObservationReliability:
        rel = data.get("reliability", data.get("overall_reliability", 0.85))
        obs_qual = data.get("observation_quality", rel)
        id_qual = data.get("identity_evidence_quality", 0.0)
        return cls(
            observation_id=str(data["observation_id"]),
            camera_id=str(data["camera_id"]),
            observation_quality=float(obs_qual),
            identity_evidence_quality=float(id_qual),
            reliability=float(rel),
            overall_reliability=float(rel),
            camera_reliability=data.get("camera_reliability"),
            detection_confidence=data.get("detection_confidence"),
            ocr_reliability=data.get("ocr_reliability"),
            reid_reliability=data.get("reid_reliability"),
            uncertainty=float(data.get("uncertainty", 1.0 - float(rel))),
            uncertainty_level=str(data.get("uncertainty_level", "low")),
            available_evidence=list(data.get("available_evidence", [])),
            missing_evidence=list(data.get("missing_evidence", [])),
            degradation_factors=list(data.get("degradation_factors", [])),
            factors=dict(data.get("factors", {})),
            explanation=str(data.get("explanation", "")),
        )


@dataclass
class IdentityMatchReliability:
    """
    Reliability and uncertainty evaluation for pairwise identity matching.

    Attributes:
        observation_a: First observation identifier.
        observation_b: Second observation identifier.
        match_probability: Estimated match likelihood from Day 2 (P in [0.0, 1.0]).
        endpoint_reliability: Geometric mean of endpoint observation reliabilities.
        evidence_quality: Quality factor of identity evidence (ReID, plate).
        combined_reliability: Propagated trust in this match.
        uncertainty: 1.0 - combined_reliability.
        explanation: Human-readable rationale.
    """
    observation_a: str
    observation_b: str
    match_probability: float
    endpoint_reliability: float
    evidence_quality: float
    combined_reliability: float
    uncertainty: float = 0.0
    explanation: str = ""

    def __post_init__(self) -> None:
        self.combined_reliability = max(0.0, min(1.0, float(self.combined_reliability)))
        self.uncertainty = round(1.0 - self.combined_reliability, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_a": self.observation_a,
            "observation_b": self.observation_b,
            "match_probability": round(self.match_probability, 4),
            "endpoint_reliability": round(self.endpoint_reliability, 4),
            "evidence_quality": round(self.evidence_quality, 4),
            "combined_reliability": round(self.combined_reliability, 4),
            "uncertainty": round(self.uncertainty, 4),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> IdentityMatchReliability:
        return cls(
            observation_a=str(data["observation_a"]),
            observation_b=str(data["observation_b"]),
            match_probability=float(data["match_probability"]),
            endpoint_reliability=float(data["endpoint_reliability"]),
            evidence_quality=float(data["evidence_quality"]),
            combined_reliability=float(data["combined_reliability"]),
            uncertainty=float(data.get("uncertainty", 1.0 - float(data["combined_reliability"]))),
            explanation=str(data.get("explanation", "")),
        )


@dataclass
class TrajectoryReliability:
    """
    Propagated reliability and uncertainty metrics for a trajectory segment or gap.

    Attributes:
        trajectory_id: Unique trajectory or gap identifier.
        overall_reliability: Composite trajectory reliability in [0.0, 1.0].
        overall_uncertainty: Composite trajectory uncertainty in [0.0, 1.0].
        uncertainty_level: Categorical uncertainty level ('low', 'moderate', 'high', 'critical').
        endpoint_reliabilities: Reliabilities of origin and destination observations.
        gap_penalty: Uncertainty penalty added for unobserved intermediate transit.
        route_entropy: Shannon entropy across feasible candidate routes.
        sources_of_uncertainty: Specific causes of remaining uncertainty.
        explanation: Human-readable rationale.
    """
    trajectory_id: str
    overall_reliability: float
    overall_uncertainty: float
    uncertainty_level: str = "low"
    endpoint_reliabilities: List[Dict[str, Any]] = field(default_factory=list)
    gap_penalty: float = 0.0
    route_entropy: float = 0.0
    sources_of_uncertainty: List[str] = field(default_factory=list)
    explanation: str = ""

    def __post_init__(self) -> None:
        self.overall_reliability = max(0.0, min(1.0, float(self.overall_reliability)))
        self.overall_uncertainty = max(0.0, min(1.0, float(self.overall_uncertainty)))
        if self.overall_uncertainty <= 0.25:
            self.uncertainty_level = "low"
        elif self.overall_uncertainty <= 0.50:
            self.uncertainty_level = "moderate"
        elif self.overall_uncertainty <= 0.75:
            self.uncertainty_level = "high"
        else:
            self.uncertainty_level = "critical"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "overall_reliability": round(self.overall_reliability, 4),
            "overall_uncertainty": round(self.overall_uncertainty, 4),
            "uncertainty_level": self.uncertainty_level,
            "endpoint_reliabilities": list(self.endpoint_reliabilities),
            "gap_penalty": round(self.gap_penalty, 4),
            "route_entropy": round(self.route_entropy, 4),
            "sources_of_uncertainty": list(self.sources_of_uncertainty),
            "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TrajectoryReliability:
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            overall_reliability=float(data["overall_reliability"]),
            overall_uncertainty=float(data["overall_uncertainty"]),
            uncertainty_level=str(data.get("uncertainty_level", "low")),
            endpoint_reliabilities=list(data.get("endpoint_reliabilities", [])),
            gap_penalty=float(data.get("gap_penalty", 0.0)),
            route_entropy=float(data.get("route_entropy", 0.0)),
            sources_of_uncertainty=list(data.get("sources_of_uncertainty", [])),
            explanation=str(data.get("explanation", "")),
        )
