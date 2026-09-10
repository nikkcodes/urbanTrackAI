"""
Anomaly Detection & Investigation Schemas for UrbanTrack AI (Day 7).

Defines structured data models for:
- DataQualityStatus: Distinguishes data validity from behavioral anomalies.
- SignalCategory: Distinguishes data quality, temporal inconsistency, physical
  inconsistency, network constraint inconsistency, behavioral anomaly, and network anomaly.
- AnomalyEvidence: Individual deviation or constraint inconsistency signal.
- AnomalyResult: Fused anomaly result for an entity (vehicle, road, network) with
  Day 5 reliability and uncertainty preserved orthogonally.
- MobilityBaseline: Baseline model abstraction (configured, synthetic, or unavailable).
- CityAnomalyReport: Consolidated city-scale anomaly report.

CRITICAL SEMANTIC PRINCIPLES:
1. INVALID DATA ≠ PHYSICAL INCONSISTENCY ≠ BEHAVIORAL ANOMALY ≠ INVESTIGATION PRIORITY.
2. Malformed/invalid input produces DataQualityStatus.INVALID / AnomalySeverity.INVALID_INPUT,
   NOT a high-priority behavioral anomaly.
3. Missing baseline produces AnomalySeverity.INSUFFICIENT_EVIDENCE, NOT confidently NORMAL.
4. Anomaly score is a behavioral/network deviation score in [0.0, 1.0], NOT a probability
   of wrongdoing, guilt, or malice.
5. Inferred routes traversing closed roads represent NETWORK_CONSTRAINT_INCONSISTENCY
   (incompatible network state), NOT suspicious driver behavior.
6. Reliability and uncertainty are NEVER multiplied into anomaly scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


class DataQualityStatus(str, Enum):
    """Integrity and validity status of input data."""
    VALID = "valid"
    INCOMPLETE = "incomplete"
    INVALID = "invalid"


class SignalCategory(str, Enum):
    """Category of signal distinguishing data quality, physical limits, and behavior."""
    DATA_QUALITY = "data_quality"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    PHYSICAL_INCONSISTENCY = "physical_inconsistency"
    NETWORK_CONSTRAINT_INCONSISTENCY = "network_constraint_inconsistency"
    BEHAVIORAL_ANOMALY = "behavioral_anomaly"
    NETWORK_ANOMALY = "network_anomaly"


class SignalSeverity(str, Enum):
    """Severity level of an individual anomaly signal."""
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    EXTREME = "extreme"


class AnomalySeverity(str, Enum):
    """Consolidated severity level for an investigated entity."""
    NORMAL = "normal"                          # verified within normal limits against baseline
    WATCH = "watch"                            # mild deviation [0.30, 0.49]
    INVESTIGATE = "investigate"                # significant deviation [0.50, 0.74]
    HIGH_PRIORITY = "high_priority"            # severe deviation or physical violation [0.75, 1.00]
    INSUFFICIENT_EVIDENCE = "insufficient_evidence" # cannot determine due to missing baseline/evidence
    INVALID_INPUT = "invalid_input"            # malformed or invalid input data

    @classmethod
    def classify(cls, score: Optional[float], has_evaluated_signals: bool = True) -> AnomalySeverity:
        """Map a normalized anomaly score in [0.0, 1.0] to an operational severity."""
        if not has_evaluated_signals or score is None:
            return cls.INSUFFICIENT_EVIDENCE
        if score < 0.30:
            return cls.NORMAL
        elif score < 0.50:
            return cls.WATCH
        elif score < 0.75:
            return cls.INVESTIGATE
        else:
            return cls.HIGH_PRIORITY


class InvestigationPriority(str, Enum):
    """
    Actionable operational review priority combining behavioral deviation score,
    data quality, and evidence reliability.
    """
    NORMAL = "normal"
    WATCH = "watch"
    INVESTIGATE = "investigate"
    HIGH_PRIORITY = "high_priority"
    DATA_QUALITY_REVIEW = "data_quality_review"

    @classmethod
    def determine(
        cls,
        anomaly_severity: AnomalySeverity,
        reliability: Optional[float] = None,
        uncertainty: Optional[float] = None,
        data_quality_status: DataQualityStatus = DataQualityStatus.VALID,
    ) -> InvestigationPriority:
        """
        Determine operational priority while preserving reliability/uncertainty separation.

        Interpretation Matrix:
        - Invalid Input -> DATA_QUALITY_REVIEW
        - Insufficient Evidence -> NORMAL (no alert warranted)
        - High Anomaly (HIGH_PRIORITY / INVESTIGATE) + High Reliability (>= 0.70) -> HIGH_PRIORITY / INVESTIGATE
        - High Anomaly (HIGH_PRIORITY / INVESTIGATE) + Low Reliability (< 0.50)  -> WATCH (cautious verification)
        - Moderate Anomaly (WATCH) + High Reliability (>= 0.70)                 -> WATCH
        - Low Anomaly (NORMAL) + High/Low Reliability                           -> NORMAL
        """
        if data_quality_status == DataQualityStatus.INVALID or anomaly_severity == AnomalySeverity.INVALID_INPUT:
            return cls.DATA_QUALITY_REVIEW

        if anomaly_severity == AnomalySeverity.INSUFFICIENT_EVIDENCE:
            return cls.NORMAL

        if reliability is None:
            if anomaly_severity in (AnomalySeverity.HIGH_PRIORITY, AnomalySeverity.INVESTIGATE, AnomalySeverity.WATCH, AnomalySeverity.NORMAL):
                return cls(anomaly_severity.value)
            return cls.NORMAL

        # High anomaly with low reliability is downgraded in priority to prevent false alerts
        if anomaly_severity in (AnomalySeverity.HIGH_PRIORITY, AnomalySeverity.INVESTIGATE):
            if reliability < 0.50:
                return cls.WATCH
            elif reliability < 0.70 and anomaly_severity == AnomalySeverity.HIGH_PRIORITY:
                return cls.INVESTIGATE
            return cls(anomaly_severity.value)

        if anomaly_severity == AnomalySeverity.WATCH:
            if reliability < 0.40:
                return cls.NORMAL
            return cls.WATCH

        return cls.NORMAL


@dataclass
class AnomalyEvidence:
    """
    A single measurable deviation signal contributing to an anomaly investigation.

    Attributes:
        signal_type: Identifier of the signal (e.g. 'travel_time_deviation').
        signal_category: Conceptual category (DATA_QUALITY, TEMPORAL_INCONSISTENCY, etc.).
        measured_value: Inferred or observed numerical measurement.
        baseline_value: Expected or configured baseline reference value.
        threshold: Threshold at which this signal transitions to elevated/anomalous.
        signal_score: Normalized deviation score in [0.0, 1.0] (NOT a probability).
        severity: Operational severity classification for this specific signal.
        available: Boolean indicating whether this signal had sufficient data to be evaluated.
        explanation: Human-readable explanation of why this signal was or was not flagged.
        metadata: Additional context (e.g. units, tolerance factor, camera ids).
    """
    signal_type: str
    signal_category: SignalCategory = SignalCategory.BEHAVIORAL_ANOMALY
    measured_value: Optional[float] = None
    baseline_value: Optional[float] = None
    threshold: Optional[float] = None
    signal_score: float = 0.0
    severity: SignalSeverity = SignalSeverity.NORMAL
    available: bool = True
    explanation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "signal_category": self.signal_category.value if isinstance(self.signal_category, SignalCategory) else str(self.signal_category),
            "measured_value": round(self.measured_value, 4) if self.measured_value is not None else None,
            "baseline_value": round(self.baseline_value, 4) if self.baseline_value is not None else None,
            "threshold": round(self.threshold, 4) if self.threshold is not None else None,
            "signal_score": round(self.signal_score, 4),
            "severity": self.severity.value if isinstance(self.severity, SignalSeverity) else str(self.severity),
            "available": self.available,
            "explanation": self.explanation,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnomalyEvidence:
        """Construct AnomalyEvidence from dictionary."""
        cat = data.get("signal_category", SignalCategory.BEHAVIORAL_ANOMALY)
        if isinstance(cat, str):
            try:
                cat = SignalCategory(cat)
            except ValueError:
                cat = SignalCategory.BEHAVIORAL_ANOMALY

        sev = data.get("severity", SignalSeverity.NORMAL)
        if isinstance(sev, str):
            try:
                sev = SignalSeverity(sev)
            except ValueError:
                sev = SignalSeverity.NORMAL

        return cls(
            signal_type=str(data["signal_type"]),
            signal_category=cat,
            measured_value=float(data["measured_value"]) if data.get("measured_value") is not None else None,
            baseline_value=float(data["baseline_value"]) if data.get("baseline_value") is not None else None,
            threshold=float(data["threshold"]) if data.get("threshold") is not None else None,
            signal_score=float(data.get("signal_score", 0.0)),
            severity=sev,
            available=bool(data.get("available", True)),
            explanation=str(data.get("explanation", "")),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class AnomalyResult:
    """
    Consolidated investigation record for an entity (vehicle trajectory, road, or network).

    Attributes:
        anomaly_id: Unique identifier for the investigation result.
        entity_type: Category of entity ('vehicle', 'road', 'network').
        entity_id: Identifier of the entity (e.g. track_id or road_id).
        data_quality_status: Status of input integrity (VALID, INCOMPLETE, INVALID).
        anomaly_types: List of triggered signal types.
        overall_score: Composite normalized score in [0.0, 1.0] across available signals (or None).
        severity: Operational severity (normal, watch, investigate, high_priority, insufficient_evidence, invalid_input).
        evidence: List of AnomalyEvidence objects.
        reliability: Day 5 evidence reliability preserved separately (sensor trust).
        uncertainty: Day 5 evidence uncertainty preserved separately.
        investigation_priority: Recommended action priority based on score and reliability.
        explanation: Synthesized human-readable investigation summary answering WHY.
        timestamp_or_window: Time window or reference timestamp of the entity.
        metadata: Domain-specific context.
    """
    anomaly_id: str
    entity_type: str
    entity_id: str
    data_quality_status: DataQualityStatus = DataQualityStatus.VALID
    anomaly_types: List[str] = field(default_factory=list)
    overall_score: Optional[float] = 0.0
    severity: AnomalySeverity = AnomalySeverity.NORMAL
    evidence: List[AnomalyEvidence] = field(default_factory=list)
    reliability: Optional[float] = None
    uncertainty: Optional[float] = None
    investigation_priority: InvestigationPriority = InvestigationPriority.NORMAL
    explanation: str = ""
    timestamp_or_window: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_anomalous(self) -> bool:
        """
        True if overall severity represents an elevated behavioral or network deviation.
        Explicitly excludes NORMAL, INSUFFICIENT_EVIDENCE, and INVALID_INPUT.
        """
        return self.severity in (
            AnomalySeverity.WATCH,
            AnomalySeverity.INVESTIGATE,
            AnomalySeverity.HIGH_PRIORITY,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "data_quality_status": (
                self.data_quality_status.value
                if isinstance(self.data_quality_status, DataQualityStatus)
                else str(self.data_quality_status)
            ),
            "anomaly_types": list(self.anomaly_types),
            "overall_score": round(self.overall_score, 4) if self.overall_score is not None else None,
            "severity": self.severity.value if isinstance(self.severity, AnomalySeverity) else str(self.severity),
            "evidence": [e.to_dict() for e in self.evidence],
            "reliability": round(self.reliability, 4) if self.reliability is not None else None,
            "uncertainty": round(self.uncertainty, 4) if self.uncertainty is not None else None,
            "investigation_priority": (
                self.investigation_priority.value
                if isinstance(self.investigation_priority, InvestigationPriority)
                else str(self.investigation_priority)
            ),
            "explanation": self.explanation,
            "timestamp_or_window": self.timestamp_or_window,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnomalyResult:
        """Construct AnomalyResult from dictionary, preserving enums and sub-structures."""
        dqs = data.get("data_quality_status", DataQualityStatus.VALID)
        if isinstance(dqs, str):
            try:
                dqs = DataQualityStatus(dqs)
            except ValueError:
                dqs = DataQualityStatus.VALID

        sev = data.get("severity", AnomalySeverity.NORMAL)
        if isinstance(sev, str):
            try:
                sev = AnomalySeverity(sev)
            except ValueError:
                sev = AnomalySeverity.NORMAL

        prio = data.get("investigation_priority", InvestigationPriority.NORMAL)
        if isinstance(prio, str):
            try:
                prio = InvestigationPriority(prio)
            except ValueError:
                prio = InvestigationPriority.NORMAL

        ev_list = [
            AnomalyEvidence.from_dict(e) if isinstance(e, dict) else e
            for e in data.get("evidence", [])
        ]

        return cls(
            anomaly_id=str(data["anomaly_id"]),
            entity_type=str(data["entity_type"]),
            entity_id=str(data["entity_id"]),
            data_quality_status=dqs,
            anomaly_types=list(data.get("anomaly_types", [])),
            overall_score=float(data["overall_score"]) if data.get("overall_score") is not None else None,
            severity=sev,
            evidence=ev_list,
            reliability=float(data["reliability"]) if data.get("reliability") is not None else None,
            uncertainty=float(data["uncertainty"]) if data.get("uncertainty") is not None else None,
            investigation_priority=prio,
            explanation=str(data.get("explanation", "")),
            timestamp_or_window=data.get("timestamp_or_window"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class MobilityBaseline:
    """
    Configured baseline reference model for urban mobility anomaly detection.

    Attributes:
        source: Provenance label ('synthetic', 'configured', 'historical', 'unavailable').
        expected_travel_times: Mapping (origin_node, dest_node) -> expected travel time (seconds).
        expected_routes: Mapping (origin_node, dest_node) -> expected node sequence.
        expected_road_demand_vph: Mapping road_id -> baseline hourly demand.
        expected_road_utilization: Mapping road_id -> baseline utilization ratio.
        speed_limits: Mapping road_id -> speed limit (km/h).
        tolerance_factors: Configurable thresholds for anomaly scoring.
    """
    source: str = "configured"
    expected_travel_times: Dict[Tuple[str, str], float] = field(default_factory=dict)
    expected_routes: Dict[Tuple[str, str], List[str]] = field(default_factory=dict)
    expected_road_demand_vph: Dict[str, float] = field(default_factory=dict)
    expected_road_utilization: Dict[str, float] = field(default_factory=dict)
    speed_limits: Dict[str, float] = field(default_factory=dict)
    tolerance_factors: Dict[str, float] = field(default_factory=dict)

    def get_expected_travel_time(self, origin: str, dest: str) -> Optional[float]:
        """Retrieve baseline travel time in seconds for an OD pair."""
        return self.expected_travel_times.get((origin, dest))

    def get_expected_route(self, origin: str, dest: str) -> Optional[List[str]]:
        """Retrieve baseline expected corridor nodes for an OD pair."""
        return self.expected_routes.get((origin, dest))

    def get_expected_demand(self, road_id: str) -> Optional[float]:
        """Retrieve baseline expected hourly demand (vph) for a road."""
        return self.expected_road_demand_vph.get(road_id)

    def get_expected_utilization(self, road_id: str) -> Optional[float]:
        """Retrieve baseline expected utilization ratio for a road."""
        return self.expected_road_utilization.get(road_id)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MobilityBaseline:
        """Construct a MobilityBaseline from a serialized dictionary."""
        source = data.get("source", "configured")

        # Parse OD travel times
        expected_travel_times = {}
        for k, v in data.get("expected_travel_times", {}).items():
            if isinstance(k, str) and "->" in k:
                orig, dest = k.split("->")
                expected_travel_times[(orig.strip(), dest.strip())] = float(v)
            elif isinstance(k, (list, tuple)) and len(k) == 2:
                expected_travel_times[(str(k[0]), str(k[1]))] = float(v)

        # Parse OD routes
        expected_routes = {}
        for k, v in data.get("expected_routes", {}).items():
            if isinstance(k, str) and "->" in k:
                orig, dest = k.split("->")
                expected_routes[(orig.strip(), dest.strip())] = list(v)
            elif isinstance(k, (list, tuple)) and len(k) == 2:
                expected_routes[(str(k[0]), str(k[1]))] = list(v)

        return cls(
            source=source,
            expected_travel_times=expected_travel_times,
            expected_routes=expected_routes,
            expected_road_demand_vph={k: float(v) for k, v in data.get("expected_road_demand_vph", {}).items()},
            expected_road_utilization={k: float(v) for k, v in data.get("expected_road_utilization", {}).items()},
            speed_limits={k: float(v) for k, v in data.get("speed_limits", {}).items()},
            tolerance_factors={k: float(v) for k, v in data.get("tolerance_factors", {}).items()},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "expected_travel_times": {
                f"{k[0]}->{k[1]}": v for k, v in self.expected_travel_times.items()
            },
            "expected_routes": {
                f"{k[0]}->{k[1]}": v for k, v in self.expected_routes.items()
            },
            "expected_road_demand_vph": dict(self.expected_road_demand_vph),
            "expected_road_utilization": dict(self.expected_road_utilization),
            "speed_limits": dict(self.speed_limits),
            "tolerance_factors": dict(self.tolerance_factors),
        }


@dataclass
class CityAnomalyReport:
    """
    Consolidated Day 7 City Anomaly Detection & Investigation Report.

    Contains vehicle-level, road-level, and network-level investigation results,
    along with summary statistics and execution metadata.
    """
    time_window: Dict[str, Any]
    vehicle_anomalies: List[AnomalyResult] = field(default_factory=list)
    road_anomalies: List[AnomalyResult] = field(default_factory=list)
    network_anomalies: List[AnomalyResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_window": dict(self.time_window),
            "vehicle_anomalies": [v.to_dict() for v in self.vehicle_anomalies],
            "road_anomalies": [r.to_dict() for r in self.road_anomalies],
            "network_anomalies": [n.to_dict() for n in self.network_anomalies],
            "summary": dict(self.summary),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)
