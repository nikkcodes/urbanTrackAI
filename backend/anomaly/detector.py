"""Evidence-based comparison of two mobility snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

from backend.anomaly.distributions import compare_distributions
from backend.anomaly.models import (
    AnomalyAnalysisResult,
    AnomalyEvent,
    BottleneckChange,
    DistributionShift,
    MobilitySnapshot,
    RoadAnomalyEvidence,
    RoadObservationStatus,
)
from backend.anomaly.spatial import SpatialAnomalyGrouper
from backend.mobility.graph import MobilityGraph


@dataclass(frozen=True, slots=True)
class AnomalyDetectorConfig:
    """Explicit thresholds and severity rules for deterministic detection."""

    flow_relative_threshold: float = 0.30
    flow_absolute_threshold: float = 1.0
    flow_strong_relative_threshold: float = 0.50
    flow_epsilon: float = 1e-6
    od_divergence_threshold: float = 0.20
    route_divergence_threshold: float = 0.20
    hhi_delta_threshold: float = 0.05
    strong_hhi_delta_threshold: float = 0.10
    critical_bottleneck_severity: str = "HEAVY"
    moderate_signal_count: int = 2
    high_signal_count: int = 2
    critical_signal_count: int = 3

    def __post_init__(self) -> None:
        numeric_fields = (
            "flow_relative_threshold", "flow_absolute_threshold", "flow_strong_relative_threshold",
            "flow_epsilon", "od_divergence_threshold", "route_divergence_threshold",
            "hhi_delta_threshold", "strong_hhi_delta_threshold",
        )
        for name in numeric_fields:
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.flow_epsilon <= 0.0:
            raise ValueError("flow_epsilon must be positive")
        if self.flow_strong_relative_threshold < self.flow_relative_threshold:
            raise ValueError("flow_strong_relative_threshold must be >= flow_relative_threshold")
        if not isinstance(self.critical_bottleneck_severity, str) or not self.critical_bottleneck_severity.strip():
            raise ValueError("critical_bottleneck_severity must be non-empty")
        for name in ("moderate_signal_count", "high_signal_count", "critical_signal_count"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not (self.moderate_signal_count <= self.high_signal_count <= self.critical_signal_count):
            raise ValueError("severity signal counts must be non-decreasing")


@dataclass(frozen=True, slots=True)
class _SignalAssessment:
    name: str
    strong: bool
    roads: Tuple[str, ...] = ()
    od_pairs: Tuple[Tuple[str, ...], ...] = ()
    explanation: str = ""
    metrics: Mapping[str, float] = None  # type: ignore[assignment]


class AnomalyDetector:
    """Compare compatible mobility states using transparent independent signals."""

    _SEVERITY_RANK = {"NORMAL": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}
    _BOTTLENECK_RANK = {"FREE": 0, "MODERATE": 1, "HEAVY": 2, "SEVERE": 3}

    def __init__(self, config: Optional[AnomalyDetectorConfig] = None) -> None:
        self.config = config or AnomalyDetectorConfig()
        self.spatial_grouper = SpatialAnomalyGrouper()

    def compare(
        self,
        baseline: MobilitySnapshot,
        current: MobilitySnapshot,
        graph: Optional[MobilityGraph] = None,
    ) -> AnomalyAnalysisResult:
        """Compare two snapshots and return deterministic evidence and events."""
        self._validate_compatibility(baseline, current)
        road_evidence = self._road_evidence(baseline, current)
        od_shift = compare_distributions(
            baseline.od_demand, current.od_demand, self.config.od_divergence_threshold
        )
        route_shift = compare_distributions(
            baseline.route_demand, current.route_demand, self.config.route_divergence_threshold
        )
        hhi_delta = current.hhi - baseline.hhi
        bottleneck_changes = self._bottleneck_changes(baseline, current)

        assessments: List[_SignalAssessment] = []
        surge_roads = tuple(item.road_id for item in road_evidence if item.flow_signal == "FLOW_SURGE")
        drop_roads = tuple(item.road_id for item in road_evidence if item.flow_signal == "FLOW_DROP")
        if surge_roads:
            assessments.append(_SignalAssessment(
                name="FLOW_SURGE", strong=any(self._is_strong_flow(item) for item in road_evidence if item.flow_signal == "FLOW_SURGE"),
                roads=surge_roads,
                explanation=f"Comparable roads increased in hourly flow: {', '.join(surge_roads)}.",
                metrics={"road_count": float(len(surge_roads))},
            ))
        if drop_roads:
            assessments.append(_SignalAssessment(
                name="FLOW_DROP", strong=any(self._is_strong_flow(item) for item in road_evidence if item.flow_signal == "FLOW_DROP"),
                roads=drop_roads,
                explanation=f"Comparable roads decreased in hourly flow: {', '.join(drop_roads)}.",
                metrics={"road_count": float(len(drop_roads))},
            ))

        od_pairs = tuple(sorted(key for key in set(od_shift.baseline_distribution) | set(od_shift.current_distribution)
                               if not math.isclose(od_shift.baseline_distribution.get(key, 0.0), od_shift.current_distribution.get(key, 0.0))))
        if od_shift.threshold_exceeded:
            assessments.append(_SignalAssessment(
                name="OD_SHIFT",
                strong=od_shift.divergence >= self.config.od_divergence_threshold * 2.0,
                od_pairs=od_pairs,
                explanation=f"OD demand distribution changed with Jensen-Shannon divergence {od_shift.divergence:.4f}.",
                metrics={"js_divergence": od_shift.divergence},
            ))

        route_keys = tuple(sorted(set(route_shift.baseline_distribution) | set(route_shift.current_distribution)))
        route_roads = tuple(sorted({road_id for key in route_keys for road_id in key
                                    if not math.isclose(route_shift.baseline_distribution.get(key, 0.0), route_shift.current_distribution.get(key, 0.0))}))
        if route_shift.threshold_exceeded:
            assessments.append(_SignalAssessment(
                name="ROUTE_REDISTRIBUTION",
                strong=route_shift.divergence >= self.config.route_divergence_threshold * 2.0,
                roads=route_roads,
                explanation=f"Probabilistic route demand redistributed with Jensen-Shannon divergence {route_shift.divergence:.4f}.",
                metrics={"js_divergence": route_shift.divergence},
            ))

        if abs(hhi_delta) >= self.config.hhi_delta_threshold:
            assessments.append(_SignalAssessment(
                name="NETWORK_CONCENTRATION_SHIFT",
                strong=abs(hhi_delta) >= self.config.strong_hhi_delta_threshold,
                explanation=f"Hourly-flow HHI changed by {hhi_delta:+.4f}.",
                metrics={"baseline_hhi": baseline.hhi, "current_hhi": current.hhi, "hhi_delta": hhi_delta},
            ))

        bottleneck_roads = tuple(change.road_id for change in bottleneck_changes)
        if bottleneck_changes:
            assessments.append(_SignalAssessment(
                name="BOTTLENECK_SHIFT",
                strong=any(self._is_critical_severity(change.current_severity) for change in bottleneck_changes),
                roads=bottleneck_roads,
                explanation=f"Bottleneck severity/state changed on {len(bottleneck_changes)} road(s).",
                metrics={"road_count": float(len(bottleneck_changes))},
            ))

        events = self._build_events(assessments, od_shift, route_shift, hhi_delta, bottleneck_changes)
        anomalous_roads = sorted({road_id for assessment in assessments for road_id in assessment.roads})
        evidence_by_road = {item.road_id: item for item in road_evidence}
        regions = self.spatial_grouper.group(anomalous_roads, graph, evidence_by_road) if graph and anomalous_roads else ()
        return AnomalyAnalysisResult(
            baseline_snapshot_id=baseline.snapshot_id,
            current_snapshot_id=current.snapshot_id,
            road_evidence=road_evidence,
            od_divergence=od_shift,
            route_divergence=route_shift,
            hhi_delta=hhi_delta,
            bottleneck_changes=bottleneck_changes,
            anomaly_events=events,
            spatial_regions=regions,
        )

    def _road_evidence(self, baseline: MobilitySnapshot, current: MobilitySnapshot) -> Tuple[RoadAnomalyEvidence, ...]:
        records: List[RoadAnomalyEvidence] = []
        for road_id in sorted(set(baseline.road_hourly_flows) | set(current.road_hourly_flows)):
            has_baseline = road_id in baseline.road_hourly_flows
            has_current = road_id in current.road_hourly_flows
            if not has_current:
                records.append(RoadAnomalyEvidence(road_id, RoadObservationStatus.MISSING_CURRENT, baseline.road_hourly_flows[road_id], None, None, None, evidence_count=0))
                continue
            if not has_baseline:
                records.append(RoadAnomalyEvidence(road_id, RoadObservationStatus.MISSING_BASELINE, None, current.road_hourly_flows[road_id], None, None, evidence_count=0))
                continue
            baseline_flow = baseline.road_hourly_flows[road_id]
            current_flow = current.road_hourly_flows[road_id]
            absolute_change = current_flow - baseline_flow
            relative_change = absolute_change / max(abs(baseline_flow), self.config.flow_epsilon)
            flow_signal = None
            if abs(relative_change) >= self.config.flow_relative_threshold and abs(absolute_change) >= self.config.flow_absolute_threshold:
                flow_signal = "FLOW_SURGE" if absolute_change > 0.0 else "FLOW_DROP"
            records.append(RoadAnomalyEvidence(
                road_id, RoadObservationStatus.COMPARABLE, baseline_flow, current_flow,
                absolute_change, relative_change, flow_signal=flow_signal,
                evidence_count=1 if flow_signal else 0,
            ))
        return tuple(records)

    def _bottleneck_changes(self, baseline: MobilitySnapshot, current: MobilitySnapshot) -> Tuple[BottleneckChange, ...]:
        before = {item.road_id: item.severity for item in baseline.bottlenecks}
        after = {item.road_id: item.severity for item in current.bottlenecks}
        changes: List[BottleneckChange] = []
        for road_id in sorted(set(before) | set(after)):
            old = before.get(road_id)
            new = after.get(road_id)
            old_rank = self._BOTTLENECK_RANK.get(old or "FREE", 0)
            new_rank = self._BOTTLENECK_RANK.get(new or "FREE", 0)
            if old is None and new is not None and new_rank >= 1:
                change_type = "NEW_BOTTLENECK"
            elif new is None and old is not None and old_rank >= 1:
                change_type = "BOTTLENECK_DISAPPEARED"
            elif new_rank > old_rank:
                change_type = "SEVERITY_INCREASE"
            elif new_rank < old_rank:
                change_type = "SEVERITY_DECREASE"
            else:
                continue
            changes.append(BottleneckChange(road_id, old, new, change_type))
        return tuple(changes)

    def _build_events(
        self,
        assessments: List[_SignalAssessment],
        od_shift: DistributionShift,
        route_shift: DistributionShift,
        hhi_delta: float,
        bottleneck_changes: Tuple[BottleneckChange, ...],
    ) -> Tuple[AnomalyEvent, ...]:
        events: List[AnomalyEvent] = []
        for index, assessment in enumerate(assessments, start=1):
            events.append(AnomalyEvent(
                anomaly_id=f"ANOM-{index:03d}",
                anomaly_type=assessment.name,
                severity=self._severity([assessment], consensus=False),
                evidence_count=1,
                triggered_signals=(assessment.name,),
                affected_roads=tuple(sorted(assessment.roads)),
                affected_od_pairs=tuple(sorted(assessment.od_pairs)),
                explanation=assessment.explanation,
                metrics=dict(assessment.metrics or {}),
            ))
        if assessments:
            signals = tuple(item.name for item in assessments)
            roads = tuple(sorted({road_id for item in assessments for road_id in item.roads}))
            od_pairs = tuple(sorted({pair for item in assessments for pair in item.od_pairs}))
            events.append(AnomalyEvent(
                anomaly_id=f"ANOM-{len(events) + 1:03d}",
                anomaly_type="MOBILITY_STATE_CHANGE",
                severity=self._severity(assessments, consensus=True),
                evidence_count=len(assessments),
                triggered_signals=signals,
                affected_roads=roads,
                affected_od_pairs=od_pairs,
                explanation="Independent mobility signals agree that the current state differs from baseline: " + "; ".join(item.explanation for item in assessments),
                metrics={"hhi_delta": hhi_delta, "od_divergence": od_shift.divergence, "route_divergence": route_shift.divergence, "bottleneck_change_count": len(bottleneck_changes)},
            ))
        return tuple(events)

    def _severity(self, assessments: List[_SignalAssessment], consensus: bool) -> str:
        if not assessments:
            return "NORMAL"
        strong_count = sum(1 for item in assessments if item.strong)
        count = len(assessments)
        if consensus and strong_count >= self.config.critical_signal_count:
            return "CRITICAL"
        if consensus and (strong_count >= self.config.high_signal_count or count >= self.config.critical_signal_count):
            return "HIGH"
        if strong_count >= 1 or count >= self.config.moderate_signal_count:
            return "MEDIUM"
        return "LOW"

    def _is_strong_flow(self, evidence: RoadAnomalyEvidence) -> bool:
        return evidence.relative_change is not None and abs(evidence.relative_change) >= self.config.flow_strong_relative_threshold

    def _is_critical_severity(self, severity: Optional[str]) -> bool:
        return self._BOTTLENECK_RANK.get(severity or "FREE", 0) >= self._BOTTLENECK_RANK.get(self.config.critical_bottleneck_severity, 2)

    def _validate_compatibility(self, baseline: MobilitySnapshot, current: MobilitySnapshot) -> None:
        if not isinstance(baseline, MobilitySnapshot) or not isinstance(current, MobilitySnapshot):
            raise TypeError("baseline and current must be MobilitySnapshot instances")
        if baseline.network_identity != current.network_identity:
            raise ValueError("baseline and current snapshots use different road-network topology")
        if not math.isclose(baseline.aggregation_duration_hours, current.aggregation_duration_hours, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError("baseline and current aggregation durations are incompatible")
        if (baseline.time_window_start, baseline.time_window_end) != (current.time_window_start, current.time_window_end):
            raise ValueError("baseline and current time windows are incompatible")
