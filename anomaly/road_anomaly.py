"""
Road Flow Anomaly Detector for UrbanTrack AI (Day 7).

Evaluates Day 6 RoadFlowMetric records against physical capacity limits and configured
traffic baselines.

Detects:
1. Road Over-Capacity (utilization_ratio > 1.0 or high utilization >= 0.80)
2. Demand Deviation (flow substantially exceeding configured baseline demand)

CRITICAL SEMANTIC PRINCIPLES:
- A high structural centrality value alone is NOT a traffic anomaly. A road is not
  congested simply because it is structurally important.
- Does NOT claim 'sudden increase' unless multiple comparable time windows exist.
- If hourly utilization is uncalibrated, leaves signal available=False rather than fabricating.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from schemas.anomaly_schema import (
    AnomalyEvidence,
    AnomalyResult,
    AnomalySeverity,
    DataQualityStatus,
    InvestigationPriority,
    MobilityBaseline,
    SignalCategory,
    SignalSeverity,
)
from schemas.mobility_schema import RoadFlowMetric, RoadStatus


class RoadAnomalyDetector:
    """
    Detector for road-segment level flow and capacity anomalies.
    """

    def __init__(self, baseline: Optional[MobilityBaseline] = None) -> None:
        self.baseline = baseline or MobilityBaseline(source="unavailable")

    def evaluate_road(self, road_metric: RoadFlowMetric) -> AnomalyResult:
        """
        Evaluate a single RoadFlowMetric and return a consolidated AnomalyResult.
        """
        rid = road_metric.road_id
        from_n = road_metric.from_node
        to_n = road_metric.to_node

        evidence_list: List[AnomalyEvidence] = []
        triggered_types: List[str] = []

        # 1. Evaluate Over-Capacity / High Utilization
        cap_evidence = self._check_capacity_utilization(road_metric)
        evidence_list.append(cap_evidence)
        if cap_evidence.signal_score >= 0.30:
            triggered_types.append(cap_evidence.signal_type)

        # 2. Evaluate Demand Deviation against Baseline
        demand_evidence = self._check_demand_deviation(road_metric)
        evidence_list.append(demand_evidence)
        if demand_evidence.signal_score >= 0.30:
            triggered_types.append(demand_evidence.signal_type)

        # Evidence fusion across available signals
        available_signals = [e for e in evidence_list if e.available]
        elevated_signals = [e for e in available_signals if e.signal_score >= 0.30]

        if not available_signals:
            overall_score = None
            severity = AnomalySeverity.INSUFFICIENT_EVIDENCE
            data_quality_status = DataQualityStatus.INCOMPLETE
        elif not elevated_signals:
            overall_score = max(e.signal_score for e in available_signals)
            severity = AnomalySeverity.NORMAL
            data_quality_status = DataQualityStatus.VALID
        elif len(elevated_signals) == 1:
            overall_score = elevated_signals[0].signal_score
            severity = AnomalySeverity.classify(overall_score, has_evaluated_signals=True)
            data_quality_status = DataQualityStatus.VALID
        else:
            sorted_elevated = sorted(elevated_signals, key=lambda e: e.signal_score, reverse=True)
            primary_score = sorted_elevated[0].signal_score
            reinforcement = sum(e.signal_score * 0.15 for e in sorted_elevated[1:])
            overall_score = min(1.0, primary_score + reinforcement)
            severity = AnomalySeverity.classify(overall_score, has_evaluated_signals=True)
            data_quality_status = DataQualityStatus.VALID

        # Extract reliability / uncertainty if present
        reliability = None
        if road_metric.reliability_summary and "mean_reliability" in road_metric.reliability_summary:
            reliability = float(road_metric.reliability_summary["mean_reliability"])

        uncertainty = None
        if road_metric.uncertainty_summary and "mean_uncertainty" in road_metric.uncertainty_summary:
            uncertainty = float(road_metric.uncertainty_summary["mean_uncertainty"])

        priority = InvestigationPriority.determine(
            severity, reliability, uncertainty, data_quality_status=data_quality_status
        )

        explanation = self._synthesize_road_explanation(
            road_metric, triggered_types, overall_score, severity, priority, evidence_list
        )

        time_window = None
        if road_metric.time_window_start is not None or road_metric.time_window_end is not None:
            time_window = {
                "start": road_metric.time_window_start,
                "end": road_metric.time_window_end,
                "duration_seconds": road_metric.duration_seconds,
            }

        return AnomalyResult(
            anomaly_id=f"anomaly_road_{rid}",
            entity_type="road",
            entity_id=rid,
            data_quality_status=data_quality_status,
            anomaly_types=triggered_types,
            overall_score=overall_score,
            severity=severity,
            evidence=evidence_list,
            reliability=reliability,
            uncertainty=uncertainty,
            investigation_priority=priority,
            explanation=explanation,
            timestamp_or_window=time_window,
            metadata={
                "from_node": from_n,
                "to_node": to_n,
                "capacity_vph": road_metric.capacity_vph,
                "contributing_trajectories": road_metric.contributing_trajectories_count,
            },
        )

    def _check_capacity_utilization(self, road_metric: RoadFlowMetric) -> AnomalyEvidence:
        """
        Check if road segment is operating over design capacity or under high utilization.
        """
        rid = road_metric.road_id

        if not road_metric.is_hourly_rate_valid or road_metric.utilization_ratio is None:
            return AnomalyEvidence(
                signal_type="road_over_capacity",
                signal_category=SignalCategory.NETWORK_ANOMALY,
                available=False,
                severity=SignalSeverity.NORMAL,
                explanation=f"Utilization evaluation unavailable: observation window duration not calibrated for road {rid}.",
            )

        util = road_metric.utilization_ratio
        cap = road_metric.capacity_vph
        vph = road_metric.expected_demand_vph or 0.0

        if util > 1.0:
            fraction = min(1.0, (util - 1.0) / 0.50)
            signal_score = 0.75 + 0.25 * fraction
            severity = SignalSeverity.EXTREME if util >= 1.25 else SignalSeverity.HIGH
            explanation = (
                f"Road '{rid}' is operating above design capacity: utilization is {util*100:.1f}% "
                f"({vph:.1f} vph vs design capacity {cap:.1f} vph)."
            )
        elif util >= 0.80:
            fraction = (util - 0.80) / 0.20
            signal_score = 0.50 + 0.25 * fraction
            severity = SignalSeverity.HIGH
            explanation = (
                f"Road '{rid}' operates near capacity: utilization is {util*100:.1f}% "
                f"({vph:.1f} vph vs capacity {cap:.1f} vph)."
            )
        elif util >= 0.60:
            fraction = (util - 0.60) / 0.20
            signal_score = 0.25 + 0.25 * fraction
            severity = SignalSeverity.ELEVATED
            explanation = f"Road '{rid}' has moderate utilization: {util*100:.1f}% of design capacity."
        else:
            signal_score = 0.0
            severity = SignalSeverity.NORMAL
            explanation = f"Road '{rid}' capacity utilization ({util*100:.1f}%) is within normal operational limits."

        return AnomalyEvidence(
            signal_type="road_over_capacity",
            signal_category=SignalCategory.NETWORK_ANOMALY,
            measured_value=util,
            baseline_value=1.0,
            threshold=0.80,
            signal_score=signal_score,
            severity=severity,
            available=True,
            explanation=explanation,
            metadata={"utilization_ratio": round(util, 4), "expected_demand_vph": round(vph, 2), "capacity_vph": cap},
        )

    def _check_demand_deviation(self, road_metric: RoadFlowMetric) -> AnomalyEvidence:
        """
        Check if road expected demand deviates substantially from configured baseline demand.
        """
        rid = road_metric.road_id
        expected_demand = self.baseline.get_expected_demand(rid)

        if expected_demand is None or expected_demand <= 0:
            return AnomalyEvidence(
                signal_type="high_demand_deviation",
                signal_category=SignalCategory.NETWORK_ANOMALY,
                available=False,
                severity=SignalSeverity.NORMAL,
                explanation=f"Baseline demand unavailable: no reference demand configured for road {rid}.",
            )

        actual_demand = (
            road_metric.expected_demand_vph
            if road_metric.is_hourly_rate_valid and road_metric.expected_demand_vph is not None
            else road_metric.expected_demand_in_window
        )

        mult = actual_demand / expected_demand

        if mult >= 2.0:
            signal_score = min(1.0, 0.65 + 0.35 * min(1.0, (mult - 2.0) / 2.0))
            severity = SignalSeverity.HIGH
            explanation = (
                f"Elevated demand on road '{rid}': flow ({actual_demand:.1f} vph) is {mult:.2f}x "
                f"configured baseline ({expected_demand:.1f} vph)."
            )
        elif mult >= 1.5:
            signal_score = 0.40 + 0.25 * ((mult - 1.5) / 0.5)
            severity = SignalSeverity.ELEVATED
            explanation = (
                f"Moderately elevated demand on road '{rid}': flow ({actual_demand:.1f} vph) is {mult:.2f}x "
                f"configured baseline ({expected_demand:.1f} vph)."
            )
        else:
            signal_score = 0.0
            severity = SignalSeverity.NORMAL
            explanation = f"Demand on road '{rid}' ({actual_demand:.1f}) is consistent with baseline ({expected_demand:.1f})."

        return AnomalyEvidence(
            signal_type="high_demand_deviation",
            signal_category=SignalCategory.NETWORK_ANOMALY,
            measured_value=actual_demand,
            baseline_value=expected_demand,
            threshold=expected_demand * 1.5,
            signal_score=signal_score,
            severity=severity,
            available=True,
            explanation=explanation,
            metadata={"actual_demand": actual_demand, "baseline_demand": expected_demand, "multiplier": round(mult, 2)},
        )

    def _synthesize_road_explanation(
        self,
        road_metric: RoadFlowMetric,
        triggered_types: List[str],
        overall_score: Optional[float],
        severity: AnomalySeverity,
        priority: InvestigationPriority,
        evidence_list: List[AnomalyEvidence],
    ) -> str:
        rid = road_metric.road_id
        from_n = road_metric.from_node
        to_n = road_metric.to_node

        if severity == AnomalySeverity.INSUFFICIENT_EVIDENCE:
            return (
                f"Cannot evaluate road flow anomaly for '{rid}' ({from_n}->{to_n}): "
                f"hourly utilization or baseline demand is unavailable."
            )

        if severity == AnomalySeverity.NORMAL:
            return (
                f"Road segment '{rid}' ({from_n}->{to_n}) operates within normal capacity limits; "
                f"no congestion or demand anomaly detected (score: {overall_score:.2f})."
            )

        reasons = [e.explanation for e in evidence_list if e.signal_score >= 0.30]
        reasons_text = "; ".join(reasons) if reasons else "Elevated operational metrics."
        score_str = f"{overall_score:.2f}" if overall_score is not None else "N/A"

        return (
            f"Road Anomaly Candidate '{rid}' ({from_n}->{to_n}) [Severity: {severity.value.upper()}, "
            f"Priority: {priority.value.upper()}]: Overall deviation score is {score_str}. "
            f"Reasons: {reasons_text}"
        )
