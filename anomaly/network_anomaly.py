"""
Network-Scale Anomaly Detector for UrbanTrack AI (Day 7).

Evaluates CityMobilityReport to identify macro-level network anomalies:
1. Critical Bottlenecks (intersections of high structural centrality and high utilization)
2. Network-Wide Congestion Concentration

CRITICAL SEMANTIC PRINCIPLES:
- High betweenness + high utilization is interpreted as 'network bottleneck candidate',
  NOT 'abnormal vehicle behavior'.
- Distinction between network condition and vehicle behavior is strictly maintained.
- Reuses Day 6 network centrality and priority road metrics without duplicating graph algorithms.
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
from schemas.mobility_schema import CityMobilityReport, RoadFlowMetric


class NetworkAnomalyDetector:
    """
    Detector for network-wide bottlenecks and structural congestion patterns.
    """

    def __init__(self, baseline: Optional[MobilityBaseline] = None) -> None:
        self.baseline = baseline or MobilityBaseline(source="unavailable")

    def evaluate_network(self, mobility_report: CityMobilityReport) -> List[AnomalyResult]:
        """
        Evaluate CityMobilityReport and return a list of network-level AnomalyResult records.
        """
        results: List[AnomalyResult] = []

        # 1. Identify Network Critical Bottlenecks
        bottleneck_result = self._check_critical_bottlenecks(mobility_report)
        if bottleneck_result:
            results.append(bottleneck_result)

        # 2. Check Network-Wide Congestion Concentration
        concentration_result = self._check_congestion_concentration(mobility_report)
        if concentration_result:
            results.append(concentration_result)

        return results

    def _check_critical_bottlenecks(self, report: CityMobilityReport) -> Optional[AnomalyResult]:
        """
        Detect roads that serve as critical topological bridges AND suffer from high utilization.
        """
        bottleneck_roads = []

        centrality_map = report.network_centrality
        for metric in report.road_metrics:
            if not metric.is_hourly_rate_valid or metric.utilization_ratio is None:
                continue

            rid = metric.road_id
            cent = centrality_map.get(rid)
            betweenness = cent.betweenness_centrality if cent else 0.0

            # High utilization combined with significant structural betweenness
            if metric.utilization_ratio >= 0.80 and betweenness > 0.04:
                bottleneck_roads.append({
                    "road_id": rid,
                    "utilization": round(metric.utilization_ratio, 4),
                    "betweenness": round(betweenness, 4),
                    "expected_demand_vph": round(metric.expected_demand_vph or 0.0, 1),
                })

        if not bottleneck_roads:
            return None

        # Sort bottlenecks by utilization descending
        bottleneck_roads.sort(key=lambda b: -b["utilization"])
        highest = bottleneck_roads[0]

        signal_score = min(1.0, 0.60 + 0.10 * len(bottleneck_roads))
        severity = AnomalySeverity.HIGH_PRIORITY if len(bottleneck_roads) >= 2 or highest["utilization"] > 1.0 else AnomalySeverity.INVESTIGATE
        priority = InvestigationPriority.HIGH_PRIORITY if severity == AnomalySeverity.HIGH_PRIORITY else InvestigationPriority.INVESTIGATE

        evidence = AnomalyEvidence(
            signal_type="network_bottleneck",
            signal_category=SignalCategory.NETWORK_ANOMALY,
            measured_value=float(len(bottleneck_roads)),
            baseline_value=0.0,
            threshold=1.0,
            signal_score=signal_score,
            severity=SignalSeverity.EXTREME if highest["utilization"] > 1.0 else SignalSeverity.HIGH,
            available=True,
            explanation=(
                f"Identified {len(bottleneck_roads)} critical network bottleneck candidate(s) where high structural centrality "
                f"coincides with severe capacity utilization. Primary critical corridor: '{highest['road_id']}' "
                f"(utilization: {highest['utilization']*100:.1f}%, betweenness centrality: {highest['betweenness']:.4f})."
            ),
            metadata={"bottlenecks": bottleneck_roads},
        )

        return AnomalyResult(
            anomaly_id="anomaly_network_critical_bottlenecks",
            entity_type="network",
            entity_id="city_network_bottlenecks",
            data_quality_status=DataQualityStatus.VALID,
            anomaly_types=["network_bottleneck"],
            overall_score=signal_score,
            severity=severity,
            evidence=[evidence],
            investigation_priority=priority,
            explanation=(
                f"Network Bottleneck Candidate [Severity: {severity.value.upper()}, Priority: {priority.value.upper()}]: "
                f"{len(bottleneck_roads)} key arterial bridge(s) are severely constrained under current traffic loading. "
                f"This reflects a network structural condition, not individual vehicle wrongdoing."
            ),
            timestamp_or_window=report.time_window,
            metadata={"affected_roads": [b["road_id"] for b in bottleneck_roads]},
        )

    def _check_congestion_concentration(self, report: CityMobilityReport) -> Optional[AnomalyResult]:
        """
        Evaluate the ratio of over-capacity segments relative to active road segments.
        """
        valid_metrics = [m for m in report.road_metrics if m.is_hourly_rate_valid and m.utilization_ratio is not None]
        if not valid_metrics:
            return None

        active_roads = [m for m in valid_metrics if (m.expected_demand_vph or 0.0) > 0.0]
        over_cap_roads = [m for m in valid_metrics if m.utilization_ratio > 1.0]

        if not over_cap_roads:
            return None

        over_cap_ratio = len(over_cap_roads) / max(len(active_roads), 1)

        signal_score = min(1.0, 0.50 + 0.50 * over_cap_ratio)
        severity = AnomalySeverity.HIGH_PRIORITY if over_cap_ratio >= 0.20 else AnomalySeverity.INVESTIGATE
        priority = InvestigationPriority(severity.value)

        evidence = AnomalyEvidence(
            signal_type="congestion_concentration",
            signal_category=SignalCategory.NETWORK_ANOMALY,
            measured_value=round(over_cap_ratio, 4),
            baseline_value=0.0,
            threshold=0.10,
            signal_score=signal_score,
            severity=SignalSeverity.HIGH,
            available=True,
            explanation=(
                f"{len(over_cap_roads)} / {len(active_roads)} active road segments ({over_cap_ratio*100:.1f}%) "
                f"are operating beyond physical capacity."
            ),
            metadata={"over_capacity_roads": [m.road_id for m in over_cap_roads]},
        )

        return AnomalyResult(
            anomaly_id="anomaly_network_congestion_concentration",
            entity_type="network",
            entity_id="city_congestion_spread",
            data_quality_status=DataQualityStatus.VALID,
            anomaly_types=["congestion_concentration"],
            overall_score=signal_score,
            severity=severity,
            evidence=[evidence],
            investigation_priority=priority,
            explanation=(
                f"Network Congestion Concentration [Severity: {severity.value.upper()}]: "
                f"{len(over_cap_roads)} road segments exceed design capacity ({over_cap_ratio*100:.1f}% of active corridors)."
            ),
            timestamp_or_window=report.time_window,
            metadata={"over_capacity_count": len(over_cap_roads), "active_count": len(active_roads)},
        )
