"""
Investigation Engine for UrbanTrack AI (Day 7).

Unified orchestrator for city-scale anomaly detection and explainable investigation.
Coordinates TrajectoryAnomalyDetector, RoadAnomalyDetector, and NetworkAnomalyDetector
to produce transparent, multi-level CityAnomalyReport outputs.

CORE OBJECTIVE:
Synthesizes explainable answers to:
1. What is unusual?
2. Where did it happen?
3. Which vehicle/road/network component is involved?
4. What evidence triggered the alert?
5. How strong is the evidence?
6. How uncertain is the inference?
7. Why was the anomaly classified at that severity?
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union

from schemas.anomaly_schema import (
    AnomalyEvidence,
    AnomalyResult,
    AnomalySeverity,
    CityAnomalyReport,
    DataQualityStatus,
    InvestigationPriority,
    MobilityBaseline,
)
from schemas.mobility_schema import CityMobilityReport, RoadFlowMetric
from schemas.normalized_trajectory_schema import NormalizedTrajectory

from .network_anomaly import NetworkAnomalyDetector
from .road_anomaly import RoadAnomalyDetector
from .trajectory_anomaly import TrajectoryAnomalyDetector


class InvestigationEngine:
    """
    Unified anomaly investigation engine for UrbanTrack AI.
    """

    def __init__(
        self,
        baseline: Optional[MobilityBaseline] = None,
        road_graph: Optional[Any] = None,
    ) -> None:
        self.baseline = baseline or MobilityBaseline(source="unavailable")
        self.road_graph = road_graph
        self.trajectory_detector = TrajectoryAnomalyDetector(
            baseline=self.baseline,
            max_physical_speed_kmh=self.baseline.tolerance_factors.get("max_physical_speed_kmh", 120.0),
        )
        self.road_detector = RoadAnomalyDetector(baseline=self.baseline)
        self.network_detector = NetworkAnomalyDetector(baseline=self.baseline)

    def investigate_trajectory(self, trajectory: NormalizedTrajectory) -> AnomalyResult:
        """
        Investigate a single vehicle trajectory.
        """
        return self.trajectory_detector.evaluate_trajectory(trajectory, road_graph=self.road_graph)

    def investigate_road(self, road_metric: RoadFlowMetric) -> AnomalyResult:
        """
        Investigate a single road segment flow metric.
        """
        return self.road_detector.evaluate_road(road_metric)

    def investigate_network(self, mobility_report: CityMobilityReport) -> List[AnomalyResult]:
        """
        Investigate macro-level network bottlenecks and concentration.
        """
        return self.network_detector.evaluate_network(mobility_report)

    def run_investigation(
        self,
        trajectories: Sequence[NormalizedTrajectory],
        mobility_report: Optional[CityMobilityReport] = None,
        time_window: Optional[Dict[str, Any]] = None,
    ) -> CityAnomalyReport:
        """
        Execute full city-scale anomaly detection across trajectories, roads, and network.

        Args:
            trajectories: Sequence of NormalizedTrajectory objects.
            mobility_report: Optional precomputed Day 6 CityMobilityReport.
            time_window: Optional time window metadata.

        Returns:
            CityAnomalyReport: Consolidated, machine-readable investigation report.
        """
        # 1. Investigate Trajectories
        vehicle_results: List[AnomalyResult] = []
        for traj in trajectories:
            res = self.investigate_trajectory(traj)
            vehicle_results.append(res)

        # 2. Investigate Roads & Network if mobility report is available
        road_results: List[AnomalyResult] = []
        network_results: List[AnomalyResult] = []

        if mobility_report:
            time_window = time_window or mobility_report.time_window
            for road_metric in mobility_report.road_metrics:
                r_res = self.investigate_road(road_metric)
                if r_res.is_anomalous:
                    road_results.append(r_res)

            network_results = self.investigate_network(mobility_report)

        time_window = time_window or {"start": 0.0, "end": 0.0, "source": "uncalibrated"}

        # Sort results by severity / priority descending
        severity_order = {
            AnomalySeverity.HIGH_PRIORITY: 4,
            AnomalySeverity.INVESTIGATE: 3,
            AnomalySeverity.WATCH: 2,
            AnomalySeverity.NORMAL: 1,
            AnomalySeverity.INSUFFICIENT_EVIDENCE: 0,
            AnomalySeverity.INVALID_INPUT: -1,
        }
        vehicle_results.sort(
            key=lambda r: (
                -severity_order.get(r.severity, 0),
                -(r.overall_score if r.overall_score is not None else -1.0),
            )
        )
        road_results.sort(
            key=lambda r: (
                -severity_order.get(r.severity, 0),
                -(r.overall_score if r.overall_score is not None else -1.0),
            )
        )

        anomalous_vehicles = [v for v in vehicle_results if v.is_anomalous]

        summary = {
            "total_trajectories_evaluated": len(trajectories),
            "anomalous_trajectories_count": len(anomalous_vehicles),
            "vehicle_severity_breakdown": {
                "high_priority": sum(1 for v in vehicle_results if v.severity == AnomalySeverity.HIGH_PRIORITY),
                "investigate": sum(1 for v in vehicle_results if v.severity == AnomalySeverity.INVESTIGATE),
                "watch": sum(1 for v in vehicle_results if v.severity == AnomalySeverity.WATCH),
                "normal": sum(1 for v in vehicle_results if v.severity == AnomalySeverity.NORMAL),
                "insufficient_evidence": sum(1 for v in vehicle_results if v.severity == AnomalySeverity.INSUFFICIENT_EVIDENCE),
                "invalid_input": sum(1 for v in vehicle_results if v.severity == AnomalySeverity.INVALID_INPUT),
            },
            "data_quality_breakdown": {
                "valid": sum(1 for v in vehicle_results if v.data_quality_status == DataQualityStatus.VALID),
                "incomplete": sum(1 for v in vehicle_results if v.data_quality_status == DataQualityStatus.INCOMPLETE),
                "invalid": sum(1 for v in vehicle_results if v.data_quality_status == DataQualityStatus.INVALID),
            },
            "anomalous_roads_count": len(road_results),
            "network_anomalies_count": len(network_results),
            "baseline_source": self.baseline.source,
        }

        return CityAnomalyReport(
            time_window=time_window,
            vehicle_anomalies=vehicle_results,
            road_anomalies=road_results,
            network_anomalies=network_results,
            summary=summary,
        )
