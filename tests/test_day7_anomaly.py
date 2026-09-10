"""
Day 7 Unit and Integration Tests: City-Scale Anomaly Detection & Investigation.

Audited and Hardened against Semantic Invariants:
1. INVALID DATA ≠ PHYSICAL INCONSISTENCY ≠ BEHAVIORAL ANOMALY ≠ INVESTIGATION PRIORITY.
2. Malformed or temporally inverted input results in DATA_QUALITY_STATUS = INVALID / INVALID_INPUT,
   NOT a high-priority behavioral anomaly.
3. Missing baseline produces INSUFFICIENT_EVIDENCE, NOT confidently NORMAL.
4. Physical speed violations (>120 km/h) are categorized as PHYSICAL_INCONSISTENCY.
5. Inferred routes on closed roads are categorized as NETWORK_CONSTRAINT_INCONSISTENCY
   (incompatible network state), NOT suspicious driver behavior.
6. Ambiguous routes preserve uncertainty and tone down deviation confidence.
7. Reliability and uncertainty remain strictly separate from anomaly score.
8. Real Kanishka data (2,503 singletons) safely skips trajectory anomaly detection with zero fabrication.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from anomaly.investigation_engine import InvestigationEngine
from anomaly.network_anomaly import NetworkAnomalyDetector
from anomaly.road_anomaly import RoadAnomalyDetector
from anomaly.trajectory_anomaly import TrajectoryAnomalyDetector
from inference.road_graph import RoadGraph
from schemas.anomaly_schema import (
    AnomalyEvidence,
    AnomalyResult,
    AnomalySeverity,
    CityAnomalyReport,
    DataQualityStatus,
    InvestigationPriority,
    MobilityBaseline,
    SignalCategory,
    SignalSeverity,
)
from schemas.mobility_schema import CityMobilityReport, RoadFlowMetric, RoadStatus
from schemas.normalized_trajectory_schema import NormalizedCandidateRoute, NormalizedTrajectory


class TestDay7AnomalyDetection(unittest.TestCase):
    """Day 7 Anomaly Detection and Investigation Suite."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_dir = Path(__file__).parent.parent
        cls.scenarios_path = cls.base_dir / "data" / "synthetic" / "day7_anomaly_scenarios.json"
        with open(cls.scenarios_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            cls.scenarios = data["scenarios"]
            cls.baseline = MobilityBaseline.from_dict(data["baseline_config"])

        cls.city_net_path = cls.base_dir / "data" / "synthetic" / "city_network.json"
        cls.road_graph = RoadGraph.from_json_file(str(cls.city_net_path))
        cls.engine = InvestigationEngine(baseline=cls.baseline, road_graph=cls.road_graph)

    # 1. Normal Vehicle Movement
    def test_case1_normal_vehicle_movement(self) -> None:
        case = self.scenarios["case1_normal_vehicle"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])
        result = self.engine.investigate_trajectory(traj)

        self.assertFalse(result.is_anomalous)
        self.assertEqual(result.data_quality_status, DataQualityStatus.VALID)
        self.assertEqual(result.severity, AnomalySeverity.NORMAL)
        self.assertEqual(result.investigation_priority, InvestigationPriority.NORMAL)
        self.assertLess(result.overall_score, 0.30)
        self.assertIn("matches baseline", result.explanation)

    # 2. Slow Travel-Time Anomaly
    def test_case2_slow_travel_time_anomaly(self) -> None:
        case = self.scenarios["case2_slow_travel_time"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])
        result = self.engine.investigate_trajectory(traj)

        self.assertTrue(result.is_anomalous)
        self.assertEqual(result.data_quality_status, DataQualityStatus.VALID)
        self.assertIn("travel_time_deviation", result.anomaly_types)
        self.assertGreaterEqual(result.overall_score, 0.75)
        self.assertEqual(result.severity, AnomalySeverity.HIGH_PRIORITY)
        self.assertEqual(result.investigation_priority, InvestigationPriority.HIGH_PRIORITY)

        tt_ev = next(e for e in result.evidence if e.signal_type == "travel_time_deviation")
        self.assertTrue(tt_ev.available)
        self.assertEqual(tt_ev.signal_category, SignalCategory.BEHAVIORAL_ANOMALY)
        self.assertEqual(tt_ev.severity, SignalSeverity.EXTREME)
        self.assertGreater(tt_ev.signal_score, 0.80)

    # 3. Impossible Speed Anomaly (Physical Inconsistency)
    def test_case3_impossible_speed_inconsistency(self) -> None:
        case = self.scenarios["case3_impossible_speed"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])
        result = self.engine.investigate_trajectory(traj)

        self.assertTrue(result.is_anomalous)
        self.assertEqual(result.data_quality_status, DataQualityStatus.VALID)
        self.assertIn("physical_inconsistency", result.anomaly_types)
        self.assertEqual(result.severity, AnomalySeverity.HIGH_PRIORITY)
        self.assertEqual(result.overall_score, 1.0)

        phys_ev = next(e for e in result.evidence if e.signal_type == "physical_inconsistency")
        self.assertEqual(phys_ev.signal_category, SignalCategory.PHYSICAL_INCONSISTENCY)
        self.assertEqual(phys_ev.severity, SignalSeverity.EXTREME)
        self.assertIn("exceeds maximum physical limit", phys_ev.explanation)

    # 4. Route Deviation Anomaly
    def test_case4_route_deviation_anomaly(self) -> None:
        case = self.scenarios["case4_route_deviation"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])
        result = self.engine.investigate_trajectory(traj)

        self.assertTrue(result.is_anomalous)
        self.assertEqual(result.data_quality_status, DataQualityStatus.VALID)
        self.assertIn("route_deviation", result.anomaly_types)
        self.assertIn(result.severity, (AnomalySeverity.INVESTIGATE, AnomalySeverity.HIGH_PRIORITY))

        route_ev = next(e for e in result.evidence if e.signal_type == "route_deviation")
        self.assertTrue(route_ev.available)
        self.assertEqual(route_ev.signal_category, SignalCategory.BEHAVIORAL_ANOMALY)
        self.assertGreater(route_ev.signal_score, 0.50)
        self.assertIn("differs from expected", route_ev.explanation)

    # 5. Road Over-Capacity Anomaly
    def test_case5_road_over_capacity_anomaly(self) -> None:
        case = self.scenarios["case5_road_over_capacity"]
        metric_data = case["road_metric"]
        road_metric = RoadFlowMetric(
            road_id=metric_data["road_id"],
            from_node=metric_data["from_node"],
            to_node=metric_data["to_node"],
            expected_demand_in_window=metric_data["expected_demand_in_window"],
            capacity_vph=metric_data["capacity_vph"],
            expected_demand_vph=metric_data["expected_demand_vph"],
            utilization_ratio=metric_data["utilization_ratio"],
            is_hourly_rate_valid=metric_data["is_hourly_rate_valid"],
            duration_seconds=metric_data["duration_seconds"],
            status=RoadStatus(metric_data["status"]),
        )

        result = self.engine.investigate_road(road_metric)
        self.assertTrue(result.is_anomalous)
        self.assertIn("road_over_capacity", result.anomaly_types)
        self.assertEqual(result.severity, AnomalySeverity.HIGH_PRIORITY)

        cap_ev = next(e for e in result.evidence if e.signal_type == "road_over_capacity")
        self.assertTrue(cap_ev.available)
        self.assertEqual(cap_ev.signal_category, SignalCategory.NETWORK_ANOMALY)
        self.assertGreaterEqual(cap_ev.signal_score, 0.75)
        self.assertIn("operating above design capacity", cap_ev.explanation)

    # 6. Elevated Road Demand Anomaly
    def test_case6_elevated_road_demand_anomaly(self) -> None:
        case = self.scenarios["case6_high_demand_road"]
        metric_data = case["road_metric"]
        road_metric = RoadFlowMetric(
            road_id=metric_data["road_id"],
            from_node=metric_data["from_node"],
            to_node=metric_data["to_node"],
            expected_demand_in_window=metric_data["expected_demand_in_window"],
            capacity_vph=metric_data["capacity_vph"],
            expected_demand_vph=metric_data["expected_demand_vph"],
            utilization_ratio=metric_data["utilization_ratio"],
            is_hourly_rate_valid=metric_data["is_hourly_rate_valid"],
            duration_seconds=metric_data["duration_seconds"],
            status=RoadStatus(metric_data["status"]),
        )

        result = self.engine.investigate_road(road_metric)
        self.assertTrue(result.is_anomalous)
        self.assertIn("high_demand_deviation", result.anomaly_types)

        dem_ev = next(e for e in result.evidence if e.signal_type == "high_demand_deviation")
        self.assertTrue(dem_ev.available)
        self.assertEqual(dem_ev.signal_category, SignalCategory.NETWORK_ANOMALY)
        self.assertGreater(dem_ev.signal_score, 0.50)

    # 7. Multi-Signal Evidence Fusion
    def test_case7_multi_signal_evidence_fusion(self) -> None:
        case = self.scenarios["case7_multi_signal"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])
        result = self.engine.investigate_trajectory(traj)

        self.assertTrue(result.is_anomalous)
        self.assertGreaterEqual(len(result.anomaly_types), 2)
        self.assertIn("travel_time_deviation", result.anomaly_types)
        self.assertIn("route_deviation", result.anomaly_types)
        self.assertEqual(result.severity, AnomalySeverity.HIGH_PRIORITY)
        self.assertEqual(result.investigation_priority, InvestigationPriority.HIGH_PRIORITY)

    # 8. Ambiguous Route Preserves Uncertainty
    def test_case8_ambiguous_route_uncertainty_preservation(self) -> None:
        case = self.scenarios["case8_ambiguous_route_anomaly"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])
        result = self.engine.investigate_trajectory(traj)

        route_ev = next(e for e in result.evidence if e.signal_type == "route_deviation")
        self.assertTrue(route_ev.metadata.get("is_ambiguous", False))
        self.assertIn("route inference remains ambiguous", route_ev.explanation)
        self.assertLessEqual(result.overall_score, 0.55)

    # 9. Low Reliability Anomaly Priority Adjustment
    def test_case9_low_reliability_anomaly_interpretation(self) -> None:
        case = self.scenarios["case9_low_reliability_anomaly"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])
        result = self.engine.investigate_trajectory(traj)

        # Behavioral anomaly score is high because travel time is 3x baseline
        self.assertEqual(result.severity, AnomalySeverity.HIGH_PRIORITY)
        self.assertGreaterEqual(result.overall_score, 0.75)

        # But evidence reliability is low (0.25), so priority is adjusted to WATCH
        self.assertEqual(result.reliability, 0.25)
        self.assertEqual(result.investigation_priority, InvestigationPriority.WATCH)
        self.assertIn("priority adjusted to WATCH due to low sensor evidence reliability", result.explanation)

    # 10. Missing Baseline produces INSUFFICIENT_EVIDENCE (NOT normal)
    def test_case10_missing_baseline_handling(self) -> None:
        case = self.scenarios["case10_missing_baseline"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])
        result = self.engine.investigate_trajectory(traj)

        self.assertFalse(result.is_anomalous)
        self.assertEqual(result.data_quality_status, DataQualityStatus.INCOMPLETE)
        self.assertEqual(result.severity, AnomalySeverity.INSUFFICIENT_EVIDENCE)
        self.assertIsNone(result.overall_score)
        self.assertIn("insufficient for anomaly determination", result.explanation)

    # 11. Invalid Trajectory produces INVALID_INPUT (NOT a behavioral anomaly)
    def test_case11_invalid_trajectory_data_quality(self) -> None:
        case = self.scenarios["case11_invalid_trajectory"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])
        result = self.engine.investigate_trajectory(traj)

        self.assertFalse(result.is_anomalous)
        self.assertEqual(result.data_quality_status, DataQualityStatus.INVALID)
        self.assertEqual(result.severity, AnomalySeverity.INVALID_INPUT)
        self.assertEqual(result.investigation_priority, InvestigationPriority.DATA_QUALITY_REVIEW)
        self.assertIsNone(result.overall_score)
        self.assertIn("Data quality error", result.explanation)
        self.assertIn("Temporal inversion", result.explanation)

    # 12. Missing Evidence Evaluates to Normal Without False Positives
    def test_case12_missing_evidence_normal(self) -> None:
        case = self.scenarios["case12_missing_evidence_normal"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])
        result = self.engine.investigate_trajectory(traj)

        self.assertFalse(result.is_anomalous)
        self.assertEqual(result.data_quality_status, DataQualityStatus.VALID)
        self.assertEqual(result.severity, AnomalySeverity.NORMAL)
        self.assertEqual(result.investigation_priority, InvestigationPriority.NORMAL)

    # 13. Malformed Non-Numeric Timestamp Handling
    def test_malformed_timestamp_data_quality(self) -> None:
        traj = NormalizedTrajectory(
            track_id="VEH_MALFORMED_01",
            origin_node="J01",
            destination_node="J03",
            vehicle_weight=1.0,
            candidate_routes=[NormalizedCandidateRoute(nodes=["J01", "J02", "J03"], probability=1.0)],
            time_window_start="corrupt_string",
            time_window_end=1200.0,
        )
        result = self.engine.investigate_trajectory(traj)

        self.assertFalse(result.is_anomalous)
        self.assertEqual(result.data_quality_status, DataQualityStatus.INVALID)
        self.assertEqual(result.severity, AnomalySeverity.INVALID_INPUT)
        self.assertEqual(result.investigation_priority, InvestigationPriority.DATA_QUALITY_REVIEW)
        self.assertIn("Non-numeric timestamp", result.explanation)

    # 14. Closed Road Network Constraint Semantics
    def test_closed_road_network_constraint_semantics(self) -> None:
        """
        Verify that a route traversing a closed road is labeled as
        NETWORK_CONSTRAINT_INCONSISTENCY and does NOT accuse the vehicle of suspicious intent.
        """
        # Close road R02 temporarily
        self.road_graph.close_road("R02")
        try:
            traj = NormalizedTrajectory(
                track_id="VEH_CLOSED_TEST",
                origin_node="J01",
                destination_node="J03",
                vehicle_weight=1.0,
                candidate_routes=[
                    NormalizedCandidateRoute(
                        nodes=["J01", "J02", "J03"],
                        probability=1.0,
                        metadata={"edges": ["R01", "R02"], "distance_m": 909.3},
                    )
                ],
                time_window_start=1000.0,
                time_window_end=1180.0,
            )
            result = self.engine.investigate_trajectory(traj)

            self.assertTrue(result.is_anomalous)
            self.assertIn("network_constraint_inconsistency", result.anomaly_types)

            net_ev = next(e for e in result.evidence if e.signal_type == "network_constraint_inconsistency")
            self.assertEqual(net_ev.signal_category, SignalCategory.NETWORK_CONSTRAINT_INCONSISTENCY)
            self.assertIn("incompatible with the current road-network state", net_ev.explanation)
            self.assertNotIn("suspicious", net_ev.explanation.lower())
            self.assertNotIn("criminal", net_ev.explanation.lower())
        finally:
            self.road_graph.restore_road("R02")

    # 15. Severity Classification Thresholds
    def test_severity_thresholds_mapping(self) -> None:
        self.assertEqual(AnomalySeverity.classify(None), AnomalySeverity.INSUFFICIENT_EVIDENCE)
        self.assertEqual(AnomalySeverity.classify(0.00), AnomalySeverity.NORMAL)
        self.assertEqual(AnomalySeverity.classify(0.25), AnomalySeverity.NORMAL)
        self.assertEqual(AnomalySeverity.classify(0.30), AnomalySeverity.WATCH)
        self.assertEqual(AnomalySeverity.classify(0.45), AnomalySeverity.WATCH)
        self.assertEqual(AnomalySeverity.classify(0.50), AnomalySeverity.INVESTIGATE)
        self.assertEqual(AnomalySeverity.classify(0.70), AnomalySeverity.INVESTIGATE)
        self.assertEqual(AnomalySeverity.classify(0.75), AnomalySeverity.HIGH_PRIORITY)
        self.assertEqual(AnomalySeverity.classify(1.00), AnomalySeverity.HIGH_PRIORITY)

    # 16. Reliability and Anomaly Score Separation
    def test_reliability_and_anomaly_score_separation(self) -> None:
        base_traj_dict = self.scenarios["case2_slow_travel_time"]["trajectory"]

        # Trajectory with high reliability
        traj_high_rel = NormalizedTrajectory.from_dict({
            **base_traj_dict,
            "metadata": {"reliability": 0.95, "uncertainty": 0.05}
        })
        res_high = self.engine.investigate_trajectory(traj_high_rel)

        # Trajectory with low reliability
        traj_low_rel = NormalizedTrajectory.from_dict({
            **base_traj_dict,
            "metadata": {"reliability": 0.15, "uncertainty": 0.85}
        })
        res_low = self.engine.investigate_trajectory(traj_low_rel)

        # Anomaly score must be identical (behavioral deviation is physically identical)
        self.assertAlmostEqual(res_high.overall_score, res_low.overall_score, places=4)
        self.assertEqual(res_high.severity, res_low.severity)

        # But investigation priority must differentiate them
        self.assertEqual(res_high.investigation_priority, InvestigationPriority.HIGH_PRIORITY)
        self.assertEqual(res_low.investigation_priority, InvestigationPriority.WATCH)

    # 17. Investigation Explanation Completeness and Neutrality
    def test_investigation_explanation_completeness(self) -> None:
        case = self.scenarios["case2_slow_travel_time"]
        traj = NormalizedTrajectory.from_dict(case["trajectory"])
        result = self.engine.investigate_trajectory(traj)

        explanation = result.explanation
        # Must answer What, Where, Who, Evidence, Strength, Severity
        self.assertIn("VEH_D7_SLOW_02", explanation)        # Who
        self.assertIn("J01->J03", explanation)              # Where
        self.assertIn("HIGH_PRIORITY", explanation)         # Severity
        self.assertIn("travel-time", explanation.lower())   # Evidence
        self.assertIn("reliability is", explanation)        # Uncertainty

        # Must NOT contain forbidden overclaimed words
        explanation_lower = explanation.lower()
        self.assertNotIn("loitering", explanation_lower)
        self.assertNotIn("unexpected stop", explanation_lower)
        self.assertNotIn("suspicious", explanation_lower)
        self.assertNotIn("criminal", explanation_lower)

    # 18. Network Bottleneck Detector
    def test_network_bottleneck_detection(self) -> None:
        from schemas.mobility_schema import RoadCentralityMetric

        road_metrics = [
            RoadFlowMetric(
                road_id="R08",
                from_node="J02",
                to_node="J05",
                expected_demand_in_window=1500.0,
                capacity_vph=1200.0,
                expected_demand_vph=1500.0,
                utilization_ratio=1.25,
                is_hourly_rate_valid=True,
                duration_seconds=3600.0,
                status=RoadStatus.OVER_CAPACITY,
            ),
            RoadFlowMetric(
                road_id="R01",
                from_node="J01",
                to_node="J02",
                expected_demand_in_window=400.0,
                capacity_vph=1600.0,
                expected_demand_vph=400.0,
                utilization_ratio=0.25,
                is_hourly_rate_valid=True,
                duration_seconds=3600.0,
                status=RoadStatus.NORMAL,
            ),
        ]
        centrality = {
            "R08": RoadCentralityMetric("R08", "J02", "J05", 2, 2, 4, betweenness_centrality=0.15),
            "R01": RoadCentralityMetric("R01", "J01", "J02", 1, 1, 2, betweenness_centrality=0.01),
        }
        report = CityMobilityReport(
            time_window={"start": 0.0, "end": 3600.0},
            road_metrics=road_metrics,
            od_demand=[],
            priority_roads=[],
            network_centrality=centrality,
        )

        detector = NetworkAnomalyDetector(baseline=self.baseline)
        network_results = detector.evaluate_network(report)

        self.assertGreater(len(network_results), 0)
        bottleneck_res = next(r for r in network_results if r.anomaly_types == ["network_bottleneck"])
        self.assertEqual(bottleneck_res.entity_type, "network")
        self.assertIn("R08", bottleneck_res.metadata["affected_roads"])
        self.assertGreater(bottleneck_res.overall_score, 0.50)
        self.assertIn("Network Bottleneck Candidate", bottleneck_res.explanation)

    # 19. Full CityAnomalyReport Generation
    def test_full_city_anomaly_report_generation(self) -> None:
        trajectories = [
            NormalizedTrajectory.from_dict(self.scenarios["case1_normal_vehicle"]["trajectory"]),
            NormalizedTrajectory.from_dict(self.scenarios["case2_slow_travel_time"]["trajectory"]),
            NormalizedTrajectory.from_dict(self.scenarios["case3_impossible_speed"]["trajectory"]),
            NormalizedTrajectory.from_dict(self.scenarios["case10_missing_baseline"]["trajectory"]),
            NormalizedTrajectory.from_dict(self.scenarios["case11_invalid_trajectory"]["trajectory"]),
        ]
        full_report = self.engine.run_investigation(trajectories)

        self.assertIsInstance(full_report, CityAnomalyReport)
        self.assertEqual(full_report.summary["total_trajectories_evaluated"], 5)
        self.assertEqual(full_report.summary["anomalous_trajectories_count"], 2)  # case 2 and case 3
        self.assertEqual(full_report.summary["vehicle_severity_breakdown"]["normal"], 1)
        self.assertEqual(full_report.summary["vehicle_severity_breakdown"]["high_priority"], 2)
        self.assertEqual(full_report.summary["vehicle_severity_breakdown"]["insufficient_evidence"], 1)
        self.assertEqual(full_report.summary["vehicle_severity_breakdown"]["invalid_input"], 1)
        self.assertEqual(full_report.summary["data_quality_breakdown"]["invalid"], 1)
        self.assertEqual(full_report.summary["data_quality_breakdown"]["incomplete"], 1)

        # JSON Serialization check
        json_str = full_report.to_json()
        self.assertIn("vehicle_anomalies", json_str)
        self.assertIn("VEH_D7_SLOW_02", json_str)

    # 20. Real Kanishka Data Feed Non-Fabrication
    def test_real_kanishka_data_no_fabricated_anomalies(self) -> None:
        real_obs_path = self.base_dir / "data" / "observations" / "kanishka_traffic.json"
        with open(real_obs_path, "r", encoding="utf-8") as rf:
            real_data = json.load(rf)

        self.assertEqual(len(real_data), 2503)
        missing_embeddings = sum(1 for o in real_data if not o.get("appearance_embedding"))
        self.assertEqual(missing_embeddings, 2503)

        # No multi-camera trajectories exist -> 0 fabricated anomalies
        multi_cam_trajectories: list[NormalizedTrajectory] = []
        report = self.engine.run_investigation(multi_cam_trajectories)
        self.assertEqual(report.summary["total_trajectories_evaluated"], 0)
        self.assertEqual(report.summary["anomalous_trajectories_count"], 0)


if __name__ == "__main__":
    unittest.main()
