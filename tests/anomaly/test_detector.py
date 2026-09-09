"""Tests for evidence-based Phase 5 anomaly detection."""

import unittest

from backend.analytics.models import Bottleneck
from backend.anomaly.detector import AnomalyDetector, AnomalyDetectorConfig
from backend.anomaly.models import MobilitySnapshot, RoadObservationStatus
from backend.mobility.graph import MobilityGraph
from backend.mobility.models import RoadSegment


class TestAnomalyDetector(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = MobilityGraph(name="AnomalyTest")
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 1.0, 40.0, 1000.0))
        self.graph.add_road(RoadSegment("R02", "J02", "J03", 1.0, 40.0, 1000.0))
        self.graph.add_road(RoadSegment("R03", "J10", "J11", 1.0, 40.0, 1000.0))

    def snapshot(self, snapshot_id: str, flows: dict, od: dict | None = None, routes: dict | None = None, hhi: float = 0.5, bottlenecks: tuple = (), duration: float = 0.25, network: str = "network") -> MobilitySnapshot:
        return MobilitySnapshot(
            snapshot_id=snapshot_id,
            network_identity=network,
            aggregation_duration_hours=duration,
            road_hourly_flows=flows,
            od_demand=od or {("J01", "J03"): 1.0},
            route_demand=routes or {("R01", "R02"): 1.0},
            hhi=hhi,
            bottlenecks=bottlenecks,
            time_window_start="08:00:00",
            time_window_end="08:15:00",
        )

    def bottleneck(self, road_id: str, severity: str) -> Bottleneck:
        return Bottleneck(road_id, severity, 1.0, 0.1, 100.0, 100.0, 1.0, 2.0)

    def test_normal_change_and_deterministic_order(self) -> None:
        baseline = self.snapshot("b", {"R01": 100.0, "R02": 100.0})
        current = self.snapshot("c", {"R01": 110.0, "R02": 90.0})
        result = AnomalyDetector().compare(baseline, current, self.graph)
        self.assertEqual(result.anomaly_events, ())
        self.assertEqual([item.road_id for item in result.road_evidence], ["R01", "R02"])

    def test_surge_drop_near_zero_and_missing_observations(self) -> None:
        baseline = self.snapshot("b", {"R01": 100.0, "R02": 0.0, "R03": 50.0})
        current = self.snapshot("c", {"R01": 180.0, "R02": 10.0, "R04": 20.0})
        result = AnomalyDetector().compare(baseline, current, self.graph)
        by_id = {item.road_id: item for item in result.road_evidence}
        self.assertEqual(by_id["R01"].flow_signal, "FLOW_SURGE")
        self.assertEqual(by_id["R02"].flow_signal, "FLOW_SURGE")
        self.assertEqual(by_id["R03"].status, RoadObservationStatus.MISSING_CURRENT)
        self.assertEqual(by_id["R04"].status, RoadObservationStatus.MISSING_BASELINE)
        self.assertIsNone(by_id["R03"].absolute_change)

    def test_distribution_hhi_and_bottleneck_signals(self) -> None:
        baseline = self.snapshot(
            "b", {"R01": 100.0, "R02": 100.0},
            od={("J01", "J03"): 1.0},
            routes={("R01", "R02"): 1.0},
            hhi=0.5,
            bottlenecks=(self.bottleneck("R01", "FREE"),),
        )
        current = self.snapshot(
            "c", {"R01": 180.0, "R02": 20.0},
            od={("J10", "J11"): 1.0},
            routes={("R03",): 1.0},
            hhi=0.8,
            bottlenecks=(self.bottleneck("R01", "SEVERE"), self.bottleneck("R02", "HEAVY")),
        )
        result = AnomalyDetector().compare(baseline, current, self.graph)
        signal_names = {event.anomaly_type for event in result.anomaly_events}
        self.assertIn("OD_SHIFT", signal_names)
        self.assertIn("ROUTE_REDISTRIBUTION", signal_names)
        self.assertIn("NETWORK_CONCENTRATION_SHIFT", signal_names)
        self.assertIn("BOTTLENECK_SHIFT", signal_names)
        self.assertIn("MOBILITY_STATE_CHANGE", signal_names)
        self.assertGreaterEqual(result.anomaly_events[-1].evidence_count, 4)
        self.assertEqual(result.bottleneck_changes[0].change_type, "SEVERITY_INCREASE")

    def test_consensus_severity_escalates_and_compatibility_is_strict(self) -> None:
        config = AnomalyDetectorConfig(flow_relative_threshold=0.2, flow_absolute_threshold=1.0)
        baseline = self.snapshot("b", {"R01": 100.0}, hhi=0.1, duration=0.25)
        current = self.snapshot("c", {"R01": 200.0}, hhi=0.3, duration=0.5)
        with self.assertRaises(ValueError):
            AnomalyDetector(config).compare(baseline, current, self.graph)
        current = self.snapshot("c", {"R01": 200.0}, hhi=0.3, duration=0.25, network="other")
        with self.assertRaises(ValueError):
            AnomalyDetector(config).compare(baseline, current, self.graph)


if __name__ == "__main__":
    unittest.main()
