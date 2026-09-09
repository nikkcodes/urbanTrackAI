"""Tests for Phase 5 snapshot and result models."""

import unittest

from backend.analytics.models import Bottleneck
from backend.anomaly.models import MobilitySnapshot, RoadObservationStatus, RoadAnomalyEvidence


class TestMobilitySnapshot(unittest.TestCase):
    def base_kwargs(self) -> dict:
        return {
            "snapshot_id": "base",
            "network_identity": "network-a",
            "aggregation_duration_hours": 0.25,
            "road_hourly_flows": {"R01": 100.0},
            "od_demand": {("J01", "J02"): 2.0},
            "route_demand": {("R01",): 2.0},
            "hhi": 1.0,
            "time_window_start": "08:00:00",
            "time_window_end": "08:15:00",
        }

    def test_valid_snapshot_and_serialization(self) -> None:
        snapshot = MobilitySnapshot(**self.base_kwargs())
        self.assertEqual(snapshot.to_dict()["snapshot_id"], "base")
        self.assertEqual(snapshot.road_hourly_flows["R01"], 100.0)

    def test_invalid_snapshot_values_and_window(self) -> None:
        kwargs = self.base_kwargs()
        kwargs["aggregation_duration_hours"] = 0.0
        with self.assertRaises(ValueError):
            MobilitySnapshot(**kwargs)
        kwargs = self.base_kwargs()
        kwargs["road_hourly_flows"] = {"R01": -1.0}
        with self.assertRaises(ValueError):
            MobilitySnapshot(**kwargs)
        kwargs = self.base_kwargs()
        kwargs["time_window_end"] = "08:00:00"
        with self.assertRaises(ValueError):
            MobilitySnapshot(**kwargs)

    def test_road_evidence_missing_states_are_explicit(self) -> None:
        evidence = RoadAnomalyEvidence(
            road_id="R01",
            status=RoadObservationStatus.MISSING_CURRENT,
            baseline_hourly_flow=100.0,
            current_hourly_flow=None,
            absolute_change=None,
            relative_change=None,
        )
        self.assertEqual(evidence.status, RoadObservationStatus.MISSING_CURRENT)


if __name__ == "__main__":
    unittest.main()
