"""Tests for the Member-2 Day-3 to Member-3 trajectory adapter."""

import copy
import unittest
from pathlib import Path

from backend.flow.adapters import Member2TrajectoryAdapter
from backend.flow.aggregation import ExpectedFlowAggregator
from backend.flow.models import ProbabilityValidationError
from backend.mobility.graph import MobilityGraph

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


TRK_001_PAYLOAD = {
    "track_id": "TRK_001",
    "origin_node": "J01",
    "destination_node": "J08",
    "vehicle_weight": 1.0,
    "candidate_routes": [
        {
            "nodes": ["J01", "J04", "J08"],
            "probability": 0.430257,
            "metadata": {"route_id": "route_01", "edges": ["R07", "R11"]},
        },
        {
            "nodes": ["J01", "J02", "J05", "J08"],
            "probability": 0.311669,
            "metadata": {"route_id": "route_02", "edges": ["R01", "R08", "R09"]},
        },
        {
            "nodes": ["J01", "J04", "J05", "J08"],
            "probability": 0.258074,
            "metadata": {"route_id": "route_03", "edges": ["R07", "R18", "R09"]},
        },
    ],
    "time_window": {"start": 1000.0, "end": 1450.0},
    "timestamp": 1000.0,
    "metadata": {"confidence": 0.4303, "is_ambiguous": True},
}


class TestMember2TrajectoryAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.adapter = Member2TrajectoryAdapter()

    def test_trk001_maps_all_fields_and_duration(self) -> None:
        trajectory = self.adapter.adapt_one(TRK_001_PAYLOAD)

        self.assertEqual(trajectory.track_id, "TRK_001")
        self.assertEqual(trajectory.origin_node, "J01")
        self.assertEqual(trajectory.destination_node, "J08")
        self.assertEqual(trajectory.vehicle_weight, 1.0)
        self.assertEqual(len(trajectory.candidate_routes), 3)
        self.assertEqual(
            [route.probability for route in trajectory.candidate_routes],
            [0.430257, 0.311669, 0.258074],
        )
        self.assertEqual(trajectory.time_window_start, 1000.0)
        self.assertEqual(trajectory.time_window_end, 1450.0)
        self.assertEqual(trajectory.metadata["aggregation_duration_seconds"], 450.0)
        self.assertEqual(trajectory.candidate_routes[0].metadata["edges"], ["R07", "R11"])

    def test_batch_and_shared_graph_aggregation(self) -> None:
        graph = MobilityGraph.load_from_json(PROJECT_ROOT / "data" / "synthetic" / "city_network.json")
        trajectories = self.adapter.adapt({"trajectories": [TRK_001_PAYLOAD]})
        result = ExpectedFlowAggregator(graph).aggregate(trajectories)

        self.assertEqual(len(result.trajectories), 1)
        self.assertGreater(result.get_flow("R07"), 0.0)
        self.assertGreater(result.get_flow("R01"), 0.0)
        self.assertGreater(result.get_flow("R09"), 0.0)

    def test_invalid_probabilities_use_existing_m3_validation(self) -> None:
        invalid = copy.deepcopy(TRK_001_PAYLOAD)
        invalid["candidate_routes"][0]["probability"] = 1.1
        with self.assertRaises(ProbabilityValidationError):
            self.adapter.adapt_one(invalid)

        invalid = copy.deepcopy(TRK_001_PAYLOAD)
        invalid["candidate_routes"][0]["probability"] = 0.4
        with self.assertRaises(ProbabilityValidationError):
            self.adapter.adapt_one(invalid)

    def test_top_level_window_values_are_preserved(self) -> None:
        payload = copy.deepcopy(TRK_001_PAYLOAD)
        payload["time_window_start"] = 2000.0
        payload["time_window_end"] = 2450.0
        trajectory = self.adapter.adapt_one(payload)
        self.assertEqual(trajectory.time_window_start, 2000.0)
        self.assertEqual(trajectory.time_window_end, 2450.0)
        self.assertEqual(trajectory.metadata["aggregation_duration_seconds"], 450.0)


if __name__ == "__main__":
    unittest.main()
