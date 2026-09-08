"""Unit tests for trajectory adapters."""

import json
import tempfile
import unittest
from pathlib import Path

from backend.flow.adapters import MockTrajectoryAdapter
from backend.flow.models import NormalizedTrajectory


class TestMockTrajectoryAdapter(unittest.TestCase):
    """Test suite for MockTrajectoryAdapter."""

    def setUp(self) -> None:
        self.adapter = MockTrajectoryAdapter()
        self.sample_data = {
            "description": "Mock test payload",
            "trajectories": [
                {
                    "track_id": "T001",
                    "origin_node": "J01",
                    "destination_node": "J05",
                    "vehicle_weight": 1.0,
                    "time_window_start": "08:00:00",
                    "time_window_end": "08:15:00",
                    "candidate_routes": [
                        {"nodes": ["J01", "J02", "J05"], "probability": 0.7},
                        {"nodes": ["J01", "J04", "J05"], "probability": 0.3},
                    ],
                },
                {
                    "track_id": "T002",
                    "origin_node": "J01",
                    "destination_node": "J05",
                    "weight": 2.5,  # alternate key
                    "candidate_routes": [
                        {"path": ["J01", "J02", "J05"], "probability": 1.0}  # alternate key 'path'
                    ],
                },
            ],
        }

    def test_adapt_from_dict(self) -> None:
        trajectories = self.adapter.adapt(self.sample_data)
        self.assertEqual(len(trajectories), 2)

        t1 = trajectories[0]
        self.assertIsInstance(t1, NormalizedTrajectory)
        self.assertEqual(t1.track_id, "T001")
        self.assertEqual(t1.origin_node, "J01")
        self.assertEqual(t1.destination_node, "J05")
        self.assertEqual(len(t1.candidate_routes), 2)
        self.assertAlmostEqual(t1.vehicle_weight, 1.0)
        self.assertEqual(t1.time_window_start, "08:00:00")

        t2 = trajectories[1]
        self.assertEqual(t2.track_id, "T002")
        self.assertAlmostEqual(t2.vehicle_weight, 2.5)
        self.assertEqual(t2.candidate_routes[0].nodes, ["J01", "J02", "J05"])

    def test_adapt_from_list(self) -> None:
        trajectories = self.adapter.adapt(self.sample_data["trajectories"])
        self.assertEqual(len(trajectories), 2)

    def test_adapt_from_file(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
            json.dump(self.sample_data, tmp)
            tmp_path = Path(tmp.name)

        try:
            trajectories = self.adapter.adapt(tmp_path)
            self.assertEqual(len(trajectories), 2)
            self.assertEqual(trajectories[0].track_id, "T001")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def test_missing_file_raises_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            self.adapter.adapt("non_existent_mock_file.json")

    def test_invalid_payload_type_raises_error(self) -> None:
        with self.assertRaises(TypeError):
            self.adapter.adapt(12345)  # type: ignore

    def test_adapt_single_item(self) -> None:
        item = self.sample_data["trajectories"][0]
        traj = self.adapter.adapt_one(item)
        self.assertEqual(traj.track_id, "T001")


if __name__ == "__main__":
    unittest.main()
