"""
Comprehensive Edge-Case Test Suite for Day 2 Identity Fusion Engine.
Validates all 14 required edge cases specified in Task 7.
"""

from datetime import datetime, timezone
import unittest

from inference import (
    IdentityGraph,
    Observation,
    match_observations,
)


class TestEdgeCases(unittest.TestCase):
    def setUp(self) -> None:
        self.camera_metadata = {
            "cam_01": {"latitude": 17.3850, "longitude": 78.4867},
            "cam_02": {"latitude": 17.3870, "longitude": 78.4900},
            "cam_03": {"latitude": 17.3950, "longitude": 78.5000},
        }

    # Edge Case 1: Clear same-vehicle match
    def test_edge_01_clear_same_vehicle_match(self):
        obs_a = Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1])
        obs_b = Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=[0.89, 0.41, 0.11])
        res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertGreater(res["same_vehicle_probability"], 0.80)
        self.assertTrue(res["evidence"]["identity_evidence_available"])

    # Edge Case 2: Clear different-vehicle match (Incompatible type)
    def test_edge_02_clear_different_vehicle_match(self):
        obs_a = Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1])
        obs_b = Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="bus", appearance_embedding=[0.9, 0.4, 0.1])
        res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertEqual(res["same_vehicle_probability"], 0.0)
        self.assertIn("Incompatible vehicle types", res["explanation"])

    # Edge Case 3: High appearance similarity but impossible travel time (> 120 km/h)
    def test_edge_03_high_appearance_impossible_travel_time(self):
        obs_a = Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1])
        obs_b = Observation(camera_id="cam_03", track_id="t3", frame_id=12, timestamp_seconds=11.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1])
        res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertEqual(res["same_vehicle_probability"], 0.0)
        self.assertIn("Physically impossible travel speed", res["explanation"])

    # Edge Case 4: Moderate / ambiguous match
    def test_edge_04_moderate_ambiguous_match(self):
        obs_a = Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.8, 0.6, 0.0])
        obs_b = Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=[0.8, 0.0, 0.6])
        res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        prob = res["same_vehicle_probability"]
        self.assertGreaterEqual(prob, 0.40)
        self.assertLessEqual(prob, 0.75)

    # Edge Case 5: Missing appearance embedding (None)
    def test_edge_05_missing_appearance_embedding(self):
        obs_a = Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=None)
        obs_b = Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1])
        res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertIsNone(res["evidence"]["appearance_similarity"])
        self.assertEqual(res["evidence"]["appearance_status"], "missing")
        self.assertFalse(res["evidence"]["identity_evidence_available"])
        self.assertEqual(res["same_vehicle_probability"], 0.5000)

    # Edge Case 6: Invalid embedding (Empty list)
    def test_edge_06_invalid_empty_embedding(self):
        obs_a = Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[])
        obs_b = Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1])
        res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertIsNone(res["evidence"]["appearance_similarity"])
        self.assertEqual(res["same_vehicle_probability"], 0.5000)

    # Edge Case 7: Mismatched embedding dimensions
    def test_edge_07_mismatched_embedding_dimensions(self):
        obs_a = Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.9, 0.4])
        obs_b = Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1, 0.2])
        res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertIsNone(res["evidence"]["appearance_similarity"])
        self.assertEqual(res["evidence"]["appearance_status"], "invalid_mismatched")
        self.assertEqual(res["same_vehicle_probability"], 0.5000)

    # Edge Case 8: Repeated observations from same camera and same track_id
    def test_edge_08_repeated_local_track_observations(self):
        obs_a = Observation(camera_id="cam_01", track_id="trk_c1_10", frame_id=100, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[1.0, 0.0])
        obs_b = Observation(camera_id="cam_01", track_id="trk_c1_10", frame_id=115, timestamp_seconds=10.5, vehicle_type="car", appearance_embedding=[1.0, 0.0])
        res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertEqual(res["same_vehicle_probability"], 1.0000)

    # Edge Case 9: Multi-camera observations belonging to one vehicle
    def test_edge_09_multi_camera_identity_clustering(self):
        obs1 = Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=0.0, vehicle_type="car", appearance_embedding=[1.0, 0.0])
        obs2 = Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=30.0, vehicle_type="car", appearance_embedding=[0.99, 0.01])
        obs3 = Observation(camera_id="cam_03", track_id="t3", frame_id=150, timestamp_seconds=150.0, vehicle_type="car", appearance_embedding=[0.98, 0.02])

        graph = IdentityGraph(min_probability_threshold=0.70)
        graph.build_graph([obs1, obs2, obs3], camera_metadata=self.camera_metadata)
        candidates = graph.get_candidate_identities()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0]["member_observations"]), 3)

    # Edge Case 10: Visually similar but different vehicles
    def test_edge_10_visually_similar_different_vehicles_different_cameras_impossible_speed(self):
        obs_a = Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1])
        obs_b = Observation(camera_id="cam_03", track_id="t3", frame_id=15, timestamp_seconds=11.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1])
        res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertEqual(res["same_vehicle_probability"], 0.0)

    # Edge Case 11: Invalid timestamp (negative timestamp)
    def test_edge_11_invalid_negative_timestamp(self):
        with self.assertRaises(ValueError):
            Observation(camera_id="cam_01", track_id="t1", timestamp_seconds=-10.0)

    # Edge Case 12: Invalid camera ID (empty string)
    def test_edge_12_invalid_camera_id(self):
        with self.assertRaises(ValueError):
            Observation(camera_id="", track_id="t1", timestamp_seconds=10.0)

    # Edge Case 13: Missing camera metadata (coordinates missing from camera mapping)
    def test_edge_13_missing_camera_metadata(self):
        obs_a = Observation(camera_id="cam_unknown_1", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car")
        obs_b = Observation(camera_id="cam_unknown_2", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car")
        res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertIsNotNone(res["same_vehicle_probability"])

    # Edge Case 14: Duplicate observation IDs (Same-camera simultaneous detections)
    def test_edge_14_same_camera_simultaneous_detections(self):
        obs_a = Observation(observation_id="obs_dup_1", camera_id="cam_01", track_id="t1", frame_id=100, timestamp_seconds=10.0, vehicle_type="car", bbox=[10, 10, 50, 50])
        obs_b = Observation(observation_id="obs_dup_2", camera_id="cam_01", track_id="t2", frame_id=100, timestamp_seconds=10.0, vehicle_type="car", bbox=[100, 100, 150, 150])
        res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertEqual(res["same_vehicle_probability"], 0.0)


if __name__ == "__main__":
    unittest.main()
