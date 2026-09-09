"""
Unit tests for Day 2 Identity Fusion Engine and Identity Graph.
Covers required test cases 1 through 6.
"""

from datetime import datetime, timezone
import unittest

from inference import (
    IdentityGraph,
    Observation,
    load_camera_metadata,
    load_observations_from_json,
    match_observations,
)


class TestIdentityFusion(unittest.TestCase):
    def setUp(self) -> None:
        """Set up camera metadata and baseline coordinates."""
        self.camera_metadata = {
            "cam_01": {"latitude": 17.3850, "longitude": 78.4867},  # Origin
            "cam_02": {"latitude": 17.3870, "longitude": 78.4900},  # ~410m away
            "cam_03": {"latitude": 17.3950, "longitude": 78.5000},  # ~1.7km away
        }

    def test_case_1_same_vehicle_high_match_probability(self):
        """
        CASE 1: Same vehicle
        - High appearance similarity
        - Compatible vehicle type (car vs car)
        - Plausible travel time (30 seconds for 410m = 49 km/h)
        Expected: High estimated match probability (> 0.8).
        """
        obs_a = Observation(
            camera_id="cam_01",
            track_id="trk_1",
            frame_id=100,
            timestamp_seconds=10.0,
            vehicle_type="car",
            appearance_embedding=[0.12, -0.04, 0.31, 0.88],
        )

        obs_b = Observation(
            camera_id="cam_02",
            track_id="trk_2",
            frame_id=130,
            timestamp_seconds=40.0,  # 30 second gap
            vehicle_type="car",
            appearance_embedding=[0.13, -0.03, 0.30, 0.87],  # Very high cosine similarity
        )

        result = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        prob = result["same_vehicle_probability"]

        self.assertGreater(prob, 0.8)
        self.assertTrue(result["evidence"]["vehicle_type_match"])
        self.assertGreater(result["evidence"]["appearance_similarity"], 0.95)
        self.assertEqual(result["evidence"]["vehicle_type_status"], "compatible")
        self.assertIn("Estimated match probability is", result["explanation"])

    def test_case_2_impossible_travel_time_rejects_match(self):
        """
        CASE 2: High appearance similarity, but physically impossible travel time
        - 1.7km distance between cam_01 and cam_03
        - Time gap of 2 seconds -> required speed ~3060 km/h (> max 120 km/h)
        Expected: Low/zero estimated match probability (0.0).
        """
        obs_a = Observation(
            camera_id="cam_01",
            track_id="trk_1",
            frame_id=100,
            timestamp_seconds=10.0,
            vehicle_type="car",
            appearance_embedding=[0.12, -0.04, 0.31, 0.88],
        )

        obs_b = Observation(
            camera_id="cam_03",
            track_id="trk_3",
            frame_id=102,
            timestamp_seconds=12.0,  # 2 seconds for 1.7km -> 3060 km/h!
            vehicle_type="car",
            appearance_embedding=[0.12, -0.04, 0.31, 0.88],  # Identical embedding
        )

        result = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        prob = result["same_vehicle_probability"]

        self.assertEqual(prob, 0.0)
        self.assertIn("Match rejected", result["explanation"])
        self.assertEqual(result["evidence"]["spatial_feasibility"], 0.0)

    def test_case_3_different_vehicle_type_rejects_match(self):
        """
        CASE 3: Low appearance similarity, incompatible vehicle type
        - 'car' vs 'bus'
        Expected: Low/zero estimated match probability (0.0).
        """
        obs_a = Observation(
            camera_id="cam_01",
            track_id="trk_1",
            frame_id=100,
            timestamp_seconds=10.0,
            vehicle_type="car",
            appearance_embedding=[0.90, 0.10, 0.00],
        )

        obs_b = Observation(
            camera_id="cam_02",
            track_id="trk_2",
            frame_id=130,
            timestamp_seconds=40.0,
            vehicle_type="bus",  # Incompatible with 'car'
            appearance_embedding=[-0.50, 0.50, 0.20],
        )

        result = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        prob = result["same_vehicle_probability"]

        self.assertEqual(prob, 0.0)
        self.assertFalse(result["evidence"]["vehicle_type_match"])
        self.assertEqual(result["evidence"]["vehicle_type_status"], "incompatible")
        self.assertIn("Match rejected", result["explanation"])

    def test_case_4_moderately_similar_vehicle_intermediate_score(self):
        """
        CASE 4: Moderately similar vehicle, plausible timing
        Expected: Intermediate/uncertain match probability (0.4 to 0.75).
        """
        obs_a = Observation(
            camera_id="cam_01",
            track_id="trk_1",
            frame_id=100,
            timestamp_seconds=10.0,
            vehicle_type="car",
            appearance_embedding=[0.80, 0.60, 0.00],
        )

        obs_b = Observation(
            camera_id="cam_02",
            track_id="trk_2",
            frame_id=130,
            timestamp_seconds=40.0,
            vehicle_type="car",
            appearance_embedding=[0.80, 0.00, 0.60],  # Moderate appearance similarity ~0.64
        )

        result = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        prob = result["same_vehicle_probability"]

        self.assertGreaterEqual(prob, 0.4)
        self.assertLessEqual(prob, 0.75)

    def test_case_5_missing_appearance_embedding_safe_fallback(self):
        """
        CASE 5: Missing appearance embedding (None)
        Expected: System continues safely, marks appearance evidence unavailable, uses spatio-temporal/type evidence.
        """
        obs_a = Observation(
            camera_id="cam_01",
            track_id="trk_1",
            frame_id=100,
            timestamp_seconds=10.0,
            vehicle_type="car",
            appearance_embedding=None,  # Missing embedding
        )

        obs_b = Observation(
            camera_id="cam_02",
            track_id="trk_2",
            frame_id=130,
            timestamp_seconds=40.0,
            vehicle_type="car",
            appearance_embedding=[0.13, -0.03, 0.30, 0.87],
        )

        result = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertIsNone(result["evidence"]["appearance_similarity"])
        self.assertGreater(result["same_vehicle_probability"], 0.0)
        self.assertIn("Appearance evidence unavailable", result["explanation"])

    def test_case_6_mismatched_embedding_dimensions_graceful_handling(self):
        """
        CASE 6: Invalid / mismatched embedding dimensions (e.g. 4-dim vs 128-dim)
        Expected: Graceful error handling (returns None for appearance_similarity, pipeline continues safely).
        """
        obs_a = Observation(
            camera_id="cam_01",
            track_id="trk_1",
            frame_id=100,
            timestamp_seconds=10.0,
            vehicle_type="car",
            appearance_embedding=[0.12, -0.04, 0.31, 0.88],  # 4-dim
        )

        obs_b = Observation(
            camera_id="cam_02",
            track_id="trk_2",
            frame_id=130,
            timestamp_seconds=40.0,
            vehicle_type="car",
            appearance_embedding=[0.01 * i for i in range(128)],  # 128-dim
        )

        result = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
        self.assertIsNone(result["evidence"]["appearance_similarity"])
        self.assertEqual(result["evidence"]["appearance_status"], "invalid_mismatched")
        self.assertIsNotNone(result["same_vehicle_probability"])

    def test_identity_graph_clustering(self):
        """Test building an IdentityGraph and grouping observations into candidate vehicle identities."""
        obs1 = Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=0.0, vehicle_type="car", appearance_embedding=[1.0, 0.0])
        obs2 = Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=30.0, vehicle_type="car", appearance_embedding=[0.98, 0.02])
        obs3 = Observation(camera_id="cam_03", track_id="t3", frame_id=10, timestamp_seconds=5.0, vehicle_type="bus", appearance_embedding=[-0.9, 0.1])

        graph = IdentityGraph(min_probability_threshold=0.5)
        graph.build_graph([obs1, obs2, obs3], camera_metadata=self.camera_metadata)

        candidates = graph.get_candidate_identities()
        self.assertGreaterEqual(len(candidates), 2)  # Car cluster + Bus cluster


if __name__ == "__main__":
    unittest.main()
