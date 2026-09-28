"""
Comprehensive test suite for vehicle observation temporal synchronization in UrbanTrack AI.

Validates:
1. Same-camera temporal comparison works.
2. Cross-camera unsynchronized observations are marked unavailable.
3. Cross-camera synchronized observations produce valid delta time.
4. Synchronization offsets are correctly applied using authoritative CityFlow metadata (C001: 0.0s, C002: +1.640s, C003: +2.049s).
5. Clock reference IDs are respected and mismatched references are rejected.
6. Video-relative timestamps are never incorrectly treated as globally synchronized.
7. Temporal ordering is correct and negative time intervals are rejected.
8. Known timestamps where expected synchronized delta can be manually verified.
"""

import unittest
from schemas.observation_schema import Observation
from inference.temporal import check_temporal_comparability, temporal_feasibility
from inference.aicity_synchronizer import AICitySynchronizer


class TestTemporalSynchronization(unittest.TestCase):
    """Rigorous validation of temporal comparability and synchronization semantics."""

    def test_01_same_camera_temporal_comparison(self):
        """1. Same-camera temporal comparison works on video-relative timeline."""
        obs_a = Observation(
            camera_id="c001",
            timestamp_seconds=10.0,
            timestamp_semantics="video_relative",
            vehicle_type="car",
        )
        obs_b = Observation(
            camera_id="c001",
            timestamp_seconds=15.5,
            timestamp_semantics="video_relative",
            vehicle_type="car",
        )

        res = check_temporal_comparability(obs_a, obs_b)
        self.assertTrue(res["comparable"])
        self.assertEqual(res["status"], "available")
        self.assertAlmostEqual(res["delta_seconds"], 5.5)
        self.assertEqual(res["time_reference_id"], "camera_c001_local")
        self.assertTrue(res["used_in_route_scoring"])

        # Negative time on same camera must be rejected
        res_neg = check_temporal_comparability(obs_b, obs_a)
        self.assertFalse(res_neg["comparable"])
        self.assertEqual(res_neg["status"], "invalid_negative_time")
        self.assertIsNone(res_neg["delta_seconds"])
        self.assertFalse(res_neg["used_in_route_scoring"])

    def test_02_cross_camera_unsynchronized_marked_unavailable(self):
        """2. Cross-camera unsynchronized observations are marked unavailable."""
        obs_a = Observation(
            camera_id="c001",
            timestamp_seconds=10.0,
            timestamp_semantics="video_relative",
            vehicle_type="car",
        )
        obs_b = Observation(
            camera_id="c002",
            timestamp_seconds=20.0,
            timestamp_semantics="video_relative",
            vehicle_type="car",
        )

        res = check_temporal_comparability(obs_a, obs_b)
        self.assertFalse(res["comparable"])
        self.assertEqual(res["status"], "unavailable")
        self.assertIsNone(res["delta_seconds"])
        self.assertFalse(res["used_in_route_scoring"])
        self.assertIn("video_relative", res["reason"])

        # temporal_feasibility must also return status='unavailable' with feasibility_score=None
        feas = temporal_feasibility(obs_a, obs_b)
        self.assertIsNone(feas["feasibility_score"])
        self.assertIsNone(feas["delta_t_seconds"])
        self.assertEqual(feas["status"], "unavailable")
        self.assertFalse(feas["used_in_route_scoring"])

    def test_03_cross_camera_synchronized_produces_valid_delta_time(self):
        """3. Cross-camera synchronized observations produce valid delta time."""
        obs_a = Observation(
            camera_id="c001",
            timestamp_seconds=10.0,
            timestamp_semantics="video_relative",
            synchronized_timestamp_seconds=10.0,
            synchronized_timestamp_semantics="aicity_official_synchronized",
            synchronized_time_reference_id="AICity_S01_Global",
            vehicle_type="car",
        )
        obs_b = Observation(
            camera_id="c002",
            timestamp_seconds=12.0,
            timestamp_semantics="video_relative",
            synchronized_timestamp_seconds=13.64,
            synchronized_timestamp_semantics="aicity_official_synchronized",
            synchronized_time_reference_id="AICity_S01_Global",
            vehicle_type="car",
        )

        res = check_temporal_comparability(obs_a, obs_b)
        self.assertTrue(res["comparable"])
        self.assertEqual(res["status"], "available")
        self.assertAlmostEqual(res["delta_seconds"], 3.64, places=3)
        self.assertEqual(res["time_reference_id"], "AICity_S01_Global")
        self.assertTrue(res["used_in_route_scoring"])

        # In temporal_feasibility
        feas = temporal_feasibility(obs_a, obs_b)
        self.assertEqual(feas["feasibility_score"], 1.0)
        self.assertAlmostEqual(feas["delta_t_seconds"], 3.64, places=3)
        self.assertEqual(feas["status"], "plausible_time_gap")

    def test_04_synchronization_offsets_correctly_applied(self):
        """4. Authoritative CityFlow synchronization offsets (C001: 0.0s, C002: +1.640s, C003: +2.049s)."""
        sync = AICitySynchronizer("data/aicity_ground_truth/cam_timestamp/S01.txt")
        self.assertAlmostEqual(sync.get_offset("c001"), 0.000)
        self.assertAlmostEqual(sync.get_offset("c002"), 1.640)
        self.assertAlmostEqual(sync.get_offset("c003"), 2.049)

        # Create observations with raw video-relative timestamps
        obs_1 = Observation(camera_id="c001", timestamp_seconds=5.000, vehicle_type="car")
        obs_2 = Observation(camera_id="c002", timestamp_seconds=10.000, vehicle_type="car")
        obs_3 = Observation(camera_id="c003", timestamp_seconds=8.000, vehicle_type="car")

        # Attach authoritative synchronization
        sync.attach_synchronization([obs_1, obs_2, obs_3])

        # Verify video-relative timestamps are preserved and never overwritten
        self.assertEqual(obs_1.timestamp_seconds, 5.000)
        self.assertEqual(obs_2.timestamp_seconds, 10.000)
        self.assertEqual(obs_3.timestamp_seconds, 8.000)

        # Verify synchronized fields
        self.assertAlmostEqual(obs_1.synchronized_timestamp_seconds, 5.000)
        self.assertAlmostEqual(obs_2.synchronized_timestamp_seconds, 11.640)
        self.assertAlmostEqual(obs_3.synchronized_timestamp_seconds, 10.049)

        # Pair c1 -> c2: delta = 11.640 - 5.000 = 6.640s
        res_12 = check_temporal_comparability(obs_1, obs_2)
        self.assertTrue(res_12["comparable"])
        self.assertAlmostEqual(res_12["delta_seconds"], 6.640, places=3)

        # Pair c1 -> c3: delta = 10.049 - 5.000 = 5.049s
        res_13 = check_temporal_comparability(obs_1, obs_3)
        self.assertTrue(res_13["comparable"])
        self.assertAlmostEqual(res_13["delta_seconds"], 5.049, places=3)

        # Pair c3 -> c2: delta = 11.640 - 10.049 = 1.591s
        res_32 = check_temporal_comparability(obs_3, obs_2)
        self.assertTrue(res_32["comparable"])
        self.assertAlmostEqual(res_32["delta_seconds"], 1.591, places=3)

    def test_05_clock_reference_ids_respected(self):
        """5. Clock reference IDs are respected; mismatched references are rejected."""
        obs_a = Observation(
            camera_id="c001",
            timestamp_seconds=10.0,
            synchronized_timestamp_seconds=10.0,
            synchronized_time_reference_id="AICity_S01_Global",
            vehicle_type="car",
        )
        obs_b_diff_ref = Observation(
            camera_id="c002",
            timestamp_seconds=12.0,
            synchronized_timestamp_seconds=13.64,
            synchronized_time_reference_id="AICity_S02_Global",  # Different scenario!
            vehicle_type="car",
        )

        res = check_temporal_comparability(obs_a, obs_b_diff_ref)
        self.assertFalse(res["comparable"])
        self.assertEqual(res["status"], "unavailable")
        self.assertIn("mismatched", res["reason"])
        self.assertIsNone(res["delta_seconds"])

    def test_06_video_relative_never_treated_as_globally_synchronized(self):
        """6. Video-relative timestamps are never incorrectly treated as globally synchronized."""
        obs_a = Observation(
            camera_id="c001",
            timestamp_seconds=10.0,
            timestamp_semantics="video_relative",
            vehicle_type="car",
        )
        obs_b = Observation(
            camera_id="c003",
            timestamp_seconds=15.0,
            timestamp_semantics="video_relative",
            vehicle_type="car",
        )

        res = check_temporal_comparability(obs_a, obs_b)
        self.assertFalse(res["comparable"])
        self.assertNotEqual(res["delta_seconds"], 5.0)
        self.assertIsNone(res["delta_seconds"])
        self.assertEqual(res["status"], "unavailable")

    def test_07_temporal_ordering_and_negative_time_detection(self):
        """7. Temporal ordering is enforced; inverted/negative delta is rejected."""
        obs_early = Observation(
            camera_id="c001",
            timestamp_seconds=5.0,
            synchronized_timestamp_seconds=5.0,
            synchronized_time_reference_id="AICity_S01_Global",
            vehicle_type="car",
        )
        obs_late = Observation(
            camera_id="c002",
            timestamp_seconds=10.0,
            synchronized_timestamp_seconds=11.64,
            synchronized_time_reference_id="AICity_S01_Global",
            vehicle_type="car",
        )

        # Correct chronological order (early -> late)
        res_fwd = check_temporal_comparability(obs_early, obs_late)
        self.assertTrue(res_fwd["comparable"])
        self.assertAlmostEqual(res_fwd["delta_seconds"], 6.64)

        # Inverted chronological order (late -> early)
        res_inv = check_temporal_comparability(obs_late, obs_early)
        self.assertFalse(res_inv["comparable"])
        self.assertEqual(res_inv["status"], "invalid_negative_time")
        self.assertIsNone(res_inv["delta_seconds"])

        feas_inv = temporal_feasibility(obs_late, obs_early)
        self.assertEqual(feas_inv["status"], "impossible_negative_time")
        self.assertEqual(feas_inv["feasibility_score"], 0.0)

    def test_08_known_timestamps_manual_verification(self):
        """8. Deterministic hand-calculated test case for camera synchronization."""
        # C001 at frame 30 (1.000s) + 0.000s offset = 1.000s
        # C002 at frame 60 (2.000s) + 1.640s offset = 3.640s
        # Expected delta = 3.640s - 1.000s = 2.640s
        obs_c1 = Observation(
            camera_id="c001",
            timestamp_seconds=1.000,
            synchronized_timestamp_seconds=1.000,
            synchronized_time_reference_id="AICity_S01_Global",
            vehicle_type="car",
        )
        obs_c2 = Observation(
            camera_id="c002",
            timestamp_seconds=2.000,
            synchronized_timestamp_seconds=3.640,
            synchronized_time_reference_id="AICity_S01_Global",
            vehicle_type="car",
        )

        comp = check_temporal_comparability(obs_c1, obs_c2)
        self.assertTrue(comp["comparable"])
        self.assertEqual(comp["status"], "available")
        self.assertEqual(comp["delta_seconds"], 2.640)


if __name__ == "__main__":
    unittest.main()
