"""
Unit Test Suite for Tracklet Reasoning, Re-ID Validation, Ablation, and Graceful Degradation.
Verifies Phase 4, Phase 11, Phase 14, and Phase 15.
"""

from datetime import datetime
import math
import unittest

from schemas.observation_schema import Observation
from inference.ablation_study import run_ablation_study
from inference.degradation_benchmark import (
    evaluate_camera_network_dropout,
    evaluate_plate_dropout_curve,
    evaluate_reid_dropout_curve,
    run_full_degradation_benchmark,
)
from inference.similarity import (
    appearance_similarity,
    evaluate_reid_distribution,
    plate_similarity,
    validate_and_normalize_embedding,
)
from inference.tracklet_engine import (
    Tracklet,
    aggregate_observations_into_tracklets,
    aggregate_plate_votes,
    match_tracklets,
    pool_embeddings,
)


class TestTrackletAndAblation(unittest.TestCase):

    def setUp(self):
        self.emb_a = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.emb_b = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        self.emb_diff = [0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1]

    # 1. Test validate_and_normalize_embedding
    def test_01_validate_and_normalize_embedding(self):
        # Valid embedding
        norm_emb = validate_and_normalize_embedding(self.emb_a)
        self.assertIsNotNone(norm_emb)
        norm = math.sqrt(sum(x * x for x in norm_emb))
        self.assertAlmostEqual(norm, 1.0, places=5)

        # Expected dim enforcement
        self.assertIsNotNone(validate_and_normalize_embedding(self.emb_a, expected_dim=8))
        self.assertIsNone(validate_and_normalize_embedding(self.emb_a, expected_dim=512))

        # Rejects NaN / Inf
        nan_emb = [0.1, float("nan"), 0.3]
        self.assertIsNone(validate_and_normalize_embedding(nan_emb))

        # Rejects all-zero
        zero_emb = [0.0] * 8
        self.assertIsNone(validate_and_normalize_embedding(zero_emb))

        # Rejects non-sequence / None
        self.assertIsNone(validate_and_normalize_embedding(None))
        self.assertIsNone(validate_and_normalize_embedding("not_a_vector"))

    # 2. Test evaluate_reid_distribution
    def test_02_evaluate_reid_distribution(self):
        same_pairs = [(self.emb_a, self.emb_b), (self.emb_a, [x + 0.01 for x in self.emb_a])]
        diff_pairs = [(self.emb_a, self.emb_diff)]

        report = evaluate_reid_distribution(same_pairs, diff_pairs)
        self.assertIn("same_vehicle_distribution", report)
        self.assertIn("different_vehicle_distribution", report)
        self.assertGreater(report["distribution_separation"], 0.5)
        self.assertGreaterEqual(report["operating_threshold"], 0.5)

    # 3. Test aggregate_plate_votes
    def test_03_aggregate_plate_votes(self):
        obs1 = Observation(camera_id="cam1", frame_id=1, timestamp_seconds=1.0, plate="KA01AB1234", plate_confidence=0.95)
        obs2 = Observation(camera_id="cam1", frame_id=2, timestamp_seconds=1.1, plate="KA01AB1234", plate_confidence=0.92)
        obs3 = Observation(camera_id="cam1", frame_id=3, timestamp_seconds=1.2, plate="KA01A81234", plate_confidence=0.60) # OCR noise

        consensus_plate, avg_conf, count = aggregate_plate_votes([obs1, obs2, obs3])
        self.assertEqual(consensus_plate, "KA01AB1234")
        self.assertGreater(avg_conf, 0.90)
        self.assertEqual(count, 3)

    # 4. Test pool_embeddings
    def test_04_pool_embeddings(self):
        obs1 = Observation(camera_id="cam1", frame_id=1, timestamp_seconds=1.0, appearance_embedding=self.emb_a, detection_confidence=0.9)
        obs2 = Observation(camera_id="cam1", frame_id=2, timestamp_seconds=1.1, appearance_embedding=self.emb_b, detection_confidence=0.8)

        pooled = pool_embeddings([obs1, obs2])
        self.assertIsNotNone(pooled)
        self.assertEqual(len(pooled), 8)
        norm = math.sqrt(sum(x * x for x in pooled))
        self.assertAlmostEqual(norm, 1.0, places=5)

    # 5. Test aggregate_observations_into_tracklets
    def test_05_aggregate_observations_into_tracklets(self):
        obs1 = Observation(camera_id="cam1", track_id="t1", frame_id=1, timestamp_seconds=10.0, vehicle_type="car", plate="KA01AB1234", plate_confidence=0.95, appearance_embedding=self.emb_a)
        obs2 = Observation(camera_id="cam1", track_id="t1", frame_id=5, timestamp_seconds=10.5, vehicle_type="car", plate="KA01AB1234", plate_confidence=0.90, appearance_embedding=self.emb_a)
        obs3 = Observation(camera_id="cam2", track_id="t2", frame_id=50, timestamp_seconds=50.0, vehicle_type="car", plate="KA01AB1234", plate_confidence=0.95, appearance_embedding=self.emb_a)

        tracklets = aggregate_observations_into_tracklets([obs1, obs2, obs3])
        self.assertEqual(len(tracklets), 2)

        trk1 = next(t for t in tracklets if t.camera_id == "cam1")
        self.assertEqual(trk1.track_id, "t1")
        self.assertEqual(trk1.observations_count, 2)
        self.assertEqual(trk1.duration_seconds, 0.5)
        self.assertEqual(trk1.aggregated_plate, "KA01AB1234")

    # 6. Test match_tracklets
    def test_06_match_tracklets(self):
        meta = {
            "cam1": {"latitude": 12.9716, "longitude": 77.5946, "time_reference_id": "city_sync_grid"},
            "cam2": {"latitude": 12.9750, "longitude": 77.5980, "time_reference_id": "city_sync_grid"},
        }
        obs1 = Observation(camera_id="cam1", track_id="t1", frame_id=1, timestamp_seconds=10.0, vehicle_type="car", plate="KA01AB1234", plate_confidence=0.95, appearance_embedding=self.emb_a, latitude=12.9716, longitude=77.5946)
        obs2 = Observation(camera_id="cam2", track_id="t2", frame_id=50, timestamp_seconds=50.0, vehicle_type="car", plate="KA01AB1234", plate_confidence=0.95, appearance_embedding=self.emb_a, latitude=12.9750, longitude=77.5980)

        tracklets = aggregate_observations_into_tracklets([obs1, obs2], camera_metadata=meta)
        trk1, trk2 = tracklets[0], tracklets[1]

        res = match_tracklets(trk1, trk2, camera_metadata=meta)
        self.assertGreater(res["same_vehicle_probability"], 0.90)
        self.assertIn("tracklet_evidence", res)

    # 7. Test 6-tier ablation study
    def test_07_ablation_study(self):
        meta = {
            "cam1": {"latitude": 12.9716, "longitude": 77.5946, "time_reference_id": "city_sync_grid"},
            "cam2": {"latitude": 12.9750, "longitude": 77.5980, "time_reference_id": "city_sync_grid"},
        }
        # Vehicle 1: 2 observations with matching plate and Re-ID
        v1_a = Observation(observation_id="v1_a", camera_id="cam1", timestamp_seconds=10.0, vehicle_type="car", plate="KA01AA1111", appearance_embedding=self.emb_a, latitude=12.9716, longitude=77.5946)
        v1_b = Observation(observation_id="v1_b", camera_id="cam2", timestamp_seconds=50.0, vehicle_type="car", plate="KA01AA1111", appearance_embedding=self.emb_a, latitude=12.9750, longitude=77.5980)

        # Vehicle 2: 1 observation with different plate and embedding
        v2_a = Observation(observation_id="v2_a", camera_id="cam1", timestamp_seconds=20.0, vehicle_type="bus", plate="KA02BB2222", appearance_embedding=self.emb_diff, latitude=12.9716, longitude=77.5946)

        obs_list = [v1_a, v1_b, v2_a]
        gt_clusters = {"V1": ["v1_a", "v1_b"], "V2": ["v2_a"]}

        abl_res = run_ablation_study(obs_list, gt_clusters, camera_metadata=meta)
        self.assertIn("tiers", abl_res)
        self.assertEqual(len(abl_res["tiers"]), 6)
        self.assertEqual(abl_res["tiers"]["F_full_urbantrack"]["cluster_purity"], 1.0)
        self.assertEqual(abl_res["tiers"]["F_full_urbantrack"]["pairwise"]["false_merge_rate"], 0.0)

    # 8. Test Graceful Degradation Suite
    def test_08_degradation_benchmark(self):
        meta = {
            "cam1": {"latitude": 12.9716, "longitude": 77.5946, "time_reference_id": "city_sync_grid"},
            "cam2": {"latitude": 12.9750, "longitude": 77.5980, "time_reference_id": "city_sync_grid"},
        }
        v1_a = Observation(observation_id="v1_a", camera_id="cam1", timestamp_seconds=10.0, vehicle_type="car", plate="KA01AA1111", appearance_embedding=self.emb_a, latitude=12.9716, longitude=77.5946)
        v1_b = Observation(observation_id="v1_b", camera_id="cam2", timestamp_seconds=50.0, vehicle_type="car", plate="KA01AA1111", appearance_embedding=self.emb_a, latitude=12.9750, longitude=77.5980)
        obs_list = [v1_a, v1_b]
        gt_clusters = {"V1": ["v1_a", "v1_b"]}

        deg_report = run_full_degradation_benchmark(obs_list, gt_clusters, camera_metadata=meta)
        self.assertIn("plate_dropout_curve", deg_report)
        self.assertIn("reid_dropout_curve", deg_report)
        self.assertIn("camera_network_dropout_curve", deg_report)
        # Verify 0 false merges across degradation levels
        for row in deg_report["plate_dropout_curve"]:
            self.assertEqual(row["false_merges"], 0)


if __name__ == "__main__":
    unittest.main()
