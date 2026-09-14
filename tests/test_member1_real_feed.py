"""
Tests for Member 1 Real Video Perception Feed Ingestion and Fusion.
Validates loading of real 512-dimensional OSNet embeddings, multi-frame OCR consensus,
frame-level camera reliability telemetry, and vehicle identity matching.
"""

import os
import unittest
from pathlib import Path

from inference.observation_loader import load_member1_perception_feed
from inference.similarity import (
    appearance_similarity,
    validate_and_normalize_embedding,
    vehicle_type_compatibility,
)
from inference.identity_fusion import match_observations
from inference.identity_graph import IdentityGraph


class TestMember1RealFeed(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tracks_path = "data/member1_perception/cam_001/track_embeddings.json"
        cls.telemetry_path = "data/member1_perception/cam_001/camera_telemetry.json"
        cls.raw_detections_path = "data/member1_perception/cam_001/raw_frame_detections.json"

        # Verify files exist
        if not os.path.isfile(cls.tracks_path):
            raise unittest.SkipTest(f"Member 1 data not present: {cls.tracks_path}")

        cls.observations = load_member1_perception_feed(
            tracks_path=cls.tracks_path,
            telemetry_path=cls.telemetry_path,
            raw_detections_path=cls.raw_detections_path,
            camera_id="CAM_001",
            fps=30.0,
        )
        cls.obs_map = {o.observation_id: o for o in cls.observations}

    def test_01_load_all_tracks(self):
        """Verify all 39 tracks are loaded into standardized Observation instances."""
        self.assertEqual(len(self.observations), 39)
        first_obs = self.observations[0]
        self.assertEqual(first_obs.camera_id, "CAM_001")
        self.assertIsNotNone(first_obs.observation_id)
        self.assertGreaterEqual(first_obs.timestamp_seconds, 0.0)

    def test_02_osnet_512d_embeddings(self):
        """Verify OSNet embeddings are 512-dimensional and pass validation."""
        for obs in self.observations:
            emb = obs.appearance_embedding
            self.assertIsNotNone(emb)
            self.assertEqual(len(emb), 512)
            normalized = validate_and_normalize_embedding(emb, expected_dim=512)
            self.assertIsNotNone(normalized)
            self.assertAlmostEqual(sum(x * x for x in normalized), 1.0, places=4)

    def test_03_plate_consensus_voting(self):
        """Verify real license plates are extracted via consensus voting."""
        obs_with_plates = [o for o in self.observations if o.plate is not None]
        self.assertGreaterEqual(len(obs_with_plates), 5)

        # Known plates from real video run
        known_plates = {"MH4JAD9203", "MA0AJK3437", "NH01DP4248", "MH01BD1383", "NH0LDD4922", "MH0ZFX9484"}
        detected_plates = {o.plate for o in obs_with_plates}
        self.assertTrue(known_plates.issubset(detected_plates) or len(detected_plates.intersection(known_plates)) >= 5)

    def test_04_camera_telemetry_attached(self):
        """Verify camera reliability and telemetry are correctly linked from video processing."""
        for obs in self.observations:
            self.assertIsNotNone(obs.camera_reliability)
            self.assertGreater(obs.camera_reliability, 0.40)
            self.assertLess(obs.camera_reliability, 0.70)
            self.assertIsNotNone(obs.detection_confidence)

    def test_05_reentry_matching_trk65_and_trk94(self):
        """Verify Track 65 and Track 94 share plate MH0ZFX9484 and high multimodal affinity."""
        o65 = self.obs_map.get("CAM_001_trk_065")
        o94 = self.obs_map.get("CAM_001_trk_094")
        self.assertIsNotNone(o65)
        self.assertIsNotNone(o94)

        # Both have plate MH0ZFX9484
        self.assertEqual(o65.plate, "MH0ZFX9484")
        self.assertEqual(o94.plate, "MH0ZFX9484")

        # OSNet Re-ID similarity is strong
        sim = appearance_similarity(o65.appearance_embedding, o94.appearance_embedding)
        self.assertGreater(sim, 0.60)

        # Full match result
        res = match_observations(o65, o94)
        self.assertGreater(res["same_vehicle_probability"], 0.65)
        self.assertEqual(res["evidence"]["plate_similarity"], 1.0)
        self.assertTrue(res["evidence"]["vehicle_type_match"])

    def test_06_incompatible_vehicle_types(self):
        """Verify car vs truck (e.g. Track 1 car vs Track 2 truck) rejects match."""
        o1 = self.obs_map.get("CAM_001_trk_001") # car
        o2 = self.obs_map.get("CAM_001_trk_002") # truck
        self.assertIsNotNone(o1)
        self.assertIsNotNone(o2)
        self.assertEqual(o1.vehicle_type, "car")
        self.assertEqual(o2.vehicle_type, "truck")

        res = match_observations(o1, o2)
        self.assertEqual(res["same_vehicle_probability"], 0.0)
        self.assertFalse(res["evidence"]["vehicle_type_match"])

    def test_07_identity_graph_clustering_on_real_feed(self):
        """Verify IdentityGraph processes the real Member 1 feed cleanly."""
        graph = IdentityGraph(min_probability_threshold=0.65)
        graph.build_graph(self.observations)
        clusters = graph.get_candidate_identities()

        # Should form candidate clusters without error
        self.assertGreater(len(clusters), 0)
        self.assertLessEqual(len(clusters), 39)


if __name__ == "__main__":
    unittest.main(verbosity=2)
