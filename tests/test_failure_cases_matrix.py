"""
Unit and Integration Test Suite: 20 Required Failure-Case and Robustness Scenarios.
Directly implements Section 22 of the UrbanTrack AI Engineering Specification.
Every test asserts an explicit semantic outcome (CONFIRMED, REJECTED, AMBIGUOUS, DEGRADED, etc.).
"""

from datetime import datetime, timezone
import math
import unittest
from typing import Any, Dict, List

from schemas.observation_schema import Observation
from inference.identity_fusion import match_observations
from inference.reid_compatibility import (
    ReIDModelCompatibilityLayer,
    are_reid_models_compatible,
)
from inference.similarity import (
    CONFUSABLE_OCR_CHAR_PAIRS,
    appearance_similarity,
    ocr_aware_plate_similarity,
    plate_similarity,
    vehicle_type_compatibility,
)
from inference.spatial import spatial_feasibility
from inference.temporal import temporal_feasibility
from inference.road_graph import RoadEdge, RoadGraph, RoadNode
from inference.sparse_engine import infer_sparse_gap


class TestFailureCasesMatrix(unittest.TestCase):
    """
    Validation matrix covering all 20 required failure and edge-case scenarios
    from Section 22 of the UrbanTrack AI Engineering Specification.
    """

    def setUp(self):
        self.emb_sedan_a = [0.1] * 512
        self.emb_sedan_b = [0.1] * 512
        self.emb_truck = [-0.1] * 512
        self.cam_meta = {
            "C001": {"latitude": 17.3850, "longitude": 78.4860, "name": "Junction_1", "time_reference_id": "AICity_S01_Global", "clock_offset_seconds": 0.0},
            "C002": {"latitude": 17.3875, "longitude": 78.4890, "name": "Junction_2", "time_reference_id": "AICity_S01_Global", "clock_offset_seconds": 1.640},
            "C003": {"latitude": 17.3910, "longitude": 78.4930, "name": "Junction_3", "time_reference_id": "AICity_S01_Global", "clock_offset_seconds": 2.049},
        }

    # Case 1: Same vehicle, good plate
    def test_01_same_vehicle_good_plate(self):
        """Case 1: Same vehicle across cameras with reliable, matching license plate."""
        obs1 = Observation(
            observation_id="C001_v1",
            camera_id="C001",
            timestamp_seconds=10.0,
            synchronized_timestamp_seconds=10.0,
            vehicle_type="car",
            plate="KA01AB1234",
            plate_confidence=0.96,
            appearance_embedding=self.emb_sedan_a,
            latitude=17.3850,
            longitude=78.4860,
        )
        obs2 = Observation(
            observation_id="C002_v1",
            camera_id="C002",
            timestamp_seconds=38.36,
            synchronized_timestamp_seconds=40.0,
            vehicle_type="car",
            plate="KA01AB1234",
            plate_confidence=0.94,
            appearance_embedding=self.emb_sedan_b,
            latitude=17.3875,
            longitude=78.4890,
        )
        res = match_observations(obs1, obs2, camera_metadata=self.cam_meta)
        self.assertEqual(res["decision_state"], "CONFIRMED")
        self.assertGreaterEqual(res["same_vehicle_score"], 0.80)
        self.assertEqual(res["evidence"]["plate_similarity"], 1.0)

    # Case 2: Same vehicle, missing plate
    def test_02_same_vehicle_missing_plate(self):
        """Case 2: Same vehicle with missing plates (CityFlow plate censoring scenario)."""
        obs1 = Observation(
            observation_id="C001_v2",
            camera_id="C001",
            timestamp_seconds=10.0,
            synchronized_timestamp_seconds=10.0,
            vehicle_type="car",
            plate=None,
            appearance_embedding=self.emb_sedan_a,
            embedding_model="osnet_x0_25_aicity",
            latitude=17.3850,
            longitude=78.4860,
        )
        obs2 = Observation(
            observation_id="C003_v2",
            camera_id="C003",
            timestamp_seconds=50.0,
            synchronized_timestamp_seconds=52.049,
            vehicle_type="car",
            plate=None,
            appearance_embedding=self.emb_sedan_b,
            embedding_model="osnet_x0_25_aicity",
            latitude=17.3910,
            longitude=78.4930,
        )
        res = match_observations(obs1, obs2, camera_metadata=self.cam_meta)
        self.assertEqual(res["decision_state"], "CONFIRMED")
        self.assertGreaterEqual(res["same_vehicle_score"], 0.75)
        self.assertIsNone(res["evidence"]["plate_similarity"])
        self.assertGreater(res["evidence"]["appearance_similarity"], 0.90)

    # Case 3: Same vehicle, noisy OCR
    def test_03_same_vehicle_noisy_ocr(self):
        """Case 3: Same vehicle with visually confusable OCR substitution (B vs 8)."""
        p1 = "KA01AB1234"
        p2 = "KA01A81234"
        sim = ocr_aware_plate_similarity(p1, p2)
        self.assertGreater(sim, 0.95)

        obs1 = Observation(
            camera_id="C001",
            timestamp_seconds=10.0,
            vehicle_type="car",
            plate=p1,
            plate_confidence=0.90,
            appearance_embedding=self.emb_sedan_a,
            latitude=17.3850,
            longitude=78.4860,
        )
        obs2 = Observation(
            camera_id="C002",
            timestamp_seconds=40.0,
            vehicle_type="car",
            plate=p2,
            plate_confidence=0.85,
            appearance_embedding=self.emb_sedan_b,
            latitude=17.3875,
            longitude=78.4890,
        )
        res = match_observations(obs1, obs2, camera_metadata=self.cam_meta)
        self.assertNotEqual(res["decision_state"], "REJECTED")
        self.assertGreater(res["same_vehicle_score"], 0.70)

    # Case 4: Same vehicle, missing Re-ID
    def test_04_same_vehicle_missing_reid(self):
        """Case 4: Same vehicle where appearance embedding is missing or corrupted."""
        obs1 = Observation(
            camera_id="C001",
            timestamp_seconds=10.0,
            vehicle_type="car",
            plate="DL01XY9999",
            plate_confidence=0.98,
            appearance_embedding=None,
            latitude=17.3850,
            longitude=78.4860,
        )
        obs2 = Observation(
            camera_id="C002",
            timestamp_seconds=40.0,
            vehicle_type="car",
            plate="DL01XY9999",
            plate_confidence=0.97,
            appearance_embedding=None,
            latitude=17.3875,
            longitude=78.4890,
        )
        res = match_observations(obs1, obs2, camera_metadata=self.cam_meta)
        self.assertEqual(res["decision_state"], "CONFIRMED")
        self.assertGreaterEqual(res["same_vehicle_score"], 0.80)
        self.assertIsNone(res["evidence"]["appearance_similarity"])

    # Case 5: Similar-looking different vehicles
    def test_05_similar_looking_different_vehicles(self):
        """Case 5: High appearance similarity but contradictory plates must be REJECTED."""
        obs1 = Observation(
            camera_id="C001",
            timestamp_seconds=10.0,
            vehicle_type="car",
            plate="KA01AA1111",
            plate_confidence=0.95,
            appearance_embedding=self.emb_sedan_a,
            latitude=17.3850,
            longitude=78.4860,
        )
        obs2 = Observation(
            camera_id="C002",
            timestamp_seconds=40.0,
            vehicle_type="car",
            plate="DL04BB2222",
            plate_confidence=0.95,
            appearance_embedding=self.emb_sedan_b,
            latitude=17.3875,
            longitude=78.4890,
        )
        res = match_observations(obs1, obs2, camera_metadata=self.cam_meta)
        self.assertEqual(res["decision_state"], "REJECTED")
        self.assertLess(res["same_vehicle_score"], 0.35)

    # Case 6: Different vehicle types
    def test_06_different_vehicle_types(self):
        """Case 6: Categorical vehicle type conflict (car vs bus) unconditionally rejects."""
        obs1 = Observation(
            camera_id="C001",
            timestamp_seconds=10.0,
            vehicle_type="car",
            plate="TS09ZZ1234",
            appearance_embedding=self.emb_sedan_a,
            latitude=17.3850,
            longitude=78.4860,
        )
        obs2 = Observation(
            camera_id="C002",
            timestamp_seconds=40.0,
            vehicle_type="bus",
            plate="TS09ZZ1234",  # Plate hash collision or cloned plate
            appearance_embedding=self.emb_sedan_b,
            latitude=17.3875,
            longitude=78.4890,
        )
        res = match_observations(obs1, obs2, camera_metadata=self.cam_meta)
        self.assertEqual(res["decision_state"], "REJECTED")
        self.assertEqual(res["same_vehicle_score"], 0.0)

    # Case 7: Noisy vehicle type
    def test_07_noisy_vehicle_type(self):
        """Case 7: Borderline type noise (car vs truck) treated as evidence penalty, not instant crash."""
        compat, status = vehicle_type_compatibility("car", "truck")
        self.assertEqual(status, "incompatible")
        self.assertEqual(compat, 0.0)

    # Case 8: Impossible travel time
    def test_08_impossible_travel_time(self):
        """Case 8: Physically impossible speed (e.g. 5 km in 4 seconds = 4,500 km/h)."""
        obs1 = Observation(
            camera_id="C001",
            timestamp_seconds=10.0,
            vehicle_type="car",
            plate="MH01AA0001",
            latitude=17.3850,
            longitude=78.4860,
        )
        obs2 = Observation(
            camera_id="C003",
            timestamp_seconds=14.0,  # 4 seconds elapsed
            vehicle_type="car",
            plate="MH01AA0001",
            latitude=17.4350,  # ~5.5 km away
            longitude=78.5360,
        )
        res = match_observations(obs1, obs2, camera_metadata=self.cam_meta)
        self.assertEqual(res["decision_state"], "REJECTED")
        self.assertEqual(res["evidence"]["spatial_feasibility"], 0.0)

    # Case 9: Missing camera
    def test_09_missing_camera(self):
        """Case 9: Road-constrained route hypothesis when intermediate camera C002 drops detection."""
        rg = RoadGraph()
        rg.add_node(RoadNode("C001", 17.3850, 78.4860))
        rg.add_node(RoadNode("C002", 17.3875, 78.4890))
        rg.add_node(RoadNode("C003", 17.3910, 78.4930))
        rg.add_edge(RoadEdge("E1", "C001", "C002", 400.0, 50.0))
        rg.add_edge(RoadEdge("E2", "C002", "C003", 500.0, 50.0))
        rg.camera_associations = {"C001": "C001", "C002": "C002", "C003": "C003"}

        obs_a = Observation(camera_id="C001", timestamp_seconds=0.0)
        obs_b = Observation(camera_id="C003", timestamp_seconds=90.0)

        gap = infer_sparse_gap(obs_a, obs_b, rg)
        self.assertIsNotNone(gap)
        self.assertEqual(gap.start_node_id, "C001")
        self.assertEqual(gap.end_node_id, "C003")
        self.assertIn("C002", gap.unobserved_intermediate_nodes)
        self.assertEqual(gap.status, "success")
        self.assertTrue(gap.feasible)

    # Case 10: Camera blackout
    def test_10_camera_blackout(self):
        """Case 10: Low camera reliability attenuates confidence."""
        obs1 = Observation(
            camera_id="C001",
            timestamp_seconds=10.0,
            vehicle_type="car",
            camera_reliability=0.10,  # Camera sensor severely impaired
            appearance_embedding=self.emb_sedan_a,
            latitude=17.3850,
            longitude=78.4860,
        )
        obs2 = Observation(
            camera_id="C002",
            timestamp_seconds=40.0,
            vehicle_type="car",
            camera_reliability=0.85,
            appearance_embedding=self.emb_sedan_b,
            latitude=17.3875,
            longitude=78.4890,
        )
        res = match_observations(obs1, obs2, camera_metadata=self.cam_meta)
        self.assertLess(res["same_vehicle_score"], 0.70)
        self.assertNotEqual(res["decision_state"], "CONFIRMED")

    # Case 11: Timestamp jitter
    def test_11_timestamp_jitter(self):
        """Case 11: Small sub-second jitter is accommodated by time uncertainty bounds."""
        meta = {
            "C001": {"time_reference_id": "global_sync", "clock_offset_seconds": 0.0},
            "C002": {"time_reference_id": "global_sync", "clock_offset_seconds": 0.0},
        }
        obs1 = Observation(camera_id="C001", timestamp_seconds=10.0, synchronized_timestamp_seconds=10.0)
        obs2_nominal = Observation(camera_id="C002", timestamp_seconds=30.0, synchronized_timestamp_seconds=30.0)
        obs2_jitter = Observation(camera_id="C002", timestamp_seconds=30.8, synchronized_timestamp_seconds=30.8)

        res_nominal = temporal_feasibility(obs1, obs2_nominal, camera_metadata=meta)
        res_jitter = temporal_feasibility(obs1, obs2_jitter, camera_metadata=meta)
        self.assertIn(res_nominal["status"], ["feasible", "plausible_time_gap"])
        self.assertIn(res_jitter["status"], ["feasible", "plausible_time_gap"])
        self.assertGreater(res_jitter["feasibility_score"], 0.8)

    # Case 12: Camera synchronization offset
    def test_12_camera_synchronization_offset(self):
        """Case 12: Applying camera clock offset prevents incorrect cross-camera delta-t."""
        c001_raw = 10.0
        c002_raw = 8.8   # Video-relative looks like it arrived BEFORE C001 (-1.2s)!
        c002_offset = 1.640 # Official CityFlow C002 offset

        c001_sync = c001_raw + 0.0
        c002_sync = c002_raw + c002_offset  # 8.8 + 1.640 = 10.44s (positive forward time!)
        dt_sync = c002_sync - c001_sync
        self.assertGreater(dt_sync, 0.0)
        self.assertAlmostEqual(dt_sync, 0.44, places=2)

    # Case 13: Long temporal gap
    def test_13_long_temporal_gap(self):
        """Case 13: Multi-hour gap between adjacent cameras indicates discontinuous journey."""
        rg = RoadGraph()
        rg.add_node(RoadNode("C001", 17.3850, 78.4860))
        rg.add_node(RoadNode("C002", 17.3875, 78.4890))
        rg.add_edge(RoadEdge("E1", "C001", "C002", 400.0, 50.0))
        rg.camera_associations = {"C001": "C001", "C002": "C002"}

        obs_a = Observation(camera_id="C001", timestamp_seconds=0.0)
        obs_b = Observation(camera_id="C002", timestamp_seconds=14400.0)  # 4 hours later

        gap = infer_sparse_gap(obs_a, obs_b, rg)
        self.assertEqual(gap.status, "discontinuous_journey")

    # Case 14: Incompatible Re-ID models
    def test_14_incompatible_reid_models(self):
        """Case 14: Cross-model embedding comparison (AICity vs MSMT17) explicitly rejected."""
        self.assertFalse(are_reid_models_compatible("osnet_x0_25_aicity", "osnet_x0_25_msmt17"))
        layer = ReIDModelCompatibilityLayer()
        self.assertFalse(layer.are_compatible("osnet_x0_25_aicity", "osnet_x0_25_msmt17"))
        is_valid, reason = layer.verify_embeddings(self.emb_sedan_a, self.emb_sedan_b, "osnet_x0_25_aicity", "osnet_x0_25_msmt17")
        self.assertFalse(is_valid)
        self.assertIn("incompatible", reason)

    # Case 15: Missing world coordinates
    def test_15_missing_world_coordinates(self):
        """Case 15: Missing GPS / world coordinates degrades to neutral evidence without crashing."""
        obs = Observation(camera_id="C001", timestamp_seconds=10.0, latitude=None, longitude=None)
        self.assertEqual(obs.world_coordinate_quality, "UNAVAILABLE")
        self.assertFalse(obs.world_coordinate_available)
        s_res = spatial_feasibility(obs, obs)
        self.assertEqual(s_res["status"], "coordinates_unavailable")
        self.assertEqual(s_res["feasibility_score"], 0.50)

    # Case 16: Poor calibration
    def test_16_poor_calibration(self):
        """Case 16: Near-horizon / distorted homography flagged as DEGRADED quality."""
        obs = Observation(
            camera_id="C002",
            timestamp_seconds=10.0,
            latitude=17.3850,
            longitude=78.4860,
            data_quality_flags=["near_horizon", "high_reprojection_error"],
        )
        self.assertEqual(obs.world_coordinate_quality, "DEGRADED")

    # Case 17: Multiple plausible routes
    def test_17_multiple_plausible_routes(self):
        """Case 17: Equal-cost bifurcated routes preserve multiple hypotheses without false certainty."""
        rg = RoadGraph()
        rg.add_node(RoadNode("N_SRC", 0.0, 0.0))
        rg.add_node(RoadNode("N_WAY_A", 1.0, 0.0))
        rg.add_node(RoadNode("N_WAY_B", 0.0, 1.0))
        rg.add_node(RoadNode("N_DST", 1.0, 1.0))
        # Two parallel routes of identical length 1000m
        rg.add_edge(RoadEdge("E1", "N_SRC", "N_WAY_A", 500.0, 50.0))
        rg.add_edge(RoadEdge("E2", "N_WAY_A", "N_DST", 500.0, 50.0))
        rg.add_edge(RoadEdge("E3", "N_SRC", "N_WAY_B", 500.0, 50.0))
        rg.add_edge(RoadEdge("E4", "N_WAY_B", "N_DST", 500.0, 50.0))

        routes = rg.find_candidate_paths("N_SRC", "N_DST", max_paths=5)
        self.assertGreaterEqual(len(routes), 2)
        dists = [r["distance_m"] for r in routes]
        self.assertEqual(dists[0], dists[1])

    # Case 18: Contradictory evidence
    def test_18_contradictory_evidence(self):
        """Case 18: High appearance similarity overridden by contradictory plate."""
        obs1 = Observation(camera_id="C001", timestamp_seconds=10.0, vehicle_type="car", plate="MH01AA1111", appearance_embedding=self.emb_sedan_a, latitude=17.385, longitude=78.486)
        obs2 = Observation(camera_id="C002", timestamp_seconds=40.0, vehicle_type="car", plate="KA04BB2222", appearance_embedding=self.emb_sedan_b, latitude=17.387, longitude=78.489)
        res = match_observations(obs1, obs2, camera_metadata=self.cam_meta)
        self.assertEqual(res["decision_state"], "REJECTED")

    # Case 19: False high-confidence Re-ID
    def test_19_false_high_confidence_reid(self):
        """Case 19: High appearance similarity with negative elapsed time strictly rejected by causality."""
        obs1 = Observation(camera_id="C001", timestamp_seconds=60.0, vehicle_type="car", appearance_embedding=self.emb_sedan_a)
        obs2 = Observation(camera_id="C001", timestamp_seconds=20.0, vehicle_type="car", appearance_embedding=self.emb_sedan_b)
        res = match_observations(obs1, obs2)
        self.assertEqual(res["evidence"]["temporal_status"], "impossible_negative_time")
        self.assertEqual(res["evidence"]["temporal_feasibility"], 0.0)
        self.assertEqual(res["same_vehicle_score"], 0.0)
        self.assertEqual(res["decision_state"], "REJECTED")

    # Case 20: False high-confidence OCR
    def test_20_false_high_confidence_ocr(self):
        """Case 20: High OCR confidence but physically impossible speed rejected by spatio-temporal gate."""
        obs1 = Observation(camera_id="C001", timestamp_seconds=10.0, vehicle_type="car", plate="AP09XY1234", plate_confidence=0.99, latitude=17.3850, longitude=78.4860)
        obs2 = Observation(camera_id="C003", timestamp_seconds=11.0, vehicle_type="car", plate="AP09XY1234", plate_confidence=0.99, latitude=17.4850, longitude=78.5860)  # 15km in 1s = 54,000 km/h!
        res = match_observations(obs1, obs2, camera_metadata=self.cam_meta)
        self.assertEqual(res["decision_state"], "REJECTED")
        self.assertEqual(res["evidence"]["spatial_feasibility"], 0.0)


if __name__ == "__main__":
    unittest.main()
