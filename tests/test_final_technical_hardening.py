"""
UrbanTrack AI — Comprehensive 22-Scenario Technical Hardening & Adversarial Validation Suite.

Adversarial Stress Testing across all SIH Technical Jury Invariants:
1. Visually identical vehicles at different cameras (doppelgangers)
2. OCR character substitution (plausible variation vs contradiction)
3. Strong license plate contradiction (verified OCR confidence >= 0.50)
4. Missing plate with appearance evidence only
5. Missing appearance with plate evidence only
6. Physically impossible travel speed
7. Network topology violation (one-way reversal)
8. Single missing camera sparse reasoning without observation fabrication
9. Multiple missing cameras in corridor
10. Unsynchronized cameras with unverified temporal reference
11. Equal timestamps across distinct cameras with shared clock
12. Duplicate observation handling
13. Input permutation invariance / determinism
14. Malformed observation inputs
15. Disconnected camera snapping rejection (no fabrication)
16. Disconnected road network (no feasible route)
17. Two plausible routes with Shannon entropy quantification
18. Transitive contradiction splitting and explain_non_merge ledger
19. Route topology sequence discontinuity
20. Low-quality evidence without positive identity modalities
21. Spatio-temporal candidate generator scalability and semantic equivalence
22. Privacy architecture: role views, pseudonymization, retention, and audit trails
"""

import copy
from datetime import datetime
import math
import random
import unittest

from schemas.observation_schema import Observation
from schemas.trajectory_schema import CandidateRoute
from inference.candidate_generation import CandidateGenerator
from inference.identity_fusion import match_observations
from inference.identity_graph import IdentityGraph
from inference.road_graph import RoadEdge, RoadGraph, RoadNode
from inference.sparse_engine import infer_sparse_gap
from inference.trajectory_engine import (
    evaluate_global_trajectory_hypotheses,
    reconstruct_identity_trajectory,
    score_trajectory_hypothesis,
)
from privacy.privacy_guard import AccessRole, PrivacyGuard, pseudonymize_plate


def _make_obs(
    obs_id: str,
    cam_id: str,
    ts: float,
    vtype: str = "car",
    plate: str = None,
    plate_conf: float = None,
    emb: list = None,
    lat: float = 12.9716,
    lon: float = 77.5946,
    sync_status: str = "synchronized",
    time_ref: str = "city_sync_grid_ptp",
) -> Observation:
    return Observation(
        observation_id=obs_id,
        camera_id=cam_id,
        timestamp=datetime.fromtimestamp(max(ts, 0.0)),
        timestamp_seconds=float(ts),
        vehicle_type=vtype,
        plate=plate,
        plate_confidence=plate_conf,
        appearance_embedding=emb,
        latitude=lat,
        longitude=lon,
        timestamp_semantics=sync_status,
        time_reference_id=time_ref,
    )


class TestAdversarialHardening(unittest.TestCase):

    def setUp(self):
        self.camera_meta = {
            "CAM_1": {"latitude": 12.9716, "longitude": 77.5946, "time_reference_id": "city_sync_grid_ptp"},
            "CAM_2": {"latitude": 12.9750, "longitude": 77.5980, "time_reference_id": "city_sync_grid_ptp"},
            "CAM_3": {"latitude": 12.9790, "longitude": 77.6020, "time_reference_id": "city_sync_grid_ptp"},
            "CAM_4": {"latitude": 12.9830, "longitude": 77.6060, "time_reference_id": "city_sync_grid_ptp"},
        }

    # 1. Visually identical vehicles at different cameras
    def test_01_visually_identical_vehicles_different_cameras(self):
        emb = [0.5] * 8
        o1 = _make_obs("ADV_1A", "CAM_1", 1000.0, emb=emb, lat=12.9716, lon=77.5946)
        o2 = _make_obs("ADV_1B", "CAM_3", 1000.0, emb=emb, lat=12.9790, lon=77.6020)
        res = match_observations(o1, o2, camera_metadata=self.camera_meta)
        self.assertEqual(res["same_vehicle_probability"], 0.0)
        self.assertIn("Physically impossible", res["explanation"])

    # 2. OCR character substitution
    def test_02_ocr_character_substitution(self):
        o1 = _make_obs("ADV_2A", "CAM_1", 1000.0, plate="KA01AB1234", plate_conf=0.90)
        o2 = _make_obs("ADV_2B", "CAM_2", 1050.0, plate="KA01AB1284", plate_conf=0.85, lat=12.9750, lon=77.5980)
        res = match_observations(o1, o2, camera_metadata=self.camera_meta)
        self.assertGreater(res["same_vehicle_probability"], 0.70)
        self.assertNotIn("Strong license plate contradiction", res["explanation"])

    # 3. Strong license plate contradiction
    def test_03_strong_plate_contradiction(self):
        o1 = _make_obs("ADV_3A", "CAM_1", 1000.0, plate="KA01AB1234", plate_conf=0.95)
        o2 = _make_obs("ADV_3B", "CAM_2", 1050.0, plate="DL05XY9999", plate_conf=0.95, lat=12.9750, lon=77.5980)
        res = match_observations(o1, o2, camera_metadata=self.camera_meta)
        self.assertEqual(res["same_vehicle_probability"], 0.0)
        self.assertIn("Strong license plate contradiction", res["explanation"])

    # 4. Missing plate with appearance evidence only
    def test_04_missing_plate_appearance_only(self):
        emb = [0.3] * 8
        o1 = _make_obs("ADV_4A", "CAM_1", 1000.0, plate=None, emb=emb)
        o2 = _make_obs("ADV_4B", "CAM_2", 1050.0, plate=None, emb=emb, lat=12.9750, lon=77.5980)
        res = match_observations(o1, o2, camera_metadata=self.camera_meta)
        self.assertGreaterEqual(res["same_vehicle_probability"], 0.80)
        self.assertIsNone(o1.plate)
        self.assertIsNone(o2.plate)

    # 5. Missing appearance with plate evidence only
    def test_05_missing_appearance_plate_only(self):
        o1 = _make_obs("ADV_5A", "CAM_1", 1000.0, plate="KA03MH5555", plate_conf=0.95, emb=None)
        o2 = _make_obs("ADV_5B", "CAM_2", 1050.0, plate="KA03MH5555", plate_conf=0.95, emb=None, lat=12.9750, lon=77.5980)
        res = match_observations(o1, o2, camera_metadata=self.camera_meta)
        self.assertGreaterEqual(res["same_vehicle_probability"], 0.85)
        self.assertIsNone(o1.appearance_embedding)

    # 6. Physically impossible travel speed
    def test_06_impossible_speed(self):
        o1 = _make_obs("ADV_6A", "CAM_1", 1000.0, plate="KA01SPEED", lat=12.9716, lon=77.5946)
        o2 = _make_obs("ADV_6B", "CAM_4", 1001.0, plate="KA01SPEED", lat=12.9830, lon=77.6060)
        res = match_observations(o1, o2, camera_metadata=self.camera_meta)
        self.assertEqual(res["same_vehicle_probability"], 0.0)
        self.assertIn("Physically impossible travel speed", res["explanation"])

    # 7. Network topology violation (one-way reversal)
    def test_07_impossible_topology_one_way(self):
        rg = RoadGraph()
        rg.add_node(RoadNode("N1", 12.9716, 77.5946))
        rg.add_node(RoadNode("N2", 12.9750, 77.5980))
        rg.add_edge(RoadEdge("E12", "N1", "N2", 520.0, speed_limit_kmh=50.0, one_way=True))
        rg.camera_associations["CAM_1"] = "N1"
        rg.camera_associations["CAM_2"] = "N2"

        o_start = _make_obs("ADV_7A", "CAM_2", 1000.0, lat=12.9750, lon=77.5980)
        o_end = _make_obs("ADV_7B", "CAM_1", 1050.0, lat=12.9716, lon=77.5946)
        seg = reconstruct_identity_trajectory({"member_observations": [o_start, o_end]}, rg)
        self.assertEqual(len(seg.complete_route_edges), 0)

    # 8. Single missing camera sparse reasoning
    def test_08_missing_camera_sparse_reasoning(self):
        rg = RoadGraph()
        rg.add_node(RoadNode("N1", 12.9716, 77.5946))
        rg.add_node(RoadNode("N2", 12.9750, 77.5980))
        rg.add_node(RoadNode("N3", 12.9790, 77.6020))
        rg.add_edge(RoadEdge("E12", "N1", "N2", 520.0))
        rg.add_edge(RoadEdge("E23", "N2", "N3", 550.0))
        rg.camera_associations["CAM_1"] = "N1"
        rg.camera_associations["CAM_3"] = "N3"

        o1 = _make_obs("ADV_8A", "CAM_1", 1000.0, lat=12.9716, lon=77.5946)
        o2 = _make_obs("ADV_8B", "CAM_3", 1120.0, lat=12.9790, lon=77.6020)
        gap = infer_sparse_gap(o1, o2, rg)
        self.assertEqual(gap.unobserved_intermediate_nodes, ["N2"])
        self.assertTrue(gap.feasible)

    # 9. Multiple missing cameras in corridor
    def test_09_multiple_missing_cameras(self):
        rg = RoadGraph()
        for i in range(1, 6):
            rg.add_node(RoadNode(f"N{i}", 12.9716 + 0.003*i, 77.5946 + 0.003*i))
        for i in range(1, 5):
            rg.add_edge(RoadEdge(f"E{i}{i+1}", f"N{i}", f"N{i+1}", 500.0))
        rg.camera_associations["CAM_1"] = "N1"
        rg.camera_associations["CAM_5"] = "N5"

        o1 = _make_obs("ADV_9A", "CAM_1", 1000.0, lat=12.9746, lon=77.5976)
        o2 = _make_obs("ADV_9B", "CAM_5", 1300.0, lat=12.9866, lon=77.6096)
        gap = infer_sparse_gap(o1, o2, rg)
        self.assertEqual(gap.unobserved_intermediate_nodes, ["N2", "N3", "N4"])

    # 10. Unsynchronized cameras with unverified temporal reference
    def test_10_unsynchronized_cameras_temporal_unverified(self):
        o1 = _make_obs("ADV_10A", "CAM_1", 1000.0, sync_status="unsynchronized", time_ref="cam1_local")
        o2 = _make_obs("ADV_10B", "CAM_2", 1000.0, sync_status="unsynchronized", time_ref="cam2_local")
        res = match_observations(o1, o2)
        # Missing sync is not contradictory evidence
        self.assertEqual(res["evidence_ledger"]["temporal"]["status"], "unavailable")
        self.assertNotEqual(res["same_vehicle_probability"], 0.0)

    # 11. Equal timestamps across distinct cameras with shared clock
    def test_11_equal_timestamps_distinct_cameras(self):
        o1 = _make_obs("ADV_11A", "CAM_1", 1000.0, sync_status="synchronized", time_ref="city_sync_grid_ptp")
        o2 = _make_obs("ADV_11B", "CAM_2", 1000.0, sync_status="synchronized", time_ref="city_sync_grid_ptp")
        res = match_observations(o1, o2, camera_metadata=self.camera_meta)
        self.assertEqual(res["same_vehicle_probability"], 0.0)
        self.assertIn("Simultaneous detection at different cameras", res["explanation"])

    # 12. Duplicate observation handling
    def test_12_duplicate_observations(self):
        o1 = _make_obs("ADV_12A", "CAM_1", 1000.0, plate="KA01DUP")
        g = IdentityGraph()
        g.build_graph([o1, o1], camera_metadata=self.camera_meta)
        self.assertEqual(len(g.nodes), 1)

    # 13. Input permutation invariance / determinism
    def test_13_input_permutation_determinism(self):
        emb = [0.4] * 8
        obs_list = [
            _make_obs(f"ADV_13_{i}", f"CAM_{i}", 1000.0 + i*40, plate="KA01PERM", emb=emb, lat=12.9716 + i*0.003, lon=77.5946 + i*0.003)
            for i in [1, 2, 3]
        ]
        g1 = IdentityGraph()
        g1.build_graph(obs_list, camera_metadata=self.camera_meta)
        c1 = g1.get_final_identity_hypotheses()

        shuffled = list(reversed(obs_list))
        g2 = IdentityGraph()
        g2.build_graph(shuffled, camera_metadata=self.camera_meta)
        c2 = g2.get_final_identity_hypotheses()

        self.assertEqual(c1[0]["observation_ids"], c2[0]["observation_ids"])
        self.assertEqual(c1[0]["identity_confidence"], c2[0]["identity_confidence"])

    # 14. Malformed observation inputs
    def test_14_malformed_observation_handling(self):
        o1 = _make_obs("ADV_14A", "CAM_1", 1000.0, emb=[1.0, 2.0]) # wrong dimension
        o2 = _make_obs("ADV_14B", "CAM_2", 1050.0, emb=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        res = match_observations(o1, o2, camera_metadata=self.camera_meta)
        self.assertEqual(res["evidence_ledger"]["appearance"]["status"], "invalid")

    # 15. Disconnected camera snapping rejection
    def test_15_disconnected_camera_snapping_rejection(self):
        rg = RoadGraph()
        rg.add_node(RoadNode("N1", 12.9716, 77.5946))
        # Camera is 50 km away
        snapped = rg.associate_camera("CAM_FAR", latitude=13.5000, longitude=78.0000, max_snapping_distance_m=150.0)
        self.assertIsNone(snapped)

    # 16. Disconnected road network
    def test_16_no_feasible_route_between_cameras(self):
        rg = RoadGraph()
        rg.add_node(RoadNode("N1", 12.9716, 77.5946))
        rg.add_node(RoadNode("N2", 12.9750, 77.5980))
        # No edge between N1 and N2
        rg.camera_associations["CAM_1"] = "N1"
        rg.camera_associations["CAM_2"] = "N2"
        o1 = _make_obs("ADV_16A", "CAM_1", 1000.0, lat=12.9716, lon=77.5946)
        o2 = _make_obs("ADV_16B", "CAM_2", 1050.0, lat=12.9750, lon=77.5980)
        seg = reconstruct_identity_trajectory({"member_observations": [o1, o2]}, rg)
        self.assertEqual(len(seg.complete_route_edges), 0)

    # 17. Two plausible routes with Shannon entropy quantification
    def test_17_two_plausible_routes_entropy_quantification(self):
        rg = RoadGraph()
        rg.add_node(RoadNode("J1", 12.9716, 77.5946))
        rg.add_node(RoadNode("J2", 12.9750, 77.5980))
        rg.add_node(RoadNode("J3", 12.9790, 77.6020))
        # Main road J1 -> J3
        rg.add_edge(RoadEdge("E13", "J1", "J3", 1000.0, speed_limit_kmh=50.0))
        # Alternative road J1 -> J2 -> J3
        rg.add_edge(RoadEdge("E12", "J1", "J2", 550.0, speed_limit_kmh=50.0))
        rg.add_edge(RoadEdge("E23", "J2", "J3", 550.0, speed_limit_kmh=50.0))
        rg.camera_associations["CAM_1"] = "J1"
        rg.camera_associations["CAM_3"] = "J3"

        o1 = _make_obs("ADV_17A", "CAM_1", 1000.0, lat=12.9716, lon=77.5946)
        o2 = _make_obs("ADV_17B", "CAM_3", 1090.0, lat=12.9790, lon=77.6020)
        traj = reconstruct_identity_trajectory({"member_observations": [o1, o2]}, rg)
        eval_h = evaluate_global_trajectory_hypotheses(traj)
        self.assertTrue(eval_h["ambiguous"])
        self.assertGreater(eval_h["route_entropy_bits"], 0.80)
        self.assertGreater(eval_h["normalized_route_dispersion"], 0.80)

    # 18. Transitive contradiction splitting and explain_non_merge ledger
    def test_18_transitive_contradiction_splitting(self):
        emb = [0.2] * 8
        # A at t=1000, B at t=1050 (feasible), C at t=1001 (impossible speed to A)
        oa = _make_obs("ADV_18A", "CAM_1", 1000.0, emb=emb, lat=12.9716, lon=77.5946)
        ob = _make_obs("ADV_18B", "CAM_2", 1050.0, emb=emb, lat=12.9750, lon=77.5980)
        oc = _make_obs("ADV_18C", "CAM_3", 1001.0, emb=emb, lat=12.9790, lon=77.6020)

        graph = IdentityGraph()
        graph.build_graph([oa, ob, oc], camera_metadata=self.camera_meta)
        final_hyps = graph.get_final_identity_hypotheses(camera_metadata=self.camera_meta)
        self.assertGreaterEqual(len(final_hyps), 2)

        # Check explain_non_merge
        exp = graph.explain_non_merge("ADV_18A", "ADV_18C", camera_metadata=self.camera_meta)
        self.assertFalse(exp["merged"])
        self.assertEqual(exp["rejection_stage"], "cluster_contradiction_split")

    # 19. Route topology sequence discontinuity
    def test_19_route_topology_discontinuity(self):
        from inference.benchmark_suite import make_benchmark_route
        from schemas.trajectory_schema import TrajectorySegment
        r1 = make_benchmark_route("R1", ["N1", "N2"], ["E1"], dist=500.0, feasible=True)
        r2 = make_benchmark_route("R2", ["N3", "N4"], ["E3"], dist=500.0, feasible=True)
        seg1 = TrajectorySegment(segment_id="s1", identity_id="v1", start_observation_id="TOPO_A", start_camera_id="CAM_1", start_timestamp=1000.0, end_observation_id="TOPO_B", end_camera_id="CAM_2", end_timestamp=1050.0)
        seg2 = TrajectorySegment(segment_id="s2", identity_id="v1", start_observation_id="TOPO_B", start_camera_id="CAM_2", start_timestamp=1050.0, end_observation_id="TOPO_C", end_camera_id="CAM_3", end_timestamp=1100.0)
        h = score_trajectory_hypothesis([r1, r2], [seg1, seg2])
        self.assertFalse(h["is_globally_feasible"])
        self.assertIn("topology_discontinuity", str(h["contradictions"]))

    # 20. Low-quality evidence without positive identity modalities
    def test_20_low_quality_evidence_missing_both_modalities(self):
        o1 = _make_obs("ADV_20A", "CAM_1", 1000.0, plate=None, emb=None)
        o2 = _make_obs("ADV_20B", "CAM_2", 1050.0, plate=None, emb=None, lat=12.9750, lon=77.5980)
        res = match_observations(o1, o2, camera_metadata=self.camera_meta)
        self.assertLessEqual(res["same_vehicle_probability"], 0.50)
        self.assertFalse(res["evidence"]["identity_evidence_available"])

    # 21. Candidate generator scalability and semantic equivalence
    def test_21_candidate_generator_scalability_and_equivalence(self):
        obs = []
        random.seed(123)
        coords = [(12.9716, 77.5946), (12.9750, 77.5980), (12.9790, 77.6020)]
        for i in range(50):
            c_idx = i % 3
            lat, lon = coords[c_idx]
            obs.append(_make_obs(f"CG_{i}", f"CAM_{c_idx+1}", 1000.0 + i*15, lat=lat, lon=lon, plate=f"KA01TEST{i}"))
        cg = CandidateGenerator(camera_metadata=self.camera_meta)
        rep = cg.evaluate_scalability(obs)
        self.assertTrue(rep.semantic_equivalence_verified)
        self.assertEqual(rep.false_exclusions_count, 0)
        self.assertGreater(rep.pruned_pairs_count, 0)

    # 22. Privacy architecture: role views, pseudonymization, retention, and audit trails
    def test_22_privacy_guard_role_views_and_retention(self):
        guard = PrivacyGuard(salt="unit_test_salt_secret")
        o = _make_obs("PRIV_1", "CAM_1", 1000.0, plate="KA01SAFE99", emb=[0.1]*8)

        # 1. Pseudonymization
        pseudo = pseudonymize_plate("KA01SAFE99", salt="unit_test_salt_secret")
        self.assertTrue(pseudo.startswith("PSEUDO_PLATE_"))

        # 2. Roles
        analytics = guard.filter_observation(o, AccessRole.ANALYTICS)
        self.assertIsNone(analytics["plate"])
        self.assertIsNone(analytics["appearance_embedding"])

        audit = guard.filter_observation(o, AccessRole.AUDIT, requester_id="auditor_1", purpose="audit")
        self.assertEqual(audit["plate"], pseudo)

        admin = guard.filter_observation(o, AccessRole.ADMIN, requester_id="admin_1", purpose="admin")
        self.assertEqual(admin["plate"], "KA01SAFE99")

        # 3. Audit trail
        trail = guard.get_audit_trail()
        self.assertEqual(len(trail), 2)

        # 4. Retention
        retained, purged = guard.enforce_retention([o], current_time_seconds=1000.0 + 3000000.0)
        self.assertEqual(purged, 1)
        self.assertIsNone(retained[0].plate)


if __name__ == "__main__":
    unittest.main()
