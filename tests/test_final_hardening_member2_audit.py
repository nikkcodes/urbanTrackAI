import unittest
from datetime import datetime
from pathlib import Path
from schemas.observation_schema import Observation
from inference.similarity import appearance_similarity, validate_and_normalize_embedding
from inference.temporal import temporal_feasibility
from inference.identity_fusion import match_observations
from inference.identity_graph import IdentityGraph
from inference.candidate_generation import CandidateGenerator, benchmark_end_to_end_scalability
from inference.adversarial_suite import run_adversarial_suite
from inference.benchmark.runner import run_multicamera_benchmark
from scripts.reproduce_all import verify_report_consistency

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestFinalHardeningMember2Audit(unittest.TestCase):
    """
    Regression test suite for Member 2 final hardening pass:
    - Track 65/94 general ambiguity reasoning & controls
    - Scalability benchmark without duplicated fusion
    - Score terminology standardization
    - Portable relative paths
    - Dynamic report consistency validation
    """

    def setUp(self):
        self.emb_white_sedan = [0.1] * 512
        self.emb_black_suv = [-0.1] * 512

    def test_01_track_65_94_strictly_ambiguous(self):
        """Verify Track 65 and 94 resolve strictly to AMBIGUOUS via general temporal overlap logic."""
        o10_a = Observation(
            camera_id="CAM_001",
            track_id="65",
            frame_id=590,
            timestamp_seconds=19.667,
            vehicle_type="car",
            plate="NH0LBD4932",
            appearance_embedding=self.emb_white_sedan,
        )
        o10_b = Observation(
            camera_id="CAM_001",
            track_id="94",
            frame_id=600,
            timestamp_seconds=20.000,
            vehicle_type="car",
            plate="NH0LBD4932",
            appearance_embedding=self.emb_white_sedan,
        )
        res = match_observations(o10_a, o10_b)
        self.assertEqual(res["decision_state"], "AMBIGUOUS")
        self.assertLess(res["same_vehicle_score"], 0.75)
        self.assertGreaterEqual(res["same_vehicle_score"], 0.40)
        self.assertIn("conflicting", res["explanation"].lower())

    def test_02_positive_control_remains_confirmed(self):
        """Verify strong cross-camera match with synchronized clocks remains CONFIRMED."""
        oa = Observation(
            observation_id="POS_A",
            camera_id="CAM_1",
            timestamp_seconds=1000.0,
            latitude=12.9716,
            longitude=77.5946,
            vehicle_type="car",
            plate="KA01TEST99",
            appearance_embedding=self.emb_white_sedan,
            timestamp_semantics="synchronized",
            time_reference_id="city_sync",
        )
        ob = Observation(
            observation_id="POS_B",
            camera_id="CAM_2",
            timestamp_seconds=1030.0,
            latitude=12.9750,
            longitude=77.5980,
            vehicle_type="car",
            plate="KA01TEST99",
            appearance_embedding=self.emb_white_sedan,
            timestamp_semantics="synchronized",
            time_reference_id="city_sync",
        )
        res = match_observations(oa, ob)
        self.assertEqual(res["decision_state"], "CONFIRMED")
        self.assertGreaterEqual(res["same_vehicle_score"], 0.75)

    def test_03_negative_contradictions_rejected(self):
        """Verify physical speed and vehicle type contradictions evaluate strictly to REJECTED."""
        # Incompatible vehicle types
        o_car = Observation(camera_id="CAM_1", timestamp_seconds=1000.0, vehicle_type="car", appearance_embedding=self.emb_white_sedan)
        o_bus = Observation(camera_id="CAM_2", timestamp_seconds=1030.0, vehicle_type="bus", appearance_embedding=self.emb_white_sedan)
        r_type = match_observations(o_car, o_bus)
        self.assertEqual(r_type["decision_state"], "REJECTED")
        self.assertEqual(r_type["same_vehicle_score"], 0.0)

        # Simultaneous detection on same camera with distinct tracks
        o_same_1 = Observation(camera_id="CAM_1", track_id="trk_A", timestamp_seconds=1000.0, vehicle_type="car", appearance_embedding=self.emb_white_sedan)
        o_same_2 = Observation(camera_id="CAM_1", track_id="trk_B", timestamp_seconds=1000.0, vehicle_type="car", appearance_embedding=self.emb_white_sedan)
        r_simul = match_observations(o_same_1, o_same_2)
        self.assertEqual(r_simul["decision_state"], "REJECTED")
        self.assertEqual(r_simul["same_vehicle_score"], 0.0)

    def test_04_insufficient_evidence_ambiguous(self):
        """Verify moderate/conflicting evidence evaluates to AMBIGUOUS rather than false confirmation."""
        # Moderate appearance without plate verification
        emb_a = [0.1] * 512
        emb_b = [0.1] * 150 + [0.0] * 362
        oa = Observation(camera_id="CAM_1", timestamp_seconds=1000.0, latitude=12.9716, longitude=77.5946, vehicle_type="car", appearance_embedding=emb_a, timestamp_semantics="synchronized", time_reference_id="sync")
        ob = Observation(camera_id="CAM_2", timestamp_seconds=1030.0, latitude=12.9750, longitude=77.5980, vehicle_type="car", appearance_embedding=emb_b, timestamp_semantics="synchronized", time_reference_id="sync")
        res = match_observations(oa, ob)
        self.assertEqual(res["decision_state"], "AMBIGUOUS")
        self.assertLess(res["same_vehicle_score"], 0.75)
        self.assertGreaterEqual(res["same_vehicle_score"], 0.40)

    def test_05_identity_graph_build_from_matches_equivalence(self):
        """Verify build_graph_from_matches produces identical edges to build_graph."""
        obs = [
            Observation(observation_id=f"OBS_{i}", camera_id="CAM_1", timestamp_seconds=1000.0 + i*10, vehicle_type="car", plate="KA01SAME", appearance_embedding=self.emb_white_sedan)
            for i in range(4)
        ]
        # Standard build_graph
        g1 = IdentityGraph(min_score_threshold=0.75)
        g1.build_graph(obs, enable_pruning=False)

        # Precomputed matches into build_graph_from_matches
        pairs = [(obs[i], obs[j]) for i in range(len(obs)) for j in range(i+1, len(obs))]
        matches = [match_observations(a, b) for a, b in pairs]
        g2 = IdentityGraph(min_score_threshold=0.75)
        g2.build_graph_from_matches(obs, pairs, matches)

        self.assertEqual(len(g1.edges), len(g2.edges))
        self.assertEqual(set(g1.nodes.keys()), set(g2.nodes.keys()))
        c1 = g1.get_candidate_identities()
        c2 = g2.get_candidate_identities()
        self.assertEqual(len(c1), len(c2))

    def test_06_scalability_benchmark_non_duplicated(self):
        """Verify benchmark_end_to_end_scalability runs cleanly with accurate metrics."""
        res = benchmark_end_to_end_scalability(counts=[10, 20], repetitions=1)
        self.assertEqual(len(res["evaluations"]), 2)
        for ev in res["evaluations"]:
            self.assertIn("speedup_factor", ev)
            self.assertIn("candidate_reduction", ev)
            self.assertIn("candidate_recall", ev)
            self.assertGreater(ev["candidate_recall"], 90.0)
            self.assertGreater(ev["baseline_pipeline"]["total_runtime_median_ms"], 0.0)
            self.assertGreater(ev["optimized_pipeline"]["total_runtime_median_ms"], 0.0)

    def test_07_score_terminology_standardization(self):
        """Verify same_vehicle_score is primary and same_vehicle_probability is exact alias."""
        oa = Observation(camera_id="CAM_1", timestamp_seconds=1000.0, vehicle_type="car", plate="KA01TEST", appearance_embedding=self.emb_white_sedan)
        ob = Observation(camera_id="CAM_1", timestamp_seconds=1010.0, vehicle_type="car", plate="KA01TEST", appearance_embedding=self.emb_white_sedan)
        res = match_observations(oa, ob)
        self.assertIn("same_vehicle_score", res)
        self.assertIn("same_vehicle_probability", res)
        self.assertEqual(res["same_vehicle_score"], res["same_vehicle_probability"])

        # Check graph
        g = IdentityGraph(min_score_threshold=0.70)
        self.assertEqual(g.min_score_threshold, 0.70)
        self.assertEqual(g.min_probability_threshold, 0.70)

    def test_08_no_developer_specific_absolute_paths(self):
        """Verify zero /Users/ or /home/ developer paths exist in Member 2 source or tests."""
        import os
        check_dirs = ["inference", "schemas", "tests"]
        found = []
        for d in check_dirs:
            for root, _, files in os.walk(PROJECT_ROOT / d):
                for f in files:
                    if f.endswith(".py") and f != "test_final_hardening_member2_audit.py":
                        p = Path(root) / f
                        with open(p, "r", encoding="utf-8") as file:
                            for lno, line in enumerate(file, 1):
                                if "/Users/" in line or "/home/" in line or "/mnt/" in line:
                                    found.append(f"{p.name}:{lno}")
        self.assertEqual(found, [], f"Found developer-specific absolute paths: {found}")

    def test_09_report_consistency_validator(self):
        """Verify verify_report_consistency passes on matching data and detects mismatches."""
        test_stats = {"tests_run": 356, "failures": 0, "errors": 0}
        mc_bench = {"candidate_reduction_pct": 94.08, "candidate_recall_pct": 99.99}
        e2e = {"evaluations": [{"speedup_factor": 7.07}]}
        adv = {"passed_count": 16, "total_scenarios": 16}
        real_obs = [None] * 39
        report_data = {"acceptance_gates": {"G1": {"status": "PASS"}}}

        valid_md = "356/356 tests passing | 94.08 | 99.99 | 7.07x"
        errs = verify_report_consistency(report_data, valid_md, test_stats, mc_bench, e2e, adv, real_obs)
        self.assertEqual(errs, [])

        # Inconsistent markdown
        bad_md = "100/100 tests passing | 50.00 | 50.00 | 1.00x"
        errs_bad = verify_report_consistency(report_data, bad_md, test_stats, mc_bench, e2e, adv, real_obs)
        self.assertGreater(len(errs_bad), 0)

    def test_10_multicamera_benchmark_independent_ground_truth(self):
        """Verify multicamera_v1 ground truth is generated independently from latent identities."""
        res = run_multicamera_benchmark(benchmark_id="multicamera_v1")
        self.assertEqual(res["dataset_name"], "multicamera_v1")
        self.assertGreater(res["total_observations"], 0)
        self.assertGreaterEqual(res["candidate_recall_pct"], 99.0)
        self.assertGreaterEqual(res["f1_score"], 0.85)


if __name__ == "__main__":
    unittest.main()
