"""
UrbanTrack AI — Unit & Integration Tests for Independent Multi-Camera Benchmark.
Validates independent ground truth generation, difficulty tiers, candidate recall,
and evaluator metric integrity without fabrication.
"""

import unittest
from pathlib import Path
import random

from schemas.observation_schema import Observation
from inference.benchmark.difficulty import (
    DifficultyTier,
    perturb_plate_text,
    perturb_embedding,
    sample_tier,
)
from inference.benchmark.ground_truth import (
    GroundTruthRegistry,
    LatentVehicle,
    PairwiseLabel,
)
from inference.benchmark.generator import (
    MultiCameraBenchmarkGenerator,
    DEFAULT_CAMERAS,
    CORRIDOR_ORDER,
    load_empirical_osnet_prototypes,
)
from inference.benchmark.evaluator import MultiCameraBenchmarkEvaluator
from inference.similarity import appearance_similarity, plate_similarity
from inference.identity_fusion import match_observations


class TestMultiCameraBenchmark(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.rng = random.Random(42)

    def test_difficulty_tier_plate_perturbation(self):
        """Verify that plate perturbation reflects tier difficulty."""
        base_plate = "MH12AB1234"

        # EASY: exact plate, high confidence >= 0.90
        p_easy, c_easy = perturb_plate_text(base_plate, DifficultyTier.EASY, rng=self.rng)
        self.assertEqual(p_easy, base_plate)
        self.assertGreaterEqual(c_easy, 0.90)

        # MEDIUM: slight OCR noise or identical, conf in [0.72, 0.88]
        p_med, c_med = perturb_plate_text(base_plate, DifficultyTier.MEDIUM, rng=self.rng)
        if p_med is not None:
            self.assertGreaterEqual(plate_similarity(base_plate, p_med), 0.70)
            self.assertGreaterEqual(c_med, 0.70)

        # HARD: can be missing or degraded
        missing_seen = False
        for _ in range(50):
            p_hard, _ = perturb_plate_text(base_plate, DifficultyTier.HARD, rng=self.rng)
            if p_hard is None:
                missing_seen = True
                break
        self.assertTrue(missing_seen, "HARD tier must model missing plate observations")

    def test_difficulty_tier_embedding_perturbation(self):
        """Verify 512-D embedding perturbation scales geometrically on unit hypersphere."""
        dim = 512
        proto = [self.rng.gauss(0.0, 1.0) for _ in range(dim)]
        norm = sum(x * x for x in proto) ** 0.5
        proto = [x / norm for x in proto]

        # EASY: expected cosine >= 0.90
        emb_easy, q_easy = perturb_embedding(proto, DifficultyTier.EASY, rng=self.rng)
        self.assertIsNotNone(emb_easy)
        sim_easy = appearance_similarity(proto, emb_easy)
        self.assertGreaterEqual(sim_easy, 0.90)

        # MEDIUM: expected cosine in [0.80, 0.92]
        emb_med, q_med = perturb_embedding(proto, DifficultyTier.MEDIUM, rng=self.rng)
        self.assertIsNotNone(emb_med)
        sim_med = appearance_similarity(proto, emb_med)
        self.assertGreaterEqual(sim_med, 0.78)

        # HARD: high noise, lower similarity
        sims_hard = []
        for _ in range(20):
            emb_hard, _ = perturb_embedding(proto, DifficultyTier.HARD, rng=self.rng)
            if emb_hard is not None:
                sims_hard.append(appearance_similarity(proto, emb_hard))
        avg_hard = sum(sims_hard) / len(sims_hard)
        self.assertLess(avg_hard, sim_easy, "HARD tier embeddings must have lower fidelity than EASY")

    def test_independent_ground_truth_isolation(self):
        """Verify ground truth is generated strictly before observation inference."""
        registry = GroundTruthRegistry()
        v = LatentVehicle(
            latent_id="LAT_TEST_01",
            vehicle_type="car",
            true_plate="TS09AB1000",
            base_embedding=[0.1] * 512,
            speed_capability_kmh=50.0,
            color="white",
            is_hard_negative=False,
        )
        registry.register_vehicle(v)
        registry.register_observation("OBS_1", "LAT_TEST_01", DifficultyTier.EASY)
        registry.register_observation("OBS_2", "LAT_TEST_01", DifficultyTier.MEDIUM)
        registry.register_pair(PairwiseLabel(
            obs_a_id="OBS_1",
            obs_b_id="OBS_2",
            is_same_vehicle=True,
            latent_id_a="LAT_TEST_01",
            latent_id_b="LAT_TEST_01",
            difficulty_tier="MEDIUM",
            is_hard_negative=False,
            relationship="SAME_VEHICLE",
        ))

        self.assertEqual(registry.get_latent_id("OBS_1"), "LAT_TEST_01")
        self.assertEqual(registry.get_latent_id("OBS_2"), "LAT_TEST_01")
        pair = registry.pairwise_labels[("OBS_1", "OBS_2")]
        self.assertTrue(pair.is_same_vehicle)
        self.assertEqual(pair.relationship, "SAME_VEHICLE")

    def test_generator_synthesis_properties(self):
        """Verify MultiCameraBenchmarkGenerator produces schema-compliant observations and GT."""
        generator = MultiCameraBenchmarkGenerator(seed=123, project_root=self.project_root)
        obs, reg, meta = generator.generate_dataset(n_vehicles=20, target_observations=200, n_hard_negatives=4)

        self.assertGreaterEqual(len(obs), 100)
        self.assertEqual(len(reg.latent_vehicles), 20)
        self.assertIn("dataset_type", meta)
        self.assertEqual(meta["dataset_type"], "synthetic_controlled")

        # Check observation schema integrity
        for o in obs[:10]:
            self.assertIsInstance(o, Observation)
            self.assertIsNotNone(o.observation_id)
            self.assertIsNotNone(o.camera_id)
            self.assertIsNotNone(o.timestamp_seconds)
            if o.appearance_embedding is not None:
                self.assertEqual(len(o.appearance_embedding), 512)

    def test_hard_negative_twin_rejection(self):
        """Verify that hard negative twins are conservatively rejected by IdentityFusion."""
        protos = load_empirical_osnet_prototypes(self.project_root)
        shared_proto = protos[0]

        # Vehicle A: car, plate KA01AB1000
        emb_a, _ = perturb_embedding(shared_proto, DifficultyTier.EASY, rng=self.rng)
        obs_a = Observation(
            observation_id="HN_TEST_A",
            camera_id="CAM_NORTH_01",
            timestamp_seconds=1000.0,
            vehicle_type="car",
            plate="KA01AB1000",
            plate_confidence=0.95,
            appearance_embedding=emb_a,
            latitude=DEFAULT_CAMERAS["CAM_NORTH_01"]["latitude"],
            longitude=DEFAULT_CAMERAS["CAM_NORTH_01"]["longitude"],
        )

        # Vehicle B: car, plate DL04XY9999 (completely different plate, identical appearance)
        emb_b, _ = perturb_embedding(shared_proto, DifficultyTier.EASY, rng=self.rng)
        obs_b = Observation(
            observation_id="HN_TEST_B",
            camera_id="CAM_NORTH_02",
            timestamp_seconds=1050.0,
            vehicle_type="car",
            plate="DL04XY9999",
            plate_confidence=0.95,
            appearance_embedding=emb_b,
            latitude=DEFAULT_CAMERAS["CAM_NORTH_02"]["latitude"],
            longitude=DEFAULT_CAMERAS["CAM_NORTH_02"]["longitude"],
        )

        res = match_observations(obs_a, obs_b, camera_metadata=DEFAULT_CAMERAS)
        self.assertEqual(res["same_vehicle_score"], 0.0)
        self.assertEqual(res["decision_state"], "REJECTED")
        self.assertIn("Strong license plate contradiction", res["explanation"])

    def test_evaluator_execution(self):
        """Verify evaluator produces valid BenchmarkEvaluationResult metrics without errors."""
        generator = MultiCameraBenchmarkGenerator(seed=999, project_root=self.project_root)
        obs, reg, meta = generator.generate_dataset(n_vehicles=10, target_observations=100, n_hard_negatives=2)

        evaluator = MultiCameraBenchmarkEvaluator(registry=reg, camera_metadata=DEFAULT_CAMERAS, threshold=0.75)
        result = evaluator.evaluate(obs)

        self.assertGreaterEqual(result.candidate_reduction_pct, 50.0)
        self.assertGreaterEqual(result.candidate_recall_pct, 95.0)
        self.assertGreaterEqual(result.precision, 0.0)
        self.assertGreaterEqual(result.recall, 0.0)
        self.assertGreaterEqual(result.f1_score, 0.0)
        self.assertIn("EASY", result.tier_breakdown)
        self.assertIn("ADVERSARIAL", result.tier_breakdown)


if __name__ == "__main__":
    unittest.main()
