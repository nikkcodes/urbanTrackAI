"""
Synthetic Benchmark Runner for Day 2 Identity Fusion Engine.
Validates existing identity fusion and identity graph implementations against ground truth.
"""

import json
from pathlib import Path
import unittest

from inference import (
    IdentityGraph,
    Observation,
    load_camera_metadata,
    load_observations_from_json,
    match_observations,
)


class TestIdentityBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base_dir = Path(__file__).parent.parent
        cls.bench_file = base_dir / "data" / "benchmarks" / "identity_fusion_benchmark.json"
        cls.gt_file = base_dir / "data" / "benchmarks" / "identity_fusion_ground_truth.json"

        with open(cls.gt_file, "r", encoding="utf-8") as f:
            cls.gt_data = json.load(f)

        cls.camera_metadata = load_camera_metadata(cls.gt_data["camera_metadata"])
        cls.observations = load_observations_from_json(cls.bench_file, camera_metadata=cls.camera_metadata)
        cls.obs_map = {obs.observation_id: obs for obs in cls.observations}

        # Build pair -> ground truth mapping (Same Vehicle = True, Different = False)
        cls.vehicle_gt = cls.gt_data["vehicle_ground_truth"]
        cls.gt_pair_map = {}
        cls.gt_obs_to_veh = {}

        for veh_id, obs_ids in cls.vehicle_gt.items():
            for o_id in obs_ids:
                cls.gt_obs_to_veh[o_id] = veh_id

        obs_ids_list = list(cls.obs_map.keys())
        for i in range(len(obs_ids_list)):
            for j in range(i + 1, len(obs_ids_list)):
                id1 = obs_ids_list[i]
                id2 = obs_ids_list[j]
                same_gt = cls.gt_obs_to_veh.get(id1) == cls.gt_obs_to_veh.get(id2)
                pair_key = tuple(sorted([id1, id2]))
                cls.gt_pair_map[pair_key] = same_gt

    def test_run_benchmark(self) -> None:
        """Run complete benchmark suite and print detailed metrics and test case evaluations."""
        print("\n" + "=" * 70)
        print("     URBANTRACK AI - DAY 2 IDENTITY FUSION BENCHMARK RUNNER     ")
        print("=" * 70 + "\n")

        # 1. Evaluate All Pairs using EXISTING Engine
        n_obs = len(self.observations)
        n_pairs = (n_obs * (n_obs - 1)) // 2

        tp = 0  # Same-vehicle correctly identified (prob >= 0.5)
        tn = 0  # Different-vehicle correctly rejected (prob < 0.5)
        fp = 0  # Different-vehicle incorrectly matched (prob >= 0.5)
        fn = 0  # Same-vehicle incorrectly rejected (prob < 0.5)
        ambiguous_count = 0
        missing_data_count = 0
        invalid_data_count = 0

        eval_pair_results = {}

        for pair_key, is_same_gt in self.gt_pair_map.items():
            obs_a = self.obs_map[pair_key[0]]
            obs_b = self.obs_map[pair_key[1]]

            # Ensure chronological order
            if obs_a.timestamp_seconds > obs_b.timestamp_seconds:
                obs_a, obs_b = obs_b, obs_a

            res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
            prob = float(res["same_vehicle_probability"])
            eval_pair_results[pair_key] = res

            # Track missing/invalid cases
            if obs_a.appearance_embedding is None or obs_b.appearance_embedding is None:
                missing_data_count += 1
            elif len(obs_a.appearance_embedding) != len(obs_b.appearance_embedding):
                invalid_data_count += 1

            if 0.4 <= prob <= 0.75:
                ambiguous_count += 1

            # Threshold at 0.5
            pred_same = prob >= 0.5

            if is_same_gt and pred_same:
                tp += 1
            elif not is_same_gt and not pred_same:
                tn += 1
            elif not is_same_gt and pred_same:
                fp += 1
            elif is_same_gt and not pred_same:
                fn += 1

        # 2. Build Identity Graph
        graph = IdentityGraph()
        graph.build_graph(self.observations, camera_metadata=self.camera_metadata)
        candidate_identities = graph.get_candidate_identities()

        # 3. Print Scenario Evaluations
        scenarios = [
            ("SCENARIO 1: CLEAR SAME-VEHICLE MATCH", ("obs_001", "obs_002"), True, 0.7, ">="),
            ("SCENARIO 2: CLEAR DIFFERENT-VEHICLE CASE", ("obs_001", "obs_005"), False, 0.3, "<="),
            ("SCENARIO 3: HIGH APPEARANCE BUT IMPOSSIBLE TRAVEL", ("obs_001", "obs_008"), False, 0.1, "<="),
            ("SCENARIO 4: MODERATE / AMBIGUOUS CASE", ("obs_009", "obs_010"), True, 0.5, ">="),
            ("SCENARIO 5: MISSING EMBEDDING", ("obs_013", "obs_014"), True, 0.5, "no_crash"),
            ("SCENARIO 6: INVALID EMBEDDING", ("obs_015", "obs_016"), True, 0.5, "no_crash"),
            ("SCENARIO 7: REPEATED LOCAL OBSERVATIONS", ("obs_017", "obs_018"), True, 0.5, ">="),
            ("SCENARIO 8: MULTI-CAMERA IDENTITY (GT_VEHICLE_01)", ("obs_001", "obs_004"), True, 0.5, ">="),
            ("SCENARIO 9: VISUALLY SIMILAR DIFFERENT VEHICLES", ("obs_001", "obs_008"), False, 0.1, "<="),
        ]

        print("--- SPECIFIC TEST SCENARIO RESULTS ---")
        scenario_passes = 0

        for name, (id1, id2), expected_rel, threshold, op in scenarios:
            obs_a = self.obs_map[id1]
            obs_b = self.obs_map[id2]
            if obs_a.timestamp_seconds > obs_b.timestamp_seconds:
                obs_a, obs_b = obs_b, obs_a

            res = match_observations(obs_a, obs_b, camera_metadata=self.camera_metadata)
            prob = float(res["same_vehicle_probability"])

            passed = False
            if op == ">=":
                passed = prob >= threshold
            elif op == "<=":
                passed = prob <= threshold
            elif op == "range":
                passed = threshold[0] <= prob <= threshold[1]
            elif op == "no_crash":
                passed = True  # Ran without exception

            if passed:
                scenario_passes += 1

            status_str = "PASS" if passed else "FAIL"
            print(f"\n{name}")
            print(f"  Expected Relationship : {'Same Vehicle' if expected_rel else 'Different Vehicle'}")
            print(f"  Actual Prob Output    : {prob:.4f}")
            print(f"  Result                : [{status_str}]")
            print(f"  Explanation           : {res['explanation']}")

        # 4. Print Overall Summary Benchmark Report
        print("\n" + "=" * 70)
        print("                  SUMMARY BENCHMARK REPORT                      ")
        print("=" * 70)
        print(f"Total Observations Evaluated       : {n_obs}")
        print(f"Total Possible Pairs (N*(N-1)/2)   : {n_pairs}")
        print(f"Total Pairs Evaluated              : {n_pairs}")
        print(f"Same-Vehicle Pairs Correct (TP)    : {tp}")
        print(f"Different-Vehicle Correct (TN)     : {tn}")
        print(f"False Positive Matches (FP)        : {fp}")
        print(f"False Negative Matches (FN)        : {fn}")
        print(f"Ambiguous Cases (0.4 <= prob <= 0.75): {ambiguous_count}")
        print(f"Missing-Data Cases                 : {missing_data_count}")
        print(f"Invalid-Data Cases                 : {invalid_data_count}")
        print(f"Test Scenarios Passing             : {scenario_passes} / {len(scenarios)}")
        print(f"Discovered Candidate Identities    : {len(candidate_identities)} clusters")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    unittest.main()
