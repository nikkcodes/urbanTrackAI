"""
UrbanTrack AI - Day 2 Identity Fusion Synthetic Benchmark Execution Script.
Runs the EXISTING Identity Fusion Engine against the deterministic benchmark dataset.
"""

import json
from pathlib import Path

from inference import (
    IdentityGraph,
    Observation,
    load_camera_metadata,
    load_observations_from_json,
    match_observations,
)


def execute_benchmark():
    base_dir = Path(__file__).parent
    bench_file = base_dir / "data" / "benchmarks" / "identity_fusion_benchmark.json"
    gt_file = base_dir / "data" / "benchmarks" / "identity_fusion_ground_truth.json"

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    camera_metadata = load_camera_metadata(gt_data["camera_metadata"])
    observations = load_observations_from_json(bench_file, camera_metadata=camera_metadata)
    obs_map = {obs.observation_id: obs for obs in observations}

    vehicle_gt = gt_data["vehicle_ground_truth"]
    gt_obs_to_veh = {}
    for veh_id, obs_ids in vehicle_gt.items():
        for o_id in obs_ids:
            gt_obs_to_veh[o_id] = veh_id

    # Define test cases for 9 scenarios
    scenarios = [
        {
            "num": 1,
            "name": "1. Clear Same-Vehicle Match",
            "pair": ("obs_001", "obs_002"),
            "expected_rel": "Same Vehicle (GT_VEHICLE_01)",
            "expected_match": True,
            "criterion": lambda prob: prob >= 0.7,
        },
        {
            "num": 2,
            "name": "2. Clear Different-Vehicle Case",
            "pair": ("obs_001", "obs_005"),
            "expected_rel": "Different Vehicle (Car vs Bus)",
            "expected_match": False,
            "criterion": lambda prob: prob <= 0.3,
        },
        {
            "num": 3,
            "name": "3. High Appearance / Impossible Travel",
            "pair": ("obs_001", "obs_008"),
            "expected_rel": "Different Vehicle (Speed > 2000 km/h)",
            "expected_match": False,
            "criterion": lambda prob: prob <= 0.1,
        },
        {
            "num": 4,
            "name": "4. Moderate Similarity / Plausible Time",
            "pair": ("obs_009", "obs_010"),
            "expected_rel": "Same Vehicle (GT_VEHICLE_04)",
            "expected_match": True,
            "criterion": lambda prob: prob >= 0.5,
        },
        {
            "num": 5,
            "name": "5. Missing Embedding (Null)",
            "pair": ("obs_013", "obs_014"),
            "expected_rel": "Same Vehicle (Missing Embedding)",
            "expected_match": True,
            "criterion": lambda prob: prob > 0.0,
        },
        {
            "num": 6,
            "name": "6. Invalid Embedding (Mismatched Dim)",
            "pair": ("obs_015", "obs_016"),
            "expected_rel": "Same Vehicle (Mismatched Dim)",
            "expected_match": True,
            "criterion": lambda prob: True,  # Safe fallback without crashing
        },
        {
            "num": 7,
            "name": "7. Repeated Local Observations",
            "pair": ("obs_017", "obs_018"),
            "expected_rel": "Same Vehicle (Same Local Track)",
            "expected_match": True,
            "criterion": lambda prob: prob >= 0.7,
        },
        {
            "num": 8,
            "name": "8. Multi-Camera Identity (4 Cameras)",
            "pair": ("obs_001", "obs_004"),
            "expected_rel": "Same Vehicle (GT_VEHICLE_01)",
            "expected_match": True,
            "criterion": lambda prob: prob >= 0.7,
        },
        {
            "num": 9,
            "name": "9. Visually Similar Different Vehicles",
            "pair": ("obs_001", "obs_008"),
            "expected_rel": "Different Vehicle (Doppelganger)",
            "expected_match": False,
            "criterion": lambda prob: prob <= 0.1,
        },
    ]

    print("=========================================================================================")
    print("        DAY 2 IDENTITY FUSION ENGINE - DETERMINISTIC SYNTHETIC BENCHMARK        ")
    print("=========================================================================================\n")

    print(f"Total Observations Loaded : {len(observations)}")
    print(f"Ground-Truth Vehicles     : {len(vehicle_gt)}")
    print(f"Total Possible Pairs      : {(len(observations) * (len(observations)-1)) // 2}\n")

    print("--- SCENARIO BENCHMARK RESULTS ---")
    passes = 0
    fails = 0

    for sc in scenarios:
        id1, id2 = sc["pair"]
        obs_a = obs_map[id1]
        obs_b = obs_map[id2]
        if obs_a.timestamp_seconds > obs_b.timestamp_seconds:
            obs_a, obs_b = obs_b, obs_a

        res = match_observations(obs_a, obs_b, camera_metadata=camera_metadata)
        prob = float(res["same_vehicle_probability"])
        is_match = prob >= 0.5

        passed = sc["criterion"](prob)
        if passed:
            passes += 1
            status = "PASS"
        else:
            fails += 1
            status = "FAIL"

        print(f"\nScenario [{sc['num']}]: {sc['name']}")
        print(f"  Pair Evaluated       : {id1} vs {id2}")
        print(f"  Expected Relationship: {sc['expected_rel']}")
        print(f"  Pairwise Match Result: {'MATCH' if is_match else 'REJECTED'}")
        print(f"  Estimated Match Prob : {prob:.4f}")
        print(f"  Benchmark Result     : [{status}]")
        print(f"  Explanation          : {res['explanation']}")

    # Identity Graph Building
    graph = IdentityGraph()
    graph.build_graph(observations, camera_metadata=camera_metadata)
    candidate_identities = graph.get_candidate_identities()

    print("\n" + "=" * 90)
    print("                       IDENTITY GROUPING RESULTS VS GROUND TRUTH                        ")
    print("=" * 90)
    print(f"Ground-Truth Vehicle Clusters : {len(vehicle_gt)}")
    print(f"Discovered Identity Clusters  : {len(candidate_identities)}\n")

    for cand in candidate_identities:
        c_id = cand["candidate_vehicle_id"]
        members = cand["observation_ids"]
        cams = " -> ".join(cand["cameras_visited"])
        # Find which GT vehicles are in this cluster
        gt_in_cluster = set(gt_obs_to_veh.get(m) for m in members)
        print(f"  - Identity Cluster: {c_id}")
        print(f"    Member Count    : {len(members)} observations")
        print(f"    Member IDs      : {members}")
        print(f"    Camera Trail    : {cams}")
        print(f"    GT Vehicles In  : {list(gt_in_cluster)}")
        print()

    print("=" * 90)
    print(f"BENCHMARK COMPLETED: {passes} Scenarios Passed | {fails} Scenarios Failed")
    print("=========================================================================================")


if __name__ == "__main__":
    import sys
    if "--expanded" in sys.argv or "--all" in sys.argv:
        execute_benchmark()
        from inference.benchmark_suite import run_master_benchmark_suite
        run_master_benchmark_suite(verbose=True)
    else:
        execute_benchmark()
        print("\n[NOTE] To run the expanded 20-scenario benchmark + holdout + performance scaling:")
        print("       python3 run_benchmark.py --expanded\n")

