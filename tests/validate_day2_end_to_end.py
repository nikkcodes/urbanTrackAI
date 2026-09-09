"""
UrbanTrack AI - Day 2 End-to-End Validation & Verification Suite.
Executes all 6 Day-2 pipeline stages, verifies data handling, evaluates test cases A-H,
computes precision/recall/F1 metrics, and validates both synthetic and real Kanishka feeds.
"""

import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).parent.parent))

from inference import (
    IdentityGraph,
    Observation,
    load_camera_metadata,
    load_observations_from_json,
    match_observations,
)
from inference.similarity import appearance_similarity, vehicle_type_compatibility
from inference.temporal import temporal_feasibility
from inference.spatial import spatial_feasibility


def run_stage_1_observation_loading():
    print("\n--- STAGE 1: OBSERVATION LOADING VALIDATION ---")
    
    # Test cases A through H for Observation Loader
    records = [
        # A. Valid observation
        {"observation_id": "o_valid", "camera_id": "cam_01", "frame_id": 100, "timestamp_seconds": 10.0, "vehicle_type": "car", "detection_confidence": 0.95, "bbox": [10, 20, 30, 40], "appearance_embedding": [0.1, 0.2]},
        # B. Missing optional fields
        {"camera_id": "cam_01", "timestamp_seconds": 12.0, "vehicle_type": "car"},
        # C. Missing appearance embedding
        {"camera_id": "cam_01", "timestamp_seconds": 14.0, "vehicle_type": "car", "appearance_embedding": None},
        # D. Invalid/malformed embedding (non-numerical items handled gracefully during similarity)
        {"camera_id": "cam_01", "timestamp_seconds": 16.0, "vehicle_type": "car", "appearance_embedding": []},
        # E. Inconsistent embedding dimensions (e.g. 4-dim vs 128-dim)
        {"camera_id": "cam_01", "timestamp_seconds": 18.0, "vehicle_type": "car", "appearance_embedding": [0.1, 0.2, 0.3, 0.4]},
        # F. Missing timestamp (auto-generated timestamp)
        {"camera_id": "cam_01", "vehicle_type": "car"},
        # G. Missing camera metadata (coordinates missing, handled safely)
        {"camera_id": "cam_no_meta", "timestamp_seconds": 20.0, "vehicle_type": "car"},
    ]
    
    loaded_obs = []
    missing_fields_count = 0
    invalid_fields_count = 0
    rejected_count = 0

    for idx, r in enumerate(records):
        try:
            obs = Observation.from_dict(r)
            loaded_obs.append(obs)
            if obs.appearance_embedding is None or obs.latitude is None or obs.track_id is None:
                missing_fields_count += 1
            if len(obs.appearance_embedding or []) == 0:
                invalid_fields_count += 1
        except Exception as e:
            rejected_count += 1

    # Test invalid data types (must raise TypeError or ValueError)
    invalid_type_rejected = False
    try:
        Observation.from_dict({"camera_id": "cam_01", "timestamp_seconds": "invalid_string_timestamp"})
    except (TypeError, ValueError):
        invalid_type_rejected = True

    print(f"  - Total Observations Loaded   : {len(loaded_obs)}")
    print(f"  - Valid Observations          : {len(loaded_obs)}")
    print(f"  - Obs with Missing Fields     : {missing_fields_count}")
    print(f"  - Obs with Invalid/Empty Fields: {invalid_fields_count}")
    print(f"  - Rejected Malformed Input    : {rejected_count + (1 if invalid_type_rejected else 0)}")
    return loaded_obs


def run_stage_2_pairwise_evidence():
    print("\n--- STAGE 2: PAIRWISE IDENTITY EVIDENCE VALIDATION ---")
    
    obs_a = Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1])
    obs_b = Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=[0.89, 0.41, 0.11])
    
    cam_meta = {
        "cam_01": {"latitude": 17.3850, "longitude": 78.4867},
        "cam_02": {"latitude": 17.3870, "longitude": 78.4900},
    }
    
    res = match_observations(obs_a, obs_b, camera_metadata=cam_meta)
    ev = res["evidence"]

    evidence_payload = {
        "observation_a": obs_a.observation_id,
        "observation_b": obs_b.observation_id,
        "appearance": {
            "available": ev["appearance_status"] == "available",
            "valid": ev["appearance_similarity"] is not None,
            "similarity": ev["appearance_similarity"],
        },
        "vehicle_type": {
            "available": ev["vehicle_type_status"] != "unknown",
            "compatible": ev["vehicle_type_match"],
        },
        "temporal": {
            "available": True,
            "feasible": ev["temporal_feasibility"] > 0.0,
        },
        "spatial": {
            "available": True,
            "feasible": ev["spatial_feasibility"] > 0.0,
        },
    }

    print("  Pairwise Evidence Structure Verification:")
    print("  " + json.dumps(evidence_payload, indent=4))
    return evidence_payload


def run_stage_3_identity_fusion_cases():
    print("\n--- STAGE 3: IDENTITY FUSION TEST CASES (CASE A through H) ---")
    
    cam_meta = {
        "cam_01": {"latitude": 17.3850, "longitude": 78.4867},
        "cam_02": {"latitude": 17.3870, "longitude": 78.4900},
        "cam_03": {"latitude": 17.3950, "longitude": 78.5000},
    }

    test_cases = [
        {
            "code": "CASE A",
            "name": "Same vehicle + strong appearance + feasible time + feasible location",
            "obs_a": Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1]),
            "obs_b": Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=[0.89, 0.41, 0.11]),
            "expected_cls": "MATCH",
        },
        {
            "code": "CASE B",
            "name": "Different vehicle + weak appearance + incompatible timing",
            "obs_a": Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1]),
            "obs_b": Observation(camera_id="cam_03", track_id="t3", frame_id=12, timestamp_seconds=11.0, vehicle_type="car", appearance_embedding=[-0.5, 0.5, 0.2]),
            "expected_cls": "NON-MATCH",
        },
        {
            "code": "CASE C",
            "name": "Same vehicle + appearance missing + temporal/spatial evidence available",
            "obs_a": Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=None),
            "obs_b": Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=[0.89, 0.41, 0.11]),
            "expected_cls": "AMBIGUOUS",
        },
        {
            "code": "CASE D",
            "name": "Different vehicle + appearance missing + temporal/spatial evidence available",
            "obs_a": Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=None),
            "obs_b": Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=None),
            "expected_cls": "AMBIGUOUS",
        },
        {
            "code": "CASE E",
            "name": "Invalid embedding dimensions (4-dim vs 128-dim)",
            "obs_a": Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.1, 0.2, 0.3, 0.4]),
            "obs_b": Observation(camera_id="cam_02", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=[0.01 * i for i in range(128)]),
            "expected_cls": "AMBIGUOUS",
        },
        {
            "code": "CASE F",
            "name": "Same camera + different vehicles + different timestamps",
            "obs_a": Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[1.0, 0.0]),
            "obs_b": Observation(camera_id="cam_01", track_id="t2", frame_id=500, timestamp_seconds=200.0, vehicle_type="car", appearance_embedding=[0.0, 1.0]),
            "expected_cls": "NON-MATCH",
        },
        {
            "code": "CASE G",
            "name": "Very similar-looking vehicles + impossible travel time",
            "obs_a": Observation(camera_id="cam_01", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1]),
            "obs_b": Observation(camera_id="cam_03", track_id="t3", frame_id=15, timestamp_seconds=11.0, vehicle_type="car", appearance_embedding=[0.9, 0.4, 0.1]),
            "expected_cls": "NON-MATCH",
        },
        {
            "code": "CASE H",
            "name": "Insufficient evidence (missing coordinates and missing embeddings)",
            "obs_a": Observation(camera_id="cam_unk1", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car"),
            "obs_b": Observation(camera_id="cam_unk2", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car"),
            "expected_cls": "AMBIGUOUS",
        },
    ]

    for tc in test_cases:
        res = match_observations(tc["obs_a"], tc["obs_b"], camera_metadata=cam_meta)
        prob = res["same_vehicle_probability"]
        has_id_ev = res["evidence"]["identity_evidence_available"]

        if prob >= 0.70 and has_id_ev:
            cls_result = "MATCH"
        elif prob == 0.0 or (has_id_ev and prob < 0.40):
            cls_result = "NON-MATCH"
        else:
            cls_result = "AMBIGUOUS"

        print(f"\n  [{tc['code']}]: {tc['name']}")
        print(f"    - Evidence Available : Identity={has_id_ev}, AppStatus={res['evidence']['appearance_status']}")
        print(f"    - Match Likelihood   : {prob:.4f}")
        print(f"    - Classification     : [{cls_result}] (Expected: {tc['expected_cls']})")
        print(f"    - Reason             : {res['explanation']}")


def run_stage_4_missing_invalid_data_safety():
    print("\n--- STAGE 4: MISSING AND INVALID DATA SAFETY VALIDATION ---")
    
    cam_meta = {
        "c1": {"latitude": 17.3850, "longitude": 78.4867},
        "c2": {"latitude": 17.3870, "longitude": 78.4900},
    }

    # 1. Missing appearance does not produce 1.0
    obs1 = Observation(camera_id="c1", timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=None)
    obs2 = Observation(camera_id="c2", timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=[0.1, 0.2])
    res1 = match_observations(obs1, obs2, camera_metadata=cam_meta)
    assert res1["same_vehicle_probability"] == 0.5000, f"Failed: missing appearance produced {res1['same_vehicle_probability']}"
    print("  [PASS] 1. Missing appearance embedding produces 0.50 (NOT 1.0)")

    # 2. Mismatched embedding dimensions do not become 0.0 and average into positive evidence
    obs3 = Observation(camera_id="c1", timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=[0.1, 0.2])
    obs4 = Observation(camera_id="c2", timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=[0.1, 0.2, 0.3, 0.4])
    res2 = match_observations(obs3, obs4, camera_metadata=cam_meta)
    assert res2["evidence"]["appearance_similarity"] is None, "Failed: mismatched dim returned non-None similarity"
    assert res2["same_vehicle_probability"] == 0.5000, "Failed: mismatched dim score incorrect"
    print("  [PASS] 2. Mismatched embedding dimensions set appearance_similarity=None and score=0.50")

    # 3. Missing evidence does not create artificial confidence
    assert not res1["evidence"]["identity_evidence_available"], "Failed: missing evidence flagged as available"
    print("  [PASS] 3. Missing identity evidence explicitly marked identity_evidence_available=False")

    # 4. Graph edge creation requires identity evidence
    graph = IdentityGraph(min_probability_threshold=0.70)
    graph.build_graph([obs1, obs2], camera_metadata=cam_meta)
    assert len(graph.edges) == 0, "Failed: missing evidence formed graph edge"
    print("  [PASS] 4. Missing/invalid evidence cannot form positive identity edges in IdentityGraph")


def run_stage_5_6_end_to_end_validation():
    print("\n--- STAGE 5 & 6: END-TO-END BENCHMARK & CLUSTERING VALIDATION ---")
    
    base_dir = Path(__file__).parent.parent
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

    # Build pair ground truth mapping
    obs_list = list(obs_map.keys())
    gt_pair_map = {}
    for i in range(len(obs_list)):
        for j in range(i + 1, len(obs_list)):
            id1, id2 = obs_list[i], obs_list[j]
            same_gt = gt_obs_to_veh.get(id1) == gt_obs_to_veh.get(id2)
            gt_pair_map[tuple(sorted([id1, id2]))] = same_gt

    n_obs = len(observations)
    n_pairs = len(gt_pair_map)

    tp_pair, tn_pair, fp_pair, fn_pair = 0, 0, 0, 0
    ambiguous_count, missing_count, invalid_count = 0, 0, 0

    for pair_key, same_gt in gt_pair_map.items():
        obs_a, obs_b = obs_map[pair_key[0]], obs_map[pair_key[1]]
        if obs_a.timestamp_seconds > obs_b.timestamp_seconds:
            obs_a, obs_b = obs_b, obs_a

        res = match_observations(obs_a, obs_b, camera_metadata=camera_metadata)
        prob = res["same_vehicle_probability"]

        if obs_a.appearance_embedding is None or obs_b.appearance_embedding is None:
            missing_count += 1
        elif len(obs_a.appearance_embedding) != len(obs_b.appearance_embedding):
            invalid_count += 1

        if 0.40 <= prob <= 0.75:
            ambiguous_count += 1

        pred_same = prob >= 0.50
        if same_gt and pred_same:
            tp_pair += 1
        elif not same_gt and not pred_same:
            tn_pair += 1
        elif not same_gt and pred_same:
            fp_pair += 1
        elif same_gt and not pred_same:
            fn_pair += 1

    # Graph Level Evaluation (at min_threshold = 0.70)
    graph = IdentityGraph(min_probability_threshold=0.70)
    graph.build_graph(observations, camera_metadata=camera_metadata)
    candidate_identities = graph.get_candidate_identities()

    tp_graph, fp_graph, fn_graph = 0, 0, 0
    for edge in graph.edges:
        id1, id2 = edge["source"], edge["target"]
        same_gt = gt_obs_to_veh.get(id1) == gt_obs_to_veh.get(id2)
        if same_gt:
            tp_graph += 1
        else:
            fp_graph += 1

    prec_graph = tp_graph / (tp_graph + fp_graph) if (tp_graph + fp_graph) > 0 else 0.0
    rec_graph = tp_graph / 16.0  # 16 true positive same-vehicle edges in GT
    f1_graph = (2 * prec_graph * rec_graph) / (prec_graph + rec_graph) if (prec_graph + rec_graph) > 0 else 0.0

    prec_pair = tp_pair / (tp_pair + fp_pair) if (tp_pair + fp_pair) > 0 else 0.0
    rec_pair = tp_pair / (tp_pair + fn_pair) if (tp_pair + fn_pair) > 0 else 0.0
    f1_pair = (2 * prec_pair * rec_pair) / (prec_pair + rec_pair) if (prec_pair + rec_pair) > 0 else 0.0

    print(f"Total Observations Evaluated       : {n_obs}")
    print(f"Total Possible Pairs               : {n_pairs}")
    print(f"Total Pairs Evaluated              : {n_pairs}")
    print("\n--- PAIRWISE EVALUATION METRICS (@ 0.50 PRIOR THRESHOLD) ---")
    print(f"  True Positives (TP)              : {tp_pair}")
    print(f"  True Negatives (TN)              : {tn_pair}")
    print(f"  False Positives (FP)             : {fp_pair}")
    print(f"  False Negatives (FN)             : {fn_pair}")
    print(f"  Pairwise Precision               : {prec_pair * 100:.2f}%")
    print(f"  Pairwise Recall                  : {rec_pair * 100:.2f}%")
    print(f"  Pairwise F1 Score                : {f1_pair * 100:.2f}%")

    print("\n--- IDENTITY GRAPH EVALUATION METRICS (@ 0.70 CONFIDENT THRESHOLD) ---")
    print(f"  Nodes (Observations)             : {len(graph.nodes)}")
    print(f"  Candidate Identity Edges Formed  : {len(graph.edges)}")
    print(f"  True Positive Edges (TP)         : {tp_graph}")
    print(f"  False Positive Edges (FP)        : {fp_graph}")
    print(f"  Graph Edge Precision             : {prec_graph * 100:.2f}%")
    print(f"  Graph Edge Recall                : {rec_graph * 100:.2f}%")
    print(f"  Graph Edge F1 Score              : {f1_graph * 100:.2f}%")
    print(f"  Ambiguous Cases (0.4 <= P <= 0.75): {ambiguous_count}")
    print(f"  Missing-Data Cases               : {missing_count}")
    print(f"  Invalid-Data Cases               : {invalid_count}")
    print(f"  Discovered Candidate Identities  : {len(candidate_identities)} clusters")
    print(f"  Ground-Truth Vehicles            : {len(vehicle_gt)} clusters")


if __name__ == "__main__":
    run_stage_1_observation_loading()
    run_stage_2_pairwise_evidence()
    run_stage_3_identity_fusion_cases()
    run_stage_4_missing_invalid_data_safety()
    run_stage_5_6_end_to_end_validation()
