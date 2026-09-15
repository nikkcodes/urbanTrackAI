"""
UrbanTrack AI — Deterministic Train/Development vs Holdout Benchmark (Phases 12 & 13).

Strict Data Split Methodology:
1. Development Split: Used solely to select the optimal operating decision threshold tau*.
2. Holdout Split: Evaluated ONCE using the frozen development threshold.
Zero post-hoc threshold tuning or rule adjustments on holdout data.
"""

import hashlib
import json
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple
from datetime import datetime

from schemas.observation_schema import Observation
from .identity_fusion import match_observations
from .identity_graph import IdentityGraph
from .similarity import appearance_similarity, plate_similarity


def generate_corridor_split(
    seed: int,
    num_vehicles: int,
    cameras: List[Dict[str, Any]],
    start_time: float = 1000.0,
    prefix: str = "DEV",
    num_hard_negatives: int = 2,
) -> Tuple[List[Observation], Dict[str, List[str]]]:
    """
    Generate deterministic benchmark split with empirical OSNet 512-D prototypes,
    realistic Indian registration plates, difficulty tiers, and hard negatives.
    """
    from .benchmark.generator import load_empirical_osnet_prototypes
    from .benchmark.difficulty import DifficultyTier, perturb_plate_text, perturb_embedding, sample_tier

    rng = random.Random(seed)
    observations: List[Observation] = []
    ground_truth_clusters: Dict[str, List[str]] = {}

    state_codes = ["MH", "KA", "TS", "DL"]
    vehicle_types = ["car", "car", "motorcycle", "truck", "bus"]
    osnet_protos = load_empirical_osnet_prototypes()

    def _make_plate(state: str, num: int) -> str:
        series = chr(65 + (num // 1000) % 26) + chr(65 + (num // 100) % 26)
        rto = f"{(num % 90) + 1:02d}"
        seq = f"{num % 9000 + 1000:04d}"
        return f"{state}{rto}{series}{seq}"

    # 1. Regular vehicles
    n_reg = max(2, num_vehicles - num_hard_negatives * 2)
    for v_idx in range(n_reg):
        veh_id = f"{prefix}_VEH_{v_idx:03d}"
        ground_truth_clusters[veh_id] = []
        v_type = rng.choice(vehicle_types)
        plate = _make_plate(rng.choice(state_codes), 1000 + v_idx * 17)
        proto = rng.choice(osnet_protos)

        num_cams_visited = rng.randint(2, len(cameras))
        visited_cams = cameras[:num_cams_visited]
        current_time = start_time + v_idx * 60.0 + rng.uniform(-5.0, 5.0)

        for c_idx, cam in enumerate(visited_cams):
            obs_id = f"{prefix}_OBS_{v_idx:03d}_C{c_idx+1:02d}"
            ground_truth_clusters[veh_id].append(obs_id)

            tier = sample_tier(rng)
            obs_plate, plate_conf = perturb_plate_text(plate, tier, rng=rng)
            noisy_emb, emb_q = perturb_embedding(proto, tier, rng=rng)

            if c_idx > 0:
                dist_m = 400.0
                speed_mps = rng.uniform(10.0, 15.0)
                travel_time = dist_m / speed_mps
                current_time += travel_time

            obs = Observation(
                observation_id=obs_id,
                camera_id=cam["camera_id"],
                timestamp=datetime.fromtimestamp(current_time),
                timestamp_seconds=round(current_time, 2),
                latitude=cam.get("latitude"),
                longitude=cam.get("longitude"),
                vehicle_type=v_type,
                plate=obs_plate,
                plate_confidence=plate_conf,
                ocr_confidence=plate_conf,
                appearance_embedding=noisy_emb,
                source_provenance={"reid_model": "osnet_x0_25_msmt17", "embedding_quality": emb_q},
                timestamp_semantics="synchronized",
                time_reference_id="city_network_sync",
            )
            observations.append(obs)

    # 2. Hard negative vehicle pairs (unseen twins sharing visual prototype but distinct plates)
    for h in range(num_hard_negatives):
        shared_proto = rng.choice(osnet_protos)
        for twin in [1, 2]:
            veh_id = f"{prefix}_HN_{h:02d}_T{twin}"
            ground_truth_clusters[veh_id] = []
            plate = _make_plate("MH" if twin == 1 else "DL", 5000 + h * 50 + twin * 25)
            num_cams_visited = rng.randint(2, len(cameras))
            visited_cams = cameras[:num_cams_visited]
            current_time = start_time + (n_reg + h) * 60.0 + (twin - 1) * 45.0

            for c_idx, cam in enumerate(visited_cams):
                obs_id = f"{prefix}_HN_OBS_{h:02d}_T{twin}_C{c_idx+1:02d}"
                ground_truth_clusters[veh_id].append(obs_id)

                tier = sample_tier(rng)
                obs_plate, plate_conf = perturb_plate_text(plate, tier, rng=rng)
                noisy_emb, emb_q = perturb_embedding(shared_proto, tier, rng=rng)

                if c_idx > 0:
                    dist_m = 400.0
                    speed_mps = rng.uniform(10.0, 15.0)
                    travel_time = dist_m / speed_mps
                    current_time += travel_time

                obs = Observation(
                    observation_id=obs_id,
                    camera_id=cam["camera_id"],
                    timestamp=datetime.fromtimestamp(current_time),
                    timestamp_seconds=round(current_time, 2),
                    latitude=cam.get("latitude"),
                    longitude=cam.get("longitude"),
                    vehicle_type="car",
                    plate=obs_plate,
                    plate_confidence=plate_conf,
                    ocr_confidence=plate_conf,
                    appearance_embedding=noisy_emb,
                    source_provenance={"reid_model": "osnet_x0_25_msmt17", "embedding_quality": emb_q},
                    timestamp_semantics="synchronized",
                    time_reference_id="city_network_sync",
                )
                observations.append(obs)

    return observations, ground_truth_clusters


def evaluate_split_at_threshold(
    observations: List[Observation],
    ground_truth_clusters: Dict[str, List[str]],
    threshold: float,
    camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    obs_map = {o.observation_id: o for o in observations}
    all_obs_ids = sorted(obs_map.keys())
    n = len(all_obs_ids)

    gt_same_pairs: Set[Tuple[str, str]] = set()
    for veh_id, o_ids in ground_truth_clusters.items():
        present_ids = [oid for oid in o_ids if oid in obs_map]
        for i in range(len(present_ids)):
            for j in range(i + 1, len(present_ids)):
                u, v = present_ids[i], present_ids[j]
                gt_same_pairs.add((min(u, v), max(u, v)))

    all_pairs: List[Tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            all_pairs.append((min(all_obs_ids[i], all_obs_ids[j]), max(all_obs_ids[i], all_obs_ids[j])))

    gt_diff_pairs = set(all_pairs) - gt_same_pairs

    graph = IdentityGraph(min_probability_threshold=threshold)
    graph.build_graph(observations, camera_metadata=camera_metadata)
    clusters = graph.get_candidate_identities(camera_metadata=camera_metadata, resolve_contradictions=True)

    predicted_edges: Set[Tuple[str, str]] = set()
    for cand in clusters:
        c_ids = cand.get("observation_ids", [])
        for i in range(len(c_ids)):
            for j in range(i + 1, len(c_ids)):
                predicted_edges.add((min(c_ids[i], c_ids[j]), max(c_ids[i], c_ids[j])))

    tp = len(predicted_edges & gt_same_pairs)
    fp = len(predicted_edges & gt_diff_pairs)
    fn = len(gt_same_pairs - predicted_edges)
    tn = len(gt_diff_pairs - predicted_edges)

    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    false_merge_rate = (fp / len(gt_diff_pairs)) if len(gt_diff_pairs) > 0 else 0.0
    false_split_rate = (fn / len(gt_same_pairs)) if len(gt_same_pairs) > 0 else 0.0

    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "false_merge_rate": round(false_merge_rate, 4),
        "false_split_rate": round(false_split_rate, 4),
        "clusters_count": len(clusters),
    }


def run_train_holdout_benchmark(
    dev_seed: int = 42,
    holdout_seed: int = 1337,
    num_dev_vehicles: int = 12,
    num_holdout_vehicles: int = 15,
) -> Dict[str, Any]:
    cameras_dev = [
        {"camera_id": "CAM_J01", "latitude": 17.3850, "longitude": 78.4867},
        {"camera_id": "CAM_J02", "latitude": 17.3886, "longitude": 78.4867},
        {"camera_id": "CAM_J03", "latitude": 17.3922, "longitude": 78.4867},
        {"camera_id": "CAM_J04", "latitude": 17.3958, "longitude": 78.4867},
    ]
    cameras_holdout = [
        {"camera_id": "CAM_J05", "latitude": 17.4000, "longitude": 78.4900},
        {"camera_id": "CAM_J06", "latitude": 17.4036, "longitude": 78.4900},
        {"camera_id": "CAM_J07", "latitude": 17.4072, "longitude": 78.4900},
        {"camera_id": "CAM_J08", "latitude": 17.4108, "longitude": 78.4900},
    ]

    dev_obs, dev_gt = generate_corridor_split(dev_seed, num_dev_vehicles, cameras_dev, start_time=1000.0, prefix="DEV")
    holdout_obs, holdout_gt = generate_corridor_split(holdout_seed, num_holdout_vehicles, cameras_holdout, start_time=5000.0, prefix="HOLDOUT")

    dev_hash = hashlib.sha256(json.dumps([o.observation_id for o in dev_obs]).encode()).hexdigest()[:16]
    holdout_hash = hashlib.sha256(json.dumps([o.observation_id for o in holdout_obs]).encode()).hexdigest()[:16]

    candidate_thresholds = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
    sweep_results = []
    best_f1 = -1.0
    optimal_threshold = 0.70

    for tau in candidate_thresholds:
        res = evaluate_split_at_threshold(dev_obs, dev_gt, tau)
        sweep_results.append(res)
        if res["f1_score"] > best_f1:
            best_f1 = res["f1_score"]
            optimal_threshold = tau

    holdout_metrics = evaluate_split_at_threshold(holdout_obs, holdout_gt, optimal_threshold)

    return {
        "benchmark_name": "train_holdout_generalization",
        "methodology": "Threshold selected on DEV split, frozen, and evaluated ONCE on HOLDOUT split",
        "dev_split": {
            "seed": dev_seed,
            "dataset_hash": dev_hash,
            "num_vehicles": num_dev_vehicles,
            "num_observations": len(dev_obs),
            "optimal_threshold": optimal_threshold,
            "best_dev_f1": best_f1,
            "threshold_sweep": sweep_results,
        },
        "holdout_split": {
            "seed": holdout_seed,
            "dataset_hash": holdout_hash,
            "num_vehicles": num_holdout_vehicles,
            "num_observations": len(holdout_obs),
            "frozen_evaluation_threshold": optimal_threshold,
            "metrics": holdout_metrics,
        },
    }
