"""
UrbanTrack AI — Clean Modality Ablation Study Runner (Phase 10 Hardened).

Evaluates 6 strictly isolated evidence tiers on identical datasets, identical ground truth,
and identical decision threshold methodology without silent fallbacks between modalities.

Tier A: Re-ID Only (OSNet appearance cosine similarity alone)
Tier B: Plate Only (License plate string similarity alone)
Tier C: Re-ID + Plate (Fixed equal fusion; no fallback to unimodal when one is missing)
Tier D: Re-ID + Plate + Temporal (Adds temporal order and cross-camera simultaneity checks)
Tier E: Re-ID + Plate + Spatial (Adds spatial speed limit bounds)
Tier F: Full UrbanTrack (Multimodal fusion + vehicle type + camera reliability + contradiction logic)
"""

from collections import Counter
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from schemas.observation_schema import Observation
from .identity_fusion import match_observations
from .identity_graph import IdentityGraph
from .similarity import appearance_similarity, plate_similarity, geographic_distance
from .temporal import check_temporal_comparability


def run_ablation_study(
    observations: List[Observation],
    ground_truth_clusters: Dict[str, List[str]],
    camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Execute empirical ablation across all 6 tiers on a ground-truth observation set.
    Ensures zero silent fallbacks across tiers.
    """
    obs_map = {o.observation_id: o for o in observations}
    all_obs_ids = sorted(obs_map.keys())
    n = len(all_obs_ids)

    # Build ground truth same/different pairs
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
    total_pairs_count = len(all_pairs)

    tiers = [
        "A_reid_only",
        "B_plate_only",
        "C_reid_and_plate",
        "D_reid_plate_temporal",
        "E_reid_plate_spatial",
        "F_full_urbantrack",
    ]

    tier_descriptions = {
        "A_reid_only": "OSNet appearance cosine similarity only (no plate, kinematics, or spatial constraints)",
        "B_plate_only": "License plate string similarity only (no Re-ID, kinematics, or spatial constraints)",
        "C_reid_and_plate": "Re-ID + Plate fixed 50/50 fusion (strictly zero unimodal fallback if one modality is absent)",
        "D_reid_plate_temporal": "Re-ID + Plate + Temporal order & cross-camera simultaneity gating",
        "E_reid_plate_spatial": "Re-ID + Plate + Spatial travel speed feasibility gating (<= 120 km/h)",
        "F_full_urbantrack": "Full UrbanTrack (Multimodal fusion + vehicle type compatibility + camera reliability + contradiction logic)",
    }

    tier_results = {}

    for tier in tiers:
        predicted_edges: Set[Tuple[str, str]] = set()
        eligible_pairs_count = 0

        if tier == "A_reid_only":
            for u, v in all_pairs:
                oa, ob = obs_map[u], obs_map[v]
                if oa.appearance_embedding and ob.appearance_embedding:
                    eligible_pairs_count += 1
                    a_sim = appearance_similarity(oa.appearance_embedding, ob.appearance_embedding)
                    if a_sim is not None and a_sim >= threshold:
                        predicted_edges.add((u, v))

        elif tier == "B_plate_only":
            for u, v in all_pairs:
                oa, ob = obs_map[u], obs_map[v]
                if oa.plate is not None and ob.plate is not None:
                    eligible_pairs_count += 1
                    p_sim = plate_similarity(oa.plate, ob.plate)
                    if p_sim >= threshold:
                        predicted_edges.add((u, v))

        elif tier == "C_reid_and_plate":
            for u, v in all_pairs:
                oa, ob = obs_map[u], obs_map[v]
                has_plate = oa.plate is not None and ob.plate is not None
                has_reid = bool(oa.appearance_embedding and ob.appearance_embedding)
                if has_plate or has_reid:
                    eligible_pairs_count += 1
                p_sim = plate_similarity(oa.plate, ob.plate) if has_plate else 0.0
                a_sim = (appearance_similarity(oa.appearance_embedding, ob.appearance_embedding) or 0.0) if has_reid else 0.0
                # Strict 50/50 fusion without fallback to 1.0 weight
                score = 0.5 * p_sim + 0.5 * a_sim
                if score >= threshold:
                    predicted_edges.add((u, v))

        elif tier == "D_reid_plate_temporal":
            for u, v in all_pairs:
                oa, ob = obs_map[u], obs_map[v]
                has_plate = oa.plate is not None and ob.plate is not None
                has_reid = bool(oa.appearance_embedding and ob.appearance_embedding)
                if has_plate or has_reid:
                    eligible_pairs_count += 1

                # Temporal check
                if oa.timestamp_seconds > ob.timestamp_seconds:
                    ea, eb = ob, oa
                else:
                    ea, eb = oa, ob
                t_comp = check_temporal_comparability(ea, eb, camera_metadata=camera_metadata)
                if t_comp.get("comparable"):
                    dt = float(t_comp.get("delta_seconds", 0.0))
                    if dt == 0.0 and ea.camera_id != eb.camera_id:
                        # Impossible simultaneous on distinct cameras
                        continue
                    if dt < 0.0 and ea.camera_id == eb.camera_id:
                        # Impossible negative elapsed time on same camera
                        continue

                p_sim = plate_similarity(oa.plate, ob.plate) if has_plate else 0.0
                a_sim = (appearance_similarity(oa.appearance_embedding, ob.appearance_embedding) or 0.0) if has_reid else 0.0
                score = 0.5 * p_sim + 0.5 * a_sim
                if score >= threshold:
                    predicted_edges.add((u, v))

        elif tier == "E_reid_plate_spatial":
            for u, v in all_pairs:
                oa, ob = obs_map[u], obs_map[v]
                has_plate = oa.plate is not None and ob.plate is not None
                has_reid = bool(oa.appearance_embedding and ob.appearance_embedding)
                if has_plate or has_reid:
                    eligible_pairs_count += 1

                # Spatial speed check
                if oa.latitude is not None and oa.longitude is not None and ob.latitude is not None and ob.longitude is not None:
                    dist_m = geographic_distance(oa.latitude, oa.longitude, ob.latitude, ob.longitude)
                    dt = abs(oa.timestamp_seconds - ob.timestamp_seconds)
                    if dt > 0:
                        speed_kmh = (dist_m / dt) * 3.6
                        if speed_kmh > 120.0:
                            continue

                p_sim = plate_similarity(oa.plate, ob.plate) if has_plate else 0.0
                a_sim = (appearance_similarity(oa.appearance_embedding, ob.appearance_embedding) or 0.0) if has_reid else 0.0
                score = 0.5 * p_sim + 0.5 * a_sim
                if score >= threshold:
                    predicted_edges.add((u, v))

        elif tier == "F_full_urbantrack":
            graph_f = IdentityGraph(min_probability_threshold=threshold)
            graph_f.build_graph(observations, camera_metadata=camera_metadata)
            clusters = graph_f.get_candidate_identities(camera_metadata=camera_metadata, resolve_contradictions=True)
            eligible_pairs_count = total_pairs_count
            for cand in clusters:
                c_ids = cand.get("observation_ids", [])
                for i in range(len(c_ids)):
                    for j in range(i + 1, len(c_ids)):
                        predicted_edges.add((min(c_ids[i], c_ids[j]), max(c_ids[i], c_ids[j])))

        # Compute classification metrics against ground truth
        tp = len(predicted_edges & gt_same_pairs)
        fp = len(predicted_edges & gt_diff_pairs)
        fn = len(gt_same_pairs - predicted_edges)
        tn = len(gt_diff_pairs - predicted_edges)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        false_merge_rate = (fp / len(gt_diff_pairs)) if len(gt_diff_pairs) > 0 else 0.0
        false_split_rate = (fn / len(gt_same_pairs)) if len(gt_same_pairs) > 0 else 0.0

        # Connected component clustering for purity calculation
        adj: Dict[str, Set[str]] = {oid: set() for oid in all_obs_ids}
        for u, v in predicted_edges:
            adj[u].add(v)
            adj[v].add(u)

        visited: Set[str] = set()
        clusters_found = []
        for oid in all_obs_ids:
            if oid not in visited:
                comp = set()
                q = [oid]
                visited.add(oid)
                while q:
                    curr = q.pop()
                    comp.add(curr)
                    for nxt in adj[curr]:
                        if nxt not in visited:
                            visited.add(nxt)
                            q.append(nxt)
                clusters_found.append(comp)

        # Ground truth mapping: obs_id -> true vehicle_id
        obs_to_gt = {}
        for veh_id, o_ids in ground_truth_clusters.items():
            for oid in o_ids:
                obs_to_gt[oid] = veh_id

        # Cluster purity: sum of majority ground truth count / total observations
        total_pure_obs = 0
        for comp in clusters_found:
            labels = [obs_to_gt[oid] for oid in comp if oid in obs_to_gt]
            if labels:
                most_common_cnt = Counter(labels).most_common(1)[0][1]
                total_pure_obs += most_common_cnt
            else:
                total_pure_obs += len(comp)

        purity = (total_pure_obs / n) if n > 0 else 0.0
        coverage = (eligible_pairs_count / total_pairs_count * 100.0) if total_pairs_count > 0 else 100.0

        tier_results[tier] = {
            "tier_name": tier,
            "description": tier_descriptions.get(tier, ""),
            "evaluated_pairs": total_pairs_count,
            "eligible_pairs": eligible_pairs_count,
            "coverage_pct": round(coverage, 2),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_merge_rate": round(false_merge_rate, 4),
            "false_split_rate": round(false_split_rate, 4),
            "cluster_count": len(clusters_found),
            "cluster_purity": round(purity, 4),
            "pairwise": {
                "false_merge_rate": round(false_merge_rate, 4),
                "false_split_rate": round(false_split_rate, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
            },
        }

    return {
        "benchmark_name": "modality_ablation_study",
        "threshold": threshold,
        "n_observations": n,
        "total_pairs": total_pairs_count,
        "gt_same_pairs": len(gt_same_pairs),
        "gt_diff_pairs": len(gt_diff_pairs),
        "tiers": tier_results,
    }
