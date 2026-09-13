"""
UrbanTrack AI — Ablation Study Runner.

Quantifies the empirical contribution of each evidence modality across 6 tiers:
Tier A: Plate Only
Tier B: Re-ID Only
Tier C: Plate + Re-ID
Tier D: Plate + Re-ID + Temporal Feasibility
Tier E: Plate + Re-ID + Temporal + Spatial Feasibility
Tier F: Full UrbanTrack (Multi-Modal Fusion + All-Pairs Contradiction Resolution)

Metrics Computed:
- Precision, Recall, F1
- False Merges, False Splits
- Cluster Purity
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
from .similarity import appearance_similarity, plate_similarity


def run_ablation_study(
    observations: List[Observation],
    ground_truth_clusters: Dict[str, List[str]],
    camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Execute empirical ablation across all 6 tiers on a ground-truth observation set.
    """
    obs_map = {o.observation_id: o for o in observations}
    all_obs_ids = sorted(obs_map.keys())
    n = len(all_obs_ids)

    # Build ground truth pairs
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

    tiers = [
        "A_plate_only",
        "B_reid_only",
        "C_plate_and_reid",
        "D_plate_reid_temporal",
        "E_plate_reid_temporal_spatial",
        "F_full_urbantrack",
    ]

    tier_descriptions = {
        "A_plate_only": "Plate string similarity only (no Re-ID, kinematics, or spatial constraints)",
        "B_reid_only": "Re-ID appearance cosine similarity only (no plate or kinematics)",
        "C_plate_and_reid": "Plate + Re-ID feature fusion (no temporal/spatial physical constraints)",
        "D_plate_reid_temporal": "Plate + Re-ID + Temporal order & synchronization checks",
        "E_plate_reid_temporal_spatial": "Plate + Re-ID + Spatio-temporal speed bounds (naive transitive clustering)",
        "F_full_urbantrack": "Full Multi-Modal Evidence Fusion + All-Pairs Contradiction Resolution",
    }

    tier_results = {}

    for tier in tiers:
        predicted_edges: Set[Tuple[str, str]] = set()

        if tier == "A_plate_only":
            for u, v in all_pairs:
                oa, ob = obs_map[u], obs_map[v]
                if oa.plate is not None and ob.plate is not None:
                    p_sim = plate_similarity(oa.plate, ob.plate)
                    if p_sim >= threshold:
                        predicted_edges.add((u, v))

        elif tier == "B_reid_only":
            for u, v in all_pairs:
                oa, ob = obs_map[u], obs_map[v]
                if oa.appearance_embedding and ob.appearance_embedding:
                    a_sim = appearance_similarity(oa.appearance_embedding, ob.appearance_embedding)
                    if a_sim is not None and a_sim >= threshold:
                        predicted_edges.add((u, v))

        elif tier == "C_plate_and_reid":
            for u, v in all_pairs:
                oa, ob = obs_map[u], obs_map[v]
                has_plate = oa.plate is not None and ob.plate is not None
                has_reid = oa.appearance_embedding and ob.appearance_embedding
                if has_plate and has_reid:
                    p_sim = plate_similarity(oa.plate, ob.plate)
                    a_sim = appearance_similarity(oa.appearance_embedding, ob.appearance_embedding) or 0.0
                    score = 0.5 * p_sim + 0.5 * a_sim
                elif has_plate:
                    score = plate_similarity(oa.plate, ob.plate)
                elif has_reid:
                    score = appearance_similarity(oa.appearance_embedding, ob.appearance_embedding) or 0.0
                else:
                    score = 0.0
                if score >= threshold:
                    predicted_edges.add((u, v))

        elif tier == "D_plate_reid_temporal":
            for u, v in all_pairs:
                oa, ob = obs_map[u], obs_map[v]
                # Check temporal ordering
                if oa.timestamp_seconds > ob.timestamp_seconds:
                    ea, eb = ob, oa
                else:
                    ea, eb = oa, ob
                from .temporal import temporal_feasibility
                tf = temporal_feasibility(ea, eb, camera_metadata=camera_metadata)
                t_status = tf.get("status", "unavailable")
                if t_status in ("impossible_negative_time", "impossible_simultaneous_different_cameras"):
                    continue

                # Combine plate and reid
                has_plate = ea.plate is not None and eb.plate is not None
                has_reid = ea.appearance_embedding and eb.appearance_embedding
                if has_plate and has_reid:
                    score = 0.5 * plate_similarity(ea.plate, eb.plate) + 0.5 * (appearance_similarity(ea.appearance_embedding, eb.appearance_embedding) or 0.0)
                elif has_plate:
                    score = plate_similarity(ea.plate, eb.plate)
                elif has_reid:
                    score = appearance_similarity(ea.appearance_embedding, eb.appearance_embedding) or 0.0
                else:
                    score = 0.0

                if score >= threshold:
                    predicted_edges.add((u, v))

        elif tier == "E_plate_reid_temporal_spatial":
            # Uses match_observations but naive connected-components without contradiction splitting
            graph_e = IdentityGraph(min_probability_threshold=threshold)
            for obs in observations:
                graph_e.add_observation(obs)
            graph_e.build_graph(observations, camera_metadata=camera_metadata)
            # Naive connected components (resolve_contradictions=False)
            candidate_clusters = graph_e.get_candidate_identities(camera_metadata=camera_metadata, resolve_contradictions=False)
            for cand in candidate_clusters:
                c_ids = cand.get("observation_ids", [])
                for i in range(len(c_ids)):
                    for j in range(i + 1, len(c_ids)):
                        predicted_edges.add((min(c_ids[i], c_ids[j]), max(c_ids[i], c_ids[j])))

        elif tier == "F_full_urbantrack":
            # Full system with contradiction resolution
            graph_f = IdentityGraph(min_probability_threshold=threshold)
            for obs in observations:
                graph_f.add_observation(obs)
            graph_f.build_graph(observations, camera_metadata=camera_metadata)
            final_clusters = graph_f.get_final_identity_hypotheses(camera_metadata=camera_metadata)
            for clus in final_clusters:
                c_ids = clus.get("observation_ids", [])
                for i in range(len(c_ids)):
                    for j in range(i + 1, len(c_ids)):
                        predicted_edges.add((min(c_ids[i], c_ids[j]), max(c_ids[i], c_ids[j])))

        # Compute Pairwise Metrics
        tp = len(predicted_edges & gt_same_pairs)
        fp = len(predicted_edges - gt_same_pairs)
        fn = len(gt_same_pairs - predicted_edges)
        tn = len(gt_diff_pairs - predicted_edges)

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        false_merge_rate = (fp / len(gt_diff_pairs)) if gt_diff_pairs else 0.0
        false_split_rate = (fn / len(gt_same_pairs)) if gt_same_pairs else 0.0

        # Compute Cluster Purity via connected components of predicted_edges
        parent = {oid: oid for oid in all_obs_ids}
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for u, v in predicted_edges:
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[rv] = ru

        pred_clusters: Dict[str, Set[str]] = {}
        for oid in all_obs_ids:
            r = find(oid)
            if r not in pred_clusters:
                pred_clusters[r] = set()
            pred_clusters[r].add(oid)

        # Ground truth mapping: obs_id -> ground_truth_vehicle_id
        gt_mapping = {}
        for v_id, o_ids in ground_truth_clusters.items():
            for oid in o_ids:
                gt_mapping[oid] = v_id

        # Cluster purity: sum of max class counts in each predicted cluster / total observations
        purity_sum = 0
        for r, members in pred_clusters.items():
            class_counts = Counter(gt_mapping.get(m, "unknown") for m in members)
            purity_sum += class_counts.most_common(1)[0][1]

        cluster_purity = (purity_sum / n) if n > 0 else 1.0

        tier_results[tier] = {
            "name": tier,
            "description": tier_descriptions[tier],
            "pairwise": {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
                "false_merge_rate": round(false_merge_rate, 4),
                "false_split_rate": round(false_split_rate, 4),
            },
            "clusters_formed_count": len(pred_clusters),
            "cluster_purity": round(cluster_purity, 4),
        }

    return {
        "observations_evaluated": n,
        "total_pairs": len(all_pairs),
        "ground_truth_same_pairs": len(gt_same_pairs),
        "ground_truth_diff_pairs": len(gt_diff_pairs),
        "tiers": tier_results,
    }
