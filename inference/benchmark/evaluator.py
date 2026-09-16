"""
UrbanTrack AI — Multi-Camera Benchmark Evaluator.
Rigorously evaluates production CandidateGenerator, IdentityFusion, and IdentityGraph
against independent ground truth with difficulty tier breakdowns and hard-negative safety.
"""

from dataclasses import asdict, dataclass
import math
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from schemas.observation_schema import Observation
from inference.candidate_generation import CandidateGenerator
from inference.identity_fusion import match_observations
from inference.identity_graph import IdentityGraph
from .difficulty import DifficultyTier
from .ground_truth import GroundTruthRegistry, PairwiseLabel


@dataclass
class BenchmarkEvaluationResult:
    dataset_name: str
    total_observations: int
    total_possible_pairs: int
    candidate_pairs_count: int
    candidate_reduction_pct: float
    candidate_recall_pct: float
    elapsed_total_seconds: float
    elapsed_candidate_gen_seconds: float
    elapsed_fusion_seconds: float
    elapsed_graph_seconds: float
    threshold_used: float
    tp: int
    fp: int
    tn: int
    fn: int
    ambiguous_count: int
    precision: float
    recall: float
    f1_score: float
    false_merge_rate: float
    false_split_rate: float
    ambiguity_rate: float
    hard_negatives_evaluated: int
    hard_negative_false_merges: int
    hard_negative_safe_rate: float
    tier_breakdown: Dict[str, Dict[str, Any]]
    cluster_count: int
    cluster_purity: float
    reid_metrics: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiCameraBenchmarkEvaluator:
    """
    Evaluator executing the canonical production reasoning path against independent ground truth.
    """

    def __init__(
        self,
        registry: GroundTruthRegistry,
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        threshold: float = 0.75,
        max_time_window_seconds: float = 1800.0,
        max_speed_kmh: float = 120.0,
    ) -> None:
        self.registry = registry
        self.camera_metadata = camera_metadata or {}
        self.threshold = threshold
        self.max_time_window_seconds = max_time_window_seconds
        self.max_speed_kmh = max_speed_kmh


    def evaluate_reid_quality(
        self,
        ground_truth_registry: GroundTruthRegistry,
        observations: List[Observation],
    ) -> Dict[str, Any]:
        """
        Rigorously evaluate ReID embedding quality, pairwise matching distribution,
        ROC-AUC, and threshold sensitivity against ground truth.
        """
        from inference.similarity import appearance_similarity

        obs_by_id = {o.observation_id: o for o in observations}

        # 1. Embedding Extraction Quality
        valid_dim_count = 0
        finite_count = 0
        norms = []
        for o in observations:
            emb = o.appearance_embedding
            if emb and isinstance(emb, (list, tuple)) and len(emb) == 512:
                valid_dim_count += 1
                if not any(math.isnan(x) or math.isinf(x) for x in emb):
                    finite_count += 1
                    n_val = math.sqrt(sum(x * x for x in emb))
                    norms.append(n_val)

        mean_norm = (sum(norms) / len(norms)) if norms else 0.0

        # 2. Pairwise ReID Matching Distribution
        same_scores = []
        hard_neg_scores = []
        diff_scores = []

        for pair in ground_truth_registry.pairwise_labels.values():
            oa = obs_by_id.get(pair.obs_a_id)
            ob = obs_by_id.get(pair.obs_b_id)
            if not oa or not ob or not oa.appearance_embedding or not ob.appearance_embedding:
                continue
            sim = appearance_similarity(oa.appearance_embedding, ob.appearance_embedding)
            if sim is None:
                continue
            if pair.relationship == "SAME_VEHICLE":
                same_scores.append(sim)
            elif pair.relationship == "HARD_NEGATIVE":
                hard_neg_scores.append(sim)
            else:
                diff_scores.append(sim)

        # 3. ROC-AUC & Threshold Sweep
        all_pos = same_scores
        all_neg = diff_scores + hard_neg_scores

        roc_auc = 0.0
        best_f1 = 0.0
        best_tau = 0.70
        sweep_data = []

        if all_pos and all_neg:
            thresholds = [round(i * 0.05, 2) for i in range(21)]
            tprs = []
            fprs = []
            for t in thresholds:
                tp_c = sum(1 for s in all_pos if s >= t)
                fp_c = sum(1 for s in all_neg if s >= t)
                fn_c = len(all_pos) - tp_c
                tn_c = len(all_neg) - fp_c
                tpr = tp_c / len(all_pos)
                fpr = fp_c / len(all_neg)
                tprs.append(tpr)
                fprs.append(fpr)

                prec = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
                rec = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
                f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                fmr = fp_c / len(all_neg)
                fnmr = fn_c / len(all_pos)

                if f1 > best_f1:
                    best_f1 = f1
                    best_tau = t

                sweep_data.append({
                    "threshold": t,
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "f1_score": round(f1, 4),
                    "false_match_rate": round(fmr, 4),
                    "false_non_match_rate": round(fnmr, 4),
                })

            for i in range(len(fprs) - 1):
                roc_auc += abs(fprs[i] - fprs[i + 1]) * (tprs[i] + tprs[i + 1]) / 2.0

        return {
            "embedding_extraction_quality": {
                "total_observations": len(observations),
                "valid_512d_count": valid_dim_count,
                "finite_count": finite_count,
                "mean_l2_norm": round(mean_norm, 4),
                "unit_hypersphere_verified": abs(mean_norm - 1.0) < 0.01,
            },
            "pairwise_reid_matching": {
                "roc_auc": round(roc_auc, 4),
                "optimal_reid_threshold": best_tau,
                "optimal_reid_f1": round(best_f1, 4),
                "same_vehicle_similarity_mean": round(sum(same_scores) / len(same_scores), 4) if same_scores else None,
                "hard_negative_similarity_mean": round(sum(hard_neg_scores) / len(hard_neg_scores), 4) if hard_neg_scores else None,
                "different_vehicle_similarity_mean": round(sum(diff_scores) / len(diff_scores), 4) if diff_scores else None,
                "threshold_sensitivity_sweep": sweep_data,
            },
            "ground_truth_context": (
                "Controlled Multi-Camera ReID Evaluation ('multicamera_v1'). "
                "Real perception feed ('CAM_001') contains single-camera CCTV data with zero cross-camera ground truth."
            ),
        }

    def evaluate(
        self,
        observations: List[Observation],
        config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkEvaluationResult:
        """
        Execute production CandidateGenerator -> IdentityFusion -> IdentityGraph
        and compute independent empirical validation metrics.
        """
        t_start = time.perf_counter()
        n_obs = len(observations)
        total_possible = (n_obs * (n_obs - 1)) // 2

        cfg = dict(config or {})
        cfg.setdefault("confirmed_threshold", self.threshold)
        cfg.setdefault("ambiguous_threshold", 0.40)

        # 1. Candidate Generation (Production Path)
        t_gen_0 = time.perf_counter()
        generator = CandidateGenerator(
            max_speed_kmh=self.max_speed_kmh,
            max_time_window_seconds=self.max_time_window_seconds,
            camera_metadata=self.camera_metadata,
            config=cfg,
        )
        candidates, _ = generator.generate_candidates(observations)
        t_gen = time.perf_counter() - t_gen_0

        candidate_count = len(candidates)
        candidate_reduction = (
            ((total_possible - candidate_count) / total_possible * 100.0)
            if total_possible > 0
            else 0.0
        )

        candidate_set: Set[Tuple[str, str]] = set(
            (min(a.observation_id, b.observation_id), max(a.observation_id, b.observation_id))
            for a, b in candidates
        )

        # Measure Candidate Recall against all positive ground-truth pairs
        true_same_pairs_in_gt = [
            label for label in self.registry.pairwise_labels.values() if label.is_same_vehicle
        ]
        retained_same_pairs = sum(
            1 for label in true_same_pairs_in_gt if label.pair_key() in candidate_set
        )
        candidate_recall = (
            (retained_same_pairs / len(true_same_pairs_in_gt) * 100.0)
            if true_same_pairs_in_gt
            else 100.0
        )

        # 2. Identity Fusion on Candidates (Production Path)
        t_fuse_0 = time.perf_counter()
        predictions: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for obs_a, obs_b in candidates:
            res = match_observations(
                obs_a,
                obs_b,
                camera_metadata=self.camera_metadata,
                config=cfg,
            )
            key = (min(obs_a.observation_id, obs_b.observation_id), max(obs_a.observation_id, obs_b.observation_id))
            predictions[key] = res
        t_fuse = time.perf_counter() - t_fuse_0

        # 3. IdentityGraph Assembly and Clustering (Production Path)
        t_graph_0 = time.perf_counter()
        graph = IdentityGraph(
            min_probability_threshold=self.threshold,
        )
        sorted_obs_input = sorted(observations, key=lambda o: (o.timestamp_seconds, o.observation_id))
        for obs in sorted_obs_input:
            graph.add_observation(obs)

        for (oa_id, ob_id), match_result in predictions.items():
            prob = float(match_result.get("same_vehicle_score", match_result.get("same_vehicle_probability", 0.0)))
            has_id_ev = match_result.get("evidence", {}).get("identity_evidence_available", True)

            if prob >= self.threshold and has_id_ev:
                edge_data = {
                    "source": oa_id,
                    "target": ob_id,
                    "probability": prob,
                    "evidence": match_result.get("evidence", {}),
                    "evidence_ledger": match_result.get("evidence_ledger"),
                    "explanation": match_result.get("explanation", ""),
                }
                graph.edges.append(edge_data)
                graph.adjacency[oa_id].append((ob_id, prob))
                graph.adjacency[ob_id].append((oa_id, prob))
            else:
                expl = match_result.get("explanation", "")
                if prob < self.threshold:
                    tag = "below_threshold"
                    reason = f"Estimated probability {prob:.4f} is below configured threshold {self.threshold:.2f}. {expl}"
                else:
                    tag = "insufficient_identity_evidence"
                    reason = f"Probability {prob:.4f} meets threshold but lacked positive identity evidence."
                graph._record_rejection(oa_id, ob_id, tag, reason, prob)

        for nid in graph.adjacency:
            graph.adjacency[nid].sort(key=lambda item: (-item[1], item[0]))

        clusters = graph.get_candidate_identities()
        t_graph = time.perf_counter() - t_graph_0

        # Compute Cluster Purity against latent vehicle ground truth
        pure_clusters = 0
        for cluster in clusters:
            latents = set(self.registry.get_latent_id(oid) for oid in cluster.get("observation_ids", []))
            latents.discard(None)
            if len(latents) <= 1:
                pure_clusters += 1
        cluster_purity = (pure_clusters / len(clusters)) if clusters else 1.0

        # 4. Pairwise Evaluation against Independent Pairwise Ground Truth
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        ambiguous_count = 0

        # Per-tier tracking
        tier_stats: Dict[str, Dict[str, int]] = {
            tier.value: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "ambiguous": 0, "total": 0}
            for tier in DifficultyTier
        }

        hard_negatives_evaluated = 0
        hard_negative_false_merges = 0

        for key, label in self.registry.pairwise_labels.items():
            tier = label.difficulty_tier
            if tier not in tier_stats:
                tier_stats[tier] = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "ambiguous": 0, "total": 0}
            tier_stats[tier]["total"] += 1

            pred = predictions.get(key)
            if pred is None:
                # Pruned by candidate generator or un-evaluated:
                # If it was pruned, predicted score is 0.0 (REJECTED)
                pred_decision = "REJECTED"
                pred_score = 0.0
            else:
                pred_decision = pred.get("decision_state", "REJECTED")
                pred_score = float(pred.get("same_vehicle_score", pred.get("same_vehicle_probability", 0.0)))

            if label.is_hard_negative:
                hard_negatives_evaluated += 1
                if pred_decision == "CONFIRMED":
                    hard_negative_false_merges += 1

            if label.is_same_vehicle:
                if pred_decision == "CONFIRMED":
                    tp += 1
                    tier_stats[tier]["tp"] += 1
                elif pred_decision == "AMBIGUOUS":
                    ambiguous_count += 1
                    tier_stats[tier]["ambiguous"] += 1
                    # In conservative evaluation, AMBIGUOUS is treated as non-merge (safe non-false-merge)
                    fn += 1
                    tier_stats[tier]["fn"] += 1
                else:
                    fn += 1
                    tier_stats[tier]["fn"] += 1
            else:
                if pred_decision == "CONFIRMED":
                    fp += 1
                    tier_stats[tier]["fp"] += 1
                elif pred_decision == "AMBIGUOUS":
                    ambiguous_count += 1
                    tier_stats[tier]["ambiguous"] += 1
                    tn += 1
                    tier_stats[tier]["tn"] += 1
                else:
                    tn += 1
                    tier_stats[tier]["tn"] += 1

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fmr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fsr = (fn / (tp + fn)) if (tp + fn) > 0 else 0.0
        total_eval = len(self.registry.pairwise_labels)
        ambiguity_rate = (ambiguous_count / total_eval) if total_eval > 0 else 0.0

        hard_neg_safe_rate = (
            ((hard_negatives_evaluated - hard_negative_false_merges) / hard_negatives_evaluated * 100.0)
            if hard_negatives_evaluated > 0
            else 100.0
        )

        # Compute tier metrics
        tier_breakdown: Dict[str, Dict[str, Any]] = {}
        for t_name, s in tier_stats.items():
            t_tp = s["tp"]
            t_fp = s["fp"]
            t_tn = s["tn"]
            t_fn = s["fn"]
            t_prec = (t_tp / (t_tp + t_fp)) if (t_tp + t_fp) > 0 else 0.0
            t_rec = (t_tp / (t_tp + t_fn)) if (t_tp + t_fn) > 0 else 0.0
            t_f1 = (2 * t_prec * t_rec / (t_prec + t_rec)) if (t_prec + t_rec) > 0 else 0.0
            t_fmr = (t_fp / (t_fp + t_tn)) if (t_fp + t_tn) > 0 else 0.0
            tier_breakdown[t_name] = {
                "total_pairs": s["total"],
                "tp": t_tp,
                "fp": t_fp,
                "tn": t_tn,
                "fn": t_fn,
                "ambiguous": s["ambiguous"],
                "precision": round(t_prec, 4),
                "recall": round(t_rec, 4),
                "f1_score": round(t_f1, 4),
                "false_merge_rate": round(t_fmr, 4),
            }

        t_total = time.perf_counter() - t_start
        reid_results = self.evaluate_reid_quality(self.registry, observations)

        return BenchmarkEvaluationResult(
            dataset_name="multicamera_v1",
            total_observations=n_obs,
            total_possible_pairs=total_possible,
            candidate_pairs_count=candidate_count,
            candidate_reduction_pct=round(candidate_reduction, 2),
            candidate_recall_pct=round(candidate_recall, 2),
            elapsed_total_seconds=round(t_total, 3),
            elapsed_candidate_gen_seconds=round(t_gen, 3),
            elapsed_fusion_seconds=round(t_fuse, 3),
            elapsed_graph_seconds=round(t_graph, 3),
            threshold_used=self.threshold,
            tp=tp,
            fp=fp,
            tn=tn,
            fn=fn,
            ambiguous_count=ambiguous_count,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
            false_merge_rate=round(fmr, 4),
            false_split_rate=round(fsr, 4),
            ambiguity_rate=round(ambiguity_rate, 4),
            hard_negatives_evaluated=hard_negatives_evaluated,
            hard_negative_false_merges=hard_negative_false_merges,
            hard_negative_safe_rate=round(hard_neg_safe_rate, 2),
            tier_breakdown=tier_breakdown,
            cluster_count=len(clusters),
            cluster_purity=round(cluster_purity, 4),
            reid_metrics=reid_results,
        )
