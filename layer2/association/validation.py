"""
UrbanTrack AI — Layer 2 Identity Association: Benchmark Validation.

Evaluates association decisions against S01 official ground truth:
- Precision, Recall, F1 score
- Candidate recall vs Association recall
- True Positives, False Positives, False Negatives, True Negatives
- False Merge Rate and Cluster Purity
- Hard-negative sample analysis
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple


def parse_canonical_from_tracklet_str(s: str) -> str:
    """Parses CAM_S01_C001_trk_006 into S01:CAM_S01_C001:6."""
    m = re.match(r"(CAM_S01_C\d{3})_trk_(\d+)", s)
    if m:
        return f"S01:{m.group(1)}:{int(m.group(2))}"
    if s.startswith("S01:"):
        return s
    raise ValueError(f"Cannot parse tracklet string: {s}")


class AssociationValidator:
    """
    Validates association decisions against official ground truth.
    """

    def __init__(
        self,
        gt_path: str = "results/gt_positive_loss_audit.json",
    ) -> None:
        self.gt_path = Path(gt_path)
        self.gt_pairs: List[Dict[str, Any]] = []
        self.gt_pairs_set: Set[Tuple[str, str]] = set()
        self.gt_vehicle_map: Dict[str, int] = {}
        self._load_ground_truth()

    def _load_ground_truth(self) -> None:
        if not self.gt_path.is_file():
            raise FileNotFoundError(f"GT file not found: {self.gt_path}")
        with open(self.gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.gt_pairs = data.get("pairs_audit", [])
        for p in self.gt_pairs:
            id_a = parse_canonical_from_tracklet_str(p["tracklet_a"])
            id_b = parse_canonical_from_tracklet_str(p["tracklet_b"])
            pair_key = tuple(sorted([id_a, id_b]))
            self.gt_pairs_set.add(pair_key)
            self.gt_vehicle_map[id_a] = p["gt_vehicle_id"]
            self.gt_vehicle_map[id_b] = p["gt_vehicle_id"]

    @property
    def total_gt_positives(self) -> int:
        return len(self.gt_pairs_set)

    def is_gt_positive(self, tracklet_a: str, tracklet_b: str) -> bool:
        return tuple(sorted([tracklet_a, tracklet_b])) in self.gt_pairs_set

    def evaluate_associations(
        self,
        evaluated_pairs: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Evaluates a set of candidate association decisions against ground truth.
        """
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        ambiguous_gt = 0
        ambiguous_neg = 0

        gt_decision_breakdown = {"CONFIRMED": 0, "AMBIGUOUS": 0, "REJECTED": 0}
        gt_found_in_candidates = 0

        for item in evaluated_pairs:
            orig = item["origin_tracklet"]
            dest = item["destination_tracklet"]
            decision = item["decision"]
            is_pos = self.is_gt_positive(orig, dest)

            if is_pos:
                gt_found_in_candidates += 1
                gt_decision_breakdown[decision] += 1
                if decision == "CONFIRMED":
                    tp += 1
                elif decision == "AMBIGUOUS":
                    ambiguous_gt += 1
                elif decision == "REJECTED":
                    fn += 1
            else:
                if decision == "CONFIRMED":
                    fp += 1
                elif decision == "AMBIGUOUS":
                    ambiguous_neg += 1
                elif decision == "REJECTED":
                    tn += 1

        total_gt = len(self.gt_pairs_set)
        # Pairs missed before association (if any)
        gt_missed_before_fusion = total_gt - gt_found_in_candidates

        # Candidate recall = fraction of GT pairs that reached fusion
        candidate_recall = (gt_found_in_candidates / total_gt * 100.0) if total_gt else 0.0

        # Strict identity precision and recall (CONFIRMED only)
        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        confirmed_recall = (tp / total_gt * 100.0) if total_gt else 0.0
        # Permissive recall (CONFIRMED + AMBIGUOUS preserved for downstream)
        permissive_recall = ((tp + ambiguous_gt) / total_gt * 100.0) if total_gt else 0.0

        f1 = (2 * precision * confirmed_recall / (precision + confirmed_recall)) if (precision + confirmed_recall) > 0 else 0.0
        false_merge_rate = (fp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0

        return {
            "dataset": "AI City Challenge 2022 Track 1 / CityFlowV2 S01",
            "total_gt_positives": total_gt,
            "gt_positives_reaching_fusion": gt_found_in_candidates,
            "candidate_recall_percentage": round(candidate_recall, 2),
            "gt_decision_breakdown": gt_decision_breakdown,
            "confusion_matrix": {
                "TP_confirmed_gt": tp,
                "FP_false_merge": fp,
                "FN_rejected_gt": fn,
                "TN_rejected_neg": tn,
                "ambiguous_gt_preserved": ambiguous_gt,
                "ambiguous_neg_preserved": ambiguous_neg,
            },
            "metrics": {
                "precision_percentage": round(precision, 2),
                "confirmed_recall_percentage": round(confirmed_recall, 2),
                "permissive_recall_percentage": round(permissive_recall, 2),
                "f1_score": round(f1 / 100.0, 4),
                "false_merge_rate_percentage": round(false_merge_rate, 2),
                "cluster_purity_percentage": round(100.0 - false_merge_rate, 2),
            },
        }
