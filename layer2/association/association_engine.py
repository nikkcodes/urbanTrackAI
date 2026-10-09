"""
UrbanTrack AI — Layer 2 Production Association Engine.

Orchestrates streaming evidence fusion and decision policy evaluation across
all candidate pairs:
- Ingests canonical tracklets and streams frozen candidate pairs.
- Evaluates multimodal evidence (appearance, temporal, motion, type, OCR, topology, quality).
- Enforces strict Re-ID model compatibility.
- Applies decision policy producing CONFIRMED, AMBIGUOUS, and REJECTED outcomes.
- Emits comprehensive audit ledgers and benchmarks against official ground truth.
"""

from __future__ import annotations

import json
import os
import resource
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from layer2.association.decision_policy import DecisionPolicy, PolicyThresholds
from layer2.association.evidence_fusion import EvidenceFusionScorer
from layer2.association.evidence_ledger import (
    AssociationDecision,
    CandidateEvidenceLedger,
)
from layer2.association.validation import AssociationValidator
from layer2.ingestion.canonical_models import (
    ANPRData,
    AppearanceData,
    CanonicalTracklet,
    MotionData,
    QualityData,
    SpatialData,
    TemporalData,
    VehicleData,
)


class AssociationEngine:
    """
    Production-grade multimodal identity association engine.
    """

    def __init__(
        self,
        canonical_tracklets_path: Union[str, Path] = "results/layer2_ingestion/canonical_tracklets.json",
        candidate_pairs_path: Union[str, Path] = "results/layer2_candidates/candidate_pairs.json",
        output_dir: Union[str, Path] = "results/layer2_association",
        gt_path: Optional[Union[str, Path]] = "results/gt_positive_loss_audit.json",
        thresholds: Optional[PolicyThresholds] = None,
        base_weights: Optional[Dict[str, float]] = None,
    ) -> None:
        self.canonical_tracklets_path = Path(canonical_tracklets_path)
        self.candidate_pairs_path = Path(candidate_pairs_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.thresholds = thresholds or PolicyThresholds()
        self.scorer = EvidenceFusionScorer(base_weights=base_weights)
        self.policy = DecisionPolicy(thresholds=self.thresholds)

        self.gt_path = Path(gt_path) if gt_path else None
        self.validator = AssociationValidator(str(self.gt_path)) if self.gt_path and self.gt_path.is_file() else None

        self.tracklets: Dict[str, CanonicalTracklet] = {}

    def load_tracklets(self) -> int:
        """
        Loads canonical tracklets into memory index.
        """
        if not self.canonical_tracklets_path.is_file():
            raise FileNotFoundError(f"Canonical tracklets not found: {self.canonical_tracklets_path}")

        with open(self.canonical_tracklets_path, "r", encoding="utf-8") as f:
            raw_list = json.load(f)

        self.tracklets.clear()
        for d in raw_list:
            trk = CanonicalTracklet(
                scenario_id=d["scenario_id"],
                camera_id=d["camera_id"],
                track_id=d["track_id"],
                global_vehicle_id=d.get("global_vehicle_id"),
                temporal=TemporalData(**d["temporal"]),
                motion=MotionData(**d["motion"]),
                appearance=AppearanceData(**d["appearance"]),
                vehicle=VehicleData(**d["vehicle"]),
                anpr=ANPRData(**d["anpr"]),
                spatial=SpatialData(**d["spatial"]),
                quality=QualityData(**d["quality"]),
            )
            self.tracklets[trk.canonical_id] = trk

        return len(self.tracklets)

    def evaluate_candidate_pair(
        self,
        candidate_pair: Dict[str, Any],
    ) -> CandidateEvidenceLedger:
        """
        Evaluates a single candidate pair dict and returns the complete CandidateEvidenceLedger.
        """
        candidate_id = candidate_pair.get("candidate_pair_id", "")
        scenario_id = candidate_pair.get("scenario_id", "")
        orig_id = candidate_pair["origin_tracklet_id"]
        dest_id = candidate_pair["destination_tracklet_id"]

        orig = self.tracklets[orig_id]
        dest = self.tracklets[dest_id]

        topo_meta = candidate_pair.get("topology", {})
        has_topo_edge = topo_meta.get("has_directed_edge", True)
        topo_rel = topo_meta.get("relationship")
        edge_dist_m = topo_meta.get("edge_distance_m")
        cam_dist_m = candidate_pair.get("spatial", {}).get("camera_distance_m", 30.0)

        score, evidence_dict, active_weights, avail_mods, miss_mods, incomp_mods = self.scorer.fuse_evidence(
            origin=orig,
            dest=dest,
            has_topo_edge=has_topo_edge,
            topo_rel=topo_rel,
            edge_dist_m=edge_dist_m,
            cam_dist_m=cam_dist_m,
        )

        decision, reason_codes = self.policy.evaluate_decision(
            association_score=score,
            evidence=evidence_dict,
            available_modalities=avail_mods,
            missing_modalities=miss_mods,
            incompatible_modalities=incomp_mods,
        )

        return CandidateEvidenceLedger(
            candidate_id=candidate_id,
            scenario_id=scenario_id,
            origin_tracklet=orig_id,
            destination_tracklet=dest_id,
            decision=decision,
            association_score=score,
            evidence=evidence_dict,
            active_weights=active_weights,
            available_modalities=avail_mods,
            missing_modalities=miss_mods,
            incompatible_modalities=incomp_mods,
            reason_codes=reason_codes,
        )

    def run_association_pipeline(
        self,
        progress_interval: int = 500000,
        max_ledger_samples: int = 2000,
    ) -> Dict[str, Any]:
        """
        Streams all candidate pairs through evidence fusion and decision evaluation,
        serializing auditable artifacts.
        """
        t_start = time.time()

        if not self.tracklets:
            self.load_tracklets()

        confirmed_file = self.output_dir / "identity_associations.json"
        rejected_file = self.output_dir / "rejected_associations.json"
        ledger_file = self.output_dir / "association_evidence_ledger.json"
        summary_file = self.output_dir / "association_summary.json"
        validation_file = self.output_dir / "association_validation.json"

        # Counters and metrics
        total_candidates = 0
        decisions_counter = Counter()
        reason_codes_counter = Counter()
        missing_mods_counter = Counter()
        incompatible_mods_counter = Counter()

        # Ground truth tracking
        s01_evaluated_pairs = []
        hard_negatives_sample = []

        # Sample ledger storage
        ledger_samples: List[Dict[str, Any]] = []

        print(f"[AssociationEngine] Streaming candidate pairs from: {self.candidate_pairs_path}")

        with open(confirmed_file, "w", encoding="utf-8") as f_conf, \
             open(rejected_file, "w", encoding="utf-8") as f_rej, \
             open(self.candidate_pairs_path, "r", encoding="utf-8") as f_in:

            # Initialize JSON streaming arrays
            f_conf.write("[\n")
            f_rej.write("[\n")
            first_conf = True
            first_rej = True

            # Skip opening bracket '['
            f_in.readline()

            for line in f_in:
                line_str = line.strip()
                if not line_str or line_str == "]":
                    break
                if line_str.endswith(","):
                    line_str = line_str[:-1]

                candidate_pair = json.loads(line_str)
                total_candidates += 1

                ledger = self.evaluate_candidate_pair(candidate_pair)
                decision = ledger.decision
                decisions_counter[decision] += 1

                for r in ledger.reason_codes:
                    reason_codes_counter[r] += 1
                for m in ledger.missing_modalities:
                    missing_mods_counter[m] += 1
                for inc in ledger.incompatible_modalities:
                    incompatible_mods_counter[inc] += 1

                is_gt_pos = False
                if self.validator and ledger.scenario_id == "S01":
                    s01_evaluated_pairs.append({
                        "origin_tracklet": ledger.origin_tracklet,
                        "destination_tracklet": ledger.destination_tracklet,
                        "decision": decision,
                        "association_score": ledger.association_score,
                        "reasons": ledger.reason_codes,
                    })
                    is_gt_pos = self.validator.is_gt_positive(ledger.origin_tracklet, ledger.destination_tracklet)
                    if is_gt_pos:
                        # Always include GT positives in ledger
                        ledger_samples.append(ledger.to_dict())

                # Collect hard negative sample (negative candidate with high score or similar appearance)
                if not is_gt_pos and decision != AssociationDecision.REJECTED.value:
                    if len(hard_negatives_sample) < 20 and ledger.association_score >= 0.70:
                        hard_negatives_sample.append({
                            "candidate_id": ledger.candidate_id,
                            "origin_tracklet": ledger.origin_tracklet,
                            "destination_tracklet": ledger.destination_tracklet,
                            "decision": decision,
                            "association_score": ledger.association_score,
                            "appearance_cosine": ledger.evidence["appearance"].get("cosine_similarity"),
                            "reason_codes": ledger.reason_codes,
                        })

                # Sample for general evidence ledger
                if not is_gt_pos and len(ledger_samples) < max_ledger_samples:
                    if decision == AssociationDecision.CONFIRMED.value or total_candidates % 1500 == 0:
                        ledger_samples.append(ledger.to_dict())

                # Output streaming
                if decision == AssociationDecision.CONFIRMED.value:
                    entry_json = json.dumps({
                        "candidate_pair_id": ledger.candidate_id,
                        "scenario_id": ledger.scenario_id,
                        "origin_tracklet_id": ledger.origin_tracklet,
                        "destination_tracklet_id": ledger.destination_tracklet,
                        "association_score": ledger.association_score,
                        "decision": decision,
                        "reason_codes": ledger.reason_codes,
                        "appearance_cosine": ledger.evidence["appearance"].get("cosine_similarity"),
                        "temporal_regime": ledger.evidence["temporal"].get("regime"),
                        "ocr_status": ledger.evidence["ocr"].get("status"),
                    })
                    if not first_conf:
                        f_conf.write(",\n")
                    f_conf.write(entry_json)
                    first_conf = False

                elif decision == AssociationDecision.REJECTED.value:
                    entry_json = json.dumps({
                        "candidate_pair_id": ledger.candidate_id,
                        "scenario_id": ledger.scenario_id,
                        "origin_tracklet_id": ledger.origin_tracklet,
                        "destination_tracklet_id": ledger.destination_tracklet,
                        "association_score": ledger.association_score,
                        "decision": decision,
                        "reason_codes": ledger.reason_codes,
                    })
                    if not first_rej:
                        f_rej.write(",\n")
                    f_rej.write(entry_json)
                    first_rej = False

                if total_candidates % progress_interval == 0:
                    elapsed = time.time() - t_start
                    speed = total_candidates / elapsed if elapsed > 0 else 0
                    print(f"  Processed {total_candidates:,} candidates ({speed:.1f} pairs/sec)...")

            # Finalize streaming JSON arrays
            f_conf.write("\n]\n")
            f_rej.write("\n]\n")

        elapsed_time = round(time.time() - t_start, 2)
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        peak_memory_mb = round(rusage.ru_maxrss / (1024 * 1024), 2)  # macOS returns bytes
        # On macOS ru_maxrss is bytes; if < 100MB check if Linux KB:
        if peak_memory_mb < 1.0:
            peak_memory_mb = round(rusage.ru_maxrss / 1024, 2)
        candidates_per_sec = round(total_candidates / elapsed_time, 2) if elapsed_time > 0 else 0.0

        print(f"[AssociationEngine] Completed in {elapsed_time}s ({candidates_per_sec} candidates/sec). Peak memory: {peak_memory_mb} MB")

        # Write evidence ledger artifact
        with open(ledger_file, "w", encoding="utf-8") as f_led:
            json.dump(ledger_samples, f_led, indent=2)

        # Validation against ground truth
        validation_metrics = {}
        if self.validator:
            validation_metrics = self.validator.evaluate_associations(s01_evaluated_pairs)
            with open(validation_file, "w", encoding="utf-8") as f_val:
                json.dump(validation_metrics, f_val, indent=2)

        # Summary artifact
        summary = {
            "status": "COMPLETED",
            "execution_metrics": {
                "total_candidates_entering_fusion": total_candidates,
                "runtime_seconds": elapsed_time,
                "peak_memory_mb": peak_memory_mb,
                "throughput_candidates_per_second": candidates_per_sec,
            },
            "decision_breakdown": {
                "CONFIRMED": decisions_counter[AssociationDecision.CONFIRMED.value],
                "AMBIGUOUS": decisions_counter[AssociationDecision.AMBIGUOUS.value],
                "REJECTED": decisions_counter[AssociationDecision.REJECTED.value],
                "percentages": {
                    "confirmed_pct": round(decisions_counter[AssociationDecision.CONFIRMED.value] / total_candidates * 100.0, 2) if total_candidates else 0,
                    "ambiguous_pct": round(decisions_counter[AssociationDecision.AMBIGUOUS.value] / total_candidates * 100.0, 2) if total_candidates else 0,
                    "rejected_pct": round(decisions_counter[AssociationDecision.REJECTED.value] / total_candidates * 100.0, 2) if total_candidates else 0,
                },
            },
            "reasons_breakdown": dict(reason_codes_counter),
            "missing_modalities": dict(missing_mods_counter),
            "incompatible_modalities": dict(incompatible_mods_counter),
            "policy_thresholds": {
                "threshold_confirmed": self.thresholds.threshold_confirmed,
                "threshold_rejected": self.thresholds.threshold_rejected,
                "min_appearance_for_confirmation": self.thresholds.min_appearance_for_confirmation,
                "min_ocr_for_confirmation": self.thresholds.min_ocr_for_confirmation,
                "dissimilar_appearance_threshold": self.thresholds.dissimilar_appearance_threshold,
                "ocr_contradiction_rejection": self.thresholds.ocr_contradiction_rejection,
            },
            "s01_ground_truth_validation": validation_metrics,
            "hard_negatives_sample": hard_negatives_sample,
        }

        with open(summary_file, "w", encoding="utf-8") as f_sum:
            json.dump(summary, f_sum, indent=2)

        return summary
