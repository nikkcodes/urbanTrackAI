"""
UrbanTrack AI — Layer 2 Identity Association: Decision Policy.

Evaluates fused multimodal evidence against configurable thresholds to produce
tri-state identity decisions: CONFIRMED, AMBIGUOUS, or REJECTED.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from layer2.association.evidence_ledger import AssociationDecision


@dataclass
class PolicyThresholds:
    """Configurable decision policy thresholds."""
    threshold_confirmed: float = 0.68
    threshold_rejected: float = 0.40
    min_appearance_for_confirmation: float = 0.45
    min_ocr_for_confirmation: float = 0.85
    dissimilar_appearance_threshold: float = 0.18
    ocr_contradiction_rejection: bool = True


class DecisionPolicy:
    """
    Applies auditable decision policy to fused candidate evidence.
    """

    def __init__(self, thresholds: Optional[PolicyThresholds] = None) -> None:
        self.thresholds = thresholds or PolicyThresholds()

    def evaluate_decision(
        self,
        association_score: float,
        evidence: Dict[str, Any],
        available_modalities: List[str],
        missing_modalities: List[str],
        incompatible_modalities: List[str],
    ) -> Tuple[str, List[str]]:
        """
        Determines tri-state decision and produces machine-readable reason codes.

        Returns:
            (decision_str, reason_codes)
        """
        reasons: List[str] = []
        app_data = evidence.get("appearance", {})
        ocr_data = evidence.get("ocr", {})
        temp_data = evidence.get("temporal", {})
        vtype_data = evidence.get("vehicle_type", {})
        topo_data = evidence.get("topology", {})

        # 1. Negative hard constraint: OCR contradiction
        if self.thresholds.ocr_contradiction_rejection and ocr_data.get("is_contradiction"):
            reasons.append("REJECT_OCR_CONTRADICTION")
            return AssociationDecision.REJECTED.value, reasons

        # 2. Negative hard constraint: Strong visual dissimilarity on compatible models
        # (Exclude sub-second boundary handoffs where extreme lighting/angle differences may occur)
        delta_t = temp_data.get("delta_t_seconds", 0.0)
        cos_sim = app_data.get("cosine_similarity")
        if (
            app_data.get("models_compatible")
            and cos_sim is not None
            and cos_sim < self.thresholds.dissimilar_appearance_threshold
            and delta_t > 5.0
        ):
            reasons.append("REJECT_DISSIMILAR_APPEARANCE")
            return AssociationDecision.REJECTED.value, reasons

        # 3. Overall low association score
        if association_score < self.thresholds.threshold_rejected:
            reasons.append("REJECT_LOW_ASSOCIATION_SCORE")
            if cos_sim is not None and cos_sim < 0.30:
                reasons.append("LOW_APPEARANCE_SIMILARITY")
            if not vtype_data.get("is_exact_match"):
                reasons.append("VEHICLE_TYPE_MISMATCH")
            return AssociationDecision.REJECTED.value, reasons

        # 4. Confirmation Rule A: Strong OCR match with plausible temporal transition
        temp_score = temp_data.get("normalized_score") or 0.0
        if (
            ocr_data.get("is_exact_match")
            and temp_score >= 0.40
        ):
            reasons.append("CONFIRM_EXACT_OCR_MATCH")
            return AssociationDecision.CONFIRMED.value, reasons

        lev_sim = ocr_data.get("levenshtein_similarity")
        if (
            lev_sim is not None
            and lev_sim >= self.thresholds.min_ocr_for_confirmation
            and association_score >= self.thresholds.threshold_confirmed
        ):
            reasons.append("CONFIRM_PARTIAL_OCR_AND_SPATIOTEMPORAL")
            return AssociationDecision.CONFIRMED.value, reasons

        # 5. Confirmation Rule B: High multimodal score with strong compatible appearance
        if (
            "appearance" in available_modalities
            and cos_sim is not None
            and cos_sim >= self.thresholds.min_appearance_for_confirmation
            and association_score >= self.thresholds.threshold_confirmed
        ):
            reasons.append("CONFIRM_STRONG_APPEARANCE_AND_SPATIOTEMPORAL")
            if vtype_data.get("is_exact_match"):
                reasons.append("VEHICLE_TYPE_AGREEMENT")
            if topo_data.get("has_directed_edge"):
                reasons.append("TOPOLOGY_EDGE_SUPPORTED")
            return AssociationDecision.CONFIRMED.value, reasons

        # 6. Otherwise: AMBIGUOUS (Plausible transition lacking conclusive proof)
        decision = AssociationDecision.AMBIGUOUS.value

        if "appearance" in incompatible_modalities:
            reasons.append("AMBIGUOUS_INCOMPATIBLE_REID_SPACES")
        elif "appearance" in missing_modalities:
            reasons.append("AMBIGUOUS_MISSING_APPEARANCE_EMBEDDING")
        elif cos_sim is not None and cos_sim < self.thresholds.min_appearance_for_confirmation:
            reasons.append("AMBIGUOUS_MODERATE_APPEARANCE_SIMILARITY")

        if "ocr" in missing_modalities:
            reasons.append("AMBIGUOUS_MISSING_OCR_EVIDENCE")

        if not vtype_data.get("is_exact_match"):
            reasons.append("AMBIGUOUS_VEHICLE_TYPE_NOISE")

        if temp_data.get("is_overlapping"):
            reasons.append("CONCURRENT_INTERSECTION_TRANSIT")
        elif temp_data.get("regime") == "BOUNDARY_HANDOFF":
            reasons.append("ADJACENT_FOV_BOUNDARY_HANDOFF")

        return decision, reasons
