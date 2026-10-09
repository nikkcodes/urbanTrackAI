"""
UrbanTrack AI — Layer 2 Identity Association: Evidence Fusion Scorer.

Extracts, normalizes, and dynamically fuses multimodal evidence across:
- Appearance / Re-ID (with strict latent space compatibility enforcement)
- Corrected synchronized timestamps (overlapping, boundary, and corridor regimes)
- Trajectory / motion heading alignment
- Vehicle type classification (soft evidence, never hard rejection)
- License plate OCR / ANPR (exact match, Levenshtein distance, contradiction)
- Camera graph road topology priors
- Camera reliability and embedding quality modulation
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional, Tuple

from layer2.association.evidence_ledger import (
    AppearanceEvidence,
    CameraQualityEvidence,
    MotionEvidence,
    OCREvidence,
    TemporalEvidence,
    TopologyEvidenceSummary,
    VehicleTypeEvidence,
)
from layer2.ingestion.canonical_models import CanonicalTracklet
from layer2.ingestion.reid_compatibility import are_reid_compatible


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)
    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0 = list(v1)
    return v1[len(s2)]


def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Computes great-circle cosine similarity between two feature vectors."""
    dot_prod = 0.0
    norm1 = 0.0
    norm2 = 0.0
    for a, b in zip(v1, v2):
        dot_prod += a * b
        norm1 += a * a
        norm2 += b * b
    if norm1 <= 0.0 or norm2 <= 0.0:
        return 0.0
    return dot_prod / (math.sqrt(norm1) * math.sqrt(norm2))


def normalize_plate(plate: Optional[str]) -> Optional[str]:
    """Strips non-alphanumeric characters and converts to uppercase."""
    if not plate:
        return None
    cleaned = re.sub(r"[^A-Za-z0-9]", "", plate).upper()
    return cleaned if len(cleaned) >= 2 else None


class EvidenceFusionScorer:
    """
    Multimodal Evidence Fusion engine for cross-camera identity association.
    """

    def __init__(
        self,
        base_weights: Optional[Dict[str, float]] = None,
        appearance_zero_point: float = 0.15,
        appearance_scale_range: float = 0.55,
    ) -> None:
        self.base_weights = base_weights or {
            "appearance": 0.45,
            "ocr": 0.20,
            "temporal": 0.15,
            "vehicle_type": 0.10,
            "topology": 0.05,
            "motion": 0.05,
        }
        self.appearance_zero_point = float(appearance_zero_point)
        self.appearance_scale_range = float(appearance_scale_range)

    def evaluate_appearance(
        self,
        origin: CanonicalTracklet,
        dest: CanonicalTracklet,
    ) -> AppearanceEvidence:
        """
        Evaluates appearance evidence strictly enforcing Re-ID model compatibility.
        """
        has_orig = bool(origin.appearance.has_embedding and origin.appearance.appearance_embedding)
        has_dest = bool(dest.appearance.has_embedding and dest.appearance.appearance_embedding)
        model_orig = origin.appearance.reid_model
        model_dest = dest.appearance.reid_model

        if not has_orig and not has_dest:
            return AppearanceEvidence(
                origin_model=model_orig,
                destination_model=model_dest,
                models_compatible=False,
                origin_has_embedding=False,
                destination_has_embedding=False,
                cosine_similarity=None,
                normalized_score=None,
                status="MISSING_BOTH",
            )
        if not has_orig:
            return AppearanceEvidence(
                origin_model=model_orig,
                destination_model=model_dest,
                models_compatible=False,
                origin_has_embedding=False,
                destination_has_embedding=True,
                cosine_similarity=None,
                normalized_score=None,
                status="MISSING_ORIGIN",
            )
        if not has_dest:
            return AppearanceEvidence(
                origin_model=model_orig,
                destination_model=model_dest,
                models_compatible=False,
                origin_has_embedding=True,
                destination_has_embedding=False,
                cosine_similarity=None,
                normalized_score=None,
                status="MISSING_DESTINATION",
            )

        # Check compatibility
        if not are_reid_compatible(origin, dest):
            return AppearanceEvidence(
                origin_model=model_orig,
                destination_model=model_dest,
                models_compatible=False,
                origin_has_embedding=True,
                destination_has_embedding=True,
                cosine_similarity=None,
                normalized_score=None,
                status="INCOMPATIBLE_SPACES",
            )

        # Compute cosine similarity
        cos_sim = cosine_similarity(
            origin.appearance.appearance_embedding,
            dest.appearance.appearance_embedding,
        )
        norm_score = max(0.0, min(1.0, (cos_sim - self.appearance_zero_point) / self.appearance_scale_range))

        return AppearanceEvidence(
            origin_model=model_orig,
            destination_model=model_dest,
            models_compatible=True,
            origin_has_embedding=True,
            destination_has_embedding=True,
            cosine_similarity=cos_sim,
            normalized_score=norm_score,
            status="COMPATIBLE",
        )

    def evaluate_temporal(
        self,
        origin: CanonicalTracklet,
        dest: CanonicalTracklet,
        chronology_data: Optional[Dict[str, Any]] = None,
        adjacent_dist_m: float = 50.0,
        cam_dist_m: float = 30.0,
    ) -> TemporalEvidence:
        """
        Evaluates temporal consistency across overlapping, boundary, and sequential regimes.
        """
        s_o = origin.temporal.start_sync_seconds
        e_o = origin.temporal.end_sync_seconds
        s_d = dest.temporal.start_sync_seconds
        e_d = dest.temporal.end_sync_seconds

        delta_t = s_d - e_o
        is_overlap = (delta_t <= 0.0)
        overlap_dur = max(0.0, min(e_o, e_d) - s_d) if is_overlap else 0.0

        if is_overlap:
            regime = "OVERLAPPING"
            norm_score = 1.0
        elif cam_dist_m <= adjacent_dist_m and delta_t <= 2.0:
            regime = "BOUNDARY_HANDOFF"
            norm_score = 0.95
        else:
            regime = "SEQUENTIAL_TRANSIT"
            # Linear decay from 1.0 down to 0.20 over 120s horizon
            norm_score = max(0.20, 1.0 - (delta_t / 150.0))

        return TemporalEvidence(
            origin_start_sync=s_o,
            origin_end_sync=e_o,
            destination_start_sync=s_d,
            destination_end_sync=e_d,
            delta_t_seconds=delta_t,
            is_overlapping=is_overlap,
            overlap_duration_seconds=overlap_dur,
            regime=regime,
            normalized_score=norm_score,
        )

    def evaluate_vehicle_type(
        self,
        origin: CanonicalTracklet,
        dest: CanonicalTracklet,
    ) -> VehicleTypeEvidence:
        """
        Evaluates vehicle type consistency as soft evidence (never hard elimination).
        """
        t_orig = origin.vehicle.vehicle_type
        t_dest = dest.vehicle.vehicle_type
        is_match = (t_orig == t_dest)
        # Match = 1.0, Mismatch = 0.40 (soft discount reflecting detector noise)
        norm_score = 1.0 if is_match else 0.40

        return VehicleTypeEvidence(
            origin_type=t_orig,
            destination_type=t_dest,
            is_exact_match=is_match,
            normalized_score=norm_score,
        )

    def evaluate_ocr(
        self,
        origin: CanonicalTracklet,
        dest: CanonicalTracklet,
    ) -> OCREvidence:
        """
        Evaluates optional OCR / license plate consistency.
        """
        p_orig = normalize_plate(origin.anpr.aggregated_plate_text) if origin.anpr.has_readable_ocr else None
        p_dest = normalize_plate(dest.anpr.aggregated_plate_text) if dest.anpr.has_readable_ocr else None

        has_orig = (p_orig is not None)
        has_dest = (p_dest is not None)

        if not has_orig and not has_dest:
            return OCREvidence(
                origin_plate=None,
                destination_plate=None,
                origin_has_plate=False,
                destination_has_plate=False,
                is_exact_match=False,
                edit_distance=None,
                levenshtein_similarity=None,
                is_contradiction=False,
                status="MISSING_BOTH",
                normalized_score=None,
            )

        if not has_orig or not has_dest:
            return OCREvidence(
                origin_plate=p_orig,
                destination_plate=p_dest,
                origin_has_plate=has_orig,
                destination_has_plate=has_dest,
                is_exact_match=False,
                edit_distance=None,
                levenshtein_similarity=None,
                is_contradiction=False,
                status="MISSING_ONE_SIDE",
                normalized_score=None,
            )

        # Both plates available
        if p_orig == p_dest:
            return OCREvidence(
                origin_plate=p_orig,
                destination_plate=p_dest,
                origin_has_plate=True,
                destination_has_plate=True,
                is_exact_match=True,
                edit_distance=0,
                levenshtein_similarity=1.0,
                is_contradiction=False,
                status="EXACT_MATCH",
                normalized_score=1.0,
            )

        dist = levenshtein_distance(p_orig, p_dest)
        max_len = max(len(p_orig), len(p_dest))
        sim = max(0.0, 1.0 - (dist / max_len)) if max_len > 0 else 0.0

        if dist <= 1 and max_len >= 5:
            # 1-character typo/OCR noise
            return OCREvidence(
                origin_plate=p_orig,
                destination_plate=p_dest,
                origin_has_plate=True,
                destination_has_plate=True,
                is_exact_match=False,
                edit_distance=dist,
                levenshtein_similarity=sim,
                is_contradiction=False,
                status="PARTIAL_MATCH",
                normalized_score=0.85,
            )

        # Clear contradiction
        return OCREvidence(
            origin_plate=p_orig,
            destination_plate=p_dest,
            origin_has_plate=True,
            destination_has_plate=True,
            is_exact_match=False,
            edit_distance=dist,
            levenshtein_similarity=sim,
            is_contradiction=True,
            status="CONTRADICTION",
            normalized_score=0.0,
        )

    def evaluate_motion(
        self,
        origin: CanonicalTracklet,
        dest: CanonicalTracklet,
        relative_bearing_deg: Optional[float] = None,
    ) -> MotionEvidence:
        """
        Evaluates motion heading consistency.
        """
        dir_o = origin.motion.direction
        dir_d = dest.motion.direction

        if not dir_o or not dir_d:
            return MotionEvidence(
                origin_direction=dir_o,
                destination_direction=dir_d,
                relative_bearing_deg=relative_bearing_deg,
                is_aligned=True,
                normalized_score=0.75,
            )

        if dir_o == dir_d:
            score = 1.0
            aligned = True
        else:
            # Orthogonal turns are common at intersections
            turns = {
                ("north", "east"), ("north", "west"),
                ("south", "east"), ("south", "west"),
                ("east", "north"), ("east", "south"),
                ("west", "north"), ("west", "south"),
            }
            if (dir_o, dir_d) in turns:
                score = 0.70
                aligned = True
            else:
                score = 0.35
                aligned = False

        return MotionEvidence(
            origin_direction=dir_o,
            destination_direction=dir_d,
            relative_bearing_deg=relative_bearing_deg,
            is_aligned=aligned,
            normalized_score=score,
        )

    def evaluate_topology(
        self,
        has_edge: bool,
        relationship: Optional[str] = None,
        edge_dist_m: Optional[float] = None,
    ) -> TopologyEvidenceSummary:
        """
        Evaluates road topology evidence as a positive prior.
        """
        if has_edge:
            score = 1.0
        else:
            score = 0.50  # Neutral prior, never zero

        return TopologyEvidenceSummary(
            has_directed_edge=has_edge,
            relationship=relationship,
            edge_distance_m=edge_dist_m,
            normalized_score=score,
        )

    def evaluate_quality(
        self,
        origin: CanonicalTracklet,
        dest: CanonicalTracklet,
    ) -> CameraQualityEvidence:
        """
        Computes camera reliability and embedding quality factor.
        """
        r_o = origin.quality.camera_reliability
        r_d = dest.quality.camera_reliability
        q_o = origin.appearance.embedding_quality or 1.0
        q_d = dest.appearance.embedding_quality or 1.0

        factor = math.sqrt(r_o * r_d * q_o * q_d)

        return CameraQualityEvidence(
            origin_reliability=r_o,
            destination_reliability=r_d,
            origin_embedding_quality=origin.appearance.embedding_quality,
            destination_embedding_quality=dest.appearance.embedding_quality,
            combined_quality_factor=factor,
        )

    def fuse_evidence(
        self,
        origin: CanonicalTracklet,
        dest: CanonicalTracklet,
        has_topo_edge: bool = True,
        topo_rel: Optional[str] = None,
        edge_dist_m: Optional[float] = None,
        cam_dist_m: float = 30.0,
        adjacent_dist_m: float = 50.0,
    ) -> Tuple[float, Dict[str, Any], Dict[str, float], List[str], List[str], List[str]]:
        """
        Fuses all available evidence modalities with dynamic weight redistribution.

        Returns:
            (association_score, evidence_dict, active_weights, available_modalities, missing_modalities, incompatible_modalities)
        """
        app_ev = self.evaluate_appearance(origin, dest)
        temp_ev = self.evaluate_temporal(origin, dest, cam_dist_m=cam_dist_m, adjacent_dist_m=adjacent_dist_m)
        vtype_ev = self.evaluate_vehicle_type(origin, dest)
        ocr_ev = self.evaluate_ocr(origin, dest)
        motion_ev = self.evaluate_motion(origin, dest)
        topo_ev = self.evaluate_topology(has_topo_edge, topo_rel, edge_dist_m)
        qual_ev = self.evaluate_quality(origin, dest)

        evidence_dict = {
            "appearance": app_ev.to_dict(),
            "temporal": temp_ev.to_dict(),
            "vehicle_type": vtype_ev.to_dict(),
            "ocr": ocr_ev.to_dict(),
            "motion": motion_ev.to_dict(),
            "topology": topo_ev.to_dict(),
            "camera_quality": qual_ev.to_dict(),
        }

        available_mods: List[str] = ["temporal", "vehicle_type", "topology", "motion", "camera_quality"]
        missing_mods: List[str] = []
        incompatible_mods: List[str] = []

        # Check appearance
        active_weights = dict(self.base_weights)
        if app_ev.status == "COMPATIBLE":
            available_mods.append("appearance")
        elif app_ev.status == "INCOMPATIBLE_SPACES":
            incompatible_mods.append("appearance")
            active_weights["appearance"] = 0.0
        else:
            missing_mods.append("appearance")
            active_weights["appearance"] = 0.0

        # Check OCR
        if ocr_ev.status in ("EXACT_MATCH", "PARTIAL_MATCH", "CONTRADICTION"):
            available_mods.append("ocr")
        else:
            missing_mods.append("ocr")
            active_weights["ocr"] = 0.0

        # Dynamic weight redistribution to sum to 1.0
        total_active_w = sum(active_weights.values())
        if total_active_w > 0:
            active_weights = {k: v / total_active_w for k, v in active_weights.items() if v > 0.0}

        # Compute weighted sum
        score = 0.0
        if active_weights.get("appearance", 0) > 0 and app_ev.normalized_score is not None:
            # Modulate appearance contribution by camera/embedding quality factor
            q_factor = qual_ev.combined_quality_factor
            score += active_weights["appearance"] * (app_ev.normalized_score * q_factor + (1.0 - q_factor) * 0.5)

        if active_weights.get("ocr", 0) > 0 and ocr_ev.normalized_score is not None:
            score += active_weights["ocr"] * ocr_ev.normalized_score

        score += active_weights["temporal"] * temp_ev.normalized_score
        score += active_weights["vehicle_type"] * vtype_ev.normalized_score
        score += active_weights["topology"] * topo_ev.normalized_score
        score += active_weights["motion"] * motion_ev.normalized_score

        return round(score, 4), evidence_dict, active_weights, available_mods, missing_mods, incompatible_mods
