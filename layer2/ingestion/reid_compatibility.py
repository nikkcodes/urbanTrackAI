"""
UrbanTrack AI — Re-ID Compatibility Contract & Verification.

Implements strict validation of Re-ID feature vector spaces, preventing invalid
cross-dataset cosine similarity comparisons between AICity vehicle and MSMT17 person spaces.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from layer2.ingestion.canonical_models import CanonicalTracklet

AICITY_GROUP = "AICITY_VEHICLE_V1"
MSMT17_GROUP = "MSMT17_PERSON_BASELINE"
NONE_GROUP = "NONE"


def map_reid_compatibility_group(model_tag: Optional[str], has_embedding: bool) -> str:
    """
    Maps a Layer 1 Re-ID model architecture tag to a canonical compatibility domain.
    """
    if not has_embedding or not model_tag:
        return NONE_GROUP

    tag_clean = model_tag.strip().lower()
    if "aicity" in tag_clean:
        return AICITY_GROUP
    elif "msmt17" in tag_clean:
        return MSMT17_GROUP
    else:
        return NONE_GROUP


def are_reid_compatible(tracklet_a: CanonicalTracklet, tracklet_b: CanonicalTracklet) -> bool:
    """
    Evaluates whether two tracklets share a semantically and mathematically compatible
    Re-ID embedding space.

    Rules:
    - AICITY_VEHICLE_V1 <-> AICITY_VEHICLE_V1: Compatible (True)
    - MSMT17_PERSON_BASELINE <-> AICITY_VEHICLE_V1: Strictly Incompatible (False)
    - MSMT17_PERSON_BASELINE <-> MSMT17_PERSON_BASELINE: Structurally compatible space,
      but person-reid weights cannot confirm vehicle identity.
    - If either lacks a valid embedding: Incompatible (False)
    """
    grp_a = tracklet_a.appearance.reid_compatibility_group
    grp_b = tracklet_b.appearance.reid_compatibility_group

    if grp_a == NONE_GROUP or grp_b == NONE_GROUP:
        return False

    if not tracklet_a.appearance.has_embedding or not tracklet_b.appearance.has_embedding:
        return False

    if grp_a == AICITY_GROUP and grp_b == AICITY_GROUP:
        return True

    if grp_a == MSMT17_GROUP and grp_b == MSMT17_GROUP:
        # Same embedding space (dimension and weights), but person-reid model
        return True

    return False


def get_reid_compatibility_status(
    tracklet_a: CanonicalTracklet,
    tracklet_b: CanonicalTracklet,
) -> Dict[str, Any]:
    """
    Produces a detailed diagnostic status record for Re-ID compatibility between two tracklets.
    No cosine similarity is computed in this ingestion layer.
    """
    grp_a = tracklet_a.appearance.reid_compatibility_group
    grp_b = tracklet_b.appearance.reid_compatibility_group
    has_emb_a = tracklet_a.appearance.has_embedding
    has_emb_b = tracklet_b.appearance.has_embedding

    if not has_emb_a or not has_emb_b:
        missing = []
        if not has_emb_a:
            missing.append(tracklet_a.canonical_id)
        if not has_emb_b:
            missing.append(tracklet_b.canonical_id)
        return {
            "status": "MISSING_EMBEDDING",
            "is_compatible": False,
            "allows_cosine_similarity": False,
            "group_a": grp_a,
            "group_b": grp_b,
            "reason": f"Tracklet(s) lack valid appearance embedding: {', '.join(missing)}",
        }

    if grp_a == AICITY_GROUP and grp_b == AICITY_GROUP:
        return {
            "status": "COMPATIBLE",
            "is_compatible": True,
            "allows_cosine_similarity": True,
            "group_a": grp_a,
            "group_b": grp_b,
            "reason": "Both tracklets use fine-tuned AICity vehicle OSNet model (512-D).",
        }

    if (grp_a == MSMT17_GROUP and grp_b == AICITY_GROUP) or (grp_a == AICITY_GROUP and grp_b == MSMT17_GROUP):
        return {
            "status": "INCOMPATIBLE",
            "is_compatible": False,
            "allows_cosine_similarity": False,
            "group_a": grp_a,
            "group_b": grp_b,
            "reason": (
                "Incompatible feature spaces: One tracklet uses MSMT17 person Re-ID weights "
                "while the other uses AICity vehicle Re-ID weights. Direct cosine similarity prohibited."
            ),
        }

    if grp_a == MSMT17_GROUP and grp_b == MSMT17_GROUP:
        return {
            "status": "STRUCTURALLY_EQUIVALENT_NON_TARGET_DOMAIN",
            "is_compatible": True,
            "allows_cosine_similarity": True,
            "group_a": grp_a,
            "group_b": grp_b,
            "reason": "Both tracklets use MSMT17 weights. Mathematical space matches, but model is person-trained.",
        }

    return {
        "status": "UNKNOWN_INCOMPATIBLE",
        "is_compatible": False,
        "allows_cosine_similarity": False,
        "group_a": grp_a,
        "group_b": grp_b,
        "reason": f"Unrecognized model compatibility pairing: {grp_a} vs {grp_b}",
    }
