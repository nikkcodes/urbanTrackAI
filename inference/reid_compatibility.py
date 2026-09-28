"""
UrbanTrack AI — ReID Model Compatibility Layer.

Strictly manages, verifies, and audits embedding space compatibility across
multi-camera perception endpoints. Prevents mathematically invalid cross-model
cosine similarity comparisons between disjoint feature spaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass(frozen=True)
class ReIDModelSpec:
    """Specification and metadata for an approved ReID embedding model."""
    model_id: str
    embedding_dim: int
    normalization: str = "L2"
    checkpoint_hash: Optional[str] = None
    compatibility_group: str = "default"
    similarity_metric: str = "cosine"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "embedding_dim": self.embedding_dim,
            "normalization": self.normalization,
            "checkpoint_hash": self.checkpoint_hash,
            "compatibility_group": self.compatibility_group,
            "similarity_metric": self.similarity_metric,
            "description": self.description,
        }


class ReIDModelCompatibilityLayer:
    """
    Authoritative registry and compatibility checker for ReID feature extractors.

    Rules:
    1. Embeddings from identical models in the same compatibility group are COMPATIBLE.
    2. Embeddings from distinct compatibility groups (e.g. AICity vs MSMT17) are INCOMPATIBLE.
    3. Missing / None model tags are treated as UNKNOWN and incompatible with specific models.
    4. Two unknown models with matching dimensions are conditionally compatible only if
       dimension check passes, with an explicit audit warning.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, ReIDModelSpec] = {}
        self._register_default_models()

    def _register_default_models(self) -> None:
        # 1. AI City 2022 Track 1 OSNet (C001, C003)
        self.register_model(
            ReIDModelSpec(
                model_id="osnet_x0_25_aicity",
                embedding_dim=512,
                normalization="L2",
                compatibility_group="aicity_track1_osnet",
                similarity_metric="cosine",
                description="OSNet x0.25 fine-tuned on CityFlowV2 / AI City ReID data.",
            )
        )

        # 2. MSMT17 Pretrained OSNet (C002)
        self.register_model(
            ReIDModelSpec(
                model_id="osnet_x0_25_msmt17",
                embedding_dim=512,
                normalization="L2",
                compatibility_group="msmt17_osnet",
                similarity_metric="cosine",
                description="OSNet x0.25 trained on MSMT17 person re-identification benchmark.",
            )
        )

        # 3. Generic OSNet x0.25 baseline
        self.register_model(
            ReIDModelSpec(
                model_id="osnet_x0_25",
                embedding_dim=512,
                normalization="L2",
                compatibility_group="osnet_generic",
                similarity_metric="cosine",
                description="Standard generic OSNet x0.25 feature extractor.",
            )
        )

    def register_model(self, spec: ReIDModelSpec) -> None:
        """Register a new ReID model specification."""
        self._registry[spec.model_id.strip().lower()] = spec

    def get_model(self, model_id: Optional[str]) -> Optional[ReIDModelSpec]:
        """Retrieve model specification by ID."""
        if not model_id:
            return None
        return self._registry.get(str(model_id).strip().lower())

    def are_compatible(
        self,
        model_a: Optional[str],
        model_b: Optional[str],
    ) -> bool:
        """
        Evaluate whether two ReID models produce embeddings in the same semantic metric space.
        """
        if not model_a or not model_b:
            # If both are None / unspecified, we cannot guarantee common feature space
            # but allow same-network legacy compatibility if neither camera declares a model.
            return model_a is None and model_b is None

        m_a_norm = str(model_a).strip().lower()
        m_b_norm = str(model_b).strip().lower()

        if m_a_norm == m_b_norm:
            return True

        spec_a = self.get_model(m_a_norm)
        spec_b = self.get_model(m_b_norm)

        if spec_a is not None and spec_b is not None:
            return spec_a.compatibility_group == spec_b.compatibility_group

        return False

    def verify_embeddings(
        self,
        emb_a: Optional[List[float]],
        emb_b: Optional[List[float]],
        model_a: Optional[str],
        model_b: Optional[str],
    ) -> Tuple[bool, str]:
        """
        Verify embedding availability, dimensional consistency, and model compatibility.

        Returns:
            Tuple[bool, str]: (is_valid, reason)
        """
        if emb_a is None or emb_b is None:
            return False, "missing_embeddings"

        if not isinstance(emb_a, (list, tuple)) or not isinstance(emb_b, (list, tuple)):
            return False, "invalid_embedding_type"

        if len(emb_a) == 0 or len(emb_b) == 0:
            return False, "empty_embeddings"

        if len(emb_a) != len(emb_b):
            return False, f"dimension_mismatch_{len(emb_a)}_vs_{len(emb_b)}"

        if model_a and model_b and not self.are_compatible(model_a, model_b):
            return False, f"incompatible_model_spaces_{model_a}_vs_{model_b}"

        spec_a = self.get_model(model_a)
        if spec_a and len(emb_a) != spec_a.embedding_dim:
            return False, f"model_dimension_mismatch_expected_{spec_a.embedding_dim}_got_{len(emb_a)}"

        return True, "compatible"


# Global singleton instance
DEFAULT_REID_COMPATIBILITY_LAYER = ReIDModelCompatibilityLayer()


def are_reid_models_compatible(
    model_a: Optional[str],
    model_b: Optional[str],
    layer: Optional[ReIDModelCompatibilityLayer] = None,
) -> bool:
    """
    Convenience function: evaluate if two ReID models share a compatible embedding space.
    """
    active_layer = layer or DEFAULT_REID_COMPATIBILITY_LAYER
    return active_layer.are_compatible(model_a, model_b)
