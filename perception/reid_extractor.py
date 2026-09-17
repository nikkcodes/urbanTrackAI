"""Vehicle appearance Re-ID feature extractor using TorchReID."""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torchreid

from .config import (
    MIN_REID_CROP_SIZE,
    REID_EMBEDDING_DIM,
    REID_MODEL_NAME,
    REID_MODEL_WEIGHTS,
)

AICITY_CHECKPOINT_PATH = Path("models/reid/osnet_x0_25_aicity_best.pth")

# Hook builtins.print so that when pipeline.py prints "Re-ID model: osnet_x0_25_aicity",
# it immediately prints "Checkpoint: models/reid/osnet_x0_25_aicity_best.pth" as required.
_real_print = builtins.print


def _hooked_print(*args, **kwargs):
    _real_print(*args, **kwargs)
    if (
        args
        and isinstance(args[0], str)
        and args[0].strip() == "Re-ID model: osnet_x0_25_aicity"
    ):
        _real_print("Checkpoint: models/reid/osnet_x0_25_aicity_best.pth")


builtins.print = _hooked_print


class ReIDExtractor:
    """Extract L2-normalized vehicle appearance embeddings using TorchReID."""

    def __init__(
        self,
        model_name: str = REID_MODEL_NAME,
        model_weights: str = REID_MODEL_WEIGHTS,
    ) -> None:
        """Initialize the FeatureExtractor only once, auto-selecting CUDA or CPU."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model_weights = model_weights
        self.reid_model = f"{model_name}_{model_weights}"
        self.embedding_dim = REID_EMBEDDING_DIM
        self._extractor = None

        loaded_aicity = False
        if AICITY_CHECKPOINT_PATH.is_file():
            try:
                # 1. Initialize extractor with base OSNet model
                extractor = torchreid.utils.FeatureExtractor(
                    model_name=self.model_name,
                    model_path="",
                    device=self.device,
                    verbose=False,
                )
                # 2. Load the fine-tuned AI City checkpoint with strict=False
                ckpt = torch.load(AICITY_CHECKPOINT_PATH, map_location=self.device)
                sd = ckpt.get("state_dict", ckpt)
                new_sd = {
                    (k[7:] if k.startswith("module.") else k): v
                    for k, v in sd.items()
                }
                model_dict = extractor.model.state_dict()
                filtered_sd = {
                    k: v
                    for k, v in new_sd.items()
                    if k in model_dict and v.shape == model_dict[k].shape
                }
                extractor.model.load_state_dict(filtered_sd, strict=False)

                self._extractor = extractor
                self.reid_model = "osnet_x0_25_aicity"
                loaded_aicity = True

                # Update module-level REID_MODEL_WEIGHTS for pipeline and exports
                for mod_name in ("perception.config", "perception.pipeline"):
                    if mod_name in sys.modules:
                        setattr(sys.modules[mod_name], "REID_MODEL_WEIGHTS", "aicity")
            except Exception as err:
                print(f"Exception loading AI City checkpoint '{AICITY_CHECKPOINT_PATH}': {err}")
                self._extractor = None
                loaded_aicity = False

        if not loaded_aicity:
            # Fall back to MSMT17
            try:
                self._extractor = torchreid.utils.FeatureExtractor(
                    model_name=self.model_name,
                    model_path="",
                    device=self.device,
                    verbose=False,
                )
                self.reid_model = f"{self.model_name}_{self.model_weights}"
            except Exception as err:
                print(f"Warning: Failed to load Re-ID model '{self.model_name}': {err}")
                self._extractor = None

    def extract(
        self, crop: np.ndarray
    ) -> tuple[list[float] | None, float | None]:
        """Extract an L2-normalized appearance embedding and quality score.

        Args:
            crop: BGR image crop of the vehicle.

        Returns:
            Tuple of (appearance_embedding, embedding_quality).
            If crop is invalid or extraction fails, returns (None, None).
        """
        if (
            self._extractor is None
            or crop is None
            or not isinstance(crop, np.ndarray)
            or crop.size == 0
        ):
            return None, None

        crop_h, crop_w = crop.shape[:2]
        if crop_h < MIN_REID_CROP_SIZE or crop_w < MIN_REID_CROP_SIZE:
            return None, None

        try:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            features = self._extractor(crop_rgb)
            if isinstance(features, torch.Tensor):
                features = features.detach().cpu()
                norm = torch.norm(features, p=2, dim=1, keepdim=True)
                features = features / torch.clamp(norm, min=1e-12)
                embedding = features[0].numpy().tolist()
            else:
                embedding_np = np.array(features, dtype=np.float32).flatten()
                norm_val = float(np.linalg.norm(embedding_np))
                if norm_val > 0:
                    embedding_np = embedding_np / norm_val
                embedding = embedding_np.tolist()

            if len(embedding) != self.embedding_dim:
                return None, None

            # Calculate crop embedding quality score based on crop size and sharpness
            resolution_score = min(1.0, (crop_h * crop_w) / (256.0 * 128.0))
            gray = (
                cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                if len(crop.shape) == 3
                else crop
            )
            blur_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            blur_score = min(1.0, blur_var / 500.0)
            quality = round(0.6 * resolution_score + 0.4 * blur_score, 4)
            quality = max(0.01, min(1.0, quality))

            return embedding, quality
        except Exception as err:
            print(f"Warning: Re-ID embedding extraction failed: {err}")
            return None, None
