"""Vehicle appearance Re-ID feature extractor using TorchReID."""

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

        try:
            # Initialize TorchReID FeatureExtractor once
            self._extractor = torchreid.utils.FeatureExtractor(
                model_name=self.model_name,
                model_path="",
                device=self.device,
                verbose=False,
            )
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
