"""YOLO-based license plate detection for vehicle crops."""

from typing import Any, ClassVar
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from . import config


class PlateDetector:
    """Detect license plates without performing OCR."""

    _model: ClassVar[Any | None] = None
    _model_device: ClassVar[str | None] = None

    def __init__(self) -> None:
        """Load the plate model once and select the available device."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.__class__._model is None:
            model_path = Path(config.PLATE_MODEL_PATH)
            if not model_path.is_file():
                print(
                    f"License plate model not found at {config.PLATE_MODEL_PATH}"
                )
                print(
                    "Add the trained license plate YOLO weights at that path "
                    "before starting the perception pipeline."
                )
                raise SystemExit(1)
            self.__class__._model = YOLO(config.PLATE_MODEL_PATH)
            self.__class__._model.to(self.device)
            self.__class__._model_device = self.device

    def detect(self, vehicle_crop: np.ndarray) -> list[dict[str, float | list[int]]]:
        """Return plate bounding boxes and confidences for a vehicle crop."""
        results = self.__class__._model.predict(
            source=vehicle_crop,
            conf=config.PLATE_CONF_THRESHOLD,
            device=self.device,
            verbose=False,
        )
        detections: list[dict[str, float | list[int]]] = []
        for result in results:
            if result.boxes is None:
                continue
            for confidence, box in zip(
                result.boxes.conf.tolist(),
                result.boxes.xyxy.tolist(),
            ):
                detections.append(
                    {
                        "bbox": [int(coordinate) for coordinate in box],
                        "confidence": float(confidence),
                    }
                )
        return detections
