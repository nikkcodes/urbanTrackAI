"""Vehicle detection using the Ultralytics YOLOv8 Small model."""

from typing import Any, ClassVar

import numpy as np
import torch
from ultralytics import YOLO

from . import config


class VehicleDetector:
    """Run YOLO inference and return structured vehicle observations."""

    _model: ClassVar[Any | None] = None
    _model_device: ClassVar[str | None] = None
    _vehicle_class_ids: ClassVar[dict[int, str]] = dict(
        zip((2, 3, 5, 7), config.VEHICLE_CLASSES)
    )

    def __init__(self, confidence_threshold: float = config.CONFIDENCE_THRESHOLD) -> None:
        """Initialize the detector.

        Args:
            confidence_threshold: Minimum confidence required for a detection.

        Raises:
            ValueError: If the confidence threshold is outside the range 0 to 1.
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.confidence_threshold = confidence_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.__class__._model is None:
            self.__class__._model = YOLO("yolov8s.pt")
            self.__class__._model.to(self.device)
            self.__class__._model_device = self.device

        print(f"VehicleDetector using device: {self.device}")

    def detect(self, frame: np.ndarray) -> list[dict[str, str | float | int | list[int]]]:
        """Track cars, motorcycles, buses, and trucks in a BGR frame.

        Args:
            frame: An OpenCV BGR image represented as a NumPy array.

        Returns:
            A list of vehicle observations with type, confidence, and bounding box.
        """
        results = self.__class__._model.track(
            frame,
            conf=self.confidence_threshold,
            classes=list(self._vehicle_class_ids),
            device=self.device,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        detections: list[dict[str, str | float | int | list[int]]] = []

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            for track_id, class_id, confidence, box in zip(
                result.boxes.id.tolist(),
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.xyxy.tolist(),
            ):
                vehicle_type = self._vehicle_class_ids.get(int(class_id))
                if vehicle_type is None:
                    continue

                detections.append(
                    {
                        "track_id": int(track_id),
                        "vehicle_type": vehicle_type,
                        "confidence": float(confidence),
                        "bbox": [int(coordinate) for coordinate in box],
                    }
                )

        return detections
