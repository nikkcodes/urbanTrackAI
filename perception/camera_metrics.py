"""Camera health metrics for processed video frames."""

from typing import Any

import cv2
import numpy as np


class CameraMetrics:
    """Compute normalized image and detection quality metrics."""

    @staticmethod
    def compute(
        frame: np.ndarray,
        detections: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Return camera metrics and a weighted reliability score."""
        frame_height, frame_width = frame.shape[:2]
        frame_area = max(1, frame_height * frame_width)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = float(np.mean(gray)) / 255.0
        blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        blur_score = min(1.0, blur_variance / 1000.0)

        occupied = np.zeros((frame_height, frame_width), dtype=np.uint8)
        confidence_total = 0.0
        for detection in detections:
            x1, y1, x2, y2 = (
                int(value) for value in detection["bbox"]
            )
            cv2.rectangle(
                occupied,
                (max(0, x1), max(0, y1)),
                (min(frame_width - 1, x2), min(frame_height - 1, y2)),
                1,
                cv2.FILLED,
            )
            confidence_total += float(detection["confidence"])

        occlusion_ratio = min(1.0, float(np.count_nonzero(occupied)) / frame_area)
        vehicle_density = min(1.0, len(detections) / frame_area)
        detection_confidence_mean = (
            confidence_total / len(detections) if detections else 0.0
        )
        reliability = (
            0.35 * brightness
            + 0.30 * blur_score
            + 0.20 * (1.0 - occlusion_ratio)
            + 0.15 * detection_confidence_mean
        )
        return {
            "brightness": round(max(0.0, min(1.0, brightness)), 3),
            "blur_score": round(max(0.0, min(1.0, blur_score)), 3),
            "occlusion_ratio": round(occlusion_ratio, 3),
            "vehicle_density": round(vehicle_density, 6),
            "detection_confidence_mean": round(
                max(0.0, min(1.0, detection_confidence_mean)),
                3,
            ),
            "reliability": round(max(0.0, min(1.0, reliability)), 3),
        }
