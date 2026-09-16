"""Detector smoke test against a real frame from data/input/traffics.mp4.

Replaces the legacy image-based test that depended on the removed
data/input/traffic.jpg asset. Uses the existing perception pipeline input
video and the production VehicleDetector, so no images or detections are
fabricated.
"""

from __future__ import annotations

import cv2
import pytest

from perception.vehicle_detector import VehicleDetector

INPUT_VIDEO = "data/input/traffics.mp4"
VALID_VEHICLE_TYPES = {"car", "motorcycle", "bus", "truck", "auto"}


def _load_first_frame():
    """Read the first frame of the production input video."""
    capture = cv2.VideoCapture(INPUT_VIDEO)
    if not capture.isOpened():
        pytest.fail(f"Could not open input video: {INPUT_VIDEO}")
    try:
        success, frame = capture.read()
    finally:
        capture.release()
    if not success or frame is None:
        pytest.fail(f"Could not read a frame from {INPUT_VIDEO}")
    return frame


def test_detector_returns_valid_detections_on_real_frame():
    frame = _load_first_frame()
    detector = VehicleDetector()
    vehicles = detector.detect(frame)

    assert vehicles, "detector must return at least one vehicle on a real frame"
    for vehicle in vehicles:
        assert vehicle["vehicle_type"] in VALID_VEHICLE_TYPES, (
            f"unknown vehicle_type: {vehicle['vehicle_type']}"
        )
        assert 0.0 <= vehicle["confidence"] <= 1.0, (
            f"confidence out of range: {vehicle['confidence']}"
        )
        x1, y1, x2, y2 = vehicle["bbox"]
        assert isinstance(x1, int) and isinstance(y1, int)
        assert isinstance(x2, int) and isinstance(y2, int)
        assert x1 < x2 and y1 < y2, f"invalid bbox: {vehicle['bbox']}"
        height, width = frame.shape[:2]
        assert 0 <= x1 < x2 <= width, f"bbox x out of frame: {vehicle['bbox']}"
        assert 0 <= y1 < y2 <= height, f"bbox y out of frame: {vehicle['bbox']}"