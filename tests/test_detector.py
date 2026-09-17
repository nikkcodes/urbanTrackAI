"""Detector smoke test against a real frame from AI City dataset.

Uses the first camera video from data/config/aicity_manifest.json and the
production VehicleDetector, so no images or detections are fabricated.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest

from perception.vehicle_detector import VehicleDetector

MANIFEST_PATH = Path("data/config/aicity_manifest.json")
VALID_VEHICLE_TYPES = {"car", "motorcycle", "bus", "truck", "auto"}


def _get_first_camera_video_path() -> str:
    """Resolve the video path for the first camera in the AI City manifest."""
    if not MANIFEST_PATH.is_file():
        pytest.fail(f"Missing required manifest: {MANIFEST_PATH}")
    with MANIFEST_PATH.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    cameras = manifest.get("cameras", [])
    if not cameras:
        pytest.fail(f"No cameras found in manifest: {MANIFEST_PATH}")
    first_cam = cameras[0]
    video_path = first_cam.get("video_path")
    if video_path and Path(video_path).is_file():
        return video_path
    rel_path = first_cam.get("video_path_relative")
    dataset_root = manifest.get("dataset_root")
    if dataset_root and rel_path:
        combined = Path(dataset_root) / rel_path
        if combined.is_file():
            return str(combined)
    if video_path:
        return video_path
    pytest.fail(f"Could not locate video file for camera {first_cam.get('camera_id')}")


def _load_first_frame():
    """Read the first frame of the first AI City camera video."""
    video_path = _get_first_camera_video_path()
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        pytest.fail(f"Could not open input video: {video_path}")
    try:
        success, frame = capture.read()
    finally:
        capture.release()
    if not success or frame is None:
        pytest.fail(f"Could not read a frame from {video_path}")
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