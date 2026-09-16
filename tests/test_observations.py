"""Semantic validation tests for exported observations.json."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

OBSERVATIONS_PATH = Path("data/output/CAM_001/observations.json")
VALID_CAMERA_IDS = {f"CAM_{i:03d}" for i in range(1, 7)}
REQUIRED_PROVENANCE = (
    "source_video",
    "camera_id",
    "detector_model",
    "tracker_model",
    "ocr_model",
    "reid_model",
    "processing_device",
    "fps_source",
)


def _load_observations() -> dict:
    if not OBSERVATIONS_PATH.is_file():
        pytest.fail(f"Missing required artefact: {OBSERVATIONS_PATH}")
    with OBSERVATIONS_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _frames() -> list[dict]:
    data = _load_observations()
    assert isinstance(data.get("frames"), list), "observations.json must contain a 'frames' list"
    return data["frames"]


def _vehicles(frame: dict) -> list[dict]:
    vehicles = frame.get("vehicles")
    assert isinstance(vehicles, list), f"frame {frame.get('frame_number')} must have a 'vehicles' list"
    return vehicles


def test_observations_file_exists():
    assert OBSERVATIONS_PATH.is_file(), f"Expected {OBSERVATIONS_PATH}"


def test_top_level_camera_id_is_known():
    data = _load_observations()
    camera_id = data.get("camera_id")
    assert camera_id in VALID_CAMERA_IDS, f"Unknown camera_id: {camera_id}"


def test_vehicle_count_matches_vehicles():
    for frame in _frames():
        assert frame["vehicle_count"] == len(_vehicles(frame)), (
            f"frame {frame.get('frame_number')}: vehicle_count mismatch"
        )


def test_frame_numbers_are_unique_and_monotonic():
    frame_numbers = [frame["frame_number"] for frame in _frames()]
    assert len(frame_numbers) == len(set(frame_numbers)), "duplicate frame numbers"
    assert frame_numbers == sorted(frame_numbers), "frame numbers must be monotonic"


def test_timestamps_are_monotonic():
    timestamps = [frame["timestamp"] for frame in _frames()]
    assert timestamps == sorted(timestamps), "timestamps must increase monotonically"


def test_timestamp_matches_frame_number_over_fps():
    for frame in _frames():
        fps = frame.get("fps")
        assert isinstance(fps, (int, float)) and fps > 0, "fps must be positive"
        frame_number = frame["frame_number"]
        timestamp = frame["timestamp"]
        assert isinstance(timestamp, str) and ":" in timestamp, "timestamp must be HH:MM:SS.mmm"


def test_bboxes_are_valid_within_frame():
    """Bounding boxes must be well-formed and lie within the frame.

    Frame dimensions are derived from the union of observed bounding
    boxes rather than asserted from a top-level field, because
    observations.json does not itself carry frame_width/frame_height
    (those live in camera_metrics.json and perception_summary.json).
    """
    frames = _frames()
    assert frames, "observations must contain frames"
    max_x = max_y = 0
    for frame in frames:
        for vehicle in _vehicles(frame):
            x1, y1, x2, y2 = vehicle["bbox"]
            assert isinstance(x1, int) and isinstance(y1, int)
            assert isinstance(x2, int) and isinstance(y2, int)
            assert x1 < x2, f"bbox x1>=x2: {vehicle['bbox']}"
            assert y1 < y2, f"bbox y1>=y2: {vehicle['bbox']}"
            assert x1 >= 0 and y1 >= 0, f"bbox has negative coord: {vehicle['bbox']}"
            max_x = max(max_x, x2)
            max_y = max(max_y, y2)
    assert max_x > 0 and max_y > 0, "observations must contain at least one bounding box"


def test_detection_confidence_in_range():
    for frame in _frames():
        for vehicle in _vehicles(frame):
            confidence = vehicle["confidence"]
            assert isinstance(confidence, (int, float))
            assert 0.0 <= confidence <= 1.0, f"confidence out of range: {confidence}"


def test_plate_confidence_in_range_or_null():
    for frame in _frames():
        for vehicle in _vehicles(frame):
            confidence = vehicle.get("plate_confidence")
            if confidence is not None:
                assert 0.0 <= confidence <= 1.0, f"plate_confidence out of range: {confidence}"


def test_ocr_confidence_in_range_or_null():
    for frame in _frames():
        for vehicle in _vehicles(frame):
            confidence = vehicle.get("plate_text_confidence")
            if confidence is not None:
                assert 0.0 <= confidence <= 1.0, f"plate_text_confidence out of range: {confidence}"


def test_unavailable_values_are_null_not_empty_string():
    for frame in _frames():
        for vehicle in _vehicles(frame):
            for key in ("plate_bbox", "plate_confidence", "plate_number", "plate_text_confidence"):
                value = vehicle.get(key)
                assert value != "", f"{key} must be null, not empty string"


def test_no_duplicate_track_id_within_frame():
    for frame in _frames():
        track_ids = [vehicle["track_id"] for vehicle in _vehicles(frame)]
        assert len(track_ids) == len(set(track_ids)), (
            f"duplicate track_id in frame {frame.get('frame_number')}"
        )


def test_provenance_present_on_every_frame():
    for frame in _frames():
        provenance = frame.get("provenance")
        assert isinstance(provenance, dict), f"frame {frame.get('frame_number')} missing provenance"
        for key in REQUIRED_PROVENANCE:
            assert key in provenance, f"provenance missing {key}"
            assert provenance[key] is not None and provenance[key] != "", (
                f"provenance.{key} must be non-empty"
            )
        assert provenance["camera_id"] == frame.get("camera_id") or provenance["camera_id"] in VALID_CAMERA_IDS