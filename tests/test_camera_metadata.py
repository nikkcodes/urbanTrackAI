"""Semantic validation tests for camera metadata (data/config/camera_metadata.json)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from perception.config import CAMERA_METADATA_PATH

CAMERA_METADATA_PATH = Path(CAMERA_METADATA_PATH)
VALID_CAMERA_IDS = {f"CAM_{i:03d}" for i in range(1, 7)}
VALID_METADATA_SOURCES = {"video_metadata", "prototype_configuration"}


def _load_metadata() -> dict:
    if not CAMERA_METADATA_PATH.is_file():
        pytest.fail(f"Missing camera metadata: {CAMERA_METADATA_PATH}")
    with CAMERA_METADATA_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def _cameras() -> list[dict]:
    data = _load_metadata()
    cameras = data.get("cameras")
    assert isinstance(cameras, list), "camera_metadata.json must contain a 'cameras' list"
    assert cameras, "camera_metadata.json must define at least one camera"
    return cameras


def test_camera_metadata_file_exists():
    assert CAMERA_METADATA_PATH.is_file(), f"Expected {CAMERA_METADATA_PATH}"


def test_camera_ids_are_unique():
    camera_ids = [camera["camera_id"] for camera in _cameras()]
    assert len(camera_ids) == len(set(camera_ids)), "duplicate camera_id in metadata"


def test_camera_ids_are_prototype_members():
    for camera in _cameras():
        assert camera["camera_id"] in VALID_CAMERA_IDS, (
            f"unknown camera_id: {camera['camera_id']}"
        )


def test_metadata_source_is_valid():
    for camera in _cameras():
        assert camera.get("metadata_source") in VALID_METADATA_SOURCES, (
            f"{camera['camera_id']}: invalid metadata_source"
        )


def test_calibrated_is_boolean():
    for camera in _cameras():
        assert isinstance(camera.get("calibrated"), bool), (
            f"{camera['camera_id']}: calibrated must be boolean"
        )


def test_video_metadata_only_when_measured():
    for camera in _cameras():
        if camera.get("metadata_source") != "video_metadata":
            assert camera.get("fps") is None, (
                f"{camera['camera_id']}: fps present without video_metadata"
            )
            assert camera.get("resolution_width") is None, (
                f"{camera['camera_id']}: resolution_width present without video_metadata"
            )
            assert camera.get("resolution_height") is None, (
                f"{camera['camera_id']}: resolution_height present without video_metadata"
            )


def test_unknown_metadata_remains_null():
    null_fields = (
        "latitude",
        "longitude",
        "camera_heading",
        "field_of_view_deg",
        "camera_height_m",
        "synchronization_source",
        "synchronization_accuracy_ms",
        "neighboring_cameras",
    )
    for camera in _cameras():
        for field in null_fields:
            assert camera.get(field) is None, (
                f"{camera['camera_id']}: {field} must be null, got {camera.get(field)!r}"
            )


def test_required_fields_present():
    required = (
        "camera_id",
        "camera_name",
        "fps",
        "resolution_width",
        "resolution_height",
        "calibrated",
        "latitude",
        "longitude",
        "camera_heading",
        "field_of_view_deg",
        "camera_height_m",
        "synchronization_source",
        "synchronization_accuracy_ms",
        "neighboring_cameras",
        "metadata_source",
    )
    for camera in _cameras():
        for field in required:
            assert field in camera, f"{camera['camera_id']}: missing field {field}"