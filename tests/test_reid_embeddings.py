"""Semantic validation tests for exported trajectories.json and Re-ID embeddings."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from perception.config import REID_EMBEDDING_DIM, REID_MODEL_NAME, REID_MODEL_WEIGHTS

TRAJECTORIES_PATH = Path("data/output/trajectories.json")
EXPECTED_REID_MODEL = f"{REID_MODEL_NAME}_{REID_MODEL_WEIGHTS}"


def _load_trajectories() -> list[dict]:
    if not TRAJECTORIES_PATH.is_file():
        pytest.fail(f"Missing required artefact: {TRAJECTORIES_PATH}")
    with TRAJECTORIES_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)
    assert isinstance(data, list), "trajectories.json must be a JSON list"
    return data


def _records() -> list[dict]:
    records = _load_trajectories()
    assert records, "trajectories.json must contain at least one track"
    return records


def test_trajectories_file_exists():
    assert TRAJECTORIES_PATH.is_file(), f"Expected {TRAJECTORIES_PATH}"


def test_track_ids_are_unique():
    track_ids = [record["track_id"] for record in _records()]
    assert len(track_ids) == len(set(track_ids)), "duplicate track_id in trajectories"


def test_embedding_dimension_matches_config():
    for record in _records():
        embedding = record.get("appearance_embedding")
        if embedding is None:
            continue
        assert len(embedding) == REID_EMBEDDING_DIM, (
            f"track {record['track_id']}: embedding dim {len(embedding)} != {REID_EMBEDDING_DIM}"
        )


def test_embeddings_contain_finite_numeric_values():
    for record in _records():
        embedding = record.get("appearance_embedding")
        if embedding is None:
            continue
        for value in embedding:
            assert isinstance(value, (int, float)), "embedding values must be numeric"
            assert math.isfinite(value), "embedding values must be finite"


def test_embeddings_are_l2_normalized():
    for record in _records():
        embedding = record.get("appearance_embedding")
        if embedding is None:
            continue
        norm = math.sqrt(sum(value * value for value in embedding))
        assert math.isclose(norm, 1.0, abs_tol=1e-3), (
            f"track {record['track_id']}: L2 norm {norm} != 1.0"
        )


def test_missing_embeddings_remain_null():
    for record in _records():
        embedding = record.get("appearance_embedding")
        quality = record.get("embedding_quality")
        if embedding is None:
            assert quality is None, (
                f"track {record['track_id']}: embedding null but quality present"
            )
        else:
            assert quality is not None, (
                f"track {record['track_id']}: embedding present but quality null"
            )


def test_embedding_quality_in_valid_range():
    for record in _records():
        quality = record.get("embedding_quality")
        if quality is None:
            continue
        assert isinstance(quality, (int, float))
        assert 0.0 <= quality <= 1.0, f"embedding_quality out of range: {quality}"


def test_exported_model_name_matches_config():
    for record in _records():
        assert record.get("reid_model") == EXPECTED_REID_MODEL, (
            f"track {record['track_id']}: reid_model {record.get('reid_model')} != {EXPECTED_REID_MODEL}"
        )


def test_trajectory_points_are_valid():
    for record in _records():
        trajectory = record.get("trajectory")
        assert isinstance(trajectory, list), f"track {record['track_id']}: trajectory must be a list"
        for point in trajectory:
            assert isinstance(point, list) and len(point) == 2, (
                f"track {record['track_id']}: invalid trajectory point {point}"
            )
            x, y = point
            assert isinstance(x, int) and isinstance(y, int), (
                f"track {record['track_id']}: trajectory point must be ints"
            )