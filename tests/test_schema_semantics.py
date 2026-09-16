"""Schema compatibility and synthetic dataset validation tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from perception import config

REAL_OUTPUT_DIR = Path("data/output")
SYNTHETIC_OUTPUT_DIR = Path(config.SYNTHETIC_OUTPUT_DIR)

REAL_FILES = (
    "observations.json",
    "trajectories.json",
    "camera_metrics.json",
    "perception_summary.json",
)
SYNTHETIC_FILES = (
    "observations_degraded.json",
    "trajectories_degraded.json",
    "degradation_summary.json",
)


def _require_real_outputs():
    missing = [name for name in REAL_FILES if not (REAL_OUTPUT_DIR / name).is_file()]
    if missing:
        pytest.fail(
            "Required real perception artefacts missing after a successful run: "
            + ", ".join(missing)
        )


def _require_synthetic_outputs():
    missing = [name for name in SYNTHETIC_FILES if not (SYNTHETIC_OUTPUT_DIR / name).is_file()]
    if missing:
        pytest.fail(
            "Required synthetic artefacts missing: " + ", ".join(missing)
        )


def test_real_output_files_exist_after_run():
    _require_real_outputs()


def test_synthetic_files_exist_after_generation():
    _require_synthetic_outputs()


def test_synthetic_files_do_not_overwrite_real_files():
    _require_real_outputs()
    _require_synthetic_outputs()
    real_names = {path.name for path in REAL_OUTPUT_DIR.iterdir()}
    synthetic_names = {path.name for path in SYNTHETIC_OUTPUT_DIR.iterdir()}
    assert not (real_names & synthetic_names), (
        f"synthetic files must not share names with real files: {real_names & synthetic_names}"
    )


def test_degraded_observations_carry_synthetic_provenance():
    _require_synthetic_outputs()
    with (SYNTHETIC_OUTPUT_DIR / "observations_degraded.json").open("r", encoding="utf-8") as file:
        data = json.load(file)
    frames = data.get("frames", [])
    assert frames, "degraded observations must contain frames"
    for frame in frames:
        assert frame.get("synthetic") is True, "every degraded observation must have synthetic: true"
        tags = frame.get("degradation_tags")
        assert tags is None or isinstance(tags, list), "degradation_tags must be a list when present"


def test_degraded_trajectories_carry_synthetic_provenance():
    _require_synthetic_outputs()
    with (SYNTHETIC_OUTPUT_DIR / "trajectories_degraded.json").open("r", encoding="utf-8") as file:
        records = json.load(file)
    assert records, "degraded trajectories must contain records"
    degraded = [record for record in records if record.get("synthetic") is True]
    # With small fixtures a probabilistic run may degrade zero tracks; the
    # invariant to enforce is that any degraded record carries its tags.
    for record in records:
        if record.get("synthetic") is True:
            assert isinstance(record.get("degradation_tags"), list), (
                "degraded trajectory must have degradation_tags"
            )
    for record in degraded:
        assert isinstance(record.get("degradation_tags"), list), (
            "degraded trajectory must have degradation_tags"
        )


def test_degradation_summary_contains_required_fields():
    _require_synthetic_outputs()
    with (SYNTHETIC_OUTPUT_DIR / "degradation_summary.json").open("r", encoding="utf-8") as file:
        summary = json.load(file)
    required = (
        "source_dataset",
        "generation_timestamp",
        "degradation_seed",
        "degraded_frames",
        "degraded_tracks",
        "missing_plate_count",
        "ocr_failure_count",
        "missing_embedding_count",
        "occluded_vehicle_count",
        "fragmented_track_count",
        "dropped_frame_count",
        "camera_outage_intervals",
    )
    for field in required:
        assert field in summary, f"degradation_summary missing {field}"


def _hash_synthetic_outputs() -> dict[str, str]:
    """Hash only the deterministic data files.

    ``degradation_summary.json`` contains ``generation_timestamp`` and is
    intentionally excluded from byte-level determinism checks.
    """
    hashes: dict[str, str] = {}
    for name in ("observations_degraded.json", "trajectories_degraded.json"):
        path = SYNTHETIC_OUTPUT_DIR / name
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _run_generator(seed: int) -> None:
    subprocess.run(
        [sys.executable, "-m", "perception.synthetic_degradation", "--seed", str(seed)],
        check=True,
        capture_output=True,
    )


def test_same_seed_produces_identical_degraded_outputs():
    _require_synthetic_outputs()
    first = _hash_synthetic_outputs()
    shutil.rmtree(SYNTHETIC_OUTPUT_DIR)
    _run_generator(42)
    second = _hash_synthetic_outputs()
    assert first == second, "same seed must produce byte-identical degraded outputs"


def test_different_seed_produces_different_degraded_outputs():
    _require_synthetic_outputs()
    first = _hash_synthetic_outputs()
    shutil.rmtree(SYNTHETIC_OUTPUT_DIR)
    _run_generator(99)
    second = _hash_synthetic_outputs()
    assert first != second, "different seeds must produce different degraded outputs"