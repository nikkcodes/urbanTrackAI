"""Shared pytest fixtures for the UrbanTrack validation suite."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from perception import config

PREFERRED_CAMERA_ID = "CAM_S01_C001"
SYNTHETIC_OUTPUT_DIR = Path(config.SYNTHETIC_OUTPUT_DIR)
SYNTHETIC_SEED = 42


def get_camera_output_dir(base_output_dir: Path | str = "data/output") -> Path:
    """Find the first camera directory inside data/output/, preferring CAM_S01_C001."""
    base = Path(base_output_dir)
    preferred = base / PREFERRED_CAMERA_ID
    if preferred.is_dir():
        return preferred
    if base.is_dir():
        for item in sorted(base.iterdir()):
            if item.is_dir() and item.name.startswith("CAM_"):
                return item
    return preferred


def get_camera_id() -> str:
    """Return the camera ID of the resolved camera output directory."""
    return get_camera_output_dir().name


def get_synthetic_output_dir() -> Path:
    """Return the synthetic output directory for the resolved camera ID."""
    return SYNTHETIC_OUTPUT_DIR / get_camera_id()


@pytest.fixture(scope="session")
def camera_output_dir() -> Path:
    """Session fixture providing the resolved camera output directory."""
    return get_camera_output_dir()


def _run_generator(seed: int, camera_id: str | None = None) -> None:
    cid = camera_id or get_camera_id()
    cmd = [sys.executable, "-m", "perception.synthetic_degradation", "--seed", str(seed)]
    if cid:
        cmd.extend(["--camera-id", cid])
    subprocess.run(
        cmd,
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session", autouse=True)
def _ensure_synthetic_outputs():
    """Generate deterministic synthetic outputs once per test session.

    Real perception outputs in data/output/ are never modified. The
    synthetic directory is always regenerated with the fixed seed so the
    schema tests start from a known state and are immune to stale artefacts
    left by prior manual runs.
    """
    shutil.rmtree(get_synthetic_output_dir(), ignore_errors=True)
    _run_generator(SYNTHETIC_SEED)
    yield
    # Leave generated artefacts in place for inspection; do not delete.