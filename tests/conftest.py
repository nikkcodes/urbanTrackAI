"""Shared pytest fixtures for the UrbanTrack validation suite."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from perception import config

SYNTHETIC_OUTPUT_DIR = Path(config.SYNTHETIC_OUTPUT_DIR)
SYNTHETIC_SEED = 42


def _run_generator(seed: int) -> None:
    subprocess.run(
        [sys.executable, "-m", "perception.synthetic_degradation", "--seed", str(seed)],
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
    shutil.rmtree(SYNTHETIC_OUTPUT_DIR, ignore_errors=True)
    _run_generator(SYNTHETIC_SEED)
    yield
    # Leave generated artefacts in place for inspection; do not delete.