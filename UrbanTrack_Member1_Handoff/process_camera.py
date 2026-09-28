"""Process a single AI City camera video with the UrbanTrack pipeline.

Wraps ``PerceptionPipeline.process()`` so one CCTV video can be processed
without copying it into ``data/input/`` and without splitting it. Outputs
are written to ``data/output/<camera_id>/`` by the pipeline.

Usage:
    python scripts/process_camera.py \
        --video "C:/.../train/S01/c001/vdo.avi" \
        --camera-id CAM_S01_C001
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow running from the repository root without installation.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from perception.pipeline import PerceptionPipeline

DEFAULT_MANIFEST = "data/config/aicity_manifest.json"


def _lookup_video_path(camera_id: str, manifest_path: Path) -> Path:
    """Return the source video path for ``camera_id`` from the manifest."""
    if not manifest_path.is_file():
        raise SystemExit(
            f"Camera manifest not found: {manifest_path}. "
            "Run scripts/scan_aicity_dataset.py first, or pass --video."
        )
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    for entry in manifest.get("cameras", []):
        if entry.get("camera_id") == camera_id:
            return Path(entry["video_path"])
    raise SystemExit(
        f"Camera ID '{camera_id}' not found in manifest {manifest_path}."
    )


def _lookup_manifest_entry(camera_id: str, manifest_path: Path) -> dict | None:
    """Return the manifest entry for ``camera_id``, or ``None`` if absent."""
    if not manifest_path.is_file():
        return None
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    for entry in manifest.get("cameras", []):
        if entry.get("camera_id") == camera_id:
            return entry
    return None


def _inject_aicity_provenance(
    camera_id: str,
    manifest_path: Path,
    output_dir: Path,
) -> None:
    """Add an ``aicity_dataset`` block to the camera's perception summary.

    Existing summary fields are untouched; only the nested object is added
    (or overwritten if already present). This is provenance metadata only.
    """
    entry = _lookup_manifest_entry(camera_id, manifest_path)
    summary_path = output_dir / "perception_summary.json"
    if not summary_path.is_file():
        return
    with summary_path.open("r", encoding="utf-8") as file:
        summary = json.load(file)
    if not isinstance(summary, dict):
        return
    if entry is None:
        summary["aicity_dataset"] = {
            "scene": None,
            "camera": None,
            "split": None,
            "video_path_relative": None,
            "ground_truth_available": False,
            "calibration_available": False,
        }
    else:
        summary["aicity_dataset"] = {
            "scene": entry.get("scene"),
            "camera": entry.get("camera"),
            "split": entry.get("split"),
            "video_path_relative": entry.get("video_path_relative"),
            "ground_truth_available": bool(entry.get("ground_truth_path_relative")),
            "calibration_available": bool(entry.get("calibration_path_relative")),
        }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Process one AI City camera video with the UrbanTrack pipeline."
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Absolute or relative path to the source video (vdo.avi/vdo.mp4). "
        "If omitted, the path is looked up in the manifest by --camera-id.",
    )
    parser.add_argument(
        "--camera-id",
        required=True,
        help="Camera identifier, e.g. CAM_S01_C001.",
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Path to the AI City manifest. Defaults to data/config/aicity_manifest.json.",
    )
    args = parser.parse_args(argv)

    camera_id = args.camera_id
    manifest_path = Path(args.manifest)
    if args.video:
        video_path = Path(args.video)
    else:
        video_path = _lookup_video_path(camera_id, manifest_path)

    if not video_path.is_file():
        raise SystemExit(f"Source video does not exist: {video_path}")

    output_dir = PerceptionPipeline._camera_output_dir(camera_id)

    print(f"Camera ID: {camera_id}")
    print(f"Source video: {video_path}")
    print(f"Output directory: {output_dir}")

    start = time.perf_counter()
    pipeline = PerceptionPipeline()
    frame_count = pipeline.process(
        input_path=video_path,
        camera_id=camera_id,
        allow_unknown_camera=True,
    )
    elapsed = time.perf_counter() - start

    _inject_aicity_provenance(camera_id, manifest_path, output_dir)

    print(f"Frames processed: {frame_count}")
    print(f"Elapsed processing time: {elapsed:.2f} seconds")
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()