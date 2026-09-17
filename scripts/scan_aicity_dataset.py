"""Scan the AI City Challenge 2022 Track 1 MTMC Tracking dataset.

Discovers every CCTV camera video under the dataset root and writes a
manifest to ``data/config/aicity_manifest.json``. This is a pure discovery
utility: it performs no perception processing, no cleanup, and no
modification of any Member 1 code.

Each camera entry records the scene, camera, camera_id, split, absolute
and relative video paths, plus relative paths to the optional
``gt/gt.txt`` and ``calibration.txt`` files (``null`` when absent). The
top-level manifest reports how many cameras have ground truth and
calibration data.

Usage:
    python scripts/scan_aicity_dataset.py
    python scripts/scan_aicity_dataset.py --dataset-root C:/path/to/AICity22_Track1_MTMC_Tracking
    python scripts/scan_aicity_dataset.py --split train --split validation
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_DATASET_ROOT = (
    r"C:\Users\kanis\Downloads\AICity22_Track1_MTMC_Tracking"
)
DEFAULT_OUTPUT = Path("data/config/aicity_manifest.json")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")


def discover_cameras(
    dataset_root: Path,
    splits: tuple[str, ...],
) -> list[dict]:
    """Return one manifest entry per discovered camera video.

    Each entry contains scene, camera, camera_id, split, absolute video
    path, relative video path, and relative paths to the optional
    ground-truth and calibration files (``null`` when absent). Videos are
    matched by name (``vdo.mp4`` or ``vdo.avi``) inside
    ``<split>/<scene>/<camera>/``.
    """
    entries: list[dict] = []
    for split in splits:
        split_dir = dataset_root / split
        if not split_dir.is_dir():
            continue
        for scene_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            scene = scene_dir.name
            for camera_dir in sorted(p for p in scene_dir.iterdir() if p.is_dir()):
                camera = camera_dir.name
                video = _find_video(camera_dir)
                if video is None:
                    continue
                camera_id = f"CAM_{scene}_{camera.upper()}"
                gt_path = _relative_if_exists(camera_dir / "gt" / "gt.txt", dataset_root)
                cal_path = _relative_if_exists(camera_dir / "calibration.txt", dataset_root)
                entries.append(
                    {
                        "scene": scene,
                        "camera": camera,
                        "camera_id": camera_id,
                        "split": split,
                        "video_name": video.name,
                        "video_path": str(video.resolve()),
                        "video_path_relative": str(video.relative_to(dataset_root)),
                        "ground_truth_path_relative": gt_path,
                        "calibration_path_relative": cal_path,
                    }
                )
    return entries


def _relative_if_exists(path: Path, dataset_root: Path) -> str | None:
    """Return the path relative to the dataset root, or ``None`` if missing."""
    if path.is_file():
        return str(path.relative_to(dataset_root))
    return None


def _find_video(camera_dir: Path) -> Path | None:
    """Return the first supported video inside a camera directory."""
    for name in ("vdo.mp4", "vdo.avi", "vdo.mov", "vdo.mkv"):
        candidate = camera_dir / name
        if candidate.is_file():
            return candidate
    for candidate in sorted(camera_dir.iterdir()):
        if candidate.is_file() and candidate.suffix.lower() in VIDEO_EXTENSIONS:
            return candidate
    return None


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Scan the AI City 2022 Track 1 MTMC dataset and write a manifest."
    )
    parser.add_argument(
        "--dataset-root",
        default=DEFAULT_DATASET_ROOT,
        help="Root directory of the AI City dataset.",
    )
    parser.add_argument(
        "--split",
        action="append",
        choices=("train", "validation", "test"),
        default=None,
        help="Dataset split to scan. Repeatable. Defaults to train and validation.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Path for the generated manifest JSON.",
    )
    args = parser.parse_args(argv)

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset root does not exist: {dataset_root}")

    splits = tuple(args.split) if args.split else ("train", "validation")
    entries = discover_cameras(dataset_root, splits)

    scenes = sorted({entry["scene"] for entry in entries})
    ground_truth_cameras = sum(
        1 for entry in entries if entry.get("ground_truth_path_relative")
    )
    calibration_cameras = sum(
        1 for entry in entries if entry.get("calibration_path_relative")
    )
    manifest = {
        "dataset_root": str(dataset_root),
        "splits_scanned": list(splits),
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "scene_count": len(scenes),
        "scenes": scenes,
        "camera_count": len(entries),
        "ground_truth_cameras": ground_truth_cameras,
        "calibration_cameras": calibration_cameras,
        "cameras": entries,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=4)

    print(f"Dataset root: {dataset_root}")
    print(f"Splits scanned: {', '.join(splits)}")
    print(f"Scenes found: {len(scenes)} ({', '.join(scenes)})")
    print(f"Cameras found: {len(entries)}")
    print(f"Cameras with ground truth: {ground_truth_cameras}")
    print(f"Cameras with calibration: {calibration_cameras}")
    print(f"Manifest written: {output_path}")


if __name__ == "__main__":
    main()