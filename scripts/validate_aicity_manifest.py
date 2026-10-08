"""
Read-only validation script for UrbanTrack AI City Manifest.
Checks data/config/aicity_manifest.json against the actual dataset filesystem.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def validate_manifest(manifest_path: Path) -> int:
    if not manifest_path.exists():
        print(f"Error: Manifest file does not exist: {manifest_path}")
        return 1

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dataset_root = Path(manifest.get("dataset_root", ""))
    if not dataset_root.is_dir():
        print(f"Error: Dataset root does not exist: {dataset_root}")
        return 1

    cameras = manifest.get("cameras", [])
    total_entries = len(cameras)
    unique_cams = sorted({c["camera"] for c in cameras})
    scenes = sorted({c["scene"] for c in cameras})

    scene_counts = {}
    for c in cameras:
        sc = c["scene"]
        scene_counts[sc] = scene_counts.get(sc, 0) + 1

    s06_entries = [c["camera_id"] for c in cameras if c["scene"] == "S06"]

    # File validation on disk
    missing_videos = []
    missing_calibs = []
    missing_expected_gt = []

    for c in cameras:
        # 1. Video file check
        v_path = Path(c.get("video_path", ""))
        if not v_path.is_file():
            missing_videos.append(c["camera_id"])

        # 2. Calibration file check
        cal_rel = c.get("calibration_path_relative")
        if not cal_rel or not (dataset_root / cal_rel).is_file():
            missing_calibs.append(c["camera_id"])

        # 3. Expected ground truth check (S01-S05 are expected to have GT, S06 is test split without GT)
        if c["scene"] in ("S01", "S02", "S03", "S04", "S05"):
            gt_rel = c.get("ground_truth_path_relative")
            if not gt_rel or not (dataset_root / gt_rel).is_file():
                missing_expected_gt.append(c["camera_id"])
        elif c["scene"] == "S06":
            gt_rel = c.get("ground_truth_path_relative")
            if gt_rel is not None:
                print(f"Warning: S06 entry {c['camera_id']} unexpectedly has ground truth path: {gt_rel}")

    print("========================================")
    print("AICITY MANIFEST VALIDATION")
    print("========================================")
    print(f"\nTotal entries: {total_entries}")
    print(f"Unique camera numbers: {len(unique_cams)}\n")

    for sc in ("S01", "S02", "S03", "S04", "S05", "S06"):
        print(f"{sc}: {scene_counts.get(sc, 0)}")

    print("\nS06 entries:")
    for s06_id in s06_entries:
        print(s06_id)

    print(f"\nMissing video files: {len(missing_videos)}")
    print(f"Missing calibration files: {len(missing_calibs)}")
    print(f"Missing expected GT files: {len(missing_expected_gt)}")
    print("\n========================================")

    # Sanity asserts
    has_errors = False
    if total_entries != 65:
        print(f"FAILED: Expected 65 total entries, got {total_entries}")
        has_errors = True
    if len(unique_cams) != 46:
        print(f"FAILED: Expected 46 unique camera numbers, got {len(unique_cams)}")
        has_errors = True
    if len(s06_entries) != 6:
        print(f"FAILED: Expected 6 S06 entries, got {len(s06_entries)}")
        has_errors = True
    if missing_videos or missing_calibs or missing_expected_gt:
        print("FAILED: Missing files detected on disk!")
        has_errors = True

    return 1 if has_errors else 0


if __name__ == "__main__":
    default_manifest = Path(__file__).resolve().parent.parent / "data" / "config" / "aicity_manifest.json"
    manifest_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else default_manifest
    sys.exit(validate_manifest(manifest_arg))
