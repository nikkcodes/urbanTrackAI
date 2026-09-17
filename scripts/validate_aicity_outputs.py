"""Validate AI City perception outputs for completeness and consistency.

Read-only validator checking:
- Output folder and required artefact existence
- perception_summary.json metadata consistency
- observations.json camera_id and frames_processed correspondence
- Cross-reference with aicity_manifest.json and aicity_index.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MANIFEST = REPO_ROOT / "data" / "config" / "aicity_manifest.json"
DEFAULT_INDEX = REPO_ROOT / "data" / "output" / "aicity_index.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "output"

REQUIRED_FILES = (
    "observations.json",
    "trajectories.json",
    "camera_metrics.json",
    "perception_summary.json",
    "vdo_tracked.avi",
)


def load_json(path: Path) -> dict:
    """Read a JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def validate_camera(
    camera_id: str,
    output_dir: Path,
    manifest_cameras: dict[str, dict],
) -> list[str]:
    """Validate a single camera output directory.

    Returns a list of failure reason strings (empty if valid).
    """
    errors: list[str] = []
    folder = output_dir / camera_id

    # 1. Output directory exists
    if not folder.is_dir():
        return [f"Output folder does not exist: {folder}"]

    # Check manifest awareness
    if manifest_cameras and camera_id not in manifest_cameras:
        errors.append(f"Camera ID {camera_id} not found in manifest")

    # 2. Required files exist
    missing_files = [fname for fname in REQUIRED_FILES if not (folder / fname).is_file()]
    if missing_files:
        errors.append(f"Missing required file(s): {', '.join(missing_files)}")

    # 3. Validate perception_summary.json
    summary_path = folder / "perception_summary.json"
    summary_frames: int | None = None
    if summary_path.is_file():
        try:
            summary = load_json(summary_path)
            # camera_id matches
            if summary.get("camera_id") != camera_id:
                errors.append(
                    f"perception_summary.json camera_id mismatch: expected '{camera_id}', "
                    f"got '{summary.get('camera_id')}'"
                )

            # frames_processed > 0
            frames_processed = summary.get("frames_processed")
            if frames_processed is None or not isinstance(frames_processed, int) or frames_processed <= 0:
                errors.append(
                    f"perception_summary.json frames_processed must be > 0, got {frames_processed}"
                )
            else:
                summary_frames = frames_processed

            # unique_tracks exists
            if "unique_tracks" not in summary or summary.get("unique_tracks") is None:
                errors.append("perception_summary.json missing 'unique_tracks'")

            # reid_model exists
            if not summary.get("reid_model"):
                errors.append("perception_summary.json missing 'reid_model'")
        except Exception as exc:
            errors.append(f"Failed to parse perception_summary.json: {exc}")

    # 4. Validate observations.json
    obs_path = folder / "observations.json"
    if obs_path.is_file():
        try:
            obs = load_json(obs_path)
            # top-level camera_id matches
            if obs.get("camera_id") != camera_id:
                errors.append(
                    f"observations.json camera_id mismatch: expected '{camera_id}', "
                    f"got '{obs.get('camera_id')}'"
                )

            # frames_processed matches summary
            obs_frames = obs.get("frames_processed")
            if summary_frames is not None and obs_frames != summary_frames:
                errors.append(
                    f"frames_processed mismatch between observations.json ({obs_frames}) "
                    f"and perception_summary.json ({summary_frames})"
                )
        except Exception as exc:
            errors.append(f"Failed to parse observations.json: {exc}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate AI City Challenge perception outputs."
    )
    parser.add_argument(
        "--camera-id",
        default=None,
        help="Validate one camera by ID (e.g. CAM_S01_C001).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Validate every camera recorded in aicity_index.json.",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Path to aicity_manifest.json.",
    )
    parser.add_argument(
        "--index",
        default=str(DEFAULT_INDEX),
        help="Path to aicity_index.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Path to outputs directory. Default: data/output.",
    )

    args = parser.parse_args(argv)

    if not args.all and not args.camera_id:
        parser.error("Please specify either --all or --camera-id <CAMERA_ID>.")

    manifest_path = Path(args.manifest)
    index_path = Path(args.index)
    output_dir = Path(args.output_dir)

    if not manifest_path.is_file():
        print(f"Error: manifest file not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = load_json(manifest_path)
    manifest_cameras = {c["camera_id"]: c for c in manifest.get("cameras", []) if "camera_id" in c}

    if not index_path.is_file():
        print(f"Error: index file not found: {index_path}", file=sys.stderr)
        return 1

    index_data = load_json(index_path)
    indexed_cameras = index_data.get("cameras", [])

    # Select target cameras
    if args.camera_id:
        target_camera_ids = [args.camera_id]
    else:
        target_camera_ids = [c["camera_id"] for c in indexed_cameras if "camera_id" in c]

    if not target_camera_ids:
        print("No cameras to validate.")
        print(f"Processed cameras: 0")
        print(f"Passed: 0")
        print(f"Failed: 0")
        return 0

    failures: dict[str, list[str]] = {}
    passed_count = 0

    for cam_id in target_camera_ids:
        errs = validate_camera(
            camera_id=cam_id,
            output_dir=output_dir,
            manifest_cameras=manifest_cameras,
        )
        if errs:
            failures[cam_id] = errs
        else:
            passed_count += 1

    total = len(target_camera_ids)
    failed_count = len(failures)

    print(f"Processed cameras: {total}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")

    if failures:
        print("\nFailures:")
        for cam_id, err_list in failures.items():
            print(f"- {cam_id}:")
            for err in err_list:
                print(f"    * {err}")

    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
