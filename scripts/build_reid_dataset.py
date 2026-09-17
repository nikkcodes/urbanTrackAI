"""Build TorchReID-compatible training dataset from AI City videos and ground truth.

Extracts vehicle bounding-box crops from AI City videos based on gt.txt annotations,
organizing them by split and camera-namespaced vehicle identity:
    data/reid_dataset/train/<camera_id>_<identity>/<camera_id>_<frame>.jpg
    data/reid_dataset/validation/<camera_id>_<identity>/<camera_id>_<frame>.jpg

Produces data/reid_dataset/dataset_manifest.json with identity_namespace="camera_local".
"""

from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = "data/config/aicity_manifest.json"
DEFAULT_OUTPUT_DIR = "data/reid_dataset"
MIN_CROP_DIM = 32


def load_manifest(manifest_path: Path) -> dict:
    """Read the AI City dataset manifest."""
    if not manifest_path.is_file():
        raise SystemExit(f"Manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def filter_cameras(
    manifest: dict,
    split: str,
    camera_id_filter: str | None = None,
    max_cameras: int | None = None,
) -> list[dict]:
    """Filter cameras by split, camera ID, and maximum limit."""
    cameras = manifest.get("cameras", [])
    if split != "both":
        cameras = [c for c in cameras if c.get("split") == split]

    if camera_id_filter:
        cameras = [c for c in cameras if c.get("camera_id") == camera_id_filter]

    if max_cameras is not None and max_cameras > 0:
        cameras = cameras[:max_cameras]

    return list(cameras)


def parse_ground_truth(gt_path: Path) -> dict[int, list[dict]]:
    """Parse MOT-format gt.txt into frame-indexed annotations.

    Format per line:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
    """
    frame_to_annos: dict[int, list[dict]] = collections.defaultdict(list)
    with gt_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                frame_num = int(float(parts[0]))
                identity_id = int(float(parts[1]))
                bb_left = float(parts[2])
                bb_top = float(parts[3])
                bb_width = float(parts[4])
                bb_height = float(parts[5])
            except (ValueError, IndexError):
                continue

            frame_to_annos[frame_num].append(
                {
                    "identity_id": identity_id,
                    "bb_left": bb_left,
                    "bb_top": bb_top,
                    "bb_width": bb_width,
                    "bb_height": bb_height,
                }
            )
    return frame_to_annos


def resolve_gt_path(entry: dict, dataset_root: Path) -> Path | None:
    """Locate gt.txt for a camera entry."""
    gt_rel = entry.get("ground_truth_path_relative")
    if gt_rel:
        candidate = dataset_root / gt_rel
        if candidate.is_file():
            return candidate

    # Fallback to relative to video path
    video_path = entry.get("video_path")
    if video_path:
        candidate = Path(video_path).parent / "gt" / "gt.txt"
        if candidate.is_file():
            return candidate

    return None


def process_camera(
    entry: dict,
    dataset_root: Path,
    output_dir: Path,
) -> dict:
    """Extract crops for one camera and write to dataset directory."""
    camera_id = entry["camera_id"]
    split = entry.get("split", "train")
    video_path = Path(entry["video_path"])

    stats = {
        "camera_id": camera_id,
        "split": split,
        "images_exported": 0,
        "identities": set(),
        "skipped_crops": 0,
        "skipped_small_crops": 0,
    }

    if not video_path.is_file():
        print(f"[{camera_id}] Video not found: {video_path}, skipping.")
        return stats

    gt_path = resolve_gt_path(entry, dataset_root)
    if not gt_path or not gt_path.is_file():
        print(f"[{camera_id}] Ground truth not found, skipping.")
        return stats

    frame_to_annos = parse_ground_truth(gt_path)
    if not frame_to_annos:
        print(f"[{camera_id}] No annotations found in {gt_path}.")
        return stats

    max_target_frame = max(frame_to_annos.keys())
    split_dir = output_dir / split

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[{camera_id}] Failed to open video: {video_path}")
        return stats

    current_frame_idx = 1
    while current_frame_idx <= max_target_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_idx in frame_to_annos:
            frame_h, frame_w = frame.shape[:2]
            for anno in frame_to_annos[current_frame_idx]:
                bb_left = anno["bb_left"]
                bb_top = anno["bb_top"]
                bb_width = anno["bb_width"]
                bb_height = anno["bb_height"]
                identity_id = anno["identity_id"]

                x1 = max(0, int(round(bb_left)))
                y1 = max(0, int(round(bb_top)))
                x2 = min(frame_w, int(round(bb_left + bb_width)))
                y2 = min(frame_h, int(round(bb_top + bb_height)))

                crop_w = x2 - x1
                crop_h = y2 - y1

                if crop_w < MIN_CROP_DIM or crop_h < MIN_CROP_DIM:
                    stats["skipped_crops"] += 1
                    stats["skipped_small_crops"] += 1
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    stats["skipped_crops"] += 1
                    continue

                # Save crop in camera-namespaced identity folder
                identity_str = f"{camera_id}_{identity_id:06d}"
                identity_dir = split_dir / identity_str
                identity_dir.mkdir(parents=True, exist_ok=True)

                filename = f"{camera_id}_{current_frame_idx:06d}.jpg"
                crop_path = identity_dir / filename
                cv2.imwrite(str(crop_path), crop)

                stats["images_exported"] += 1
                stats["identities"].add(identity_str)

        current_frame_idx += 1

    cap.release()
    print(
        f"[{camera_id}] Processed: {stats['images_exported']} crops, "
        f"{len(stats['identities'])} identities, "
        f"{stats['skipped_crops']} skipped (small: {stats['skipped_small_crops']})"
    )
    return stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build TorchReID dataset from AI City challenge videos."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Path to aicity_manifest.json.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "both"),
        default="train",
        help="Split to build (train, validation, or both). Default: train.",
    )
    parser.add_argument(
        "--camera-id",
        default=None,
        help="Filter to a specific camera (e.g. CAM_S01_C001).",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=None,
        help="Maximum number of cameras to process.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output dataset directory. Default: data/reid_dataset.",
    )

    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean target split directory to avoid lingering unnamespaced folders
    if not args.camera_id:
        for split_name in ("train", "validation"):
            if args.split in (split_name, "both"):
                split_target = output_dir / split_name
                if split_target.exists():
                    shutil.rmtree(split_target)

    manifest = load_manifest(manifest_path)
    dataset_root = Path(manifest.get("dataset_root", "."))

    cameras = filter_cameras(
        manifest=manifest,
        split=args.split,
        camera_id_filter=args.camera_id,
        max_cameras=args.max_cameras,
    )

    print(
        f"Building Re-ID dataset: {len(cameras)} camera(s) selected "
        f"(split={args.split}, max_cameras={args.max_cameras})."
    )

    total_images = 0
    total_skipped = 0
    total_skipped_small = 0
    train_identities: set[str] = set()
    val_identities: set[str] = set()

    start_time = time.perf_counter()
    for entry in cameras:
        stats = process_camera(
            entry=entry,
            dataset_root=dataset_root,
            output_dir=output_dir,
        )
        total_images += stats["images_exported"]
        total_skipped += stats["skipped_crops"]
        total_skipped_small += stats["skipped_small_crops"]
        if stats["split"] == "train":
            train_identities.update(stats["identities"])
        elif stats["split"] == "validation":
            val_identities.update(stats["identities"])

    all_identities = train_identities | val_identities
    elapsed = time.perf_counter() - start_time

    # Generate dataset_manifest.json
    manifest_data = {
        "total_identities": len(all_identities),
        "total_images": total_images,
        "train_identities": len(train_identities),
        "validation_identities": len(val_identities),
        "skipped_crops": total_skipped,
        "skipped_small_crops": total_skipped_small,
        "identity_namespace": "camera_local",
        "identity_folder_format": "CAM_Sxx_Cyyy_000007",
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    manifest_output_path = output_dir / "dataset_manifest.json"
    with manifest_output_path.open("w", encoding="utf-8") as file:
        json.dump(manifest_data, file, indent=2)

    print("\n" + "=" * 50)
    print("Re-ID Dataset Build Complete")
    print(f"Identities exported: {len(all_identities)}")
    print(f"Images exported: {total_images}")
    print(f"Skipped crops: {total_skipped}")
    print(f"Elapsed time: {elapsed:.2f}s")
    print(f"Manifest written to: {manifest_output_path}")
    print("=" * 50)

    return 0


if __name__ == "__main__":
    sys.exit(main())
