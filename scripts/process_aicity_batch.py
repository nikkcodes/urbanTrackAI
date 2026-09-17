"""Batch processor for AI City Challenge 2022 Track 1 dataset.

Orchestration layer that automatically processes AI City camera videos
using the existing PerceptionPipeline and process_camera provenance logic.

Features:
- Manifest-driven processing using data/config/aicity_manifest.json
- Automatic runtime cleanup (preserving config, source, models, and metadata)
- Resume mode to skip already-processed cameras
- Progress display with ETA, elapsed time, and counts
- Structured logging to logs/aicity_processing.log
- Processing index generation for Member 2 (data/output/aicity_index.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from perception.pipeline import PerceptionPipeline

# Allow running from repository root without installation.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_MANIFEST = "data/config/aicity_manifest.json"
DEFAULT_OUTPUT_DIR = "data/output"
DEFAULT_SYNTHETIC_DIR = "data/synthetic_output"
DEFAULT_LOG_FILE = "logs/aicity_processing.log"
INDEX_FILE_NAME = "aicity_index.json"

PROTECTED_NAMES = {
    "aicity_manifest.json",
    "camera_metadata.json",
}

logger = logging.getLogger("aicity_batch")


def setup_logger(log_file: Path) -> None:
    """Configure logger to write to both file and console."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # File handler records full timestamps and levels
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)


def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS string."""
    if seconds < 0 or math.isinf(seconds) or math.isnan(seconds):
        return "--:--:--"
    total_sec = int(seconds)
    m, s = divmod(total_sec, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def normalize_rel_path(path_str: str | None) -> str | None:
    """Normalize file path separators to forward slashes."""
    if not path_str:
        return None
    return path_str.replace("\\", "/")


def get_rel_output_dir(path: Path) -> str:
    """Return relative posix path string for camera output directory."""
    try:
        resolved = path.resolve()
        if resolved.is_relative_to(REPO_ROOT):
            return normalize_rel_path(str(resolved.relative_to(REPO_ROOT))) or ""
    except Exception:
        pass
    return normalize_rel_path(str(path)) or ""


def load_manifest(manifest_path: Path) -> dict:
    """Read the manifest JSON file."""
    if not manifest_path.is_file():
        raise SystemExit(
            f"Manifest file not found: {manifest_path}. "
            "Please ensure data/config/aicity_manifest.json exists."
        )
    with manifest_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def filter_cameras(
    manifest: dict,
    split: str,
    camera_filter: str | None = None,
    max_cameras: int | None = None,
) -> list[dict]:
    """Filter cameras based on split, camera ID filter, and max cameras."""
    cameras = manifest.get("cameras", [])
    if split != "both":
        cameras = [c for c in cameras if c.get("split") == split]

    if camera_filter:
        allowed = {c.strip() for c in camera_filter.split(",") if c.strip()}
        cameras = [
            c
            for c in cameras
            if c.get("camera_id") in allowed or c.get("camera") in allowed
        ]

    if max_cameras is not None and max_cameras > 0:
        cameras = cameras[:max_cameras]

    return list(cameras)


def clean_runtime_artifacts(
    directories: list[Path],
    protected_names: set[str],
) -> list[str]:
    """Delete contents of runtime directories while protecting required files."""
    deleted: list[str] = []
    for directory in directories:
        if not directory.exists():
            continue
        for item in directory.iterdir():
            if item.name in protected_names:
                continue
            try:
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    item.unlink()
                deleted.append(str(item))
            except Exception as exc:
                print(f"Warning: Failed to delete {item}: {exc}")
    return deleted


def inject_aicity_provenance(
    entry: dict,
    output_dir: Path,
) -> None:
    """Add the aicity_dataset provenance block to perception_summary.json."""
    summary_path = output_dir / "perception_summary.json"
    if not summary_path.is_file():
        return
    try:
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)
        if not isinstance(summary, dict):
            return

        summary["aicity_dataset"] = {
            "scene": entry.get("scene"),
            "camera": entry.get("camera"),
            "split": entry.get("split"),
            "video_path_relative": normalize_rel_path(entry.get("video_path_relative")),
            "ground_truth_available": bool(entry.get("ground_truth_path_relative")),
            "calibration_available": bool(entry.get("calibration_path_relative")),
        }
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=4)
    except Exception as exc:
        logger.warning(
            f"Failed to inject provenance for {entry.get('camera_id')}: {exc}"
        )


def process_single_camera(
    entry: dict,
    pipeline: Any,
    output_root: Path,
    resume: bool = False,
    overwrite: bool = False,
) -> dict:
    """Process a single camera or resume from existing summary.

    Returns the camera record dictionary for aicity_index.json.
    """
    camera_id = entry["camera_id"]
    scene = entry.get("scene", "unknown")
    split = entry.get("split", "train")
    camera_output_dir = output_root / camera_id
    summary_path = camera_output_dir / "perception_summary.json"

    camera_record: dict[str, Any] = {
        "camera_id": camera_id,
        "scene": scene,
        "camera": entry.get("camera"),
        "split": split,
        "video_path_relative": normalize_rel_path(entry.get("video_path_relative")),
        "ground_truth_path_relative": normalize_rel_path(entry.get("ground_truth_path_relative")),
        "calibration_path_relative": normalize_rel_path(entry.get("calibration_path_relative")),
        "output_directory": get_rel_output_dir(camera_output_dir),
        "status": "failed",
        "frames_processed": 0,
        "elapsed_seconds": 0.0,
        "processing_timestamp": None,
        "error": None,
    }

    # Check resume mode
    if resume and not overwrite and summary_path.is_file():
        logger.info(f"Camera {camera_id} skipped (already processed)")
        try:
            with summary_path.open("r", encoding="utf-8") as file:
                existing_summary = json.load(file)
            camera_record["status"] = "success"
            camera_record["frames_processed"] = existing_summary.get("frames_processed", 0)
            camera_record["elapsed_seconds"] = existing_summary.get(
                "elapsed_seconds",
                existing_summary.get("duration_seconds", 0.0),
            )
            camera_record["processing_timestamp"] = existing_summary.get("processing_timestamp")
            # Ensure provenance is present in summary
            inject_aicity_provenance(entry, camera_output_dir)
            return camera_record
        except Exception as exc:
            logger.warning(
                f"Could not read existing summary for {camera_id}: {exc}. Reprocessing."
            )

    video_path = Path(entry["video_path"])
    if not video_path.is_file():
        err_msg = f"Video file not found: {video_path}"
        logger.error(f"Camera {camera_id} failed: {err_msg}")
        camera_record["status"] = "failed"
        camera_record["error"] = err_msg
        return camera_record

    logger.info(f"Camera {camera_id} start | Scene: {scene} | Split: {split} | Video: {video_path}")
    start_time = time.perf_counter()
    try:
        frames = pipeline.process(
            input_path=video_path,
            camera_id=camera_id,
            allow_unknown_camera=True,
        )
        elapsed = time.perf_counter() - start_time
        inject_aicity_provenance(entry, camera_output_dir)

        # Read actual summary timestamp if available
        timestamp = datetime.now(timezone.utc).isoformat()
        if summary_path.is_file():
            try:
                with summary_path.open("r", encoding="utf-8") as file:
                    summary_data = json.load(file)
                timestamp = summary_data.get("processing_timestamp", timestamp)
            except Exception:
                pass

        logger.info(
            f"Camera {camera_id} finish | Elapsed: {elapsed:.2f}s | Frames: {frames}"
        )
        camera_record["status"] = "success"
        camera_record["frames_processed"] = frames
        camera_record["elapsed_seconds"] = round(elapsed, 2)
        camera_record["processing_timestamp"] = timestamp
        camera_record["error"] = None
    except Exception as exc:
        elapsed = time.perf_counter() - start_time
        tb_str = traceback.format_exc()
        logger.error(f"Camera {camera_id} failed after {elapsed:.2f}s: {exc}")
        logger.error(f"Traceback for {camera_id}:\n{tb_str}")
        camera_record["status"] = "failed"
        camera_record["elapsed_seconds"] = round(elapsed, 2)
        camera_record["error"] = f"{type(exc).__name__}: {exc}"

    return camera_record


def write_index_file(
    index_path: Path,
    camera_records: list[dict],
) -> dict:
    """Write data/output/aicity_index.json."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    successful_count = sum(1 for c in camera_records if c.get("status") == "success")
    failed_count = sum(1 for c in camera_records if c.get("status") == "failed")

    index_data = {
        "dataset": "AI City Challenge 2022 Track 1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processed_cameras": len(camera_records),
        "successful_cameras": successful_count,
        "failed_cameras": failed_count,
        "cameras": camera_records,
    }

    with index_path.open("w", encoding="utf-8") as file:
        json.dump(index_data, file, indent=2)

    logger.info(
        f"Generated index: {index_path} (processed={len(camera_records)}, "
        f"success={successful_count}, failed={failed_count})"
    )
    return index_data


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Batch processor for AI City Challenge 2022 Track 1."
    )
    parser.add_argument(
        "--manifest",
        default=DEFAULT_MANIFEST,
        help="Path to manifest file. Defaults to data/config/aicity_manifest.json.",
    )
    parser.add_argument(
        "--split",
        choices=("train", "validation", "both"),
        default="train",
        help="Dataset split to process (train, validation, or both). Default: train.",
    )
    parser.add_argument(
        "--max-cameras",
        type=int,
        default=None,
        help="Maximum number of cameras to process.",
    )
    parser.add_argument(
        "--camera-filter",
        default=None,
        help="Filter cameras by ID (e.g. CAM_S01_C001) or comma-separated list.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume processing: skip cameras whose output summary already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing camera outputs even in resume mode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker threads (default: 1).",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write camera outputs. Default: data/output.",
    )
    parser.add_argument(
        "--log-file",
        default=DEFAULT_LOG_FILE,
        help="Path to output log file. Default: logs/aicity_processing.log.",
    )

    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    output_dir = Path(args.output_dir)
    synthetic_dir = Path(DEFAULT_SYNTHETIC_DIR)
    log_file = Path(args.log_file)
    index_path = output_dir / INDEX_FILE_NAME

    setup_logger(log_file)
    logger.info("=" * 60)
    logger.info("AI City Batch Processor started.")
    logger.info(
        f"Args: split={args.split}, max_cameras={args.max_cameras}, "
        f"camera_filter={args.camera_filter}, resume={args.resume}, "
        f"overwrite={args.overwrite}, workers={args.workers}"
    )

    # 1. Automatic Runtime Cleanup
    if args.resume:
        print("Resume mode enabled: skipping automatic runtime cleanup.")
        logger.info("Resume mode enabled: skipped automatic runtime cleanup.")
    else:
        print("Running automatic runtime cleanup of previous outputs...")
        deleted = clean_runtime_artifacts(
            [output_dir, synthetic_dir],
            PROTECTED_NAMES,
        )
        if deleted:
            print(f"Cleaned {len(deleted)} runtime artifact(s):")
            for item in deleted:
                print(f"  - {item}")
            logger.info(f"Cleaned {len(deleted)} runtime artifacts.")
        else:
            print("No previous runtime artifacts to clean.")
            logger.info("No runtime artifacts were present to clean.")

    # 2. Manifest Loading & Camera Selection
    manifest = load_manifest(manifest_path)
    cameras = filter_cameras(
        manifest=manifest,
        split=args.split,
        camera_filter=args.camera_filter,
        max_cameras=args.max_cameras,
    )

    total_cameras = len(cameras)
    print(
        f"AI City Batch Processor: {total_cameras} camera(s) selected "
        f"(split={args.split}, max_cameras={args.max_cameras})."
    )
    if total_cameras == 0:
        print("No cameras match the selection criteria. Exiting.")
        write_index_file(index_path, [])
        return 0

    batch_start_time = time.perf_counter()
    camera_records: list[dict] = []
    processed_count = 0

    # Sequential execution (workers == 1) or Multi-worker execution
    if args.workers <= 1:
        from perception.pipeline import PerceptionPipeline
        pipeline = PerceptionPipeline()

        for index, entry in enumerate(cameras, start=1):
            camera_id = entry["camera_id"]
            scene = entry.get("scene", "unknown")

            # Progress display calculations
            pct = ((index - 1) / total_cameras) * 100.0
            batch_elapsed = time.perf_counter() - batch_start_time
            avg_time = (batch_elapsed / processed_count) if processed_count > 0 else 0.0
            remaining_count = total_cameras - (index - 1)
            eta_seconds = remaining_count * avg_time if processed_count > 0 else 0.0
            eta_str = format_duration(eta_seconds) if processed_count > 0 else "--:--:--"

            print(
                f"[{index}/{total_cameras}] {camera_id} | Scene {scene} | "
                f"{pct:.1f}% | ETA {eta_str}"
            )
            print(
                f"  Elapsed: {format_duration(batch_elapsed)} | "
                f"Processed: {processed_count} | Remaining: {remaining_count}"
            )

            record = process_single_camera(
                entry=entry,
                pipeline=pipeline,
                output_root=output_dir,
                resume=args.resume,
                overwrite=args.overwrite,
            )
            camera_records.append(record)
            processed_count += 1

            if record["status"] == "success":
                print(
                    f"  -> {camera_id} completed: {record['frames_processed']} frames "
                    f"in {record['elapsed_seconds']:.2f}s"
                )
            else:
                print(f"  -> {camera_id} failed: {record['error']}")

            # Progressively persist the index file so interruptions keep progress
            write_index_file(index_path, camera_records)
    else:
        # Multi-worker execution using thread-local pipelines
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from perception.pipeline import PerceptionPipeline

        thread_local = threading.local()

        def get_thread_pipeline() -> PerceptionPipeline:
            if not hasattr(thread_local, "pipeline"):
                thread_local.pipeline = PerceptionPipeline()
            return thread_local.pipeline

        def worker_task(camera_entry: dict, idx: int) -> tuple[int, dict]:
            pipe = get_thread_pipeline()
            res = process_single_camera(
                entry=camera_entry,
                pipeline=pipe,
                output_root=output_dir,
                resume=args.resume,
                overwrite=args.overwrite,
            )
            return idx, res

        lock = threading.Lock()
        results: list[tuple[int, dict]] = []

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_entry = {
                executor.submit(worker_task, entry, idx): (idx, entry)
                for idx, entry in enumerate(cameras, start=1)
            }
            for future in as_completed(future_to_entry):
                idx, entry = future_to_entry[future]
                camera_id = entry["camera_id"]
                scene = entry.get("scene", "unknown")
                try:
                    _, record = future.result()
                except Exception as exc:
                    logger.error(f"Camera {camera_id} failed: {exc}")
                    logger.error(f"Traceback for {camera_id}:\n{traceback.format_exc()}")
                    record = {
                        "camera_id": camera_id,
                        "scene": scene,
                        "camera": entry.get("camera"),
                        "split": entry.get("split"),
                        "video_path_relative": normalize_rel_path(entry.get("video_path_relative")),
                        "ground_truth_path_relative": normalize_rel_path(entry.get("ground_truth_path_relative")),
                        "calibration_path_relative": normalize_rel_path(entry.get("calibration_path_relative")),
                        "output_directory": get_rel_output_dir(output_dir / camera_id),
                        "status": "failed",
                        "frames_processed": 0,
                        "elapsed_seconds": 0.0,
                        "processing_timestamp": None,
                        "error": str(exc),
                    }

                with lock:
                    results.append((idx, record))
                    processed_count += 1
                    batch_elapsed = time.perf_counter() - batch_start_time
                    avg_time = batch_elapsed / processed_count
                    remaining_count = total_cameras - processed_count
                    eta_seconds = remaining_count * avg_time
                    pct = (processed_count / total_cameras) * 100.0

                    print(
                        f"[{processed_count}/{total_cameras}] {camera_id} | Scene {scene} | "
                        f"{pct:.1f}% | ETA {format_duration(eta_seconds)}"
                    )
                    print(
                        f"  Elapsed: {format_duration(batch_elapsed)} | "
                        f"Processed: {processed_count} | Remaining: {remaining_count}"
                    )

                    # Update index
                    sorted_records = [r for _, r in sorted(results, key=lambda x: x[0])]
                    write_index_file(index_path, sorted_records)

        camera_records = [r for _, r in sorted(results, key=lambda x: x[0])]

    # Final Summary
    total_elapsed = time.perf_counter() - batch_start_time
    final_index = write_index_file(index_path, camera_records)

    print("\n" + "=" * 60)
    print("AI City Batch Processing Complete!")
    print(f"Total time elapsed: {format_duration(total_elapsed)} ({total_elapsed:.2f}s)")
    print(f"Total cameras: {final_index['processed_cameras']}")
    print(f"Successful: {final_index['successful_cameras']}")
    print(f"Failed: {final_index['failed_cameras']}")
    print(f"Processing index written to: {index_path}")
    print(f"Logs written to: {log_file}")
    print("=" * 60)

    return 0 if final_index["failed_cameras"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
