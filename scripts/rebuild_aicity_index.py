from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


MANIFEST_PATH = Path("data/config/aicity_manifest.json")
OUTPUT_DIR = Path("data/output")
INDEX_PATH = OUTPUT_DIR / "aicity_index.json"


def normalize_rel_path(value):
    if value is None:
        return None
    return str(value).replace("\\", "/")


def main():
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    cameras = manifest.get("cameras", [])

    if len(cameras) != 65:
        raise RuntimeError(
            f"Expected 65 manifest cameras, found {len(cameras)}."
        )

    records = []

    for entry in cameras:
        camera_id = entry["camera_id"]
        summary_path = OUTPUT_DIR / camera_id / "perception_summary.json"

        if not summary_path.exists():
            raise RuntimeError(
                f"Missing perception summary: {summary_path}"
            )

        with summary_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)

        if summary.get("camera_id") != camera_id:
            raise RuntimeError(
                f"Camera ID mismatch for {camera_id}: "
                f"summary contains {summary.get('camera_id')}"
            )

        record = {
            "camera_id": camera_id,
            "scene": entry.get("scene"),
            "camera": entry.get("camera"),
            "split": entry.get("split"),
            "status": "success",
            "video_path_relative": normalize_rel_path(
                entry.get("video_path_relative")
            ),
            "ground_truth_path_relative": normalize_rel_path(
                entry.get("ground_truth_path_relative")
            ),
            "calibration_path_relative": normalize_rel_path(
                entry.get("calibration_path_relative")
            ),
            "ground_truth_available": bool(
                entry.get("ground_truth_path_relative")
            ),
            "calibration_available": bool(
                entry.get("calibration_path_relative")
            ),
            "frames_processed": summary.get("frames_processed"),
            "unique_tracks": summary.get("unique_tracks"),
            "total_vehicle_observations": summary.get(
                "total_vehicle_observations"
            ),
            "plates_detected": summary.get("plates_detected"),
            "ocr_success_count": summary.get("ocr_success_count"),
            "embeddings_generated": summary.get("embeddings_generated"),
            "embedding_generation_failures": summary.get(
                "embedding_generation_failures"
            ),
            "processing_device": summary.get("processing_device"),
            "reid_model": summary.get("reid_model"),
            "embedding_dimension": summary.get("embedding_dimension"),
            "processing_timestamp": summary.get("processing_timestamp"),
        }

        records.append(record)

    if len(records) != 65:
        raise RuntimeError(
            f"Safety check failed: recovered {len(records)} records."
        )

    camera_ids = [r["camera_id"] for r in records]

    if len(set(camera_ids)) != 65:
        raise RuntimeError(
            "Safety check failed: duplicate camera IDs detected."
        )

    successful = sum(
        1 for r in records if r["status"] == "success"
    )

    failed = sum(
        1 for r in records if r["status"] == "failed"
    )

    index = {
        "dataset": "AI City Challenge 2022 Track 1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "processed_cameras": len(records),
        "successful_cameras": successful,
        "failed_cameras": failed,
        "cameras": records,
    }

    # Backup current index before replacing it.
    if INDEX_PATH.exists():
        backup_path = INDEX_PATH.with_suffix(".json.bak")
        INDEX_PATH.replace(backup_path)
        print(f"Previous index backed up to: {backup_path}")

    with INDEX_PATH.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

    print("=" * 60)
    print("AI City index reconstruction complete")
    print(f"Recovered cameras: {len(records)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Index: {INDEX_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()