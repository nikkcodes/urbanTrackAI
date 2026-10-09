"""
UrbanTrack AI — Execution Runner for Layer 2 Canonical Ingestion.

Executes TrackletLoader across all 65 cameras, runs validation,
and serializes artifacts into results/layer2_ingestion/.
"""

import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from layer2.ingestion.tracklet_loader import TrackletLoader
from layer2.ingestion.validation import validate_canonical_tracklets


def get_peak_memory_mb() -> float:
    """Returns peak memory in megabytes for the current process on macOS."""
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    # On macOS ru_maxrss is in bytes
    return rusage.ru_maxrss / (1024.0 * 1024.0)


def run_full_ingestion() -> int:
    print("=" * 70)
    print(" UrbanTrack AI: Layer 2 Canonical Ingestion & Normalization")
    print("=" * 70)

    out_dir = Path("results/layer2_ingestion")
    out_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    mem_before = get_peak_memory_mb()

    loader = TrackletLoader(
        handoff_root="UrbanTrack_Member1_Handoff 2",
        sync_meta_dir="data/cityflowv2/cam_timestamp",
    )

    print(f"\n[1/3] Loading and canonicalizing all 65 cameras...")
    tracklets, ingestion_meta = loader.load_all_tracklets(progress_interval=10)

    total_tracklets = len(tracklets)
    elapsed_load = time.time() - start_time
    print(f"  -> Successfully loaded {total_tracklets} tracklets in {elapsed_load:.2f}s")

    print("\n[2/3] Running machine-checkable invariant validation suite...")
    all_passed, validation_report = validate_canonical_tracklets(
        tracklets=tracklets,
        expected_camera_count=65,
        expected_tracklet_count=7448,
    )

    print(f"  -> Invariant Validation Status: {validation_report['overall_status']}")
    for inv_name, inv_res in validation_report["invariants"].items():
        status_tag = "PASS" if inv_res["passed"] else "FAIL"
        print(f"     [{status_tag}] {inv_name}")

    if not all_passed:
        print("\n[ERROR] Invariant Validation Errors:")
        for err in validation_report["errors"]:
            print(f"  - {err}")

    print("\n[3/3] Serializing Canonical Ingestion Artifacts...")

    # 1. canonical_tracklets.json
    tracklets_dict = [t.to_dict() for t in tracklets]
    tracklets_file = out_dir / "canonical_tracklets.json"
    with open(tracklets_file, "w", encoding="utf-8") as f:
        json.dump(tracklets_dict, f)
    print(f"  -> Exported canonical tracklets: {tracklets_file} ({tracklets_file.stat().st_size / (1024*1024):.1f} MB)")

    # 2. ingestion_validation.json
    val_file = out_dir / "ingestion_validation.json"
    with open(val_file, "w", encoding="utf-8") as f:
        json.dump(validation_report, f, indent=2)
    print(f"  -> Exported validation report: {val_file}")

    # Compute detailed statistics for summary
    scenario_counts = {}
    reid_groups = {}
    embeddings_present = 0
    embeddings_null = 0
    plate_detections = 0
    readable_ocr = 0

    for t in tracklets:
        sc = t.scenario_id
        scenario_counts[sc] = scenario_counts.get(sc, 0) + 1
        grp = t.appearance.reid_compatibility_group
        reid_groups[grp] = reid_groups.get(grp, 0) + 1

        if t.appearance.has_embedding:
            embeddings_present += 1
        else:
            embeddings_null += 1

        if t.anpr.has_plate_detection:
            plate_detections += 1
        if t.anpr.has_readable_ocr:
            readable_ocr += 1

    total_time = time.time() - start_time
    peak_mem = get_peak_memory_mb()

    # 3. ingestion_summary.json
    summary_data = {
        "status": "READY" if all_passed else "BLOCKED",
        "provenance": ingestion_meta["provenance"],
        "scale": {
            "total_cameras": ingestion_meta["scale"]["total_cameras"],
            "total_tracklets": total_tracklets,
            "total_observations_joined": ingestion_meta["scale"]["total_observations_joined"],
            "runtime_seconds": round(total_time, 2),
            "peak_memory_mb": round(peak_mem, 2),
        },
        "breakdown": {
            "scenario_tracklet_counts": scenario_counts,
            "reid_compatibility_groups": reid_groups,
            "embeddings_present": embeddings_present,
            "embeddings_null": embeddings_null,
            "plate_detections_count": plate_detections,
            "readable_ocr_count": readable_ocr,
        },
        "synchronization": {
            "cameras_synchronized": ingestion_meta["scale"]["total_cameras"],
            "offsets_applied": True,
            "heterogeneous_fps_handled": True,
        },
        "join_failures_count": len(ingestion_meta["join_failures"]),
        "join_failures": ingestion_meta["join_failures"],
        "validation_status": validation_report["overall_status"],
    }

    summary_file = out_dir / "ingestion_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
    print(f"  -> Exported ingestion summary: {summary_file}")

    print("\n" + "=" * 70)
    print(f" Ingestion Completed in {total_time:.2f}s | Peak Memory: {peak_mem:.1f} MB")
    print(f" Final Verdict: LAYER2_INGESTION = {'READY' if all_passed else 'BLOCKED'}")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_full_ingestion())
