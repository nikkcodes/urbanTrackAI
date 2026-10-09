#!/usr/bin/env python3
"""
UrbanTrack AI — Controlled Experiment: C002 AI City Vehicle Re-ID Extraction.

Re-extracts vehicle appearance Re-ID embeddings for CAM_S01_C002 using the
fine-tuned AI City Challenge OSNet model checkpoint (osnet_x0_25_aicity_best.pth)
on the unannotated clean source video (vdo.avi).

CRITICAL CONSTRAINTS & DATA INTEGRITY:
1. Frozen baseline inputs are strictly read-only. Baseline SHA256 hashes are recorded
   before and verified after extraction to guarantee immutability.
2. Uses ONLY the unannotated clean video source (no tracking overlays).
3. First-appearance frames and bounding boxes are strictly sourced from frozen
   Member 1 outputs. No bbox modification, no track ID alteration.
4. Output is written strictly to results/experiments/c002_aicity_reid/.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import logging
import math
from pathlib import Path
import platform
import sys
import types
from typing import Any, Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("extract_c002_aicity_reid")

# Default authoritative file paths
DEFAULT_CLEAN_VIDEO = Path(
    "/Users/yanalavivekreddy/Downloads/AICity22_Track1_MTMC_Tracking/train/S01/c002/vdo.avi"
)
DEFAULT_OBSERVATIONS = Path(
    "UrbanTrack_Member1_Handoff/output/CAM_S01_C002/observations.json"
)
DEFAULT_TRAJECTORIES = Path(
    "UrbanTrack_Member1_Handoff/output/CAM_S01_C002/trajectories.json"
)
DEFAULT_CAMERA_METRICS = Path(
    "UrbanTrack_Member1_Handoff/output/CAM_S01_C002/camera_metrics.json"
)
DEFAULT_PERCEPTION_SUMMARY = Path(
    "UrbanTrack_Member1_Handoff/output/CAM_S01_C002/perception_summary.json"
)
DEFAULT_CHECKPOINT = Path(
    "UrbanTrack_Member1_Handoff/osnet_x0_25_aicity_best.pth"
)
DEFAULT_EXTRACTOR_SCRIPT = Path(
    "UrbanTrack_Member1_Handoff/reid_extractor.py"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/experiments/c002_aicity_reid/output/CAM_S01_C002"
)
DEFAULT_MANIFEST = Path(
    "results/experiments/c002_aicity_reid/experiment_manifest.json"
)


def compute_sha256(file_path: Path) -> str:
    """Compute the SHA256 cryptographic digest of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def load_member1_reid_extractor(
    extractor_script_path: Path,
    checkpoint_path: Path,
    device: Optional[str] = None,
) -> Any:
    """
    Dynamically import and initialize Member 1's ReIDExtractor using the
    fine-tuned AI City checkpoint without modifying existing project files.
    """
    if not extractor_script_path.is_file():
        raise FileNotFoundError(
            f"Member 1 ReID extractor script not found: {extractor_script_path}"
        )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"AI City checkpoint not found: {checkpoint_path}"
        )

    # 1. Set up simulated config package matching Member 1 expectation
    mock_config = types.ModuleType("config")
    mock_config.MIN_REID_CROP_SIZE = 32
    mock_config.REID_EMBEDDING_DIM = 512
    mock_config.REID_MODEL_NAME = "osnet_x0_25"
    mock_config.REID_MODEL_WEIGHTS = "aicity"

    mock_package = types.ModuleType("UrbanTrack_Member1_Handoff")
    mock_package.config = mock_config
    sys.modules["UrbanTrack_Member1_Handoff"] = mock_package
    sys.modules["UrbanTrack_Member1_Handoff.config"] = mock_config

    # 2. Load module spec from file
    spec = importlib.util.spec_from_file_location(
        "UrbanTrack_Member1_Handoff.reid_extractor",
        str(extractor_script_path.resolve()),
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec from {extractor_script_path}")

    mod = importlib.util.module_from_spec(spec)

    # Point extractor checkpoint path to the verified AI City checkpoint
    mod.AICITY_CHECKPOINT_PATH = checkpoint_path.resolve()
    spec.loader.exec_module(mod)

    # 3. Instantiate ReIDExtractor
    extractor = mod.ReIDExtractor()

    if device is not None and hasattr(extractor, "device"):
        extractor.device = device

    logger.info(
        f"ReIDExtractor initialized successfully: model={extractor.reid_model}, "
        f"dim={extractor.embedding_dim}, device={extractor.device}"
    )
    return extractor


def run_c002_reid_extraction(
    video_path: Path = DEFAULT_CLEAN_VIDEO,
    observations_path: Path = DEFAULT_OBSERVATIONS,
    trajectories_path: Path = DEFAULT_TRAJECTORIES,
    camera_metrics_path: Path = DEFAULT_CAMERA_METRICS,
    perception_summary_path: Path = DEFAULT_PERCEPTION_SUMMARY,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    extractor_script_path: Path = DEFAULT_EXTRACTOR_SCRIPT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute controlled Re-ID embedding re-extraction on clean C002 video.
    """
    try:
        import cv2
    except ImportError as e:
        raise RuntimeError(
            f"OpenCV (cv2) is required for extraction. Run inside .venv_c002_reid: {e}"
        )

    try:
        import torch
        import torchreid
    except ImportError as e:
        raise RuntimeError(
            f"PyTorch and TorchReID are required for extraction. Run inside .venv_c002_reid: {e}"
        )

    # -------------------------------------------------------------------------
    # STEP 1: Verify all input files exist
    # -------------------------------------------------------------------------
    input_files = {
        "video": video_path,
        "observations": observations_path,
        "trajectories": trajectories_path,
        "checkpoint": checkpoint_path,
        "extractor": extractor_script_path,
    }
    for name, path in input_files.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required input '{name}' not found at: {path}")

    # -------------------------------------------------------------------------
    # STEP 2: Record baseline hashes BEFORE extraction (Safety Rule 1 & 10)
    # -------------------------------------------------------------------------
    baseline_hashes_before = {
        "observations": compute_sha256(observations_path),
        "trajectories": compute_sha256(trajectories_path),
        "checkpoint": compute_sha256(checkpoint_path),
        "extractor": compute_sha256(extractor_script_path),
    }
    logger.info("Recorded SHA256 digests of frozen baseline inputs before execution.")

    # -------------------------------------------------------------------------
    # STEP 3: Load frozen Member 1 perception outputs
    # -------------------------------------------------------------------------
    with open(observations_path, "r", encoding="utf-8") as f:
        frozen_obs = json.load(f)

    with open(trajectories_path, "r", encoding="utf-8") as f:
        frozen_trajs = json.load(f)

    total_tracks = len(frozen_trajs)
    logger.info(f"Loaded {total_tracks} frozen trajectories from {trajectories_path}")

    # Build observation index: (frame_number, track_id) -> vehicle_dict
    obs_index: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for frame in frozen_obs.get("frames", []):
        f_num = frame.get("frame_number")
        for veh in frame.get("vehicles", []):
            t_id = veh.get("track_id")
            if f_num is not None and t_id is not None:
                obs_index[(f_num, t_id)] = veh

    # -------------------------------------------------------------------------
    # STEP 4: Build frame-indexed extraction requests
    # -------------------------------------------------------------------------
    requests_by_frame: Dict[int, List[Dict[str, Any]]] = {}
    missing_obs_tracks: List[int] = []

    for traj in frozen_trajs:
        track_id = traj.get("track_id")
        start_frame = traj.get("start_frame")

        veh_obs = obs_index.get((start_frame, track_id))
        if veh_obs is None:
            missing_obs_tracks.append(track_id)
            continue

        bbox = veh_obs.get("bbox")
        if not bbox or len(bbox) != 4:
            missing_obs_tracks.append(track_id)
            continue

        requests_by_frame.setdefault(start_frame, []).append(
            {
                "track_id": track_id,
                "bbox": bbox,
                "vehicle_type": veh_obs.get("vehicle_type", "vehicle"),
            }
        )

    logger.info(
        f"Queued extraction for {total_tracks - len(missing_obs_tracks)} tracks "
        f"across {len(requests_by_frame)} unique start frames."
    )

    # -------------------------------------------------------------------------
    # STEP 5: Initialize Member 1 ReIDExtractor
    # -------------------------------------------------------------------------
    extractor = load_member1_reid_extractor(
        extractor_script_path=extractor_script_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # -------------------------------------------------------------------------
    # STEP 6: Execute video decoding & crop extraction
    # -------------------------------------------------------------------------
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open clean source video: {video_path}")

    max_target_frame = max(requests_by_frame.keys()) if requests_by_frame else -1
    current_frame_idx = 0

    results: Dict[int, Dict[str, Any]] = {}
    failures: Dict[int, str] = {}

    for t_id in missing_obs_tracks:
        failures[t_id] = "No matching observation found at start_frame"

    logger.info("Beginning single-pass sequential video frame decoding...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame_idx in requests_by_frame:
            frame_h, frame_w = frame.shape[:2]

            for req in requests_by_frame[current_frame_idx]:
                track_id = req["track_id"]
                bbox = req["bbox"]

                x1, y1, x2, y2 = [int(round(coord)) for coord in bbox]
                x1 = max(0, min(x1, frame_w))
                x2 = max(0, min(x2, frame_w))
                y1 = max(0, min(y1, frame_h))
                y2 = max(0, min(y2, frame_h))

                crop = frame[y1:y2, x1:x2]
                crop_h, crop_w = crop.shape[:2]

                # Semantic Gate: crop width >= 32 and height >= 32
                if crop_w < 32 or crop_h < 32:
                    failures[track_id] = (
                        f"Crop dimensions ({crop_w}x{crop_h}) below minimum gate 32x32"
                    )
                    continue

                try:
                    emb, quality = extractor.extract(crop)
                except Exception as extract_err:
                    failures[track_id] = f"Extractor raised exception: {extract_err}"
                    continue

                if emb is None:
                    failures[track_id] = "Extractor returned None"
                    continue

                # Invariant checks for embedding validity
                if len(emb) != 512:
                    failures[track_id] = f"Invalid embedding dimension: {len(emb)} (expected 512)"
                    continue

                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in emb):
                    failures[track_id] = "Embedding contains non-finite values"
                    continue

                l2_norm = math.sqrt(sum(v * v for v in emb))
                if abs(l2_norm - 1.0) > 1e-3:
                    failures[track_id] = f"Embedding L2 norm deviates from 1.0: {l2_norm:.6f}"
                    continue

                results[track_id] = {
                    "appearance_embedding": [float(v) for v in emb],
                    "embedding_quality": float(quality) if quality is not None else 1.0,
                    "crop_width": crop_w,
                    "crop_height": crop_h,
                }

        current_frame_idx += 1
        if current_frame_idx > max_target_frame:
            break

    cap.release()
    logger.info(
        f"Extraction pass completed. Successful: {len(results)}, Failed: {len(failures)}"
    )

    # -------------------------------------------------------------------------
    # STEP 7: Construct updated experiment copies
    # -------------------------------------------------------------------------
    # 7a. Updated trajectories.json
    new_trajs: List[Dict[str, Any]] = []
    for traj in frozen_trajs:
        traj_copy = copy.deepcopy(traj)
        t_id = traj_copy["track_id"]

        traj_copy["reid_model"] = "osnet_x0_25_aicity"
        traj_copy["embedding_dim"] = 512

        if t_id in results:
            traj_copy["appearance_embedding"] = results[t_id]["appearance_embedding"]
            traj_copy["embedding_quality"] = results[t_id]["embedding_quality"]
        else:
            traj_copy["appearance_embedding"] = None
            traj_copy["embedding_quality"] = None

        new_trajs.append(traj_copy)

    # 7b. Updated observations.json
    new_obs = copy.deepcopy(frozen_obs)
    for frame in new_obs.get("frames", []):
        prov = frame.get("provenance")
        if isinstance(prov, dict):
            prov["reid_model"] = "osnet_x0_25_aicity"
        for veh in frame.get("vehicles", []):
            veh["reid_model"] = "osnet_x0_25_aicity"
            veh["embedding_dim"] = 512

    # 7c. Updated perception_summary.json (if present)
    new_summary = None
    if perception_summary_path.is_file():
        with open(perception_summary_path, "r", encoding="utf-8") as f:
            new_summary = json.load(f)
        new_summary["reid_model"] = "osnet_x0_25_aicity"
        new_summary["embeddings_generated"] = len(results)
        new_summary["embedding_generation_failures"] = len(failures)

    # 7d. Camera metrics copy
    camera_metrics_data = None
    if camera_metrics_path.is_file():
        with open(camera_metrics_path, "r", encoding="utf-8") as f:
            camera_metrics_data = json.load(f)

    # -------------------------------------------------------------------------
    # STEP 8: Write all output files to isolated experiment directory
    # -------------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    out_trajs_file = output_dir / "trajectories.json"
    out_obs_file = output_dir / "observations.json"
    out_summary_file = output_dir / "perception_summary.json"
    out_metrics_file = output_dir / "camera_metrics.json"

    with open(out_trajs_file, "w", encoding="utf-8") as f:
        json.dump(new_trajs, f, indent=2)

    with open(out_obs_file, "w", encoding="utf-8") as f:
        json.dump(new_obs, f, indent=2)

    if new_summary is not None:
        with open(out_summary_file, "w", encoding="utf-8") as f:
            json.dump(new_summary, f, indent=2)

    if camera_metrics_data is not None:
        with open(out_metrics_file, "w", encoding="utf-8") as f:
            json.dump(camera_metrics_data, f, indent=2)

    logger.info(f"Wrote experiment outputs to: {output_dir}")

    # -------------------------------------------------------------------------
    # STEP 9: Verify baseline immutability AFTER execution (Safety Rule 1 & 10)
    # -------------------------------------------------------------------------
    baseline_hashes_after = {
        "observations": compute_sha256(observations_path),
        "trajectories": compute_sha256(trajectories_path),
        "checkpoint": compute_sha256(checkpoint_path),
        "extractor": compute_sha256(extractor_script_path),
    }

    for key, hash_before in baseline_hashes_before.items():
        hash_after = baseline_hashes_after[key]
        if hash_before != hash_after:
            raise AssertionError(
                f"SAFETY VIOLATION: Baseline file '{key}' was altered! "
                f"Before: {hash_before}, After: {hash_after}"
            )

    logger.info("Baseline immutability verified: all frozen input SHA256 hashes are identical.")

    # -------------------------------------------------------------------------
    # STEP 10: Validation checks on generated experiment output
    # -------------------------------------------------------------------------
    assert len(new_trajs) == total_tracks, (
        f"Track count mismatch: {len(new_trajs)} != {total_tracks}"
    )

    for orig, new in zip(frozen_trajs, new_trajs):
        assert orig["track_id"] == new["track_id"], "Track ID mismatch"
        assert orig["start_frame"] == new["start_frame"], "Start frame mismatch"
        assert orig["end_frame"] == new["end_frame"], "End frame mismatch"
        assert orig["duration_frames"] == new["duration_frames"], "Duration mismatch"
        assert orig["trajectory"] == new["trajectory"], "Trajectory coords mismatch"
        assert orig["vehicle_type"] == new["vehicle_type"], "Vehicle type mismatch"
        assert new["reid_model"] == "osnet_x0_25_aicity", "Model tag mismatch"
        assert new["embedding_dim"] == 512, "Dimension mismatch"

        if new["appearance_embedding"] is not None:
            emb = new["appearance_embedding"]
            assert len(emb) == 512, "Embedding length != 512"
            assert all(math.isfinite(v) for v in emb), "Embedding contains non-finite values"
            norm = math.sqrt(sum(v * v for v in emb))
            assert abs(norm - 1.0) < 1e-3, f"L2 norm != 1.0: {norm}"

    # Verify observations preservation
    orig_frames = frozen_obs.get("frames", [])
    new_frames = new_obs.get("frames", [])
    assert len(orig_frames) == len(new_frames), "Frame count mismatch in observations"

    # -------------------------------------------------------------------------
    # STEP 11: Write Experiment Manifest
    # -------------------------------------------------------------------------
    manifest = {
        "experiment_name": "c002_aicity_reid_extraction",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_video_path": str(video_path.resolve()),
        "source_observations_path": str(observations_path.resolve()),
        "source_trajectories_path": str(trajectories_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "model_name": "osnet_x0_25",
        "reid_model": "osnet_x0_25_aicity",
        "embedding_dim": 512,
        "extraction_semantics": (
            "First-appearance frame crop frame[y1:y2, x1:x2], "
            "gate min_size >= 32x32, 256x128 bilinear resize, "
            "ImageNet normalization, 512-D L2-normalized vector"
        ),
        "total_c002_tracks": total_tracks,
        "successful_embeddings": len(results),
        "failed_embeddings": len(failures),
        "failed_track_ids": sorted(list(failures.keys())),
        "failure_reasons": failures,
        "environment_metadata": {
            "python_version": platform.python_version(),
            "torch_version": getattr(sys.modules.get("torch"), "__version__", "unknown"),
            "torchvision_version": getattr(sys.modules.get("torchvision"), "__version__", "unknown"),
            "torchreid_version": getattr(sys.modules.get("torchreid"), "__version__", "0.2.5"),
            "opencv_version": cv2.__version__,
            "device": getattr(extractor, "device", "cpu"),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "baseline_verification": {
            "immutability_verified": True,
            "baseline_sha256": baseline_hashes_before,
        },
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Experiment manifest successfully written to: {manifest_path}")

    return {
        "status": "SUCCESS",
        "total_tracks": total_tracks,
        "successful_embeddings": len(results),
        "failed_embeddings": len(failures),
        "manifest_path": str(manifest_path),
        "output_dir": str(output_dir),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract C002 AI City Re-ID embeddings in an isolated experiment."
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_CLEAN_VIDEO,
        help="Path to clean unannotated C002 vdo.avi video.",
    )
    parser.add_argument(
        "--observations",
        type=Path,
        default=DEFAULT_OBSERVATIONS,
        help="Path to frozen C002 observations.json.",
    )
    parser.add_argument(
        "--trajectories",
        type=Path,
        default=DEFAULT_TRAJECTORIES,
        help="Path to frozen C002 trajectories.json.",
    )
    parser.add_argument(
        "--camera-metrics",
        type=Path,
        default=DEFAULT_CAMERA_METRICS,
        help="Path to frozen C002 camera_metrics.json.",
    )
    parser.add_argument(
        "--perception-summary",
        type=Path,
        default=DEFAULT_PERCEPTION_SUMMARY,
        help="Path to frozen C002 perception_summary.json.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to osnet_x0_25_aicity_best.pth checkpoint.",
    )
    parser.add_argument(
        "--extractor-script",
        type=Path,
        default=DEFAULT_EXTRACTOR_SCRIPT,
        help="Path to Member 1 reid_extractor.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for CAM_S01_C002 experiment output.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Destination path for experiment_manifest.json.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference ('cpu', 'cuda', 'mps'). Auto-selects if None.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Initializing C002 AI City Re-ID Extraction Experiment...")
    res = run_c002_reid_extraction(
        video_path=args.video_path,
        observations_path=args.observations,
        trajectories_path=args.trajectories,
        camera_metrics_path=args.camera_metrics,
        perception_summary_path=args.perception_summary,
        checkpoint_path=args.checkpoint,
        extractor_script_path=args.extractor_script,
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        device=args.device,
    )
    logger.info(f"Execution finished: {res}")


if __name__ == "__main__":
    main()
