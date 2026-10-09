"""
UrbanTrack AI — Production Layer 2 Tracklet Loader.

Coordinates the end-to-end ingestion pipeline:
- Loads camera configs and official synchronizer offsets.
- Streams and indexes observations per camera to avoid excessive peak memory.
- Performs deterministic joins, invariant checking, and normalization into CanonicalTracklets.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from layer2.ingestion.canonical_models import (
    ANPRData,
    AppearanceData,
    CanonicalTracklet,
    MotionData,
    QualityData,
    SpatialData,
    TemporalData,
    VehicleData,
)
from layer2.ingestion.observation_join import index_camera_observations, validate_and_extract_join
from layer2.ingestion.reid_compatibility import map_reid_compatibility_group
from layer2.ingestion.timestamp_sync import (
    TimestampSynchronizer,
    format_timestamp_seconds,
    parse_raw_timestamp,
)


class TrackletLoader:
    """
    Production-grade loader transforming frozen Layer 1 artifacts into CanonicalTracklets.
    """

    def __init__(
        self,
        handoff_root: Union[str, Path] = "UrbanTrack_Member1_Handoff 2",
        sync_meta_dir: Union[str, Path] = "data/cityflowv2/cam_timestamp",
    ) -> None:
        self.handoff_root = Path(handoff_root)
        self.output_dir = self.handoff_root / "data" / "output"
        self.config_dir = self.handoff_root / "data" / "config"
        self.sync_meta_dir = Path(sync_meta_dir)

        if not self.output_dir.is_dir():
            raise FileNotFoundError(f"Layer 1 output directory not found: {self.output_dir}")
        if not self.config_dir.is_dir():
            raise FileNotFoundError(f"Layer 1 config directory not found: {self.config_dir}")

        # Load synchronizer
        self.synchronizer = TimestampSynchronizer(primary_dir=self.sync_meta_dir)

        # Load static configuration maps
        with open(self.config_dir / "camera_locations.json", "r", encoding="utf-8") as f:
            self.camera_locations: Dict[str, Dict[str, Any]] = json.load(f)

        with open(self.config_dir / "camera_graph.json", "r", encoding="utf-8") as f:
            self.camera_graph: Dict[str, Any] = json.load(f)

        with open(self.config_dir / "aicity_manifest.json", "r", encoding="utf-8") as f:
            self.manifest: Dict[str, Any] = json.load(f)

    def load_camera_tracklets(
        self,
        camera_id: str,
    ) -> Tuple[List[CanonicalTracklet], Dict[str, Any]]:
        """
        Loads and canonicalizes all tracklets for a single camera.
        Memory for raw frame observations is released upon completion of the camera.
        """
        cam_dir = self.output_dir / camera_id
        if not cam_dir.is_dir():
            raise FileNotFoundError(f"Camera output directory not found: {cam_dir}")

        # Parse scenario
        parts = camera_id.split("_")
        if len(parts) != 3 or parts[0] != "CAM" or not parts[1].startswith("S"):
            raise ValueError(f"Malformed scenario-qualified camera ID: '{camera_id}'")
        scenario_id = parts[1]

        # Camera spatial metadata
        if camera_id not in self.camera_locations:
            raise KeyError(f"Camera {camera_id} missing from camera_locations.json")
        loc_record = self.camera_locations[camera_id]
        if loc_record.get("scenario") != scenario_id:
            raise ValueError(f"Scenario mismatch for {camera_id}: expected {scenario_id}, got {loc_record.get('scenario')}")

        cam_lat = float(loc_record["location"]["latitude"])
        cam_lon = float(loc_record["location"]["longitude"])
        cam_bearing = float(loc_record.get("direction", {}).get("bearing_deg", 0.0))
        cam_conf = loc_record.get("confidence", "HIGH")
        road_ctx = loc_record.get("road_context", {})

        # Perception summary for camera reliability and fps
        with open(cam_dir / "perception_summary.json", "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        cam_reliability = float(summary_data.get("average_camera_reliability", 1.0))
        cam_fps = float(summary_data.get("average_fps", 10.0))

        # Trajectories
        with open(cam_dir / "trajectories.json", "r", encoding="utf-8") as f:
            trajectories_data = json.load(f)

        # Observations
        with open(cam_dir / "observations.json", "r", encoding="utf-8") as f:
            observations_data = json.load(f)

        # Index observations by track_id
        track_obs_map, frame_timestamps = index_camera_observations(observations_data)

        # Clock offset for camera
        offset_sec = self.synchronizer.get_offset(scenario_id, camera_id)

        canonical_tracklets: List[CanonicalTracklet] = []
        cam_joined_obs = 0
        join_failures = []

        for traj in trajectories_data:
            tid = traj["track_id"]
            obs_list = track_obs_map.get(tid, [])

            try:
                join_res = validate_and_extract_join(
                    trajectory_record=traj,
                    vehicle_observations=obs_list,
                    camera_id=camera_id,
                    frame_timestamps=frame_timestamps,
                )
            except Exception as ex:
                join_failures.append({"camera_id": camera_id, "track_id": tid, "error": str(ex)})
                continue

            cam_joined_obs += join_res["observation_count"]

            # Temporal calculations
            start_raw_ts = join_res["start_raw_timestamp"]
            end_raw_ts = join_res["end_raw_timestamp"]
            start_raw_sec = parse_raw_timestamp(start_raw_ts)
            end_raw_sec = parse_raw_timestamp(end_raw_ts)

            start_sync_sec = start_raw_sec + offset_sec
            end_sync_sec = end_raw_sec + offset_sec
            start_sync_ts = format_timestamp_seconds(start_sync_sec)
            end_sync_ts = format_timestamp_seconds(end_sync_sec)
            duration_sec = end_sync_sec - start_sync_sec

            temporal = TemporalData(
                start_frame=traj["start_frame"],
                end_frame=traj["end_frame"],
                duration_frames=traj["duration_frames"],
                fps=cam_fps,
                start_raw_timestamp=start_raw_ts,
                end_raw_timestamp=end_raw_ts,
                start_raw_seconds=round(start_raw_sec, 4),
                end_raw_seconds=round(end_raw_sec, 4),
                start_sync_timestamp=start_sync_ts,
                end_sync_timestamp=end_sync_ts,
                start_sync_seconds=round(start_sync_sec, 4),
                end_sync_seconds=round(end_sync_sec, 4),
                duration_seconds=round(duration_sec, 4),
            )

            # Motion
            motion = MotionData(
                trajectory_pixels=traj.get("trajectory", []),
                trajectory_length=traj.get("trajectory_length", len(traj.get("trajectory", []))),
                average_velocity_px=round(float(traj.get("average_velocity_px", 0.0)), 2),
                direction=join_res["direction"],
            )

            # Appearance
            raw_emb = traj.get("appearance_embedding")
            has_emb = raw_emb is not None
            reid_model = traj.get("reid_model", "unknown")
            compat_grp = map_reid_compatibility_group(reid_model, has_emb)
            emb_quality = float(traj["embedding_quality"]) if traj.get("embedding_quality") is not None else None
            emb_dim = int(traj["embedding_dim"]) if has_emb and traj.get("embedding_dim") is not None else (len(raw_emb) if has_emb else None)

            appearance = AppearanceData(
                has_embedding=has_emb,
                appearance_embedding=raw_emb,
                embedding_dim=emb_dim,
                embedding_quality=round(emb_quality, 4) if emb_quality is not None else None,
                reid_model=reid_model,
                reid_compatibility_group=compat_grp,
            )

            # Vehicle
            vehicle = VehicleData(
                vehicle_type=traj.get("vehicle_type", "car"),
                average_detector_confidence=join_res["average_detector_confidence"],
            )

            # ANPR
            anpr: ANPRData = join_res["anpr"]

            # Spatial
            spatial = SpatialData(
                camera_latitude=cam_lat,
                camera_longitude=cam_lon,
                camera_bearing_deg=cam_bearing,
                camera_confidence=cam_conf,
                road_context=road_ctx,
                projected_vehicle_coordinates=None,
            )

            # Quality and missing evidence
            missing_evidence = []
            if not has_emb:
                missing_evidence.append("APPEARANCE_EMBEDDING")
            if not anpr.has_plate_detection:
                missing_evidence.append("PLATE_DETECTION")
            elif not anpr.has_readable_ocr:
                missing_evidence.append("READABLE_OCR")
            if not motion.direction or motion.direction == "stationary":
                missing_evidence.append("MOTION_DIRECTION")

            quality = QualityData(
                camera_reliability=round(cam_reliability, 4),
                missing_evidence=missing_evidence,
            )

            canonical_tracklet = CanonicalTracklet(
                scenario_id=scenario_id,
                camera_id=camera_id,
                track_id=tid,
                global_vehicle_id=None,
                temporal=temporal,
                motion=motion,
                appearance=appearance,
                vehicle=vehicle,
                anpr=anpr,
                spatial=spatial,
                quality=quality,
            )
            canonical_tracklets.append(canonical_tracklet)

        meta = {
            "camera_id": camera_id,
            "scenario_id": scenario_id,
            "tracks_loaded": len(canonical_tracklets),
            "observations_joined": cam_joined_obs,
            "join_failures": join_failures,
        }
        return canonical_tracklets, meta

    def load_all_tracklets(
        self,
        progress_interval: int = 10,
    ) -> Tuple[List[CanonicalTracklet], Dict[str, Any]]:
        """
        Loads and normalizes tracklets across all 65 scenario-qualified cameras.
        """
        start_time = time.time()
        camera_dirs = sorted([d.name for d in self.output_dir.iterdir() if d.is_dir()])
        total_cams = len(camera_dirs)

        all_tracklets: List[CanonicalTracklet] = []
        total_obs_joined = 0
        all_join_failures = []
        per_camera_metrics = {}

        for idx, cam_id in enumerate(camera_dirs):
            tracklets, cam_meta = self.load_camera_tracklets(cam_id)
            all_tracklets.extend(tracklets)
            total_obs_joined += cam_meta["observations_joined"]
            if cam_meta["join_failures"]:
                all_join_failures.extend(cam_meta["join_failures"])
            per_camera_metrics[cam_id] = cam_meta

            if (idx + 1) % progress_interval == 0 or idx == total_cams - 1:
                elapsed = time.time() - start_time
                print(f"  [Loader] Processed {idx + 1}/{total_cams} cameras ({len(all_tracklets)} tracklets, {total_obs_joined} obs) in {elapsed:.2f}s")

        elapsed_total = time.time() - start_time

        ingestion_metadata = {
            "provenance": {
                "source_layer1_root": str(self.handoff_root),
                "timestamp_metadata_root": str(self.sync_meta_dir),
                "contract_version": "1.0.0",
                "generation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
            "scale": {
                "total_cameras": total_cams,
                "total_tracklets": len(all_tracklets),
                "total_observations_joined": total_obs_joined,
                "runtime_seconds": round(elapsed_total, 3),
            },
            "join_failures": all_join_failures,
            "per_camera_metrics": per_camera_metrics,
        }

        return all_tracklets, ingestion_metadata
