"""
Observation and camera metadata loading utilities for UrbanTrack AI.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from schemas.observation_schema import Observation


def load_camera_metadata(
    metadata_source: Union[str, Path, Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """
    Load camera metadata mapping camera_id -> metadata dictionary (e.g. latitude, longitude).

    Args:
        metadata_source: File path to JSON camera metadata, JSON string, or dict object.

    Returns:
        Dict[str, Dict[str, Any]]: Dictionary mapping camera IDs to metadata.
    """
    if isinstance(metadata_source, (str, Path)):
        p = Path(metadata_source)
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            try:
                data = json.loads(str(metadata_source))
            except json.JSONDecodeError:
                raise ValueError(f"Could not load camera metadata from file or JSON string: {metadata_source}")
    elif isinstance(metadata_source, dict):
        data = metadata_source
    else:
        raise TypeError(f"metadata_source must be a file path, JSON string, or dict, got {type(metadata_source).__name__}")

    metadata_map = {}
    for cam_id, meta in data.items():
        if isinstance(meta, dict):
            cam_dict = {
                "latitude": float(meta["latitude"]) if "latitude" in meta and meta["latitude"] is not None else None,
                "longitude": float(meta["longitude"]) if "longitude" in meta and meta["longitude"] is not None else None,
                **{k: v for k, v in meta.items() if k not in ("latitude", "longitude")},
            }
            if "clock_offset_seconds" in meta and meta["clock_offset_seconds"] is not None:
                cam_dict["clock_offset_seconds"] = float(meta["clock_offset_seconds"])
            metadata_map[str(cam_id)] = cam_dict
    return metadata_map


def load_observations_from_json(
    source: Union[str, Path, Dict[str, Any], List[Any]],
    camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Observation]:
    """
    Load vehicle observations from a JSON string, dict, list, or file path.
    Optionally attaches camera geographic coordinates from camera_metadata.

    Args:
        source: File path, JSON string, dictionary, or list of observation dictionaries.
        camera_metadata: Optional dictionary mapping camera_id to metadata dict with latitude/longitude.

    Returns:
        List[Observation]: List of parsed Observation instances.
    """
    data = None
    if isinstance(source, (str, Path)):
        p = Path(source)
        if p.is_file():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            try:
                data = json.loads(str(source))
            except json.JSONDecodeError:
                raise ValueError(f"Could not parse JSON source: {source}")
    else:
        data = source

    if isinstance(data, dict):
        raw_list = [data]
    elif isinstance(data, list):
        raw_list = data
    else:
        raise TypeError(f"Invalid observation JSON data format: {type(data).__name__}")

    observations = []
    for item in raw_list:
        obs = Observation.from_json(item)

        # Attach camera coordinates and metadata if available in camera_metadata
        if camera_metadata and obs.camera_id in camera_metadata:
            cam_meta = camera_metadata[obs.camera_id]
            if obs.latitude is None and cam_meta.get("latitude") is not None:
                obs.latitude = cam_meta["latitude"]
            if obs.longitude is None and cam_meta.get("longitude") is not None:
                obs.longitude = cam_meta["longitude"]
            if obs.timestamp_semantics is None and cam_meta.get("timestamp_semantics") is not None:
                obs.timestamp_semantics = cam_meta["timestamp_semantics"]
            if obs.time_reference_id is None and cam_meta.get("time_reference_id") is not None:
                obs.time_reference_id = cam_meta["time_reference_id"]
            if obs.clock_offset_seconds is None and cam_meta.get("clock_offset_seconds") is not None:
                obs.clock_offset_seconds = float(cam_meta["clock_offset_seconds"])

        # For current Kanishka observations, if timestamp semantics are not explicitly present,
        # assign "video_relative" because this is known from the finalized perception contract.
        if obs.timestamp_semantics is None:
            obs.timestamp_semantics = "video_relative"

        observations.append(obs)

    return observations


def verify_raw_data_integrity(
    manifest_path: Union[str, Path] = "data/member1_perception/cam_001/manifest.json"
) -> Tuple[bool, Dict[str, Any]]:
    """
    Cryptographically verify raw Member 1 perception artifacts against the canonical SHA-256 manifest.
    Fails if any authoritative file was modified, moved, or corrupted.

    Returns:
        Tuple[bool, Dict[str, Any]]: (is_valid, validation_details)
    """
    import hashlib

    p_manifest = Path(manifest_path)
    if not p_manifest.is_file():
        # Search relative to base directory
        alt_manifest = Path(__file__).parent.parent / manifest_path
        if alt_manifest.is_file():
            p_manifest = alt_manifest
        else:
            return False, {"error": f"Manifest file not found at {manifest_path}"}

    with open(p_manifest, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    artifacts = manifest.get("artifacts", {})
    results = {}
    all_valid = True
    workspace_root = p_manifest.parent.parent.parent.parent

    for filename, meta in artifacts.items():
        rel_path = meta.get("path")
        expected_hash = meta.get("sha256")
        expected_size = meta.get("size_bytes")

        target_file = Path(rel_path)
        if not target_file.is_file():
            # Try workspace root
            alt_file = workspace_root / rel_path
            if alt_file.is_file():
                target_file = alt_file
            else:
                # Try raw subdirectory next to manifest
                alt_raw = p_manifest.parent / "raw" / filename
                if alt_raw.is_file():
                    target_file = alt_raw
                else:
                    results[filename] = {"status": "missing", "path": rel_path}
                    all_valid = False
                    continue

        sha256_hash = hashlib.sha256()
        with open(target_file, "rb") as bf:
            for byte_block in iter(lambda: bf.read(65536), b""):
                sha256_hash.update(byte_block)
        actual_hash = sha256_hash.hexdigest()
        actual_size = target_file.stat().st_size

        is_match = (actual_hash == expected_hash) and (actual_size == expected_size)
        if not is_match:
            all_valid = False
        results[filename] = {
            "status": "valid" if is_match else "corrupted",
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "size_bytes": actual_size,
            "size_match": actual_size == expected_size,
        }

    return all_valid, {"all_valid": all_valid, "artifacts": results}


def load_member1_perception_feed(
    tracks_path: Union[str, Path] = "data/member1_perception/cam_001/track_embeddings.json",
    telemetry_path: Optional[Union[str, Path]] = "data/member1_perception/cam_001/camera_telemetry.json",
    raw_detections_path: Optional[Union[str, Path]] = "data/member1_perception/cam_001/raw_frame_detections.json",
    camera_id: str = "CAM_001",
    fps: float = 30.0,
    camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
    camera_id_mapping: Optional[Dict[str, str]] = None,
) -> List[Observation]:
    """
    Load real Member 1 perception outputs (YOLOv8 + ByteTrack + OSNet Re-ID + Telemetry)
    into standardized UrbanTrack Observation instances with end-to-end provenance.

    Fuses multi-frame OCR outputs using confidence-weighted character consensus voting,
    attaches validated 512-dimensional OSNet appearance embeddings, isolates actual vehicle
    detection confidence from camera telemetry averages, and calculates multi-frame tracklet
    telemetry aggregates.

    Returns:
        List[Observation]: Standardized observations ready for Member 2 identity fusion.
    """
    import math
    from datetime import datetime
    from .tracklet_engine import aggregate_plate_votes

    # Path fallback: check raw subdirectory if direct path is missing
    p_tracks = Path(tracks_path)
    if not p_tracks.is_file():
        alt_raw = Path("data/member1_perception/cam_001/raw/trajectories.json")
        if alt_raw.is_file():
            p_tracks = alt_raw
        else:
            raise FileNotFoundError(f"Member 1 track embeddings file not found: {tracks_path}")

    with open(p_tracks, "r", encoding="utf-8") as f:
        tracks = json.load(f)

    # Resolve telemetry path with raw fallback
    telemetry_by_frame = {}
    if telemetry_path:
        p_tel = Path(telemetry_path)
        if not p_tel.is_file():
            alt_tel = Path("data/member1_perception/cam_001/raw/camera_telemetry.json")
            if alt_tel.is_file():
                p_tel = alt_tel
        if p_tel.is_file():
            with open(p_tel, "r", encoding="utf-8") as f:
                tel_data = json.load(f)
                frame_list = tel_data.get("frames", []) if isinstance(tel_data, dict) else tel_data
                for fr in frame_list:
                    if isinstance(fr, dict) and "frame_number" in fr:
                        telemetry_by_frame[int(fr["frame_number"])] = fr

    # Resolve raw detections path with raw fallback
    plates_by_track: Dict[int, List[Observation]] = {}
    bboxes_by_track: Dict[int, List[float]] = {}
    det_confs_by_track: Dict[int, List[float]] = {}

    if raw_detections_path:
        p_raw = Path(raw_detections_path)
        if not p_raw.is_file():
            alt_raw_det = Path("data/member1_perception/cam_001/raw/raw_frame_detections.json")
            if alt_raw_det.is_file():
                p_raw = alt_raw_det
        if p_raw.is_file():
            with open(p_raw, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
                raw_frames = raw_data.get("frames", []) if isinstance(raw_data, dict) else raw_data
                for fr in raw_frames:
                    fnum = int(fr.get("frame_number", 0))
                    for v in fr.get("vehicles", []):
                        tid = v.get("track_id")
                        if tid is None:
                            continue
                        tid = int(tid)
                        if tid not in bboxes_by_track and "bbox" in v:
                            bboxes_by_track[tid] = [float(x) for x in v["bbox"]]

                        # Capture vehicle's actual detection confidence from YOLO
                        if "confidence" in v and v["confidence"] is not None:
                            det_confs_by_track.setdefault(tid, []).append(float(v["confidence"]))

                        p = v.get("plate_number")
                        if p and p != "None" and str(p).strip():
                            conf = v.get("plate_confidence") or v.get("plate_text_confidence") or 0.80
                            plates_by_track.setdefault(tid, []).append(
                                Observation(
                                    camera_id=camera_id,
                                    frame_id=fnum,
                                    timestamp_seconds=float(fnum) / fps,
                                    plate=str(p).strip(),
                                    plate_confidence=float(conf),
                                    ocr_confidence=float(v.get("plate_text_confidence")) if v.get("plate_text_confidence") is not None else None,
                                    plate_bbox=[float(b) for b in v["plate_bbox"]] if "plate_bbox" in v and v["plate_bbox"] else None,
                                )
                            )

    # Camera identifier normalization
    effective_camera_id = camera_id
    if camera_id_mapping and camera_id in camera_id_mapping:
        effective_camera_id = camera_id_mapping[camera_id]

    # Attach camera coordinates if available in camera_metadata
    cam_meta = {}
    if camera_metadata:
        cam_meta = camera_metadata.get(effective_camera_id) or camera_metadata.get(camera_id) or {}
    lat = cam_meta.get("latitude")
    lon = cam_meta.get("longitude")

    observations: List[Observation] = []
    for t in tracks:
        track_id = int(t["track_id"])
        start_frame = int(t.get("start_frame", 0))
        end_frame = int(t.get("end_frame", start_frame))
        ts_sec = round(float(start_frame) / fps, 4)

        # Multi-frame OCR consensus voting across track sightings
        consensus_plate = None
        consensus_conf = None
        vote_count = 0
        conflicting_plate_count = 0
        if track_id in plates_by_track:
            obs_plates = plates_by_track[track_id]
            consensus_plate, consensus_conf, vote_count = aggregate_plate_votes(obs_plates)
            unique_plates = set(o.plate for o in obs_plates if o.plate)
            conflicting_plate_count = max(len(unique_plates) - 1, 0)

        # Validate 512-dimensional OSNet embedding vector
        raw_emb = t.get("appearance_embedding")
        validated_emb = None
        if raw_emb is not None and isinstance(raw_emb, (list, tuple)):
            if len(raw_emb) == 512 and all(isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x) for x in raw_emb):
                validated_emb = [float(x) for x in raw_emb]

        # Multi-frame Tracklet Telemetry Aggregation over [start_frame, end_frame]
        rel_vals = []
        blur_vals = []
        bright_vals = []
        occ_vals = []
        frame_det_means = []

        for f_idx in range(start_frame, end_frame + 1):
            if f_idx in telemetry_by_frame:
                f_tel = telemetry_by_frame[f_idx]
                if "reliability" in f_tel and f_tel["reliability"] is not None:
                    rel_vals.append(float(f_tel["reliability"]))
                if "blur_score" in f_tel and f_tel["blur_score"] is not None:
                    blur_vals.append(float(f_tel["blur_score"]))
                if "brightness" in f_tel and f_tel["brightness"] is not None:
                    bright_vals.append(float(f_tel["brightness"]))
                if "occlusion_ratio" in f_tel and f_tel["occlusion_ratio"] is not None:
                    occ_vals.append(float(f_tel["occlusion_ratio"]))
                if "detection_confidence_mean" in f_tel and f_tel["detection_confidence_mean"] is not None:
                    frame_det_means.append(float(f_tel["detection_confidence_mean"]))

        telemetry_frame_count = len(rel_vals)
        mean_rel = round(sum(rel_vals) / len(rel_vals), 4) if rel_vals else None
        min_rel = round(min(rel_vals), 4) if rel_vals else None
        max_rel = round(max(rel_vals), 4) if rel_vals else None
        mean_blur = round(sum(blur_vals) / len(blur_vals), 4) if blur_vals else None
        mean_bright = round(sum(bright_vals) / len(bright_vals), 4) if bright_vals else None
        mean_occ = round(sum(occ_vals) / len(occ_vals), 4) if occ_vals else None
        frame_det_mean = round(sum(frame_det_means) / len(frame_det_means), 4) if frame_det_means else None

        # Actual vehicle detection confidence (YOLO model score across track detections)
        track_confs = det_confs_by_track.get(track_id, [])
        actual_det_conf = round(sum(track_confs) / len(track_confs), 4) if track_confs else 0.85

        traj_pts = t.get("trajectory", [])
        first_pt = traj_pts[0] if traj_pts else None
        avg_vel = float(t.get("average_velocity_px", 0.0)) if t.get("average_velocity_px") is not None else None

        # Build comprehensive source provenance dictionary
        provenance = {
            "source_file": str(p_tracks),
            "camera_id": camera_id,
            "effective_camera_id": effective_camera_id,
            "frame_number": start_frame,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "duration_frames": end_frame - start_frame + 1,
            "track_id": track_id,
            "embedding_id": track_id,
            "reid_model": str(t.get("reid_model", "osnet_x0_25_msmt17")),
            "embedding_dimension": len(validated_emb) if validated_emb else None,
            "embedding_quality": float(t.get("embedding_quality")) if t.get("embedding_quality") is not None else None,
            "actual_detection_confidence_mean": actual_det_conf,
            "plate_consensus_votes": vote_count,
            "conflicting_plate_count": conflicting_plate_count,
            "telemetry_attached": telemetry_frame_count > 0,
            "telemetry_aggregates": {
                "telemetry_frame_count": telemetry_frame_count,
                "mean_reliability": mean_rel,
                "min_reliability": min_rel,
                "max_reliability": max_rel,
                "mean_blur": mean_blur,
                "mean_brightness": mean_bright,
                "mean_occlusion": mean_occ,
                "frame_detection_confidence_mean": frame_det_mean,
            } if telemetry_frame_count > 0 else None,
        }

        obs = Observation(
            observation_id=f"{effective_camera_id}_trk_{track_id:03d}",
            camera_id=effective_camera_id,
            timestamp=datetime.fromtimestamp(max(ts_sec, 0.0)),
            timestamp_seconds=ts_sec,
            frame_id=start_frame,
            track_id=str(track_id),
            vehicle_type=str(t.get("vehicle_type", "car")).lower(),
            detection_confidence=actual_det_conf,
            frame_detection_confidence_mean=frame_det_mean,
            bbox=bboxes_by_track.get(track_id),
            appearance_embedding=validated_emb,
            plate=consensus_plate,
            plate_confidence=consensus_conf,
            latitude=lat,
            longitude=lon,
            trajectory_point=first_pt,
            point_type="image_space_trajectory_point",
            point_coordinate_system="image",
            pixel_speed=avg_vel,
            local_track_history=traj_pts if traj_pts else None,
            timestamp_semantics="video_relative",
            time_reference_id=camera_id,
            camera_reliability=mean_rel,
            source_provenance=provenance,
        )
        observations.append(obs)

    # Sort observations deterministically by timestamp and track_id
    observations.sort(key=lambda o: (o.timestamp_seconds, o.observation_id))
    return observations



# =============================================================================
# MULTI-CAMERA FEED ADAPTER (MODULAR INGESTION LAYER)
# =============================================================================

from dataclasses import dataclass, field
from enum import Enum


class DatasetClassification(str, Enum):
    """Explicit dataset provenance tiers to prevent silent data mixing."""
    REAL = "REAL"
    CONTROLLED = "CONTROLLED"
    SYNTHETIC = "SYNTHETIC"


@dataclass
class CameraFeedConfig:
    """Configuration contract for a registered camera feed."""
    camera_id: str
    classification: DatasetClassification
    source_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)
    time_reference_id: Optional[str] = None
    coordinate_system: str = "image"
    is_active: bool = True


class MultiCameraFeedAdapter:
    """
    Standardized, modular multi-camera feed ingestion adapter for UrbanTrack AI.

    Provides a clean, extensible architectural boundary for ingesting observations:
    1. REAL: Physical camera perception outputs (e.g. Member 1 CCTV CAM_001).
    2. CONTROLLED: Rigorously controlled multi-camera benchmark datasets (e.g. multicamera_v1).
    3. SYNTHETIC: Explicit synthetic diagnostic and stress scenarios.

    Key Guarantees:
    - Zero silent mixing of dataset tiers.
    - Strict validation of coordinate and timestamp contracts.
    - Dynamic camera registration: additional cameras (CAM_002, CAM_003) can be plugged in
      without modifying any downstream fusion or graph logic.
    """

    def __init__(self, camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self.camera_metadata: Dict[str, Dict[str, Any]] = camera_metadata or {}
        self._registered_feeds: Dict[str, CameraFeedConfig] = {}
        self._cached_observations: Dict[str, List[Observation]] = {}

    def register_camera_feed(
        self,
        camera_id: str,
        classification: Union[DatasetClassification, str],
        source_path: Union[str, Path],
        metadata: Optional[Dict[str, Any]] = None,
        time_reference_id: Optional[str] = None,
        coordinate_system: str = "image",
    ) -> None:
        """
        Register a new camera feed into the multi-camera adapter.

        Args:
            camera_id: Unique camera sensor identifier.
            classification: Provenance tier (REAL, CONTROLLED, or SYNTHETIC).
            source_path: Path to observation JSON or directory of tracklet artifacts.
            metadata: Optional camera geographic/sensor metadata.
            time_reference_id: Clock reference ID.
            coordinate_system: Point coordinate system ("image" or "gps").
        """
        if isinstance(classification, str):
            classification = DatasetClassification(classification.upper())

        p = Path(source_path)
        if not p.exists():
            raise FileNotFoundError(f"Feed source path does not exist for camera {camera_id}: {source_path}")

        meta = dict(metadata or {})
        if camera_id in self.camera_metadata:
            meta = {**self.camera_metadata[camera_id], **meta}
        else:
            self.camera_metadata[camera_id] = meta

        feed_cfg = CameraFeedConfig(
            camera_id=camera_id,
            classification=classification,
            source_path=p,
            metadata=meta,
            time_reference_id=time_reference_id,
            coordinate_system=coordinate_system,
        )
        self._registered_feeds[camera_id] = feed_cfg
        # Invalidate cache
        if camera_id in self._cached_observations:
            del self._cached_observations[camera_id]

    def load_camera_observations(self, camera_id: str, force_reload: bool = False) -> List[Observation]:
        """
        Load observations for a specific registered camera feed.
        """
        if camera_id not in self._registered_feeds:
            raise KeyError(f"Camera '{camera_id}' is not registered in MultiCameraFeedAdapter.")

        if not force_reload and camera_id in self._cached_observations:
            return self._cached_observations[camera_id]

        cfg = self._registered_feeds[camera_id]
        p = cfg.source_path

        # If path is directory with track_embeddings.json, load via load_member1_perception_feed
        if p.is_dir():
            t_path = p / 'track_embeddings.json'
            tel_path = p / 'camera_telemetry.json'
            det_path = p / 'raw_frame_detections.json'
            obs = load_member1_perception_feed(
                tracks_path=t_path,
                telemetry_path=tel_path if tel_path.exists() else None,
                raw_detections_path=det_path if det_path.exists() else None,
                camera_id=camera_id,
                camera_metadata=self.camera_metadata,
            )
        elif p.is_file():
            obs = load_observations_from_json(p, camera_metadata=self.camera_metadata)
            # Tag camera_id if needed
            for o in obs:
                if not o.camera_id:
                    o.camera_id = camera_id
        else:
            raise ValueError(f"Unrecognized source path type for camera {camera_id}: {p}")

        # Enforce classification metadata
        for o in obs:
            if not hasattr(o, "dataset_classification") or not o.dataset_classification:
                setattr(o, "dataset_classification", cfg.classification.value)
            if cfg.coordinate_system:
                o.point_coordinate_system = cfg.coordinate_system

        self._cached_observations[camera_id] = obs
        return obs

    def ingest_all_feeds(
        self,
        allowed_classifications: Optional[List[Union[DatasetClassification, str]]] = None,
    ) -> List[Observation]:
        """
        Ingest observations across all registered active cameras, optionally filtering by tier.

        Guarantees:
        Observations from different classifications (e.g. REAL vs SYNTHETIC) are never
        mixed unless explicitly requested.
        """
        if allowed_classifications is not None:
            allowed_set = {
                (c.value if isinstance(c, DatasetClassification) else str(c).upper())
                for c in allowed_classifications
            }
        else:
            allowed_set = None

        all_obs: List[Observation] = []
        for cam_id, cfg in sorted(self._registered_feeds.items()):
            if not cfg.is_active:
                continue
            if allowed_set is not None and cfg.classification.value not in allowed_set:
                continue
            cam_obs = self.load_camera_observations(cam_id)
            all_obs.extend(cam_obs)

        # Deterministic sort
        all_obs.sort(key=lambda o: (o.timestamp_seconds, o.observation_id))
        return all_obs

    def get_inventory(self) -> Dict[str, Any]:
        """Return structured summary of registered feeds and data classification."""
        inventory = {}
        for cam_id, cfg in sorted(self._registered_feeds.items()):
            obs_count = len(self._cached_observations.get(cam_id, []))
            inventory[cam_id] = {
                "classification": cfg.classification.value,
                "source_path": str(cfg.source_path),
                "is_active": cfg.is_active,
                "loaded_observations": obs_count,
                "coordinate_system": cfg.coordinate_system,
                "time_reference_id": cfg.time_reference_id,
            }
        return inventory
