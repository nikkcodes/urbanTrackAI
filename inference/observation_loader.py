"""
Observation and camera metadata loading utilities for UrbanTrack AI.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

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
    attaches validated 512-dimensional OSNet appearance embeddings, and integrates frame-level
    camera reliability and sensor telemetry.

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

                        p = v.get("plate_number")
                        if p and p != "None" and str(p).strip():
                            conf = v.get("plate_confidence") or v.get("plate_text_confidence") or 0.80
                            if tid not in plates_by_track:
                                plates_by_track[tid] = []
                            plates_by_track[tid].append(
                                Observation(
                                    camera_id=camera_id,
                                    frame_id=fnum,
                                    timestamp_seconds=float(fnum) / fps,
                                    plate=str(p).strip(),
                                    plate_confidence=float(conf),
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
        ts_sec = round(float(start_frame) / fps, 4)

        # Consensus plate voting across track sightings
        consensus_plate = None
        consensus_conf = None
        vote_count = 0
        if track_id in plates_by_track:
            consensus_plate, consensus_conf, vote_count = aggregate_plate_votes(plates_by_track[track_id])

        # Validate 512-dimensional OSNet embedding vector
        raw_emb = t.get("appearance_embedding")
        validated_emb = None
        if raw_emb is not None and isinstance(raw_emb, (list, tuple)):
            if len(raw_emb) == 512 and all(isinstance(x, (int, float)) and not math.isnan(x) and not math.isinf(x) for x in raw_emb):
                validated_emb = [float(x) for x in raw_emb]

        # Extract telemetry for this frame
        tel = telemetry_by_frame.get(start_frame, {})
        cam_rel = float(tel.get("reliability", 0.52)) if "reliability" in tel else None
        det_conf = float(tel.get("detection_confidence_mean", 0.85)) if "detection_confidence_mean" in tel else 0.85

        traj_pts = t.get("trajectory", [])
        first_pt = traj_pts[0] if traj_pts else None
        avg_vel = float(t.get("average_velocity_px", 0.0)) if t.get("average_velocity_px") is not None else None

        # Build comprehensive source provenance dictionary (Item 6 & 7)
        provenance = {
            "source_file": str(p_tracks),
            "camera_id": camera_id,
            "effective_camera_id": effective_camera_id,
            "frame_number": start_frame,
            "track_id": track_id,
            "embedding_id": track_id,
            "reid_model": str(t.get("reid_model", "osnet_x0_25_msmt17")),
            "embedding_dimension": len(validated_emb) if validated_emb else None,
            "embedding_quality": float(t.get("embedding_quality")) if t.get("embedding_quality") is not None else None,
            "plate_consensus_votes": vote_count,
            "telemetry_attached": start_frame in telemetry_by_frame,
            "telemetry_metrics": {
                "blur_score": tel.get("blur_score"),
                "brightness": tel.get("brightness"),
                "occlusion_ratio": tel.get("occlusion_ratio"),
                "reliability": tel.get("reliability"),
            } if start_frame in telemetry_by_frame else None,
        }

        obs = Observation(
            observation_id=f"{effective_camera_id}_trk_{track_id:03d}",
            camera_id=effective_camera_id,
            timestamp=datetime.fromtimestamp(max(ts_sec, 0.0)),
            timestamp_seconds=ts_sec,
            frame_id=start_frame,
            track_id=str(track_id),
            vehicle_type=str(t.get("vehicle_type", "car")).lower(),
            detection_confidence=det_conf,
            bbox=bboxes_by_track.get(track_id),
            appearance_embedding=validated_emb,
            plate=consensus_plate,
            plate_confidence=consensus_conf,
            latitude=lat,
            longitude=lon,
            trajectory_point=first_pt,
            pixel_speed=avg_vel,
            local_track_history=traj_pts if traj_pts else None,
            timestamp_semantics="video_relative",
            time_reference_id=camera_id,
            camera_reliability=cam_rel,
            source_provenance=provenance,
        )
        observations.append(obs)

    # Sort observations deterministically by timestamp and track_id
    observations.sort(key=lambda o: (o.timestamp_seconds, o.observation_id))
    return observations

