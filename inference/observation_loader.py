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
            metadata_map[str(cam_id)] = {
                "latitude": float(meta["latitude"]) if "latitude" in meta else None,
                "longitude": float(meta["longitude"]) if "longitude" in meta else None,
                **{k: v for k, v in meta.items() if k not in ("latitude", "longitude")},
            }
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

        # Attach camera coordinates if missing on observation but available in camera_metadata
        if camera_metadata and obs.camera_id in camera_metadata:
            cam_meta = camera_metadata[obs.camera_id]
            if obs.latitude is None and cam_meta.get("latitude") is not None:
                obs.latitude = cam_meta["latitude"]
            if obs.longitude is None and cam_meta.get("longitude") is not None:
                obs.longitude = cam_meta["longitude"]

        observations.append(obs)

    return observations
