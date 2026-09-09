"""
Data model definition and JSON I/O for vehicle observations.
Supports perception pipeline outputs and standard mobility observations with backwards compatibility.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Union


@dataclass(init=False)
class Observation:
    """
    Standardized observation model representing a vehicle detection by a camera.
    """
    observation_id: str
    camera_id: str
    timestamp: datetime
    timestamp_seconds: float
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    plate: Optional[str] = None
    plate_confidence: Optional[float] = None
    appearance_embedding: Optional[List[float]] = None
    camera_reliability: Optional[float] = None
    frame_id: Optional[int] = None
    track_id: Optional[str] = None
    vehicle_type: Optional[str] = None
    detection_confidence: Optional[float] = None
    bbox: Optional[List[float]] = None

    def __init__(
        self,
        observation_id_or_camera_id: Optional[str] = None,
        camera_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        plate: Optional[str] = None,
        plate_confidence: Optional[float] = None,
        appearance_embedding: Optional[List[float]] = None,
        camera_reliability: Optional[float] = None,
        frame_id: Optional[int] = None,
        timestamp_seconds: Optional[float] = None,
        track_id: Optional[str] = None,
        vehicle_type: Optional[str] = None,
        detection_confidence: Optional[float] = None,
        bbox: Optional[List[float]] = None,
        observation_id: Optional[str] = None,
    ) -> None:
        # Handle signature resolution for backwards compatibility
        if camera_id is not None:
            # Called as Observation(obs_id, cam_id, ts, lat, lon...) or keyword camera_id="cam_01"
            resolved_obs_id = observation_id_or_camera_id or observation_id
            resolved_cam_id = camera_id
        else:
            # Called as Observation(camera_id="cam_01", ...)
            resolved_obs_id = observation_id
            resolved_cam_id = observation_id_or_camera_id

        if not resolved_cam_id or not isinstance(resolved_cam_id, str) or not resolved_cam_id.strip():
            raise ValueError("camera_id must be a non-empty string.")

        self.camera_id = resolved_cam_id.strip()

        if resolved_obs_id is not None:
            if not isinstance(resolved_obs_id, str) or not resolved_obs_id.strip():
                raise ValueError("observation_id must be a non-empty string.")
            self.observation_id = resolved_obs_id.strip()
        else:
            if track_id is not None and frame_id is not None:
                self.observation_id = f"{self.camera_id}_{track_id}_{frame_id}"
            elif track_id is not None:
                self.observation_id = f"{self.camera_id}_{track_id}"
            else:
                self.observation_id = f"{self.camera_id}_obs_{id(self)}"

        # Timestamp & timestamp_seconds
        if timestamp_seconds is not None:
            if not isinstance(timestamp_seconds, (int, float)):
                raise TypeError("timestamp_seconds must be a number.")
            val_ts = float(timestamp_seconds)
            if val_ts < 0.0:
                raise ValueError(f"timestamp_seconds must be non-negative, got {val_ts}.")
            self.timestamp_seconds = val_ts

        if timestamp is not None:
            if not isinstance(timestamp, datetime):
                raise TypeError("timestamp must be a datetime instance.")
            self.timestamp = timestamp
            if timestamp_seconds is None:
                self.timestamp_seconds = timestamp.timestamp()
        elif timestamp_seconds is not None:
            self.timestamp = datetime.fromtimestamp(self.timestamp_seconds, tz=timezone.utc)
        else:
            self.timestamp = datetime.now(timezone.utc)
            self.timestamp_seconds = self.timestamp.timestamp()

        # Coordinates
        if latitude is not None:
            if not isinstance(latitude, (int, float)) or not (-90.0 <= latitude <= 90.0):
                raise ValueError(f"latitude must be between -90.0 and 90.0, got {latitude}.")
            self.latitude = float(latitude)
        else:
            self.latitude = None

        if longitude is not None:
            if not isinstance(longitude, (int, float)) or not (-180.0 <= longitude <= 180.0):
                raise ValueError(f"longitude must be between -180.0 and 180.0, got {longitude}.")
            self.longitude = float(longitude)
        else:
            self.longitude = None

        # Plate
        if plate is not None:
            if not isinstance(plate, str):
                raise TypeError("plate must be a string or None.")
            self.plate = plate.strip() or None
        else:
            self.plate = None

        if plate_confidence is not None:
            if not isinstance(plate_confidence, (int, float)):
                raise TypeError("plate_confidence must be a number or None.")
            val = float(plate_confidence)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"plate_confidence must be between 0.0 and 1.0, got {val}.")
            self.plate_confidence = val
        else:
            self.plate_confidence = None

        # Reliability
        if camera_reliability is not None:
            if not isinstance(camera_reliability, (int, float)):
                raise TypeError("camera_reliability must be a number or None.")
            val = float(camera_reliability)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"camera_reliability must be between 0.0 and 1.0, got {val}.")
            self.camera_reliability = val
        else:
            self.camera_reliability = None

        # Perception fields
        self.frame_id = int(frame_id) if frame_id is not None else None
        self.track_id = str(track_id).strip() if track_id is not None else None

        if vehicle_type is not None:
            if not isinstance(vehicle_type, str):
                raise TypeError("vehicle_type must be a string or None.")
            self.vehicle_type = vehicle_type.strip().lower() or None
        else:
            self.vehicle_type = None

        if detection_confidence is not None:
            if not isinstance(detection_confidence, (int, float)):
                raise TypeError("detection_confidence must be a number or None.")
            val = float(detection_confidence)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"detection_confidence must be between 0.0 and 1.0, got {val}.")
            self.detection_confidence = val
        else:
            self.detection_confidence = None

        if bbox is not None:
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                raise ValueError("bbox must be a list or tuple of 4 numbers [x1, y1, x2, y2].")
            self.bbox = [float(b) for b in bbox]
        else:
            self.bbox = None

        if appearance_embedding is not None:
            if not isinstance(appearance_embedding, (list, tuple)):
                raise TypeError("appearance_embedding must be a list of numbers or None.")
            cleaned_emb = []
            for item in appearance_embedding:
                if not isinstance(item, (int, float)):
                    raise TypeError(f"All elements in appearance_embedding must be numbers, got {type(item)}.")
                cleaned_emb.append(float(item))
            self.appearance_embedding = cleaned_emb
        else:
            self.appearance_embedding = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert Observation to dictionary representation."""
        ts_str = self.timestamp.isoformat() if self.timestamp else None
        if ts_str and ts_str.endswith("+00:00"):
            ts_str = ts_str[:-6] + "Z"

        return {
            "observation_id": self.observation_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp_seconds": self.timestamp_seconds,
            "timestamp": ts_str,
            "track_id": self.track_id,
            "vehicle_type": self.vehicle_type,
            "detection_confidence": self.detection_confidence,
            "bbox": self.bbox,
            "appearance_embedding": self.appearance_embedding,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "plate": self.plate,
            "plate_confidence": self.plate_confidence,
            "camera_reliability": self.camera_reliability,
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serialize Observation to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        """Construct an Observation instance from a dictionary."""
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict, got {type(data).__name__}")

        if "camera_id" not in data and "observation_id" not in data:
            raise ValueError("Missing required key 'camera_id' for Observation")

        raw_ts = data.get("timestamp")
        parsed_ts = None
        if isinstance(raw_ts, str):
            clean_ts = raw_ts.replace("Z", "+00:00")
            parsed_ts = datetime.fromisoformat(clean_ts)
        elif isinstance(raw_ts, datetime):
            parsed_ts = raw_ts

        det_conf = data.get("detection_confidence")
        if det_conf is None and "confidence" in data:
            det_conf = data.get("confidence")

        return cls(
            observation_id_or_camera_id=data.get("observation_id"),
            camera_id=data.get("camera_id"),
            timestamp=parsed_ts,
            latitude=data.get("latitude"),
            longitude=data.get("longitude"),
            plate=data.get("plate"),
            plate_confidence=data.get("plate_confidence"),
            appearance_embedding=data.get("appearance_embedding"),
            camera_reliability=data.get("camera_reliability"),
            frame_id=data.get("frame_id"),
            timestamp_seconds=data.get("timestamp_seconds"),
            track_id=data.get("track_id"),
            vehicle_type=data.get("vehicle_type"),
            detection_confidence=det_conf,
            bbox=data.get("bbox"),
        )

    @classmethod
    def from_json(cls, json_data: Union[str, Dict[str, Any]]) -> "Observation":
        """Construct an Observation instance from a JSON string or dict."""
        if isinstance(json_data, str):
            try:
                dict_data = json.loads(json_data)
            except json.JSONDecodeError as err:
                raise ValueError(f"Invalid JSON string provided: {err}")
        elif isinstance(json_data, dict):
            dict_data = json_data
        else:
            raise TypeError(f"json_data must be a JSON string or dict, got {type(json_data).__name__}")

        return cls.from_dict(dict_data)
