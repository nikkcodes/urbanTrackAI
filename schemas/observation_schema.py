"""
Data model definition and JSON I/O for vehicle observations.
Supports perception pipeline outputs and standard mobility observations with backwards compatibility.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Union


@dataclass
class CameraMetadata:
    """
    Metadata representation for physical sensors / cameras in the city network.
    Maintains clean separation between perception observations and GIS/network registration.
    """
    camera_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    road_id: Optional[str] = None
    junction_id: Optional[str] = None
    bearing: Optional[float] = None
    time_reference_id: Optional[str] = None
    clock_offset_seconds: Optional[float] = None
    synchronization_status: Optional[str] = None
    timestamp_semantics: Optional[str] = None
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "camera_id": self.camera_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "road_id": self.road_id,
            "junction_id": self.junction_id,
            "bearing": self.bearing,
            "time_reference_id": self.time_reference_id,
            "clock_offset_seconds": self.clock_offset_seconds,
            "synchronization_status": self.synchronization_status,
            "timestamp_semantics": self.timestamp_semantics,
            "name": self.name,
        }
        if self.metadata:
            d.update(self.metadata)
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any], camera_id: Optional[str] = None) -> "CameraMetadata":
        cid = camera_id or data.get("camera_id")
        if not cid:
            raise ValueError("Missing camera_id for CameraMetadata")
        return cls(
            camera_id=str(cid),
            latitude=float(data["latitude"]) if "latitude" in data and data["latitude"] is not None else None,
            longitude=float(data["longitude"]) if "longitude" in data and data["longitude"] is not None else None,
            road_id=data.get("road_id"),
            junction_id=data.get("junction_id"),
            bearing=float(data["bearing"]) if "bearing" in data and data["bearing"] is not None else None,
            time_reference_id=data.get("time_reference_id"),
            clock_offset_seconds=float(data["clock_offset_seconds"]) if "clock_offset_seconds" in data and data["clock_offset_seconds"] is not None else None,
            synchronization_status=data.get("synchronization_status"),
            timestamp_semantics=data.get("timestamp_semantics"),
            name=data.get("name"),
            metadata={k: v for k, v in data.items() if k not in (
                "camera_id", "latitude", "longitude", "road_id", "junction_id",
                "bearing", "time_reference_id", "clock_offset_seconds",
                "synchronization_status", "timestamp_semantics", "name"
            )},
        )


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

    # Member 1 Perception Fields
    trajectory_point: Optional[List[float]] = None
    point_type: str = "vehicle_footpoint"
    point_coordinate_system: str = "image"
    heading_angle: Optional[float] = None
    heading_coordinate_system: str = "image"
    heading_type: str = "image_motion_direction"
    pixel_speed: Optional[float] = None
    plate_bbox: Optional[List[float]] = None
    plate_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    local_track_history: Optional[List[Any]] = None

    # Temporal Metadata
    timestamp_semantics: Optional[str] = None
    time_reference_id: Optional[str] = None
    clock_offset_seconds: Optional[float] = None
    time_uncertainty_seconds: Optional[float] = None

    @property
    def coordinate_system(self) -> str:
        return self.point_coordinate_system

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
        trajectory_point: Optional[List[float]] = None,
        point_type: str = "vehicle_footpoint",
        point_coordinate_system: str = "image",
        heading_angle: Optional[float] = None,
        heading_coordinate_system: str = "image",
        heading_type: str = "image_motion_direction",
        pixel_speed: Optional[float] = None,
        plate_bbox: Optional[List[float]] = None,
        plate_text: Optional[str] = None,
        ocr_confidence: Optional[float] = None,
        local_track_history: Optional[List[Any]] = None,
        timestamp_semantics: Optional[str] = None,
        time_reference_id: Optional[str] = None,
        clock_offset_seconds: Optional[float] = None,
        time_uncertainty_seconds: Optional[float] = None,
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

        # Member 1 Perception Fields
        if trajectory_point is not None:
            if not isinstance(trajectory_point, (list, tuple)) or len(trajectory_point) != 2:
                raise ValueError("trajectory_point must be a list or tuple of 2 numbers [x, y].")
            self.trajectory_point = [float(p) for p in trajectory_point]
        elif self.bbox is not None and len(self.bbox) == 4:
            # Footpoint: bottom-center of vehicle bounding box in image pixel coordinates
            self.trajectory_point = [(self.bbox[0] + self.bbox[2]) / 2.0, float(self.bbox[3])]
        else:
            self.trajectory_point = None
        self.point_type = str(point_type) if point_type else "vehicle_footpoint"
        self.point_coordinate_system = str(point_coordinate_system) if point_coordinate_system else "image"

        if heading_angle is not None:
            if not isinstance(heading_angle, (int, float)):
                raise TypeError("heading_angle must be a number or None.")
            self.heading_angle = float(heading_angle)
        else:
            self.heading_angle = None
        self.heading_coordinate_system = str(heading_coordinate_system) if heading_coordinate_system else "image"
        self.heading_type = str(heading_type) if heading_type else "image_motion_direction"

        if pixel_speed is not None:
            if not isinstance(pixel_speed, (int, float)):
                raise TypeError("pixel_speed must be a number or None.")
            self.pixel_speed = float(pixel_speed)
        else:
            self.pixel_speed = None

        if plate_bbox is not None:
            if not isinstance(plate_bbox, (list, tuple)) or len(plate_bbox) != 4:
                raise ValueError("plate_bbox must be a list or tuple of 4 numbers [x1, y1, x2, y2].")
            self.plate_bbox = [float(b) for b in plate_bbox]
        else:
            self.plate_bbox = None

        resolved_plate = plate or plate_text
        if resolved_plate is not None:
            if not isinstance(resolved_plate, str):
                raise TypeError("plate/plate_text must be a string or None.")
            self.plate = resolved_plate.strip() or None
            self.plate_text = self.plate
        else:
            self.plate = None
            self.plate_text = None

        if ocr_confidence is not None:
            if not isinstance(ocr_confidence, (int, float)):
                raise TypeError("ocr_confidence must be a number or None.")
            val = float(ocr_confidence)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"ocr_confidence must be between 0.0 and 1.0, got {val}.")
            self.ocr_confidence = val
        else:
            self.ocr_confidence = None

        if local_track_history is not None:
            if not isinstance(local_track_history, (list, tuple)):
                raise TypeError("local_track_history must be a list or tuple.")
            self.local_track_history = list(local_track_history)
        else:
            self.local_track_history = None

        # Temporal metadata
        if timestamp_semantics is not None:
            self.timestamp_semantics = str(timestamp_semantics).strip().lower()
        else:
            self.timestamp_semantics = None

        self.time_reference_id = str(time_reference_id).strip() if time_reference_id is not None else None

        if clock_offset_seconds is not None:
            if not isinstance(clock_offset_seconds, (int, float)):
                raise TypeError("clock_offset_seconds must be a number or None.")
            self.clock_offset_seconds = float(clock_offset_seconds)
        else:
            self.clock_offset_seconds = None

        if time_uncertainty_seconds is not None:
            if not isinstance(time_uncertainty_seconds, (int, float)):
                raise TypeError("time_uncertainty_seconds must be a number or None.")
            self.time_uncertainty_seconds = float(time_uncertainty_seconds)
        else:
            self.time_uncertainty_seconds = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert Observation to dictionary representation."""
        ts_str = self.timestamp.isoformat() if self.timestamp else None
        if ts_str and ts_str.endswith("+00:00"):
            ts_str = ts_str[:-6] + "Z"

        d = {
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
        if self.trajectory_point is not None:
            d["trajectory_point"] = self.trajectory_point
            d["point_type"] = self.point_type
            d["point_coordinate_system"] = self.point_coordinate_system
        if self.heading_angle is not None:
            d["heading_angle"] = self.heading_angle
            d["heading_coordinate_system"] = self.heading_coordinate_system
            d["heading_type"] = self.heading_type
        if self.pixel_speed is not None:
            d["pixel_speed"] = self.pixel_speed
        if self.plate_bbox is not None:
            d["plate_bbox"] = self.plate_bbox
        if self.plate_text is not None:
            d["plate_text"] = self.plate_text
        if self.ocr_confidence is not None:
            d["ocr_confidence"] = self.ocr_confidence
        if self.local_track_history is not None:
            d["local_track_history"] = self.local_track_history
        if self.timestamp_semantics is not None:
            d["timestamp_semantics"] = self.timestamp_semantics
        if self.time_reference_id is not None:
            d["time_reference_id"] = self.time_reference_id
        if self.clock_offset_seconds is not None:
            d["clock_offset_seconds"] = self.clock_offset_seconds
        if self.time_uncertainty_seconds is not None:
            d["time_uncertainty_seconds"] = self.time_uncertainty_seconds
        return d

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

        traj_pt = data.get("trajectory_point")
        pt_type = data.get("point_type", "vehicle_footpoint")
        pt_coord_sys = data.get("point_coordinate_system") or data.get("coordinate_system", "image")
        head_angle = data.get("heading_angle")
        head_coord_sys = data.get("heading_coordinate_system", "image")
        head_type = data.get("heading_type", "image_motion_direction")
        px_speed = data.get("pixel_speed")
        plt_bbox = data.get("plate_bbox")
        plt_text = data.get("plate_text") or data.get("plate")
        ocr_conf = data.get("ocr_confidence")
        loc_track_hist = data.get("local_track_history")
        ts_sem = data.get("timestamp_semantics")
        time_ref_id = data.get("time_reference_id")
        clock_offset = data.get("clock_offset_seconds")
        time_unc = data.get("time_uncertainty_seconds")

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
            trajectory_point=traj_pt,
            point_type=pt_type,
            point_coordinate_system=pt_coord_sys,
            heading_angle=head_angle,
            heading_coordinate_system=head_coord_sys,
            heading_type=head_type,
            pixel_speed=px_speed,
            plate_bbox=plt_bbox,
            plate_text=plt_text,
            ocr_confidence=ocr_conf,
            local_track_history=loc_track_hist,
            timestamp_semantics=ts_sem,
            time_reference_id=time_ref_id,
            clock_offset_seconds=clock_offset,
            time_uncertainty_seconds=time_unc,
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
