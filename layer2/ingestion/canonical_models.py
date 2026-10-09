"""
UrbanTrack AI — Layer 2 Canonical Data Models.

Defines the normalized, in-memory representations for Layer 2 ingestion:
- CanonicalTracklet
- TemporalData
- MotionData
- AppearanceData
- VehicleData
- ANPRData
- SpatialData
- QualityData
- ChronologicalRelationship
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TemporalData:
    start_frame: int
    end_frame: int
    duration_frames: int
    fps: float
    start_raw_timestamp: str
    end_raw_timestamp: str
    start_raw_seconds: float
    end_raw_seconds: float
    start_sync_timestamp: str
    end_sync_timestamp: str
    start_sync_seconds: float
    end_sync_seconds: float
    duration_seconds: float


@dataclass
class MotionData:
    trajectory_pixels: List[List[int]]
    trajectory_length: int
    average_velocity_px: float
    direction: Optional[str]


@dataclass
class AppearanceData:
    has_embedding: bool
    appearance_embedding: Optional[List[float]]
    embedding_dim: Optional[int]
    embedding_quality: Optional[float]
    reid_model: str
    reid_compatibility_group: str


@dataclass
class VehicleData:
    vehicle_type: str
    average_detector_confidence: Optional[float]


@dataclass
class ANPRData:
    has_plate_detection: bool
    has_readable_ocr: bool
    aggregated_plate_text: Optional[str]
    ocr_confidence: Optional[float]
    ocr_readings_count: int
    plate_detections_count: int
    ocr_consensus_ratio: Optional[float]


@dataclass
class SpatialData:
    camera_latitude: float
    camera_longitude: float
    camera_bearing_deg: float
    camera_confidence: str
    road_context: Dict[str, Any]
    projected_vehicle_coordinates: None = None


@dataclass
class QualityData:
    camera_reliability: float
    missing_evidence: List[str] = field(default_factory=list)


@dataclass
class CanonicalTracklet:
    scenario_id: str
    camera_id: str
    track_id: int
    global_vehicle_id: Optional[str]
    temporal: TemporalData
    motion: MotionData
    appearance: AppearanceData
    vehicle: VehicleData
    anpr: ANPRData
    spatial: SpatialData
    quality: QualityData

    @property
    def canonical_id(self) -> str:
        """Returns the canonical tracklet identifier: SCENARIO:CAMERA:TRACK."""
        return f"{self.scenario_id}:{self.camera_id}:{self.track_id}"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes CanonicalTracklet to dictionary conforming strictly to contract."""
        return asdict(self)


@dataclass
class ChronologicalRelationship:
    """Represents direction-independent chronological relationship between two tracklets."""
    origin_tracklet_id: str
    destination_tracklet_id: str
    chronological_direction: str  # "A_TO_B" or "B_TO_A"
    delta_t_seconds: float
    is_chronologically_valid: bool
    t_origin_departure_sync: float
    t_dest_arrival_sync: float
    overlap_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
