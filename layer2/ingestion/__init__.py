"""
UrbanTrack AI — Layer 2 Ingestion Package.
"""

from layer2.ingestion.canonical_models import (
    ANPRData,
    AppearanceData,
    CanonicalTracklet,
    ChronologicalRelationship,
    MotionData,
    QualityData,
    SpatialData,
    TemporalData,
    VehicleData,
)
from layer2.ingestion.ocr_aggregation import (
    aggregate_tracklet_ocr,
    normalize_plate_string,
)
from layer2.ingestion.reid_compatibility import (
    are_reid_compatible,
    get_reid_compatibility_status,
    map_reid_compatibility_group,
)
from layer2.ingestion.timestamp_sync import (
    TimestampSynchronizer,
    compare_chronological_order,
    format_timestamp_seconds,
    parse_raw_timestamp,
)
from layer2.ingestion.tracklet_loader import TrackletLoader
from layer2.ingestion.validation import validate_canonical_tracklets

__all__ = [
    "CanonicalTracklet",
    "TemporalData",
    "MotionData",
    "AppearanceData",
    "VehicleData",
    "ANPRData",
    "SpatialData",
    "QualityData",
    "ChronologicalRelationship",
    "TrackletLoader",
    "TimestampSynchronizer",
    "compare_chronological_order",
    "parse_raw_timestamp",
    "format_timestamp_seconds",
    "are_reid_compatible",
    "get_reid_compatibility_status",
    "map_reid_compatibility_group",
    "normalize_plate_string",
    "aggregate_tracklet_ocr",
    "validate_canonical_tracklets",
]
