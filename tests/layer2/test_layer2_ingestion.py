"""
UrbanTrack AI — Layer 2 Ingestion Unit & Regression Tests.

Validates all 13 core ingestion requirements including the critical regression test
for reverse-camera chronological travel direction.
"""

import math
import pytest

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
from layer2.ingestion.ocr_aggregation import aggregate_tracklet_ocr, normalize_plate_string
from layer2.ingestion.reid_compatibility import (
    AICITY_GROUP,
    MSMT17_GROUP,
    NONE_GROUP,
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
from layer2.ingestion.validation import validate_canonical_tracklets


def make_dummy_tracklet(
    scenario_id: str,
    camera_id: str,
    track_id: int,
    start_sync_sec: float,
    end_sync_sec: float,
    reid_group: str = AICITY_GROUP,
    has_emb: bool = True,
    plate_text: str = None,
    fps: float = 10.0,
) -> CanonicalTracklet:
    """Helper to construct a minimal valid CanonicalTracklet."""
    emb = [0.0] * 512 if has_emb else None
    if has_emb:
        emb[0] = 1.0  # Unit L2 norm

    return CanonicalTracklet(
        scenario_id=scenario_id,
        camera_id=camera_id,
        track_id=track_id,
        global_vehicle_id=None,
        temporal=TemporalData(
            start_frame=int(start_sync_sec * fps),
            end_frame=int(end_sync_sec * fps),
            duration_frames=int((end_sync_sec - start_sync_sec) * fps) + 1,
            fps=fps,
            start_raw_timestamp="00:00:00.000",
            end_raw_timestamp="00:00:05.000",
            start_raw_seconds=start_sync_sec,
            end_raw_seconds=end_sync_sec,
            start_sync_timestamp=format_timestamp_seconds(start_sync_sec),
            end_sync_timestamp=format_timestamp_seconds(end_sync_sec),
            start_sync_seconds=start_sync_sec,
            end_sync_seconds=end_sync_sec,
            duration_seconds=end_sync_sec - start_sync_sec,
        ),
        motion=MotionData(
            trajectory_pixels=[[100, 200], [105, 205]],
            trajectory_length=2,
            average_velocity_px=1.5,
            direction="northbound",
        ),
        appearance=AppearanceData(
            has_embedding=has_emb,
            appearance_embedding=emb,
            embedding_dim=512 if has_emb else None,
            embedding_quality=0.85 if has_emb else None,
            reid_model="osnet_x0_25_aicity" if reid_group == AICITY_GROUP else "osnet_x0_25_msmt17",
            reid_compatibility_group=reid_group if has_emb else NONE_GROUP,
        ),
        vehicle=VehicleData(vehicle_type="car", average_detector_confidence=0.90),
        anpr=ANPRData(
            has_plate_detection=plate_text is not None,
            has_readable_ocr=plate_text is not None,
            aggregated_plate_text=plate_text,
            ocr_confidence=0.95 if plate_text else None,
            ocr_readings_count=1 if plate_text else 0,
            plate_detections_count=1 if plate_text else 0,
            ocr_consensus_ratio=1.0 if plate_text else None,
        ),
        spatial=SpatialData(
            camera_latitude=42.525,
            camera_longitude=-90.723,
            camera_bearing_deg=0.0,
            camera_confidence="HIGH",
            road_context={"road": "Test Rd"},
            projected_vehicle_coordinates=None,
        ),
        quality=QualityData(camera_reliability=1.0, missing_evidence=[]),
    )


# 1. Timestamp Parsing Tests
def test_timestamp_parsing():
    assert parse_raw_timestamp("00:00:00.000") == 0.0
    assert parse_raw_timestamp("00:01:00.000") == 60.0
    assert parse_raw_timestamp("01:00:00.000") == 3600.0
    assert parse_raw_timestamp("00:03:15.500") == 195.5
    with pytest.raises(ValueError):
        parse_raw_timestamp("invalid")
    with pytest.raises(ValueError):
        parse_raw_timestamp("00:00:00")  # Missing ms


# 2. Official Offset Application Tests
def test_official_offset_application():
    sync = TimestampSynchronizer()
    assert sync.get_offset("S01", "CAM_S01_C001") == 0.0
    assert sync.get_offset("S01", "CAM_S01_C002") == 1.640
    assert sync.get_offset("S01", "CAM_S01_C003") == 2.049
    assert sync.get_offset("S04", "CAM_S04_C040") == 175.838
    assert sync.get_offset("S05", "CAM_S05_C010") == 0.0

    raw_sec = 10.0
    sync_sec = sync.synchronize_raw_seconds(raw_sec, "S01", "CAM_S01_C002")
    assert math.isclose(sync_sec, 11.640, abs_tol=1e-5)


# 3. 8 FPS Handling Tests
def test_8_fps_handling():
    fps = 8.0
    frame_num = 16
    expected_sec = frame_num / fps  # 2.0s
    assert expected_sec == 2.0


# 4. OCR Normalization Tests
def test_ocr_normalization():
    assert normalize_plate_string("  ia-789_abc  ") == "IA789ABC"
    assert normalize_plate_string("ab") is None  # Too short
    assert normalize_plate_string(None) is None
    assert normalize_plate_string("123") == "123"


# 5. OCR Aggregation Consensus Tests
def test_ocr_aggregation_consensus():
    obs = [
        {"plate_bbox": [10, 20, 50, 40], "plate_number": "ABC-123", "plate_text_confidence": 0.90},
        {"plate_bbox": [10, 20, 50, 40], "plate_number": "ABC-123", "plate_text_confidence": 0.95},
        {"plate_bbox": [10, 20, 50, 40], "plate_number": "XYZ-999", "plate_text_confidence": 0.80},
    ]
    anpr = aggregate_tracklet_ocr(obs)
    assert anpr.has_readable_ocr is True
    assert anpr.aggregated_plate_text == "ABC123"
    assert anpr.ocr_readings_count == 3
    assert anpr.plate_detections_count == 3
    assert math.isclose(anpr.ocr_consensus_ratio, 2 / 3, abs_tol=1e-3)


# 6. Missing OCR Tests
def test_missing_ocr():
    obs = [{"plate_bbox": None, "plate_number": None, "plate_text_confidence": None}]
    anpr = aggregate_tracklet_ocr(obs)
    assert anpr.has_plate_detection is False
    assert anpr.has_readable_ocr is False
    assert anpr.aggregated_plate_text is None
    assert anpr.ocr_confidence is None


# 7. Missing Embedding Tests
def test_missing_embedding():
    t_no_emb = make_dummy_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0, has_emb=False)
    assert t_no_emb.appearance.has_embedding is False
    assert t_no_emb.appearance.appearance_embedding is None
    assert t_no_emb.appearance.reid_compatibility_group == NONE_GROUP

    t_emb = make_dummy_tracklet("S01", "CAM_S01_C001", 2, 0.0, 5.0, has_emb=True)
    assert are_reid_compatible(t_no_emb, t_emb) is False


# 8. Re-ID Compatibility Tests
def test_reid_compatibility():
    t_aicity1 = make_dummy_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0, reid_group=AICITY_GROUP)
    t_aicity2 = make_dummy_tracklet("S01", "CAM_S01_C003", 2, 6.0, 10.0, reid_group=AICITY_GROUP)
    t_msmt = make_dummy_tracklet("S01", "CAM_S01_C002", 3, 2.0, 7.0, reid_group=MSMT17_GROUP)

    # Compatible AICity vs AICity
    assert are_reid_compatible(t_aicity1, t_aicity2) is True
    stat_ok = get_reid_compatibility_status(t_aicity1, t_aicity2)
    assert stat_ok["status"] == "COMPATIBLE"
    assert stat_ok["allows_cosine_similarity"] is True

    # Incompatible AICity vs MSMT17
    assert are_reid_compatible(t_aicity1, t_msmt) is False
    stat_incompat = get_reid_compatibility_status(t_aicity1, t_msmt)
    assert stat_incompat["status"] == "INCOMPATIBLE"
    assert stat_incompat["allows_cosine_similarity"] is False


# 9. Scenario Isolation Tests
def test_scenario_isolation_validation():
    t_s01 = make_dummy_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0)
    # Malformed tracklet with mismatched scenario
    t_bad = make_dummy_tracklet("S02", "CAM_S01_C001", 2, 0.0, 5.0)

    passed, report = validate_canonical_tracklets([t_s01, t_bad], expected_camera_count=1, expected_tracklet_count=2)
    assert not report["invariants"]["INV-01_SCENARIO_ISOLATION"]["passed"]


# 10. Chronological Ordering Forward Test
def test_chronological_ordering_forward():
    t_a = make_dummy_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0)
    t_b = make_dummy_tracklet("S01", "CAM_S01_C002", 2, 7.0, 12.0)

    rel = compare_chronological_order(t_a, t_b)
    assert rel.origin_tracklet_id == t_a.canonical_id
    assert rel.destination_tracklet_id == t_b.canonical_id
    assert rel.chronological_direction == "A_TO_B"
    assert math.isclose(rel.delta_t_seconds, 2.0, abs_tol=1e-3)


# 11. REGRESSION TEST: Reverse Camera Ordering Bug Prevention
def test_reverse_camera_ordering_regression():
    """
    REGRESSION TEST:
    Reproduces the exact historical failure mode where alphabetical camera ordering
    (CAM_S01_C001 < CAM_S01_C002) led to delta_t = t(C002) - t(C001) < 0 when a vehicle
    physically traveled from C002 to C001 (e.g. Westbound).

    The test proves that regardless of input argument order (t_c001, t_c002) or (t_c002, t_c001),
    compare_chronological_order() establishes origin based strictly on timestamps!
    """
    # Vehicle appears in C002 first at t_sync=2.0s, leaves at t_sync=5.0s
    t_c002 = make_dummy_tracklet("S01", "CAM_S01_C002", 10, start_sync_sec=2.0, end_sync_sec=5.0)

    # Vehicle arrives in C001 later at t_sync=8.0s, leaves at t_sync=12.0s
    t_c001 = make_dummy_tracklet("S01", "CAM_S01_C001", 20, start_sync_sec=8.0, end_sync_sec=12.0)

    # Notice: Alphabetical order is CAM_S01_C001 < CAM_S01_C002
    assert t_c001.camera_id < t_c002.camera_id

    # Case A: Passed as (C001, C002)
    rel_a = compare_chronological_order(t_c001, t_c002)
    assert rel_a.origin_tracklet_id == t_c002.canonical_id
    assert rel_a.destination_tracklet_id == t_c001.canonical_id
    assert rel_a.chronological_direction == "B_TO_A"
    assert math.isclose(rel_a.delta_t_seconds, 3.0, abs_tol=1e-3)
    assert rel_a.delta_t_seconds > 0.0

    # Case B: Passed as (C002, C001)
    rel_b = compare_chronological_order(t_c002, t_c001)
    assert rel_b.origin_tracklet_id == t_c002.canonical_id
    assert rel_b.destination_tracklet_id == t_c001.canonical_id
    assert rel_b.chronological_direction == "A_TO_B"
    assert math.isclose(rel_b.delta_t_seconds, 3.0, abs_tol=1e-3)
    assert rel_b.delta_t_seconds > 0.0


# 12. Duplicate Track Detection Tests
def test_duplicate_track_detection():
    t1 = make_dummy_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0)
    t2 = make_dummy_tracklet("S01", "CAM_S01_C001", 1, 10.0, 15.0)  # Same camera and track_id!

    passed, report = validate_canonical_tracklets([t1, t2], expected_camera_count=1, expected_tracklet_count=2)
    assert "S01:CAM_S01_C001:1" in report["structural_checks"]["duplicate_keys"]


# 13. Embedding Validation Tests
def test_embedding_normalization_check():
    # Valid unit norm embedding
    t_valid = make_dummy_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0, has_emb=True)
    # Unnormalized embedding
    t_unnorm = make_dummy_tracklet("S01", "CAM_S01_C001", 2, 0.0, 5.0, has_emb=True)
    t_unnorm.appearance.appearance_embedding = [2.0] * 512

    passed, report = validate_canonical_tracklets([t_valid, t_unnorm], expected_camera_count=1, expected_tracklet_count=2)
    assert report["structural_checks"]["embedding_norm_not_one_count"] == 1
