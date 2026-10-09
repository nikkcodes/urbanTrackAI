"""
UrbanTrack AI — Layer 2 Candidate Generation Unit & Regression Tests.

Validates all 15 core requirements:
1. cross-scenario isolation
2. same-camera exclusion
3. chronological ordering
4. reverse alphabetical ordering regression
5. negative time rejection
6. positive time acceptance
7. one-way camera edge direction
8. no-edge does not automatically reject
9. excessive-speed rejection
10. valid-speed candidate
11. missing embedding handling
12. incompatible ReID handling
13. vehicle-type disagreement does not automatically erase candidate
14. deterministic output
15. rejection reason completeness
"""

import math
import pytest

from layer2.candidate_generation.candidate_generator import CandidateGenerator
from layer2.candidate_generation.spatial_gate import SpatialGate, haversine_distance_m
from layer2.candidate_generation.temporal_gate import TemporalGate
from layer2.candidate_generation.topology_gate import TopologyGate
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
from layer2.ingestion.reid_compatibility import AICITY_GROUP, MSMT17_GROUP, NONE_GROUP
from layer2.ingestion.timestamp_sync import format_timestamp_seconds


def make_test_tracklet(
    scenario_id: str,
    camera_id: str,
    track_id: int,
    start_sync_sec: float,
    end_sync_sec: float,
    vehicle_type: str = "car",
    reid_group: str = AICITY_GROUP,
    has_emb: bool = True,
    has_ocr: bool = True,
    fps: float = 10.0,
) -> CanonicalTracklet:
    """Helper to construct a valid CanonicalTracklet for testing."""
    emb = [0.0] * 512 if has_emb else None
    if has_emb:
        emb[0] = 1.0

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
            trajectory_pixels=[[100, 100]],
            trajectory_length=1,
            average_velocity_px=10.0,
            direction="north",
        ),
        appearance=AppearanceData(
            has_embedding=has_emb,
            appearance_embedding=emb,
            embedding_dim=512 if has_emb else None,
            embedding_quality=0.9 if has_emb else None,
            reid_model="osnet_x0_25_aicity" if reid_group == AICITY_GROUP else "osnet_x0_25_msmt17",
            reid_compatibility_group=reid_group if has_emb else NONE_GROUP,
        ),
        vehicle=VehicleData(
            vehicle_type=vehicle_type,
            average_detector_confidence=0.95,
        ),
        anpr=ANPRData(
            has_plate_detection=has_ocr,
            has_readable_ocr=has_ocr,
            aggregated_plate_text="ABC1234" if has_ocr else None,
            ocr_confidence=0.9 if has_ocr else None,
            ocr_readings_count=5 if has_ocr else 0,
            plate_detections_count=5 if has_ocr else 0,
            ocr_consensus_ratio=1.0 if has_ocr else None,
        ),
        spatial=SpatialData(
            camera_latitude=42.52554,
            camera_longitude=-90.72348,
            camera_bearing_deg=335.0,
            camera_confidence="HIGH",
            road_context={"road": "JFK Rd"},
        ),
        quality=QualityData(
            camera_reliability=1.0,
            missing_evidence=[] if has_emb else ["appearance_embedding"],
        ),
    )


@pytest.fixture
def generator() -> CandidateGenerator:
    return CandidateGenerator(
        graph_path="UrbanTrack_Member1_Handoff 2/data/config/camera_graph.json",
        locations_path="UrbanTrack_Member1_Handoff 2/data/config/camera_locations.json",
        max_speed_mps=45.0,
    )


# 1. Cross-scenario isolation
def test_cross_scenario_isolation(generator: CandidateGenerator):
    t_s01 = make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0)
    t_s02 = make_test_tracklet("S02", "CAM_S02_C006", 1, 10.0, 15.0)

    cand, rej = generator.evaluate_pair(t_s01, t_s02)
    assert cand is None
    assert rej is not None
    assert rej.reason_code == "REJECT_CROSS_SCENARIO"
    assert "Incompatible scenarios" in rej.reason_detail

    # Separate camera entities with same numeric ID
    t_s03_c10 = make_test_tracklet("S03", "CAM_S03_C010", 1, 0.0, 5.0)
    t_s05_c10 = make_test_tracklet("S05", "CAM_S05_C010", 1, 10.0, 15.0)
    cand2, rej2 = generator.evaluate_pair(t_s03_c10, t_s05_c10)
    assert cand2 is None
    assert rej2 is not None
    assert rej2.reason_code == "REJECT_CROSS_SCENARIO"


# 2. Same-camera exclusion
def test_same_camera_exclusion(generator: CandidateGenerator):
    t1 = make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0)
    t2 = make_test_tracklet("S01", "CAM_S01_C001", 2, 10.0, 15.0)

    cand, rej = generator.evaluate_pair(t1, t2)
    assert cand is None
    assert rej is not None
    assert rej.reason_code == "REJECT_SAME_CAMERA"
    assert rej.status == "REJECTED"


# 3. Chronological ordering
def test_chronological_ordering(generator: CandidateGenerator):
    t_early = make_test_tracklet("S01", "CAM_S01_C001", 1, 10.0, 15.0)
    t_late = make_test_tracklet("S01", "CAM_S01_C002", 2, 20.0, 25.0)

    # Calling (early, late)
    cand1, _ = generator.evaluate_pair(t_early, t_late)
    assert cand1 is not None
    assert cand1.origin_tracklet_id == t_early.canonical_id
    assert cand1.destination_tracklet_id == t_late.canonical_id

    # Calling (late, early)
    cand2, _ = generator.evaluate_pair(t_late, t_early)
    assert cand2 is not None
    assert cand2.origin_tracklet_id == t_early.canonical_id
    assert cand2.destination_tracklet_id == t_late.canonical_id
    assert cand2.chronology["delta_t_seconds"] == cand1.chronology["delta_t_seconds"]


# 4. Reverse alphabetical ordering regression test
def test_reverse_alphabetical_ordering_regression(generator: CandidateGenerator):
    # Historical bug: C002 -> C001 traffic was inverted because C001 < C002 alphabetically
    t_c002 = make_test_tracklet("S01", "CAM_S01_C002", 10, 5.0, 8.0)
    t_c001 = make_test_tracklet("S01", "CAM_S01_C001", 20, 12.0, 16.0)

    # Even though C001 is alphabetically earlier, C002 started first in sync time
    cand, rej = generator.evaluate_pair(t_c001, t_c002)
    assert rej is None
    assert cand is not None
    assert cand.origin_tracklet_id == t_c002.canonical_id
    assert cand.destination_tracklet_id == t_c001.canonical_id
    assert cand.chronology["delta_t_seconds"] == 4.0  # 12.0 - 8.0 = 4.0s > 0
    assert cand.candidate_status == "CANDIDATE"


# 5. Overlapping camera / FOV admission
def test_overlapping_camera_fov_admission(generator: CandidateGenerator):
    # Cameras C001 and C002 in S01 are adjacent intersection cameras (d = 33.2m, graph edge exists)
    # Tracklets active concurrently: origin [0.0, 10.0], destination [5.0, 12.0]
    # delta_t = 5.0 - 10.0 = -5.0s (overlap duration = 5.0s)
    t_orig = make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 10.0)
    t_dest = make_test_tracklet("S01", "CAM_S01_C002", 2, 5.0, 12.0)

    cand, rej = generator.evaluate_pair(t_orig, t_dest)
    assert rej is None
    assert cand is not None
    assert cand.candidate_status == "CANDIDATE"
    assert cand.chronology["is_overlapping"] is True
    assert cand.chronology["overlap_duration_seconds"] == 5.0
    assert cand.spatial["speed_feasible"] is True


# 5b. Negative chronology rejection for distant impossible pairs
def test_negative_chronology_for_distant_impossible_pairs(generator: CandidateGenerator):
    # Cameras C016 and C025 in S04 are separated by ~1400m (no direct edge)
    # Tracklets overlapping: origin [0.0, 10.0], destination [5.0, 12.0]
    # delta_t = 5.0 - 10.0 = -5.0s
    t_orig = make_test_tracklet("S04", "CAM_S04_C016", 1, 0.0, 10.0)
    t_dest = make_test_tracklet("S04", "CAM_S04_C025", 2, 5.0, 12.0)

    cand, rej = generator.evaluate_pair(t_orig, t_dest)
    assert cand is None
    assert rej is not None
    assert rej.reason_code == "REJECT_NEGATIVE_TIME"
    assert "physically impossible across separated cameras" in rej.reason_detail


# 5c. Valid short-gap transition across adjacent camera FOV boundary
def test_valid_short_gap_adjacent_boundary_transition(generator: CandidateGenerator):
    # Cameras C001 and C002 in S01 are separated by 33.2m
    # Vehicle crosses intersection boundary in 0.15s: delta_t = 0.15s
    # In baseline, 33.2 / 0.15 = 221.3 m/s triggered false speed rejection.
    # In geometry-aware gate, adjacent cameras (d <= 50m) admit boundary handoff!
    t_orig = make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0)
    t_dest = make_test_tracklet("S01", "CAM_S01_C002", 2, 5.15, 10.0)

    cand, rej = generator.evaluate_pair(t_orig, t_dest)
    assert rej is None
    assert cand is not None
    assert cand.candidate_status == "CANDIDATE"
    assert cand.chronology["delta_t_seconds"] == 0.15
    assert cand.spatial["speed_feasible"] is True


# 6. Non-overlapping sequential case
def test_positive_time_acceptance(generator: CandidateGenerator):
    t_orig = make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0)
    t_dest = make_test_tracklet("S01", "CAM_S01_C002", 2, 9.0, 14.0)

    # Distance between C001 and C002 in S01 is 33.2m
    # delta_t = 9.0 - 5.0 = 4.0s -> implied speed = 33.2 / 4.0 = 8.3 m/s (~30 km/h)
    cand, rej = generator.evaluate_pair(t_orig, t_dest)
    assert rej is None
    assert cand is not None
    assert cand.chronology["delta_t_seconds"] == 4.0
    assert cand.spatial["speed_feasible"] is True
    assert cand.candidate_status == "CANDIDATE"


# 7. One-way camera edge direction
def test_one_way_camera_edge_direction(generator: CandidateGenerator):
    # In S02: CAM_S02_C007 -> CAM_S02_C009 is a directed one-way edge
    topo = generator.topology_gate
    assert topo.has_directed_edge("CAM_S02_C007", "CAM_S02_C009") is True
    assert topo.has_directed_edge("CAM_S02_C009", "CAM_S02_C007") is False

    t_c007 = make_test_tracklet("S02", "CAM_S02_C007", 1, 0.0, 4.0)
    t_c009 = make_test_tracklet("S02", "CAM_S02_C009", 2, 8.0, 12.0)

    cand_fwd, _ = generator.evaluate_pair(t_c007, t_c009)
    assert cand_fwd is not None
    assert cand_fwd.topology["has_directed_edge"] is True
    assert cand_fwd.topology["edge_distance_m"] is not None

    # Reverse direction: C009 -> C007
    t_c009_early = make_test_tracklet("S02", "CAM_S02_C009", 3, 0.0, 4.0)
    t_c007_late = make_test_tracklet("S02", "CAM_S02_C007", 4, 8.0, 12.0)

    cand_rev, _ = generator.evaluate_pair(t_c009_early, t_c007_late)
    assert cand_rev is not None
    assert cand_rev.topology["has_directed_edge"] is False
    assert cand_rev.topology["edge_distance_m"] is None


# 8. No-edge does not automatically reject
def test_no_edge_does_not_automatically_reject(generator: CandidateGenerator):
    # CAM_S02_C009 -> CAM_S02_C007 has NO graph edge
    t_c009 = make_test_tracklet("S02", "CAM_S02_C009", 1, 0.0, 4.0)
    t_c007 = make_test_tracklet("S02", "CAM_S02_C007", 2, 8.0, 12.0)

    cand, rej = generator.evaluate_pair(t_c009, t_c007)
    assert rej is None
    assert cand is not None
    assert cand.topology["has_directed_edge"] is False
    assert cand.spatial["speed_feasible"] is True
    assert cand.candidate_status == "CANDIDATE"


# 9. Excessive speed rejection
def test_excessive_speed_rejection(generator: CandidateGenerator):
    # Distance between CAM_S04_C016 and CAM_S04_C025 is ~1400 meters
    # If delta_t = 1.0s, implied speed = 1400 m/s >> 45.0 m/s
    t_orig = make_test_tracklet("S04", "CAM_S04_C016", 1, 0.0, 4.0)
    t_dest = make_test_tracklet("S04", "CAM_S04_C025", 2, 5.0, 9.0)  # delta_t = 1.0s

    cand, rej = generator.evaluate_pair(t_orig, t_dest)
    assert cand is None
    assert rej is not None
    assert rej.reason_code == "REJECT_EXCESSIVE_SPEED"
    assert "exceeds maximum speed threshold" in rej.reason_detail


# 10. Valid speed candidate
def test_valid_speed_candidate(generator: CandidateGenerator):
    # Distance between CAM_S01_C001 and CAM_S01_C003 is ~21.6m
    # delta_t = 2.0s -> speed = 10.8 m/s (38.9 km/h) <= 45.0 m/s
    t_orig = make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0)
    t_dest = make_test_tracklet("S01", "CAM_S01_C003", 2, 7.0, 12.0)

    cand, rej = generator.evaluate_pair(t_orig, t_dest)
    assert rej is None
    assert cand is not None
    assert cand.spatial["implied_speed_mps"] == 10.8
    assert cand.spatial["speed_feasible"] is True
    assert cand.candidate_status == "CANDIDATE"


# 11. Missing embedding handling
def test_missing_embedding_handling(generator: CandidateGenerator):
    t_emb = make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0, has_emb=True)
    t_no_emb = make_test_tracklet("S01", "CAM_S01_C003", 2, 8.0, 13.0, has_emb=False)

    cand, rej = generator.evaluate_pair(t_emb, t_no_emb)
    assert rej is None
    assert cand is not None
    assert cand.appearance["origin_available"] is True
    assert cand.appearance["destination_available"] is False
    assert cand.appearance["compatible"] is False
    assert cand.candidate_status == "CANDIDATE"


# 12. Incompatible ReID handling
def test_incompatible_reid_handling(generator: CandidateGenerator):
    # S01_C002 is MSMT17, S01_C001 is AICITY
    t_c002 = make_test_tracklet("S01", "CAM_S01_C002", 1, 0.0, 5.0, reid_group=MSMT17_GROUP)
    t_c001 = make_test_tracklet("S01", "CAM_S01_C001", 2, 9.0, 14.0, reid_group=AICITY_GROUP)

    cand, rej = generator.evaluate_pair(t_c002, t_c001)
    assert rej is None
    assert cand is not None
    assert cand.appearance["origin_available"] is True
    assert cand.appearance["destination_available"] is True
    assert cand.appearance["compatible"] is False  # Prohibits cosine similarity
    assert cand.candidate_status == "CANDIDATE"


# 13. Vehicle-type disagreement does not automatically erase candidate
def test_vehicle_type_disagreement_tolerance(generator: CandidateGenerator):
    t_car = make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0, vehicle_type="car")
    t_truck = make_test_tracklet("S01", "CAM_S01_C003", 2, 8.0, 13.0, vehicle_type="truck")

    cand, rej = generator.evaluate_pair(t_car, t_truck)
    assert rej is None
    assert cand is not None
    assert cand.vehicle_type["origin"] == "car"
    assert cand.vehicle_type["destination"] == "truck"
    assert cand.vehicle_type["agreement"] == "MISMATCH"
    assert cand.candidate_status == "CANDIDATE"


# 14. Deterministic output
def test_deterministic_output(generator: CandidateGenerator):
    t1 = make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0)
    t2 = make_test_tracklet("S01", "CAM_S01_C003", 2, 8.0, 13.0)
    t3 = make_test_tracklet("S01", "CAM_S01_C004", 3, 10.0, 15.0)

    tracklets = [t3, t1, t2]  # Unordered
    cands1, rejs1 = generator.generate_candidates_for_tracklets(list(tracklets))
    cands2, rejs2 = generator.generate_candidates_for_tracklets(list(tracklets))

    assert len(cands1) == len(cands2)
    assert len(rejs1) == len(rejs2)
    assert [c.candidate_pair_id for c in cands1] == [c.candidate_pair_id for c in cands2]
    assert [r.candidate_pair_id for r in rejs1] == [r.candidate_pair_id for r in rejs2]


# 15. Rejection reason completeness
def test_rejection_reason_completeness(generator: CandidateGenerator):
    valid_reasons = {
        "REJECT_SAME_CAMERA",
        "REJECT_CROSS_SCENARIO",
        "REJECT_NEGATIVE_TIME",
        "REJECT_ZERO_OR_INVALID_TIME",
        "REJECT_EXCESSIVE_SPEED",
        "REJECT_EXCEED_MAX_TIME",
    }

    # Generate a variety of rejections
    pairs_to_test = [
        # Cross scenario
        (make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0),
         make_test_tracklet("S02", "CAM_S02_C006", 2, 10.0, 15.0)),
        # Same camera
        (make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0),
         make_test_tracklet("S01", "CAM_S01_C001", 2, 10.0, 15.0)),
        # Negative time across separated cameras
        (make_test_tracklet("S04", "CAM_S04_C016", 1, 0.0, 10.0),
         make_test_tracklet("S04", "CAM_S04_C025", 2, 5.0, 12.0)),
        # Instantaneous transit across separated cameras
        (make_test_tracklet("S04", "CAM_S04_C016", 1, 0.0, 5.0),
         make_test_tracklet("S04", "CAM_S04_C025", 2, 5.0, 10.0)),
        # Excessive speed
        (make_test_tracklet("S04", "CAM_S04_C016", 1, 0.0, 4.0),
         make_test_tracklet("S04", "CAM_S04_C025", 2, 4.05, 8.0)),
        # Exceed maximum temporal limit (> 120s)
        (make_test_tracklet("S01", "CAM_S01_C001", 1, 0.0, 5.0),
         make_test_tracklet("S01", "CAM_S01_C002", 2, 150.0, 155.0)),
    ]

    for t_a, t_b in pairs_to_test:
        cand, rej = generator.evaluate_pair(t_a, t_b)
        assert cand is None
        assert rej is not None
        assert rej.status == "REJECTED"
        assert rej.reason_code in valid_reasons
        assert isinstance(rej.reason_detail, str) and len(rej.reason_detail) > 0
        assert isinstance(rej.candidate_pair_id, str) and len(rej.candidate_pair_id) > 0
