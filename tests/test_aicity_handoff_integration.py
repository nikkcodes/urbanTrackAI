"""
UrbanTrack AI — Test Suite: AI City Challenge 2022 Track 1 Handoff Integration.

Validates:
1. Multi-camera folder discovery (CAM_S01_C001, CAM_S01_C002, CAM_S01_C003)
2. Canonical Observation schema mapping and provenance metadata
3. Authoritative video-relative timestamp semantics (no fake UTC conversion)
4. Dynamic appearance embedding validation (512-D, unit norm, non-zero, finite)
5. Re-ID model provenance and C002 msmt17 cross-model compatibility guard
6. Zero data fabrication enforcement (plates, GPS, calibration, sync offsets)
7. Unsynchronized candidate pair generation
8. MultiCameraFeedAdapter registration and ingestion contracts
9. Unsupervised identity clustering and image-space trajectory reconstruction
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List
import pytest

from inference.candidate_generation import CandidateGenerator
from inference.identity_fusion import match_observations
from inference.identity_graph import IdentityGraph
from inference.observation_loader import (
    DatasetClassification,
    MultiCameraFeedAdapter,
    load_aicity_member1_camera,
    load_aicity_member1_feed,
)
from inference.road_graph import RoadGraph
from inference.similarity import are_reid_models_compatible
from inference.trajectory_engine import reconstruct_identity_trajectory
from schemas.observation_schema import Observation

HANDOFF_OUTPUT_DIR = Path("UrbanTrack_Member1_Handoff/output")


@pytest.fixture(scope="module")
def handoff_observations() -> List[Observation]:
    """Module-level fixture loading all observations from the 3-camera AI City handoff."""
    if not HANDOFF_OUTPUT_DIR.is_dir():
        pytest.skip(f"Member 1 handoff output directory not found at {HANDOFF_OUTPUT_DIR}")
    return load_aicity_member1_feed(output_dir=HANDOFF_OUTPUT_DIR, fps=10.0)


# =============================================================================
# 1. DISCOVERY & INGESTION TESTS
# =============================================================================

def test_three_camera_directory_discovery(handoff_observations: List[Observation]):
    """
    Ensure all 3 camera streams are discovered and ingested directly from the directory,
    even though CAM_S01_C003 was missing from Member 1's aicity_index.json.
    """
    discovered_cams = sorted(list({o.camera_id for o in handoff_observations}))
    expected_cams = ["CAM_S01_C001", "CAM_S01_C002", "CAM_S01_C003"]
    assert discovered_cams == expected_cams, (
        f"Expected {expected_cams}, but discovered {discovered_cams}."
    )
    assert len(handoff_observations) > 0, "No observations loaded from handoff."


def test_observation_schema_compliance(handoff_observations: List[Observation]):
    """
    Verify all ingested tracklets map strictly to canonical Observation schema
    with valid IDs, frame numbers, and source provenance.
    """
    for obs in handoff_observations:
        assert isinstance(obs, Observation)
        assert obs.observation_id.startswith(obs.camera_id)
        assert obs.camera_id in ["CAM_S01_C001", "CAM_S01_C002", "CAM_S01_C003"]
        assert obs.frame_id is not None and obs.frame_id >= 0
        assert obs.track_id is not None
        assert obs.vehicle_type in ["car", "truck", "bus", "suv", "vehicle"]

        # Provenance audit
        prov = obs.source_provenance
        assert isinstance(prov, dict), f"Missing source provenance for {obs.observation_id}"
        assert prov.get("source_dataset") == "AICity_2022_Track1_Member1_Handoff"
        assert prov.get("camera_id") == obs.camera_id
        assert prov.get("track_id") == int(obs.track_id)
        assert "reid_model" in prov


# =============================================================================
# 2. TIMESTAMP SEMANTICS TESTS
# =============================================================================

def test_authoritative_video_relative_timestamp_semantics(handoff_observations: List[Observation]):
    """
    Verify timestamp semantics adhere strictly to user requirements:
    - Authoritative time: timestamp_seconds = frame_number / fps
    - timestamp_semantics = 'video_relative'
    - time_reference_id = camera_id (independent timelines, no assumed synchrony)
    - clock_offset_seconds is None (no fake synchronization offset fabricated)
    """
    fps = 10.0
    for obs in handoff_observations:
        expected_ts_sec = round(float(obs.frame_id) / fps, 4)
        assert abs(obs.timestamp_seconds - expected_ts_sec) < 1e-3, (
            f"Timestamp {obs.timestamp_seconds} does not match frame {obs.frame_id} / {fps}"
        )
        assert obs.timestamp_semantics == "video_relative", (
            f"Expected timestamp_semantics='video_relative', got '{obs.timestamp_semantics}'"
        )
        assert obs.time_reference_id == obs.camera_id, (
            f"time_reference_id must reflect camera-local clock ({obs.camera_id}), got {obs.time_reference_id}"
        )
        assert obs.clock_offset_seconds is None, (
            "No clock_offset_seconds should be fabricated without ground-truth synchronization."
        )


# =============================================================================
# 3. DYNAMIC EMBEDDING AUDIT TESTS
# =============================================================================

def test_dynamic_embedding_validation(handoff_observations: List[Observation]):
    """
    Dynamically validate all appearance embeddings without hardcoding static dataset counts.
    Criteria:
    - Dimension is exactly 512
    - All values are finite (no NaN, no Inf)
    - Non-zero vector (norm > 1e-6)
    - Normalization tolerance: unit L2 norm within reasonable margin (|norm - 1.0| < 0.02)
    - Valid track_id association
    - Provenance contains embedding metadata
    """
    total_embeddings = 0
    cam_emb_counts: Dict[str, int] = {}

    for obs in handoff_observations:
        emb = obs.appearance_embedding
        if emb is None:
            continue

        total_embeddings += 1
        cam_emb_counts[obs.camera_id] = cam_emb_counts.get(obs.camera_id, 0) + 1

        # 1. Dimension
        assert len(emb) == 512, f"Embedding dimension for {obs.observation_id} is {len(emb)}, expected 512"

        # 2. Finite values
        for idx, val in enumerate(emb):
            assert not math.isnan(val), f"NaN found in {obs.observation_id} at index {idx}"
            assert not math.isinf(val), f"Inf found in {obs.observation_id} at index {idx}"

        # 3. Non-zero & L2 unit normalization
        l2_norm = math.sqrt(sum(v * v for v in emb))
        assert l2_norm > 1e-6, f"All-zero embedding vector in {obs.observation_id}"
        assert abs(l2_norm - 1.0) < 0.02, (
            f"Embedding vector for {obs.observation_id} is not unit normalized: L2 norm = {l2_norm}"
        )

        # 4. Valid track association
        assert obs.track_id is not None

        # 5. Provenance metadata
        prov = obs.source_provenance or {}
        assert prov.get("embedding_dimension") == 512

    assert total_embeddings > 0, "Expected at least one valid appearance embedding across the handoff."
    # Report observed counts as diagnostic log, not rigid hardcoded requirement
    print(f"\n[Dynamic Embedding Audit] Total validated 512-D vectors: {total_embeddings} across cameras: {cam_emb_counts}")


# =============================================================================
# 4. RE-ID MODEL COMPATIBILITY & C002 GUARD TESTS
# =============================================================================

def test_reid_model_provenance(handoff_observations: List[Observation]):
    """
    Verify model provenance is correctly tagged per camera:
    - CAM_S01_C001: osnet_x0_25_aicity
    - CAM_S01_C002: osnet_x0_25_msmt17
    - CAM_S01_C003: osnet_x0_25_aicity
    """
    for obs in handoff_observations:
        prov = obs.source_provenance or {}
        model = prov.get("reid_model")
        if obs.camera_id in ("CAM_S01_C001", "CAM_S01_C003"):
            assert model == "osnet_x0_25_aicity", (
                f"Camera {obs.camera_id} must have reid_model='osnet_x0_25_aicity', got '{model}'"
            )
        elif obs.camera_id == "CAM_S01_C002":
            assert model == "osnet_x0_25_msmt17", (
                f"Camera {obs.camera_id} must have reid_model='osnet_x0_25_msmt17', got '{model}'"
            )


def test_reid_model_compatibility_function():
    """Verify the model compatibility predicate correctly rejects cross-model pairings."""
    assert are_reid_models_compatible("osnet_x0_25_aicity", "osnet_x0_25_aicity") is True
    assert are_reid_models_compatible("osnet_x0_25_msmt17", "osnet_x0_25_msmt17") is True
    assert are_reid_models_compatible("osnet_x0_25_aicity", "osnet_x0_25_msmt17") is False
    assert are_reid_models_compatible("osnet_x0_25_msmt17", "osnet_x0_25_aicity") is False
    assert are_reid_models_compatible(None, "osnet_x0_25_aicity") is False


def test_c002_reid_model_guard_in_identity_fusion(handoff_observations: List[Observation]):
    """
    Verify the C002 compatibility guard blocks cross-model Re-ID comparison between
    C002 (msmt17) and C001/C003 (aicity).
    """
    obs_c001 = next(o for o in handoff_observations if o.camera_id == "CAM_S01_C001" and o.appearance_embedding is not None)
    obs_c002 = next(o for o in handoff_observations if o.camera_id == "CAM_S01_C002" and o.appearance_embedding is not None)
    obs_c003 = next(o for o in handoff_observations if o.camera_id == "CAM_S01_C003" and o.appearance_embedding is not None)

    # 1. C001 (aicity) vs C002 (msmt17) -> Must trigger incompatible_models guard
    res_1_2 = match_observations(obs_c001, obs_c002)
    ev_1_2 = res_1_2.get("evidence", {})
    assert ev_1_2.get("appearance_status") == "incompatible_models", (
        f"Expected appearance_status='incompatible_models', got '{ev_1_2.get('appearance_status')}'"
    )
    assert ev_1_2.get("appearance_similarity") is None

    # 2. C002 (msmt17) vs C003 (aicity) -> Must trigger incompatible_models guard
    res_2_3 = match_observations(obs_c002, obs_c003)
    ev_2_3 = res_2_3.get("evidence", {})
    assert ev_2_3.get("appearance_status") == "incompatible_models"
    assert ev_2_3.get("appearance_similarity") is None

    # 3. C001 (aicity) vs C003 (aicity) -> Compatible models! Re-ID must be evaluated
    res_1_3 = match_observations(obs_c001, obs_c003)
    ev_1_3 = res_1_3.get("evidence", {})
    assert ev_1_3.get("appearance_status") == "available", (
        f"Expected appearance_status='available', got '{ev_1_3.get('appearance_status')}'"
    )
    assert ev_1_3.get("appearance_similarity") is not None
    assert 0.0 <= ev_1_3.get("appearance_similarity") <= 1.0


# =============================================================================
# 5. ZERO DATA FABRICATION TESTS
# =============================================================================

def test_zero_data_fabrication_integrity(handoff_observations: List[Observation]):
    """
    Enforce strict zero-fabrication guarantees:
    - plate text is 100% None (never synthesized)
    - latitude and longitude are 100% None (never synthesized)
    - point_coordinate_system is explicitly 'image'
    - clock_offset_seconds is None
    - plate_bbox and plate_confidence are preserved from perception without hallucinated characters
    """
    for obs in handoff_observations:
        assert obs.plate is None, f"Fabricated plate found on {obs.observation_id}: {obs.plate}"
        assert obs.plate_text is None, f"Fabricated plate_text found on {obs.observation_id}: {obs.plate_text}"
        assert obs.latitude is None, f"Fabricated latitude on {obs.observation_id}: {obs.latitude}"
        assert obs.longitude is None, f"Fabricated longitude on {obs.observation_id}: {obs.longitude}"
        assert obs.point_coordinate_system == "image", (
            f"Coordinate system must be 'image', got '{obs.point_coordinate_system}'"
        )
        assert obs.clock_offset_seconds is None

    # Verify plate bboxes are preserved from perception
    obs_with_plate_bbox = [o for o in handoff_observations if o.plate_bbox is not None]
    assert len(obs_with_plate_bbox) > 0, "Expected perception plate_bbox values to be preserved."
    for o in obs_with_plate_bbox:
        assert len(o.plate_bbox) == 4
        assert o.plate is None  # Bbox exists, but plate string remains strictly None


# =============================================================================
# 6. UNSYNCHRONIZED CANDIDATE GENERATION TESTS
# =============================================================================

def test_unsynchronized_candidate_generation(handoff_observations: List[Observation]):
    """
    Verify candidate generator in unsynchronized_mode:
    - Generates intra-camera candidate pairs bounded by temporal window
    - Generates cross-camera candidate pairs across independent camera clocks
    - Prunes incompatible vehicle types and missing identity evidence
    - Does not falsely reject pairs due to non-existent cross-camera synchrony
    """
    generator = CandidateGenerator(
        max_time_window_seconds=7200.0,
        min_probability_threshold=0.65,
        unsynchronized_mode=True,
    )
    candidates, rejections = generator.generate_candidates(handoff_observations)

    assert len(candidates) > 0, "Unsynchronized mode failed to generate candidate pairs."

    # Verify both same-camera and cross-camera pairs exist
    same_cam = [c for c in candidates if c[0].camera_id == c[1].camera_id]
    diff_cam = [c for c in candidates if c[0].camera_id != c[1].camera_id]
    assert len(same_cam) > 0, "No same-camera candidate pairs generated."
    assert len(diff_cam) > 0, "No cross-camera candidate pairs generated."

    # Verify vehicle type pruning was active
    for a, b in candidates:
        if a.vehicle_type and b.vehicle_type:
            # Synonyms check (car vs suv might map to car, but never car vs truck)
            assert not (a.vehicle_type == "car" and b.vehicle_type == "truck")


# =============================================================================
# 7. MULTI-CAMERA FEED ADAPTER INTEGRATION
# =============================================================================

def test_feed_adapter_registration():
    """
    Verify MultiCameraFeedAdapter cleanly registers the AI City handoff under
    DatasetClassification.REAL with independent camera feeds.
    """
    adapter = MultiCameraFeedAdapter()
    adapter.register_aicity_handoff(
        handoff_dir=HANDOFF_OUTPUT_DIR,
        camera_ids=["CAM_S01_C001", "CAM_S01_C002", "CAM_S01_C003"],
        fps=10.0,
    )

    active_cams = adapter.get_active_cameras()
    assert sorted(active_cams) == ["CAM_S01_C001", "CAM_S01_C002", "CAM_S01_C003"]

    cfg_c001 = adapter.get_camera_config("CAM_S01_C001")
    assert cfg_c001 is not None
    assert cfg_c001.classification == DatasetClassification.REAL
    assert cfg_c001.coordinate_system == "image"
    assert cfg_c001.time_reference_id == "CAM_S01_C001"

    all_obs = adapter.load_all_observations()
    assert len(all_obs) > 0


# =============================================================================
# 8. IDENTITY GRAPH & IMAGE-SPACE TRAJECTORY INFERENCE
# =============================================================================

def test_identity_graph_and_image_space_trajectories(handoff_observations: List[Observation]):
    """
    Verify identity graph assembly, cluster formation, and image-space trajectory reconstruction.
    """
    # Use a focused subset (e.g. 50 tracklets) for fast end-to-end testing
    subset = handoff_observations[:50]
    generator = CandidateGenerator(
        max_time_window_seconds=7200.0,
        min_probability_threshold=0.65,
        unsynchronized_mode=True,
    )
    candidates, _ = generator.generate_candidates(subset)

    match_results = [match_observations(a, b) for a, b in candidates]

    graph = IdentityGraph(min_score_threshold=0.70)
    graph.assemble_from_pairs(
        observations=subset,
        pairs=candidates,
        match_results=match_results,
    )

    clusters = graph.get_candidate_identities()
    assert len(clusters) > 0, "No identity clusters produced."

    # Trajectory reconstruction
    dummy_road_graph = RoadGraph()
    traj_configs = {"mode": "image_space", "fallback_enabled": True}

    for idx, cl in enumerate(clusters[:5]):
        cl["candidate_vehicle_id"] = f"UT_TEST_{idx:03d}"
        traj = reconstruct_identity_trajectory(
            identity_data=cl,
            road_graph=dummy_road_graph,
            config=traj_configs,
        )
        traj_dict = traj.to_dict()
        assert "complete_route_nodes" in traj_dict
        assert "cameras_visited" in traj_dict
        assert "observations_count" in traj_dict
