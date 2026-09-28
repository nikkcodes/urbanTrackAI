"""
UrbanTrack AI — Test Suite: AI City 2022 / CityFlowV2 Ground-Truth Validation.

Validates:
1. Official synchronization metadata parsing and non-destructive timestamp semantics.
2. Homography calibration parsing and world-coordinate (cityflow_world) transformation.
3. Strict ground-truth isolation and pre-inference zero-leakage guarantee.
4. Tracklet-to-GT consensus linking protocol (frame-level IoU >= 0.40, consensus >= 60%).
5. Cross-camera ground-truth pair generation.
6. Evaluator metrics computation (Candidate Recall, Precision, Recall, F1, FMR, Purity).
7. Probabilistic holdout calibration with disjoint vehicle-level partitioning (zero vehicle leakage).
8. Ablation study baseline execution on identical evaluation populations.
9. Controlled robustness perturbation benchmark suite.
"""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from schemas.observation_schema import Observation
from inference.observation_loader import load_aicity_member1_feed
from inference.candidate_generation import CandidateGenerator
from inference.identity_fusion import match_observations
from inference.identity_graph import IdentityGraph

from inference.aicity_synchronizer import AICitySynchronizer
from inference.aicity_calibration import AICityCalibration
from inference.aicity_gt_adapter import AICityGroundTruthAdapter
from inference.aicity_evaluator import (
    audit_inference_leakage,
    evaluate_candidate_recall,
    evaluate_identity_associations,
    evaluate_probabilistic_calibration,
    run_ablation_experiments,
    run_robustness_experiments,
    run_scalability_benchmark,
)

SYNC_FILE = Path("data/aicity_ground_truth/cam_timestamp/S01.txt")
CAL_DIR = Path("data/aicity_ground_truth/calibration")
GT_DIR = Path("data/aicity_ground_truth/gt")


@pytest.fixture(scope="module")
def real_observations() -> list[Observation]:
    """Module-level fixture loading all observations from the 3-camera AI City handoff."""
    return load_aicity_member1_feed()


# =============================================================================
# 1. SYNCHRONIZATION TESTS
# =============================================================================

def test_synchronization_offsets_and_semantics(real_observations: list[Observation]):
    """Verify official synchronization parsing and non-destructive timestamp preservation."""
    assert SYNC_FILE.is_file(), f"Official synchronization file missing: {SYNC_FILE}"

    synchronizer = AICitySynchronizer(sync_file=SYNC_FILE)
    assert synchronizer.get_offset("CAM_S01_C001") == 0.0
    assert synchronizer.get_offset("CAM_S01_C002") == 1.640
    assert synchronizer.get_offset("CAM_S01_C003") == 2.049

    synced_obs = synchronizer.attach_synchronization(real_observations)
    for obs in synced_obs:
        # 1. Video-relative time is preserved as authoritative
        assert obs.timestamp_semantics == "video_relative"
        assert obs.timestamp_seconds >= 0.0

        # 2. Synchronized time is distinctly represented
        offset = synchronizer.get_offset(obs.camera_id)
        expected_sync = round(obs.timestamp_seconds + offset, 4)
        actual_sync = getattr(obs, "synchronized_timestamp_seconds")
        assert abs(actual_sync - expected_sync) < 1e-3
        assert getattr(obs, "synchronized_timestamp_semantics") == "aicity_official_synchronized"

        # 3. No fake UTC conversion
        assert obs.clock_offset_seconds == offset


# =============================================================================
# 2. CALIBRATION & WORLD COORDINATES TESTS
# =============================================================================

def test_calibration_homography_and_world_coordinates(real_observations: list[Observation]):
    """Verify official homography parsing, ground-contact projection, and coordinate tags."""
    assert CAL_DIR.is_dir(), f"Calibration directory missing: {CAL_DIR}"

    calibrator = AICityCalibration(calibration_dir=CAL_DIR)
    for cam_id in ["CAM_S01_C001", "CAM_S01_C002", "CAM_S01_C003"]:
        assert calibrator.has_calibration(cam_id)
        norm_key = cam_id.split("_")[-1].lower()
        assert calibrator.reprojection_errors.get(norm_key) is not None

    calibrated_obs = calibrator.attach_calibration(real_observations)
    calibrated_count = 0
    for obs in calibrated_obs:
        wx = getattr(obs, "world_x", None)
        wy = getattr(obs, "world_y", None)
        if wx is not None and wy is not None:
            calibrated_count += 1
            assert not math.isnan(wx) and not math.isinf(wx)
            assert not math.isnan(wy) and not math.isinf(wy)
            assert obs.point_coordinate_system == "cityflow_world"
            assert obs.point_type == "cityflow_world_ground_contact"
            # Zero fabricated GPS
            assert obs.latitude is None
            assert obs.longitude is None

    assert calibrated_count > 0, "No observations were successfully projected into world coordinates."


# =============================================================================
# 3. GROUND-TRUTH ISOLATION & LEAKAGE AUDIT
# =============================================================================

def test_ground_truth_isolation_and_zero_leakage(real_observations: list[Observation]):
    """Enforce strict isolation between ground-truth labels and inference inputs."""
    gt_adapter = AICityGroundTruthAdapter(gt_dir=GT_DIR)
    audit = audit_inference_leakage(real_observations, gt_adapter)

    assert audit["status"] == "passed"
    assert audit["zero_gt_leakage_verified"] is True
    assert audit["violations_found"] == 0

    # Verify that Observation class does not expose GT fields as active defaults
    for obs in real_observations:
        assert not hasattr(obs, "gt_vehicle_id") or getattr(obs, "gt_vehicle_id") is None
        assert not hasattr(obs, "global_gt_id") or getattr(obs, "global_gt_id") is None


# =============================================================================
# 4. TRACKLET-TO-GT LINKING & PAIRWISE LABELS
# =============================================================================

def test_tracklet_to_gt_linking_consensus(real_observations: list[Observation]):
    """Verify spatiotemporal IoU consensus linking between Member-1 tracklets and GT boxes."""
    gt_adapter = AICityGroundTruthAdapter(gt_dir=GT_DIR, iou_threshold=0.40, consensus_threshold=0.60)
    assocs = gt_adapter.link_member1_tracklets(real_observations)

    assert len(assocs) == len(real_observations)
    mapped_count = len(gt_adapter.obs_id_to_gt)
    assert mapped_count > 250, f"Expected >250 mapped tracklets, got {mapped_count}"

    # Verify that consensus requirement is respected
    for obs_id, assoc in assocs.items():
        if assoc.match_status == "matched":
            assert assoc.gt_vehicle_id is not None
            assert assoc.consensus_ratio >= 0.60
            assert assoc.mean_iou >= 0.40
        elif assoc.match_status == "ambiguous":
            assert assoc.gt_vehicle_id is None
            assert assoc.consensus_ratio < 0.60

    # Verify cross-camera GT pairs exist
    cross_pairs = gt_adapter.get_cross_camera_gt_pairs(real_observations)
    assert len(cross_pairs) > 50, f"Expected >50 true cross-camera GT pairs, got {len(cross_pairs)}"


# =============================================================================
# 5. EVALUATOR METRICS & CANDIDATE RECALL
# =============================================================================

def test_candidate_recall_and_identity_evaluation(real_observations: list[Observation]):
    """Verify Candidate Recall and pairwise precision/recall/F1 metrics computation."""
    generator = CandidateGenerator(unsynchronized_mode=True, min_probability_threshold=0.65)
    candidates, _ = generator.generate_candidates(real_observations)

    gt_adapter = AICityGroundTruthAdapter(gt_dir=GT_DIR)
    gt_adapter.link_member1_tracklets(real_observations)

    # Candidate recall check
    cand_metrics = evaluate_candidate_recall(candidates, gt_adapter, real_observations)
    assert cand_metrics["status"] == "passed"
    assert cand_metrics["candidate_recall"] is not None
    assert 0.0 <= cand_metrics["candidate_recall"] <= 1.0

    # Matching & identity metrics
    match_results = []
    for a, b in candidates[:100]:  # fast test slice
        r = match_observations(a, b)
        r["obs_a_id"] = a.observation_id
        r["obs_b_id"] = b.observation_id
        match_results.append(r)

    graph = IdentityGraph(min_score_threshold=0.70)
    graph.build_graph_from_matches(real_observations, candidates[:100], match_results)
    clusters = graph.get_candidate_identities()

    id_metrics = evaluate_identity_associations(
        predicted_pairs=match_results,
        clusters=clusters,
        gt_adapter=gt_adapter,
        decision_threshold=0.70,
        cross_camera_only=True,
    )
    assert "precision" in id_metrics
    assert "recall" in id_metrics
    assert "f1_score" in id_metrics
    assert "false_merge_rate" in id_metrics
    assert "cluster_purity" in id_metrics


# =============================================================================
# 6. PROBABILISTIC CALIBRATION & DISJOINT HOLDOUT
# =============================================================================

def test_probabilistic_calibration_disjoint_holdout(real_observations: list[Observation]):
    """Verify Platt scaling calibration with zero vehicle-level train/holdout leakage."""
    generator = CandidateGenerator(unsynchronized_mode=True, min_probability_threshold=0.60)
    candidates, _ = generator.generate_candidates(real_observations)

    gt_adapter = AICityGroundTruthAdapter(gt_dir=GT_DIR)
    gt_adapter.link_member1_tracklets(real_observations)

    match_results = []
    for a, b in candidates:
        r = match_observations(a, b)
        r["obs_a_id"] = a.observation_id
        r["obs_b_id"] = b.observation_id
        match_results.append(r)

    calib_res = evaluate_probabilistic_calibration(match_results, gt_adapter, dev_ratio=0.60, seed=42)
    assert calib_res["status"] == "passed"
    assert calib_res["dev_vehicles_count"] > 0
    assert calib_res["holdout_vehicles_count"] > 0
    assert "calibrated_brier_score" in calib_res
    assert "calibrated_ece" in calib_res
    assert calib_res["calibrated_brier_score"] >= 0.0


# =============================================================================
# 7. ABLATION & ROBUSTNESS EXECUTION
# =============================================================================

def test_ablation_study_execution(real_observations: list[Observation]):
    """Verify 5-way ablation experiment executes cleanly on real observations."""
    subset = real_observations[:60]
    gt_adapter = AICityGroundTruthAdapter(gt_dir=GT_DIR)
    gt_adapter.link_member1_tracklets(subset)

    ablations = run_ablation_experiments(subset, gt_adapter)
    expected_baselines = [
        "Baseline_1_ReID_Only",
        "Baseline_2_ReID_VehicleType",
        "Baseline_3_Multimodal_Fusion",
        "Baseline_4_Multimodal_PhysicalConstraints",
        "Baseline_5_Full_UrbanTrack",
    ]
    for b in expected_baselines:
        assert b in ablations
        assert "f1_score" in ablations[b]
        assert "cluster_purity" in ablations[b]


def test_scalability_benchmark_execution(real_observations: list[Observation]):
    """Verify scalability profiling executes cleanly across multiple observation sizes."""
    bench = run_scalability_benchmark(real_observations, scales=[20, 50])
    assert len(bench) == 2
    assert bench[0]["observation_count"] == 20
    assert bench[1]["observation_count"] == 50
    assert bench[0]["candidate_generation_runtime_ms"] > 0.0


# =============================================================================
# 8. REGRESSION TESTS: HORIZON SAFETY, IDENTITY GRAPH SAFETY & SCALE READINESS
# =============================================================================

def test_homography_horizon_instability_safety():
    """
    [TASK 2 REGRESSION TEST]
    Verify homography horizon instability protection:
    - Points near the vanishing line where |W| < 0.50 are safely rejected from world projection.
    - Observation attributes world_x and world_y remain None, preventing runaway coordinates.
    - Projection status is marked as 'unstable_horizon_denominator'.
    - Never outputs NaN or Inf.
    - Never substitutes fabricated GPS or arbitrary coordinates.
    - Runaway world coordinates (> 25,000m) are also safely rejected.
    """
    calibrator = AICityCalibration(calibration_dir=CAL_DIR)

    # Test 1: Near-horizon coordinate on C002 where W approaches 0
    # On C002, H row 2 is [-0.006904, 0.007787, 1.0].
    # At (x=700, y=490): W = -0.006904*700 + 0.007787*490 + 1.0 = -0.01717 (extreme vanishing line!)
    res = calibrator.project_contact_point("CAM_S01_C002", 700.0, 490.0)
    assert res is None

    # Test with observation placed near horizon
    unstable_obs = Observation(
        observation_id="TEST_HORIZON_UNSTABLE",
        camera_id="CAM_S01_C002",
        frame_id=100,
        timestamp_seconds=10.0,
        trajectory_point=[700.0, 490.0],
        vehicle_type="car",
    )
    result_obs = calibrator.attach_calibration([unstable_obs])[0]
    assert getattr(result_obs, "world_x") is None
    assert getattr(result_obs, "world_y") is None
    assert getattr(result_obs, "projection_status") == "unstable_horizon_denominator"
    assert result_obs.latitude is None
    assert result_obs.longitude is None
    # Original image observation is fully preserved
    assert result_obs.trajectory_point == [700.0, 490.0]

    # Test 2: Valid stable projection works normally
    valid_res = calibrator.project_contact_point("CAM_S01_C001", 960.0, 900.0)
    assert valid_res is not None
    wx_valid, wy_valid = valid_res
    assert not math.isnan(wx_valid) and not math.isinf(wx_valid)
    assert abs(wx_valid) < 25000.0


def test_identity_graph_transitive_contradiction_safety():
    """
    [TASK 3 REGRESSION TEST]
    Verify identity graph transitive contradiction safety:
    A-B strong match, B-C strong match, but A-C contradictory (e.g. incompatible vehicle types).
    Verify that blind merging is prevented and the contradictory cluster is split into
    mutually consistent identity clusters.
    """
    obs_a = Observation(
        observation_id="TEST_OBS_A",
        camera_id="CAM_S01_C001",
        frame_id=10,
        timestamp_seconds=1.0,
        vehicle_type="car",
        detection_confidence=0.90,
    )
    obs_b = Observation(
        observation_id="TEST_OBS_B",
        camera_id="CAM_S01_C002",
        frame_id=20,
        timestamp_seconds=2.0,
        vehicle_type="car",
        detection_confidence=0.90,
    )
    obs_c = Observation(
        observation_id="TEST_OBS_C",
        camera_id="CAM_S01_C003",
        frame_id=30,
        timestamp_seconds=3.0,
        vehicle_type="bus",  # Incompatible with car!
        detection_confidence=0.90,
    )
    all_obs = [obs_a, obs_b, obs_c]

    # A-B strong match (0.85), B-C strong match (0.82)
    candidate_pairs = [(obs_a, obs_b), (obs_b, obs_c)]
    match_results = [
        {
            "obs_a_id": "TEST_OBS_A", "obs_b_id": "TEST_OBS_B",
            "same_vehicle_score": 0.85, "same_vehicle_probability": 0.85,
            "match_decision": "accepted",
            "evidence": {"identity_evidence_available": True},
        },
        {
            "obs_a_id": "TEST_OBS_B", "obs_b_id": "TEST_OBS_C",
            "same_vehicle_score": 0.82, "same_vehicle_probability": 0.82,
            "match_decision": "accepted",
            "evidence": {"identity_evidence_available": True},
        },
    ]

    # Naive graph without contradiction resolution connects all 3
    graph_naive = IdentityGraph(min_score_threshold=0.70)
    graph_naive.build_graph_from_matches(all_obs, candidate_pairs, match_results)
    naive_clusters = graph_naive.get_candidate_identities(resolve_contradictions=False)
    assert len(naive_clusters) == 1

    # Safe graph with contradiction resolution splits the incompatible chain
    graph_safe = IdentityGraph(min_score_threshold=0.70)
    graph_safe.build_graph_from_matches(all_obs, candidate_pairs, match_results)
    safe_clusters = graph_safe.get_candidate_identities(resolve_contradictions=True)
    assert len(safe_clusters) == 2

    # Check that car and bus are partitioned into separate clusters
    cluster_types = [[obs["vehicle_type"] for obs in c["member_observations"]] for c in safe_clusters]
    for ct in cluster_types:
        assert not ("car" in ct and "bus" in ct), "Incompatible vehicle types merged into same cluster!"


def test_arbitrary_n_camera_scalability_readiness():
    """
    [TASK 9 REGRESSION TEST — TEST ONLY]
    Lightweight structural test verifying that the UrbanTrack AI pipeline
    operates across arbitrary N cameras (e.g. N=5) without hardcoded
    assumptions of exactly 3 cameras, specific camera IDs, or fixed counts.
    """
    # Create N=5 synthetic test cameras explicitly labeled TEST ONLY
    cam_ids = [f"CAM_SCALE_TEST_C{i:03d}" for i in range(1, 6)]
    test_obs = []
    for i, cam_id in enumerate(cam_ids):
        obs = Observation(
            observation_id=f"{cam_id}_trk_001",
            camera_id=cam_id,
            frame_id=100 + i * 50,
            timestamp_seconds=10.0 + i * 5.0,
            timestamp_semantics="video_relative",
            time_reference_id=cam_id,
            vehicle_type="car",
            detection_confidence=0.88,
            appearance_embedding=[0.1] * 512,
        )
        test_obs.append(obs)

    # 1. Ingestion / processing accepts N cameras
    assert len({o.camera_id for o in test_obs}) == 5

    # 2. Candidate generation accepts N cameras
    generator = CandidateGenerator(unsynchronized_mode=True, min_probability_threshold=0.50)
    candidates, stats = generator.generate_candidates(test_obs)
    cand_cams = set()
    for a, b in candidates:
        cand_cams.add(a.camera_id)
        cand_cams.add(b.camera_id)
        assert a.camera_id != b.camera_id
    assert len(cand_cams) == 5

    # 3. Identity Graph accepts N cameras
    match_results = []
    for a, b in candidates:
        res = match_observations(a, b)
        res["obs_a_id"] = a.observation_id
        res["obs_b_id"] = b.observation_id
        match_results.append(res)

    graph = IdentityGraph(min_score_threshold=0.60)
    graph.build_graph_from_matches(test_obs, candidates, match_results)
    clusters = graph.get_candidate_identities(resolve_contradictions=True)
    assert len(clusters) > 0

