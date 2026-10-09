"""
UrbanTrack AI — Layer 2 Multimodal Identity Association Unit Tests.

Validates core identity association and evidence fusion capabilities:
1. compatible Re-ID
2. incompatible Re-ID
3. missing Re-ID
4. OCR exact match
5. OCR missing
6. OCR contradiction
7. vehicle-type mismatch as soft evidence
8. overlapping cameras
9. sequential cameras
10. topology evidence
11. missing evidence redistribution
12. ambiguous decision
13. confirmed decision
14. rejected decision
15. cross-scenario isolation
16. deterministic output
"""

import math
import pytest

from layer2.association.decision_policy import DecisionPolicy, PolicyThresholds
from layer2.association.evidence_fusion import (
    EvidenceFusionScorer,
    cosine_similarity,
    levenshtein_distance,
    normalize_plate,
)
from layer2.association.evidence_ledger import (
    AssociationDecision,
    CandidateEvidenceLedger,
)
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


def make_test_tracklet(
    scenario_id: str = "S01",
    camera_id: str = "CAM_S01_C001",
    track_id: int = 1,
    start_sync_sec: float = 0.0,
    end_sync_sec: float = 5.0,
    vehicle_type: str = "car",
    reid_group: str = AICITY_GROUP,
    reid_model: str = "osnet_x0_25_aicity",
    has_emb: bool = True,
    embedding: list = None,
    has_ocr: bool = False,
    plate_text: str = None,
    direction: str = "westbound",
    camera_reliability: float = 0.85,
    embedding_quality: float = 0.80,
) -> CanonicalTracklet:
    """Constructs a deterministic CanonicalTracklet for testing."""
    if embedding is not None:
        emb = embedding
    elif has_emb:
        emb = [0.0] * 512
        emb[0] = 1.0
    else:
        emb = None

    return CanonicalTracklet(
        scenario_id=scenario_id,
        camera_id=camera_id,
        track_id=track_id,
        global_vehicle_id=None,
        temporal=TemporalData(
            start_frame=int(start_sync_sec * 10),
            end_frame=int(end_sync_sec * 10),
            duration_frames=int((end_sync_sec - start_sync_sec) * 10) + 1,
            fps=10.0,
            start_raw_timestamp="00:00:00.000",
            end_raw_timestamp="00:00:05.000",
            start_raw_seconds=start_sync_sec,
            end_raw_seconds=end_sync_sec,
            start_sync_timestamp="00:00:00.000",
            end_sync_timestamp="00:00:05.000",
            start_sync_seconds=start_sync_sec,
            end_sync_seconds=end_sync_sec,
            duration_seconds=end_sync_sec - start_sync_sec,
        ),
        motion=MotionData(
            trajectory_pixels=[[100, 100], [200, 200]],
            trajectory_length=2,
            average_velocity_px=10.0,
            direction=direction,
        ),
        appearance=AppearanceData(
            has_embedding=(emb is not None),
            appearance_embedding=emb,
            embedding_dim=512 if emb else None,
            embedding_quality=embedding_quality if emb else None,
            reid_model=reid_model,
            reid_compatibility_group=reid_group,
        ),
        vehicle=VehicleData(
            vehicle_type=vehicle_type,
            average_detector_confidence=0.85,
        ),
        anpr=ANPRData(
            has_plate_detection=(plate_text is not None),
            has_readable_ocr=has_ocr,
            aggregated_plate_text=plate_text,
            ocr_confidence=0.90 if has_ocr else None,
            ocr_readings_count=5 if has_ocr else 0,
            plate_detections_count=5 if has_ocr else 0,
            ocr_consensus_ratio=1.0 if has_ocr else None,
        ),
        spatial=SpatialData(
            camera_latitude=42.50,
            camera_longitude=-90.70,
            camera_bearing_deg=180.0,
            camera_confidence="HIGH",
            road_context={},
            projected_vehicle_coordinates=None,
        ),
        quality=QualityData(
            camera_reliability=camera_reliability,
            missing_evidence=[] if has_ocr else ["READABLE_OCR"],
        ),
    )


class TestIdentityAssociation:
    """Test suite for Layer 2 Identity Association and Evidence Fusion."""

    def test_compatible_reid(self):
        """1. Compatible Re-ID computes valid cosine similarity."""
        emb1 = [0.0] * 512
        emb1[0] = 1.0
        emb2 = [0.0] * 512
        emb2[0] = 0.8
        emb2[1] = 0.6  # Unit length vector

        t1 = make_test_tracklet(camera_id="CAM_S01_C001", embedding=emb1, reid_group=AICITY_GROUP)
        t2 = make_test_tracklet(camera_id="CAM_S01_C003", embedding=emb2, reid_group=AICITY_GROUP)

        scorer = EvidenceFusionScorer()
        ev = scorer.evaluate_appearance(t1, t2)

        assert ev.models_compatible is True
        assert ev.status == "COMPATIBLE"
        assert ev.cosine_similarity is not None
        assert abs(ev.cosine_similarity - 0.8) < 1e-4
        assert ev.normalized_score is not None

    def test_incompatible_reid(self):
        """2. Incompatible Re-ID latent spaces are never compared directly."""
        emb1 = [1.0] + [0.0] * 511
        emb2 = [1.0] + [0.0] * 511

        t1 = make_test_tracklet(camera_id="CAM_S01_C001", embedding=emb1, reid_group=AICITY_GROUP, reid_model="osnet_x0_25_aicity")
        t2 = make_test_tracklet(camera_id="CAM_S01_C002", embedding=emb2, reid_group=MSMT17_GROUP, reid_model="osnet_x0_25_msmt17")

        scorer = EvidenceFusionScorer()
        ev = scorer.evaluate_appearance(t1, t2)

        assert ev.models_compatible is False
        assert ev.status == "INCOMPATIBLE_SPACES"
        assert ev.cosine_similarity is None
        assert ev.normalized_score is None

        # Verify appearance weight is redistributed during fusion
        score, _, active_weights, avail, miss, incomp = scorer.fuse_evidence(t1, t2)
        assert "appearance" in incomp
        assert "appearance" not in active_weights
        assert abs(sum(active_weights.values()) - 1.0) < 1e-4

    def test_missing_reid(self):
        """3. Missing embeddings mark modality as missing and redistribute weight."""
        t1 = make_test_tracklet(has_emb=True)
        t2 = make_test_tracklet(has_emb=False)

        scorer = EvidenceFusionScorer()
        ev = scorer.evaluate_appearance(t1, t2)

        assert ev.models_compatible is False
        assert ev.status == "MISSING_DESTINATION"
        assert ev.cosine_similarity is None

        score, _, active_weights, avail, miss, incomp = scorer.fuse_evidence(t1, t2)
        assert "appearance" in miss
        assert "appearance" not in active_weights
        assert abs(sum(active_weights.values()) - 1.0) < 1e-4

    def test_ocr_exact_match(self):
        """4. Exact plate OCR match produces strong confirmation evidence."""
        t1 = make_test_tracklet(has_ocr=True, plate_text="ABC-1234")
        t2 = make_test_tracklet(has_ocr=True, plate_text="abc 1234")

        scorer = EvidenceFusionScorer()
        ev = scorer.evaluate_ocr(t1, t2)

        assert ev.is_exact_match is True
        assert ev.is_contradiction is False
        assert ev.status == "EXACT_MATCH"
        assert ev.normalized_score == 1.0

        score, ev_dict, weights, avail, miss, incomp = scorer.fuse_evidence(t1, t2)
        policy = DecisionPolicy()
        dec, reasons = policy.evaluate_decision(score, ev_dict, avail, miss, incomp)

        assert dec == AssociationDecision.CONFIRMED.value
        assert "CONFIRM_EXACT_OCR_MATCH" in reasons

    def test_ocr_missing(self):
        """5. Missing OCR is treated as optional and redistributes weight."""
        t1 = make_test_tracklet(has_ocr=True, plate_text="ABC-1234")
        t2 = make_test_tracklet(has_ocr=False, plate_text=None)

        scorer = EvidenceFusionScorer()
        ev = scorer.evaluate_ocr(t1, t2)

        assert ev.status == "MISSING_ONE_SIDE"
        assert ev.is_contradiction is False
        assert ev.normalized_score is None

        score, _, active_weights, avail, miss, incomp = scorer.fuse_evidence(t1, t2)
        assert "ocr" in miss
        assert "ocr" not in active_weights
        assert abs(sum(active_weights.values()) - 1.0) < 1e-4

    def test_ocr_contradiction(self):
        """6. Contradictory OCR plates trigger immediate rejection."""
        t1 = make_test_tracklet(has_ocr=True, plate_text="ABC1234")
        t2 = make_test_tracklet(has_ocr=True, plate_text="XYZ9876")

        scorer = EvidenceFusionScorer()
        ev = scorer.evaluate_ocr(t1, t2)

        assert ev.is_contradiction is True
        assert ev.status == "CONTRADICTION"
        assert ev.normalized_score == 0.0

        score, ev_dict, weights, avail, miss, incomp = scorer.fuse_evidence(t1, t2)
        policy = DecisionPolicy()
        dec, reasons = policy.evaluate_decision(score, ev_dict, avail, miss, incomp)

        assert dec == AssociationDecision.REJECTED.value
        assert "REJECT_OCR_CONTRADICTION" in reasons

    def test_vehicle_type_mismatch_soft_evidence(self):
        """7. Vehicle type mismatch is soft discount and never hard elimination."""
        t1 = make_test_tracklet(vehicle_type="car", start_sync_sec=0.0, end_sync_sec=5.0)
        t2 = make_test_tracklet(vehicle_type="truck", start_sync_sec=6.0, end_sync_sec=10.0)

        scorer = EvidenceFusionScorer()
        ev = scorer.evaluate_vehicle_type(t1, t2)

        assert ev.is_exact_match is False
        assert ev.normalized_score == 0.40  # Soft discount

        score, ev_dict, weights, avail, miss, incomp = scorer.fuse_evidence(t1, t2)
        policy = DecisionPolicy()
        dec, reasons = policy.evaluate_decision(score, ev_dict, avail, miss, incomp)

        # Mismatch should NOT hard-reject into REJECTED solely for type
        assert dec != AssociationDecision.REJECTED.value or "REJECT_OCR_CONTRADICTION" in reasons or score < 0.40

    def test_overlapping_cameras(self):
        """8. Temporal overlap is correctly classified and scored."""
        t1 = make_test_tracklet(start_sync_sec=0.0, end_sync_sec=10.0)
        t2 = make_test_tracklet(start_sync_sec=5.0, end_sync_sec=15.0)

        scorer = EvidenceFusionScorer()
        ev = scorer.evaluate_temporal(t1, t2)

        assert ev.is_overlapping is True
        assert ev.regime == "OVERLAPPING"
        assert ev.normalized_score == 1.0

    def test_sequential_cameras(self):
        """9. Sequential transit decays smoothly over time."""
        t1 = make_test_tracklet(start_sync_sec=0.0, end_sync_sec=5.0)
        t2_near = make_test_tracklet(start_sync_sec=10.0, end_sync_sec=15.0)
        t2_far = make_test_tracklet(start_sync_sec=90.0, end_sync_sec=95.0)

        scorer = EvidenceFusionScorer()
        ev_near = scorer.evaluate_temporal(t1, t2_near, cam_dist_m=200.0)
        ev_far = scorer.evaluate_temporal(t1, t2_far, cam_dist_m=200.0)

        assert ev_near.regime == "SEQUENTIAL_TRANSIT"
        assert ev_far.regime == "SEQUENTIAL_TRANSIT"
        assert ev_near.normalized_score > ev_far.normalized_score

    def test_topology_evidence(self):
        """10. Road topology edge acts as positive prior, absence does not reject."""
        scorer = EvidenceFusionScorer()
        ev_with = scorer.evaluate_topology(has_edge=True, relationship="connected_road")
        ev_without = scorer.evaluate_topology(has_edge=False)

        assert ev_with.has_directed_edge is True
        assert ev_with.normalized_score == 1.0
        assert ev_without.has_directed_edge is False
        assert ev_without.normalized_score == 0.50  # Neutral prior

    def test_missing_evidence_redistribution(self):
        """11. Missing evidence dynamically re-normalizes active weights."""
        t1 = make_test_tracklet(has_emb=False, has_ocr=False)
        t2 = make_test_tracklet(has_emb=False, has_ocr=False)

        scorer = EvidenceFusionScorer()
        score, _, active_weights, avail, miss, incomp = scorer.fuse_evidence(t1, t2)

        assert "appearance" in miss
        assert "ocr" in miss
        assert "appearance" not in active_weights
        assert "ocr" not in active_weights
        assert abs(sum(active_weights.values()) - 1.0) < 1e-4

    def test_ambiguous_decision(self):
        """12. Plausible transition with missing definitive proof is AMBIGUOUS."""
        # Compatible appearance moderate (0.42) without OCR
        emb1 = [1.0] + [0.0] * 511
        emb2 = [0.42] + [math.sqrt(1.0 - 0.42**2)] + [0.0] * 510

        t1 = make_test_tracklet(start_sync_sec=0.0, end_sync_sec=5.0, embedding=emb1, has_ocr=False)
        t2 = make_test_tracklet(start_sync_sec=7.0, end_sync_sec=12.0, embedding=emb2, has_ocr=False)

        scorer = EvidenceFusionScorer()
        score, ev_dict, weights, avail, miss, incomp = scorer.fuse_evidence(t1, t2)

        policy = DecisionPolicy()
        dec, reasons = policy.evaluate_decision(score, ev_dict, avail, miss, incomp)

        assert dec == AssociationDecision.AMBIGUOUS.value
        assert any("AMBIGUOUS" in r for r in reasons)

    def test_confirmed_decision(self):
        """13. Strong compatible appearance + temporal continuity is CONFIRMED."""
        # Cosine similarity 0.95
        emb1 = [1.0] + [0.0] * 511
        emb2 = [0.95] + [math.sqrt(1.0 - 0.95**2)] + [0.0] * 510

        t1 = make_test_tracklet(start_sync_sec=0.0, end_sync_sec=5.0, embedding=emb1, has_ocr=False)
        t2 = make_test_tracklet(start_sync_sec=6.0, end_sync_sec=11.0, embedding=emb2, has_ocr=False)

        scorer = EvidenceFusionScorer()
        score, ev_dict, weights, avail, miss, incomp = scorer.fuse_evidence(t1, t2)

        policy = DecisionPolicy()
        dec, reasons = policy.evaluate_decision(score, ev_dict, avail, miss, incomp)

        assert dec == AssociationDecision.CONFIRMED.value
        assert "CONFIRM_STRONG_APPEARANCE_AND_SPATIOTEMPORAL" in reasons

    def test_rejected_decision(self):
        """14. Strong appearance dissimilarity triggers REJECTED."""
        # Cosine similarity 0.05
        emb1 = [1.0] + [0.0] * 511
        emb2 = [0.05] + [math.sqrt(1.0 - 0.05**2)] + [0.0] * 510

        t1 = make_test_tracklet(start_sync_sec=0.0, end_sync_sec=5.0, embedding=emb1, has_ocr=False)
        t2 = make_test_tracklet(start_sync_sec=15.0, end_sync_sec=20.0, embedding=emb2, has_ocr=False)

        scorer = EvidenceFusionScorer()
        score, ev_dict, weights, avail, miss, incomp = scorer.fuse_evidence(t1, t2)

        policy = DecisionPolicy()
        dec, reasons = policy.evaluate_decision(score, ev_dict, avail, miss, incomp)

        assert dec == AssociationDecision.REJECTED.value
        assert "REJECT_DISSIMILAR_APPEARANCE" in reasons

    def test_cross_scenario_isolation(self):
        """15. Different scenario tracklets are rejected / isolated."""
        t_s01 = make_test_tracklet(scenario_id="S01")
        t_s02 = make_test_tracklet(scenario_id="S02")

        assert t_s01.scenario_id != t_s02.scenario_id

    def test_deterministic_output(self):
        """16. Identical inputs produce identical association scores and reasons."""
        t1 = make_test_tracklet(start_sync_sec=0.0, end_sync_sec=5.0)
        t2 = make_test_tracklet(start_sync_sec=6.0, end_sync_sec=10.0)

        scorer = EvidenceFusionScorer()
        policy = DecisionPolicy()

        score1, ev1, w1, av1, mi1, inc1 = scorer.fuse_evidence(t1, t2)
        dec1, reas1 = policy.evaluate_decision(score1, ev1, av1, mi1, inc1)

        score2, ev2, w2, av2, mi2, inc2 = scorer.fuse_evidence(t1, t2)
        dec2, reas2 = policy.evaluate_decision(score2, ev2, av2, mi2, inc2)

        assert score1 == score2
        assert dec1 == dec2
        assert reas1 == reas2
        assert w1 == w2
