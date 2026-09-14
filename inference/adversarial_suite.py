"""
UrbanTrack AI — 15-Case Adversarial Evaluation Suite.
Hardens the identity fusion engine against deliberate adversarial inputs,
edge cases, and degraded perception scenarios per Phase 21.

15 Scenarios:
1. Identical-looking vehicles (same appearance, impossible simultaneous presence)
2. Visually similar vehicles (close appearance, exceeding physical speed limit)
3. OCR one-character error (e.g., NH0LBD4932 vs NH0LDD4922 - soft penalty)
4. Wrong OCR (completely divergent plate strings - negative evidence)
5. Missing OCR (one or both plates absent - neutral evidence)
6. Missing OSNet (one or both embeddings absent - neutral evidence)
7. Corrupted OSNet (NaN, Inf, empty, or mismatched dimension)
8. Simultaneous observations across distinct cameras (dt = 0, distance > 0)
9. Impossible temporal transition (backward time / negative delta)
10. Tracker fragmentation (same camera, same car split into consecutive tracks)
11. Tracker ID switch (different vehicles swap IDs under occlusion)
12. Duplicate detections (same camera, same frame, near-identical bounding box)
13. Contradictory vehicle type (e.g., car vs heavy bus)
14. Unreliable camera (camera reliability score < 0.20 attenuating evidence)
15. Conflicting modalities (high appearance match + strongly conflicting plates)
"""

from typing import Any, Dict, List, Optional
import math

from schemas.observation_schema import Observation
from .identity_fusion import match_observations
from .similarity import appearance_similarity, plate_similarity, vehicle_type_compatibility


def run_adversarial_suite(camera_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Execute all 15 adversarial scenarios and verify correct decision states.
    Target: zero unjustified CONFIRMED decisions on conflicting evidence;
    unresolved conflicts must resolve to AMBIGUOUS or REJECTED.
    """
    if camera_metadata is None:
        camera_metadata = {
            "cam_01": {"latitude": 17.3850, "longitude": 78.4867, "timestamp_semantics": "synchronized", "time_reference_id": "city_sync"},
            "cam_02": {"latitude": 17.3870, "longitude": 78.4900, "timestamp_semantics": "synchronized", "time_reference_id": "city_sync"},
            "cam_03": {"latitude": 17.4000, "longitude": 78.5100, "timestamp_semantics": "synchronized", "time_reference_id": "city_sync"},
            "cam_low_rel": {"latitude": 17.3855, "longitude": 78.4870, "reliability": 0.15, "timestamp_semantics": "synchronized", "time_reference_id": "city_sync"},
        }

    # Reference embeddings (512-D unit vectors)
    emb_white_sedan = [1.0 / math.sqrt(512)] * 512
    emb_white_sedan_clone = [1.0 / math.sqrt(512)] * 512
    emb_black_suv = [-1.0 / math.sqrt(512)] * 512
    emb_corrupted_nan = [float("nan")] * 512

    scenarios = []

    # 1. Identical-looking vehicles (same appearance, simultaneous on different cameras)
    o1_a = Observation(camera_id="cam_01", timestamp_seconds=100.0, vehicle_type="car", appearance_embedding=emb_white_sedan, latitude=17.3850, longitude=78.4867)
    o1_b = Observation(camera_id="cam_03", timestamp_seconds=100.0, vehicle_type="car", appearance_embedding=emb_white_sedan_clone, latitude=17.4000, longitude=78.5100)
    r1 = match_observations(o1_a, o1_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_01",
        "name": "Identical-looking vehicles (Simultaneous presence)",
        "expected_state": ["REJECTED", "AMBIGUOUS"],
        "actual_state": r1["decision_state"],
        "score": r1["same_vehicle_score"],
        "explanation": r1["explanation"],
        "passed": r1["decision_state"] in ("REJECTED", "AMBIGUOUS") and r1["same_vehicle_score"] < 0.40,
    })

    # 2. Visually similar vehicles (High appearance, exceeding physical speed)
    o2_a = Observation(camera_id="cam_01", timestamp_seconds=100.0, vehicle_type="car", appearance_embedding=emb_white_sedan, latitude=17.3850, longitude=78.4867)
    o2_b = Observation(camera_id="cam_03", timestamp_seconds=105.0, vehicle_type="car", appearance_embedding=emb_white_sedan, latitude=17.4000, longitude=78.5100)
    r2 = match_observations(o2_a, o2_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_02",
        "name": "Visually similar vehicles (Impossible speed)",
        "expected_state": ["REJECTED"],
        "actual_state": r2["decision_state"],
        "score": r2["same_vehicle_score"],
        "explanation": r2["explanation"],
        "passed": r2["decision_state"] == "REJECTED" and r2["same_vehicle_score"] == 0.0,
    })

    # 3. OCR one-character error (e.g. NH0LBD4932 vs NH0LDD4922)
    o3_a = Observation(camera_id="cam_01", timestamp_seconds=10.0, vehicle_type="car", plate="NH0LBD4932", appearance_embedding=emb_white_sedan, latitude=17.3850, longitude=78.4867)
    o3_b = Observation(camera_id="cam_02", timestamp_seconds=40.0, vehicle_type="car", plate="NH0LDD4932", appearance_embedding=emb_white_sedan, latitude=17.3870, longitude=78.4900)
    r3 = match_observations(o3_a, o3_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_03",
        "name": "OCR 1-character OCR noise (Soft penalty)",
        "expected_state": ["CONFIRMED", "AMBIGUOUS"],
        "actual_state": r3["decision_state"],
        "score": r3["same_vehicle_score"],
        "explanation": r3["explanation"],
        "passed": r3["same_vehicle_score"] >= 0.60,
    })

    # 4. Wrong OCR (Completely different plate strings)
    o4_a = Observation(camera_id="cam_01", timestamp_seconds=10.0, vehicle_type="car", plate="TS09EA1234", appearance_embedding=emb_white_sedan, latitude=17.3850, longitude=78.4867)
    o4_b = Observation(camera_id="cam_02", timestamp_seconds=40.0, vehicle_type="car", plate="DL01XY9999", appearance_embedding=emb_white_sedan, latitude=17.3870, longitude=78.4900)
    r4 = match_observations(o4_a, o4_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_04",
        "name": "Wrong OCR (Plate contradiction)",
        "expected_state": ["REJECTED", "AMBIGUOUS"],
        "actual_state": r4["decision_state"],
        "score": r4["same_vehicle_score"],
        "explanation": r4["explanation"],
        "passed": r4["decision_state"] != "CONFIRMED",
    })

    # 5. Missing OCR (Plates absent -> neutral evidence)
    o5_a = Observation(camera_id="cam_01", timestamp_seconds=10.0, vehicle_type="car", plate=None, appearance_embedding=emb_white_sedan, latitude=17.3850, longitude=78.4867)
    o5_b = Observation(camera_id="cam_02", timestamp_seconds=40.0, vehicle_type="car", plate=None, appearance_embedding=emb_white_sedan, latitude=17.3870, longitude=78.4900)
    r5 = match_observations(o5_a, o5_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_05",
        "name": "Missing OCR (Neutral fallback)",
        "expected_state": ["CONFIRMED", "AMBIGUOUS"],
        "actual_state": r5["decision_state"],
        "score": r5["same_vehicle_score"],
        "explanation": r5["explanation"],
        "passed": r5["same_vehicle_score"] > 0.60,
    })

    # 6. Missing OSNet (Embeddings absent -> neutral evidence)
    o6_a = Observation(camera_id="cam_01", timestamp_seconds=10.0, vehicle_type="car", plate="KA01AA1111", appearance_embedding=None, latitude=17.3850, longitude=78.4867)
    o6_b = Observation(camera_id="cam_02", timestamp_seconds=40.0, vehicle_type="car", plate="KA01AA1111", appearance_embedding=None, latitude=17.3870, longitude=78.4900)
    r6 = match_observations(o6_a, o6_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_06",
        "name": "Missing OSNet (Plate fallback)",
        "expected_state": ["CONFIRMED"],
        "actual_state": r6["decision_state"],
        "score": r6["same_vehicle_score"],
        "explanation": r6["explanation"],
        "passed": r6["decision_state"] == "CONFIRMED",
    })

    # 7. Corrupted OSNet (NaN values in embedding)
    o7_a = Observation(camera_id="cam_01", timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=emb_corrupted_nan, latitude=17.3850, longitude=78.4867)
    o7_b = Observation(camera_id="cam_02", timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=emb_white_sedan, latitude=17.3870, longitude=78.4900)
    r7 = match_observations(o7_a, o7_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_07",
        "name": "Corrupted OSNet (NaN vectors safely neutral)",
        "expected_state": ["AMBIGUOUS"],
        "actual_state": r7["decision_state"],
        "score": r7["same_vehicle_score"],
        "explanation": r7["explanation"],
        "passed": r7["evidence"]["appearance_similarity"] is None and r7["same_vehicle_score"] == 0.50,
    })

    # 8. Simultaneous observations across distinct cameras (dt=0, dx>0)
    o8_a = Observation(camera_id="cam_01", timestamp_seconds=50.0, vehicle_type="car", appearance_embedding=emb_white_sedan, latitude=17.3850, longitude=78.4867)
    o8_b = Observation(camera_id="cam_02", timestamp_seconds=50.0, vehicle_type="car", appearance_embedding=emb_white_sedan, latitude=17.3870, longitude=78.4900)
    r8 = match_observations(o8_a, o8_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_08",
        "name": "Simultaneous presence across cameras (dt=0.0s)",
        "expected_state": ["REJECTED"],
        "actual_state": r8["decision_state"],
        "score": r8["same_vehicle_score"],
        "explanation": r8["explanation"],
        "passed": r8["decision_state"] == "REJECTED" and r8["same_vehicle_score"] == 0.0,
    })

    # 9. Impossible temporal transition (Reverse time / negative delta)
    o9_a = Observation(camera_id="cam_01", timestamp_seconds=60.0, vehicle_type="car", appearance_embedding=emb_white_sedan)
    o9_b = Observation(camera_id="cam_01", timestamp_seconds=20.0, vehicle_type="car", appearance_embedding=emb_white_sedan)
    r9 = match_observations(o9_a, o9_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_09",
        "name": "Negative elapsed time on same camera",
        "expected_state": ["REJECTED"],
        "actual_state": r9["decision_state"],
        "score": r9["same_vehicle_score"],
        "explanation": r9["explanation"],
        "passed": r9["decision_state"] == "REJECTED" and r9["same_vehicle_score"] == 0.0,
    })

    # 10. Tracker fragmentation (Track 65 and Track 94 simultaneous overlap)
    o10_a = Observation(camera_id="CAM_001", track_id="65", frame_id=590, timestamp_seconds=19.667, vehicle_type="car", plate="NH0LBD4932", appearance_embedding=emb_white_sedan)
    o10_b = Observation(camera_id="CAM_001", track_id="94", frame_id=600, timestamp_seconds=20.000, vehicle_type="car", plate="NH0LBD4932", appearance_embedding=emb_white_sedan)
    r10 = match_observations(o10_a, o10_b)
    scenarios.append({
        "id": "ADV_10",
        "name": "Tracker fragmentation with temporal overlap (Tracks 65 & 94)",
        "expected_state": ["AMBIGUOUS", "CONFIRMED"],
        "actual_state": r10["decision_state"],
        "score": r10["same_vehicle_score"],
        "explanation": r10["explanation"],
        "passed": r10["decision_state"] in ("AMBIGUOUS", "CONFIRMED"),
    })

    # 11. Tracker ID switch (Different appearance, same local track label under occlusion)
    o11_a = Observation(camera_id="cam_01", track_id="switched_trk", frame_id=10, timestamp_seconds=1.0, vehicle_type="car", appearance_embedding=emb_white_sedan)
    o11_b = Observation(camera_id="cam_01", track_id="switched_trk", frame_id=80, timestamp_seconds=8.0, vehicle_type="car", appearance_embedding=emb_black_suv)
    r11 = match_observations(o11_a, o11_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_11",
        "name": "Tracker ID switch (Severe visual drift)",
        "expected_state": ["REJECTED", "AMBIGUOUS"],
        "actual_state": r11["decision_state"],
        "score": r11["same_vehicle_score"],
        "explanation": r11["explanation"],
        "passed": r11["decision_state"] in ("REJECTED", "AMBIGUOUS") and r11["same_vehicle_score"] < 0.40,
    })

    # 12. Duplicate detections (same frame, same camera)
    o12_a = Observation(camera_id="cam_01", track_id="trk_a", frame_id=100, timestamp_seconds=10.0, vehicle_type="car", bbox=[100, 100, 200, 200], appearance_embedding=emb_white_sedan)
    o12_b = Observation(camera_id="cam_01", track_id="trk_b", frame_id=100, timestamp_seconds=10.0, vehicle_type="car", bbox=[105, 102, 202, 201], appearance_embedding=emb_white_sedan)
    r12 = match_observations(o12_a, o12_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_12",
        "name": "Duplicate detections in same frame",
        "expected_state": ["REJECTED", "AMBIGUOUS"],
        "actual_state": r12["decision_state"],
        "score": r12["same_vehicle_score"],
        "explanation": r12["explanation"],
        "passed": r12["decision_state"] in ("REJECTED", "AMBIGUOUS") and r12["same_vehicle_score"] <= 0.50,
    })

    # 13. Contradictory vehicle type (car vs bus)
    o13_a = Observation(camera_id="cam_01", timestamp_seconds=10.0, vehicle_type="car", plate="KA01AA1111", appearance_embedding=emb_white_sedan, latitude=17.3850, longitude=78.4867)
    o13_b = Observation(camera_id="cam_02", timestamp_seconds=40.0, vehicle_type="bus", plate="KA01AA1111", appearance_embedding=emb_white_sedan, latitude=17.3870, longitude=78.4900)
    r13 = match_observations(o13_a, o13_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_13",
        "name": "Contradictory vehicle type (Car vs Bus)",
        "expected_state": ["REJECTED"],
        "actual_state": r13["decision_state"],
        "score": r13["same_vehicle_score"],
        "explanation": r13["explanation"],
        "passed": r13["decision_state"] == "REJECTED" and r13["same_vehicle_score"] == 0.0,
    })

    # 14. Unreliable camera (Reliability 0.15 attenuating score)
    o14_a = Observation(camera_id="cam_low_rel", timestamp_seconds=10.0, vehicle_type="car", camera_reliability=0.15, appearance_embedding=emb_white_sedan, latitude=17.3855, longitude=78.4870)
    o14_b = Observation(camera_id="cam_02", timestamp_seconds=40.0, vehicle_type="car", camera_reliability=0.90, appearance_embedding=emb_white_sedan, latitude=17.3870, longitude=78.4900)
    r14 = match_observations(o14_a, o14_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_14",
        "name": "Low camera reliability (Attenuated weight)",
        "expected_state": ["CONFIRMED", "AMBIGUOUS"],
        "actual_state": r14["decision_state"],
        "score": r14["same_vehicle_score"],
        "explanation": r14["explanation"],
        "passed": r14["same_vehicle_score"] <= 0.85,
    })

    # 15. Conflicting modalities (High appearance but completely conflicting plates)
    o15_a = Observation(camera_id="cam_01", timestamp_seconds=10.0, vehicle_type="car", plate="AP09AA1111", appearance_embedding=emb_white_sedan, latitude=17.3850, longitude=78.4867)
    o15_b = Observation(camera_id="cam_02", timestamp_seconds=40.0, vehicle_type="car", plate="MH12ZZ9999", appearance_embedding=emb_white_sedan, latitude=17.3870, longitude=78.4900)
    r15 = match_observations(o15_a, o15_b, camera_metadata=camera_metadata)
    scenarios.append({
        "id": "ADV_15",
        "name": "Conflicting modalities (High appearance vs Conflicting plate)",
        "expected_state": ["REJECTED", "AMBIGUOUS"],
        "actual_state": r15["decision_state"],
        "score": r15["same_vehicle_score"],
        "explanation": r15["explanation"],
        "passed": r15["decision_state"] != "CONFIRMED",
    })

    passed_count = sum(1 for s in scenarios if s["passed"])
    return {
        "suite": "ADVERSARIAL_15_SCENARIOS",
        "total_scenarios": len(scenarios),
        "passed_count": passed_count,
        "pass_rate": round(passed_count / len(scenarios), 4),
        "all_passed": passed_count == len(scenarios),
        "scenarios": scenarios,
    }
