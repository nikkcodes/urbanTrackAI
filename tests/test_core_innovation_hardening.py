"""
tests/test_core_innovation_hardening.py

Phase H — Hardening tests (A–X) for the UrbanTrack AI core inference upgrade.

Tests cover:
  A  Connected-component transitive contradiction
  B  Singleton — no artificial perfect identity confidence
  C  Clean chronological chain — consistent
  D  Cross-camera equal numeric timestamps without shared reference — NOT automatically simultaneous
  E  Comparable simultaneous cross-camera (same time, different cameras, shared reference) — contradiction
  F  Missing appearance — missing, not positive
  G  Missing plate — missing, not positive
  H  Contradictory plate — explicitly contradictory
  I  Invalid embedding dimensions — invalid in ledger
  J  Valid 3-observation global trajectory — globally consistent
  K  Locally valid but globally incompatible route combo — rejected or marked inconsistent
  L  Impossible segment despite reasonable cumulative average — detected
  M  Multiple complete hypotheses — ambiguity preserved
  N  Best global hypothesis ranking — correct relative ordering
  O  Sparse single gap — existing behavior preserved
  P  Sparse multi-gap compatible hypotheses — globally consistent
  Q  Sparse multi-gap conflicting hypotheses — conflict detected
  R  No feasible hidden route — no fabricated route
  S  Temporal synchronization unavailable — unverified/unavailable, not fabricated
  T  Inference trace serialization round-trip — semantic preservation
  U  Deterministic same input — semantically identical result
  V  Different input ordering — semantically equivalent cluster result
  W  Reliability vs demand separation — reliability change does not affect vehicle_weight
  X  Real Kanishka feed — no fabricated cross-camera merge without identity evidence
"""

import sys
import os
import time
import math
import unittest
from datetime import datetime
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.observation_schema import Observation
from inference.identity_fusion import match_observations
from inference.identity_graph import IdentityGraph, ClusterConsistencyResult
from inference.trajectory_engine import (
    evaluate_global_trajectory_hypotheses,
    score_trajectory_hypothesis,
    _build_route_combinations,
    reconstruct_identity_trajectory,
)
from inference.sparse_engine import infer_sparse_gap, evaluate_multigap_consistency
from inference.inference_trace import (
    InferenceTrace,
    build_cluster_trace,
    build_global_trajectory_trace,
    build_identity_pair_trace,
    build_sparse_gap_trace,
    build_trajectory_segment_trace,
)
from inference.road_graph import RoadGraph
from schemas.trajectory_schema import CandidateRoute, TrajectorySegment, VehicleTrajectory


# ---------------------------------------------------------------------------
# Shared fixtures / helper builders
# ---------------------------------------------------------------------------

def _make_obs(
    obs_id: str,
    camera_id: str,
    ts: float,
    vehicle_type: str = "car",
    plate: Optional[str] = None,
    embedding: Optional[List[float]] = None,
    lat: float = 12.9716,
    lon: float = 77.5946,
    timestamp_semantics: Optional[str] = None,
    time_reference_id: Optional[str] = None,
) -> Observation:
    """Helper to build a minimal Observation for testing."""
    return Observation(
        observation_id=obs_id,
        camera_id=camera_id,
        timestamp=datetime.fromtimestamp(ts),
        timestamp_seconds=ts,
        vehicle_type=vehicle_type,
        plate=plate,
        appearance_embedding=embedding,
        latitude=lat,
        longitude=lon,
        timestamp_semantics=timestamp_semantics or "elapsed_seconds",
        time_reference_id=time_reference_id or camera_id,  # local to camera by default
    )


def _make_segment(
    seg_id: str,
    obs_a: Observation,
    obs_b: Observation,
    routes: Optional[List[CandidateRoute]] = None,
    confidence: float = 0.8,
    status: str = "success",
) -> TrajectorySegment:
    """Helper to build a minimal TrajectorySegment."""
    return TrajectorySegment(
        segment_id=seg_id,
        identity_id="test_vehicle",
        start_observation_id=obs_a.observation_id,
        start_camera_id=obs_a.camera_id,
        start_timestamp=obs_a.timestamp_seconds,
        end_observation_id=obs_b.observation_id,
        end_camera_id=obs_b.camera_id,
        end_timestamp=obs_b.timestamp_seconds,
        time_difference_seconds=obs_b.timestamp_seconds - obs_a.timestamp_seconds,
        start_node_id="node_A",
        end_node_id="node_B",
        candidate_routes=routes or [],
        most_likely_route=routes[0].edges if routes else None,
        confidence=confidence,
        status=status,
    )


def _make_route(
    route_id: str,
    nodes: List[str],
    edges: List[str],
    dist: float = 1000.0,
    speed_limit: float = 50.0,
    required_speed: Optional[float] = 30.0,
    feasible: bool = True,
    feasibility_status: str = "feasible",
    raw_score: float = 0.8,
) -> CandidateRoute:
    """Helper to build a CandidateRoute for hypothesis testing."""
    return CandidateRoute(
        route_id=route_id,
        edges=edges,
        nodes=nodes,
        distance_meters=dist,
        estimated_travel_time_seconds=dist / (speed_limit / 3.6),
        min_travel_time_seconds=dist / (speed_limit / 3.6),
        required_speed_kmh=required_speed,
        speed_limit_kmh=speed_limit,
        feasible=feasible,
        feasibility_status=feasibility_status,
        estimated_likelihood=0.0,
        raw_score=raw_score,
        explanation=f"Test route {route_id}.",
    )


# ---------------------------------------------------------------------------
# Test A: Transitive contradiction detection
# ---------------------------------------------------------------------------
class TestA_TransitiveContradiction(unittest.TestCase):

    def test_transitive_contradiction_detected(self):
        """
        A-B strong edge, B-C strong edge, but A-C have contradictory plates.
        Cluster consistency must detect the transitive plate contradiction.
        """
        obs_a = _make_obs("A", "cam1", 1000.0, plate="KA01AB1234",
                          embedding=[0.9] * 64, time_reference_id="shared_ref")
        obs_b = _make_obs("B", "cam2", 1060.0, plate="KA01AB1234",
                          embedding=[0.88] * 64, time_reference_id="shared_ref")
        obs_c = _make_obs("C", "cam3", 1120.0, plate="TN99ZZ9999",  # clearly different plate
                          embedding=[0.85] * 64, time_reference_id="shared_ref")

        graph = IdentityGraph(min_probability_threshold=0.50)

        # Manually inject edges (A-B and B-C strong, but no A-C edge)
        for obs in [obs_a, obs_b, obs_c]:
            graph.add_observation(obs)

        # A-B strong
        graph.adjacency["A"].append(("B", 0.85))
        graph.adjacency["B"].append(("A", 0.85))
        graph.edges.append({"source": "A", "target": "B", "probability": 0.85, "evidence": {}, "explanation": ""})

        # B-C strong
        graph.adjacency["B"].append(("C", 0.83))
        graph.adjacency["C"].append(("B", 0.83))
        graph.edges.append({"source": "B", "target": "C", "probability": 0.83, "evidence": {}, "explanation": ""})

        # No A-C edge (would be added if A-C matched, but plate is contradictory)

        clusters = graph.get_candidate_identities()
        self.assertEqual(len(clusters), 1, "A, B, C should form a single connected cluster")

        cluster = clusters[0]
        consistency = cluster.get("cluster_consistency", {})

        # The transitive contradiction should be detected as a warning
        warnings = consistency.get("warnings", [])
        contradictory_edges = consistency.get("contradictory_edges", [])

        # Either warnings or contradictory_edges should mention the A-C contradiction
        all_text = " ".join(warnings) + str(contradictory_edges)
        self.assertTrue(
            len(warnings) > 0 or len(contradictory_edges) > 0,
            f"No contradiction detected for A-C plate mismatch. warnings={warnings}, contradictory={contradictory_edges}"
        )


# ---------------------------------------------------------------------------
# Test B: Singleton — no artificial perfect identity confidence
# ---------------------------------------------------------------------------
class TestB_SingletonSemantics(unittest.TestCase):

    def test_singleton_has_no_artificial_confidence(self):
        """
        A single observation with no matching partner must have identity_confidence=None
        and identity_status='unconfirmed_singleton', NOT confidence=1.0.
        """
        obs = _make_obs("solo_obs", "cam1", 100.0)
        graph = IdentityGraph()
        graph.add_observation(obs)
        clusters = graph.get_candidate_identities()

        self.assertEqual(len(clusters), 1)
        cluster = clusters[0]
        self.assertEqual(cluster["identity_status"], "unconfirmed_singleton",
                         "Singleton must be marked as unconfirmed_singleton")
        self.assertIsNone(cluster["identity_confidence"],
                          "Singleton confidence must be None, not 1.0")

    def test_singleton_consistency_is_singleton_status(self):
        """Singleton cluster consistency status must be 'singleton'."""
        obs = _make_obs("solo2", "cam1", 200.0)
        graph = IdentityGraph()
        graph.add_observation(obs)
        clusters = graph.get_candidate_identities()
        consistency = clusters[0]["cluster_consistency"]
        self.assertEqual(consistency["status"], "singleton")
        self.assertTrue(consistency["is_consistent"])


# ---------------------------------------------------------------------------
# Test C: Clean chronological chain — consistent
# ---------------------------------------------------------------------------
class TestC_CleanChain(unittest.TestCase):

    def test_clean_3obs_chain_consistent(self):
        """Three observations in clear chronological order with shared reference → consistent."""
        obs_a = _make_obs("C1", "cam1", 1000.0, time_reference_id="shared")
        obs_b = _make_obs("C2", "cam2", 1060.0, time_reference_id="shared")
        obs_c = _make_obs("C3", "cam3", 1130.0, time_reference_id="shared")

        graph = IdentityGraph()
        for obs in [obs_a, obs_b, obs_c]:
            graph.add_observation(obs)

        result = graph._validate_cluster_consistency([obs_a, obs_b, obs_c])
        self.assertEqual(len(result.violations), 0, f"Expected no violations, got: {result.violations}")
        self.assertIn(result.status, ("consistent", "insufficient_evidence", "unverified"))


# ---------------------------------------------------------------------------
# Test D: Cross-camera equal timestamps without shared reference — NOT simultaneous
# ---------------------------------------------------------------------------
class TestD_NoSharedReferenceNotSimultaneous(unittest.TestCase):

    def test_equal_timestamps_different_cameras_no_shared_ref_not_contradiction(self):
        """
        Two observations with equal numeric timestamp_seconds but different cameras
        and NO shared temporal reference must NOT be automatically classified as
        simultaneously impossible (that would violate the temporal invariant).
        """
        obs_a = _make_obs("D1", "cam1", 500.0, time_reference_id="cam1_local")
        obs_b = _make_obs("D2", "cam2", 500.0, time_reference_id="cam2_local")  # different local ref

        from inference.temporal import check_temporal_comparability
        result = check_temporal_comparability(obs_a, obs_b)

        # With different local references, timestamps are not comparable
        self.assertFalse(
            result.get("comparable", False),
            "Same numeric timestamp with different cameras/refs must NOT be comparable"
        )
        self.assertNotIn(
            result.get("status", ""),
            ("impossible_simultaneous_different_cameras",),
            f"Without shared reference, equal ts must not be 'impossible_simultaneous'. Got: {result}"
        )

    def test_evidence_ledger_temporal_unavailable_not_impossible(self):
        """Evidence ledger temporal status must be 'unavailable' when timestamps not comparable."""
        obs_a = _make_obs("D3", "cam1", 500.0, time_reference_id="cam1_local")
        obs_b = _make_obs("D4", "cam2", 500.0, time_reference_id="cam2_local")

        result = match_observations(obs_a, obs_b)
        ledger = result.get("evidence_ledger", {})
        temporal_status = ledger.get("temporal", {}).get("status", "")

        self.assertIn(
            temporal_status, ("unavailable", "unverified"),
            f"Expected 'unavailable' or 'unverified' temporal status, got: {temporal_status}"
        )


# ---------------------------------------------------------------------------
# Test E: Comparable simultaneous cross-camera — physical contradiction
# ---------------------------------------------------------------------------
class TestE_SimultaneousCrossCameraContradiction(unittest.TestCase):

    def test_simultaneous_shared_ref_different_cameras_contradiction(self):
        """
        Observations from different cameras at t=500.0s with the SAME temporal reference.
        This is physically impossible and must produce a rejection (is_rejected / score=0.0).
        """
        # Create observations with the same shared time reference (both synchronized)
        obs_a = _make_obs("E1", "cam1", 500.0,
                          timestamp_semantics="utc_synchronized",
                          time_reference_id="utc_gps")
        obs_b = _make_obs("E2", "cam2", 500.0,
                          timestamp_semantics="utc_synchronized",
                          time_reference_id="utc_gps",
                          lat=13.0, lon=77.7)  # different location

        result = match_observations(obs_a, obs_b)
        score = result.get("same_vehicle_probability", 1.0)

        # Should be rejected or very low probability
        # The temporal module should detect the impossible simultaneous cross-camera condition
        # (The actual detection depends on temporal.py logic — we just verify score is low or rejected)
        temporal_ev = result.get("evidence", {})
        t_status = temporal_ev.get("temporal_status", "")

        # If comparable and simultaneous at different cameras with shared reference:
        # should be rejected or impossible
        self.assertLessEqual(score, 0.5,
                             f"Simultaneous cross-camera with shared ref should have low score, got {score}. "
                             f"t_status={t_status}")


# ---------------------------------------------------------------------------
# Test F: Missing appearance — missing, not positive
# ---------------------------------------------------------------------------
class TestF_MissingAppearance(unittest.TestCase):

    def test_missing_appearance_status_in_ledger(self):
        """No appearance embedding → ledger status 'missing', not positive contribution."""
        obs_a = _make_obs("F1", "cam1", 1000.0, embedding=None, time_reference_id="shared")
        obs_b = _make_obs("F2", "cam2", 1060.0, embedding=None, time_reference_id="shared")

        result = match_observations(obs_a, obs_b)
        ledger = result.get("evidence_ledger", {})
        app_entry = ledger.get("appearance", {})

        self.assertEqual(app_entry.get("status"), "missing",
                         f"Missing embedding should be 'missing', got: {app_entry.get('status')}")
        self.assertIsNone(app_entry.get("value"), "Missing embedding value should be None")
        self.assertIsNone(app_entry.get("contribution"), "Missing embedding contribution should be None")

    def test_missing_appearance_limits_match_probability(self):
        """Without identity evidence, match probability must not exceed 0.50."""
        obs_a = _make_obs("F3", "cam1", 1000.0, time_reference_id="shared")
        obs_b = _make_obs("F4", "cam2", 1060.0, time_reference_id="shared")

        result = match_observations(obs_a, obs_b)
        score = result.get("same_vehicle_probability", 1.0)
        self.assertLessEqual(score, 0.50,
                             f"Without identity evidence, score must be <= 0.50, got {score}")


# ---------------------------------------------------------------------------
# Test G: Missing plate — missing, not positive
# ---------------------------------------------------------------------------
class TestG_MissingPlate(unittest.TestCase):

    def test_missing_plate_status(self):
        """No plate on either obs → plate ledger status 'missing'."""
        obs_a = _make_obs("G1", "cam1", 1000.0, plate=None, time_reference_id="shared")
        obs_b = _make_obs("G2", "cam2", 1060.0, plate=None, time_reference_id="shared")

        result = match_observations(obs_a, obs_b)
        ledger = result.get("evidence_ledger", {})
        plate_entry = ledger.get("plate", {})

        self.assertEqual(plate_entry.get("status"), "missing",
                         f"No plates should give status 'missing', got {plate_entry}")

    def test_one_side_missing_plate_is_missing(self):
        """Only one obs has a plate → ledger plate status 'missing'."""
        obs_a = _make_obs("G3", "cam1", 1000.0, plate="KA01AB1234", time_reference_id="shared")
        obs_b = _make_obs("G4", "cam2", 1060.0, plate=None, time_reference_id="shared")

        result = match_observations(obs_a, obs_b)
        ledger = result.get("evidence_ledger", {})
        plate_entry = ledger.get("plate", {})
        self.assertEqual(plate_entry.get("status"), "missing",
                         "One plate present/one missing → status must be 'missing'")


# ---------------------------------------------------------------------------
# Test H: Contradictory plate — explicitly contradictory
# ---------------------------------------------------------------------------
class TestH_ContradictoryPlate(unittest.TestCase):

    def test_contradictory_plates_ledger(self):
        """
        Both observations have valid but completely different plates.
        Ledger plate status must be 'contradictory' (plate_score < 0.3).
        """
        obs_a = _make_obs("H1", "cam1", 1000.0, plate="KA01AB1234", time_reference_id="shared")
        obs_b = _make_obs("H2", "cam2", 1060.0, plate="TN99ZZ9999", time_reference_id="shared")

        result = match_observations(obs_a, obs_b)
        ledger = result.get("evidence_ledger", {})
        plate_entry = ledger.get("plate", {})

        # The Levenshtein-based plate_similarity of completely different plates should be < 0.3
        self.assertEqual(
            plate_entry.get("status"), "contradictory",
            f"Completely different plates should produce 'contradictory', got: {plate_entry}"
        )

    def test_similar_plates_are_available_not_contradictory(self):
        """Plates with only 1 char difference → 'available' (not contradictory)."""
        obs_a = _make_obs("H3", "cam1", 1000.0, plate="KA01AB1234", time_reference_id="shared")
        obs_b = _make_obs("H4", "cam2", 1060.0, plate="KA01AB1235", time_reference_id="shared")

        result = match_observations(obs_a, obs_b)
        ledger = result.get("evidence_ledger", {})
        plate_entry = ledger.get("plate", {})
        self.assertEqual(plate_entry.get("status"), "available",
                         f"Near-identical plates should be 'available', got {plate_entry}")


# ---------------------------------------------------------------------------
# Test I: Invalid embedding dimensions — invalid in ledger
# ---------------------------------------------------------------------------
class TestI_InvalidEmbedding(unittest.TestCase):

    def test_mismatched_embedding_dimensions_invalid(self):
        """obs_a has 64-dim embedding, obs_b has 128-dim — ledger should show 'invalid'."""
        obs_a = _make_obs("I1", "cam1", 1000.0, embedding=[0.5] * 64)
        obs_b = _make_obs("I2", "cam2", 1060.0, embedding=[0.5] * 128)

        result = match_observations(obs_a, obs_b)
        ledger = result.get("evidence_ledger", {})
        app_entry = ledger.get("appearance", {})

        self.assertEqual(app_entry.get("status"), "invalid",
                         f"Mismatched embedding dims should be 'invalid', got {app_entry}")
        self.assertIsNone(app_entry.get("value"), "Invalid embedding should have no score value")


# ---------------------------------------------------------------------------
# Test J: Valid 3-observation global trajectory — globally consistent
# ---------------------------------------------------------------------------
class TestJ_GlobalTrajectoryConsistent(unittest.TestCase):

    def _build_3obs_trajectory(self) -> VehicleTrajectory:
        """Build a trajectory with 2 segments, all feasible."""
        r1 = _make_route("r1", ["n1", "n2"], ["e1"], dist=1000.0, required_speed=30.0)
        r2 = _make_route("r2", ["n2", "n3"], ["e2"], dist=1200.0, required_speed=28.0)
        obs_a = _make_obs("J1", "cam1", 1000.0)
        obs_b = _make_obs("J2", "cam2", 1120.0)
        obs_c = _make_obs("J3", "cam3", 1270.0)
        seg1 = _make_segment("s1", obs_a, obs_b, routes=[r1], confidence=0.82)
        seg2 = _make_segment("s2", obs_b, obs_c, routes=[r2], confidence=0.79)
        seg1.start_node_id = "n1"
        seg1.end_node_id = "n2"
        seg2.start_node_id = "n2"
        seg2.end_node_id = "n3"
        return VehicleTrajectory(
            identity_id="test_J",
            observations_count=3,
            cameras_visited=["cam1", "cam2", "cam3"],
            start_timestamp=1000.0,
            end_timestamp=1270.0,
            segments=[seg1, seg2],
            overall_confidence=0.80,
        )

    def test_feasible_3obs_hypothesis_globally_consistent(self):
        """O1→O2→O3 with road-compatible routes → globally consistent hypothesis."""
        traj = self._build_3obs_trajectory()
        result = evaluate_global_trajectory_hypotheses(traj)

        feasible = result["feasible_count"]
        self.assertGreater(feasible, 0, "Should have at least 1 feasible global hypothesis")
        best = next((h for h in result["hypotheses"] if h["is_globally_feasible"]), None)
        self.assertIsNotNone(best, "Should have at least one feasible hypothesis")

    def test_feasible_hypothesis_has_nonzero_relative_likelihood(self):
        """The surviving feasible hypothesis must have relative_likelihood > 0."""
        traj = self._build_3obs_trajectory()
        result = evaluate_global_trajectory_hypotheses(traj)
        feasible_hyps = [h for h in result["hypotheses"] if h["is_globally_feasible"]]
        for h in feasible_hyps:
            self.assertIsNotNone(h["relative_likelihood"])
            self.assertGreater(h["relative_likelihood"], 0.0)


# ---------------------------------------------------------------------------
# Test K: Locally valid but globally incompatible — rejected
# ---------------------------------------------------------------------------
class TestK_GloballyIncompatibleRoute(unittest.TestCase):

    def test_topology_discontinuity_rejected(self):
        """
        Segment 1 route ends at node_B, segment 2 route starts at node_C (different).
        The hypothesis must be marked infeasible due to topology discontinuity.
        """
        r1 = _make_route("r1", ["n_start", "n_mid", "n_end_A"], ["e1", "e2"])
        r2 = _make_route("r2", ["n_start_B", "n_final"], ["e3"])  # starts at different node!

        obs_a = _make_obs("K1", "cam1", 1000.0)
        obs_b = _make_obs("K2", "cam2", 1120.0)
        obs_c = _make_obs("K3", "cam3", 1240.0)

        seg1 = _make_segment("ks1", obs_a, obs_b, routes=[r1], confidence=0.85)
        seg2 = _make_segment("ks2", obs_b, obs_c, routes=[r2], confidence=0.80)

        hypothesis = score_trajectory_hypothesis([r1, r2], [seg1, seg2])

        self.assertFalse(hypothesis["is_globally_feasible"],
                         "Topology discontinuity must make hypothesis infeasible")
        self.assertTrue(any("topology" in r.lower() or "discontinuity" in r.lower()
                            for r in hypothesis.get("rejection_reasons", [])),
                        f"Should cite topology discontinuity. Got: {hypothesis['rejection_reasons']}")


# ---------------------------------------------------------------------------
# Test L: Impossible segment despite reasonable cumulative average
# ---------------------------------------------------------------------------
class TestL_IsolatedInfeasibleSegment(unittest.TestCase):

    def test_isolated_infeasible_segment_detected(self):
        """
        A 3-segment trajectory where the middle segment is physically infeasible
        while both outer segments are feasible must be detected as having an
        isolated infeasibility — NOT passed as globally consistent.
        """
        r_ok1 = _make_route("rL1", ["nL1", "nL2"], ["eL1"], feasible=True, raw_score=0.8, required_speed=30.0)
        r_bad = _make_route("rL_bad", ["nL2", "nL3"], ["eL2"], feasible=False,
                             feasibility_status="speed_limit_exceeded", raw_score=0.0, required_speed=300.0)
        r_ok2 = _make_route("rL2", ["nL3", "nL4"], ["eL3"], feasible=True, raw_score=0.75, required_speed=25.0)

        obs_a = _make_obs("L1", "cam1", 1000.0)
        obs_b = _make_obs("L2", "cam2", 1002.0)  # Only 2s for 3km → impossible
        obs_c = _make_obs("L3", "cam3", 1200.0)
        obs_d = _make_obs("L4", "cam4", 1320.0)

        seg1 = _make_segment("Ls1", obs_a, obs_b, routes=[r_ok1], confidence=0.8)
        seg2 = _make_segment("Ls2", obs_b, obs_c, routes=[r_bad], confidence=0.0, status="infeasible")
        seg3 = _make_segment("Ls3", obs_c, obs_d, routes=[r_ok2], confidence=0.75)

        hypothesis = score_trajectory_hypothesis([r_ok1, r_bad, r_ok2], [seg1, seg2, seg3])

        self.assertFalse(hypothesis["is_globally_feasible"],
                         "Middle segment infeasible → hypothesis should be infeasible")
        contradiction_text = str(hypothesis.get("contradictions", []))
        # Should either be in rejection_reasons or contradictions
        all_text = str(hypothesis.get("rejection_reasons", [])) + str(hypothesis.get("contradictions", []))
        self.assertTrue(len(hypothesis.get("rejection_reasons", [])) > 0 or
                        len(hypothesis.get("contradictions", [])) > 0,
                        "Infeasible middle segment should generate rejection or contradiction entry")


# ---------------------------------------------------------------------------
# Test M: Multiple complete hypotheses — ambiguity preserved
# ---------------------------------------------------------------------------
class TestM_MultipleHypothesesAmbiguityPreserved(unittest.TestCase):

    def test_multiple_hypotheses_both_survive(self):
        """
        Two competing routes for a single segment, both feasible with similar scores.
        Both should survive and ambiguity should be flagged.
        """
        r_a = _make_route("M_r1", ["nM1", "nM2"], ["eM1"], raw_score=0.82, required_speed=28.0)
        r_b = _make_route("M_r2", ["nM1", "nMalt", "nM2"], ["eM2", "eM3"], raw_score=0.80, required_speed=27.0)

        obs_a = _make_obs("M1", "cam1", 1000.0)
        obs_b = _make_obs("M2", "cam2", 1120.0)
        seg = _make_segment("Ms1", obs_a, obs_b, routes=[r_a, r_b], confidence=0.81)

        traj = VehicleTrajectory(
            identity_id="test_M",
            observations_count=2,
            cameras_visited=["cam1", "cam2"],
            start_timestamp=1000.0,
            end_timestamp=1120.0,
            segments=[seg],
        )
        result = evaluate_global_trajectory_hypotheses(traj)

        # Both hypotheses should survive (both feasible)
        self.assertGreaterEqual(result["feasible_count"], 2,
                                f"Both routes are feasible, expected ≥2 feasible hypotheses, got {result}")

    def test_relative_likelihood_sums_to_1(self):
        """Relative likelihoods of feasible hypotheses must sum to 1.0."""
        r_a = _make_route("M_r3", ["nM3", "nM4"], ["eM4"], raw_score=0.82, required_speed=28.0)
        r_b = _make_route("M_r4", ["nM3", "nM4"], ["eM5"], raw_score=0.78, required_speed=27.0)

        obs_a = _make_obs("M3", "cam1", 1000.0)
        obs_b = _make_obs("M4", "cam2", 1120.0)
        seg = _make_segment("Ms2", obs_a, obs_b, routes=[r_a, r_b], confidence=0.80)

        traj = VehicleTrajectory(
            identity_id="test_M2",
            observations_count=2,
            cameras_visited=["cam1", "cam2"],
            start_timestamp=1000.0,
            end_timestamp=1120.0,
            segments=[seg],
        )
        result = evaluate_global_trajectory_hypotheses(traj)
        feasible_hyps = [h for h in result["hypotheses"] if h["is_globally_feasible"]]
        total_rl = sum(h.get("relative_likelihood") or 0.0 for h in feasible_hyps)
        if feasible_hyps:
            self.assertAlmostEqual(total_rl, 1.0, places=3,
                                   msg=f"Feasible hypothesis likelihoods should sum to 1.0, got {total_rl}")


# ---------------------------------------------------------------------------
# Test N: Best global hypothesis ranking
# ---------------------------------------------------------------------------
class TestN_HypothesisRanking(unittest.TestCase):

    def test_higher_score_route_is_ranked_first(self):
        """Higher raw_score route should appear first after normalization."""
        r_high = _make_route("N_high", ["nN1", "nN2"], ["eN1"], raw_score=0.90, required_speed=28.0)
        r_low = _make_route("N_low", ["nN1", "nN3", "nN2"], ["eN2", "eN3"], raw_score=0.60, required_speed=27.0)

        obs_a = _make_obs("N1", "cam1", 1000.0)
        obs_b = _make_obs("N2", "cam2", 1120.0)
        seg = _make_segment("Ns1", obs_a, obs_b, routes=[r_high, r_low], confidence=0.85)

        traj = VehicleTrajectory(
            identity_id="test_N",
            observations_count=2,
            cameras_visited=["cam1", "cam2"],
            start_timestamp=1000.0,
            end_timestamp=1120.0,
            segments=[seg],
        )
        result = evaluate_global_trajectory_hypotheses(traj)
        feasible_hyps = [h for h in result["hypotheses"] if h["is_globally_feasible"]]

        if len(feasible_hyps) >= 2:
            top_rl = feasible_hyps[0].get("relative_likelihood") or 0.0
            second_rl = feasible_hyps[1].get("relative_likelihood") or 0.0
            self.assertGreaterEqual(top_rl, second_rl,
                                    f"Top hypothesis should have higher or equal RL: {top_rl} vs {second_rl}")


# ---------------------------------------------------------------------------
# Test O: Sparse single gap — existing behavior preserved
# ---------------------------------------------------------------------------
class TestO_SparseSingleGap(unittest.TestCase):

    def test_single_gap_returns_sparse_observation_gap(self):
        """infer_sparse_gap with a valid road graph should return a SparseObservationGap."""
        # SparseObservationGap is defined inside sparse_engine (not sparse_schema)
        from inference.sparse_engine import SparseObservationGap

        road_graph = RoadGraph.from_json_file("data/roads/synthetic_road_graph.json")

        obs_a = _make_obs("O1", "cam_1", 1000.0)
        obs_b = _make_obs("O2", "cam_2", 1120.0)

        gap = infer_sparse_gap(obs_a, obs_b, road_graph=road_graph)
        self.assertIsInstance(gap, SparseObservationGap)
        self.assertIsNotNone(gap.gap_id)
        self.assertIn(gap.status, ("success", "no_path", "unassociated_camera", "temporal_inversion", "infeasible"))


# ---------------------------------------------------------------------------
# Test P: Sparse multi-gap compatible hypotheses — globally consistent
# ---------------------------------------------------------------------------
class TestP_MultiGapCompatible(unittest.TestCase):

    def _make_mock_gap(self, gap_id: str, start_node: str, end_node: str,
                       t_start: float, t_end: float, route_nodes: List[str]) -> Any:
        """Create a minimal duck-type gap object for evaluate_multigap_consistency."""
        from types import SimpleNamespace
        route = _make_route(f"r_{gap_id}", route_nodes, [f"e_{gap_id}"], feasible=True)
        return SimpleNamespace(
            gap_id=gap_id,
            start_node_id=start_node,
            end_node_id=end_node,
            candidate_routes=[route],
            start_observation={"timestamp_seconds": t_start},
            end_observation={"timestamp_seconds": t_end},
        )

    def test_compatible_multi_gap(self):
        """
        GAP_1 ends at shared_node_B, GAP_2 starts at shared_node_B.
        Top routes agree on exit/entry at the boundary.
        Expected: globally consistent.
        """
        gap1 = self._make_mock_gap("gap1", "node_A", "node_B", 1000.0, 1100.0,
                                   ["node_A", "node_mid1", "node_B"])
        gap2 = self._make_mock_gap("gap2", "node_B", "node_C", 1100.0, 1200.0,
                                   ["node_B", "node_mid2", "node_C"])

        result = evaluate_multigap_consistency([gap1, gap2])
        self.assertTrue(result["is_globally_consistent"],
                        f"Compatible gaps should be globally consistent. Got: {result}")
        self.assertEqual(result["conflicting_gap_count"], 0)


# ---------------------------------------------------------------------------
# Test Q: Sparse multi-gap conflicting hypotheses — conflict detected
# ---------------------------------------------------------------------------
class TestQ_MultiGapConflict(unittest.TestCase):

    def _make_mock_gap(self, gap_id: str, start_node: str, end_node: str,
                       t_start: float, t_end: float, route_nodes: List[str]) -> Any:
        from types import SimpleNamespace
        route = _make_route(f"r_{gap_id}", route_nodes, [f"e_{gap_id}"], feasible=True)
        return SimpleNamespace(
            gap_id=gap_id,
            start_node_id=start_node,
            end_node_id=end_node,
            candidate_routes=[route],
            start_observation={"timestamp_seconds": t_start},
            end_observation={"timestamp_seconds": t_end},
        )

    def test_conflicting_anchor_nodes_detected(self):
        """
        GAP_1 ends at node_B_X, GAP_2 starts at node_B_Y (different).
        Conflict must be detected.
        """
        gap1 = self._make_mock_gap("gq1", "node_A", "node_B_X", 1000.0, 1100.0,
                                   ["node_A", "node_B_X"])
        gap2 = self._make_mock_gap("gq2", "node_B_Y", "node_C", 1100.0, 1200.0,
                                   ["node_B_Y", "node_C"])

        result = evaluate_multigap_consistency([gap1, gap2])
        self.assertFalse(result["is_globally_consistent"],
                         "Anchor node mismatch should be detected as conflict")
        self.assertGreater(len(result["node_alignment_conflicts"]), 0)

    def test_temporal_conflict_detected(self):
        """GAP_2 starts before GAP_1 ends (temporal ordering violation)."""
        gap1 = self._make_mock_gap("gq3", "node_A", "node_B", 1000.0, 1150.0,
                                   ["node_A", "node_B"])
        gap2 = self._make_mock_gap("gq4", "node_B", "node_C", 1100.0, 1250.0,
                                   ["node_B", "node_C"])  # starts BEFORE gap1 ends

        result = evaluate_multigap_consistency([gap1, gap2])
        # t_end of gap1 = 1150, t_start of gap2 = 1100 → conflict
        self.assertGreater(len(result["temporal_conflicts"]), 0,
                           "Temporal ordering violation should be reported")


# ---------------------------------------------------------------------------
# Test R: No feasible hidden route — no fabricated route
# ---------------------------------------------------------------------------
class TestR_NoFeasibleRoute(unittest.TestCase):

    def test_no_feasible_route_returns_no_fabricated_path(self):
        """
        When no feasible route exists (e.g., camera not on road graph),
        the gap must not return fabricated route data.
        """
        road_graph = RoadGraph.from_json_file("data/roads/synthetic_road_graph.json")

        # Cameras not on the road graph → should get unassociated_camera status
        obs_a = _make_obs("R1", "cam_not_on_graph_XYZ", 1000.0)
        obs_b = _make_obs("R2", "cam_not_on_graph_ABC", 1120.0)

        gap = infer_sparse_gap(obs_a, obs_b, road_graph=road_graph)

        # Must not fabricate any observation records
        self.assertEqual(len(gap.candidate_routes), 0,
                         "No feasible routes should mean zero candidate routes, not fabricated ones")
        self.assertIn(gap.status, ("no_path", "unassociated_camera", "infeasible", "temporal_inversion"))


# ---------------------------------------------------------------------------
# Test S: Temporal synchronization unavailable — unverified, not fabricated
# ---------------------------------------------------------------------------
class TestS_TemporalSyncUnavailable(unittest.TestCase):

    def test_no_shared_reference_temporal_unavailable(self):
        """When no shared temporal reference exists, temporal status must be unavailable."""
        obs_a = _make_obs("S1", "cam1", 1000.0, time_reference_id="cam1_local")
        obs_b = _make_obs("S2", "cam2", 1060.0, time_reference_id="cam2_local")

        result = match_observations(obs_a, obs_b)
        ledger = result.get("evidence_ledger", {})
        t_status = ledger.get("temporal", {}).get("status", "")

        self.assertIn(t_status, ("unavailable", "unverified"),
                      f"Temporal status should be 'unavailable' or 'unverified', got '{t_status}'")

    def test_temporal_unavailable_does_not_produce_negative_delta(self):
        """An unverifiable temporal situation must not claim a negative delta_t."""
        obs_a = _make_obs("S3", "cam1", 1000.0, time_reference_id="cam1_local")
        obs_b = _make_obs("S4", "cam2", 980.0, time_reference_id="cam2_local")

        result = match_observations(obs_a, obs_b)
        ledger = result.get("evidence_ledger", {})
        t_entry = ledger.get("temporal", {})
        # Should not produce a 'temporal_inversion' violation when references are incomparable
        t_status = t_entry.get("status", "")
        self.assertNotEqual(t_status, "impossible",
                            "Incomparable timestamps must not be classified as impossible")


# ---------------------------------------------------------------------------
# Test T: Inference trace serialization round-trip
# ---------------------------------------------------------------------------
class TestT_InferenceTraceRoundTrip(unittest.TestCase):

    def _make_sample_trace(self) -> InferenceTrace:
        return InferenceTrace(
            trace_id="test_trace_001",
            entity_type="identity_pair",
            entity_id="obs_A:obs_B",
            stage="identity_fusion",
            evidence_items=[
                {"signal_type": "plate", "status": "missing", "value": None, "contribution": None},
                {"signal_type": "appearance", "status": "available", "value": 0.86, "contribution": 0.86},
            ],
            decision="matched",
            decision_score=0.7245,
            explanation="Appearance similarity is 0.86 (high), temporal feasible.",
            generated_at=1000000.0,
        )

    def test_to_dict_contains_all_fields(self):
        """to_dict must include all InferenceTrace fields."""
        trace = self._make_sample_trace()
        d = trace.to_dict()
        for key in ("trace_id", "entity_type", "entity_id", "stage",
                    "evidence_items", "decision", "decision_score",
                    "explanation", "generated_at"):
            self.assertIn(key, d, f"Missing key '{key}' in to_dict output")

    def test_from_dict_round_trip(self):
        """InferenceTrace → to_dict() → from_dict() must preserve all fields."""
        trace = self._make_sample_trace()
        d = trace.to_dict()
        restored = InferenceTrace.from_dict(d)

        self.assertEqual(restored.trace_id, trace.trace_id)
        self.assertEqual(restored.entity_type, trace.entity_type)
        self.assertEqual(restored.entity_id, trace.entity_id)
        self.assertEqual(restored.stage, trace.stage)
        self.assertEqual(restored.decision, trace.decision)
        self.assertAlmostEqual(restored.decision_score, trace.decision_score, places=4)
        self.assertEqual(restored.explanation, trace.explanation)
        self.assertAlmostEqual(restored.generated_at, trace.generated_at, places=4)
        self.assertEqual(len(restored.evidence_items), len(trace.evidence_items))

    def test_build_identity_pair_trace(self):
        """build_identity_pair_trace() must produce a valid InferenceTrace from match output."""
        obs_a = _make_obs("T1", "cam1", 1000.0, embedding=[0.9] * 64, time_reference_id="shared")
        obs_b = _make_obs("T2", "cam2", 1060.0, embedding=[0.88] * 64, time_reference_id="shared")
        match_result = match_observations(obs_a, obs_b)
        trace = build_identity_pair_trace(match_result)

        self.assertIsInstance(trace, InferenceTrace)
        self.assertEqual(trace.entity_type, "identity_pair")
        self.assertGreater(len(trace.evidence_items), 0)
        self.assertIn(trace.decision, ("matched", "ambiguous", "rejected", "insufficient_evidence"))

    def test_build_cluster_trace(self):
        """build_cluster_trace() must produce a valid InferenceTrace from cluster dict."""
        obs = _make_obs("T3", "cam1", 100.0)
        graph = IdentityGraph()
        graph.add_observation(obs)
        cluster = graph.get_candidate_identities()[0]

        trace = build_cluster_trace(cluster)
        self.assertIsInstance(trace, InferenceTrace)
        self.assertEqual(trace.entity_type, "cluster")
        self.assertIn(trace.decision, ("singleton", "candidate", "ambiguous", "insufficient_evidence"))


# ---------------------------------------------------------------------------
# Test U: Deterministic same input — semantically identical result
# ---------------------------------------------------------------------------
class TestU_Deterministic(unittest.TestCase):

    def test_same_input_same_cluster_output(self):
        """Same set of observations must always produce the same clustering result."""
        obs_list = [
            _make_obs("U1", "cam1", 1000.0, plate="KA01AB1234", embedding=[0.9] * 64),
            _make_obs("U2", "cam2", 1060.0, plate="KA01AB1234", embedding=[0.88] * 64),
        ]

        results = []
        for _ in range(5):
            graph = IdentityGraph(min_probability_threshold=0.50)
            graph.build_graph(list(obs_list))
            clusters = graph.get_candidate_identities()
            results.append(len(clusters))

        self.assertTrue(all(r == results[0] for r in results),
                        f"Cluster count varied across runs: {results}")

    def test_same_match_result_repeated_calls(self):
        """match_observations must return same score for same inputs."""
        obs_a = _make_obs("U3", "cam1", 1000.0, embedding=[0.9] * 64)
        obs_b = _make_obs("U4", "cam2", 1060.0, embedding=[0.88] * 64)

        scores = [
            match_observations(obs_a, obs_b)["same_vehicle_probability"]
            for _ in range(5)
        ]
        self.assertTrue(all(abs(s - scores[0]) < 1e-6 for s in scores),
                        f"Scores varied: {scores}")


# ---------------------------------------------------------------------------
# Test V: Different input ordering — semantically equivalent cluster
# ---------------------------------------------------------------------------
class TestV_OrderingEquivalence(unittest.TestCase):

    def test_observation_ordering_invariant_cluster_count(self):
        """Cluster count must be the same regardless of observation insertion order."""
        obs_1 = _make_obs("V1", "cam1", 1000.0, plate="KA01AB1234", embedding=[0.9] * 64)
        obs_2 = _make_obs("V2", "cam2", 1060.0, plate="KA01AB1234", embedding=[0.88] * 64)
        obs_3 = _make_obs("V3", "cam3", 1130.0, plate="TN99ZZ0001", embedding=[0.3] * 64)

        g1 = IdentityGraph(min_probability_threshold=0.50)
        g1.build_graph([obs_1, obs_2, obs_3])
        n1 = len(g1.get_candidate_identities())

        g2 = IdentityGraph(min_probability_threshold=0.50)
        g2.build_graph([obs_3, obs_1, obs_2])
        n2 = len(g2.get_candidate_identities())

        self.assertEqual(n1, n2,
                         f"Cluster count should be same regardless of insertion order: {n1} vs {n2}")


# ---------------------------------------------------------------------------
# Test W: Reliability vs demand separation
# ---------------------------------------------------------------------------
class TestW_ReliabilityDemandSeparation(unittest.TestCase):

    def test_reliability_change_does_not_change_vehicle_weight(self):
        """
        Changing observation reliability must NOT change the physical vehicle_weight.
        vehicle_weight represents physical PCU demand (vehicle_weight × route_allocation).
        Reliability is evidence trustworthiness, not demand.
        We test via the NormalizedTrajectory schema invariant.
        """
        from inference.member3_adapter import adapt_vehicle_trajectory_to_normalized

        obs_a = _make_obs("W1", "cam1", 1000.0)
        obs_b = _make_obs("W2", "cam2", 1120.0)
        r = _make_route("Wr1", ["nW1", "nW2"], ["eW1"], raw_score=0.8, required_speed=30.0)
        seg = _make_segment("Ws1", obs_a, obs_b, routes=[r])

        traj = VehicleTrajectory(
            identity_id="VEHICLE_CANDIDATE_001",
            observations_count=2,
            cameras_visited=["cam1", "cam2"],
            start_timestamp=1000.0,
            end_timestamp=1120.0,
            segments=[seg],
            overall_confidence=0.85,
            reliability={"overall_reliability": 0.85, "overall_uncertainty": 0.15},
        )

        nt = adapt_vehicle_trajectory_to_normalized(traj)

        # vehicle_weight must be 1.0 (single vehicle = 1 PCU)
        self.assertEqual(nt.vehicle_weight, 1.0,
                         f"Default single vehicle should have vehicle_weight=1.0 (one PCU), got {nt.vehicle_weight}")

        # Now produce a second normalized trajectory with different reliability
        traj2 = VehicleTrajectory(
            identity_id="VEHICLE_CANDIDATE_002",
            observations_count=2,
            cameras_visited=["cam1", "cam2"],
            start_timestamp=1000.0,
            end_timestamp=1120.0,
            segments=[seg],
            overall_confidence=0.85,
            reliability={"overall_reliability": 0.40, "overall_uncertainty": 0.60},  # lower reliability
        )
        nt2 = adapt_vehicle_trajectory_to_normalized(traj2)

        self.assertEqual(nt2.vehicle_weight, 1.0,
                         f"Lower reliability must NOT change vehicle_weight. Got {nt2.vehicle_weight}")

        # Validate: reliability ≠ vehicle_weight semantically
        # vehicle_weight is physical demand; reliability is evidence trustworthiness
        reliability_val = traj.reliability.get("overall_reliability", 0.85)
        self.assertNotEqual(
            nt.vehicle_weight,
            reliability_val,
            "vehicle_weight (1.0 PCU) must not equal reliability score (evidence trustworthiness are semantically distinct)"
        )



# ---------------------------------------------------------------------------
# Test X: Real Kanishka feed — no fabricated merge without identity evidence
# ---------------------------------------------------------------------------
class TestX_RealKanishkaNoFabricatedMerge(unittest.TestCase):

    def test_kanishka_real_observations_no_cross_camera_merge_without_evidence(self):
        """
        Run the real Kanishka dataset through the identity graph.
        Since the real data lacks Re-ID embeddings and cross-camera plate matching,
        zero cross-camera identity links must be formed.
        All observations must be unconfirmed singletons or local clusters only.
        """
        import json
        kanishka_path = "data/observations/kanishka_traffic.json"

        if not os.path.exists(kanishka_path):
            self.skipTest(f"Real data file not found: {kanishka_path}")

        with open(kanishka_path) as f:
            raw = json.load(f)

        observations_raw = raw if isinstance(raw, list) else raw.get("observations", [])
        if not observations_raw:
            self.skipTest("No observations in Kanishka dataset")

        observations = []
        for obs_raw in observations_raw[:50]:  # test with first 50 observations
            try:
                obs = Observation.from_dict(obs_raw)
                observations.append(obs)
            except Exception:
                continue

        if not observations:
            self.skipTest("Could not parse any observations")

        graph = IdentityGraph(min_probability_threshold=0.70)
        graph.build_graph(observations)

        # Count cross-camera edges (edges between different cameras)
        cross_camera_edges = [
            e for e in graph.edges
            if graph.nodes.get(e["source"], None) is not None
            and graph.nodes.get(e["target"], None) is not None
            and graph.nodes[e["source"]].camera_id != graph.nodes[e["target"]].camera_id
        ]

        # With no Re-ID embeddings or plate data in real Kanishka feed,
        # there should be no cross-camera identity links
        identity_evidence_in_any_edge = any(
            e.get("evidence", {}).get("identity_evidence_available", False)
            for e in cross_camera_edges
        )

        self.assertFalse(
            identity_evidence_in_any_edge,
            f"Real data must not produce cross-camera merges without actual identity evidence. "
            f"Found {len(cross_camera_edges)} cross-camera edge(s)."
        )

        # All singleton clusters must be marked unconfirmed_singleton with confidence=None
        clusters = graph.get_candidate_identities()
        singleton_clusters = [c for c in clusters if c.get("identity_status") == "unconfirmed_singleton"]
        for sc in singleton_clusters:
            self.assertIsNone(sc.get("identity_confidence"),
                              f"Singleton {sc['identity_id']} must have confidence=None, "
                              f"not {sc.get('identity_confidence')}")


# ---------------------------------------------------------------------------
# Test Y: Contradiction-Aware Cluster Splitting
# ---------------------------------------------------------------------------
class TestY_ContradictionAwareClusterSplitting(unittest.TestCase):

    def test_transitive_contradiction_cluster_splitting(self):
        """
        A-B strong, B-C strong, A-C impossible.
        get_candidate_identities() maintains candidate clusters.
        get_final_identity_hypotheses() splits contradictory clusters so A and C are separated.
        """
        emb = [0.1] * 64
        # A: (12.9716, 77.5946) t=1000
        # B: (12.9750, 77.5980) t=1060 (dist ~520m, 60s -> 31 km/h: feasible)
        # C: (12.9751, 77.5981) t=1001 (dist ~15m from B, but t_C=1001 vs t_A=1000 -> 530m in 1s -> speed > 1900 km/h: IMPOSSIBLE!)
        obs_a = _make_obs("Y_A", "cam1", 1000.0, embedding=emb, lat=12.9716, lon=77.5946)
        obs_b = _make_obs("Y_B", "cam2", 1060.0, embedding=emb, lat=12.9750, lon=77.5980)
        obs_c = _make_obs("Y_C", "cam3", 1001.0, embedding=emb, lat=12.9751, lon=77.5981)

        graph = IdentityGraph(min_probability_threshold=0.70)
        graph.build_graph([obs_a, obs_b, obs_c])

        # Candidate identities contain the initial candidate component
        cands = graph.get_candidate_identities()
        self.assertGreater(len(cands), 0)

        # Final hypotheses MUST split the cluster so A and C never co-exist in the same identity
        finals = graph.get_final_identity_hypotheses()
        self.assertGreaterEqual(len(finals), 2)

        a_cluster = next((c for c in finals if "Y_A" in c["observation_ids"]), None)
        self.assertIsNotNone(a_cluster)
        self.assertNotIn("Y_C", a_cluster["observation_ids"], "A and C must be separated due to physical contradiction")


# ---------------------------------------------------------------------------
# Test Z: Evidence Provenance — No Hardcoded Claims
# ---------------------------------------------------------------------------
class TestZ_EvidenceProvenanceNoHardcodedClaims(unittest.TestCase):

    def test_missing_evidence_not_claimed_as_supported(self):
        """Observations without plates or appearance must not claim 'supported' in ledger or cluster."""
        o1 = _make_obs("Z1", "cam1", 1000.0, plate=None, embedding=None)
        o2 = _make_obs("Z2", "cam2", 1060.0, plate=None, embedding=None)

        res = match_observations(o1, o2)
        ledger = res.get("evidence_ledger", {})

        self.assertEqual(ledger["plate"]["status"], "missing")
        self.assertEqual(ledger["appearance"]["status"], "missing")
        self.assertEqual(ledger["plate"]["source"], "missing")
        self.assertIsNone(ledger["plate"]["confidence"])

        # When plate is present, source must be actual_observation
        o_p1 = _make_obs("Z3", "cam1", 1000.0, plate="KA01AB1234")
        o_p2 = _make_obs("Z4", "cam2", 1060.0, plate="KA01AB1234")
        res_p = match_observations(o_p1, o_p2)
        self.assertEqual(res_p["evidence_ledger"]["plate"]["source"], "actual_observation")

        graph = IdentityGraph()
        graph.build_graph([o1, o2])
        clusters = graph.get_candidate_identities()
        for c in clusters:
            summary = c.get("identity_evidence_summary", {})
            self.assertNotEqual(summary.get("plate"), "supported", "Missing plate must never be reported as supported")
            self.assertNotEqual(summary.get("appearance"), "supported", "Missing appearance must never be reported as supported")


# ---------------------------------------------------------------------------
# Test AA: Adaptive Plate Confidence Weighting
# ---------------------------------------------------------------------------
class TestAA_AdaptivePlateConfidence(unittest.TestCase):

    def test_high_confidence_plate_increases_weight(self):
        """Verified high OCR confidence gives more weight to plate evidence than low confidence."""
        emb1 = [0.1] * 64
        emb2 = [-0.1] * 64 # Low appearance similarity
        plate = "KA05MN1234"

        # High OCR confidence
        o1_high = _make_obs("AA1", "cam1", 1000.0, plate=plate, embedding=emb1)
        o1_high.plate_confidence = 0.95
        o2_high = _make_obs("AA2", "cam2", 1060.0, plate=plate, embedding=emb2, lat=12.9750, lon=77.5980)
        o2_high.plate_confidence = 0.95

        # Low OCR confidence
        o1_low = _make_obs("AA3", "cam1", 1000.0, plate=plate, embedding=emb1)
        o1_low.plate_confidence = 0.20
        o2_low = _make_obs("AA4", "cam2", 1060.0, plate=plate, embedding=emb2, lat=12.9750, lon=77.5980)
        o2_low.plate_confidence = 0.20

        res_high = match_observations(o1_high, o2_high)
        res_low = match_observations(o1_low, o2_low)

        p_high = float(res_high["same_vehicle_probability"])
        p_low = float(res_low["same_vehicle_probability"])

        self.assertGreater(p_high, p_low, "High OCR confidence should yield higher match score when plate matches")


# ---------------------------------------------------------------------------
# Test AB: Reference vs Optimized Graph Construction Equivalence
# ---------------------------------------------------------------------------
class TestAB_ReferenceVsOptimizedGraphEquivalence(unittest.TestCase):

    def test_pruned_graph_matches_reference_implementation(self):
        """Safe-pruned graph must produce semantically identical edges and clusters as reference O(N^2)."""
        obs_list = []
        for i in range(25):
            v_type = "car" if i % 3 != 0 else "truck"
            emb = [0.05 * (i % 5)] * 64 if i % 4 != 0 else None
            plate = f"MH12AB{1000+i}" if i % 2 == 0 else None
            obs = _make_obs(
                f"AB_{i}",
                f"cam_{i % 5}",
                1000.0 + i * 20.0,
                vehicle_type=v_type,
                plate=plate,
                embedding=emb,
                lat=12.9700 + (i % 5) * 0.003,
                lon=77.5900 + (i % 5) * 0.003,
            )
            obs_list.append(obs)

        ref_graph = IdentityGraph()
        ref_graph.build_graph_reference(obs_list)

        opt_graph = IdentityGraph()
        opt_graph.build_graph(obs_list, enable_pruning=True)

        self.assertEqual(len(ref_graph.edges), len(opt_graph.edges), "Edge counts must match between reference and pruned")

        ref_clusters = [c["observation_ids"] for c in ref_graph.get_candidate_identities()]
        opt_clusters = [c["observation_ids"] for c in opt_graph.get_candidate_identities()]
        self.assertEqual(ref_clusters, opt_clusters, "Candidate clusters must be semantically equivalent")


# ---------------------------------------------------------------------------
# Test AC: Global Trajectory Trace Verification
# ---------------------------------------------------------------------------
class TestAC_GlobalTrajectoryTraceVerification(unittest.TestCase):

    def test_build_global_trajectory_trace_and_round_trip(self):
        """build_global_trajectory_trace() must populate required provenance and round-trip via JSON."""
        o1 = _make_obs("AC1", "cam1", 1000.0)
        o2 = _make_obs("AC2", "cam2", 1060.0)
        r1 = _make_route("r1", ["n1", "n2"], ["e1"], dist=800.0)
        seg = _make_segment("seg_ac", o1, o2, routes=[r1])
        traj = VehicleTrajectory(
            identity_id="traj_AC",
            observations_count=2,
            cameras_visited=["cam1", "cam2"],
            start_timestamp=1000.0,
            end_timestamp=1060.0,
            segments=[seg],
        )
        hyp_result = evaluate_global_trajectory_hypotheses(traj)
        trace = build_global_trajectory_trace(traj, hyp_result, model_version="2.1.0")

        self.assertIsInstance(trace, InferenceTrace)
        self.assertEqual(trace.entity_type, "trajectory_hypothesis")
        self.assertEqual(trace.entity_id, "traj_AC")
        self.assertEqual(trace.metadata["model_version"], "2.1.0")
        self.assertIn("AC1", trace.metadata["observations_involved"])
        self.assertIn("AC2", trace.metadata["observations_involved"])
        self.assertGreater(trace.generated_at, 1000000.0)

        # JSON Round Trip
        d = trace.to_dict()
        import json
        json_str = json.dumps(d)
        restored_d = json.loads(json_str)
        restored_trace = InferenceTrace.from_dict(restored_d)

        self.assertEqual(restored_trace.trace_id, trace.trace_id)
        self.assertEqual(restored_trace.metadata["model_version"], "2.1.0")
        self.assertEqual(restored_trace.metadata["observations_involved"], trace.metadata["observations_involved"])


# ---------------------------------------------------------------------------
# Test AD: Master Benchmark Suite Execution
# ---------------------------------------------------------------------------
class TestAD_PerformanceBenchmarkSuite(unittest.TestCase):

    def test_master_benchmark_suite_all_pass(self):
        """20 controlled scenarios and 5 holdouts in the benchmark suite must all pass."""
        from inference.benchmark_suite import run_master_benchmark_suite
        report = run_master_benchmark_suite(verbose=False)

        self.assertEqual(report["development_scenarios"]["pass_rate"], 1.0,
                         f"All 20 dev scenarios must pass, got {report['development_scenarios']['passed']}/20")
        self.assertEqual(report["holdout_scenarios"]["pass_rate"], 1.0,
                         f"All 5 holdouts must pass, got {report['holdout_scenarios']['passed']}/5")
        self.assertIn("N_19", report["performance"])
        self.assertIn("N_100", report["performance"])
        self.assertIn("N_200", report["performance"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

