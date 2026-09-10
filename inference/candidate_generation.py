"""
UrbanTrack AI — Spatio-Temporal Indexed Candidate Pair Generator.

Replaces brute-force O(N^2) pairwise iteration with an indexed candidate generation
layer using safe physical, temporal, and topological constraints.

Zero-Tolerance Invariant:
Candidate pruning MUST NOT remove reference-feasible candidates (semantic equivalence).
"""

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from schemas.observation_schema import Observation
from .identity_fusion import match_observations
from .similarity import geographic_distance, plate_similarity, vehicle_type_compatibility
from .temporal import check_temporal_comparability


@dataclass
class CandidateGenerationReport:
    """Detailed diagnostic and benchmarking metrics for candidate generation."""
    n_observations: int
    brute_force_pairs: int
    candidate_pairs_generated: int
    pruned_pairs_count: int
    candidate_reduction_pct: float
    elapsed_generation_ms: float
    brute_force_reference_edges: int
    candidate_pipeline_edges: int
    false_exclusions_count: int
    semantic_equivalence_verified: bool
    rejection_breakdown: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_observations": self.n_observations,
            "brute_force_pairs": self.brute_force_pairs,
            "candidate_pairs_generated": self.candidate_pairs_generated,
            "pruned_pairs_count": self.pruned_pairs_count,
            "candidate_reduction_pct": round(self.candidate_reduction_pct, 2),
            "elapsed_generation_ms": round(self.elapsed_generation_ms, 2),
            "brute_force_reference_edges": self.brute_force_reference_edges,
            "candidate_pipeline_edges": self.candidate_pipeline_edges,
            "false_exclusions_count": self.false_exclusions_count,
            "semantic_equivalence_verified": self.semantic_equivalence_verified,
            "rejection_breakdown": dict(sorted(self.rejection_breakdown.items())),
        }


class CandidateGenerator:
    """
    Spatio-temporal indexed candidate generator for multi-camera vehicle tracking.

    Applies safe physical and topological pruning filters before expensive identity
    fusion reasoning, drastically reducing the pairwise search space while preserving
    every reference-feasible candidate edge (semantic equivalence).
    """

    def __init__(
        self,
        max_speed_kmh: float = 120.0,
        max_time_window_seconds: float = 7200.0,  # 2-hour maximum corridor horizon
        min_probability_threshold: float = 0.70,
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.max_speed_kmh = max_speed_kmh
        self.max_time_window_seconds = max_time_window_seconds
        self.min_probability_threshold = min_probability_threshold
        self.camera_metadata = camera_metadata or {}
        self.config = config or {}

    def generate_candidates(
        self,
        observations: List[Observation],
    ) -> Tuple[List[Tuple[Observation, Observation]], Dict[str, int]]:
        """
        Generate candidate pairs using safe spatio-temporal and attribute indexing.

        Returns:
            Tuple of:
            - List of (obs_a, obs_b) candidate pairs ordered chronologically.
            - Dictionary of rejection reason counts for diagnostic explainability.
        """
        rejection_counts: Dict[str, int] = {
            "temporal_horizon_exceeded": 0,
            "simultaneous_different_cameras": 0,
            "physically_impossible_speed": 0,
            "incompatible_vehicle_type": 0,
            "strong_plate_contradiction": 0,
            "missing_identity_evidence": 0,
        }

        # Deterministic chronological sort: (timestamp_seconds, observation_id)
        sorted_obs = sorted(observations, key=lambda o: (o.timestamp_seconds, o.observation_id))
        n = len(sorted_obs)
        candidates: List[Tuple[Observation, Observation]] = []

        for i in range(n):
            obs_a = sorted_obs[i]
            has_id_a = (obs_a.appearance_embedding is not None and len(obs_a.appearance_embedding) > 0) or (obs_a.plate is not None)
            clean_plate_a = "".join(c for c in str(obs_a.plate or "").upper() if c.isalnum())
            conf_a = float(obs_a.plate_confidence) if obs_a.plate_confidence is not None else 1.0

            for j in range(i + 1, n):
                obs_b = sorted_obs[j]
                has_id_b = (obs_b.appearance_embedding is not None and len(obs_b.appearance_embedding) > 0) or (obs_b.plate is not None)

                # 1. Temporal window indexing:
                # If clocks are shared or synchronized, time cannot exceed max_time_window_seconds.
                # Since array is sorted, any subsequent observation k > j will also exceed window.
                t_check = check_temporal_comparability(obs_a, obs_b, camera_metadata=self.camera_metadata)
                is_sync = t_check.get("comparable", False)

                if is_sync:
                    dt = float(t_check.get("delta_seconds", 0.0))
                    if dt > self.max_time_window_seconds:
                        # Can break inner loop because sorted_obs[k].timestamp_seconds >= sorted_obs[j].timestamp_seconds
                        rejection_counts["temporal_horizon_exceeded"] += (n - j)
                        break

                    # Simultaneous on different cameras
                    if dt == 0.0 and obs_a.camera_id != obs_b.camera_id:
                        rejection_counts["simultaneous_different_cameras"] += 1
                        continue

                    # Physically impossible travel speed
                    if (
                        obs_a.latitude is not None and obs_a.longitude is not None
                        and obs_b.latitude is not None and obs_b.longitude is not None
                        and dt > 0.0
                    ):
                        dist_m = geographic_distance(obs_a.latitude, obs_a.longitude, obs_b.latitude, obs_b.longitude)
                        speed_kmh = (dist_m / dt) * 3.6
                        if speed_kmh > self.max_speed_kmh:
                            rejection_counts["physically_impossible_speed"] += 1
                            continue

                # 2. Vehicle type compatibility (only prune when both are known and incompatible)
                if obs_a.vehicle_type and obs_b.vehicle_type:
                    _, vt_stat = vehicle_type_compatibility(obs_a.vehicle_type, obs_b.vehicle_type)
                    if vt_stat == "incompatible":
                        rejection_counts["incompatible_vehicle_type"] += 1
                        continue

                # 3. Strong license plate contradiction
                if obs_a.plate is not None and obs_b.plate is not None:
                    clean_plate_b = "".join(c for c in str(obs_b.plate or "").upper() if c.isalnum())
                    conf_b = float(obs_b.plate_confidence) if obs_b.plate_confidence is not None else 1.0
                    if len(clean_plate_a) >= 4 and len(clean_plate_b) >= 4 and conf_a >= 0.50 and conf_b >= 0.50:
                        sim = plate_similarity(obs_a.plate, obs_b.plate)
                        if sim < 0.35:
                            rejection_counts["strong_plate_contradiction"] += 1
                            continue

                # 4. Absence of identity evidence when threshold > 0.50
                if self.min_probability_threshold > 0.50 and not (has_id_a or has_id_b):
                    rejection_counts["missing_identity_evidence"] += 1
                    continue

                # All safe filters passed: add candidate pair
                candidates.append((obs_a, obs_b))

        return candidates, rejection_counts

    def evaluate_scalability(
        self,
        observations: List[Observation],
    ) -> CandidateGenerationReport:
        """
        Evaluate candidate generation efficiency and verify semantic equivalence
        against brute-force reference matching.
        """
        n = len(observations)
        total_pairs = (n * (n - 1)) // 2

        # 1. Run indexed candidate generation
        t0 = time.perf_counter()
        candidates, rejection_breakdown = self.generate_candidates(observations)
        elapsed_gen_ms = (time.perf_counter() - t0) * 1000.0

        n_candidates = len(candidates)
        pruned_count = total_pairs - n_candidates
        reduction_pct = (pruned_count / total_pairs * 100.0) if total_pairs > 0 else 0.0

        # 2. Evaluate candidates to find candidate pipeline edges
        candidate_edges = set()
        for obs_a, obs_b in candidates:
            if obs_a.timestamp_seconds > obs_b.timestamp_seconds:
                ea, eb = obs_b, obs_a
            else:
                ea, eb = obs_a, obs_b
            res = match_observations(ea, eb, camera_metadata=self.camera_metadata, config=self.config)
            p = float(res.get("same_vehicle_probability", 0.0))
            has_id = res.get("evidence", {}).get("identity_evidence_available", True)
            if p >= self.min_probability_threshold and has_id:
                candidate_edges.add((min(ea.observation_id, eb.observation_id), max(ea.observation_id, eb.observation_id)))

        # 3. Evaluate reference brute force on all pairs to check semantic equivalence
        reference_edges = set()
        sorted_all = sorted(observations, key=lambda o: (o.timestamp_seconds, o.observation_id))
        for i in range(n):
            for j in range(i + 1, n):
                oa, ob = sorted_all[i], sorted_all[j]
                if oa.timestamp_seconds > ob.timestamp_seconds:
                    ea, eb = ob, oa
                else:
                    ea, eb = oa, ob
                res = match_observations(ea, eb, camera_metadata=self.camera_metadata, config=self.config)
                p = float(res.get("same_vehicle_probability", 0.0))
                has_id = res.get("evidence", {}).get("identity_evidence_available", True)
                if p >= self.min_probability_threshold and has_id:
                    reference_edges.add((min(ea.observation_id, eb.observation_id), max(ea.observation_id, eb.observation_id)))

        # 4. Check for false exclusions (edges found by reference but missed by candidate generation)
        missing_edges = reference_edges - candidate_edges
        false_exclusions = len(missing_edges)
        semantic_equivalent = (false_exclusions == 0 and len(candidate_edges) == len(reference_edges))

        return CandidateGenerationReport(
            n_observations=n,
            brute_force_pairs=total_pairs,
            candidate_pairs_generated=n_candidates,
            pruned_pairs_count=pruned_count,
            candidate_reduction_pct=reduction_pct,
            elapsed_generation_ms=elapsed_gen_ms,
            brute_force_reference_edges=len(reference_edges),
            candidate_pipeline_edges=len(candidate_edges),
            false_exclusions_count=false_exclusions,
            semantic_equivalence_verified=semantic_equivalent,
            rejection_breakdown=rejection_breakdown,
        )
