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
        min_probability_threshold: Optional[float] = None,
        min_score_threshold: float = 0.70,
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.max_speed_kmh = max_speed_kmh
        self.max_time_window_seconds = max_time_window_seconds
        thresh = float(min_probability_threshold) if min_probability_threshold is not None else float(min_score_threshold)
        self.min_score_threshold = thresh
        self.min_probability_threshold = thresh
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

        import bisect

        # Deterministic chronological sort: (timestamp_seconds, observation_id)
        sorted_obs = sorted(observations, key=lambda o: (o.timestamp_seconds, o.observation_id))
        n = len(sorted_obs)
        candidates: List[Tuple[Observation, Observation]] = []

        if n < 2:
            return candidates, rejection_counts

        # Extract indexed sorted timestamps for sub-linear window boundary lookup
        timestamps = [o.timestamp_seconds for o in sorted_obs]

        # 1. Indexed retrieval using temporal window bisect search
        for i in range(n):
            obs_a = sorted_obs[i]
            t_a = obs_a.timestamp_seconds
            has_id_a = (obs_a.appearance_embedding is not None and len(obs_a.appearance_embedding) > 0) or (obs_a.plate is not None)
            clean_plate_a = "".join(c for c in str(obs_a.plate or "").upper() if c.isalnum())
            conf_a = float(obs_a.plate_confidence) if obs_a.plate_confidence is not None else 1.0

            # Find upper bound index in O(log N) using bisect_right
            horizon_limit = t_a + self.max_time_window_seconds
            upper_bound_idx = bisect.bisect_right(timestamps, horizon_limit)

            # Count pruned pairs beyond temporal horizon
            pruned_beyond_horizon = n - upper_bound_idx
            if pruned_beyond_horizon > 0:
                rejection_counts["temporal_horizon_exceeded"] += pruned_beyond_horizon

            # Only iterate through temporally plausible candidate window [i + 1, upper_bound_idx)
            for j in range(i + 1, upper_bound_idx):
                obs_b = sorted_obs[j]
                has_id_b = (obs_b.appearance_embedding is not None and len(obs_b.appearance_embedding) > 0) or (obs_b.plate is not None)

                # Temporal comparability check
                t_check = check_temporal_comparability(obs_a, obs_b, camera_metadata=self.camera_metadata)
                is_sync = t_check.get("comparable", False)

                if is_sync:
                    dt = float(t_check.get("delta_seconds", 0.0))

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
                t_comp = check_temporal_comparability(ea, eb, camera_metadata=self.camera_metadata)
                if t_comp.get("comparable", False) and float(t_comp.get("delta_seconds", 0.0)) > self.max_time_window_seconds:
                    continue
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


def benchmark_candidate_scaling(
    counts: Optional[List[int]] = None,
    base_observations: Optional[List[Observation]] = None,
) -> Dict[str, Any]:
    """
    Benchmark naive brute-force O(N^2) pair generation vs indexed candidate retrieval.
    Measures pair reduction, candidate recall, true exclusions, and runtime across scale levels.

    Args:
        counts: List of observation counts to evaluate (default [100, 250, 500, 1000]).
        base_observations: Optional template observations to replicate/scale.

    Returns:
        Dict[str, Any]: Benchmark summary dictionary reporting per-N scaling metrics.
    """
    import random
    from datetime import datetime

    if counts is None:
        counts = [50, 100, 250, 500]

    results = []
    generator = CandidateGenerator(max_speed_kmh=120.0, max_time_window_seconds=1800.0)

    vehicle_types = ["car", "truck", "bus", "motorcycle"]

    random.seed(42)
    for n in counts:
        # Generate synthetic test observations with realistic temporal spread across 4 cameras
        test_obs = []
        for i in range(n):
            cam_idx = (i % 4) + 1
            t_sec = float(i * 12.0)  # 12s spacing
            vtype = vehicle_types[i % len(vehicle_types)]
            emb = [random.uniform(-0.1, 0.1) for _ in range(8)]
            plate = f"KA01TEST{i % 20:02d}" if (i % 3 != 0) else None
            test_obs.append(
                Observation(
                    observation_id=f"SCALE_OBS_{i:04d}",
                    camera_id=f"CAM_{cam_idx:02d}",
                    timestamp=datetime.fromtimestamp(1000.0 + t_sec),
                    timestamp_seconds=1000.0 + t_sec,
                    vehicle_type=vtype,
                    plate=plate,
                    plate_confidence=0.90 if plate else None,
                    appearance_embedding=emb,
                    timestamp_semantics="synchronized",
                    time_reference_id="city_network_sync",
                )
            )

        total_pairs = (n * (n - 1)) // 2

        # 1. Naive enumeration runtime
        t0 = time.perf_counter()
        naive_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                naive_pairs.append((test_obs[i], test_obs[j]))
        naive_ms = (time.perf_counter() - t0) * 1000.0

        # 2. Indexed retrieval runtime
        t1 = time.perf_counter()
        candidates, breakdown = generator.generate_candidates(test_obs)
        indexed_ms = (time.perf_counter() - t1) * 1000.0

        candidate_count = len(candidates)
        reduction_pct = ((total_pairs - candidate_count) / total_pairs * 100.0) if total_pairs > 0 else 0.0

        # Candidate recall verification: ensure no temporally plausible identical plate is excluded
        candidate_pair_ids = set((min(a.observation_id, b.observation_id), max(a.observation_id, b.observation_id)) for a, b in candidates)
        true_matches_total = 0
        true_matches_found = 0
        for oa, ob in naive_pairs:
            if oa.plate and ob.plate and oa.plate == ob.plate and abs(oa.timestamp_seconds - ob.timestamp_seconds) <= 1800.0:
                true_matches_total += 1
                pid = (min(oa.observation_id, ob.observation_id), max(oa.observation_id, ob.observation_id))
                if pid in candidate_pair_ids:
                    true_matches_found += 1

        candidate_recall = (true_matches_found / true_matches_total * 100.0) if true_matches_total > 0 else 100.0
        speedup = (naive_ms / indexed_ms) if indexed_ms > 0 else 1.0

        results.append({
            "n_observations": n,
            "theoretical_pairs": total_pairs,
            "candidates_generated": candidate_count,
            "pruned_pairs": total_pairs - candidate_count,
            "reduction_pct": round(reduction_pct, 2),
            "candidate_recall_pct": round(candidate_recall, 2),
            "naive_enumeration_ms": round(naive_ms, 2),
            "indexed_retrieval_ms": round(indexed_ms, 2),
            "speedup_factor": round(speedup, 2),
        })

    return {
        "benchmark_name": "candidate_generation_scaling",
        "timestamp": datetime.now().isoformat(),
        "evaluations": results,
    }



def benchmark_end_to_end_scalability(
    counts: Optional[List[int]] = None,
    repetitions: int = 3,
) -> Dict[str, Any]:
    """
    Fair end-to-end scalability benchmark comparing:
    1. Baseline Naive Pipeline: All N*(N-1)/2 pairs -> IdentityFusion -> IdentityGraph construction
    2. Optimized Production Pipeline: CandidateGenerator -> Candidate IdentityFusion -> IdentityGraph construction

    Measures median and p95 runtimes across both candidate generation, fusion, and graph assembly.
    """
    import random
    import statistics
    from datetime import datetime
    from .identity_graph import IdentityGraph

    if counts is None:
        counts = [50, 100, 200, 500]

    vehicle_types = ['car', 'truck', 'bus', 'motorcycle']
    evaluations = []

    for n in counts:
        # Build synthetic observations
        test_obs = []
        for i in range(n):
            cam_idx = (i % 4) + 1
            t_sec = float(i * 10.0)
            vtype = vehicle_types[i % len(vehicle_types)]
            emb = [random.uniform(-0.1, 0.1) for _ in range(8)]
            plate = f'KA01TEST{i % 20:02d}' if (i % 3 != 0) else None
            test_obs.append(
                Observation(
                    observation_id=f'E2E_OBS_{i:04d}',
                    camera_id=f'CAM_{cam_idx:02d}',
                    timestamp=datetime.fromtimestamp(1000.0 + t_sec),
                    timestamp_seconds=1000.0 + t_sec,
                    vehicle_type=vtype,
                    plate=plate,
                    plate_confidence=0.90 if plate else None,
                    appearance_embedding=emb,
                    timestamp_semantics='synchronized',
                    time_reference_id='city_network_sync',
                )
            )

        total_pairs = (n * (n - 1)) // 2

        # 0. Warm-up pass to ensure cold start / import effects do not distort measurements
        warm_pairs = [(test_obs[i], test_obs[j]) for i in range(min(5, n)) for j in range(i + 1, min(5, n))]
        _ = [match_observations(a, b) for a, b in warm_pairs]

        # 1. Baseline Full Pipeline Repetitions (Naive O(N^2) Pairs -> Fusion -> Graph Assembly)
        base_runtimes = []
        base_fusion_runtimes = []
        base_graph_runtimes = []
        base_fused_count = 0
        base_edges = 0
        base_pred_edges = set()

        for _ in range(repetitions):
            t_start = time.perf_counter()
            # Naive pair enumeration
            pairs = [(test_obs[i], test_obs[j]) for i in range(n) for j in range(i + 1, n)]
            t_fusion_start = time.perf_counter()
            fused_res = [match_observations(a, b) for a, b in pairs]
            t_fusion = (time.perf_counter() - t_fusion_start) * 1000.0

            t_graph_start = time.perf_counter()
            g_base = IdentityGraph(min_score_threshold=0.75)
            g_base.build_graph_from_matches(test_obs, pairs, fused_res)
            _ = g_base.get_candidate_identities()
            t_graph = (time.perf_counter() - t_graph_start) * 1000.0
            t_total = (time.perf_counter() - t_start) * 1000.0

            base_runtimes.append(t_total)
            base_fusion_runtimes.append(t_fusion)
            base_graph_runtimes.append(t_graph)
            base_fused_count = len(pairs)
            base_edges = len(g_base.edges)
            base_pred_edges = set((min(e["source"], e["target"]), max(e["source"], e["target"])) for e in g_base.edges)

        # 2. Optimized Candidate Pipeline Repetitions (CandidateGenerator -> Fusion -> Graph Assembly)
        opt_runtimes = []
        opt_gen_runtimes = []
        opt_fusion_runtimes = []
        opt_graph_runtimes = []
        candidates_count = 0
        opt_edges = 0
        opt_pred_edges = set()

        generator = CandidateGenerator(max_speed_kmh=120.0, max_time_window_seconds=1800.0)

        for _ in range(repetitions):
            t_start = time.perf_counter()
            t_gen_start = time.perf_counter()
            candidates, _ = generator.generate_candidates(test_obs)
            t_gen = (time.perf_counter() - t_gen_start) * 1000.0

            t_fusion_start = time.perf_counter()
            fused_res = [match_observations(a, b) for a, b in candidates]
            t_fusion = (time.perf_counter() - t_fusion_start) * 1000.0

            t_graph_start = time.perf_counter()
            g_opt = IdentityGraph(min_score_threshold=0.75)
            g_opt.build_graph_from_matches(test_obs, candidates, fused_res)
            _ = g_opt.get_candidate_identities()
            t_graph = (time.perf_counter() - t_graph_start) * 1000.0
            t_total = (time.perf_counter() - t_start) * 1000.0

            opt_runtimes.append(t_total)
            opt_gen_runtimes.append(t_gen)
            opt_fusion_runtimes.append(t_fusion)
            opt_graph_runtimes.append(t_graph)
            candidates_count = len(candidates)
            opt_edges = len(g_opt.edges)
            opt_pred_edges = set((min(e["source"], e["target"]), max(e["source"], e["target"])) for e in g_opt.edges)

        # Calculate candidate recall of true identical plates within plausible window
        candidate_pair_ids = set((min(a.observation_id, b.observation_id), max(a.observation_id, b.observation_id)) for a, b in candidates)
        true_matches_total = 0
        true_matches_found = 0
        true_pair_ids = set()
        for i in range(n):
            for j in range(i + 1, n):
                oa, ob = test_obs[i], test_obs[j]
                if oa.plate and ob.plate and oa.plate == ob.plate and abs(oa.timestamp_seconds - ob.timestamp_seconds) <= 1800.0:
                    true_matches_total += 1
                    pid = (min(oa.observation_id, ob.observation_id), max(oa.observation_id, ob.observation_id))
                    true_pair_ids.add(pid)
                    if pid in candidate_pair_ids:
                        true_matches_found += 1

        candidate_recall = (true_matches_found / true_matches_total * 100.0) if true_matches_total > 0 else 100.0
        pruned_pairs = total_pairs - candidates_count
        reduction_pct = (pruned_pairs / total_pairs * 100.0) if total_pairs > 0 else 0.0

        base_med = statistics.median(base_runtimes)
        opt_med = statistics.median(opt_runtimes)
        base_p95 = sorted(base_runtimes)[int(0.95 * len(base_runtimes))]
        opt_p95 = sorted(opt_runtimes)[int(0.95 * len(opt_runtimes))]
        speedup = (base_med / opt_med) if opt_med > 0 else 1.0

        # Compute precision, recall, f1 for optimized pipeline edges against true matches
        tp = sum(1 for edge in opt_pred_edges if edge in true_pair_ids)
        fp = len(opt_pred_edges) - tp
        fn = len(true_pair_ids) - tp
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 1.0

        evaluations.append({
            'n_observations': n,
            'theoretical_pairs': total_pairs,
            'candidates_generated': candidates_count,
            'candidate_pairs': candidates_count,
            'pruned_pairs': pruned_pairs,
            'reduction_pct': round(reduction_pct, 2),
            'candidate_reduction': round(reduction_pct, 2),
            'candidate_recall_pct': round(candidate_recall, 2),
            'candidate_recall': round(candidate_recall, 2),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'baseline_pipeline': {
                'total_runtime_median_ms': round(base_med, 2),
                'total_runtime_p95_ms': round(base_p95, 2),
                'fusion_runtime_median_ms': round(statistics.median(base_fusion_runtimes), 2),
                'graph_runtime_median_ms': round(statistics.median(base_graph_runtimes), 2),
                'fused_pairs_count': base_fused_count,
                'final_edges': base_edges,
            },
            'optimized_pipeline': {
                'total_runtime_median_ms': round(opt_med, 2),
                'total_runtime_p95_ms': round(opt_p95, 2),
                'candidate_gen_median_ms': round(statistics.median(opt_gen_runtimes), 2),
                'fusion_runtime_median_ms': round(statistics.median(opt_fusion_runtimes), 2),
                'graph_runtime_median_ms': round(statistics.median(opt_graph_runtimes), 2),
                'fused_pairs_count': candidates_count,
                'final_edges': opt_edges,
            },
            'speedup_factor': round(speedup, 2),
        })

    return {
        'benchmark_name': 'fair_end_to_end_scalability_benchmark',
        'repetitions': repetitions,
        'timestamp': datetime.now().isoformat(),
        'evaluations': evaluations,
    }
