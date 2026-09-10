"""
Identity Graph representation and candidate vehicle identity clustering for UrbanTrack AI.

Phase C: Adds cluster-level consistency validation.
Phase D: Fixes singleton semantics — singletons receive identity_status='unconfirmed_singleton',
         NOT artificial identity_confidence=1.0.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Set, Tuple

from schemas.observation_schema import Observation
from .identity_fusion import match_observations


# ---------------------------------------------------------------------------
# Cluster Consistency Result
# ---------------------------------------------------------------------------

class ClusterConsistencyResult(NamedTuple):
    """
    Structured outcome of cluster-level consistency validation.

    Fields:
        is_consistent: True only when no temporal, spatial, or vehicle-type violations detected.
        status: 'consistent' | 'inconsistent' | 'insufficient_evidence' | 'unverified' | 'singleton'
        consistency_score: Heuristic score in [0.0, 1.0] (lower = more violated; None = insufficient evidence).
        temporal_consistency: 'consistent' | 'inconsistent' | 'unavailable' | 'unverified'
        spatial_consistency: 'consistent' | 'inconsistent' | 'unavailable'
        vehicle_type_consistency: 'compatible' | 'incompatible' | 'unknown'
        violations: List of violation description strings.
        warnings: List of non-blocking warning strings.
        contradictory_edges: List of (obs_id_a, obs_id_b) pairs whose direct transition is contradictory.
    """
    is_consistent: bool
    status: str
    consistency_score: Optional[float]
    temporal_consistency: str
    spatial_consistency: str
    vehicle_type_consistency: str
    violations: List[str]
    warnings: List[str]
    contradictory_edges: List[Tuple[str, str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_consistent": self.is_consistent,
            "status": self.status,
            "consistency_score": round(self.consistency_score, 4) if self.consistency_score is not None else None,
            "temporal_consistency": self.temporal_consistency,
            "spatial_consistency": self.spatial_consistency,
            "vehicle_type_consistency": self.vehicle_type_consistency,
            "violations": list(self.violations),
            "warnings": list(self.warnings),
            "contradictory_edges": [list(e) for e in self.contradictory_edges],
        }


class IdentityGraph:
    """
    Graph representation of vehicle observations and identity match probabilities.

    Nodes: Vehicle observations.
    Edges: Pairwise candidate identity links formed when evidence score meets the configured threshold.
    """

    def __init__(self, min_probability_threshold: float = 0.70) -> None:
        """
        Initialize the IdentityGraph.

        Args:
            min_probability_threshold: Configured identity-link threshold: 0.70 evidence score.
                                       A decision parameter requiring sufficient evidence to avoid transitive over-clustering.
                                       Note: This is an evidence-score decision threshold, not a calibrated probability.
        """
        self.min_threshold = min_probability_threshold
        self.nodes: Dict[str, Observation] = {}
        self.edges: List[Dict[str, Any]] = []
        self.adjacency: Dict[str, List[Tuple[str, float]]] = {}

    def add_observation(self, obs: Observation) -> None:
        """Add an observation node to the graph."""
        if obs.observation_id not in self.nodes:
            self.nodes[obs.observation_id] = obs
            self.adjacency[obs.observation_id] = []

    def build_graph(
        self,
        observations: List[Observation],
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Build pairwise edges across observations using the Identity Fusion Engine.

        Args:
            observations: List of vehicle observations to add and compare.
            camera_metadata: Optional camera metadata mapping.
            config: Optional fusion engine configuration settings.
        """
        for obs in observations:
            self.add_observation(obs)

        obs_list = list(self.nodes.values())
        n = len(obs_list)

        for i in range(n):
            obs_a = obs_list[i]
            has_id_a = (obs_a.appearance_embedding is not None and len(obs_a.appearance_embedding) > 0) or (obs_a.plate is not None)

            for j in range(i + 1, n):
                obs_b = obs_list[j]
                has_id_b = (obs_b.appearance_embedding is not None and len(obs_b.appearance_embedding) > 0) or (obs_b.plate is not None)

                # Early pruning: if threshold > 0.50 and neither observation has identity features,
                # maximum achievable match probability is 0.50, which cannot form a graph edge.
                if self.min_threshold > 0.50 and not (has_id_a or has_id_b):
                    continue

                # Ensure chronological ordering A -> B for pairwise evaluation
                if obs_a.timestamp_seconds > obs_b.timestamp_seconds:
                    eval_a, eval_b = obs_b, obs_a
                else:
                    eval_a, eval_b = obs_a, obs_b

                match_result = match_observations(eval_a, eval_b, camera_metadata=camera_metadata, config=config)
                prob = float(match_result["same_vehicle_probability"])
                has_id_ev = match_result.get("evidence", {}).get("identity_evidence_available", True)

                # Form edge only if probability meets threshold AND positive identity evidence is present
                if prob >= self.min_threshold and has_id_ev:
                    edge_data = {
                        "source": obs_a.observation_id,
                        "target": obs_b.observation_id,
                        "probability": prob,
                        "evidence": match_result["evidence"],
                        "evidence_ledger": match_result.get("evidence_ledger"),
                        "explanation": match_result["explanation"],
                    }
                    self.edges.append(edge_data)
                    self.adjacency[obs_a.observation_id].append((obs_b.observation_id, prob))
                    self.adjacency[obs_b.observation_id].append((obs_a.observation_id, prob))

    # ---------------------------------------------------------------------------
    # Phase C: Cluster Consistency Validation
    # ---------------------------------------------------------------------------

    def _validate_cluster_consistency(
        self,
        member_obs: List[Observation],
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> ClusterConsistencyResult:
        """
        Validate whether a set of chronologically ordered member observations form a
        physically consistent vehicle identity cluster.

        Rules evaluated:
        1. Vehicle type compatibility across all members.
        2. Temporal ordering: no two members at different cameras with identical timestamps
           unless an appropriate shared time reference confirms simultaneity.
        3. Consecutive observation transitions: physically impossible transitions
           (e.g. impossible speed) are flagged as violations, not silently discarded.
        4. Chronological ordering relies on timestamp_seconds; if comparability is unavailable
           the transition is marked 'unverified', NOT 'violation'.
        5. Missing spatio-temporal evidence is 'insufficient_evidence', NOT 'inconsistent'.

        Args:
            member_obs: Chronologically ordered list of Observation objects.
            camera_metadata: Optional camera metadata for temporal comparability.
            config: Optional configuration dictionary.

        Returns:
            ClusterConsistencyResult
        """
        n = len(member_obs)

        # Singleton: always consistent by definition (no transitions to check)
        if n <= 1:
            return ClusterConsistencyResult(
                is_consistent=True,
                status="singleton",
                consistency_score=None,
                temporal_consistency="unavailable",
                spatial_consistency="unavailable",
                vehicle_type_consistency="unknown",
                violations=[],
                warnings=[],
                contradictory_edges=[],
            )

        violations: List[str] = []
        warnings: List[str] = []
        contradictory_edges: List[Tuple[str, str]] = []
        temporal_states: List[str] = []
        spatial_states: List[str] = []

        # 1. Vehicle type consistency
        types_present = [obs.vehicle_type for obs in member_obs if obs.vehicle_type]
        if len(set(t.strip().lower() for t in types_present)) > 1:
            # Normalize synonyms the same way similarity.py does
            _synonyms = {"auto": "rickshaw", "suv": "car", "sedan": "car", "hatchback": "car", "van": "car"}
            norm_types = set(_synonyms.get(t.strip().lower(), t.strip().lower()) for t in types_present)
            if len(norm_types) > 1:
                violations.append(
                    f"Inconsistent vehicle types in cluster: {sorted(norm_types)}."
                )
                vehicle_type_consistency = "incompatible"
            else:
                vehicle_type_consistency = "compatible"
        elif types_present:
            vehicle_type_consistency = "compatible"
        else:
            vehicle_type_consistency = "unknown"

        # 2. Pairwise consecutive transition analysis
        from .temporal import check_temporal_comparability
        from .spatial import spatial_feasibility

        max_speed_kmh = float((config or {}).get("max_plausible_speed_kmh", 120.0))

        for i in range(n - 1):
            obs_a = member_obs[i]
            obs_b = member_obs[i + 1]
            pair_label = f"({obs_a.observation_id} → {obs_b.observation_id})"

            # Temporal comparability check
            t_comp = check_temporal_comparability(obs_a, obs_b, camera_metadata=camera_metadata)

            if t_comp.get("comparable"):
                delta_t = t_comp["delta_seconds"]
                if delta_t < 0:
                    # Genuinely inverted timestamps that ARE comparable — a real violation
                    violations.append(
                        f"Temporal inversion at transition {pair_label}: "
                        f"delta_t={delta_t:.2f}s with a shared temporal reference."
                    )
                    contradictory_edges.append((obs_a.observation_id, obs_b.observation_id))
                    temporal_states.append("inconsistent")
                elif delta_t == 0 and obs_a.camera_id != obs_b.camera_id:
                    # Simultaneous at confirmed-comparable timestamps but different cameras
                    violations.append(
                        f"Physically impossible simultaneous cross-camera observation at {pair_label}: "
                        f"delta_t=0.0s on different cameras with shared temporal reference."
                    )
                    contradictory_edges.append((obs_a.observation_id, obs_b.observation_id))
                    temporal_states.append("inconsistent")
                else:
                    temporal_states.append("consistent")
            else:
                t_status = t_comp.get("status", "unavailable")
                if t_status in ("impossible_negative_time",):
                    violations.append(
                        f"Temporal inversion at transition {pair_label}: {t_comp.get('reason', '')}."
                    )
                    contradictory_edges.append((obs_a.observation_id, obs_b.observation_id))
                    temporal_states.append("inconsistent")
                else:
                    # Timestamps not comparable (no shared reference): unverified, NOT violation
                    temporal_states.append("unverified")
                    warnings.append(
                        f"Temporal comparability unavailable at transition {pair_label}: "
                        f"status={t_status}. Transition treated as unverified."
                    )
                delta_t = None

            # Spatial feasibility check (only when coordinates are present)
            if (obs_a.latitude is not None and obs_a.longitude is not None and
                    obs_b.latitude is not None and obs_b.longitude is not None):
                s_res = spatial_feasibility(
                    obs_a, obs_b,
                    max_plausible_speed_kmh=max_speed_kmh,
                    camera_metadata=camera_metadata,
                )
                s_status = str(s_res["status"])
                if s_status in ("impossible_speed", "physically_impossible_speed"):
                    violations.append(
                        f"Physically impossible travel speed at transition {pair_label}: "
                        f"{s_res.get('explanation', s_status)}."
                    )
                    contradictory_edges.append((obs_a.observation_id, obs_b.observation_id))
                    spatial_states.append("inconsistent")
                else:
                    spatial_states.append("consistent")
            else:
                spatial_states.append("unavailable")

        # 3. Transitive contradiction check (A–B strong, B–C strong, but A–C contradictory)
        # Check all non-consecutive pairs for explicit edge contradiction in the adjacency
        for i in range(n):
            for j in range(i + 2, n):  # skip consecutive — already checked
                obs_i = member_obs[i]
                obs_j = member_obs[j]
                # Both must be in adjacency
                i_neighbours = {nid: p for nid, p in self.adjacency.get(obs_i.observation_id, [])}
                if obs_j.observation_id not in i_neighbours:
                    # No direct edge — could be transitive only; check for direct contradiction
                    # by looking at ledger of a fresh match result if both have identity evidence
                    has_id_i = (obs_i.appearance_embedding is not None and len(obs_i.appearance_embedding) > 0) or (obs_i.plate is not None)
                    has_id_j = (obs_j.appearance_embedding is not None and len(obs_j.appearance_embedding) > 0) or (obs_j.plate is not None)
                    if has_id_i and has_id_j:
                        # Check if a direct match would be contradictory
                        eval_i, eval_j = (obs_i, obs_j) if obs_i.timestamp_seconds <= obs_j.timestamp_seconds else (obs_j, obs_i)
                        try:
                            mr = match_observations(eval_i, eval_j, camera_metadata=camera_metadata, config=config)
                            ledger = mr.get("evidence_ledger", {})
                            # Plate directly contradictory
                            if ledger.get("plate", {}).get("status") == "contradictory":
                                warnings.append(
                                    f"Transitive contradiction warning: {obs_i.observation_id} and "
                                    f"{obs_j.observation_id} are in the same cluster but have "
                                    f"contradictory plate evidence."
                                )
                                contradictory_edges.append((obs_i.observation_id, obs_j.observation_id))
                        except Exception:
                            pass  # Do not crash cluster validation on a match error

        # 4. Derive aggregate statuses
        if any(s == "inconsistent" for s in temporal_states):
            temporal_consistency = "inconsistent"
        elif all(s == "unavailable" for s in temporal_states):
            temporal_consistency = "unavailable"
        elif any(s == "unverified" for s in temporal_states):
            temporal_consistency = "unverified"
        else:
            temporal_consistency = "consistent"

        if any(s == "inconsistent" for s in spatial_states):
            spatial_consistency = "inconsistent"
        elif all(s == "unavailable" for s in spatial_states):
            spatial_consistency = "unavailable"
        else:
            spatial_consistency = "consistent"

        # 5. Overall status
        has_violations = len(violations) > 0
        all_unverified = temporal_consistency in ("unverified", "unavailable") and spatial_consistency == "unavailable"

        if has_violations:
            overall_status = "inconsistent"
            is_consistent = False
        elif all_unverified:
            overall_status = "insufficient_evidence"
            is_consistent = True  # Cannot assert inconsistency without evidence
        else:
            overall_status = "consistent"
            is_consistent = True

        # 6. Consistency score: fraction of non-violated transitions
        n_transitions = n - 1
        n_violated = len(set(contradictory_edges))  # unique violated pairs
        if n_transitions > 0:
            consistency_score = max(0.0, 1.0 - (n_violated / n_transitions))
        else:
            consistency_score = None

        return ClusterConsistencyResult(
            is_consistent=is_consistent,
            status=overall_status,
            consistency_score=round(consistency_score, 4) if consistency_score is not None else None,
            temporal_consistency=temporal_consistency,
            spatial_consistency=spatial_consistency,
            vehicle_type_consistency=vehicle_type_consistency,
            violations=violations,
            warnings=warnings,
            contradictory_edges=contradictory_edges,
        )

    # ---------------------------------------------------------------------------
    # Candidate Identity Clustering
    # ---------------------------------------------------------------------------

    def get_candidate_identities(
        self,
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Group connected observation nodes into candidate vehicle identity clusters,
        then validate each cluster's internal consistency.

        Singleton observations receive identity_status='unconfirmed_singleton' and
        identity_confidence=None — NOT artificial 1.0 confidence.

        Returns:
            List[Dict[str, Any]]: List of candidate vehicle identities with member observations,
                                   cluster consistency validation, and Day 3 handoff metadata.
        """
        visited: Set[str] = set()
        candidate_clusters: List[Dict[str, Any]] = []
        cluster_idx = 1

        for node_id in self.nodes:
            if node_id not in visited:
                component: Set[str] = set()
                queue = [node_id]
                visited.add(node_id)

                while queue:
                    curr = queue.pop(0)
                    component.add(curr)
                    for neighbor, prob in self.adjacency.get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                # Order member observations chronologically
                member_obs = [self.nodes[nid] for nid in component]
                member_obs.sort(key=lambda x: x.timestamp_seconds)

                is_singleton = len(member_obs) == 1

                # Phase D: Fix singleton semantics
                if is_singleton:
                    identity_status = "unconfirmed_singleton"
                    avg_conf = None        # No cross-camera evidence to support confidence
                    has_app = bool(
                        member_obs[0].appearance_embedding is not None
                        and len(member_obs[0].appearance_embedding) > 0
                    )
                else:
                    identity_status = "candidate"
                    # Calculate cluster identity confidence from internal edge probabilities
                    cluster_probs = []
                    has_app = False
                    for i_m in range(len(member_obs)):
                        if member_obs[i_m].appearance_embedding is not None and len(member_obs[i_m].appearance_embedding) > 0:
                            has_app = True
                        for j_m in range(i_m + 1, len(member_obs)):
                            m_a = member_obs[i_m].observation_id
                            m_b = member_obs[j_m].observation_id
                            for adj_target, adj_prob in self.adjacency.get(m_a, []):
                                if adj_target == m_b:
                                    cluster_probs.append(adj_prob)
                    avg_conf = round(sum(cluster_probs) / len(cluster_probs), 4) if cluster_probs else None

                # Phase C: Cluster consistency validation
                consistency_result = self._validate_cluster_consistency(
                    member_obs, camera_metadata=camera_metadata, config=config
                )

                identity_id = f"VEHICLE_CANDIDATE_{cluster_idx:03d}"
                cluster_entry = {
                    "identity_id": identity_id,
                    "candidate_vehicle_id": identity_id,
                    "identity_status": identity_status,
                    "observation_ids": [obs.observation_id for obs in member_obs],
                    "member_observations": [
                        {
                            "observation_id": obs.observation_id,
                            "camera_id": obs.camera_id,
                            "timestamp_seconds": obs.timestamp_seconds,
                            "timestamp": obs.timestamp_seconds,
                            "latitude": obs.latitude,
                            "longitude": obs.longitude,
                            "vehicle_type": obs.vehicle_type,
                        }
                        for obs in member_obs
                    ],
                    "identity_confidence": avg_conf,
                    "identity_evidence_summary": {
                        "appearance": "available" if has_app else "unavailable",
                        "temporal": "supported",
                        "spatial": "supported",
                        "vehicle_type": "compatible" if consistency_result.vehicle_type_consistency == "compatible" else consistency_result.vehicle_type_consistency,
                    },
                    "cameras_visited": list(dict.fromkeys([obs.camera_id for obs in member_obs])),
                    "start_time": member_obs[0].timestamp.isoformat() if member_obs[0].timestamp else None,
                    "end_time": member_obs[-1].timestamp.isoformat() if member_obs[-1].timestamp else None,
                    "member_observations_count": len(member_obs),
                    "cluster_consistency": consistency_result.to_dict(),
                    "consistency_warning": not consistency_result.is_consistent,
                }
                candidate_clusters.append(cluster_entry)
                cluster_idx += 1

        return candidate_clusters

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph structure to a serializable dictionary representation."""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "threshold": self.min_threshold,
            "nodes": [obs.to_dict() for obs in self.nodes.values()],
            "edges": self.edges,
            "candidate_identities": self.get_candidate_identities(),
        }


