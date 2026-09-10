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
        admission_status: 'confirmed' | 'unconfirmed_singleton' | 'ambiguous' | 'rejected_merge'
        supporting_edges: List of (obs_id_a, obs_id_b) pairs with positive match evidence.
        unavailable_evidence: List of modality names where evidence was unavailable.
        temporal_conflicts: Count of temporal conflict violations.
        topology_conflicts: Count of spatial / topology conflict violations.
        identity_evidence_coverage: Fraction of member observations carrying Re-ID or plate evidence.
        internal_consistency: Heuristic score in [0.0, 1.0].
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
    admission_status: str = "ambiguous"
    supporting_edges: List[Tuple[str, str]] = ()
    unavailable_evidence: List[str] = ()
    temporal_conflicts: int = 0
    topology_conflicts: int = 0
    identity_evidence_coverage: float = 0.0
    internal_consistency: float = 0.0

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
            "admission_status": self.admission_status,
            "supporting_edges": [list(e) for e in self.supporting_edges],
            "unavailable_evidence": list(self.unavailable_evidence),
            "temporal_conflicts": self.temporal_conflicts,
            "topology_conflicts": self.topology_conflicts,
            "identity_evidence_coverage": round(self.identity_evidence_coverage, 4),
            "internal_consistency": round(self.internal_consistency, 4),
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
        self.rejected_merges: List[Dict[str, Any]] = []

    def add_observation(self, obs: Observation) -> None:
        """Add an observation node to the graph."""
        if obs.observation_id not in self.nodes:
            self.nodes[obs.observation_id] = obs
            self.adjacency[obs.observation_id] = []

    def build_graph_reference(
        self,
        observations: List[Observation],
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Reference O(N^2) graph construction without early pruning.
        Used for verification and benchmark comparison against the optimized implementation.
        """
        self.build_graph(observations, camera_metadata=camera_metadata, config=config, enable_pruning=False)

    def build_graph(
        self,
        observations: List[Observation],
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        enable_pruning: bool = True,
    ) -> None:
        """
        Build pairwise edges across observations using the Identity Fusion Engine.
        Deterministic and input-order invariant.

        Args:
            observations: List of vehicle observations to add and compare.
            camera_metadata: Optional camera metadata mapping.
            config: Optional fusion engine configuration settings.
            enable_pruning: If True, applies safe deterministic pruning rules that cannot remove valid matches.
        """
        config = config or {}
        # Deterministic sorting of input observations by timestamp then ID
        sorted_obs_input = sorted(observations, key=lambda o: (o.timestamp_seconds, o.observation_id))
        for obs in sorted_obs_input:
            self.add_observation(obs)

        # Ensure deterministic node iteration
        obs_list = sorted(self.nodes.values(), key=lambda o: (o.timestamp_seconds, o.observation_id))
        n = len(obs_list)

        max_speed_kmh = float(config.get("max_plausible_speed_kmh", 120.0))

        for i in range(n):
            obs_a = obs_list[i]
            has_id_a = (obs_a.appearance_embedding is not None and len(obs_a.appearance_embedding) > 0) or (obs_a.plate is not None)

            for j in range(i + 1, n):
                obs_b = obs_list[j]
                has_id_b = (obs_b.appearance_embedding is not None and len(obs_b.appearance_embedding) > 0) or (obs_b.plate is not None)

                # Safe deterministic pruning: only prune when mathematically/physically impossible to match
                if enable_pruning:
                    # 1. Missing identity evidence on both observations when threshold > 0.50
                    if self.min_threshold > 0.50 and not (has_id_a or has_id_b):
                        continue

                    # 2. Incompatible vehicle types (when both types are known, non-empty, and incompatible)
                    if obs_a.vehicle_type and obs_b.vehicle_type:
                        from .similarity import vehicle_type_compatibility
                        _, v_stat = vehicle_type_compatibility(obs_a.vehicle_type, obs_b.vehicle_type)
                        if v_stat == "incompatible":
                            continue

                    # 3. Physically impossible speed over known coordinates
                    if (
                        obs_a.latitude is not None and obs_a.longitude is not None
                        and obs_b.latitude is not None and obs_b.longitude is not None
                    ):
                        from .temporal import check_temporal_comparability
                        t_check = check_temporal_comparability(obs_a, obs_b, camera_metadata=camera_metadata)
                        if t_check.get("comparable"):
                            dt = abs(t_check.get("delta_seconds", 0.0))
                            if dt > 0:
                                from .similarity import geographic_distance
                                dist_m = geographic_distance(obs_a.latitude, obs_a.longitude, obs_b.latitude, obs_b.longitude)
                                speed_kmh = (dist_m / dt) * 3.6
                                if speed_kmh > max_speed_kmh:
                                    continue
                            elif obs_a.camera_id != obs_b.camera_id:
                                # Simultaneous on different cameras with shared clock
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

        # Deterministic sorting of adjacency lists: decreasing by probability, then neighbor ID
        for nid in self.adjacency:
            self.adjacency[nid].sort(key=lambda item: (-item[1], item[0]))

    # ---------------------------------------------------------------------------
    # Phase C: Cluster Consistency Validation & Admission
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

        Evaluates:
        1. Supporting edges vs contradictory edges.
        2. Vehicle type compatibility.
        3. Temporal ordering and comparability.
        4. Spatial feasibility and speed limits.
        5. Plate compatibility (detects explicit plate contradictions).
        6. Missing vs unavailable vs invalid evidence.
        7. Cluster admission status: 'confirmed' | 'unconfirmed_singleton' | 'ambiguous' | 'rejected_merge'.
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
                admission_status="unconfirmed_singleton",
                supporting_edges=[],
                unavailable_evidence=["cross_camera_transition"],
                temporal_conflicts=0,
                topology_conflicts=0,
                identity_evidence_coverage=1.0 if n == 1 and (member_obs[0].appearance_embedding is not None or member_obs[0].plate is not None) else 0.0,
                internal_consistency=1.0,
            )

        violations: List[str] = []
        warnings: List[str] = []
        contradictory_edges: List[Tuple[str, str]] = []
        supporting_edges: List[Tuple[str, str]] = []
        temporal_states: List[str] = []
        spatial_states: List[str] = []
        unavailable_evidence: List[str] = []

        # 1. Vehicle type consistency
        types_present = [obs.vehicle_type for obs in member_obs if obs.vehicle_type]
        if len(set(t.strip().lower() for t in types_present)) > 1:
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
            unavailable_evidence.append("vehicle_type")

        # 2. Pairwise transition analysis across all consecutive pairs
        from .temporal import check_temporal_comparability
        from .spatial import spatial_feasibility

        max_speed_kmh = float((config or {}).get("max_plausible_speed_kmh", 120.0))

        for i in range(n - 1):
            obs_a = member_obs[i]
            obs_b = member_obs[i + 1]
            pair_label = f"({obs_a.observation_id} → {obs_b.observation_id})"
            pair_key = (min(obs_a.observation_id, obs_b.observation_id), max(obs_a.observation_id, obs_b.observation_id))

            # Temporal comparability check
            t_comp = check_temporal_comparability(obs_a, obs_b, camera_metadata=camera_metadata)

            if t_comp.get("comparable"):
                delta_t = t_comp["delta_seconds"]
                if delta_t < 0:
                    violations.append(
                        f"Temporal inversion at transition {pair_label}: "
                        f"delta_t={delta_t:.2f}s with a shared temporal reference."
                    )
                    contradictory_edges.append(pair_key)
                    temporal_states.append("inconsistent")
                elif delta_t == 0 and obs_a.camera_id != obs_b.camera_id:
                    violations.append(
                        f"Physically impossible simultaneous cross-camera observation at {pair_label}: "
                        f"delta_t=0.0s on different cameras with shared temporal reference."
                    )
                    contradictory_edges.append(pair_key)
                    temporal_states.append("inconsistent")
                else:
                    temporal_states.append("consistent")
            else:
                t_status = t_comp.get("status", "unavailable")
                if t_status in ("impossible_negative_time",):
                    violations.append(
                        f"Temporal inversion at transition {pair_label}: {t_comp.get('reason', '')}."
                    )
                    contradictory_edges.append(pair_key)
                    temporal_states.append("inconsistent")
                else:
                    temporal_states.append("unverified")
                    warnings.append(
                        f"Temporal comparability unavailable at transition {pair_label}: "
                        f"status={t_status}. Transition treated as unverified."
                    )
                    unavailable_evidence.append(f"temporal_{pair_label}")

            # Spatial feasibility check
            if (
                obs_a.latitude is not None and obs_a.longitude is not None
                and obs_b.latitude is not None and obs_b.longitude is not None
            ):
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
                    contradictory_edges.append(pair_key)
                    spatial_states.append("inconsistent")
                else:
                    spatial_states.append("consistent")
            else:
                spatial_states.append("unavailable")
                unavailable_evidence.append(f"spatial_{pair_label}")

        # 3. All-pairs identity contradiction check (plate, speed, vehicle type across non-consecutive pairs)
        for i in range(n):
            for j in range(i + 1, n):
                obs_i = member_obs[i]
                obs_j = member_obs[j]
                pair_key = (min(obs_i.observation_id, obs_j.observation_id), max(obs_i.observation_id, obs_j.observation_id))

                # Check if this pair has a supporting graph edge
                i_neighbours = dict(self.adjacency.get(obs_i.observation_id, []))
                if obs_j.observation_id in i_neighbours:
                    supporting_edges.append(pair_key)

                # Plate contradiction check: both plates present, valid, length >= 4, and clearly mismatching
                if obs_i.plate is not None and obs_j.plate is not None:
                    from .similarity import plate_similarity
                    p_sim = plate_similarity(obs_i.plate, obs_j.plate)
                    clean_i = "".join(c for c in str(obs_i.plate).upper() if c.isalnum())
                    clean_j = "".join(c for c in str(obs_j.plate).upper() if c.isalnum())
                    if len(clean_i) >= 4 and len(clean_j) >= 4 and p_sim < 0.30:
                        warnings.append(
                            f"Plate contradiction between {obs_i.observation_id} ('{obs_i.plate}') and "
                            f"{obs_j.observation_id} ('{obs_j.plate}') in same cluster."
                        )
                        contradictory_edges.append(pair_key)

                # Physical speed contradiction across all pairs
                if (
                    obs_i.latitude is not None and obs_i.longitude is not None
                    and obs_j.latitude is not None and obs_j.longitude is not None
                ):
                    t_ij = check_temporal_comparability(obs_i, obs_j, camera_metadata=camera_metadata)
                    if t_ij.get("comparable") and t_ij.get("delta_seconds", 0) > 0:
                        dt_ij = t_ij["delta_seconds"]
                        from .similarity import geographic_distance
                        d_ij = geographic_distance(obs_i.latitude, obs_i.longitude, obs_j.latitude, obs_j.longitude)
                        sp_ij = (d_ij / dt_ij) * 3.6
                        if sp_ij > max_speed_kmh:
                            violations.append(
                                f"Physically impossible speed between {obs_i.observation_id} and {obs_j.observation_id}: "
                                f"{sp_ij:.1f} km/h exceeds limit ({max_speed_kmh:.1f} km/h)."
                            )
                            contradictory_edges.append(pair_key)

        # De-duplicate edges
        contradictory_edges = sorted(list(set(contradictory_edges)))
        supporting_edges = sorted(list(set(supporting_edges)))

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

        # 5. Overall status and admission
        has_violations = len(violations) > 0 or len(contradictory_edges) > 0
        all_unverified = temporal_consistency in ("unverified", "unavailable") and spatial_consistency == "unavailable"

        has_id_count = sum(
            1 for o in member_obs
            if (o.appearance_embedding is not None and len(o.appearance_embedding) > 0) or o.plate is not None
        )
        coverage = has_id_count / float(n) if n > 0 else 0.0

        if has_violations:
            overall_status = "inconsistent"
            is_consistent = False
            admission_status = "rejected_merge"
        elif all_unverified:
            overall_status = "insufficient_evidence"
            is_consistent = True
            admission_status = "ambiguous"
        else:
            overall_status = "consistent"
            is_consistent = True
            admission_status = "confirmed" if coverage >= 0.50 else "ambiguous"

        # 6. Consistency score
        n_transitions = n - 1
        n_violated = len(contradictory_edges)
        if n_transitions > 0:
            consistency_score = max(0.0, 1.0 - (n_violated / n_transitions))
        else:
            consistency_score = None

        internal_consistency = consistency_score if consistency_score is not None else 1.0

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
            admission_status=admission_status,
            supporting_edges=supporting_edges,
            unavailable_evidence=list(dict.fromkeys(unavailable_evidence)),
            temporal_conflicts=sum(1 for s in temporal_states if s == "inconsistent"),
            topology_conflicts=sum(1 for s in spatial_states if s == "inconsistent"),
            identity_evidence_coverage=round(coverage, 4),
            internal_consistency=round(internal_consistency, 4),
        )

    # ---------------------------------------------------------------------------
    # Phase 7: Contradiction-Aware Cluster Splitting
    # ---------------------------------------------------------------------------

    def _split_contradictory_cluster(
        self,
        member_obs: List[Observation],
        contradictory_pairs: Set[Tuple[str, str]],
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[List[Observation]]:
        """
        Split a candidate cluster containing contradictions into clean, non-contradictory sub-clusters.
        Uses a deterministic Kruskal-style constrained maximum-weight spanning forest:
        candidate edges are considered in descending order of evidence probability, and a merge
        is rejected if it would combine two components containing an irreconcilable contradiction.
        """
        member_ids = [obs.observation_id for obs in member_obs]
        obs_map = {obs.observation_id: obs for obs in member_obs}

        # Parent pointer for disjoint-set
        parent: Dict[str, str] = {nid: nid for nid in member_ids}
        components: Dict[str, Set[str]] = {nid: {nid} for nid in member_ids}

        def find(u: str) -> str:
            while parent[u] != u:
                parent[u] = parent[parent[u]]
                u = parent[u]
            return u

        # Gather all candidate edges strictly within this cluster
        internal_edges = []
        for e in self.edges:
            s, t = e["source"], e["target"]
            if s in obs_map and t in obs_map:
                pair_key = (min(s, t), max(s, t))
                if pair_key not in contradictory_pairs:
                    internal_edges.append((float(e["probability"]), pair_key[0], pair_key[1]))

        # Sort edges descending by weight, breaking ties deterministically by node IDs
        internal_edges.sort(key=lambda item: (-item[0], item[1], item[2]))

        # Greedily merge components if no contradiction exists between them
        for prob, u, v in internal_edges:
            root_u = find(u)
            root_v = find(v)
            if root_u == root_v:
                continue

            comp_u = components[root_u]
            comp_v = components[root_v]

            # Check if any pair across comp_u and comp_v is contradictory
            has_conflict = False
            for x in comp_u:
                for y in comp_v:
                    key = (min(x, y), max(x, y))
                    if key in contradictory_pairs:
                        has_conflict = True
                        break
                if has_conflict:
                    break

            if not has_conflict:
                # Merge root_v into root_u
                parent[root_v] = root_u
                components[root_u] = comp_u | comp_v
                del components[root_v]
            else:
                self.rejected_merges.append({
                    "source": u,
                    "target": v,
                    "probability": prob,
                    "reason": "Cluster-level contradiction between component members",
                })

        # Assemble resulting sub-clusters sorted deterministically
        sub_clusters: List[List[Observation]] = []
        for root in sorted(components.keys()):
            sub_members = [obs_map[nid] for nid in sorted(components[root])]
            sub_members.sort(key=lambda o: (o.timestamp_seconds, o.observation_id))
            sub_clusters.append(sub_members)

        return sub_clusters

    # ---------------------------------------------------------------------------
    # Candidate Identity Clustering & Final Hypotheses
    # ---------------------------------------------------------------------------

    def get_candidate_identities(
        self,
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
        resolve_contradictions: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Group connected observation nodes into candidate vehicle identity clusters.

        If resolve_contradictions=True, candidate clusters with hard contradictions are
        split deterministically into non-contradictory final identity hypotheses.

        Singleton observations receive identity_status='unconfirmed_singleton' and
        identity_confidence=None — NOT artificial 1.0 confidence.

        Returns:
            List[Dict[str, Any]]: List of candidate or final vehicle identities.
        """
        visited: Set[str] = set()
        raw_components: List[List[Observation]] = []

        # Deterministic node iteration
        sorted_node_ids = sorted(self.nodes.keys(), key=lambda nid: (self.nodes[nid].timestamp_seconds, nid))

        for node_id in sorted_node_ids:
            if node_id not in visited:
                component: Set[str] = set()
                queue = [node_id]
                visited.add(node_id)

                while queue:
                    curr = queue.pop(0)
                    component.add(curr)
                    # Iterate neighbors in deterministic order
                    for neighbor, prob in self.adjacency.get(curr, []):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

                # Order member observations chronologically then by ID
                member_obs = [self.nodes[nid] for nid in component]
                member_obs.sort(key=lambda x: (x.timestamp_seconds, x.observation_id))
                raw_components.append(member_obs)

        # Process components into candidate clusters or split if requested
        final_groups: List[List[Observation]] = []
        for member_obs in raw_components:
            consistency_check = self._validate_cluster_consistency(
                member_obs, camera_metadata=camera_metadata, config=config
            )
            if resolve_contradictions and consistency_check.admission_status == "rejected_merge":
                contra_set = set(consistency_check.contradictory_edges)
                split_groups = self._split_contradictory_cluster(
                    member_obs, contra_set, camera_metadata=camera_metadata, config=config
                )
                final_groups.extend(split_groups)
            else:
                final_groups.append(member_obs)

        # Sort candidate clusters deterministically
        final_groups.sort(key=lambda group: (group[0].timestamp_seconds, group[0].observation_id))

        candidate_clusters: List[Dict[str, Any]] = []
        cluster_idx = 1

        for member_obs in final_groups:
            is_singleton = len(member_obs) == 1
            consistency_result = self._validate_cluster_consistency(
                member_obs, camera_metadata=camera_metadata, config=config
            )

            # Phase 8: Stable singleton semantics
            if is_singleton:
                identity_status = "unconfirmed_singleton"
                avg_conf = None
                admission_status = "unconfirmed_singleton"
            else:
                identity_status = "candidate"
                cluster_probs = []
                for i_m in range(len(member_obs)):
                    for j_m in range(i_m + 1, len(member_obs)):
                        m_a = member_obs[i_m].observation_id
                        m_b = member_obs[j_m].observation_id
                        for adj_target, adj_prob in self.adjacency.get(m_a, []):
                            if adj_target == m_b:
                                cluster_probs.append(adj_prob)
                avg_conf = round(sum(cluster_probs) / len(cluster_probs), 4) if cluster_probs else None
                admission_status = consistency_result.admission_status

            # Phase 5: Grounded identity evidence summary (no hardcoded claims)
            has_app = any(o.appearance_embedding is not None and len(o.appearance_embedding) > 0 for o in member_obs)
            invalid_app = any(
                o.appearance_embedding is not None and not isinstance(o.appearance_embedding, (list, tuple))
                for o in member_obs
            )

            if invalid_app:
                app_summary = "invalid"
            elif not has_app:
                app_summary = "missing"
            elif is_singleton:
                app_summary = "available"
            else:
                app_summary = "available_supportive"

            has_plate = any(o.plate is not None for o in member_obs)
            if not has_plate:
                plate_summary = "missing"
            elif is_singleton:
                plate_summary = "available"
            elif any(
                edge in consistency_result.contradictory_edges
                for edge in consistency_result.contradictory_edges
                if "Plate" in str(consistency_result.warnings)
            ):
                plate_summary = "available_contradictory"
            else:
                plate_summary = "available_supportive"

            if is_singleton:
                temp_summary = "unavailable"
                spat_summary = "unavailable"
            elif consistency_result.temporal_consistency == "inconsistent":
                temp_summary = "available_contradictory"
            elif consistency_result.temporal_consistency == "consistent":
                temp_summary = "available_supportive"
            else:
                temp_summary = "unavailable"

            if is_singleton:
                spat_summary = "unavailable"
            elif consistency_result.spatial_consistency == "inconsistent":
                spat_summary = "available_contradictory"
            elif consistency_result.spatial_consistency == "consistent":
                spat_summary = "available_supportive"
            else:
                spat_summary = "unavailable"

            if consistency_result.vehicle_type_consistency == "compatible":
                type_summary = "available_supportive" if not is_singleton else "available"
            elif consistency_result.vehicle_type_consistency == "incompatible":
                type_summary = "available_contradictory"
            else:
                type_summary = "unavailable"

            identity_evidence_summary = {
                "appearance": app_summary,
                "plate": plate_summary,
                "temporal": temp_summary,
                "spatial": spat_summary,
                "vehicle_type": type_summary,
            }

            identity_id = f"VEHICLE_CANDIDATE_{cluster_idx:03d}"
            cluster_entry = {
                "identity_id": identity_id,
                "candidate_vehicle_id": identity_id,
                "identity_status": identity_status,
                "admission_status": admission_status,
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
                "identity_evidence_summary": identity_evidence_summary,
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

    def get_final_identity_hypotheses(
        self,
        camera_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate contradiction-aware final identity hypotheses.
        Splits contradictory candidate clusters so that no final identity hypothesis
        contains an irreconcilable physical or identity contradiction.
        """
        return self.get_candidate_identities(
            camera_metadata=camera_metadata, config=config, resolve_contradictions=True
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert graph structure to a serializable dictionary representation."""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "threshold": self.min_threshold,
            "nodes": [obs.to_dict() for obs in self.nodes.values()],
            "edges": self.edges,
            "candidate_identities": self.get_candidate_identities(),
            "final_identity_hypotheses": self.get_final_identity_hypotheses(),
        }


