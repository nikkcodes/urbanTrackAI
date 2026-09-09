"""
Identity Graph representation and candidate vehicle identity clustering for UrbanTrack AI.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

from schemas.observation_schema import Observation
from .identity_fusion import match_observations


class IdentityGraph:
    """
    Graph representation of vehicle observations and identity match probabilities.

    Nodes: Vehicle observations.
    Edges: Estimated probability that two observations represent the same physical vehicle.
    """

    def __init__(self, min_probability_threshold: float = 0.70) -> None:
        """
        Initialize the IdentityGraph.

        Args:
            min_probability_threshold: Minimum match probability to form an edge between observations (default 0.70).
                                       Requires strong identity match evidence to avoid transitive over-clustering.
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
                        "explanation": match_result["explanation"],
                    }
                    self.edges.append(edge_data)
                    self.adjacency[obs_a.observation_id].append((obs_b.observation_id, prob))
                    self.adjacency[obs_b.observation_id].append((obs_a.observation_id, prob))

    def get_candidate_identities(self) -> List[Dict[str, Any]]:
        """
        Group connected observation nodes into candidate vehicle identity clusters.

        Returns:
            List[Dict[str, Any]]: List of candidate vehicle identities with member observations
                                   and Task 11 Day 2 Output Contract metadata for Day 3 handoff.
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

                avg_conf = round(sum(cluster_probs) / len(cluster_probs), 4) if cluster_probs else 1.0000

                identity_id = f"VEHICLE_CANDIDATE_{cluster_idx:03d}"
                candidate_clusters.append({
                    "identity_id": identity_id,
                    "candidate_vehicle_id": identity_id,
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
                        "vehicle_type": "compatible",
                    },
                    "cameras_visited": list(dict.fromkeys([obs.camera_id for obs in member_obs])),
                    "start_time": member_obs[0].timestamp.isoformat() if member_obs[0].timestamp else None,
                    "end_time": member_obs[-1].timestamp.isoformat() if member_obs[-1].timestamp else None,
                    "member_observations_count": len(member_obs),
                })
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
