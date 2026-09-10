"""
Network Analytics and Structural Importance for UrbanTrack AI (Day 6).

Implements network topology metrics:
1. In-degree, out-degree, total degree for junctions and roads.
2. Pure-Python Brandes directed edge betweenness centrality (zero external dependencies).
3. Traffic priority road ranking combining capacity utilization, expected demand,
   and structural importance.

Critical Semantic Note:
Centrality metrics represent purely structural connectivity, labeled as
'network structural importance', separate from actual traffic flow.
"""

from __future__ import annotations

from collections import defaultdict, deque
import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

from inference.road_graph import RoadGraph, RoadEdge
from schemas.mobility_schema import RoadCentralityMetric, RoadFlowMetric, RoadPriority


class NetworkCentralityAnalyzer:
    """
    Computes structural topology metrics on a directed RoadGraph:
    degree metrics and Brandes edge betweenness centrality.
    """

    def __init__(self, road_graph: RoadGraph) -> None:
        if not isinstance(road_graph, RoadGraph):
            raise TypeError(f"Expected RoadGraph, got: {type(road_graph)}")
        self.graph = road_graph

    def compute_degree_metrics(self) -> Dict[str, Tuple[int, int, int]]:
        """
        Compute in-degree, out-degree, and total degree for all junctions in the graph.

        Returns:
            Dict mapping junction_node_id -> (in_degree, out_degree, total_degree)
        """
        in_degree: Dict[str, int] = defaultdict(int)
        out_degree: Dict[str, int] = defaultdict(int)

        # Initialize all nodes
        for nid in self.graph.nodes:
            in_degree[nid] = 0
            out_degree[nid] = 0

        for edge in self.graph.edges.values():
            if edge.is_closed:
                continue
            out_degree[edge.from_node] += 1
            in_degree[edge.to_node] += 1

        degree_map: Dict[str, Tuple[int, int, int]] = {}
        for nid in self.graph.nodes:
            ind = in_degree[nid]
            outd = out_degree[nid]
            degree_map[nid] = (ind, outd, ind + outd)

        return degree_map

    def compute_edge_betweenness(self, normalized: bool = True) -> Dict[str, float]:
        """
        Compute directed edge betweenness centrality using the Brandes algorithm.

        Operates in O(V * E) time with zero external dependencies.
        For a 14-node, 28-edge graph this completes in under 1 millisecond.

        Args:
            normalized: If True, normalizes scores by 1 / (N * (N - 1)) where N = |V|.

        Returns:
            Dict mapping road_id -> betweenness_centrality (float)
        """
        nodes = list(self.graph.nodes.keys())
        n = len(nodes)
        edge_betweenness: Dict[str, float] = {rid: 0.0 for rid in self.graph.edges}

        if n <= 1:
            return edge_betweenness

        # Run BFS from each source node s
        for s in nodes:
            stack: List[str] = []
            # predecessors: v -> list of (u, road_id)
            pred: Dict[str, List[Tuple[str, str]]] = {v: [] for v in nodes}
            sigma: Dict[str, int] = {v: 0 for v in nodes}
            sigma[s] = 1
            dist: Dict[str, int] = {v: -1 for v in nodes}
            dist[s] = 0

            queue: deque[str] = deque([s])

            while queue:
                v = queue.popleft()
                stack.append(v)

                for w, rid, _, edge_obj in self.graph.adjacency.get(v, []):
                    if edge_obj.is_closed:
                        continue

                    # Path discovery
                    if dist[w] < 0:
                        dist[w] = dist[v] + 1
                        queue.append(w)

                    # Path counting
                    if dist[w] == dist[v] + 1:
                        sigma[w] += sigma[v]
                        pred[w].append((v, rid))

            # Accumulation phase
            delta: Dict[str, float] = {v: 0.0 for v in nodes}
            while stack:
                w = stack.pop()
                for v, rid in pred[w]:
                    if sigma[w] > 0:
                        c = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                        edge_betweenness[rid] += c
                        delta[v] += c

        # Normalization for directed graph: 1 / (N * (N - 1))
        if normalized and n > 1:
            scale = 1.0 / (n * (n - 1))
            for rid in edge_betweenness:
                edge_betweenness[rid] *= scale

        return edge_betweenness

    def analyze_network(self) -> Dict[str, RoadCentralityMetric]:
        """
        Analyze structural centrality for all road segments in the network.

        Returns:
            Dict mapping road_id -> RoadCentralityMetric
        """
        node_degrees = self.compute_degree_metrics()
        edge_cb = self.compute_edge_betweenness(normalized=True)

        centrality_metrics: Dict[str, RoadCentralityMetric] = {}
        for rid, edge in self.graph.edges.items():
            u, v = edge.from_node, edge.to_node
            u_in, u_out, _ = node_degrees.get(u, (0, 0, 0))
            v_in, v_out, _ = node_degrees.get(v, (0, 0, 0))

            # Segment degrees: from_node out_degree, to_node in_degree, combined
            in_deg = v_in
            out_deg = u_out
            tot_deg = in_deg + out_deg

            cb = edge_cb.get(rid, 0.0)

            centrality_metrics[rid] = RoadCentralityMetric(
                road_id=rid,
                from_node=u,
                to_node=v,
                in_degree=in_deg,
                out_degree=out_deg,
                total_degree=tot_deg,
                betweenness_centrality=cb,
                metric_type="network_structural_importance",
            )

        return centrality_metrics


class PriorityRoadRanker:
    """
    Ranks city road segments by traffic priority using transparent operational metrics:
    utilization ratio, expected demand, and network structural importance.
    """

    def __init__(
        self,
        weight_utilization: float = 0.60,
        weight_demand: float = 0.30,
        weight_centrality: float = 0.10,
    ) -> None:
        """
        Initialize priority ranker with documented scoring weights.

        Default formula:
            priority_score = 0.60 * utilization + 0.30 * norm_demand + 0.10 * centrality
        """
        self.w_util = weight_utilization
        self.w_demand = weight_demand
        self.w_cent = weight_centrality

    def rank_roads(
        self,
        road_metrics: Iterable[RoadFlowMetric],
        centrality_metrics: Optional[Dict[str, RoadCentralityMetric]] = None,
        limit: Optional[int] = None,
    ) -> List[RoadPriority]:
        """
        Rank road segments by priority score descending.

        Handles both dimensionally calibrated hourly rates and window-only demand:
        - If hourly rate is valid: combines utilization ratio, normalized hourly demand, and centrality.
        - If uncalibrated window: combines normalized window demand and centrality without false hourly utilization.

        Args:
            road_metrics: Collection of RoadFlowMetric objects.
            centrality_metrics: Optional mapping of road_id -> RoadCentralityMetric.
            limit: Maximum number of priority roads to return (default: all).

        Returns:
            List of RoadPriority instances sorted from highest to lowest priority.
        """
        metric_list = list(road_metrics)
        if not metric_list:
            return []

        # Check if batch has valid hourly rates
        is_hourly_valid = any(m.is_hourly_rate_valid for m in metric_list)

        if is_hourly_valid:
            max_hourly = max((m.expected_demand_vph or 0.0 for m in metric_list), default=0.0)
            norm_hourly_scale = max_hourly if max_hourly > 0.0 else 1.0

            candidates: List[Tuple[float, Optional[float], float, RoadFlowMetric, str]] = []
            for m in metric_list:
                cent_val = 0.0
                if centrality_metrics and m.road_id in centrality_metrics:
                    cent_val = centrality_metrics[m.road_id].betweenness_centrality

                vph = m.expected_demand_vph or 0.0
                norm_vph = vph / norm_hourly_scale
                util = m.utilization_ratio if m.utilization_ratio is not None else 0.0

                score = (
                    self.w_util * util
                    + self.w_demand * norm_vph
                    + self.w_cent * cent_val
                )
                formula_desc = (
                    f"{self.w_util:.2f}*util({util:.4f}) + "
                    f"{self.w_demand:.2f}*norm_vph({norm_vph:.4f}) + "
                    f"{self.w_cent:.2f}*centrality"
                )
                candidates.append((score, util, vph, m, formula_desc))

            candidates.sort(key=lambda x: (-x[0], -(x[1] or 0.0), -x[2], x[3].road_id))
        else:
            # Window-only mode: avoid multiplying or comparing uncalibrated window counts with hourly capacity
            max_window_demand = max((m.expected_demand_in_window for m in metric_list), default=0.0)
            norm_window_scale = max_window_demand if max_window_demand > 0.0 else 1.0

            candidates = []
            for m in metric_list:
                cent_val = 0.0
                if centrality_metrics and m.road_id in centrality_metrics:
                    cent_val = centrality_metrics[m.road_id].betweenness_centrality

                dem = m.expected_demand_in_window
                norm_dem = dem / norm_window_scale

                # Window-only priority score: 80% normalized window demand + 20% centrality
                score = 0.80 * norm_dem + 0.20 * cent_val
                formula_desc = (
                    f"priority based on inferred demand and structural importance (window-only): "
                    f"0.80*norm_window_demand({norm_dem:.4f}) + 0.20*centrality"
                )
                candidates.append((score, None, dem, m, formula_desc))

            candidates.sort(key=lambda x: (-x[0], -x[2], x[3].road_id))

        ranked_priorities: List[RoadPriority] = []
        for rank_idx, (score, util, _, m, formula_desc) in enumerate(candidates, start=1):
            ranked_priorities.append(RoadPriority(
                road_id=m.road_id,
                priority_score=score,
                rank=rank_idx,
                expected_demand_in_window=m.expected_demand_in_window,
                capacity_vph=m.capacity_vph,
                expected_demand_vph=m.expected_demand_vph,
                utilization_ratio=m.utilization_ratio,
                is_hourly_rate_valid=m.is_hourly_rate_valid,
                formula=formula_desc,
            ))

        return ranked_priorities if limit is None else ranked_priorities[:limit]
