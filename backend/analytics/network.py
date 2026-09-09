"""Network flow concentration, topology, and mobility-priority analytics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from backend.analytics.bottlenecks import BottleneckDetector
from backend.analytics.models import (
    Bottleneck,
    NetworkIntelligence,
    ODAnalysisResult,
    RoadPriority,
    RouteDemand,
    UrbanMobilityAnalysisResult,
)
from backend.analytics.od import ODAnalyzer
from backend.flow.models import FlowAggregationResult
from backend.mobility.graph import MobilityGraph
from backend.traffic.models import TrafficMetric


@dataclass(frozen=True, slots=True)
class PriorityWeights:
    """Configurable additive weights for descriptive road prioritization."""

    flow_share: float = 1.0
    utilization: float = 1.0
    congestion: float = 1.0
    delay_ratio: float = 1.0
    structural_importance: float = 1.0

    def __post_init__(self) -> None:
        values = (self.flow_share, self.utilization, self.congestion, self.delay_ratio, self.structural_importance)
        if any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("priority weights must be finite and non-negative")
        if sum(values) <= 0.0:
            raise ValueError("at least one priority weight must be positive")


class NetworkIntelligenceAnalyzer:
    """Compute hourly-flow concentration and descriptive graph importance."""

    def analyze(self, metrics: Iterable[TrafficMetric], graph: MobilityGraph) -> NetworkIntelligence:
        """Calculate HHI, active road shares, and normalized edge centrality."""
        if not isinstance(graph, MobilityGraph):
            raise TypeError(f"Expected MobilityGraph, got: {type(graph)}")
        metric_list = self._validate_metrics(metrics)
        total_hourly_flow = sum(metric.hourly_flow for metric in metric_list if metric.hourly_flow is not None)
        shares = {
            metric.road_id: (metric.hourly_flow / total_hourly_flow if total_hourly_flow > 0.0 else 0.0)
            for metric in metric_list
            if metric.hourly_flow is not None and metric.hourly_flow > 0.0
        }
        hhi = sum(share * share for share in shares.values()) if total_hourly_flow > 0.0 else 0.0
        edge_centrality = nx.edge_betweenness_centrality(graph.nx_graph, normalized=True)
        structural_importance: Dict[str, float] = {}
        for road in graph.all_roads(include_closed=False):
            structural_importance[road.road_id] = float(edge_centrality.get((road.from_node, road.to_node), 0.0))

        return NetworkIntelligence(
            total_hourly_flow=total_hourly_flow,
            flow_concentration_hhi=hhi,
            road_flow_shares=shares,
            structural_importance=structural_importance,
            road_count=graph.road_count,
            active_roads_with_flow=sum(1 for metric in metric_list if metric.hourly_flow and metric.hourly_flow > 0.0),
            evaluated_roads_count=len(metric_list),
        )

    def rank_priority_roads(
        self,
        metrics: Iterable[TrafficMetric],
        network: NetworkIntelligence,
        weights: Optional[PriorityWeights] = None,
        limit: Optional[int] = None,
    ) -> Tuple[RoadPriority, ...]:
        """Rank roads while retaining each underlying signal beside the score.

        Formula:
            flow_share*w1 + utilization*w2 + congestion*w3
            + delay_ratio*w4 + structural_importance*w5
        """
        if not isinstance(network, NetworkIntelligence):
            raise TypeError(f"Expected NetworkIntelligence, got: {type(network)}")
        if limit is not None and (not isinstance(limit, int) or limit < 0):
            raise ValueError("limit must be a non-negative integer or None")
        weights = weights or PriorityWeights()
        total_hourly_flow = network.total_hourly_flow
        priorities: List[RoadPriority] = []
        for metric in self._validate_metrics(metrics):
            if metric.hourly_flow is None:
                raise ValueError(f"TrafficMetric '{metric.road_id}' must provide hourly_flow")
            flow_share = network.road_flow_shares.get(metric.road_id, 0.0)
            delay_ratio = max(0.0, metric.estimated_travel_time_minutes - metric.free_flow_time_minutes) / metric.free_flow_time_minutes
            structural = network.structural_importance.get(metric.road_id, 0.0)
            score = (
                weights.flow_share * flow_share
                + weights.utilization * metric.utilization_ratio
                + weights.congestion * metric.congestion_score
                + weights.delay_ratio * delay_ratio
                + weights.structural_importance * structural
            )
            priorities.append(
                RoadPriority(
                    road_id=metric.road_id,
                    priority_score=score,
                    hourly_flow=metric.hourly_flow,
                    flow_share=flow_share,
                    utilization_ratio=metric.utilization_ratio,
                    congestion_score=metric.congestion_score,
                    delay_ratio=delay_ratio,
                    structural_importance=structural,
                )
            )
        priorities.sort(key=lambda item: (-item.priority_score, item.road_id))
        return tuple(priorities if limit is None else priorities[:limit])

    @staticmethod
    def _validate_metrics(metrics: Iterable[TrafficMetric]) -> List[TrafficMetric]:
        records = list(metrics)
        if any(not isinstance(metric, TrafficMetric) for metric in records):
            raise TypeError("metrics must contain only TrafficMetric instances")
        for metric in records:
            if metric.hourly_flow is None or not math.isfinite(metric.hourly_flow) or metric.hourly_flow < 0.0:
                raise ValueError(f"TrafficMetric '{metric.road_id}' must provide a valid hourly_flow")
        return records


class UrbanMobilityAnalyzer:
    """Compose all Phase 4 analytics without adding a UI or service dependency."""

    def __init__(
        self,
        bottleneck_detector: Optional[BottleneckDetector] = None,
        priority_weights: Optional[PriorityWeights] = None,
    ) -> None:
        self.od_analyzer = ODAnalyzer()
        self.network_analyzer = NetworkIntelligenceAnalyzer()
        self.bottleneck_detector = bottleneck_detector or BottleneckDetector()
        self.priority_weights = priority_weights or PriorityWeights()

    def analyze(
        self,
        flow_result: FlowAggregationResult,
        metrics: Iterable[TrafficMetric],
        graph: MobilityGraph,
        top_n: Optional[int] = None,
    ) -> UrbanMobilityAnalysisResult:
        """Run OD, route, bottleneck, network, and priority analyses."""
        metric_list = list(metrics)
        od_analysis: ODAnalysisResult = self.od_analyzer.analyze(flow_result)
        route_demands = self.od_analyzer.analyze_route_demand(flow_result, graph)
        bottlenecks = self.bottleneck_detector.detect(metric_list)
        network = self.network_analyzer.analyze(metric_list, graph)
        priorities = self.network_analyzer.rank_priority_roads(
            metric_list, network, weights=self.priority_weights, limit=top_n
        )
        return UrbanMobilityAnalysisResult(
            od_analysis=od_analysis,
            route_demands=self.od_analyzer.top_route_demands(route_demands),
            bottlenecks=bottlenecks,
            network=network,
            top_priority_roads=priorities,
        )
