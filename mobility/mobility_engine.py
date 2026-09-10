"""
City Mobility Intelligence Engine for UrbanTrack AI (Day 6).

Coordinates trajectory flow aggregation, OD demand analysis, network centrality,
and priority road ranking into a unified CityMobilityReport.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

from inference.road_graph import RoadGraph
from schemas.normalized_trajectory_schema import NormalizedTrajectory
from schemas.mobility_schema import CityMobilityReport
from .flow_engine import MobilityFlowEngine
from .network_analytics import NetworkCentralityAnalyzer, PriorityRoadRanker


class CityMobilityEngine:
    """
    High-level orchestrator for Day 6 City Mobility Analytics.

    Ingests NormalizedTrajectory collections, executes probabilistic route flow
    allocation, verifies demand conservation, computes structural network metrics,
    and produces a standardized, machine-readable CityMobilityReport.
    """

    def __init__(
        self,
        road_graph: Union[RoadGraph, str, Path],
        priority_ranker: Optional[PriorityRoadRanker] = None,
        probability_tolerance: float = 1e-4,
        allow_normalization: bool = True,
    ) -> None:
        """
        Initialize the CityMobilityEngine.

        Args:
            road_graph: A RoadGraph instance or path to the city network JSON.
            priority_ranker: Optional custom PriorityRoadRanker instance.
            probability_tolerance: Allowable numerical tolerance for route probability sums.
            allow_normalization: Whether to safely normalize near-1.0 probability sums.
        """
        if isinstance(road_graph, (str, Path)):
            self.graph = RoadGraph.from_json_file(road_graph)
        elif isinstance(road_graph, RoadGraph):
            self.graph = road_graph
        else:
            raise TypeError(f"Expected RoadGraph or path string/Path, got: {type(road_graph)}")

        self.flow_engine = MobilityFlowEngine(
            road_graph=self.graph,
            probability_tolerance=probability_tolerance,
            allow_normalization=allow_normalization,
        )
        self.centrality_analyzer = NetworkCentralityAnalyzer(self.graph)
        self.priority_ranker = priority_ranker or PriorityRoadRanker()

    def process_trajectories(
        self,
        trajectories: Iterable[NormalizedTrajectory],
        time_window: Optional[Dict[str, Any]] = None,
        top_priority_count: Optional[int] = 10,
    ) -> CityMobilityReport:
        """
        Execute full Day 6 mobility analytics on a collection of trajectories.

        Args:
            trajectories: Iterable of NormalizedTrajectory instances.
            time_window: Optional time window dict e.g. {"start": 1000.0, "end": 1450.0}.
            top_priority_count: Limit for priority road output.

        Returns:
            CityMobilityReport containing road metrics, OD matrix, centrality,
            priority ranking, and audit summary.
        """
        traj_list = list(trajectories)

        # 1. Flow & OD Aggregation
        road_metrics, od_matrix, validation_issues, conservation_records = (
            self.flow_engine.aggregate_flows(traj_list, time_window=time_window)
        )

        # 2. Network Centrality Analytics
        centrality_map = self.centrality_analyzer.analyze_network()

        # 3. Priority Road Ranking
        ranked_priorities = self.priority_ranker.rank_roads(
            road_metrics=road_metrics,
            centrality_metrics=centrality_map,
            limit=top_priority_count,
        )

        # 4. Summary & Invariant Verification
        total_input = len(traj_list)
        valid_input = sum(1 for c in conservation_records if c.get("is_conserved", False))
        all_conserved = (
            len(conservation_records) > 0 and all(c.get("is_conserved", False) for c in conservation_records)
        )

        total_physical_weight = sum(getattr(t, "vehicle_weight", 0.0) for t in traj_list)
        active_roads = sum(1 for m in road_metrics if m.expected_demand > 0.0)
        is_hourly_valid = any(m.is_hourly_rate_valid for m in road_metrics)
        duration_sec = road_metrics[0].duration_seconds if road_metrics else None

        valid_utils = [m.utilization_ratio for m in road_metrics if m.utilization_ratio is not None]
        avg_util = (sum(valid_utils) / len(valid_utils)) if valid_utils else None

        max_util_metric = max(
            road_metrics,
            key=lambda m: (m.utilization_ratio if m.utilization_ratio is not None else -1.0),
            default=None,
        )

        # Build effective time window
        win_dict = {
            "start": time_window.get("start") if time_window else None,
            "end": time_window.get("end") if time_window else None,
        }
        if win_dict["start"] is None and road_metrics and road_metrics[0].time_window_start is not None:
            win_dict["start"] = road_metrics[0].time_window_start
            win_dict["end"] = road_metrics[0].time_window_end

        summary = {
            "total_trajectories_input": total_input,
            "valid_trajectories_count": len(conservation_records),
            "demand_conservation_verified": all_conserved,
            "total_physical_demand_od": round(od_matrix.total_demand, 4),
            "total_input_vehicle_weight": round(total_physical_weight, 4),
            "total_roads_in_network": len(self.graph.edges),
            "active_roads_with_demand": active_roads,
            "observation_duration_seconds": duration_sec,
            "is_hourly_rate_valid": is_hourly_valid,
            "flow_rate_unit": "vph" if is_hourly_valid else "window_count",
            "average_network_utilization": round(avg_util, 6) if avg_util is not None else None,
            "max_road_utilization": round(max_util_metric.utilization_ratio, 6) if (max_util_metric and max_util_metric.utilization_ratio is not None) else None,
            "highest_utilization_road": max_util_metric.road_id if (max_util_metric and max_util_metric.utilization_ratio is not None and max_util_metric.utilization_ratio > 0) else None,
            "validation_issues_count": len(validation_issues),
        }

        return CityMobilityReport(
            time_window=win_dict,
            road_metrics=road_metrics,
            od_demand=od_matrix.pairs,
            priority_roads=ranked_priorities,
            network_centrality=centrality_map,
            validation_issues=validation_issues,
            summary=summary,
        )
