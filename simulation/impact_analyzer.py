"""
Counterfactual Impact Analyzer for UrbanTrack AI (Day 8).

Analyzes differences between baseline network state and counterfactual scenario state:
- Road-level deltas (demand, safe percent delta, utilization).
- Network-level summaries (affected demand, displaced demand, unroutable demand).
- Operational state transitions (new bottlenecks, relieved roads).
- Alternate corridors absorbing displaced demand.
- Graph-topological bottleneck spillover.
- Origin-Destination impact matrix.

Audit Invariants Preserved:
- Safe division by zero (never fabricate percentage changes when baseline is 0.0).
- Topological connectivity used for spillover rather than vague proximity.
- Uncalibrated time windows remain uncalibrated without fake hourly rates.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set, Tuple

from inference.road_graph import RoadGraph
from schemas.mobility_schema import (
    ODMatrix,
    ODPairDemand,
    RoadFlowMetric,
    RoadStatus,
)
from schemas.scenario_schema import (
    ODImpactMetric,
    RoadImpactMetric,
    ScenarioDefinition,
)


class ImpactAnalyzer:
    """
    Evaluates and quantifies comparative network consequences between
    a preserved baseline state and a counterfactual scenario state.
    """

    def __init__(self, road_graph: RoadGraph) -> None:
        """Initialize analyzer with the base road network topology."""
        self.graph = road_graph

    def analyze(
        self,
        scenario: ScenarioDefinition,
        baseline_roads: List[RoadFlowMetric],
        counterfactual_roads: List[RoadFlowMetric],
        baseline_od: ODMatrix,
        counterfactual_od: ODMatrix,
        displaced_demand: float = 0.0,
        unroutable_demand: float = 0.0,
        unroutable_od_demands: Optional[Dict[Tuple[str, str], float]] = None,
        travel_time_deltas: Optional[List[float]] = None,
        rerouted_corridors: Optional[Set[str]] = None,
    ) -> Dict[str, Any]:
        """
        Produce a comprehensive comparative impact analysis.

        Args:
            scenario: The applied ScenarioDefinition.
            baseline_roads: Baseline road flow metrics.
            counterfactual_roads: Counterfactual road flow metrics.
            baseline_od: Baseline OD matrix.
            counterfactual_od: Counterfactual OD matrix.
            displaced_demand: Demand shifted due to trajectory rerouting.
            unroutable_demand: Demand that could not be routed.
            unroutable_od_demands: Unroutable demand broken down by (origin, dest).
            travel_time_deltas: List of travel time differences for rerouted vehicles.
            rerouted_corridors: Roads that received flow from rerouted trajectories.

        Returns:
            Dict containing detailed road impacts, OD impacts, network summaries, and classifications.
        """
        base_road_map: Dict[str, RoadFlowMetric] = {m.road_id: m for m in baseline_roads}
        cf_road_map: Dict[str, RoadFlowMetric] = {m.road_id: m for m in counterfactual_roads}
        unroutable_map = unroutable_od_demands or {}
        rerouted_roads = rerouted_corridors or set()

        road_impacts: List[RoadImpactMetric] = []
        newly_congested_roads: List[str] = []
        relieved_roads: List[str] = []
        alternate_corridors: List[str] = []
        bottleneck_spillover: List[str] = []

        intervened_roads: Set[str] = set(scenario.affected_roads) | set(scenario.capacity_changes.keys())
        intervened_nodes: Set[str] = set()
        for rid in intervened_roads:
            edge = self.graph.edges.get(rid)
            if edge:
                intervened_nodes.add(edge.from_node)
                intervened_nodes.add(edge.to_node)

        all_rids = sorted(set(base_road_map.keys()) | set(cf_road_map.keys()))

        for rid in all_rids:
            base_m = base_road_map.get(rid)
            cf_m = cf_road_map.get(rid)

            b_demand = base_m.expected_demand_in_window if base_m else 0.0
            cf_demand = cf_m.expected_demand_in_window if cf_m else 0.0
            demand_delta = cf_demand - b_demand

            # Safe percent delta
            demand_delta_pct: Optional[float] = None
            if b_demand > 1e-6:
                demand_delta_pct = round((demand_delta / b_demand) * 100.0, 2)
            elif cf_demand > 1e-6:
                # Baseline had no demand, now has demand
                demand_delta_pct = None

            b_util = base_m.utilization_ratio if base_m and base_m.is_hourly_rate_valid else None
            cf_util = cf_m.utilization_ratio if cf_m and cf_m.is_hourly_rate_valid else None

            util_delta: Optional[float] = None
            if b_util is not None and cf_util is not None:
                util_delta = round(cf_util - b_util, 6)

            b_status_str = base_m.status.value if base_m and isinstance(base_m.status, RoadStatus) else "normal"
            cf_status_str = cf_m.status.value if cf_m and isinstance(cf_m.status, RoadStatus) else "normal"

            # Bottleneck detection: was normal/moderate, now high/over_capacity
            is_new_bottleneck = False
            if cf_status_str in ("high", "over_capacity") and b_status_str not in ("high", "over_capacity"):
                is_new_bottleneck = True
                newly_congested_roads.append(rid)

            # Relieved roads: demand or utilization dropped noticeably
            is_relieved = False
            if demand_delta < -1e-4:
                is_relieved = True
                relieved_roads.append(rid)

            # Alternate corridor:
            # A surviving road (not directly closed/intervened) that received increased demand
            # attributable to rerouted trajectories
            is_alternate = False
            if rid not in intervened_roads and (demand_delta > 1e-4):
                if not rerouted_roads or rid in rerouted_roads:
                    is_alternate = True
                    alternate_corridors.append(rid)

            # Bottleneck spillover:
            # A new bottleneck that is topologically connected (shares a junction node)
            # with an intervened corridor
            is_spillover = False
            if is_new_bottleneck and rid not in intervened_roads:
                edge = self.graph.edges.get(rid)
                if edge and (edge.from_node in intervened_nodes or edge.to_node in intervened_nodes):
                    is_spillover = True
                    bottleneck_spillover.append(rid)

            from_n = cf_m.from_node if cf_m else (base_m.from_node if base_m else "")
            to_n = cf_m.to_node if cf_m else (base_m.to_node if base_m else "")

            metric = RoadImpactMetric(
                road_id=rid,
                from_node=from_n,
                to_node=to_n,
                baseline_demand=b_demand,
                counterfactual_demand=cf_demand,
                demand_delta=demand_delta,
                demand_delta_percent=demand_delta_pct,
                baseline_utilization=b_util,
                counterfactual_utilization=cf_util,
                utilization_delta=util_delta,
                baseline_status=b_status_str,
                counterfactual_status=cf_status_str,
                is_new_bottleneck=is_new_bottleneck,
                is_relieved=is_relieved,
                is_alternate_corridor=is_alternate,
                is_bottleneck_spillover=is_spillover,
            )
            road_impacts.append(metric)

        # ---------------------------------------------------------------------
        # OD IMPACT ANALYSIS
        # ---------------------------------------------------------------------
        od_impacts: List[ODImpactMetric] = []
        base_pairs: Dict[Tuple[str, str], float] = {
            (p.origin, p.destination): p.demand for p in baseline_od.pairs
        }
        cf_pairs: Dict[Tuple[str, str], float] = {
            (p.origin, p.destination): p.demand for p in counterfactual_od.pairs
        }
        all_od_keys = sorted(set(base_pairs.keys()) | set(cf_pairs.keys()) | set(unroutable_map.keys()))

        for orig, dest in all_od_keys:
            b_dem = base_pairs.get((orig, dest), 0.0)
            cf_dem = cf_pairs.get((orig, dest), 0.0)
            unroute = unroutable_map.get((orig, dest), 0.0)
            od_impacts.append(ODImpactMetric(
                origin=orig,
                destination=dest,
                baseline_demand=b_dem,
                counterfactual_demand=cf_dem,
                demand_delta=cf_dem - b_dem,
                unroutable_demand=unroute,
            ))

        # ---------------------------------------------------------------------
        # TRAVEL TIME SUMMARY
        # ---------------------------------------------------------------------
        avg_travel_time_delta: Optional[float] = None
        if travel_time_deltas:
            avg_travel_time_delta = round(sum(travel_time_deltas) / len(travel_time_deltas), 2)

        # ---------------------------------------------------------------------
        # NETWORK SUMMARY
        # ---------------------------------------------------------------------
        total_baseline_demand = baseline_od.total_demand
        total_cf_demand = counterfactual_od.total_demand

        affected_roads_list = sorted(list(intervened_roads | {
            m.road_id for m in road_impacts if abs(m.demand_delta) > 1e-4
        }))

        impact_summary = {
            "affected_roads": affected_roads_list,
            "displaced_demand": round(displaced_demand, 4),
            "unroutable_demand": round(unroutable_demand, 4),
            "total_baseline_demand": round(total_baseline_demand, 4),
            "total_counterfactual_demand": round(total_cf_demand, 4),
            "new_bottlenecks": sorted(newly_congested_roads),
            "relieved_roads": sorted(relieved_roads),
            "alternate_corridors": sorted(alternate_corridors),
            "bottleneck_spillover": sorted(bottleneck_spillover),
            "travel_time_change": avg_travel_time_delta,
            "road_impacts": [m.to_dict() for m in road_impacts],
            "od_impacts": [od.to_dict() for od in od_impacts],
        }

        return impact_summary
