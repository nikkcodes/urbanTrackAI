"""
Counterfactual Traffic Simulation Engine for UrbanTrack AI (Day 8).

Executes what-if scenarios on city road networks:
- Evaluates plausible consequences of hypothetical interventions (closures, capacity changes, demand variations, recoveries).
- Guarantees strict baseline immutability (original graph, trajectories, probabilities, and flow metrics are never mutated).
- Preserves relative likelihood semantics for baseline route probabilities while calculating separate counterfactual allocation probabilities.
- Reallocates demand strictly as: vehicle_weight * counterfactual_allocation_probability (never multiplied by reliability).
- Reuses existing Day 3 route discovery (find_candidate_paths) and Day 6 flow aggregation (MobilityFlowEngine).
- Reports explicit unroutable demand when no feasible corridor survives without fabricating routes.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from inference.road_graph import RoadGraph, RoadEdge
from mobility.flow_engine import MobilityFlowEngine
from mobility.network_analytics import NetworkCentralityAnalyzer, PriorityRoadRanker
from schemas.mobility_schema import ODMatrix, RoadFlowMetric
from schemas.normalized_trajectory_schema import NormalizedCandidateRoute, NormalizedTrajectory
from schemas.scenario_schema import (
    CounterfactualReport,
    ScenarioDefinition,
    ScenarioStatus,
    ScenarioType,
)
from simulation.impact_analyzer import ImpactAnalyzer


class CounterfactualEngine:
    """
    Simulation engine for comparative what-if network analysis.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize engine with optional configuration."""
        self.config = config or {}

    def simulate_counterfactual(
        self,
        trajectories: Iterable[NormalizedTrajectory],
        road_graph: RoadGraph,
        scenario: ScenarioDefinition,
        config: Optional[Dict[str, Any]] = None,
    ) -> CounterfactualReport:
        """
        Execute a counterfactual scenario against a preserved baseline state.

        Args:
            trajectories: Batch of baseline NormalizedTrajectory instances.
            road_graph: Baseline city RoadGraph.
            scenario: ScenarioDefinition specifying the hypothetical intervention.
            config: Optional run-time configuration overrides.

        Returns:
            CounterfactualReport: Detailed comparative analysis of baseline vs counterfactual.
        """
        traj_list = list(trajectories)
        cfg = {**self.config, **(config or {})}

        # ---------------------------------------------------------------------
        # 1. SCENARIO VALIDATION
        # ---------------------------------------------------------------------
        is_valid, validation_err = scenario.validate(road_graph)
        if not is_valid:
            return CounterfactualReport(
                scenario_id=scenario.scenario_id,
                scenario_type=(
                    scenario.scenario_type.value
                    if isinstance(scenario.scenario_type, ScenarioType)
                    else str(scenario.scenario_type)
                ),
                status=ScenarioStatus.INVALID_INPUT,
                baseline={},
                counterfactual={},
                impact={
                    "status": "invalid_input",
                    "error": validation_err,
                    "displaced_demand": 0.0,
                    "unroutable_demand": 0.0,
                    "affected_roads": [],
                    "new_bottlenecks": [],
                    "relieved_roads": [],
                    "alternate_corridors": [],
                    "bottleneck_spillover": [],
                },
                assumptions=list(scenario.assumptions),
                uncertainties=[f"Scenario validation rejected: {validation_err}"],
                metadata={"validation_passed": False, "error_message": validation_err},
            )

        # ---------------------------------------------------------------------
        # 2. BASELINE STATE CALCULATION (STRICT IMMUTABILITY)
        # ---------------------------------------------------------------------
        # Deep copy to ensure baseline objects are 100% isolated
        base_graph_copy = copy.deepcopy(road_graph)
        scenario_graph = copy.deepcopy(road_graph)

        baseline_flow_engine = MobilityFlowEngine(base_graph_copy)
        b_road_metrics, b_od_matrix, b_issues, _ = baseline_flow_engine.aggregate_flows(traj_list)

        b_ranker = PriorityRoadRanker()
        b_priorities = b_ranker.rank_roads(b_road_metrics)

        b_analyzer = NetworkCentralityAnalyzer(base_graph_copy)
        b_betweenness = b_analyzer.compute_edge_betweenness()

        baseline_summary = {
            "roads": [m.to_dict() for m in b_road_metrics],
            "od_metrics": b_od_matrix.to_dict(),
            "priority_roads": [p.road_id for p in b_priorities],
            "network_metrics": {
                "betweenness": {rid: round(v, 6) for rid, v in b_betweenness.items()}
            },
        }

        # ---------------------------------------------------------------------
        # 3. APPLY NETWORK INTERVENTIONS TO SCENARIO GRAPH
        # ---------------------------------------------------------------------
        stype = (
            scenario.scenario_type
            if isinstance(scenario.scenario_type, ScenarioType)
            else ScenarioType(scenario.scenario_type)
        )

        if stype in (ScenarioType.ROAD_CLOSURE, ScenarioType.MULTI_ROAD_CLOSURE):
            for rid in scenario.affected_roads:
                scenario_graph.close_road(rid)

        elif stype == ScenarioType.ROAD_RECOVERY:
            for rid in scenario.affected_roads:
                scenario_graph.restore_road(rid)

        elif stype in (ScenarioType.CAPACITY_REDUCTION, ScenarioType.CAPACITY_INCREASE):
            for rid, new_cap in scenario.capacity_changes.items():
                edge = scenario_graph.edges.get(rid)
                if edge:
                    edge.capacity_vph = float(new_cap)

        # Apply any explicit capacity changes regardless of scenario type
        for rid, new_cap in scenario.capacity_changes.items():
            if rid in scenario_graph.edges:
                scenario_graph.edges[rid].capacity_vph = float(new_cap)

        # ---------------------------------------------------------------------
        # 4. TRAJECTORY REROUTING & DEMAND REALLOCATION
        # ---------------------------------------------------------------------
        demand_mult = scenario.demand_multiplier
        cf_trajectories: List[NormalizedTrajectory] = []
        displaced_demand: float = 0.0
        unroutable_demand: float = 0.0
        unroutable_od_map: Dict[Tuple[str, str], float] = {}
        rerouted_corridors: Set[str] = set()
        travel_time_deltas: List[float] = []

        for traj in traj_list:
            w_base = float(traj.vehicle_weight)
            w_cf = w_base * demand_mult

            # Inspect baseline candidate routes and find which survive
            surviving_routes: List[Tuple[NormalizedCandidateRoute, List[str]]] = []
            eliminated_routes: List[Tuple[NormalizedCandidateRoute, List[str]]] = []

            for r in traj.candidate_routes:
                rids, _ = baseline_flow_engine.map_route_to_roads(r)
                # Check if route has any closed roads in scenario graph
                is_closed = any(scenario_graph.is_road_closed(rid) for rid in rids)
                if is_closed:
                    eliminated_routes.append((r, rids))
                else:
                    surviving_routes.append((r, rids))

            if surviving_routes:
                # -------------------------------------------------------------
                # Case A: Surviving baseline candidate routes exist
                # Reallocate probabilities among surviving routes:
                # allocation_i = orig_p_i / sum(orig_p_j for surviving)
                # -------------------------------------------------------------
                surv_prob_sum = sum(r.probability for r, _ in surviving_routes)

                # Track displaced demand if any route was eliminated
                if eliminated_routes:
                    elim_prob_sum = sum(r.probability for r, _ in eliminated_routes)
                    displaced_demand += w_base * elim_prob_sum

                    # Track alternate corridors that absorbed this traffic
                    for _, rids in surviving_routes:
                        rerouted_corridors.update(rids)

                cf_candidate_routes: List[NormalizedCandidateRoute] = []
                for r, rids in surviving_routes:
                    if surv_prob_sum > 0.0:
                        alloc_p = r.probability / surv_prob_sum
                    else:
                        alloc_p = 1.0 / len(surviving_routes)

                    meta = dict(r.metadata)
                    meta["original_route_probability"] = r.probability
                    meta["counterfactual_allocation_probability"] = alloc_p
                    meta["network_constraint_applied"] = len(eliminated_routes) > 0
                    if "edges" not in meta:
                        meta["edges"] = rids

                    cf_candidate_routes.append(NormalizedCandidateRoute(
                        nodes=list(r.nodes),
                        probability=alloc_p,
                        metadata=meta,
                    ))

                # Normalize rounding deviations if any
                p_sum = sum(cr.probability for cr in cf_candidate_routes)
                if abs(p_sum - 1.0) > 1e-7 and len(cf_candidate_routes) > 0:
                    for cr in cf_candidate_routes:
                        cr.probability /= p_sum

                cf_traj = NormalizedTrajectory(
                    track_id=traj.track_id,
                    origin_node=traj.origin_node,
                    destination_node=traj.destination_node,
                    candidate_routes=cf_candidate_routes,
                    vehicle_weight=w_cf,
                    timestamp=traj.timestamp,
                    time_window_start=traj.time_window_start,
                    time_window_end=traj.time_window_end,
                    metadata=dict(traj.metadata),
                )
                cf_trajectories.append(cf_traj)

            else:
                # -------------------------------------------------------------
                # Case B: All baseline routes eliminated by closures
                # Attempt to discover surviving alternatives via existing engine
                # -------------------------------------------------------------
                new_paths = scenario_graph.find_candidate_paths(
                    traj.origin_node,
                    traj.destination_node,
                    max_paths=5,
                )

                if new_paths:
                    # Displaced demand: 100% of baseline vehicle weight was displaced
                    displaced_demand += w_base

                    # Assign counterfactual allocation probabilities based on path distance
                    # (inverse distance weighting normalized to 1.0)
                    inv_dists = [1.0 / max(p["distance_m"], 1.0) for p in new_paths]
                    total_inv = sum(inv_dists)

                    cf_candidate_routes = []
                    for p, inv_d in zip(new_paths, inv_dists):
                        alloc_p = inv_d / total_inv if total_inv > 0.0 else (1.0 / len(new_paths))
                        meta = {
                            "edges": list(p["edges"]),
                            "distance_m": p["distance_m"],
                            "speed_limit_kmh": p.get("speed_limit_kmh", 50.0),
                            "estimated_travel_time_s": p.get("estimated_travel_time_s", 0.0),
                            "scenario_generated_alternative": True,
                            "original_route_probability": 0.0,
                            "counterfactual_allocation_probability": alloc_p,
                        }
                        rerouted_corridors.update(p["edges"])
                        cf_candidate_routes.append(NormalizedCandidateRoute(
                            nodes=list(p["nodes"]),
                            probability=alloc_p,
                            metadata=meta,
                        ))

                    # Normalize sum exactly to 1.0
                    p_sum = sum(cr.probability for cr in cf_candidate_routes)
                    if abs(p_sum - 1.0) > 1e-7:
                        for cr in cf_candidate_routes:
                            cr.probability /= p_sum

                    cf_traj = NormalizedTrajectory(
                        track_id=traj.track_id,
                        origin_node=traj.origin_node,
                        destination_node=traj.destination_node,
                        candidate_routes=cf_candidate_routes,
                        vehicle_weight=w_cf,
                        timestamp=traj.timestamp,
                        time_window_start=traj.time_window_start,
                        time_window_end=traj.time_window_end,
                        metadata=dict(traj.metadata),
                    )
                    cf_trajectories.append(cf_traj)

                else:
                    # ---------------------------------------------------------
                    # Case C: No feasible alternative route exists
                    # DO NOT invent a route. Explicitly report unroutable demand.
                    # ---------------------------------------------------------
                    unroutable_demand += w_cf
                    od_key = (traj.origin_node, traj.destination_node)
                    unroutable_od_map[od_key] = unroutable_od_map.get(od_key, 0.0) + w_cf

        # ---------------------------------------------------------------------
        # 5. DETERMINE SCENARIO STATUS
        # ---------------------------------------------------------------------
        if traj_list and not cf_trajectories:
            status = ScenarioStatus.NO_FEASIBLE_ROUTE
        elif unroutable_demand > 0.0:
            status = ScenarioStatus.PARTIALLY_ROUTED
        else:
            status = ScenarioStatus.SUCCESS

        # ---------------------------------------------------------------------
        # 6. COUNTERFACTUAL MOBILITY FLOW AGGREGATION
        # ---------------------------------------------------------------------
        cf_flow_engine = MobilityFlowEngine(scenario_graph)
        cf_road_metrics, cf_od_matrix, cf_issues, _ = cf_flow_engine.aggregate_flows(cf_trajectories)

        cf_ranker = PriorityRoadRanker()
        cf_priorities = cf_ranker.rank_roads(cf_road_metrics)

        cf_analyzer = NetworkCentralityAnalyzer(scenario_graph)
        cf_betweenness = cf_analyzer.compute_edge_betweenness()

        counterfactual_summary = {
            "roads": [m.to_dict() for m in cf_road_metrics],
            "od_metrics": cf_od_matrix.to_dict(),
            "priority_roads": [p.road_id for p in cf_priorities],
            "network_metrics": {
                "betweenness": {rid: round(v, 6) for rid, v in cf_betweenness.items()}
            },
        }

        # ---------------------------------------------------------------------
        # 7. COMPARATIVE IMPACT ANALYSIS
        # ---------------------------------------------------------------------
        analyzer = ImpactAnalyzer(scenario_graph)
        impact_summary = analyzer.analyze(
            scenario=scenario,
            baseline_roads=b_road_metrics,
            counterfactual_roads=cf_road_metrics,
            baseline_od=b_od_matrix,
            counterfactual_od=cf_od_matrix,
            displaced_demand=displaced_demand,
            unroutable_demand=unroutable_demand,
            unroutable_od_demands=unroutable_od_map,
            travel_time_deltas=travel_time_deltas if travel_time_deltas else None,
            rerouted_corridors=rerouted_corridors,
        )

        # ---------------------------------------------------------------------
        # 8. ASSUMPTIONS & UNCERTAINTIES
        # ---------------------------------------------------------------------
        assumptions = list(scenario.assumptions)
        assumptions.append(
            "UrbanTrack evaluates plausible network consequences under a specified hypothetical intervention."
        )
        assumptions.append(
            "Surviving vehicle demand reallocates across feasible open corridors without assuming exact future predictions."
        )
        if demand_mult != 1.0:
            assumptions.append(f"Scenario demand multiplier of {demand_mult:.2f}x applied hypothetically.")

        uncertainties = []
        if unroutable_demand > 0.0:
            uncertainties.append(
                f"Network disconnection: {unroutable_demand:.2f} PCU demand is unroutable due to missing feasible paths."
            )
        if any(not m.is_hourly_rate_valid for m in cf_road_metrics):
            uncertainties.append(
                "Observation window duration unavailable or uncalibrated; hourly demand rates (vph) and utilization preserved as uncalibrated."
            )

        return CounterfactualReport(
            scenario_id=scenario.scenario_id,
            scenario_type=stype.value,
            status=status,
            baseline=baseline_summary,
            counterfactual=counterfactual_summary,
            impact=impact_summary,
            assumptions=assumptions,
            uncertainties=uncertainties,
            metadata={
                "demand_multiplier": demand_mult,
                "input_trajectories_count": len(traj_list),
                "routed_trajectories_count": len(cf_trajectories),
                "unroutable_trajectories_count": len(traj_list) - len(cf_trajectories),
            },
        )
