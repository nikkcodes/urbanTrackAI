"""Construction of isolated counterfactual mobility graphs."""

from __future__ import annotations

from backend.mobility.graph import MobilityGraph
from backend.simulation.models import Scenario


class ScenarioGraphBuilder:
    """Apply a validated Scenario to a deep graph copy only."""

    def build(self, baseline_graph: MobilityGraph, scenario: Scenario) -> MobilityGraph:
        if not isinstance(baseline_graph, MobilityGraph):
            raise TypeError(f"Expected MobilityGraph, got: {type(baseline_graph)}")
        if not isinstance(scenario, Scenario):
            raise TypeError(f"Expected Scenario, got: {type(scenario)}")
        referenced = set(scenario.closed_road_ids) | set(scenario.capacity_modifications_vph) | set(scenario.speed_modifications_kmph)
        missing = sorted(road_id for road_id in referenced if not baseline_graph.has_road(road_id))
        if missing:
            raise KeyError(f"Scenario references unknown roads: {missing}")

        graph = baseline_graph.copy(deep=True)
        for road_id in scenario.closed_road_ids:
            graph.close_road(road_id)
        for road_id, capacity in scenario.capacity_modifications_vph.items():
            road = graph.get_road(road_id)
            assert road is not None
            road.capacity_vph = float(capacity)
            if not road.is_closed:
                graph.nx_graph[road.from_node][road.to_node]["capacity_vph"] = road.capacity_vph
        for road_id, speed in scenario.speed_modifications_kmph.items():
            road = graph.get_road(road_id)
            assert road is not None
            road.speed_limit_kmph = float(speed)
            road.free_flow_time_min = round((road.distance_km / road.speed_limit_kmph) * 60.0, 4)
            if not road.is_closed:
                graph.nx_graph[road.from_node][road.to_node]["speed_limit_kmph"] = road.speed_limit_kmph
                graph.nx_graph[road.from_node][road.to_node]["free_flow_time_min"] = road.free_flow_time_min
                graph.nx_graph[road.from_node][road.to_node]["weight"] = road.free_flow_time_min
        return graph
