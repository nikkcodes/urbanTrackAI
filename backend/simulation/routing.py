"""Deterministic route assignment for counterfactual scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from backend.flow.models import CandidateRoute, NormalizedTrajectory
from backend.mobility.graph import MobilityGraph
from backend.mobility.routes import RouteInfo, get_candidate_routes, get_route_road_ids, get_route_travel_time
from backend.simulation.models import RouteAssignment, UnroutableDemand


@dataclass(frozen=True, slots=True)
class RoutingOutcome:
    assignments: Tuple[RouteAssignment, ...]
    unroutable: Tuple[UnroutableDemand, ...]


class DeterministicRouter:
    """Assign route demand by preserving feasible baseline paths, then shortest time."""

    def __init__(self, max_candidate_routes: int = 20) -> None:
        if not isinstance(max_candidate_routes, int) or max_candidate_routes <= 0:
            raise ValueError("max_candidate_routes must be a positive integer")
        self.max_candidate_routes = max_candidate_routes

    def assign_baseline(
        self,
        trajectories: Iterable[NormalizedTrajectory],
        graph: MobilityGraph,
    ) -> RoutingOutcome:
        """Expand each trajectory's candidate routes into probability-weighted assignments."""
        records: List[RouteAssignment] = []
        for trajectory in trajectories:
            if not isinstance(trajectory, NormalizedTrajectory):
                raise TypeError(f"Expected NormalizedTrajectory, got: {type(trajectory)}")
            for index, route in enumerate(trajectory.candidate_routes):
                if not self._route_is_feasible(graph, route.nodes):
                    raise ValueError(f"Baseline candidate route for '{trajectory.track_id}' is not feasible")
                records.append(
                    self._assignment(
                        track_id=f"{trajectory.track_id}:candidate:{index}",
                        origin=trajectory.origin_node,
                        destination=trajectory.destination_node,
                        demand=trajectory.vehicle_weight * route.probability,
                        route_nodes=tuple(route.nodes),
                        graph=graph,
                        route_changed=False,
                    )
                )
        records.sort(key=lambda item: (item.origin, item.destination, item.track_id, item.route_nodes))
        return RoutingOutcome(tuple(records), ())

    def assign_counterfactual(
        self,
        trajectories: Iterable[NormalizedTrajectory],
        baseline_assignments: Iterable[RouteAssignment],
        graph: MobilityGraph,
    ) -> RoutingOutcome:
        """Preserve each baseline candidate route when feasible, otherwise reroute it."""
        trajectory_by_key = {
            (trajectory.track_id, index): trajectory
            for trajectory in trajectories
            for index, _ in enumerate(trajectory.candidate_routes)
        }
        assignments: List[RouteAssignment] = []
        unroutable: List[UnroutableDemand] = []
        for baseline in baseline_assignments:
            if self._route_is_feasible(graph, baseline.route_nodes):
                assignments.append(
                    self._assignment(
                        track_id=baseline.track_id,
                        origin=baseline.origin,
                        destination=baseline.destination,
                        demand=baseline.demand,
                        route_nodes=baseline.route_nodes,
                        graph=graph,
                        route_changed=False,
                    )
                )
                continue
            alternatives = get_candidate_routes(
                graph,
                baseline.origin,
                baseline.destination,
                max_routes=self.max_candidate_routes,
                weight="free_flow_time_min",
            )
            if not alternatives:
                unroutable.append(UnroutableDemand(
                    origin=baseline.origin,
                    destination=baseline.destination,
                    demand=baseline.demand,
                    reason="no feasible route remains after scenario intervention",
                ))
                continue
            selected = min(alternatives, key=lambda route: (route.free_flow_time_min, tuple(route.path), tuple(route.road_ids)))
            assignments.append(
                self._assignment(
                    track_id=baseline.track_id,
                    origin=baseline.origin,
                    destination=baseline.destination,
                    demand=baseline.demand,
                    route_nodes=tuple(selected.path),
                    graph=graph,
                    route_changed=True,
                )
            )
        assignments.sort(key=lambda item: (item.origin, item.destination, item.track_id, item.route_nodes))
        unroutable.sort(key=lambda item: (item.origin, item.destination, item.reason))
        return RoutingOutcome(tuple(assignments), tuple(unroutable))

    @staticmethod
    def _route_is_feasible(graph: MobilityGraph, nodes: Iterable[str]) -> bool:
        nodes = tuple(nodes)
        if len(nodes) < 2:
            return False
        return all(graph.nx_graph.has_edge(source, target) for source, target in zip(nodes[:-1], nodes[1:]))

    @staticmethod
    def _assignment(
        track_id: str,
        origin: str,
        destination: str,
        demand: float,
        route_nodes: Tuple[str, ...],
        graph: MobilityGraph,
        route_changed: bool,
    ) -> RouteAssignment:
        return RouteAssignment(
            track_id=track_id,
            origin=origin,
            destination=destination,
            demand=demand,
            road_ids=tuple(get_route_road_ids(graph, list(route_nodes))),
            route_nodes=route_nodes,
            route_travel_time_minutes=get_route_travel_time(graph, list(route_nodes)),
            route_changed=route_changed,
        )
