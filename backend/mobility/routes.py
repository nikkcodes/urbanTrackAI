"""Route discovery and mobility metric calculations for UrbanTrackAI.

Provides candidate path finding across the directed mobility graph and
computes route-level physical metrics (distance, free-flow travel time).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import islice
from typing import List, Optional

import networkx as nx

from backend.mobility.config import DEFAULT_MAX_CANDIDATE_ROUTES
from backend.mobility.graph import MobilityGraph


@dataclass(slots=True)
class RouteInfo:
    """Detailed information for a candidate route."""

    path: List[str]
    road_ids: List[str]
    distance_km: float
    free_flow_time_min: float


def get_candidate_routes(
    graph: MobilityGraph,
    source: str,
    destination: str,
    max_routes: int = DEFAULT_MAX_CANDIDATE_ROUTES,
    weight: Optional[str] = "free_flow_time_min",
) -> List[RouteInfo]:
    """Discover multiple candidate routes between source and destination.

    Uses NetworkX shortest simple paths to identify distinct, loop-free routes
    ordered by the specified weight (default: free-flow travel time).

    Closed road segments are automatically excluded because they are absent
    from the active routing graph.

    Args:
        graph: MobilityGraph instance.
        source: Starting junction node ID.
        destination: Ending junction node ID.
        max_routes: Maximum number of alternative paths to return (> 0).
        weight: Edge attribute to weight path ranking ('free_flow_time_min', 'distance_km', or None).

    Returns:
        List of RouteInfo objects representing feasible alternative routes.
        Returns an empty list if no valid route exists or endpoints are invalid.
    """
    if not isinstance(graph, MobilityGraph):
        raise TypeError(f"Expected MobilityGraph instance, got: {type(graph)}")

    if not graph.has_node(source) or not graph.has_node(destination):
        return []

    if source == destination:
        return []

    if max_routes <= 0:
        raise ValueError(f"max_routes must be greater than 0, got: {max_routes}")

    nx_g = graph.nx_graph

    if not nx_g.has_node(source) or not nx_g.has_node(destination):
        return []

    try:
        paths_generator = nx.shortest_simple_paths(
            nx_g,
            source=source,
            target=destination,
            weight=weight,
        )
        candidate_paths = list(islice(paths_generator, max_routes))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []

    routes: List[RouteInfo] = []
    for path in candidate_paths:
        try:
            road_ids = get_route_road_ids(graph, path)
            dist = get_route_distance(graph, path)
            travel_time = get_route_travel_time(graph, path)
            routes.append(
                RouteInfo(
                    path=path,
                    road_ids=road_ids,
                    distance_km=round(dist, 4),
                    free_flow_time_min=round(travel_time, 4),
                )
            )
        except (KeyError, ValueError):
            continue

    return routes


def get_route_road_ids(graph: MobilityGraph, path: List[str]) -> List[str]:
    """Retrieve ordered road IDs corresponding to a sequence of nodes.

    Args:
        graph: MobilityGraph instance.
        path: List of node IDs representing a continuous route.

    Returns:
        List of road segment IDs traversed.

    Raises:
        ValueError: If path has fewer than 2 nodes or an edge does not exist.
    """
    if len(path) < 2:
        raise ValueError(f"A route path must contain at least 2 nodes, got: {len(path)}")

    road_ids: List[str] = []
    for u, v in zip(path[:-1], path[1:]):
        road = graph.get_road_by_nodes(u, v)
        if road is None:
            raise ValueError(f"No registered road segment exists between '{u}' and '{v}'.")
        road_ids.append(road.road_id)

    return road_ids


def get_route_distance(graph: MobilityGraph, path: List[str]) -> float:
    """Calculate total travel distance (km) for a route path.

    Args:
        graph: MobilityGraph instance.
        path: List of node IDs representing the route.

    Returns:
        Total distance in kilometers.

    Raises:
        ValueError: If path is invalid or any road along the route does not exist.
    """
    if len(path) < 2:
        raise ValueError(f"Route path must have at least 2 nodes, got: {len(path)}")

    total_distance = 0.0
    for u, v in zip(path[:-1], path[1:]):
        road = graph.get_road_by_nodes(u, v)
        if road is None:
            raise ValueError(f"Missing road segment between '{u}' and '{v}'.")
        total_distance += road.distance_km

    return round(total_distance, 4)


def get_route_travel_time(graph: MobilityGraph, path: List[str]) -> float:
    """Calculate free-flow travel time (minutes) for a route path.

    Units: minutes = (distance_km / speed_limit_kmph) * 60.

    Args:
        graph: MobilityGraph instance.
        path: List of node IDs representing the route.

    Returns:
        Total free-flow travel time in minutes.

    Raises:
        ValueError: If path is invalid, road is missing, or speed limit is non-positive.
    """
    if len(path) < 2:
        raise ValueError(f"Route path must have at least 2 nodes, got: {len(path)}")

    total_time_min = 0.0
    for u, v in zip(path[:-1], path[1:]):
        road = graph.get_road_by_nodes(u, v)
        if road is None:
            raise ValueError(f"Missing road segment between '{u}' and '{v}'.")
        if road.speed_limit_kmph <= 0:
            raise ValueError(f"Road '{road.road_id}' has non-positive speed limit: {road.speed_limit_kmph}")

        if road.free_flow_time_min is not None and road.free_flow_time_min > 0:
            total_time_min += road.free_flow_time_min
        else:
            time_min = (road.distance_km / road.speed_limit_kmph) * 60.0
            total_time_min += time_min

    return round(total_time_min, 4)
