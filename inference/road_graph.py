"""
Road Graph representation and spatial network utilities for UrbanTrack AI.
Provides RoadGraph, RoadNode, RoadEdge, camera-to-graph association, and candidate path search.
"""

from dataclasses import dataclass, field
import heapq
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .similarity import geographic_distance


@dataclass
class RoadNode:
    """Represents an intersection / road junction in the spatial network."""
    node_id: str
    latitude: float
    longitude: float
    name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "name": self.name or self.node_id,
            "latitude": self.latitude,
            "longitude": self.longitude,
        }


@dataclass
class RoadEdge:
    """Represents a physical road segment connecting two junctions."""
    road_id: str
    from_node: str
    to_node: str
    distance_m: float
    name: str = ""
    speed_limit_kmh: float = 50.0
    expected_speed_kmh: float = 40.0
    one_way: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "road_id": self.road_id,
            "name": self.name,
            "from": self.from_node,
            "to": self.to_node,
            "distance_m": self.distance_m,
            "speed_limit_kmh": self.speed_limit_kmh,
            "expected_speed_kmh": self.expected_speed_kmh,
            "one_way": self.one_way,
        }


class RoadGraph:
    """
    Spatial Road Network Graph for vehicle trajectory inference.
    Supports camera-to-network association, candidate route generation, and travel time estimation.
    """

    def __init__(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        self.metadata: Dict[str, Any] = metadata or {}
        self.nodes: Dict[str, RoadNode] = {}
        self.edges: Dict[str, RoadEdge] = {}
        # adjacency: node_id -> list of (neighbor_node_id, road_id, distance_m, edge_obj)
        self.adjacency: Dict[str, List[Tuple[str, str, float, RoadEdge]]] = {}
        self.camera_associations: Dict[str, str] = {}

    def add_node(self, node: RoadNode) -> None:
        """Add a junction node to the graph."""
        self.nodes[node.node_id] = node
        if node.node_id not in self.adjacency:
            self.adjacency[node.node_id] = []

    def add_edge(self, edge: RoadEdge) -> None:
        """Add a road segment to the graph (supports one-way and bidirectional)."""
        self.edges[edge.road_id] = edge
        if edge.from_node not in self.adjacency:
            self.adjacency[edge.from_node] = []
        if edge.to_node not in self.adjacency:
            self.adjacency[edge.to_node] = []

        # Forward direction
        self.adjacency[edge.from_node].append((edge.to_node, edge.road_id, edge.distance_m, edge))

        # Reverse direction if bidirectional
        if not edge.one_way:
            self.adjacency[edge.to_node].append((edge.from_node, edge.road_id, edge.distance_m, edge))

    def associate_camera(
        self,
        camera_id: str,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        max_snapping_distance_m: float = 150.0,
    ) -> Optional[str]:
        """
        Associate a camera ID with a junction node in the road graph.

        Rules:
            1. Explicit association: If camera_id is already mapped, return associated node_id.
            2. Spatial snapping: If coordinates are provided, find nearest graph node within max_snapping_distance_m.
            3. If distance exceeds max_snapping_distance_m or coordinates are absent, return None (fail safely).

        Args:
            camera_id: Identifier of the camera.
            latitude: Optional camera latitude coordinate.
            longitude: Optional camera longitude coordinate.
            max_snapping_distance_m: Maximum allowable snapping distance in meters (default 150m).

        Returns:
            Optional[str]: Associated node_id or None if association cannot be established safely.
        """
        if camera_id in self.camera_associations:
            return self.camera_associations[camera_id]

        if latitude is None or longitude is None:
            return None

        best_node_id: Optional[str] = None
        min_dist = float("inf")

        for node_id, node in self.nodes.items():
            dist = geographic_distance(latitude, longitude, node.latitude, node.longitude)
            if dist < min_dist:
                min_dist = dist
                best_node_id = node_id

        if best_node_id is not None and min_dist <= max_snapping_distance_m:
            return best_node_id

        # Distance exceeds threshold: do NOT fabricate an association
        return None

    def find_shortest_distance(self, start_node: str, end_node: str) -> Optional[float]:
        """Compute the shortest road distance between two nodes using Dijkstra's algorithm."""
        if start_node not in self.nodes or end_node not in self.nodes:
            return None
        if start_node == end_node:
            return 0.0

        distances: Dict[str, float] = {start_node: 0.0}
        pq: List[Tuple[float, str]] = [(0.0, start_node)]

        while pq:
            d, u = heapq.heappop(pq)
            if d > distances.get(u, float("inf")):
                continue
            if u == end_node:
                return d

            for v, _, dist, _ in self.adjacency.get(u, []):
                new_d = d + dist
                if new_d < distances.get(v, float("inf")):
                    distances[v] = new_d
                    heapq.heappush(pq, (new_d, v))

        return None

    def find_candidate_paths(
        self,
        start_node: str,
        end_node: str,
        max_paths: int = 5,
        max_distance_factor: float = 2.0,
        max_depth: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple plausible loop-free candidate routes between start and end junctions.

        Does NOT return only the shortest path: explores reasonable alternative corridors
        while bounding search depth to avoid combinatorial explosion.

        Args:
            start_node: Origin junction ID.
            end_node: Destination junction ID.
            max_paths: Maximum candidate paths to return (top-K, default 5).
            max_distance_factor: Allowable length ratio relative to shortest path (default 2.0).
            max_depth: Maximum edge hops allowed in a candidate path (default 10).

        Returns:
            List[Dict[str, Any]]: Ranked list of candidate paths with edges, nodes, and distance.
        """
        if start_node not in self.nodes or end_node not in self.nodes:
            return []

        shortest_dist = self.find_shortest_distance(start_node, end_node)
        if shortest_dist is None:
            return []  # No path exists in graph

        # Handle same-node stationary case
        if start_node == end_node:
            return [{
                "edges": [],
                "nodes": [start_node],
                "distance_m": 0.0,
                "speed_limit_kmh": 50.0,
                "min_travel_time_s": 0.0,
                "estimated_travel_time_s": 0.0,
            }]

        cutoff_dist = max(shortest_dist * max_distance_factor, shortest_dist + 500.0)

        # Priority Queue for bounded exploration: (current_dist, current_node, path_nodes, path_edges)
        pq: List[Tuple[float, str, List[str], List[str]]] = [(0.0, start_node, [start_node], [])]
        found_paths: List[Dict[str, Any]] = []
        seen_path_signatures: Set[Tuple[str, ...]] = set()

        while pq and len(found_paths) < max_paths:
            dist, curr, nodes_visited, edges_traversed = heapq.heappop(pq)

            if curr == end_node:
                sig = tuple(edges_traversed)
                if sig not in seen_path_signatures:
                    seen_path_signatures.add(sig)

                    # Compute route travel times and speed limits
                    edge_objs = [self.edges[rid] for rid in edges_traversed]
                    min_speed_limit = min((e.speed_limit_kmh for e in edge_objs), default=50.0)
                    min_t = sum(e.distance_m / (e.speed_limit_kmh / 3.6) for e in edge_objs)
                    est_t = sum(e.distance_m / (e.expected_speed_kmh / 3.6) for e in edge_objs)

                    found_paths.append({
                        "edges": edges_traversed,
                        "nodes": nodes_visited,
                        "distance_m": dist,
                        "speed_limit_kmh": min_speed_limit,
                        "min_travel_time_s": min_t,
                        "estimated_travel_time_s": est_t,
                    })
                continue

            if len(nodes_visited) > max_depth or dist > cutoff_dist:
                continue

            for neighbor, road_id, edge_dist, _ in self.adjacency.get(curr, []):
                # Avoid loops
                if neighbor not in nodes_visited:
                    new_dist = dist + edge_dist
                    if new_dist <= cutoff_dist:
                        heapq.heappush(
                            pq,
                            (new_dist, neighbor, nodes_visited + [neighbor], edges_traversed + [road_id]),
                        )

        # Sort paths by distance ascending
        found_paths.sort(key=lambda p: p["distance_m"])
        return found_paths[:max_paths]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RoadGraph":
        """Construct RoadGraph from a dictionary representation."""
        graph = cls(metadata=data.get("metadata", {}))

        for n in data.get("nodes", []):
            node = RoadNode(
                node_id=str(n["id"]),
                name=str(n.get("name", n["id"])),
                latitude=float(n["latitude"]),
                longitude=float(n["longitude"]),
            )
            graph.add_node(node)

        for e in data.get("edges", []):
            edge = RoadEdge(
                road_id=str(e["road_id"]),
                name=str(e.get("name", e["road_id"])),
                from_node=str(e["from"]),
                to_node=str(e["to"]),
                distance_m=float(e["distance_m"]),
                speed_limit_kmh=float(e.get("speed_limit_kmh", 50.0)),
                expected_speed_kmh=float(e.get("expected_speed_kmh", 40.0)),
                one_way=bool(e.get("one_way", False)),
            )
            graph.add_edge(edge)

        graph.camera_associations = dict(data.get("camera_associations", {}))
        return graph

    @classmethod
    def from_json_file(cls, filepath: Union[str, Path]) -> "RoadGraph":
        """Load and parse RoadGraph from a JSON file."""
        p = Path(filepath)
        if not p.is_file():
            raise FileNotFoundError(f"Road graph file not found: {filepath}")
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert RoadGraph to serializable dictionary."""
        return {
            "metadata": self.metadata,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges.values()],
            "camera_associations": self.camera_associations,
        }
