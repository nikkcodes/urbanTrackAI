"""Directed road network graph for UrbanTrackAI mobility engine.

Encapsulates NetworkX DiGraph with typed validation, road registry,
and dynamic road closure/restoration mechanisms.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx

from backend.mobility.models import Node, RoadSegment


class MobilityGraph:
    """Represents a city-scale directed road network graph."""

    def __init__(self, name: str = "UrbanTrackNetwork") -> None:
        self.name: str = name
        self._graph: nx.DiGraph = nx.DiGraph()
        self._nodes: Dict[str, Node] = {}
        self._roads: Dict[str, RoadSegment] = {}
        self._edge_to_road: Dict[Tuple[str, str], str] = {}
        self._closed_roads: Set[str] = set()

    @property
    def node_count(self) -> int:
        """Return total number of registered nodes in the network."""
        return len(self._nodes)

    @property
    def active_node_count(self) -> int:
        """Return number of nodes in the active routing graph."""
        return self._graph.number_of_nodes()

    @property
    def road_count(self) -> int:
        """Return total number of registered road segments (active + closed)."""
        return len(self._roads)

    @property
    def active_road_count(self) -> int:
        """Return number of currently traversable road segments in the routing graph."""
        return self._graph.number_of_edges()

    @property
    def closed_road_count(self) -> int:
        """Return number of currently closed road segments."""
        return len(self._closed_roads)

    def add_node(self, node: Node | str, **kwargs: Any) -> Node:
        """Add a node to the network.

        Args:
            node: Either a Node instance or a string node_id.
            **kwargs: Extra fields (name, lat, lon, metadata) if node is str.

        Returns:
            The added Node instance.
        """
        if isinstance(node, str):
            node_obj = Node(node_id=node, **kwargs)
        elif isinstance(node, Node):
            node_obj = node
        else:
            raise TypeError(f"Expected Node or str, got: {type(node)}")

        self._nodes[node_obj.node_id] = node_obj
        if not self._graph.has_node(node_obj.node_id):
            self._graph.add_node(node_obj.node_id, **node_obj.to_dict())

        return node_obj

    def get_node(self, node_id: str) -> Optional[Node]:
        """Retrieve a registered Node by ID."""
        return self._nodes.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node ID is registered."""
        return node_id in self._nodes

    def all_nodes(self) -> List[Node]:
        """Return a list of all registered nodes."""
        return list(self._nodes.values())

    def add_road(self, road: RoadSegment) -> RoadSegment:
        """Add a directed road segment to the network.

        Automatically ensures from_node and to_node are registered.
        If the road is closed, it is tracked in registry but omitted from active routing graph.

        Args:
            road: Validated RoadSegment instance.

        Returns:
            The added RoadSegment instance.
        """
        if not isinstance(road, RoadSegment):
            raise TypeError(f"Expected RoadSegment instance, got: {type(road)}")

        if road.road_id in self._roads:
            raise ValueError(f"Road with ID '{road.road_id}' already exists in graph.")

        edge_key = (road.from_node, road.to_node)
        if edge_key in self._edge_to_road:
            existing_road_id = self._edge_to_road[edge_key]
            raise ValueError(
                f"A directed road segment '{existing_road_id}' already connects "
                f"'{road.from_node}' -> '{road.to_node}'."
            )

        # Ensure endpoints exist
        if not self.has_node(road.from_node):
            self.add_node(road.from_node)
        if not self.has_node(road.to_node):
            self.add_node(road.to_node)

        self._roads[road.road_id] = road
        self._edge_to_road[edge_key] = road.road_id

        if road.is_closed:
            self._closed_roads.add(road.road_id)
        else:
            self._graph.add_edge(
                road.from_node,
                road.to_node,
                road_id=road.road_id,
                distance_km=road.distance_km,
                speed_limit_kmph=road.speed_limit_kmph,
                capacity_vph=road.capacity_vph,
                free_flow_time_min=road.free_flow_time_min,
                weight=road.free_flow_time_min,
            )

        return road

    def get_road(self, road_id: str) -> Optional[RoadSegment]:
        """Retrieve a road segment by ID."""
        return self._roads.get(road_id)

    def get_road_by_nodes(self, from_node: str, to_node: str) -> Optional[RoadSegment]:
        """Retrieve a road segment by its directed endpoints."""
        road_id = self._edge_to_road.get((from_node, to_node))
        if road_id:
            return self._roads.get(road_id)
        return None

    def has_road(self, road_id: str) -> bool:
        """Check if a road ID is registered."""
        return road_id in self._roads

    def remove_road(self, road_id: str) -> RoadSegment:
        """Permanently remove a road segment from the graph and registry.

        Args:
            road_id: ID of the road to remove.

        Returns:
            The removed RoadSegment.
        """
        if road_id not in self._roads:
            raise KeyError(f"Road '{road_id}' does not exist.")

        road = self._roads.pop(road_id)
        edge_key = (road.from_node, road.to_node)
        self._edge_to_road.pop(edge_key, None)
        self._closed_roads.discard(road_id)

        if self._graph.has_edge(road.from_node, road.to_node):
            self._graph.remove_edge(road.from_node, road.to_node)

        return road

    def close_road(self, road_id: str) -> bool:
        """Temporarily close a road segment without destroying its definition.

        Removes the edge from active routing graph while retaining definition.

        Args:
            road_id: ID of the road to close.

        Returns:
            True if closure state changed, False if already closed.
        """
        if road_id not in self._roads:
            raise KeyError(f"Road '{road_id}' does not exist.")

        road = self._roads[road_id]
        if road.is_closed:
            return False

        road.is_closed = True
        self._closed_roads.add(road_id)

        if self._graph.has_edge(road.from_node, road.to_node):
            self._graph.remove_edge(road.from_node, road.to_node)

        return True

    def restore_road(self, road_id: str) -> bool:
        """Restore a previously closed road segment back into the routing graph.

        Args:
            road_id: ID of the road to restore.

        Returns:
            True if restored, False if it was not closed.
        """
        if road_id not in self._roads:
            raise KeyError(f"Road '{road_id}' does not exist.")

        road = self._roads[road_id]
        if not road.is_closed:
            return False

        road.is_closed = False
        self._closed_roads.discard(road_id)

        self._graph.add_edge(
            road.from_node,
            road.to_node,
            road_id=road.road_id,
            distance_km=road.distance_km,
            speed_limit_kmph=road.speed_limit_kmph,
            capacity_vph=road.capacity_vph,
            free_flow_time_min=road.free_flow_time_min,
            weight=road.free_flow_time_min,
        )

        return True

    def is_road_closed(self, road_id: str) -> bool:
        """Check whether a road is currently marked as closed."""
        if road_id not in self._roads:
            raise KeyError(f"Road '{road_id}' does not exist.")
        return road_id in self._closed_roads

    def get_neighbors(self, node_id: str, include_closed: bool = False) -> List[str]:
        """Return outgoing neighbor nodes reachable from node_id.

        Args:
            node_id: Source node identifier.
            include_closed: If True, include neighbors reachable via closed roads.

        Returns:
            List of neighbor node IDs.
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node '{node_id}' does not exist.")

        if include_closed:
            neighbors: List[str] = []
            for (u, v) in self._edge_to_road.keys():
                if u == node_id:
                    neighbors.append(v)
            return neighbors

        if self._graph.has_node(node_id):
            return list(self._graph.successors(node_id))
        return []

    def all_roads(self, include_closed: bool = True) -> List[RoadSegment]:
        """Return all registered road segments."""
        if include_closed:
            return list(self._roads.values())
        return [r for r in self._roads.values() if not r.is_closed]

    @property
    def nx_graph(self) -> nx.DiGraph:
        """Expose underlying NetworkX DiGraph for graph algorithms (read-only use recommended)."""
        return self._graph

    def copy(self, deep: bool = True) -> MobilityGraph:
        """Create an independent copy of the mobility graph.

        Crucial for counterfactual simulations (e.g. testing road closure impacts)
        without mutating the base graph.
        """
        new_mg = MobilityGraph(name=f"{self.name}_copy")
        for node in self._nodes.values():
            new_mg.add_node(copy.deepcopy(node) if deep else node)

        for road in self._roads.values():
            road_copy = copy.deepcopy(road) if deep else road
            new_mg.add_road(road_copy)

        return new_mg

    def to_dict(self) -> Dict[str, Any]:
        """Serialize complete mobility network to dictionary."""
        return {
            "name": self.name,
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "roads": [road.to_dict() for road in self._roads.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MobilityGraph:
        """Construct MobilityGraph from dictionary representation."""
        graph = cls(name=data.get("name", "UrbanTrackNetwork"))
        for n_data in data.get("nodes", []):
            graph.add_node(Node.from_dict(n_data))
        for r_data in data.get("roads", []):
            graph.add_road(RoadSegment.from_dict(r_data))
        return graph

    def save_to_json(self, filepath: str | Path) -> None:
        """Save network to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_json(cls, filepath: str | Path) -> MobilityGraph:
        """Load network from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Network file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
