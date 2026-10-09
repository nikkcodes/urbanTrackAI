"""
UrbanTrack AI — Layer 2 Candidate Generation: Topology Gate.

Evaluates camera topology relationships using camera_graph.json as a soft prior.
Respects directed and one-way edges without hard-rejecting pairs that lack graph edges.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple, Union


@dataclass
class TopologyEvidence:
    """Directed topology relationship between two cameras."""
    has_directed_edge: bool
    relationship: Optional[str]
    edge_distance_m: Optional[float]
    edge_bearing_deg: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "has_directed_edge": self.has_directed_edge,
            "relationship": self.relationship,
            "edge_distance_m": self.edge_distance_m,
            "edge_bearing_deg": self.edge_bearing_deg,
        }


class TopologyGate:
    """
    Evaluates directed camera topology relationships.

    The camera graph acts as a soft prior:
    - Directed edges provide positive topological transition evidence.
    - Lack of an edge indicates weaker topological evidence, NOT proof of impossibility.
    - One-way edges (e.g. CAM_S02_C007 -> CAM_S02_C009) are strictly respected.
    """

    def __init__(
        self,
        graph_path: Union[str, Path] = "UrbanTrack_Member1_Handoff 2/data/config/camera_graph.json",
    ) -> None:
        self.graph_path = Path(graph_path)
        self._nodes: Set[str] = set()
        self._edges: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self._load_graph()

    def _load_graph(self) -> None:
        if not self.graph_path.is_file():
            raise FileNotFoundError(f"Camera graph file not found at {self.graph_path}")

        with open(self.graph_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._nodes = set(data.get("nodes", []))
        for edge in data.get("edges", []):
            src = edge.get("source")
            tgt = edge.get("target")
            if src and tgt:
                self._edges[(src, tgt)] = edge

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    def has_node(self, camera_id: str) -> bool:
        return camera_id in self._nodes

    def has_directed_edge(self, source_cam: str, target_cam: str) -> bool:
        return (source_cam, target_cam) in self._edges

    def get_topology_evidence(self, source_cam: str, target_cam: str) -> TopologyEvidence:
        """
        Retrieves directed topology evidence for transitions from source_cam to target_cam.
        Does NOT automatically reject if edge is absent.
        """
        edge = self._edges.get((source_cam, target_cam))
        if edge is not None:
            return TopologyEvidence(
                has_directed_edge=True,
                relationship=edge.get("relationship", "possible_transition"),
                edge_distance_m=edge.get("distance_m"),
                edge_bearing_deg=edge.get("bearing_deg"),
            )
        return TopologyEvidence(
            has_directed_edge=False,
            relationship=None,
            edge_distance_m=None,
            edge_bearing_deg=None,
        )
