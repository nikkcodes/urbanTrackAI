"""Topology-based grouping of anomalous roads."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, Mapping, Set, Tuple

from backend.anomaly.models import RoadAnomalyEvidence, SpatialAnomalyRegion
from backend.mobility.graph import MobilityGraph


class SpatialAnomalyGrouper:
    """Group anomalous roads into deterministic connected topology regions."""

    def group(
        self,
        road_ids: Iterable[str],
        graph: MobilityGraph,
        evidence: Mapping[str, RoadAnomalyEvidence] | None = None,
    ) -> Tuple[SpatialAnomalyRegion, ...]:
        """Connect roads sharing a graph endpoint; no geographic dependency is used."""
        if not isinstance(graph, MobilityGraph):
            raise TypeError(f"Expected MobilityGraph, got: {type(graph)}")
        selected = sorted(set(road_ids))
        if any(not isinstance(road_id, str) or not road_id.strip() for road_id in selected):
            raise ValueError("road_ids must contain non-empty strings")
        known = {road.road_id: road for road in graph.all_roads(include_closed=True)}
        unknown = [road_id for road_id in selected if road_id not in known]
        if unknown:
            raise KeyError(f"Roads not found in MobilityGraph: {unknown}")
        adjacency: Dict[str, Set[str]] = {road_id: set() for road_id in selected}
        for index, road_id in enumerate(selected):
            first = known[road_id]
            first_nodes = {first.from_node, first.to_node}
            for other_id in selected[index + 1:]:
                second = known[other_id]
                if first_nodes & {second.from_node, second.to_node}:
                    adjacency[road_id].add(other_id)
                    adjacency[other_id].add(road_id)

        components = []
        unvisited = set(selected)
        while unvisited:
            root = min(unvisited)
            stack = [root]
            unvisited.remove(root)
            component = []
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in sorted(adjacency[current], reverse=True):
                    if neighbor in unvisited:
                        unvisited.remove(neighbor)
                        stack.append(neighbor)
            components.append(tuple(sorted(component)))

        regions = []
        for index, component in enumerate(sorted(components)):
            records = [evidence[road_id] for road_id in component if evidence and road_id in evidence]
            evidence_count = sum(record.evidence_count for record in records) if records else len(component)
            severity = self._region_severity(records)
            regions.append(
                SpatialAnomalyRegion(
                    region_id=f"REGION-{index + 1:03d}",
                    road_ids=component,
                    evidence_count=evidence_count,
                    severity=severity,
                    explanation=f"Connected anomalous roads share MobilityGraph endpoints: {', '.join(component)}.",
                )
            )
        return tuple(regions)

    @staticmethod
    def _region_severity(records: list[RoadAnomalyEvidence]) -> str:
        if not records:
            return "LOW"
        counts = {"CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1, "NORMAL": 0}
        highest = 0
        for record in records:
            if record.bottleneck_signal:
                highest = max(highest, counts.get(record.bottleneck_signal.split(":", 1)[-1], 1))
            if record.flow_signal:
                highest = max(highest, 2 if record.flow_signal in {"FLOW_SURGE", "FLOW_DROP"} else 1)
        return next((name for name, value in counts.items() if value == highest), "LOW")
