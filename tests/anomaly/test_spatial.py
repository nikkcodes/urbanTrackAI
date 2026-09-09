"""Tests for topology-based Phase 5 spatial grouping."""

import unittest

from backend.anomaly.models import RoadAnomalyEvidence, RoadObservationStatus
from backend.anomaly.spatial import SpatialAnomalyGrouper
from backend.mobility.graph import MobilityGraph
from backend.mobility.models import RoadSegment


class TestSpatialAnomalyGrouper(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = MobilityGraph(name="SpatialTest")
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 1.0, 40.0, 1000.0))
        self.graph.add_road(RoadSegment("R02", "J02", "J03", 1.0, 40.0, 1000.0))
        self.graph.add_road(RoadSegment("R03", "J10", "J11", 1.0, 40.0, 1000.0))

    def evidence(self, road_id: str) -> RoadAnomalyEvidence:
        return RoadAnomalyEvidence(road_id, RoadObservationStatus.COMPARABLE, 10.0, 20.0, 10.0, 1.0, "FLOW_SURGE", evidence_count=1)

    def test_connected_and_isolated_regions(self) -> None:
        evidence = {road_id: self.evidence(road_id) for road_id in ("R01", "R02", "R03")}
        regions = SpatialAnomalyGrouper().group(evidence, self.graph, evidence)
        self.assertEqual(len(regions), 2)
        self.assertEqual(regions[0].road_ids, ("R01", "R02"))
        self.assertEqual(regions[1].road_ids, ("R03",))
        self.assertEqual(regions[0].region_id, "REGION-001")

    def test_deterministic_and_invalid_road_behavior(self) -> None:
        grouper = SpatialAnomalyGrouper()
        first = grouper.group(["R02", "R01"], self.graph)
        second = grouper.group(["R01", "R02"], self.graph)
        self.assertEqual(first, second)
        with self.assertRaises(KeyError):
            grouper.group(["R99"], self.graph)


if __name__ == "__main__":
    unittest.main()
