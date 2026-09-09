"""
Unit tests for Day 3 Road Graph and Spatial Network Interface.
Tests node/edge construction, Dijkstra shortest path, top-K candidate paths,
and camera-to-road-graph association.
"""

import json
import os
import unittest

from inference.road_graph import RoadEdge, RoadGraph, RoadNode


class TestRoadGraph(unittest.TestCase):
    """Tests for RoadGraph data structures, search algorithms, and camera association."""

    def setUp(self):
        """Set up a small test road graph."""
        self.graph = RoadGraph(metadata={"name": "test_grid"})
        # Add 4 nodes forming a diamond/square network
        # N1 -> N2 -> N4 (Path 1)
        # N1 -> N3 -> N4 (Path 2)
        self.graph.add_node(RoadNode(node_id="N1", name="Junction 1", latitude=17.3850, longitude=78.4867))
        self.graph.add_node(RoadNode(node_id="N2", name="Junction 2", latitude=17.3880, longitude=78.4867))
        self.graph.add_node(RoadNode(node_id="N3", name="Junction 3", latitude=17.3850, longitude=78.4900))
        self.graph.add_node(RoadNode(node_id="N4", name="Junction 4", latitude=17.3880, longitude=78.4900))

        # Add directed edges
        # Path 1: N1 -> N2 -> N4 (Distances: 350m + 350m = 700m, speed 50 km/h)
        self.graph.add_edge(RoadEdge(
            road_id="R12", name="North Road", from_node="N1", to_node="N2", distance_m=350.0, speed_limit_kmh=50.0
        ))
        self.graph.add_edge(RoadEdge(
            road_id="R24", name="North-East Road", from_node="N2", to_node="N4", distance_m=350.0, speed_limit_kmh=50.0
        ))

        # Path 2: N1 -> N3 -> N4 (Distances: 350m + 450m = 800m, speed 80 km/h highway)
        self.graph.add_edge(RoadEdge(
            road_id="R13", name="East Highway", from_node="N1", to_node="N3", distance_m=350.0, speed_limit_kmh=80.0
        ))
        self.graph.add_edge(RoadEdge(
            road_id="R34", name="East-North Highway", from_node="N3", to_node="N4", distance_m=450.0, speed_limit_kmh=80.0
        ))

        # Explicit camera associations
        self.graph.camera_associations["cam_start"] = "N1"
        self.graph.camera_associations["cam_end"] = "N4"

    def test_node_and_edge_storage(self):
        """Verify nodes and edges are stored correctly."""
        self.assertEqual(len(self.graph.nodes), 4)
        self.assertEqual(len(self.graph.edges), 4)
        self.assertIn("N1", self.graph.nodes)
        self.assertIn("R12", self.graph.edges)
        edge = self.graph.edges.get("R12")
        self.assertIsNotNone(edge)
        self.assertEqual(edge.distance_m, 350.0)
        self.assertEqual(edge.speed_limit_kmh, 50.0)

    def test_shortest_distance_dijkstra(self):
        """Verify Dijkstra finds the shortest distance."""
        dist = self.graph.find_shortest_distance("N1", "N4")
        self.assertIsNotNone(dist)
        # Path 1 is 700m (350+350), Path 2 is 800m (350+450)
        self.assertEqual(dist, 700.0)

    def test_candidate_paths_generation(self):
        """Verify loop-free alternative paths are generated and bounded by max_paths."""
        paths = self.graph.find_candidate_paths("N1", "N4", max_paths=5)
        self.assertEqual(len(paths), 2)

        # First path should be the shorter one
        p1 = paths[0]
        self.assertEqual(p1["edges"], ["R12", "R24"])
        self.assertEqual(p1["distance_m"], 700.0)

        # Second path should be the alternative route
        p2 = paths[1]
        self.assertEqual(p2["edges"], ["R13", "R34"])
        self.assertEqual(p2["distance_m"], 800.0)

    def test_explicit_camera_association(self):
        """Verify explicit camera association takes precedence."""
        node_id = self.graph.associate_camera("cam_start", latitude=0.0, longitude=0.0)
        self.assertEqual(node_id, "N1")

    def test_geographic_snapping_camera_association(self):
        """Verify unmapped camera snaps to nearest node within threshold."""
        # Query near N2 (17.3880, 78.4867) with slight offset
        lat = 17.3881
        lon = 78.4867
        node_id = self.graph.associate_camera(
            "cam_unmapped_close", latitude=lat, longitude=lon, max_snapping_distance_m=100.0
        )
        self.assertEqual(node_id, "N2")

    def test_camera_association_rejection_when_far(self):
        """Verify camera far from all nodes is rejected with max threshold."""
        # Lat/Lon far away (Delhi coordinates vs Hyderabad)
        lat = 28.6139
        lon = 77.2090
        node_id = self.graph.associate_camera(
            "cam_far_away", latitude=lat, longitude=lon, max_snapping_distance_m=1000.0
        )
        self.assertIsNone(node_id)

    def test_synthetic_road_graph_file_loading(self):
        """Verify data/roads/synthetic_road_graph.json loads and contains required cameras."""
        filepath = os.path.join(
            os.path.dirname(__file__), "..", "data", "roads", "synthetic_road_graph.json"
        )
        self.assertTrue(os.path.exists(filepath), f"Road graph file missing at {filepath}")
        graph = RoadGraph.from_json_file(filepath)
        self.assertEqual(len(graph.nodes), 7)
        self.assertEqual(len(graph.edges), 9)

        # Check cameras 01 to 05 are mapped
        for cam_id in ["cam_01", "cam_02", "cam_03", "cam_04", "cam_05"]:
            self.assertIn(cam_id, graph.camera_associations)



if __name__ == "__main__":
    unittest.main()
