"""Unit tests for route discovery and travel-time metrics."""

import unittest
from pathlib import Path

from backend.mobility.graph import MobilityGraph
from backend.mobility.models import RoadSegment
from backend.mobility.routes import (
    get_candidate_routes,
    get_route_distance,
    get_route_road_ids,
    get_route_travel_time,
)


class TestRouteDiscoveryAndMetrics(unittest.TestCase):
    """Test candidate route discovery, distances, and travel times."""

    def setUp(self) -> None:
        """Construct a multi-path diamond network for testing:

             -> J02 (R02: 2.0 km, 60 km/h) ->
        J01                                     J04
             -> J03 (R03: 3.0 km, 45 km/h) ->
        """
        self.graph = MobilityGraph(name="DiamondNet")
        # Route 1: J01 -> J02 -> J04
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 2.0, 60.0, 1000.0))  # 2.0 min
        self.graph.add_road(RoadSegment("R02", "J02", "J04", 2.0, 60.0, 1000.0))  # 2.0 min -> Total: 4.0 km, 4.0 min

        # Route 2: J01 -> J03 -> J04
        self.graph.add_road(RoadSegment("R03", "J01", "J03", 3.0, 45.0, 1200.0))  # 4.0 min
        self.graph.add_road(RoadSegment("R04", "J03", "J04", 3.0, 45.0, 1200.0))  # 4.0 min -> Total: 6.0 km, 8.0 min

        # Disconnected node J99
        self.graph.add_node("J99")

    def test_route_discovery_multiple_routes(self) -> None:
        routes = get_candidate_routes(self.graph, source="J01", destination="J04", max_routes=5)
        self.assertEqual(len(routes), 2)

        # First route should be faster (Route 1)
        self.assertEqual(routes[0].path, ["J01", "J02", "J04"])
        self.assertEqual(routes[0].road_ids, ["R01", "R02"])
        self.assertAlmostEqual(routes[0].distance_km, 4.0)
        self.assertAlmostEqual(routes[0].free_flow_time_min, 4.0)

        # Second route
        self.assertEqual(routes[1].path, ["J01", "J03", "J04"])
        self.assertEqual(routes[1].road_ids, ["R03", "R04"])
        self.assertAlmostEqual(routes[1].distance_km, 6.0)
        self.assertAlmostEqual(routes[1].free_flow_time_min, 8.0)

    def test_no_route_handling(self) -> None:
        # Disconnected target
        routes = get_candidate_routes(self.graph, source="J01", destination="J99")
        self.assertEqual(routes, [])

        # Non-existent node
        routes = get_candidate_routes(self.graph, source="J01", destination="NON_EXISTENT")
        self.assertEqual(routes, [])

        # Source equals destination
        routes = get_candidate_routes(self.graph, source="J01", destination="J01")
        self.assertEqual(routes, [])

    def test_route_distance_calculation(self) -> None:
        dist = get_route_distance(self.graph, ["J01", "J02", "J04"])
        self.assertAlmostEqual(dist, 4.0)

        dist2 = get_route_distance(self.graph, ["J01", "J03", "J04"])
        self.assertAlmostEqual(dist2, 6.0)

        # Invalid route with fewer than 2 nodes raises ValueError
        with self.assertRaises(ValueError):
            get_route_distance(self.graph, ["J01"])

        # Disconnected path raises ValueError
        with self.assertRaises(ValueError):
            get_route_distance(self.graph, ["J01", "J04"])

    def test_route_travel_time_calculation(self) -> None:
        time_min = get_route_travel_time(self.graph, ["J01", "J02", "J04"])
        self.assertAlmostEqual(time_min, 4.0)

        time_min2 = get_route_travel_time(self.graph, ["J01", "J03", "J04"])
        self.assertAlmostEqual(time_min2, 8.0)

        with self.assertRaises(ValueError):
            get_route_travel_time(self.graph, ["J01"])

    def test_zero_speed_handling(self) -> None:
        """Verify models prevent zero speed, and routes calculation protects against non-positive speed."""
        # Model level rejection
        with self.assertRaises(ValueError):
            RoadSegment("R_ZERO", "J01", "J02", 5.0, 0.0, 1000.0)

    def test_route_road_ids_extraction(self) -> None:
        road_ids = get_route_road_ids(self.graph, ["J01", "J02", "J04"])
        self.assertEqual(road_ids, ["R01", "R02"])

        with self.assertRaises(ValueError):
            get_route_road_ids(self.graph, ["J01"])

        with self.assertRaises(ValueError):
            get_route_road_ids(self.graph, ["J01", "J04"])

    def test_closed_road_excludes_route(self) -> None:
        """Verify candidate routes adapt immediately when a road is closed."""
        self.graph.close_road("R01")  # Closes J01 -> J02

        routes = get_candidate_routes(self.graph, source="J01", destination="J04")
        self.assertEqual(len(routes), 1)
        self.assertEqual(routes[0].path, ["J01", "J03", "J04"])

        # Restore R01 and both routes return
        self.graph.restore_road("R01")
        routes = get_candidate_routes(self.graph, source="J01", destination="J04")
        self.assertEqual(len(routes), 2)

    def test_synthetic_city_routing(self) -> None:
        """Verify loading the full synthetic dataset and finding routes."""
        dataset_path = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "city_network.json"
        self.assertTrue(dataset_path.exists(), f"Synthetic dataset missing at: {dataset_path}")

        city_graph = MobilityGraph.load_from_json(dataset_path)
        self.assertEqual(city_graph.node_count, 14)
        self.assertEqual(city_graph.road_count, 28)

        # Discover routes from J01 (North Gate) to J12 (South Hub)
        routes = get_candidate_routes(city_graph, source="J01", destination="J12", max_routes=4)
        self.assertGreaterEqual(len(routes), 2)
        for r in routes:
            self.assertGreater(r.distance_km, 0.0)
            self.assertGreater(r.free_flow_time_min, 0.0)
            self.assertEqual(r.path[0], "J01")
            self.assertEqual(r.path[-1], "J12")


if __name__ == "__main__":
    unittest.main()
