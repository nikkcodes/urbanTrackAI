"""Unit tests for MobilityGraph, Node, and RoadSegment data models."""

import tempfile
import unittest
from pathlib import Path

from backend.mobility.graph import MobilityGraph
from backend.mobility.models import Node, RoadSegment


class TestNodeModel(unittest.TestCase):
    """Test Node initialization and validation."""

    def test_valid_node_creation(self) -> None:
        node = Node(node_id="J01", name="Junction 1", lat=28.61, lon=77.20)
        self.assertEqual(node.node_id, "J01")
        self.assertEqual(node.name, "Junction 1")
        self.assertAlmostEqual(node.lat, 28.61)
        self.assertAlmostEqual(node.lon, 77.20)

    def test_node_id_empty_rejection(self) -> None:
        with self.assertRaises(ValueError):
            Node(node_id="")
        with self.assertRaises(ValueError):
            Node(node_id="   ")

    def test_node_invalid_coordinates(self) -> None:
        with self.assertRaises(ValueError):
            Node(node_id="J01", lat=95.0)  # Invalid latitude (> 90)
        with self.assertRaises(ValueError):
            Node(node_id="J01", lon=-200.0)  # Invalid longitude (< -180)
        with self.assertRaises(ValueError):
            Node(node_id="J01", lat=float("nan"))


class TestRoadSegmentModel(unittest.TestCase):
    """Test RoadSegment initialization and physical validation."""

    def test_valid_road_creation(self) -> None:
        road = RoadSegment(
            road_id="R01",
            from_node="J01",
            to_node="J02",
            distance_km=2.5,
            speed_limit_kmph=50.0,
            capacity_vph=1500.0,
        )
        self.assertEqual(road.road_id, "R01")
        self.assertEqual(road.from_node, "J01")
        self.assertEqual(road.to_node, "J02")
        self.assertAlmostEqual(road.distance_km, 2.5)
        # Expected free-flow time: 2.5 / 50 * 60 = 3.0 minutes
        self.assertAlmostEqual(road.free_flow_time_min, 3.0)

    def test_empty_road_id_rejection(self) -> None:
        with self.assertRaises(ValueError):
            RoadSegment(" ", "J01", "J02", 1.0, 50.0, 1000.0)

    def test_self_loop_rejection(self) -> None:
        with self.assertRaises(ValueError):
            RoadSegment("R01", "J01", "J01", 1.0, 50.0, 1000.0)

    def test_negative_and_zero_distance_rejection(self) -> None:
        with self.assertRaises(ValueError):
            RoadSegment("R01", "J01", "J02", -1.5, 50.0, 1000.0)
        with self.assertRaises(ValueError):
            RoadSegment("R01", "J01", "J02", 0.0, 50.0, 1000.0)

    def test_negative_and_zero_speed_rejection(self) -> None:
        with self.assertRaises(ValueError):
            RoadSegment("R01", "J01", "J02", 1.5, -40.0, 1000.0)
        with self.assertRaises(ValueError):
            RoadSegment("R01", "J01", "J02", 1.5, 0.0, 1000.0)

    def test_negative_and_zero_capacity_rejection(self) -> None:
        with self.assertRaises(ValueError):
            RoadSegment("R01", "J01", "J02", 1.5, 50.0, -500.0)
        with self.assertRaises(ValueError):
            RoadSegment("R01", "J01", "J02", 1.5, 50.0, 0.0)

    def test_nan_float_rejection(self) -> None:
        with self.assertRaises(ValueError):
            RoadSegment("R01", "J01", "J02", float("nan"), 50.0, 1000.0)


class TestMobilityGraph(unittest.TestCase):
    """Test MobilityGraph construction, node/road additions, closures, and integrity."""

    def setUp(self) -> None:
        self.graph = MobilityGraph(name="TestNet")

    def test_graph_creation(self) -> None:
        self.assertEqual(self.graph.name, "TestNet")
        self.assertEqual(self.graph.node_count, 0)
        self.assertEqual(self.graph.road_count, 0)
        self.assertEqual(self.graph.active_road_count, 0)

    def test_node_insertion(self) -> None:
        node = self.graph.add_node("J01", name="Junction 1")
        self.assertEqual(self.graph.node_count, 1)
        self.assertTrue(self.graph.has_node("J01"))
        self.assertEqual(self.graph.get_node("J01").name, "Junction 1")

    def test_road_insertion(self) -> None:
        road = RoadSegment("R01", "J01", "J02", 2.0, 60.0, 1200.0)
        self.graph.add_road(road)

        # Endpoints should be automatically registered
        self.assertEqual(self.graph.node_count, 2)
        self.assertEqual(self.graph.road_count, 1)
        self.assertEqual(self.graph.active_road_count, 1)
        self.assertTrue(self.graph.has_road("R01"))
        self.assertEqual(self.graph.get_road_by_nodes("J01", "J02").road_id, "R01")

    def test_duplicate_road_id_handling(self) -> None:
        road1 = RoadSegment("R01", "J01", "J02", 2.0, 60.0, 1200.0)
        road2 = RoadSegment("R01", "J03", "J04", 1.5, 45.0, 1000.0)
        self.graph.add_road(road1)
        with self.assertRaises(ValueError):
            self.graph.add_road(road2)

    def test_duplicate_directed_edge_handling(self) -> None:
        road1 = RoadSegment("R01", "J01", "J02", 2.0, 60.0, 1200.0)
        road2 = RoadSegment("R02", "J01", "J02", 2.0, 60.0, 1200.0)
        self.graph.add_road(road1)
        with self.assertRaises(ValueError):
            self.graph.add_road(road2)

    def test_neighbor_lookup(self) -> None:
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 1.0, 50.0, 1000.0))
        self.graph.add_road(RoadSegment("R02", "J01", "J03", 2.0, 50.0, 1000.0))

        neighbors = self.graph.get_neighbors("J01")
        self.assertCountEqual(neighbors, ["J02", "J03"])

        # J02 has no outgoing roads
        self.assertEqual(self.graph.get_neighbors("J02"), [])

        # Non-existent node raises KeyError
        with self.assertRaises(KeyError):
            self.graph.get_neighbors("J99")

    def test_road_closure_and_restoration(self) -> None:
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 1.0, 50.0, 1000.0))
        self.assertEqual(self.graph.active_road_count, 1)
        self.assertFalse(self.graph.is_road_closed("R01"))

        # Close road
        closed = self.graph.close_road("R01")
        self.assertTrue(closed)
        self.assertTrue(self.graph.is_road_closed("R01"))
        self.assertEqual(self.graph.road_count, 1)  # Still registered
        self.assertEqual(self.graph.active_road_count, 0)  # But not in active graph
        self.assertEqual(self.graph.closed_road_count, 1)

        # Neighbors without include_closed should now be empty
        self.assertEqual(self.graph.get_neighbors("J01", include_closed=False), [])
        # With include_closed should show J02
        self.assertEqual(self.graph.get_neighbors("J01", include_closed=True), ["J02"])

        # Closing already closed road returns False
        self.assertFalse(self.graph.close_road("R01"))

        # Restore road
        restored = self.graph.restore_road("R01")
        self.assertTrue(restored)
        self.assertFalse(self.graph.is_road_closed("R01"))
        self.assertEqual(self.graph.active_road_count, 1)
        self.assertEqual(self.graph.closed_road_count, 0)
        self.assertEqual(self.graph.get_neighbors("J01"), ["J02"])

        # Restoring non-closed road returns False
        self.assertFalse(self.graph.restore_road("R01"))

    def test_graph_integrity_after_close_and_restore(self) -> None:
        """Verify underlying graph topology remains consistent across operations."""
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 2.0, 60.0, 1000.0))
        self.graph.add_road(RoadSegment("R02", "J02", "J03", 3.0, 60.0, 1000.0))

        initial_active = self.graph.active_road_count
        self.graph.close_road("R01")
        self.graph.restore_road("R01")

        self.assertEqual(self.graph.active_road_count, initial_active)
        self.assertEqual(self.graph.closed_road_count, 0)
        self.assertTrue(self.graph.nx_graph.has_edge("J01", "J02"))

    def test_graph_copy_isolation(self) -> None:
        """Verify modifying a copied graph does not affect the original."""
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 2.0, 60.0, 1000.0))
        copy_graph = self.graph.copy()

        copy_graph.close_road("R01")
        self.assertTrue(copy_graph.is_road_closed("R01"))
        self.assertFalse(self.graph.is_road_closed("R01"))  # Original unaffected

    def test_json_serialization_round_trip(self) -> None:
        """Verify saving to and loading from JSON preserves network structure."""
        self.graph.add_road(RoadSegment("R01", "J01", "J02", 2.0, 60.0, 1000.0, metadata={"tag": "express"}))
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        try:
            self.graph.save_to_json(tmp_path)
            loaded = MobilityGraph.load_from_json(tmp_path)
            self.assertEqual(loaded.node_count, self.graph.node_count)
            self.assertEqual(loaded.road_count, self.graph.road_count)
            road = loaded.get_road("R01")
            self.assertIsNotNone(road)
            self.assertEqual(road.metadata.get("tag"), "express")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()


if __name__ == "__main__":
    unittest.main()
