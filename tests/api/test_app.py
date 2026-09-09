"""Focused endpoint tests for the Phase 7 REST adapter."""

import unittest

from fastapi.testclient import TestClient

from backend.api.app import app


class TestMember3Api(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok", "engine": "operational"})

    def test_network_and_traffic(self) -> None:
        network = self.client.get("/api/network")
        self.assertEqual(network.status_code, 200)
        self.assertEqual(len(network.json()["nodes"]), 14)
        self.assertEqual(len(network.json()["roads"]), 28)
        traffic = self.client.get("/api/traffic")
        self.assertEqual(traffic.status_code, 200)
        self.assertEqual(traffic.json()["evaluated_roads_count"], 21)
        self.assertIn("hourly_flow", traffic.json()["metrics"][0])

    def test_analytics_and_anomalies(self) -> None:
        od = self.client.get("/api/analytics/od")
        self.assertEqual(od.status_code, 200)
        self.assertGreater(od.json()["total_demand"], 0.0)
        self.assertIn("route_demands", od.json())
        bottlenecks = self.client.get("/api/analytics/bottlenecks")
        self.assertEqual(bottlenecks.status_code, 200)
        self.assertIn("bottlenecks", bottlenecks.json())
        anomalies = self.client.get("/api/anomalies")
        self.assertEqual(anomalies.status_code, 200)
        self.assertIn("road_evidence", anomalies.json())
        self.assertIn("hhi_delta", anomalies.json())

    def test_trajectory_list_and_lookup(self) -> None:
        response = self.client.get("/api/trajectories")
        self.assertEqual(response.status_code, 200)
        trajectories = response.json()["trajectories"]
        self.assertEqual(len(trajectories), 7)
        track_id = trajectories[0]["track_id"]
        lookup = self.client.get(f"/api/trajectories/{track_id}")
        self.assertEqual(lookup.status_code, 200)
        self.assertEqual(lookup.json()["track_id"], track_id)
        missing = self.client.get("/api/trajectories/does-not-exist")
        self.assertEqual(missing.status_code, 404)

    def test_simulation_success_and_invalid_request(self) -> None:
        response = self.client.post(
            "/api/simulation",
            json={
                "scenario_id": "api-close-r01",
                "name": "Close R01",
                "closed_road_ids": ["R01"],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["scenario"]["scenario_id"], "api-close-r01")
        self.assertIn("decision", payload)
        self.assertIn("road_impacts", payload)

        invalid = self.client.post(
            "/api/simulation",
            json={
                "scenario_id": "bad",
                "name": "Bad",
                "closed_road_ids": ["R999"],
            },
        )
        self.assertEqual(invalid.status_code, 400)

    def test_simulation_json_serialization(self) -> None:
        response = self.client.post(
            "/api/simulation",
            json={"scenario_id": "speed", "name": "Slow", "speed_modifications_kmph": {"R01": 20.0}},
        )
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), dict)
        self.assertIn("baseline_summary", response.json())


if __name__ == "__main__":
    unittest.main()
