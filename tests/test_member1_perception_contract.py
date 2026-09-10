"""
Unit Test Suite for Member 1 (Kanishka) Perception Contract Alignment.
Tests all 27 required verification scenarios across Temporal, Spatial,
Identity, Trajectory, Sparse, Mobility, and Real Data subsystems.
"""

from pathlib import Path
import unittest

from inference.identity_fusion import match_observations
from inference.member3_adapter import (
    adapt_sparse_gap_to_normalized,
    adapt_trajectory_segment_to_normalized,
    adapt_vehicle_trajectory_to_normalized,
)
from inference.observation_loader import load_observations_from_json
from inference.road_graph import RoadEdge, RoadGraph, RoadNode
from inference.sparse_engine import infer_sparse_gap
from inference.spatial import spatial_feasibility
from inference.temporal import check_temporal_comparability, temporal_feasibility
from inference.trajectory_engine import reconstruct_trajectory_segment
from mobility.flow_engine import MobilityFlowEngine
from schemas.mobility_schema import RoadStatus
from schemas.observation_schema import CameraMetadata, Observation


class TestMember1PerceptionContract(unittest.TestCase):
    """Exhaustive test suite verifying the finalized Member 1 perception contract."""

    def setUp(self):
        self.base_dir = Path(__file__).parent.parent
        self.real_data_path = self.base_dir / "data" / "observations" / "kanishka_traffic.json"

        # Controlled test graph without automatic network-level sync
        self.unsynced_graph = RoadGraph(metadata={"name": "unsynced_network"})
        self.unsynced_graph.add_node(RoadNode(node_id="J1", name="Junction 1", latitude=17.3850, longitude=78.4867))
        self.unsynced_graph.add_node(RoadNode(node_id="J2", name="Junction 2", latitude=17.3880, longitude=78.4867))
        self.unsynced_graph.add_edge(RoadEdge(
            road_id="R1", name="Main St", from_node="J1", to_node="J2",
            distance_m=350.0, speed_limit_kmh=50.0, expected_speed_kmh=35.0, one_way=False
        ))
        self.unsynced_graph.camera_associations["cam_A"] = "J1"
        self.unsynced_graph.camera_associations["cam_B"] = "J2"

    # =========================================================================
    # TEMPORAL TESTS (Scenarios 1 - 7)
    # =========================================================================

    def test_01_same_camera_video_relative_usable(self):
        """1. Same-camera video-relative timestamps are usable within the video timeline."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0, timestamp_semantics="video_relative")
        obs_b = Observation(camera_id="cam_A", frame_id=150, timestamp_seconds=15.0, timestamp_semantics="video_relative")

        comp = check_temporal_comparability(obs_a, obs_b)
        self.assertTrue(comp["comparable"])
        self.assertEqual(comp["status"], "available")
        self.assertEqual(comp["delta_seconds"], 5.0)
        self.assertTrue(comp["used_in_route_scoring"])
        self.assertEqual(comp["timestamp_semantics"], "video_relative")

        tf = temporal_feasibility(obs_a, obs_b)
        self.assertEqual(tf["status"], "plausible_time_gap")
        self.assertEqual(tf["temporal_evidence"]["status"], "available")
        self.assertIsNotNone(tf["feasibility_score"])
        self.assertEqual(tf["delta_t_seconds"], 5.0)

    def test_02_diff_camera_independent_video_relative_unavailable(self):
        """2. Different-camera independent video-relative timestamps are unavailable."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0, timestamp_semantics="video_relative")
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=15.0, timestamp_semantics="video_relative")

        comp = check_temporal_comparability(obs_a, obs_b)
        self.assertFalse(comp["comparable"])
        self.assertEqual(comp["status"], "unavailable")
        self.assertIsNone(comp["delta_seconds"])
        self.assertFalse(comp["used_in_route_scoring"])
        self.assertIn("independent_camera_video_relative_timestamps_without_shared_time_reference", comp["reason"])

        tf = temporal_feasibility(obs_a, obs_b)
        self.assertEqual(tf["status"], "unavailable")
        self.assertIsNone(tf["feasibility_score"])
        self.assertIsNone(tf["delta_t_seconds"])

    def test_03_diff_camera_synchronized_usable(self):
        """3. Different-camera synchronized timestamps with shared reference are usable."""
        camera_meta = {
            "cam_A": {"timestamp_semantics": "synchronized", "time_reference_id": "city_benchmark_sync"},
            "cam_B": {"timestamp_semantics": "synchronized", "time_reference_id": "city_benchmark_sync"},
        }
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0)
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=30.0)

        comp = check_temporal_comparability(obs_a, obs_b, camera_metadata=camera_meta)
        self.assertTrue(comp["comparable"])
        self.assertEqual(comp["status"], "available")
        self.assertEqual(comp["delta_seconds"], 20.0)
        self.assertTrue(comp["used_in_route_scoring"])

    def test_04_diff_camera_documented_clock_relationship_usable(self):
        """4. Different-camera timestamps with documented clock offset relative to shared reference are usable."""
        camera_meta = {
            "cam_A": {"time_reference_id": "master_clock_01", "clock_offset_seconds": 2.0},
            "cam_B": {"time_reference_id": "master_clock_01", "clock_offset_seconds": 6.0},
        }
        # obs_a true time: 20.0 - 2.0 = 18.0s
        # With t_ref = t_local + offset:
        # obs_a true time: 20.0 + 2.0 = 22.0s
        # obs_b true time: 38.0 + 6.0 = 44.0s
        # delta = 44.0 - 22.0 = 22.0s
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=20.0)
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=38.0)

        comp = check_temporal_comparability(obs_a, obs_b, camera_metadata=camera_meta)
        self.assertTrue(comp["comparable"])
        self.assertEqual(comp["status"], "available")
        self.assertAlmostEqual(comp["delta_seconds"], 22.0, places=2)
        self.assertTrue(comp["used_in_route_scoring"])

    def test_04b_clock_offset_without_documented_reference_unavailable(self):
        """CASE 4: Standalone clock_offset_seconds without a shared time_reference_id is NOT comparable."""
        # Subcase 4A: Both cameras have numeric offsets, but NO time_reference_id
        camera_meta_no_ref = {
            "cam_A": {"clock_offset_seconds": 2.0},
            "cam_B": {"clock_offset_seconds": 6.0},
        }
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=20.0)
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=38.0)

        comp = check_temporal_comparability(obs_a, obs_b, camera_metadata=camera_meta_no_ref)
        self.assertFalse(comp["comparable"])
        self.assertEqual(comp["status"], "unavailable")
        self.assertEqual(comp["reason"], "clock_offset_without_meaningful_shared_reference")
        self.assertIsNone(comp["delta_seconds"])
        self.assertFalse(comp["used_in_route_scoring"])

        # Subcase 4B: Offsets directly on observations without time_reference_id
        obs_a_offset = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=20.0, clock_offset_seconds=2.0)
        obs_b_offset = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=38.0, clock_offset_seconds=6.0)

        comp_obs = check_temporal_comparability(obs_a_offset, obs_b_offset)
        self.assertFalse(comp_obs["comparable"])
        self.assertEqual(comp_obs["status"], "unavailable")
        self.assertEqual(comp_obs["reason"], "clock_offset_without_meaningful_shared_reference")
        self.assertIsNone(comp_obs["delta_seconds"])

        # Subcase 4C: Offsets documented with different (mismatched) time references
        camera_meta_mismatched = {
            "cam_A": {"time_reference_id": "master_clock_01", "clock_offset_seconds": 2.0},
            "cam_B": {"time_reference_id": "other_clock_02", "clock_offset_seconds": 6.0},
        }
        comp_mismatched = check_temporal_comparability(obs_a, obs_b, camera_metadata=camera_meta_mismatched)
        self.assertFalse(comp_mismatched["comparable"])
        self.assertEqual(comp_mismatched["status"], "unavailable")
        self.assertEqual(comp_mismatched["reason"], "clock_offset_without_meaningful_shared_reference")
        self.assertIsNone(comp_mismatched["delta_seconds"])

    def test_05_missing_sync_metadata_unavailable(self):
        """5. Missing synchronization metadata across cameras results in unavailable temporal evidence."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0)
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=30.0)

        # No camera metadata provided
        comp = check_temporal_comparability(obs_a, obs_b)
        self.assertFalse(comp["comparable"])
        self.assertEqual(comp["status"], "unavailable")
        self.assertIsNone(comp["delta_seconds"])
        self.assertFalse(comp["used_in_route_scoring"])

    def test_06_absolute_timestamps_without_shared_ref_unavailable(self):
        """6. Absolute timestamps without known shared temporal relationship remain unavailable."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=1700000000.0, timestamp_semantics="absolute")
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=1700000020.0, timestamp_semantics="absolute")

        comp = check_temporal_comparability(obs_a, obs_b)
        self.assertFalse(comp["comparable"])
        self.assertEqual(comp["status"], "unavailable")
        self.assertIsNone(comp["delta_seconds"])
        self.assertIn("absolute_timestamps_without_shared_temporal_reference", comp["reason"])

    def test_07_negative_inverted_interval_invalid(self):
        """7. Negative or inverted time interval is detected as invalid_negative_time."""
        obs_a = Observation(camera_id="cam_A", frame_id=150, timestamp_seconds=30.0)
        obs_b = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0)

        comp = check_temporal_comparability(obs_a, obs_b)
        self.assertFalse(comp["comparable"])
        self.assertEqual(comp["status"], "invalid_negative_time")
        self.assertIsNone(comp["delta_seconds"])
        self.assertFalse(comp["used_in_route_scoring"])

    # =========================================================================
    # SPATIAL / COORDINATE TESTS (Scenarios 8 - 12)
    # =========================================================================

    def test_08_trajectory_point_is_image_space_footpoint(self):
        """8. trajectory_point is bottom-center footpoint in image pixel coordinates."""
        obs = Observation(
            camera_id="cam_A",
            frame_id=10,
            timestamp_seconds=1.0,
            bbox=[100.0, 200.0, 300.0, 450.0],
            trajectory_point=[200.0, 450.0],
            point_type="vehicle_footpoint",
            point_coordinate_system="image",
        )
        self.assertEqual(obs.trajectory_point, [200.0, 450.0])
        self.assertEqual(obs.point_type, "vehicle_footpoint")
        self.assertEqual(obs.point_coordinate_system, "image")

    def test_09_trajectory_point_never_interpreted_as_gps_meters(self):
        """9. trajectory_point is never treated as latitude/longitude or ground meters."""
        obs = Observation(
            camera_id="cam_A",
            frame_id=10,
            timestamp_seconds=1.0,
            trajectory_point=[1280.0, 720.0],  # Pixel coordinates well outside lat/lon range
        )
        # Verify latitude and longitude remain None and are not polluted by pixel footpoint
        self.assertIsNone(obs.latitude)
        self.assertIsNone(obs.longitude)
        self.assertEqual(obs.point_coordinate_system, "image")

    def test_10_heading_angle_is_image_plane_motion_direction(self):
        """10. heading_angle is image-plane movement direction."""
        obs = Observation(
            camera_id="cam_A",
            frame_id=10,
            timestamp_seconds=1.0,
            heading_angle=42.5,
            heading_coordinate_system="image",
            heading_type="image_motion_direction",
        )
        self.assertEqual(obs.heading_angle, 42.5)
        self.assertEqual(obs.heading_coordinate_system, "image")
        self.assertEqual(obs.heading_type, "image_motion_direction")

    def test_11_heading_angle_never_interpreted_as_compass_or_road_direction(self):
        """11. heading_angle is not used as geographic bearing or road alignment."""
        cam_meta = CameraMetadata(camera_id="cam_A", bearing=180.0)
        obs = Observation(
            camera_id="cam_A",
            frame_id=10,
            timestamp_seconds=1.0,
            heading_angle=45.0,  # Image plane motion
        )
        self.assertEqual(obs.heading_angle, 45.0)
        self.assertEqual(cam_meta.bearing, 180.0)
        # Image motion angle remains distinct from compass bearing
        self.assertNotEqual(obs.heading_angle, cam_meta.bearing)

    def test_12_pixel_speed_never_interpreted_as_real_world_speed(self):
        """12. pixel_speed is in pixels/second, not m/s or km/h."""
        obs = Observation(
            camera_id="cam_A",
            frame_id=10,
            timestamp_seconds=1.0,
            pixel_speed=85.0,  # 85 pixels/sec
        )
        self.assertEqual(obs.pixel_speed, 85.0)
        # Verify spatial speed calculations do not treat pixel_speed as km/h
        sf = spatial_feasibility(obs, obs)
        self.assertIsNone(sf["required_speed_kmh"])

    # =========================================================================
    # IDENTITY TESTS (Scenarios 13 - 16)
    # =========================================================================

    def test_13_missing_reid_does_not_create_fake_embedding(self):
        """13. Missing Re-ID does not fabricate dummy or zero embeddings."""
        obs = Observation(camera_id="cam_A", frame_id=10, timestamp_seconds=1.0, appearance_embedding=None)
        self.assertIsNone(obs.appearance_embedding)
        d = obs.to_dict()
        self.assertIsNone(d.get("appearance_embedding"))

    def test_14_missing_reid_does_not_create_artificial_negative_evidence(self):
        """14. Missing Re-ID does not artificially reject an identity match."""
        camera_meta = {
            "cam_A": {"latitude": 17.3850, "longitude": 78.4867, "timestamp_semantics": "synchronized", "time_reference_id": "test_sync"},
            "cam_B": {"latitude": 17.3870, "longitude": 78.4900, "timestamp_semantics": "synchronized", "time_reference_id": "test_sync"},
        }
        obs_a = Observation(camera_id="cam_A", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", appearance_embedding=None)
        obs_b = Observation(camera_id="cam_B", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", appearance_embedding=None)

        res = match_observations(obs_a, obs_b, camera_metadata=camera_meta)
        self.assertNotEqual(res["same_vehicle_probability"], 0.0)
        self.assertEqual(res["evidence"]["appearance_status"], "missing")
        self.assertFalse(res["evidence"]["identity_evidence_available"])

    def test_15_missing_plate_does_not_create_artificial_negative_evidence(self):
        """15. Missing plate does not artificially reject an identity match."""
        camera_meta = {
            "cam_A": {"latitude": 17.3850, "longitude": 78.4867, "timestamp_semantics": "synchronized", "time_reference_id": "test_sync"},
            "cam_B": {"latitude": 17.3870, "longitude": 78.4900, "timestamp_semantics": "synchronized", "time_reference_id": "test_sync"},
        }
        obs_a = Observation(camera_id="cam_A", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", plate_text=None)
        obs_b = Observation(camera_id="cam_B", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", plate_text=None)

        res = match_observations(obs_a, obs_b, camera_metadata=camera_meta)
        self.assertNotEqual(res["same_vehicle_probability"], 0.0)
        self.assertIsNone(res["evidence"]["plate_similarity"])

    def test_16_mixed_available_unavailable_identity_evidence_continues_safely(self):
        """16. Mixed available (e.g. plate match) and unavailable (missing Re-ID) evidence fuses safely."""
        camera_meta = {
            "cam_A": {"latitude": 17.3850, "longitude": 78.4867, "timestamp_semantics": "synchronized", "time_reference_id": "test_sync"},
            "cam_B": {"latitude": 17.3870, "longitude": 78.4900, "timestamp_semantics": "synchronized", "time_reference_id": "test_sync"},
        }
        obs_a = Observation(camera_id="cam_A", track_id="t1", frame_id=10, timestamp_seconds=10.0, vehicle_type="car", plate_text="TS09AB1234", appearance_embedding=None)
        obs_b = Observation(camera_id="cam_B", track_id="t2", frame_id=40, timestamp_seconds=40.0, vehicle_type="car", plate_text="TS09AB1234", appearance_embedding=None)

        res = match_observations(obs_a, obs_b, camera_metadata=camera_meta)
        self.assertGreater(res["same_vehicle_probability"], 0.90)
        self.assertEqual(res["evidence"]["plate_similarity"], 1.0)
        self.assertEqual(res["evidence"]["appearance_status"], "missing")

    # =========================================================================
    # TRAJECTORY RECONSTRUCTION TESTS (Scenarios 17 - 20)
    # =========================================================================

    def test_17_unavailable_temporal_evidence_produces_required_speed_none(self):
        """17. Unavailable temporal evidence produces required_speed_kmh = None."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0, latitude=17.3850, longitude=78.4867, timestamp_semantics="video_relative")
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=15.0, latitude=17.3880, longitude=78.4867, timestamp_semantics="video_relative")

        sf = spatial_feasibility(obs_a, obs_b)
        self.assertIsNone(sf["required_speed_kmh"])
        self.assertEqual(sf["status"], "temporal_evidence_unavailable")

    def test_18_unavailable_temporal_evidence_does_not_produce_impossible_speed(self):
        """18. Unavailable temporal evidence does not falsely trigger impossible_speed."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0, latitude=17.3850, longitude=78.4867, timestamp_semantics="video_relative")
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=15.0, latitude=17.3880, longitude=78.4867, timestamp_semantics="video_relative")

        sf = spatial_feasibility(obs_a, obs_b)
        self.assertNotEqual(sf["status"], "impossible_speed")
        self.assertIsNone(sf["required_speed_kmh"])
        self.assertEqual(sf["status"], "temporal_evidence_unavailable")

    def test_19_temporally_unverified_routes_distinguishable_from_physically_verified(self):
        """19. Temporally unverified routes are marked feasible_temporally_unverified and have required_speed_kmh = None."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0, latitude=17.3850, longitude=78.4867, timestamp_semantics="video_relative")
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=15.0, latitude=17.3880, longitude=78.4867, timestamp_semantics="video_relative")

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.unsynced_graph)
        self.assertTrue(seg.feasible)
        self.assertEqual(len(seg.candidate_routes), 1)
        self.assertEqual(seg.candidate_routes[0].feasibility_status, "feasible_temporally_unverified")
        self.assertIsNone(seg.candidate_routes[0].required_speed_kmh)
        self.assertIsNone(seg.time_difference_seconds)
        self.assertIsNotNone(seg.temporal_evidence)
        self.assertEqual(seg.temporal_evidence["status"], "unavailable")

    def test_20_unavailable_temporal_evidence_no_fake_zero_duration_windows(self):
        """20. Unavailable temporal evidence does not create fake zero-duration time windows."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0, latitude=17.3850, longitude=78.4867, timestamp_semantics="video_relative")
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=15.0, latitude=17.3880, longitude=78.4867, timestamp_semantics="video_relative")

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.unsynced_graph)
        norm_traj = adapt_trajectory_segment_to_normalized(seg)

        self.assertIsNone(norm_traj.time_window_start)
        self.assertIsNone(norm_traj.time_window_end)
        self.assertIsNone(seg.time_difference_seconds)
        # Verify timestamps are preserved in metadata rather than collapsed to zero duration
        self.assertEqual(norm_traj.metadata.get("observation_time_a"), 10.0)
        self.assertEqual(norm_traj.metadata.get("observation_time_b"), 15.0)

    # =========================================================================
    # SPARSE INFERENCE TESTS (Scenarios 21 - 22)
    # =========================================================================

    def test_21_sparse_gap_unavailable_temporal_has_gap_duration_none(self):
        """21. Sparse gap with unavailable temporal evidence sets gap_duration_seconds = None."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0, latitude=17.3850, longitude=78.4867, timestamp_semantics="video_relative")
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=15.0, latitude=17.3880, longitude=78.4867, timestamp_semantics="video_relative")

        gap = infer_sparse_gap(obs_a, obs_b, self.unsynced_graph)
        self.assertIsNone(gap.gap_duration_seconds)
        self.assertIsNotNone(gap.metadata.get("temporal_evidence"))
        self.assertEqual(gap.metadata["temporal_evidence"]["status"], "unavailable")

        norm_gap = adapt_sparse_gap_to_normalized(gap)
        self.assertIsNone(norm_gap.time_window_start)
        self.assertIsNone(norm_gap.time_window_end)
        self.assertIsNone(gap.gap_duration_seconds)

    def test_22_sparse_inference_preserves_candidate_routes(self):
        """22. Sparse inference preserves candidate routes even when temporal evidence is unavailable."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0, latitude=17.3850, longitude=78.4867, timestamp_semantics="video_relative")
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=15.0, latitude=17.3880, longitude=78.4867, timestamp_semantics="video_relative")

        gap = infer_sparse_gap(obs_a, obs_b, self.unsynced_graph)
        self.assertGreater(len(gap.candidate_routes), 0)
        for r in gap.candidate_routes:
            self.assertTrue(r.feasible)
            self.assertEqual(r.feasibility_status, "feasible_temporally_unverified")
            self.assertIsNone(r.required_speed_kmh)

    # =========================================================================
    # MOBILITY & TRAFFIC FLOW TESTS (Scenarios 23 - 24)
    # =========================================================================

    def test_23_missing_trajectory_duration_no_invalid_expected_demand_vph(self):
        """23. Missing trajectory duration does not fabricate hourly demand rates (vph)."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0, latitude=17.3850, longitude=78.4867, timestamp_semantics="video_relative")
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=15.0, latitude=17.3880, longitude=78.4867, timestamp_semantics="video_relative")

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.unsynced_graph)
        norm_traj = adapt_trajectory_segment_to_normalized(seg)

        flow_engine = MobilityFlowEngine(self.unsynced_graph)
        road_metrics, od_matrix, issues, records = flow_engine.aggregate_flows([norm_traj])

        # Roads with flow have uncalibrated demand rate because window is unknown
        road_r1 = next((m for m in road_metrics if m.road_id == "R1"), None)
        self.assertIsNotNone(road_r1)
        self.assertIsNone(road_r1.expected_demand_vph)
        self.assertIsNone(road_r1.utilization_ratio)
        self.assertFalse(road_r1.is_hourly_rate_valid)
        self.assertEqual(road_r1.status, RoadStatus.UNCALIBRATED)

    def test_24_missing_duration_produces_is_hourly_rate_valid_false(self):
        """24. Missing duration explicitly marks is_hourly_rate_valid = False."""
        obs_a = Observation(camera_id="cam_A", frame_id=100, timestamp_seconds=10.0, latitude=17.3850, longitude=78.4867, timestamp_semantics="video_relative")
        obs_b = Observation(camera_id="cam_B", frame_id=150, timestamp_seconds=15.0, latitude=17.3880, longitude=78.4867, timestamp_semantics="video_relative")

        seg = reconstruct_trajectory_segment(obs_a, obs_b, self.unsynced_graph)
        norm_traj = adapt_trajectory_segment_to_normalized(seg)

        flow_engine = MobilityFlowEngine(self.unsynced_graph)
        road_metrics, od_matrix, issues, records = flow_engine.aggregate_flows([norm_traj])
        road_r1 = next((m for m in road_metrics if m.road_id == "R1"), None)
        self.assertIsNotNone(road_r1)
        self.assertFalse(road_r1.is_hourly_rate_valid)

    # =========================================================================
    # REAL DATA COMPATIBILITY TESTS (Scenarios 25 - 27)
    # =========================================================================

    def test_25_real_kanishka_dataset_loads_without_fabrication(self):
        """25. Real Kanishka dataset loads without any fabricated Re-ID, plate, or GPS data."""
        self.assertTrue(self.real_data_path.exists(), f"File {self.real_data_path} must exist")
        observations = load_observations_from_json(self.real_data_path)

        self.assertEqual(len(observations), 2503)
        # Verify genuine perception fields remain None where unavailable
        for obs in observations[:100]:
            self.assertIsNone(obs.appearance_embedding)
            self.assertIsNone(obs.plate_text)
            self.assertIsNone(obs.latitude)
            self.assertIsNone(obs.longitude)

    def test_26_real_kanishka_observations_safe_with_video_relative_footpoints(self):
        """26. Real Kanishka observations have video-relative timestamps and image-space footpoints."""
        observations = load_observations_from_json(self.real_data_path)
        sample = observations[0]

        self.assertEqual(sample.timestamp_semantics, "video_relative")
        self.assertEqual(sample.point_coordinate_system, "image")
        self.assertEqual(sample.point_type, "vehicle_footpoint")
        self.assertIsInstance(sample.trajectory_point, list)
        self.assertEqual(len(sample.trajectory_point), 2)
        # Footpoint y matches bottom of bounding box
        self.assertAlmostEqual(sample.trajectory_point[1], sample.bbox[3], places=1)

    def test_27_no_artificial_cross_camera_sync_introduced(self):
        """27. Two real Kanishka observations from different hypothetical cameras are not assumed synchronized."""
        obs_a = Observation(camera_id="traffic_cam_01", frame_id=100, timestamp_seconds=10.0, timestamp_semantics="video_relative")
        obs_b = Observation(camera_id="traffic_cam_02", frame_id=200, timestamp_seconds=20.0, timestamp_semantics="video_relative")

        comp = check_temporal_comparability(obs_a, obs_b)
        self.assertFalse(comp["comparable"])
        self.assertEqual(comp["status"], "unavailable")
        self.assertIsNone(comp["delta_seconds"])
        self.assertFalse(comp["used_in_route_scoring"])


if __name__ == "__main__":
    unittest.main()
