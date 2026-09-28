"""
Unit tests for Observation schema and JSON serialization/deserialization.
"""

from datetime import datetime, timezone
import json
import unittest

from inference.observation import Observation


class TestObservation(unittest.TestCase):
    def test_valid_observation_creation(self):
        """Test creating an Observation with all valid fields."""
        obs = Observation(
            observation_id="OBS001",
            camera_id="CAM01",
            timestamp=datetime(2026, 9, 7, 10, 2, 13, tzinfo=timezone.utc),
            latitude=17.3850,
            longitude=78.4867,
            plate="AP09AB1234",
            plate_confidence=0.92,
            appearance_embedding=[0.12, -0.04, 0.31],
            camera_reliability=0.87,
        )
        self.assertEqual(obs.observation_id, "OBS001")
        self.assertEqual(obs.camera_id, "CAM01")
        self.assertEqual(obs.latitude, 17.3850)
        self.assertEqual(obs.longitude, 78.4867)
        self.assertEqual(obs.plate, "AP09AB1234")
        self.assertEqual(obs.plate_confidence, 0.92)
        self.assertEqual(obs.appearance_embedding, [0.12, -0.04, 0.31])
        self.assertEqual(obs.camera_reliability, 0.87)

    def test_json_roundtrip_serialization(self):
        """Test converting Observation to JSON and parsing back."""
        original_json = json.dumps({
            "observation_id": "OBS001",
            "camera_id": "CAM01",
            "timestamp": "2026-09-07T10:02:13Z",
            "plate": "AP09AB1234",
            "plate_confidence": 0.92,
            "appearance_embedding": [0.12, -0.04, 0.31, 0.88, -0.15],
            "camera_reliability": 0.87,
            "latitude": 17.3850,
            "longitude": 78.4867
        })

        obs = Observation.from_json(original_json)
        self.assertEqual(obs.observation_id, "OBS001")
        self.assertEqual(len(obs.appearance_embedding), 5)

        exported_json = obs.to_json()
        reconstructed_obs = Observation.from_json(exported_json)

        self.assertEqual(obs.observation_id, reconstructed_obs.observation_id)
        self.assertEqual(obs.camera_id, reconstructed_obs.camera_id)
        self.assertEqual(obs.latitude, reconstructed_obs.latitude)
        self.assertEqual(obs.longitude, reconstructed_obs.longitude)
        self.assertEqual(obs.plate, reconstructed_obs.plate)
        self.assertEqual(obs.plate_confidence, reconstructed_obs.plate_confidence)
        self.assertEqual(obs.appearance_embedding, reconstructed_obs.appearance_embedding)
        self.assertEqual(obs.camera_reliability, reconstructed_obs.camera_reliability)

    def test_missing_optional_fields(self):
        """Test observation creation when optional fields are None."""
        obs = Observation(
            observation_id="OBS002",
            camera_id="CAM02",
            timestamp=datetime(2026, 9, 7, 10, 5, 0),
            latitude=17.3900,
            longitude=78.4900,
        )
        self.assertIsNone(obs.plate)
        self.assertIsNone(obs.plate_confidence)
        self.assertIsNone(obs.appearance_embedding)
        self.assertIsNone(obs.camera_reliability)

        data = obs.to_dict()
        self.assertIsNone(data["plate"])

    def test_variable_length_embedding_support(self):
        """Test support for arbitrary dimension appearance embeddings."""
        emb_dim_128 = [0.01 * i for i in range(128)]
        obs = Observation(
            observation_id="OBS_VAR_DIM",
            camera_id="CAM_HIGH_RES",
            timestamp=datetime.now(),
            latitude=17.0,
            longitude=78.0,
            appearance_embedding=emb_dim_128,
        )
        self.assertEqual(len(obs.appearance_embedding), 128)

    def test_invalid_coordinates_raises_error(self):
        """Test that out-of-range latitude/longitude values raise ValueError."""
        with self.assertRaises(ValueError):
            Observation(
                observation_id="OBS_ERR",
                camera_id="CAM1",
                timestamp=datetime.now(),
                latitude=95.0,  # Invalid (>90)
                longitude=78.0,
            )

        with self.assertRaises(ValueError):
            Observation(
                observation_id="OBS_ERR",
                camera_id="CAM1",
                timestamp=datetime.now(),
                latitude=17.0,
                longitude=-190.0,  # Invalid (<-180)
            )

    def test_invalid_confidence_and_reliability_raises_error(self):
        """Test that confidence and reliability out of [0, 1] raise ValueError."""
        with self.assertRaises(ValueError):
            Observation(
                observation_id="OBS_ERR",
                camera_id="CAM1",
                timestamp=datetime.now(),
                latitude=17.0,
                longitude=78.0,
                plate_confidence=1.5,  # Invalid (>1.0)
            )

        with self.assertRaises(ValueError):
            Observation(
                observation_id="OBS_ERR",
                camera_id="CAM1",
                timestamp=datetime.now(),
                latitude=17.0,
                longitude=78.0,
                camera_reliability=-0.1,  # Invalid (<0.0)
            )

    def test_invalid_observation_id_raises_error(self):
        """Test that empty observation_id raises ValueError."""
        with self.assertRaises(ValueError):
            Observation(
                observation_id="   ",
                camera_id="CAM1",
                timestamp=datetime.now(),
                latitude=17.0,
                longitude=78.0,
            )

    def test_observation_quality_profile_categorical_states(self):
        """Verify Section 4 observation categorical quality states (VALID, DEGRADED, UNAVAILABLE)."""
        # Complete valid observation
        obs_valid = Observation(
            observation_id="OBS_VAL",
            camera_id="CAM1",
            timestamp=datetime.now(),
            latitude=17.385,
            longitude=78.486,
            track_id="trk_01",
            detection_confidence=0.88,
            plate="KA01AB1234",
            plate_bbox=[100.0, 150.0, 200.0, 180.0],
            appearance_embedding=[0.1] * 512,
            embedding_model="osnet_x0_25_aicity",
            synchronized_timestamp_seconds=102.5,
        )
        self.assertTrue(obs_valid.plate_bbox_available)
        self.assertTrue(obs_valid.plate_text_available)
        self.assertTrue(obs_valid.embedding_available)
        self.assertTrue(obs_valid.world_coordinate_available)
        self.assertEqual(obs_valid.world_coordinate_quality, "VALID")
        self.assertTrue(obs_valid.synchronized_timestamp_available)
        self.assertEqual(obs_valid.timestamp_quality, "VALID")
        self.assertEqual(obs_valid.track_quality, "VALID")

        prof = obs_valid.get_quality_profile()
        self.assertEqual(prof["world_coordinate_quality"], "VALID")
        self.assertEqual(prof["timestamp_quality"], "VALID")
        self.assertEqual(prof["track_quality"], "VALID")
        self.assertTrue(prof["plate_text_available"])
        self.assertTrue(prof["embedding_available"])

        # Degraded / unavailable observation
        obs_deg = Observation(
            observation_id="OBS_DEG",
            camera_id="CAM2",
            timestamp_seconds=50.0,
            data_quality_flags=["near_horizon", "fragmented_track"],
            detection_confidence=0.30,
        )
        self.assertFalse(obs_deg.plate_bbox_available)
        self.assertFalse(obs_deg.plate_text_available)
        self.assertFalse(obs_deg.embedding_available)
        self.assertFalse(obs_deg.world_coordinate_available)
        self.assertEqual(obs_deg.world_coordinate_quality, "UNAVAILABLE")
        self.assertFalse(obs_deg.synchronized_timestamp_available)
        self.assertEqual(obs_deg.timestamp_quality, "DEGRADED")  # video-relative fallback
        self.assertEqual(obs_deg.track_quality, "UNAVAILABLE")  # no track_id


if __name__ == "__main__":
    unittest.main()
