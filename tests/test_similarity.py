"""
Unit tests for similarity calculation utilities in UrbanTrack AI.
Validates synthetic test scenarios for plate similarity, appearance similarity,
time difference, and geographic distance calculations.
"""

from datetime import datetime, timezone
import unittest

from inference.observation import Observation
from inference.similarity import (
    appearance_similarity,
    geographic_distance,
    plate_similarity,
    time_difference,
)


class TestSimilarity(unittest.TestCase):
    # Scenario 1: Identical plates
    def test_identical_plates(self):
        """Two observations from the same vehicle with identical plates."""
        p1 = "AP09AB1234"
        p2 = "AP09AB1234"
        score = plate_similarity(p1, p2)
        self.assertEqual(score, 1.0)

    # Scenario 2: Small OCR error
    def test_small_ocr_error_plate(self):
        """Same vehicle with a single digit OCR error (AP09AB1234 vs AP09AB1284)."""
        p1 = "AP09AB1234"
        p2 = "AP09AB1284"
        score = plate_similarity(p1, p2)
        # Length 10, 1 substitution -> 1 - 1/10 = 0.90
        self.assertAlmostEqual(score, 0.90, places=2)
        self.assertGreater(score, 0.8)

    # Scenario 3: Clearly different plates
    def test_clearly_different_plates(self):
        """Two completely different license plate strings."""
        p1 = "AP09AB1234"
        p2 = "KA01CD5678"
        score = plate_similarity(p1, p2)
        self.assertLess(score, 0.3)

    # Scenario 4: Missing plate
    def test_missing_plate(self):
        """Missing/null plate on one or both observations."""
        self.assertEqual(plate_similarity(None, "AP09AB1234"), 0.0)
        self.assertEqual(plate_similarity("AP09AB1234", None), 0.0)
        self.assertEqual(plate_similarity(None, None), 0.0)

    # Scenario 5: Similar appearance embeddings
    def test_similar_appearance_embeddings(self):
        """Appearance vectors of similar/same vehicle."""
        v1 = [0.90, 0.40, 0.15]
        v2 = [0.88, 0.42, 0.14]
        score = appearance_similarity(v1, v2)
        self.assertGreater(score, 0.95)

    # Scenario 6: Different appearance embeddings
    def test_different_appearance_embeddings(self):
        """Appearance vectors of clearly different vehicles."""
        v1 = [1.0, 0.0, 0.0]
        v2 = [0.0, 1.0, 0.0]
        score = appearance_similarity(v1, v2)
        self.assertAlmostEqual(score, 0.0, places=4)

    # Scenario 7: Missing appearance embedding
    def test_missing_appearance_embedding(self):
        """Missing embedding (None, empty, or mismatched dimension)."""
        v1 = [0.5, 0.5, 0.5]
        self.assertIsNone(appearance_similarity(None, v1))
        self.assertIsNone(appearance_similarity(v1, None))
        self.assertIsNone(appearance_similarity([], v1))
        self.assertIsNone(appearance_similarity(v1, [0.5, 0.5]))  # Mismatched dim

    # Scenario 8: Known time difference
    def test_known_time_difference(self):
        """Two observations with a known time difference in seconds."""
        t1 = datetime(2026, 9, 7, 10, 2, 13, tzinfo=timezone.utc)
        t2 = datetime(2026, 9, 7, 10, 5, 0, tzinfo=timezone.utc)
        # Gap: 2 minutes 47 seconds = 167 seconds
        diff = time_difference(t1, t2)
        self.assertEqual(diff, 167.0)

        # Test using Observation instances
        obs1 = Observation("OBS1", "CAM1", t1, 17.385, 78.486)
        obs2 = Observation("OBS2", "CAM2", t2, 17.389, 78.490)
        diff_obs = time_difference(obs1, obs2)
        self.assertEqual(diff_obs, 167.0)

    # Scenario 9: Geographic distance
    def test_geographic_distance_calculation(self):
        """Two observations with different geographic locations."""
        # Coordinates in Hyderabad (approx 625 meters apart)
        lat1, lon1 = 17.3850, 78.4867
        lat2, lon2 = 17.3890, 78.4910

        dist = geographic_distance(lat1, lon1, lat2, lon2)
        self.assertGreater(dist, 500)
        self.assertLess(dist, 750)

        # Test using Observation instances
        t = datetime.now()
        obs1 = Observation("OBS1", "CAM1", t, lat1, lon1)
        obs2 = Observation("OBS2", "CAM2", t, lat2, lon2)
        dist_obs = geographic_distance(obs1, obs2)
        self.assertAlmostEqual(dist, dist_obs, places=4)

    # Scenario 10: Invalid confidence/reliability values
    def test_invalid_confidence_and_reliability_values(self):
        """Verify that invalid camera reliability and plate confidence raise ValueError."""
        t = datetime.now()
        with self.assertRaises(ValueError):
            Observation("O1", "C1", t, 17.38, 78.48, camera_reliability=1.5)

        with self.assertRaises(ValueError):
            Observation("O1", "C1", t, 17.38, 78.48, plate_confidence=-0.5)


if __name__ == "__main__":
    unittest.main()
