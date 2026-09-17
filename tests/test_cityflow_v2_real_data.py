import math
import unittest
from pathlib import Path

from inference.cityflow_adapter import (
    CityFlowV2Adapter,
    parse_cityflow_calibration,
    parse_cityflow_camera_timestamps,
    parse_cityflow_mot_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CITYFLOW_ROOT = PROJECT_ROOT / "data" / "cityflowv2"


class CityFlowV2NativeDataTests(unittest.TestCase):
    def test_native_files_parse_without_fabricating_fields(self):
        rows = parse_cityflow_mot_file(
            CITYFLOW_ROOT / "train" / "S01" / "c001" / "mtsc" / "mtsc_deepsort_mask_rcnn.txt"
        )
        self.assertEqual(len(rows), 23110)
        self.assertEqual(rows[0]["frame"], 3)
        self.assertEqual(rows[0]["track_id"], 1)
        self.assertEqual(rows[0]["bbox"], [1359.44, 266.45, 1477.16, 308.19])

    def test_offsets_and_homography_are_loaded_as_native_metadata(self):
        offsets = parse_cityflow_camera_timestamps(CITYFLOW_ROOT / "cam_timestamp" / "S01.txt")
        self.assertAlmostEqual(offsets["c002"], 1.640)
        matrix, reprojection_error = parse_cityflow_calibration(
            CITYFLOW_ROOT / "train" / "S01" / "c001" / "calibration.txt"
        )
        self.assertEqual([len(row) for row in matrix], [3, 3, 3])
        self.assertTrue(reprojection_error is not None and reprojection_error > 0)

    def test_scenario_ingestion_isolated_from_ground_truth(self):
        adapter = CityFlowV2Adapter(dataset_root=CITYFLOW_ROOT)
        observations, ground_truth = adapter.load_scenario_from_directory()
        self.assertEqual(len(observations), 98180)
        self.assertEqual(sorted({obs.camera_id for obs in observations}), ["c001", "c002", "c003", "c004", "c005"])
        self.assertEqual(sum(obs.world_position is not None for obs in observations), 98180)
        self.assertEqual(sum(obs.appearance_embedding is not None for obs in observations), 0)
        self.assertEqual(sum(obs.plate is not None for obs in observations), 0)
        self.assertFalse(any(hasattr(obs, "ground_truth_vehicle_id") for obs in observations))
        self.assertTrue(all(obs.source_dataset == "CityFlowV2_2022" for obs in observations))
        self.assertTrue(all(obs.source_provenance["ground_truth_isolation"] == "evaluation_store_only" for obs in observations))
        self.assertEqual(len(ground_truth), 5)
        self.assertTrue(all(math.isfinite(obs.timestamp_seconds) for obs in observations))

    def test_invalid_osnet_embedding_is_rejected(self):
        adapter = CityFlowV2Adapter()
        with self.assertRaises(ValueError):
            adapter.parse_detections_and_embeddings(
                "c001",
                [{"frame": 1, "track_id": 1, "bbox": [0, 0, 10, 10], "appearance_embedding": [0.0] * 511}],
            )


if __name__ == "__main__":
    unittest.main()
