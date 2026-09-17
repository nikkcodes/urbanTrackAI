"""
CityFlowV2 Benchmark Adapter for UrbanTrack AI.
Provides clean ingestion of CityFlowV2 (AI City Challenge Multi-Camera Tracking) data
into the canonical UrbanTrack Observation schema while strictly isolating ground-truth labels.
"""

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from schemas.observation_schema import Observation
from .observation_loader import DatasetClassification


@dataclass
class CityFlowCameraConfig:
    """Camera metadata and calibration settings for a CityFlowV2 camera."""
    camera_id: str
    scenario_id: str
    fps: float = 10.0
    time_offset_seconds: float = 0.0
    homography_matrix: Optional[List[List[float]]] = None
    coordinate_system: str = "cityflow_world"
    metadata: Dict[str, Any] = field(default_factory=dict)


class CityFlowV2Adapter:
    """
    Standardized adapter for the CityFlowV2 Multi-Target Multi-Camera (MTMC) tracking dataset.

    Key Architectural Principles:
    1. STRICT GROUND-TRUTH ISOLATION:
       Ground truth vehicle identities are parsed into an external evaluation dictionary
       `ground_truth_identities: Dict[str, int]` (mapping observation_id -> gt_vehicle_id).
       The ground truth label is NEVER injected into the `Observation` inference fields,
       preventing any accidental label leakage during identity fusion.

    2. COORDINATE PRESERVATION:
       CityFlow world coordinates (from homography/calibration) are stored with explicit
       coordinate system tags (`cityflow_world`), without fabricating GPS lat/lon values.

    3. PROVENANCE INTEGRITY:
       Every observation retains source scenario, camera, frame number, timestamp, and model provenance.
    """

    def __init__(
        self,
        dataset_root: Optional[Union[str, Path]] = None,
        default_fps: float = 10.0,
    ) -> None:
        self.dataset_root = Path(dataset_root) if dataset_root else None
        self.default_fps = default_fps
        self.camera_configs: Dict[str, CityFlowCameraConfig] = {}
        # External ground-truth storage isolated from model inputs
        self.ground_truth_identities: Dict[str, int] = {}

    def register_camera(
        self,
        camera_id: str,
        scenario_id: str,
        fps: Optional[float] = None,
        time_offset_seconds: float = 0.0,
        homography_matrix: Optional[List[List[float]]] = None,
        coordinate_system: str = "cityflow_world",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a CityFlow camera sensor configuration."""
        cfg = CityFlowCameraConfig(
            camera_id=camera_id,
            scenario_id=scenario_id,
            fps=fps if fps is not None else self.default_fps,
            time_offset_seconds=time_offset_seconds,
            homography_matrix=homography_matrix,
            coordinate_system=coordinate_system,
            metadata=metadata or {},
        )
        self.camera_configs[camera_id] = cfg

    def parse_detections_and_embeddings(
        self,
        camera_id: str,
        detections_records: List[Dict[str, Any]],
        embeddings_records: Optional[Dict[int, List[float]]] = None,
        ground_truth_records: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Observation]:
        """
        Convert CityFlow raw detection/tracklet records into canonical UrbanTrack Observations.

        Args:
            camera_id: Camera identifier (e.g. 'c001', 'c002').
            detections_records: List of tracklet/detection dicts with keys:
                - frame / frame_id: int
                - track_id / local_track_id: int
                - bbox: [x, y, w, h] or [x1, y1, x2, y2]
                - confidence / detection_confidence: float
                - vehicle_type: Optional[str]
                - world_pos: Optional[[x, y]]
            embeddings_records: Optional mapping of local_track_id -> 512-D embedding.
            ground_truth_records: Optional ground truth records for evaluation only.

        Returns:
            List[Observation]: Canonical observations ready for UrbanTrack inference.
        """
        cfg = self.camera_configs.get(
            camera_id,
            CityFlowCameraConfig(camera_id=camera_id, scenario_id="default", fps=self.default_fps)
        )

        embeddings_map = embeddings_records or {}

        # 1. Parse Ground Truth into isolated evaluation store (if provided)
        if ground_truth_records:
            for gt in ground_truth_records:
                f_id = gt.get("frame") or gt.get("frame_id", 0)
                t_id = gt.get("track_id") or gt.get("local_track_id", 0)
                gt_veh_id = gt.get("target_id") or gt.get("vehicle_id") or gt.get("gt_identity")
                if gt_veh_id is not None:
                    obs_id = f"CF_{cfg.scenario_id}_{camera_id}_t{t_id}_f{f_id}"
                    self.ground_truth_identities[obs_id] = int(gt_veh_id)

        # 2. Ingest Observations (without ground-truth identity fields)
        observations: List[Observation] = []
        for det in detections_records:
            f_id = int(det.get("frame") or det.get("frame_id", 0))
            t_id = int(det.get("track_id") or det.get("local_track_id", 0))
            obs_id = f"CF_{cfg.scenario_id}_{camera_id}_t{t_id}_f{f_id}"

            # Calculate deterministic video-relative timestamp
            t_sec = round(cfg.time_offset_seconds + (f_id / cfg.fps), 3)

            # Bounding box normalization
            raw_box = det.get("bbox", [0, 0, 0, 0])
            if len(raw_box) == 4:
                if raw_box[2] > raw_box[0] and raw_box[3] > raw_box[1]:
                    bbox = [int(raw_box[0]), int(raw_box[1]), int(raw_box[2]), int(raw_box[3])]
                else:
                    bbox = [int(raw_box[0]), int(raw_box[1]), int(raw_box[0] + raw_box[2]), int(raw_box[1] + raw_box[3])]
            else:
                bbox = [0, 0, 0, 0]

            det_conf = float(det.get("confidence") or det.get("detection_confidence", 0.85))
            v_type = det.get("vehicle_type", "car")

            # Embedding retrieval
            emb = embeddings_map.get(t_id) or det.get("appearance_embedding")
            if emb is not None:
                norm = math.sqrt(sum(x * x for x in emb))
                if norm > 1e-6:
                    emb = [round(x / norm, 6) for x in emb]

            # Trajectory point / World Coordinates
            world_pos = det.get("world_pos")
            if world_pos is None and cfg.homography_matrix is not None and len(bbox) == 4:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = bbox[3]
                H = cfg.homography_matrix
                if len(H) == 3 and len(H[0]) == 3:
                    denom = H[2][0] * cx + H[2][1] * cy + H[2][2]
                    if abs(denom) > 1e-6:
                        wx = (H[0][0] * cx + H[0][1] * cy + H[0][2]) / denom
                        wy = (H[1][0] * cx + H[1][1] * cy + H[1][2]) / denom
                        world_pos = [round(wx, 2), round(wy, 2)]

            provenance = {
                "source_dataset": "CityFlowV2_2022",
                "scenario_id": cfg.scenario_id,
                "camera_id": camera_id,
                "frame_id": f_id,
                "local_track_id": t_id,
                "coordinate_system": cfg.coordinate_system,
                "fps": cfg.fps,
                "processing_version": "cityflow_adapter_v1",
            }

            obs = Observation(
                observation_id_or_camera_id=obs_id,
                camera_id=camera_id,
                frame_id=f_id,
                timestamp_seconds=t_sec,
                track_id=t_id,
                vehicle_type=v_type,
                detection_confidence=det_conf,
                bbox=bbox,
                appearance_embedding=emb,
                trajectory_point=world_pos,
                point_coordinate_system=cfg.coordinate_system,
                timestamp_semantics="video_relative",
                time_reference_id=cfg.scenario_id,
                source_provenance=provenance,
            )
            setattr(obs, "dataset_classification", DatasetClassification.CONTROLLED.value)
            observations.append(obs)

        observations.sort(key=lambda o: (o.timestamp_seconds, o.observation_id))
        return observations

    def get_ground_truth_identity(self, observation_id: str) -> Optional[int]:
        """Retrieve the isolated ground-truth identity label for evaluation."""
        return self.ground_truth_identities.get(observation_id)
