"""
CityFlowV2 Benchmark Adapter for UrbanTrack AI.
Provides clean ingestion of CityFlowV2 (AI City Challenge Multi-Camera Tracking) data
into the canonical UrbanTrack Observation schema while strictly isolating ground-truth labels.
"""

from dataclasses import dataclass, field
import csv
import json
import math
from pathlib import Path
import re
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


def parse_cityflow_mot_file(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Parse a CityFlow MOT/MTSC text file without adding unavailable fields.

    CityFlow uses the MOT format ``frame,id,left,top,width,height,score,...``.
    The returned bounding box is converted to UrbanTrack's ``[x1,y1,x2,y2]``
    convention.  No vehicle class, plate, or appearance is inferred from MOT.
    """
    records: List[Dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                values = next(csv.reader([line], skipinitialspace=True))
                if len(values) < 6:
                    raise ValueError("fewer than six MOT columns")
                frame = int(float(values[0]))
                track_id = int(float(values[1]))
                left, top, width, height = (float(values[i]) for i in range(2, 6))
                if frame < 1 or track_id < 1 or width <= 0 or height <= 0:
                    raise ValueError("invalid frame, track, or bounding-box values")
                record: Dict[str, Any] = {
                    "frame": frame,
                    "track_id": track_id,
                    "bbox": [left, top, left + width, top + height],
                }
                if len(values) >= 7:
                    score = float(values[6])
                    if math.isfinite(score):
                        record["confidence"] = score
                records.append(record)
            except (TypeError, ValueError, StopIteration) as exc:
                raise ValueError(f"Invalid CityFlow MOT record at {source}:{line_number}: {line}") from exc
    return records


def parse_cityflow_camera_timestamps(path: Union[str, Path]) -> Dict[str, float]:
    """Read the official per-camera start offsets from cam_timestamp/*.txt."""
    offsets: Dict[str, float] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            values = raw_line.strip().split()
            if not values:
                continue
            if len(values) != 2:
                raise ValueError(f"Invalid CityFlow camera timestamp at {path}:{line_number}")
            camera_id, offset = values
            offsets[camera_id] = float(offset)
    return offsets


def parse_cityflow_calibration(path: Union[str, Path]) -> Tuple[List[List[float]], Optional[float]]:
    """Parse the documented 3x3 homography and reprojection error."""
    text = Path(path).read_text(encoding="utf-8")
    matrix_match = re.search(r"Homography matrix:\s*([^\n]+)", text)
    if not matrix_match:
        raise ValueError(f"Missing homography matrix in CityFlow calibration: {path}")
    rows: List[List[float]] = []
    for row in matrix_match.group(1).split(";"):
        values = [float(value) for value in row.split()]
        if len(values) != 3:
            raise ValueError(f"Homography row must contain three values: {path}")
        rows.append(values)
    error_match = re.search(r"Reprojection error:\s*([-+0-9.eE]+)", text)
    return rows, float(error_match.group(1)) if error_match else None


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
        embedding_model: Optional[str] = None,
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
                f_id = gt.get("frame", gt.get("frame_id"))
                t_id = gt.get("track_id", gt.get("local_track_id"))
                gt_veh_id = gt.get("target_id", gt.get("vehicle_id", gt.get("gt_identity")))
                if gt_veh_id is not None:
                    obs_id = f"CF_{cfg.scenario_id}_{camera_id}_t{t_id}_f{f_id}"
                    self.ground_truth_identities[obs_id] = int(gt_veh_id)

        # 2. Ingest Observations (without ground-truth identity fields)
        observations: List[Observation] = []
        for det in detections_records:
            if det.get("frame", det.get("frame_id")) is None or det.get("track_id", det.get("local_track_id")) is None:
                raise ValueError("CityFlow detections require frame and local track identifiers.")
            f_id = int(det.get("frame", det.get("frame_id")))
            t_id = int(det.get("track_id", det.get("local_track_id")))
            obs_id = f"CF_{cfg.scenario_id}_{camera_id}_t{t_id}_f{f_id}"

            # Calculate deterministic video-relative timestamp
            # CityFlow's published camera-time convention uses frame_number/fps
            # plus the scenario camera offset (kept consistent with the native
            # benchmark's frame timestamps and the legacy adapter contract).
            t_sec = round(cfg.time_offset_seconds + (f_id / cfg.fps), 3)

            # Bounding box normalization
            raw_box = det.get("bbox")
            bbox = None
            quality_flags: List[str] = []
            if isinstance(raw_box, (list, tuple)) and len(raw_box) == 4:
                if raw_box[2] > raw_box[0] and raw_box[3] > raw_box[1]:
                    bbox = [float(raw_box[0]), float(raw_box[1]), float(raw_box[2]), float(raw_box[3])]
                else:
                    bbox = [float(raw_box[0]), float(raw_box[1]), float(raw_box[0] + raw_box[2]), float(raw_box[1] + raw_box[3])]
            else:
                quality_flags.append("missing_or_invalid_bbox")

            raw_conf = det.get("confidence", det.get("detection_confidence"))
            det_conf = float(raw_conf) if raw_conf is not None else None
            if det_conf is None:
                quality_flags.append("missing_detection_confidence")
            v_type = det.get("vehicle_type")
            if v_type is None:
                quality_flags.append("missing_vehicle_class")

            # Embedding retrieval
            emb = embeddings_map.get(t_id) or det.get("appearance_embedding")
            if emb is not None:
                if not isinstance(emb, (list, tuple)) or len(emb) != 512:
                    raise ValueError(f"CityFlow OSNet embedding for track {t_id} must be 512-D.")
                if not all(isinstance(x, (int, float)) and math.isfinite(float(x)) for x in emb):
                    raise ValueError(f"CityFlow OSNet embedding for track {t_id} contains NaN, Inf, or non-numeric values.")
                norm = math.sqrt(sum(float(x) * float(x) for x in emb))
                if norm <= 1e-6:
                    raise ValueError(f"CityFlow OSNet embedding for track {t_id} has zero norm.")
                emb = [float(x) / norm for x in emb]
            else:
                quality_flags.append("missing_appearance_embedding")

            # Trajectory point / World Coordinates
            world_pos = det.get("world_pos")
            if world_pos is None and cfg.homography_matrix is not None and bbox is not None:
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
                "processing_version": "cityflow_adapter_v2",
                "timestamp_source": "camera_start_offset_plus_frame_number_over_fps",
                "source_video": f"{cfg.metadata.get('source_video', f'{cfg.scenario_id}/{camera_id}/vdo.avi')}",
                "ground_truth_isolation": "evaluation_store_only",
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
                embedding_model=embedding_model,
                trajectory_point=world_pos,
                point_type=("world_position_from_homography" if world_pos is not None else "missing_world_position"),
                point_coordinate_system=cfg.coordinate_system,
                world_position=world_pos,
                world_coordinate_system=cfg.coordinate_system if world_pos is not None else None,
                # CityFlow provides per-camera start offsets for each scenario;
                # after applying that offset, timestamps are comparable on the
                # scenario clock (with documented frame-level jitter).
                timestamp_semantics="synchronized",
                time_reference_id=cfg.scenario_id,
                source_provenance=provenance,
                source_dataset="CityFlowV2_2022",
                source_video=cfg.metadata.get("source_video"),
                frame_number=f_id,
                local_track_id=str(t_id),
                vehicle_class=v_type,
                data_quality_flags=quality_flags,
            )
            setattr(obs, "dataset_classification", DatasetClassification.CONTROLLED.value)
            observations.append(obs)

        observations.sort(key=lambda o: (o.timestamp_seconds, o.observation_id))
        return observations

    def load_scenario_from_directory(
        self,
        split: str = "train",
        scenario_id: str = "S01",
        tracker_filename: str = "mtsc_deepsort_mask_rcnn.txt",
        embedding_model: Optional[str] = None,
    ) -> Tuple[List[Observation], Dict[str, List[Dict[str, Any]]]]:
        """Load an extracted CityFlow scenario using its native files.

        Only prediction/track files are passed to inference.  Ground truth is
        returned in a separate dictionary keyed by camera and never attached
        to an Observation.
        """
        if self.dataset_root is None:
            raise ValueError("dataset_root is required for directory-based CityFlow loading.")
        scenario_root = self.dataset_root / split / scenario_id
        if not scenario_root.is_dir():
            raise FileNotFoundError(f"CityFlow scenario directory not found: {scenario_root}")

        timestamp_path = self.dataset_root / "cam_timestamp" / f"{scenario_id}.txt"
        offsets = parse_cityflow_camera_timestamps(timestamp_path) if timestamp_path.is_file() else {}
        camera_paths = sorted(scenario_root.glob("c*/"))
        if not camera_paths:
            raise FileNotFoundError(f"No camera directories found under {scenario_root}")

        observations: List[Observation] = []
        ground_truth: Dict[str, List[Dict[str, Any]]] = {}
        for camera_path in camera_paths:
            camera_id = camera_path.name
            tracker_path = camera_path / "mtsc" / tracker_filename
            gt_path = camera_path / "gt" / "gt.txt"
            calibration_path = camera_path / "calibration.txt"
            if not tracker_path.is_file():
                continue
            homography = None
            reprojection_error = None
            if calibration_path.is_file():
                homography, reprojection_error = parse_cityflow_calibration(calibration_path)
            self.register_camera(
                camera_id=camera_id,
                scenario_id=scenario_id,
                fps=8.0 if scenario_id == "S03" and camera_id == "c015" else self.default_fps,
                time_offset_seconds=offsets.get(camera_id, 0.0),
                homography_matrix=homography,
                coordinate_system="cityflow_world_meters",
                metadata={
                    "source_video": f"{split}/{scenario_id}/{camera_id}/vdo.avi",
                    "calibration_path": str(calibration_path.relative_to(self.dataset_root)) if calibration_path.is_file() else None,
                    "reprojection_error_pixels": reprojection_error,
                },
            )
            detections = parse_cityflow_mot_file(tracker_path)
            observations.extend(self.parse_detections_and_embeddings(
                camera_id=camera_id,
                detections_records=detections,
                embedding_model=embedding_model,
            ))
            ground_truth[f"{scenario_id}/{camera_id}"] = parse_cityflow_mot_file(gt_path) if gt_path.is_file() else []

        return observations, ground_truth

    def get_ground_truth_identity(self, observation_id: str) -> Optional[int]:
        """Retrieve the isolated ground-truth identity label for evaluation."""
        return self.ground_truth_identities.get(observation_id)
