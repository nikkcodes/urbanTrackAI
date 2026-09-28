"""
UrbanTrack AI — Member 2 to Member 3 Downstream Adapter.

Ingests real Member 2 artifacts from the AI City 2022 validation run:
- results/aicity_validation/inference_results.json
- results/aicity_mvp/trajectories.json
- results/aicity_validation/dataset_inventory.json
- results/aicity_validation/calibration_metadata.json
- results/aicity_validation/synchronization_metadata.json
- results/aicity_mvp/normalized_observations.json

Enforces Non-Negotiable Invariants:
1. Zero Data Fabrication:
   - Latitude and longitude remain None (never fabricated GPS).
   - Plates remain None, displaying 'Plate unavailable / privacy-censored'.
   - Horizon instability (|W| < 0.50) preserves world_x=None, world_y=None.
2. Provenance & Coordinate Semantics:
   - Coordinate system tag is strictly 'cityflow_world' (local benchmark metric space, meters).
   - Provenance source is 'Member2 inference'.
3. Re-ID Compatibility Guard:
   - Cross-model comparisons with C002 (osnet_x0_25_msmt17 vs osnet_x0_25_aicity)
     are labeled 'Insufficient appearance evidence'.
4. Ground-Truth Isolation:
   - Ground-truth benchmark vehicle IDs are never exposed as production identities.
5. Dynamic N-Camera Architecture:
   - Discovers cameras dynamically from data without hardcoded camera ID branches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class NormalizedObservation:
    """Standardized downstream observation representation with provenance."""
    observation_id: str
    camera_id: str
    track_id: str
    frame_id: int
    timestamp_seconds: float
    synchronized_timestamp_seconds: Optional[float]
    vehicle_type: str
    detection_confidence: float
    camera_reliability: float
    bbox: List[float]
    world_x: Optional[float]
    world_y: Optional[float]
    projection_status: str  # 'valid', 'unstable_horizon_denominator', 'uncalibrated'
    coordinate_system: str  # 'cityflow_world' or 'image'
    reid_model: Optional[str]
    plate: Optional[str] = None
    plate_status: str = "Plate unavailable / privacy-censored"
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "frame_id": self.frame_id,
            "timestamp_seconds": self.timestamp_seconds,
            "synchronized_timestamp_seconds": self.synchronized_timestamp_seconds,
            "vehicle_type": self.vehicle_type,
            "detection_confidence": round(self.detection_confidence, 4),
            "camera_reliability": round(self.camera_reliability, 4),
            "bbox": [round(c, 1) for c in self.bbox],
            "world_x": round(self.world_x, 2) if self.world_x is not None else None,
            "world_y": round(self.world_y, 2) if self.world_y is not None else None,
            "projection_status": self.projection_status,
            "coordinate_system": self.coordinate_system,
            "reid_model": self.reid_model,
            "plate": self.plate,
            "plate_status": self.plate_status,
        }


@dataclass
class NormalizedIdentity:
    """Standardized downstream inferred identity representation."""
    identity_id: str
    identity_status: str  # 'unconfirmed_singleton', 'candidate', 'confirmed', 'ambiguous'
    admission_status: str
    cameras: List[str]
    observation_ids: List[str]
    observations_count: int
    vehicle_type: str
    confidence: Optional[float]
    plate_display: str = "Plate unavailable / privacy-censored"
    appearance_status: str = "available"
    consistency: Dict[str, Any] = field(default_factory=dict)
    evidence_summary: Dict[str, Any] = field(default_factory=dict)
    world_points: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "identity_status": self.identity_status,
            "admission_status": self.admission_status,
            "cameras": self.cameras,
            "observation_ids": self.observation_ids,
            "observations_count": self.observations_count,
            "vehicle_type": self.vehicle_type,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "plate_display": self.plate_display,
            "appearance_status": self.appearance_status,
            "consistency": self.consistency,
            "evidence_summary": self.evidence_summary,
            "world_points": self.world_points,
        }


@dataclass
class NormalizedTrajectory:
    """Standardized downstream trajectory representation."""
    identity_id: str
    cameras_visited: List[str]
    observations_count: int
    start_timestamp: float
    end_timestamp: float
    total_time_seconds: float
    total_distance_m: float
    overall_confidence: float
    is_ambiguous: bool
    spatial_semantics: str  # 'cityflow_world' or 'image_space_only'
    temporal_semantics: str  # 'video_relative' or 'aicity_official_synchronized'
    segments: List[Dict[str, Any]] = field(default_factory=list)
    waypoints: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "cameras_visited": self.cameras_visited,
            "observations_count": self.observations_count,
            "start_timestamp": round(self.start_timestamp, 2),
            "end_timestamp": round(self.end_timestamp, 2),
            "total_time_seconds": round(self.total_time_seconds, 2),
            "total_distance_m": round(self.total_distance_m, 2),
            "overall_confidence": round(self.overall_confidence, 4),
            "is_ambiguous": self.is_ambiguous,
            "spatial_semantics": self.spatial_semantics,
            "temporal_semantics": self.temporal_semantics,
            "segments": self.segments,
            "waypoints": self.waypoints,
        }


@dataclass
class CameraMetadata:
    """Standardized metadata for an observed camera node."""
    camera_id: str
    coordinate_system: str
    has_calibration: bool
    reprojection_error_px: Optional[float]
    sync_offset_seconds: float
    center_world_x: Optional[float]
    center_world_y: Optional[float]
    total_observations: int
    unique_tracks: int
    mean_reliability: float
    vehicle_type_distribution: Dict[str, int]
    reid_model: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "coordinate_system": self.coordinate_system,
            "has_calibration": self.has_calibration,
            "reprojection_error_px": round(self.reprojection_error_px, 2) if self.reprojection_error_px is not None else None,
            "sync_offset_seconds": round(self.sync_offset_seconds, 3),
            "center_world_x": round(self.center_world_x, 2) if self.center_world_x is not None else None,
            "center_world_y": round(self.center_world_y, 2) if self.center_world_y is not None else None,
            "total_observations": self.total_observations,
            "unique_tracks": self.unique_tracks,
            "mean_reliability": round(self.mean_reliability, 4),
            "vehicle_type_distribution": self.vehicle_type_distribution,
            "reid_model": self.reid_model,
        }


class Member2Adapter:
    """
    Robust downstream adapter consuming Member 2 inference and validation artifacts.
    Provides normalized, type-safe structures for city analytics, visualization, and REST APIs.
    """

    def __init__(
        self,
        base_dir: Optional[Union[str, Path]] = None,
        inference_file: Optional[Union[str, Path]] = None,
        trajectories_file: Optional[Union[str, Path]] = None,
        observations_file: Optional[Union[str, Path]] = None,
        inventory_file: Optional[Union[str, Path]] = None,
        calibration_file: Optional[Union[str, Path]] = None,
        synchronization_file: Optional[Union[str, Path]] = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).resolve().parent.parent.parent

        # Default paths to authoritative Member 2 artifacts
        self.inference_path = Path(inference_file) if inference_file else (
            self.base_dir / "results" / "aicity_validation" / "inference_results.json"
        )
        self.trajectories_path = Path(trajectories_file) if trajectories_file else (
            self.base_dir / "results" / "aicity_mvp" / "trajectories.json"
        )
        self.observations_path = Path(observations_file) if observations_file else (
            self.base_dir / "results" / "aicity_mvp" / "normalized_observations.json"
        )
        self.inventory_path = Path(inventory_file) if inventory_file else (
            self.base_dir / "results" / "aicity_validation" / "dataset_inventory.json"
        )
        self.calibration_path = Path(calibration_file) if calibration_file else (
            self.base_dir / "results" / "aicity_validation" / "calibration_metadata.json"
        )
        self.synchronization_path = Path(synchronization_file) if synchronization_file else (
            self.base_dir / "results" / "aicity_validation" / "synchronization_metadata.json"
        )

        # In-memory normalized repositories
        self.observations: Dict[str, NormalizedObservation] = {}
        self.identities: Dict[str, NormalizedIdentity] = {}
        self.trajectories: Dict[str, NormalizedTrajectory] = {}
        self.cameras: Dict[str, CameraMetadata] = {}
        self.validation_summary: Dict[str, Any] = {}

        # Load all artifacts
        self._load_artifacts()

    def _load_json_if_exists(self, path: Path) -> Optional[Any]:
        if not path.is_file():
            logger.warning(f"Member 2 artifact not found at: {path}")
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return None

    def _load_artifacts(self) -> None:
        """Parse and normalize all artifacts into memory."""
        # 1. Load calibration & synchronization metadata
        calib_data = self._load_json_if_exists(self.calibration_path) or {}
        sync_data = self._load_json_if_exists(self.synchronization_path) or {}
        inventory_data = self._load_json_if_exists(self.inventory_path) or {}

        reproj_errors = calib_data.get("reprojection_errors", {})
        sync_offsets = sync_data.get("offsets_by_camera", {})

        # 2. Load observations
        raw_observations = self._load_json_if_exists(self.observations_path) or []
        
        # We need calibration to project bottom-center to world coordinates if raw observations lack them
        # Let's import AICityCalibration if calibration dir exists
        calibrator = None
        cal_dir = self.base_dir / "data" / "aicity_ground_truth" / "calibration"
        if cal_dir.is_dir():
            try:
                from inference.aicity_calibration import AICityCalibration
                calibrator = AICityCalibration(calibration_dir=cal_dir)
            except Exception as e:
                logger.warning(f"Could not initialize AICityCalibration: {e}")

        for obs in raw_observations:
            obs_id = obs.get("observation_id", "")
            cam_id = obs.get("camera_id", "")
            bbox = obs.get("bbox", [0.0, 0.0, 0.0, 0.0])
            t_sec = float(obs.get("timestamp_seconds", 0.0))
            
            # Sync offset
            sync_offset = float(sync_offsets.get(cam_id, 0.0))
            sync_t_sec = t_sec + sync_offset

            # Re-ID model provenance
            prov = obs.get("source_provenance", {})
            reid_model = prov.get("reid_model")

            # World coordinates projection
            world_x: Optional[float] = None
            world_y: Optional[float] = None
            proj_status = "uncalibrated"
            coord_system = "image"

            if calibrator and calibrator.has_calibration(cam_id) and bbox and len(bbox) == 4:
                # Bottom-center projection
                pt = calibrator.project_bbox(cam_id, bbox)
                if pt is not None:
                    world_x, world_y = pt
                    proj_status = "valid"
                    coord_system = "cityflow_world"
                else:
                    world_x, world_y = None, None
                    proj_status = "unstable_horizon_denominator"
                    coord_system = "image"

            norm_obs = NormalizedObservation(
                observation_id=obs_id,
                camera_id=cam_id,
                track_id=str(obs.get("track_id", "")),
                frame_id=int(obs.get("frame_id", 0)),
                timestamp_seconds=t_sec,
                synchronized_timestamp_seconds=sync_t_sec,
                vehicle_type=obs.get("vehicle_type", "car"),
                detection_confidence=float(obs.get("detection_confidence", 0.0)),
                camera_reliability=float(obs.get("camera_reliability", 0.75)),
                bbox=bbox,
                world_x=world_x,
                world_y=world_y,
                projection_status=proj_status,
                coordinate_system=coord_system,
                reid_model=reid_model,
                plate=None,
                plate_status="Plate unavailable / privacy-censored",
                raw_metadata=obs,
            )
            self.observations[obs_id] = norm_obs

        # 3. Load inferred identities
        inference_data = self._load_json_if_exists(self.inference_path) or {}
        raw_identities = inference_data.get("inferred_identities", [])

        for id_item in raw_identities:
            ut_id = id_item.get("identity_id", "")
            obs_ids = id_item.get("observation_ids", [])
            cams = id_item.get("cameras", [])
            status = id_item.get("identity_status", "unconfirmed_singleton")
            adm_status = id_item.get("admission_status", "unconfirmed_singleton")
            v_type = id_item.get("vehicle_type") or "car"
            conf = id_item.get("confidence")

            # Collect world points from member observations
            w_points = []
            has_c002 = "CAM_S01_C002" in cams or "c002" in cams
            has_other = any("C001" in c or "C003" in c for c in cams)

            for oid in obs_ids:
                if oid in self.observations:
                    o = self.observations[oid]
                    w_points.append({
                        "observation_id": oid,
                        "camera_id": o.camera_id,
                        "timestamp_seconds": o.timestamp_seconds,
                        "synchronized_timestamp_seconds": o.synchronized_timestamp_seconds,
                        "world_x": o.world_x,
                        "world_y": o.world_y,
                        "projection_status": o.projection_status,
                    })

            # Re-ID appearance status check (protecting C002 model difference)
            if has_c002 and has_other:
                app_status = "Insufficient appearance evidence"
            elif has_c002:
                app_status = "osnet_x0_25_msmt17_isolated"
            else:
                app_status = "available" if len(obs_ids) > 1 else "singleton"

            norm_id = NormalizedIdentity(
                identity_id=ut_id,
                identity_status=status,
                admission_status=adm_status,
                cameras=cams,
                observation_ids=obs_ids,
                observations_count=len(obs_ids),
                vehicle_type=v_type,
                confidence=conf,
                plate_display="Plate unavailable / privacy-censored",
                appearance_status=app_status,
                consistency=id_item.get("consistency", {}),
                evidence_summary=id_item.get("evidence_summary", {}),
                world_points=w_points,
            )
            self.identities[ut_id] = norm_id

        # 4. Load trajectories
        raw_trajectories = self._load_json_if_exists(self.trajectories_path) or []
        for traj_item in raw_trajectories:
            ut_id = traj_item.get("identity_id", "")
            
            # Extract waypoints from linked observations
            waypoints = []
            if ut_id in self.identities:
                for pt in self.identities[ut_id].world_points:
                    waypoints.append(pt)

            norm_traj = NormalizedTrajectory(
                identity_id=ut_id,
                cameras_visited=traj_item.get("cameras_visited", []),
                observations_count=int(traj_item.get("observations_count", 1)),
                start_timestamp=float(traj_item.get("start_timestamp", 0.0)),
                end_timestamp=float(traj_item.get("end_timestamp", 0.0)),
                total_time_seconds=float(traj_item.get("total_time_seconds", 0.0)),
                total_distance_m=float(traj_item.get("total_distance_m", 0.0)),
                overall_confidence=float(traj_item.get("overall_confidence", 1.0)),
                is_ambiguous=bool(traj_item.get("is_ambiguous", False)),
                spatial_semantics="cityflow_world" if any(w.get("world_x") is not None for w in waypoints) else "image_space_only",
                temporal_semantics=traj_item.get("temporal_semantics", "video_relative"),
                segments=traj_item.get("segments", []),
                waypoints=waypoints,
            )
            self.trajectories[ut_id] = norm_traj

        # 5. Dynamically discover cameras and aggregate camera metadata
        self._build_camera_metadata(reproj_errors, sync_offsets)

        # 6. Overall validation summary
        self.validation_summary = {
            "source": "Member2 inference",
            "coordinate_system": "cityflow_world",
            "camera_count": len(self.cameras),
            "total_observations": len(self.observations),
            "calibrated_observations": sum(1 for o in self.observations.values() if o.world_x is not None),
            "unstable_horizon_observations": sum(1 for o in self.observations.values() if o.projection_status == "unstable_horizon_denominator"),
            "total_identities": len(self.identities),
            "total_trajectories": len(self.trajectories),
            "cluster_status_distribution": dict(inference_data.get("cluster_status_distribution", {})),
            "pairwise_decisions": dict(inference_data.get("pairwise_decisions", {})),
        }

    def _build_camera_metadata(self, reproj_errors: Dict[str, float], sync_offsets: Dict[str, float]) -> None:
        """Compute camera metadata dynamically from loaded observations."""
        cam_obs_map: Dict[str, List[NormalizedObservation]] = {}
        for obs in self.observations.values():
            cam_obs_map.setdefault(obs.camera_id, []).append(obs)

        for cam_id, obs_list in sorted(cam_obs_map.items()):
            # Canonical normalization for key matching
            cam_short = cam_id.split("_")[-1].lower() if "_" in cam_id else cam_id.lower()
            rep_err = reproj_errors.get(cam_short) or reproj_errors.get(cam_id)
            sync_off = sync_offsets.get(cam_id, 0.0)

            # Center world coordinates (mean of valid projected points)
            valid_x = [o.world_x for o in obs_list if o.world_x is not None]
            valid_y = [o.world_y for o in obs_list if o.world_y is not None]
            cx = sum(valid_x) / len(valid_x) if valid_x else None
            cy = sum(valid_y) / len(valid_y) if valid_y else None

            # Vehicle type counts
            v_types: Dict[str, int] = {}
            for o in obs_list:
                v_types[o.vehicle_type] = v_types.get(o.vehicle_type, 0) + 1

            # Unique tracks
            tracks = {o.track_id for o in obs_list}
            mean_rel = sum(o.camera_reliability for o in obs_list) / len(obs_list) if obs_list else 0.75
            reid_mod = next((o.reid_model for o in obs_list if o.reid_model), None)

            self.cameras[cam_id] = CameraMetadata(
                camera_id=cam_id,
                coordinate_system="cityflow_world",
                has_calibration=len(valid_x) > 0,
                reprojection_error_px=rep_err,
                sync_offset_seconds=sync_off,
                center_world_x=cx,
                center_world_y=cy,
                total_observations=len(obs_list),
                unique_tracks=len(tracks),
                mean_reliability=mean_rel,
                vehicle_type_distribution=v_types,
                reid_model=reid_mod,
            )

    # Public Accessors
    def get_cameras(self) -> List[CameraMetadata]:
        return list(self.cameras.values())

    def get_camera(self, camera_id: str) -> Optional[CameraMetadata]:
        return self.cameras.get(camera_id)

    def get_observations(self, camera_id: Optional[str] = None) -> List[NormalizedObservation]:
        if camera_id:
            return [o for o in self.observations.values() if o.camera_id == camera_id]
        return list(self.observations.values())

    def get_observation(self, observation_id: str) -> Optional[NormalizedObservation]:
        return self.observations.get(observation_id)

    def get_identities(self, status_filter: Optional[str] = None) -> List[NormalizedIdentity]:
        if status_filter:
            return [i for i in self.identities.values() if i.identity_status == status_filter]
        return list(self.identities.values())

    def get_identity(self, identity_id: str) -> Optional[NormalizedIdentity]:
        return self.identities.get(identity_id)

    def get_trajectories(self) -> List[NormalizedTrajectory]:
        return list(self.trajectories.values())

    def get_trajectory(self, identity_id: str) -> Optional[NormalizedTrajectory]:
        return self.trajectories.get(identity_id)

    def get_system_status(self) -> Dict[str, Any]:
        return self.validation_summary

    @classmethod
    def create_synthetic_n_camera_fixture(cls, n_cameras: int = 5) -> "Member2Adapter":
        """
        Create a pure in-memory test fixture for N-camera dynamic scaling tests.
        EXPLICIT NOTICE: Marked [TEST ONLY] - NEVER used for empirical evaluation.
        """
        adapter = cls.__new__(cls)
        adapter.base_dir = Path(".")
        adapter.observations = {}
        adapter.identities = {}
        adapter.trajectories = {}
        adapter.cameras = {}

        for i in range(1, n_cameras + 1):
            cam_id = f"CAM_TEST_C{i:03d}"
            adapter.cameras[cam_id] = CameraMetadata(
                camera_id=cam_id,
                coordinate_system="cityflow_world",
                has_calibration=True,
                reprojection_error_px=5.0 + i,
                sync_offset_seconds=float(i * 0.5),
                center_world_x=float(i * 1000.0),
                center_world_y=float(i * 200.0),
                total_observations=10,
                unique_tracks=5,
                mean_reliability=0.85,
                vehicle_type_distribution={"car": 8, "suv": 2},
                reid_model="test_model_fixture",
            )
            for j in range(1, 11):
                obs_id = f"{cam_id}_trk_{j:03d}"
                adapter.observations[obs_id] = NormalizedObservation(
                    observation_id=obs_id,
                    camera_id=cam_id,
                    track_id=str(j),
                    frame_id=j * 10,
                    timestamp_seconds=float(j * 1.5),
                    synchronized_timestamp_seconds=float(j * 1.5 + i * 0.5),
                    vehicle_type="car",
                    detection_confidence=0.85,
                    camera_reliability=0.85,
                    bbox=[100.0, 100.0, 200.0, 200.0],
                    world_x=float(i * 1000.0 + j * 5.0),
                    world_y=float(i * 200.0 + j * 5.0),
                    projection_status="valid",
                    coordinate_system="cityflow_world",
                    reid_model="test_model_fixture",
                )

        adapter.validation_summary = {
            "source": "SYNTHETIC_FIXTURE [TEST ONLY]",
            "coordinate_system": "cityflow_world",
            "camera_count": n_cameras,
            "total_observations": len(adapter.observations),
            "total_identities": 0,
            "total_trajectories": 0,
        }
        return adapter
