"""
UrbanTrack AI — Official AI City 2022 Track 1 Calibration & World-Coordinate Module.

Parses official CityFlowV2 / AI City homography calibration matrices and projects
image-space vehicle detections into planar CityFlow world coordinates (cityflow_world).
Guarantees:
- Zero fabricated GPS / latitude / longitude.
- Preserves raw bounding boxes and image centroids.
- Ground-contact point (bottom-center) projection.
- Numerical validation (finite coordinates, non-zero denominator).
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from schemas.observation_schema import Observation


def normalize_camera_key(camera_id: str) -> str:
    """Normalize camera key to both short (c001) and long (CAM_S01_C001) lookup forms."""
    cam_clean = camera_id.strip().lower()
    if "_" in cam_clean:
        parts = cam_clean.split("_")
        return parts[-1]
    return cam_clean


MIN_HOMOGRAPHY_DENOMINATOR = 0.50
MAX_WORLD_COORDINATE_BOUND = 25000.0


class AICityCalibration:
    """
    Authoritative calibration parser and world-coordinate transformer for AI City 2022 S01.
    """

    def __init__(
        self,
        calibration_dir: Union[str, Path] = "data/aicity_ground_truth/calibration",
    ) -> None:
        self.calibration_dir = Path(calibration_dir)
        self.homographies: Dict[str, List[List[float]]] = {}
        self.reprojection_errors: Dict[str, float] = {}
        self.file_hashes: Dict[str, str] = {}
        self._load_all()

    def _load_all(self) -> None:
        if not self.calibration_dir.is_dir():
            raise FileNotFoundError(f"Calibration directory not found: {self.calibration_dir}")

        for cal_file in sorted(self.calibration_dir.glob("*calibration*.txt")):
            cam_key = cal_file.name.split("_")[0].lower()  # e.g. c001
            content = cal_file.read_bytes()
            sha256 = hashlib.sha256(content).hexdigest()
            self.file_hashes[cam_key] = sha256

            text = content.decode("utf-8")
            lines = [l.strip() for l in text.splitlines() if l.strip()]

            # Parse Homography matrix
            h_line = next((l for l in lines if l.startswith("Homography matrix:")), None)
            if h_line:
                raw_matrix = h_line.replace("Homography matrix:", "").strip()
                rows = raw_matrix.split(";")
                H = [[float(v) for v in r.strip().split()] for r in rows if r.strip()]
                if len(H) == 3 and all(len(row) == 3 for row in H):
                    self.homographies[cam_key] = H
                    # Also register long canonical key
                    long_key = f"CAM_S01_C{cam_key[1:].upper()}"
                    self.homographies[long_key] = H

            # Parse Reprojection error
            err_line = next((l for l in lines if l.startswith("Reprojection error:")), None)
            if err_line:
                err_val = float(err_line.replace("Reprojection error:", "").strip())
                self.reprojection_errors[cam_key] = err_val
                long_key = f"CAM_S01_C{cam_key[1:].upper()}"
                self.reprojection_errors[long_key] = err_val

    def has_calibration(self, camera_id: str) -> bool:
        """Check if calibration matrix exists for this camera."""
        if camera_id in self.homographies:
            return True
        key = normalize_camera_key(camera_id)
        return key in self.homographies

    def project_contact_point(
        self,
        camera_id: str,
        image_x: float,
        image_y: float,
        min_denominator: float = MIN_HOMOGRAPHY_DENOMINATOR,
        max_world_bound: float = MAX_WORLD_COORDINATE_BOUND,
    ) -> Optional[Tuple[float, float]]:
        """
        Project 2D image coordinates (typically vehicle ground-contact bottom-center)
        into CityFlow planar world space via the official 3x3 homography matrix.

        Applies principled horizon safety:
        - Rejects near-zero projective denominator (|W| < min_denominator).
        - Rejects runaway coordinates exceeding realistic intersection bounds.
        - Never outputs NaN/Inf or fabricated coordinates.

        Returns (world_x, world_y) in CityFlow metric world units, or None if invalid.
        """
        H = self.homographies.get(camera_id) or self.homographies.get(normalize_camera_key(camera_id))
        if H is None:
            return None

        # Homogeneous transformation
        X = H[0][0] * image_x + H[0][1] * image_y + H[0][2]
        Y = H[1][0] * image_x + H[1][1] * image_y + H[1][2]
        W = H[2][0] * image_x + H[2][1] * image_y + H[2][2]

        # Task 2 Horizon Safety Guard: reject vanishing line singularities
        if abs(W) < min_denominator:
            return None

        world_x = X / W
        world_y = Y / W

        if math.isnan(world_x) or math.isnan(world_y) or math.isinf(world_x) or math.isinf(world_y):
            return None

        if abs(world_x) > max_world_bound or abs(world_y) > max_world_bound:
            return None

        return (round(world_x, 3), round(world_y, 3))

    def project_bbox(
        self,
        camera_id: str,
        bbox: List[float] | Tuple[float, float, float, float],
        min_denominator: float = MIN_HOMOGRAPHY_DENOMINATOR,
        max_world_bound: float = MAX_WORLD_COORDINATE_BOUND,
    ) -> Optional[Tuple[float, float]]:
        """
        Project vehicle ground-contact point (bottom-center of bounding box):
        x = (x1 + x2) / 2.0, y = y2
        """
        if not bbox or len(bbox) != 4:
            return None
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        contact_y = float(y2)
        return self.project_contact_point(
            camera_id, cx, contact_y,
            min_denominator=min_denominator,
            max_world_bound=max_world_bound,
        )

    def attach_calibration(
        self,
        observations: List[Observation],
        min_denominator: float = MIN_HOMOGRAPHY_DENOMINATOR,
        max_world_bound: float = MAX_WORLD_COORDINATE_BOUND,
    ) -> List[Observation]:
        """
        Attach CityFlow world coordinates to observations using official calibration.

        Guarantees:
        - raw bbox and centroid are untouched.
        - valid world projections populate trajectory_point with [world_x, world_y].
        - unstable/horizon observations preserve original image coordinates.
        - projection_status is recorded ('valid' vs 'unstable_horizon_denominator').
        - latitude and longitude remain None (zero fabricated GPS).
        - Reprojection error and calibration provenance are recorded.
        """
        for obs in observations:
            if not self.has_calibration(obs.camera_id):
                continue

            world_pt = None
            if obs.bbox and len(obs.bbox) == 4:
                world_pt = self.project_bbox(
                    obs.camera_id, obs.bbox,
                    min_denominator=min_denominator,
                    max_world_bound=max_world_bound,
                )
            elif obs.trajectory_point and len(obs.trajectory_point) == 2:
                world_pt = self.project_contact_point(
                    obs.camera_id, obs.trajectory_point[0], obs.trajectory_point[1],
                    min_denominator=min_denominator,
                    max_world_bound=max_world_bound,
                )

            cam_norm = normalize_camera_key(obs.camera_id)
            rep_err = self.reprojection_errors.get(cam_norm)

            if world_pt is not None:
                wx, wy = world_pt
                obs.trajectory_point = [wx, wy]
                obs.point_coordinate_system = "cityflow_world"
                obs.point_type = "cityflow_world_ground_contact"
                setattr(obs, "world_x", wx)
                setattr(obs, "world_y", wy)
                setattr(obs, "projection_status", "valid")

                if obs.source_provenance is None:
                    obs.source_provenance = {}
                obs.source_provenance["calibration"] = {
                    "coordinate_system": "cityflow_world",
                    "homography_source": f"c{cam_norm[1:]}_calibration.txt",
                    "file_sha256": self.file_hashes.get(cam_norm),
                    "reprojection_error_px": rep_err,
                    "world_coordinates": [wx, wy],
                    "contact_point_convention": "bottom_center",
                    "projection_status": "valid",
                }
            else:
                # Horizon safety guard: preserve raw image observation, mark world projection invalid/uncertain
                setattr(obs, "world_x", None)
                setattr(obs, "world_y", None)
                setattr(obs, "projection_status", "unstable_horizon_denominator")
                if obs.source_provenance is None:
                    obs.source_provenance = {}
                obs.source_provenance["calibration"] = {
                    "coordinate_system": "image",
                    "homography_source": f"c{cam_norm[1:]}_calibration.txt",
                    "file_sha256": self.file_hashes.get(cam_norm),
                    "reprojection_error_px": rep_err,
                    "projection_status": "unstable_horizon_denominator",
                    "explanation": "Projective denominator near vanishing line (|W| < epsilon or bounds exceeded); world coordinates suppressed for physical safety.",
                }

        return observations

    def to_dict(self) -> Dict[str, Any]:
        """Export calibration summary for diagnostics and provenance artifacts."""
        return {
            "calibration_dir": str(self.calibration_dir),
            "cameras_calibrated": sorted(list(self.file_hashes.keys())),
            "reprojection_errors": {
                k: v for k, v in self.reprojection_errors.items() if not k.startswith("CAM_")
            },
            "file_hashes": self.file_hashes,
        }
