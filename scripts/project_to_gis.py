"""
UrbanTrack AI - GIS Projection Utility (Phase 1)
=================================================
Provides mathematically validated transformations from camera pixel coordinates (u, v)
to geographic WGS84 ground-plane coordinates (latitude, longitude) using camera homography.

Dataset Context: AI City Challenge 2022 Track 1 MTMC (CityFlowV2)

CRITICAL COORDINATE CONVENTIONS:
- Internal Python APIs return dicts or tuples formatted as:
    latitude = first coordinate
    longitude = second coordinate
- GeoJSON outputs strictly follow RFC 7946:
    [longitude, latitude]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


class GISProjectionError(Exception):
    """Exception raised when coordinate projection fails numerical or geographic validation."""
    pass


class GISProjector:
    """
    Projector for transforming image pixel coordinates to WGS84 ground-plane coordinates.
    """

    def __init__(self, calibration_path: Optional[Union[str, Path]] = None, H: Optional[np.ndarray] = None):
        """
        Initialize the GIS projector with either a calibration file path or a 3x3 homography matrix.
        """
        self.calibration_path: Optional[Path] = Path(calibration_path) if calibration_path else None
        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None

        if H is not None:
            self.set_homography(H)
        elif self.calibration_path is not None:
            self.load_calibration(self.calibration_path)

    def load_calibration(self, calibration_path: Union[str, Path]) -> np.ndarray:
        """
        Parse calibration.txt and compute the matrix inverse.
        """
        calib_file = Path(calibration_path)
        if not calib_file.exists():
            raise FileNotFoundError(f"Calibration file not found: {calib_file}")

        h_matrix = None
        with open(calib_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Homography matrix:"):
                    raw_vals = line.split("Homography matrix:")[1].strip()
                    rows = raw_vals.split(";")
                    matrix_rows = []
                    for row in rows:
                        cols = [float(x) for x in row.split() if x.strip()]
                        if len(cols) != 3:
                            raise ValueError(f"Malformed homography row in {calib_file}: '{row}'")
                        matrix_rows.append(cols)
                    if len(matrix_rows) != 3:
                        raise ValueError(f"Homography matrix must have 3 rows in {calib_file}")
                    h_matrix = np.array(matrix_rows, dtype=np.float64)
                    break

        if h_matrix is None:
            raise ValueError(f"Could not find 'Homography matrix:' in {calib_file}")

        self.calibration_path = calib_file
        self.set_homography(h_matrix)
        return self.H

    def set_homography(self, H: np.ndarray) -> None:
        """
        Validate and set the homography matrix and compute its inverse.
        """
        if H.shape != (3, 3):
            raise ValueError(f"Homography matrix must be 3x3, got shape {H.shape}")

        det = np.linalg.det(H)
        if math.isclose(det, 0.0, abs_tol=1e-12) or np.isnan(det) or np.isinf(det):
            raise GISProjectionError(f"Homography matrix is singular or non-invertible (det={det})")

        self.H = H.astype(np.float64)
        self.H_inv = np.linalg.inv(self.H)

    def project_pixel(self, u: float, v: float) -> Dict[str, float]:
        """
        Project an image pixel coordinate (u, v) onto the calibrated ground plane.

        Transformation:
            [lat, lon, 1]^T ~ H^-1 [u, v, 1]^T

        Returns:
            {"latitude": float, "longitude": float}
        """
        if self.H_inv is None:
            raise GISProjectionError("Projector has not been initialized with a valid homography matrix.")

        if not (math.isfinite(u) and math.isfinite(v)):
            raise GISProjectionError(f"Invalid pixel coordinates: u={u}, v={v}")

        pt_homogeneous = np.array([float(u), float(v), 1.0], dtype=np.float64)
        res = self.H_inv @ pt_homogeneous

        w = res[2]
        if math.isclose(w, 0.0, abs_tol=1e-12) or not math.isfinite(w):
            raise GISProjectionError(f"Division by zero or invalid scale factor w={w} projecting pixel ({u}, {v})")

        lat = res[0] / w
        lon = res[1] / w

        # Geographic range validation
        if not (math.isfinite(lat) and math.isfinite(lon)):
            raise GISProjectionError(f"Non-finite coordinates projected from ({u}, {v}): lat={lat}, lon={lon}")

        if not (-90.0 <= lat <= 90.0):
            raise GISProjectionError(f"Latitude out of WGS84 range [-90, 90]: {lat} from pixel ({u}, {v})")

        if not (-180.0 <= lon <= 180.0):
            raise GISProjectionError(f"Longitude out of WGS84 range [-180, 180]: {lon} from pixel ({u}, {v})")

        return {
            "latitude": float(lat),
            "longitude": float(lon)
        }

    def project_trajectory(self, observations: List[Any]) -> List[Dict[str, Any]]:
        """
        Accept a sequence of vehicle observations and project bottom-center points to GIS.

        Supported observation formats:
        - Dict with:
            "frame" / "frame_number": int
            "bbox": [x, y, width, height] OR [x1, y1, x2, y2]
            OR "centroid": [u, v]
        - Tuple/list:
            (frame, bbox) where bbox is [x, y, w, h] or [x1, y1, x2, y2]
            OR (frame, u, v)

        Bottom-center calculation:
            u = x + width / 2.0
            v = y + height

        Returns:
            List of dicts: [{"frame": int, "latitude": float, "longitude": float}, ...]
        """
        projected_trajectory: List[Dict[str, Any]] = []

        for obs in observations:
            frame_num = None
            u = None
            v = None

            if isinstance(obs, dict):
                frame_num = obs.get("frame")
                if frame_num is None:
                    frame_num = obs.get("frame_number", 0)

                if "bbox" in obs and obs["bbox"] is not None:
                    bbox = obs["bbox"]
                    if len(bbox) == 4:
                        # Determine if [x, y, w, h] or [x1, y1, x2, y2]
                        # UrbanTrack observations.json uses [x1, y1, x2, y2]
                        # MOT format uses [left, top, w, h]
                        # If bbox has explicit centroid or if x2 > x1:
                        if "centroid" in obs and obs["centroid"] is not None:
                            u, v = float(obs["centroid"][0]), float(obs["centroid"][1])
                        else:
                            b0, b1, b2, b3 = [float(x) for x in bbox]
                            # If b2 and b3 look like coordinates rather than width/height (b2 > b0 and b3 > b1)
                            # UrbanTrack standard: x1, y1, x2, y2
                            if b2 > b0 and b3 > b1 and b2 <= 3840 and b3 <= 2160 and (b2 - b0) < 1920:
                                # We can support width/height or x1, y1, x2, y2
                                # If width is given: u = x + width / 2, v = y + height
                                # Check if 'width' / 'height' are dict keys
                                if "width" in obs and "height" in obs:
                                    u = float(obs.get("x", b0)) + float(obs["width"]) / 2.0
                                    v = float(obs.get("y", b1)) + float(obs["height"])
                                else:
                                    # Interpret as [x1, y1, x2, y2]
                                    u = (b0 + b2) / 2.0
                                    v = b3
                            else:
                                # Standard MOT format: [x, y, w, h]
                                u = b0 + b2 / 2.0
                                v = b1 + b3
                elif "centroid" in obs and obs["centroid"] is not None:
                    u, v = float(obs["centroid"][0]), float(obs["centroid"][1])
                elif "u" in obs and "v" in obs:
                    u, v = float(obs["u"]), float(obs["v"])

            elif isinstance(obs, (list, tuple)):
                if len(obs) == 2:
                    frame_num = int(obs[0])
                    bbox_or_pt = obs[1]
                    if len(bbox_or_pt) == 4:
                        b0, b1, b2, b3 = [float(x) for x in bbox_or_pt]
                        # Assume MOT format [x, y, w, h] unless specified
                        u = b0 + b2 / 2.0
                        v = b1 + b3
                    elif len(bbox_or_pt) == 2:
                        u, v = float(bbox_or_pt[0]), float(bbox_or_pt[1])
                elif len(obs) == 3:
                    frame_num = int(obs[0])
                    u = float(obs[1])
                    v = float(obs[2])

            if frame_num is None or u is None or v is None:
                continue

            try:
                coords = self.project_pixel(u, v)
                projected_trajectory.append({
                    "frame": int(frame_num),
                    "latitude": coords["latitude"],
                    "longitude": coords["longitude"]
                })
            except GISProjectionError:
                # Omit points outside valid ground projection
                continue

        return projected_trajectory

    def project_image_boundary(self, width: int = 1920, height: int = 1080) -> List[List[float]]:
        """
        Project the 4 corners of an image of size (width, height) to WGS84 ground plane.

        Corners evaluated:
            (0, 0)
            (width, 0)
            (width, height)
            (0, height)

        Returns:
            [
                [lat, lon],
                [lat, lon],
                [lat, lon],
                [lat, lon]
            ]
        """
        corners = [
            (0.0, 0.0),
            (float(width), 0.0),
            (float(width), float(height)),
            (0.0, float(height))
        ]

        polygon: List[List[float]] = []
        for u, v in corners:
            coords = self.project_pixel(u, v)
            polygon.append([coords["latitude"], coords["longitude"]])

        return polygon


# Standalone functional API matching Task specifications

def load_calibration_homography(calibration_path: Union[str, Path]) -> np.ndarray:
    """Load and parse the 3x3 homography matrix from calibration.txt."""
    projector = GISProjector(calibration_path=calibration_path)
    return projector.H


def project_pixel_to_gis(
    u: float,
    v: float,
    H: Optional[np.ndarray] = None,
    H_inv: Optional[np.ndarray] = None,
    calibration_path: Optional[Union[str, Path]] = None
) -> Dict[str, float]:
    """
    Project image pixel coordinates (u, v) onto WGS84 ground plane using H^-1.

    Output format:
        {
            "latitude": float,
            "longitude": float
        }
    """
    projector = GISProjector(calibration_path=calibration_path, H=H)
    if H_inv is not None:
        projector.H_inv = H_inv
    return projector.project_pixel(u, v)


def project_trajectory_to_gis(
    observations: List[Any],
    H: Optional[np.ndarray] = None,
    calibration_path: Optional[Union[str, Path]] = None
) -> List[Dict[str, Any]]:
    """
    Project a sequence of vehicle observations to a trajectory of (frame, latitude, longitude).
    """
    projector = GISProjector(calibration_path=calibration_path, H=H)
    return projector.project_trajectory(observations)


def project_image_boundary_to_gis(
    H: Optional[np.ndarray] = None,
    H_inv: Optional[np.ndarray] = None,
    calibration_path: Optional[Union[str, Path]] = None,
    width: int = 1920,
    height: int = 1080
) -> List[List[float]]:
    """
    Project the four image corners to ground plane WGS84 coordinates.
    Returns:
        [
            [lat, lon],
            [lat, lon],
            [lat, lon],
            [lat, lon]
        ]
    """
    projector = GISProjector(calibration_path=calibration_path, H=H)
    if H_inv is not None:
        projector.H_inv = H_inv
    return projector.project_image_boundary(width=width, height=height)


def generate_camera_fov_geojson(
    camera_id: str,
    scene: str,
    calibration_path: Union[str, Path],
    width: int = 1920,
    height: int = 1080,
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generate an RFC 7946 compliant GeoJSON Polygon feature representing camera FOV ground footprint.

    Note: GeoJSON coordinates are strictly [longitude, latitude].
    """
    projector = GISProjector(calibration_path=calibration_path)
    poly_lat_lon = projector.project_image_boundary(width=width, height=height)

    # Convert [lat, lon] to GeoJSON RFC 7946 standard: [longitude, latitude]
    # Linear rings must close by repeating the first vertex
    coordinates_lon_lat = [[pt[1], pt[0]] for pt in poly_lat_lon]
    coordinates_lon_lat.append([poly_lat_lon[0][1], poly_lat_lon[0][0]])

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coordinates_lon_lat]
        },
        "properties": {
            "camera_id": camera_id,
            "scene": scene,
            "source_calibration": str(calibration_path),
            "coordinate_system": "WGS84",
            "projection_method": "inverse_homography",
            "image_dimensions": [width, height],
            "note": "Represents the 4-corner ground-plane projection footprint; may exceed physical road surface."
        }
    }

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(feature, f, indent=2)

    return feature


def convert_trajectory_to_geojson(
    camera_id: str,
    track_id: Union[int, str],
    trajectory_points: List[Tuple[float, float]],
    calibration_path: Union[str, Path],
    start_frame: int = 0,
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Convert an UrbanTrack trajectory (list of [x, y] pixel coordinates) into an RFC 7946 GeoJSON LineString.

    GeoJSON coordinates are strictly [longitude, latitude].
    """
    projector = GISProjector(calibration_path=calibration_path)

    line_coordinates_lon_lat = []
    frame_metadata = []

    for idx, pt in enumerate(trajectory_points):
        u, v = float(pt[0]), float(pt[1])
        try:
            coords = projector.project_pixel(u, v)
            line_coordinates_lon_lat.append([coords["longitude"], coords["latitude"]])
            frame_metadata.append(start_frame + idx)
        except GISProjectionError:
            continue

    if len(line_coordinates_lon_lat) < 2:
        # LineString requires at least 2 positions
        # If 1 point, duplicate it or create point
        if len(line_coordinates_lon_lat) == 1:
            line_coordinates_lon_lat.append(line_coordinates_lon_lat[0])
        else:
            line_coordinates_lon_lat = []

    feature = {
        "type": "Feature",
        "geometry": {
            "type": "LineString",
            "coordinates": line_coordinates_lon_lat
        },
        "properties": {
            "camera_id": camera_id,
            "track_id": track_id,
            "coordinate_system": "WGS84",
            "projection_method": "inverse_homography",
            "point_count": len(line_coordinates_lon_lat),
            "frame_range": [start_frame, start_frame + len(trajectory_points) - 1] if trajectory_points else []
        }
    }

    if output_path:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(feature, f, indent=2)

    return feature


def resolve_manifest_camera(camera_query: str, manifest_path: Path) -> Optional[Dict[str, Any]]:
    """Helper to locate a camera entry in aicity_manifest.json."""
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    query_norm = camera_query.strip().lower().replace("/", "_").replace("\\", "_")
    dataset_root = Path(data.get("dataset_root", ""))

    for cam in data.get("cameras", []):
        cam_id = cam.get("camera_id", "").lower()
        cam_name = cam.get("camera", "").lower()
        scene = cam.get("scene", "").lower()
        combo = f"{scene}_{cam_name}"

        if query_norm in (cam_id, cam_name, combo):
            calib_rel = cam.get("calibration_path_relative", "")
            calib_full = dataset_root / calib_rel
            return {
                "camera_id": cam.get("camera_id"),
                "scene": cam.get("scene"),
                "camera": cam.get("camera"),
                "calibration_path": str(calib_full),
                "dataset_root": str(dataset_root)
            }
    return None


def main():
    parser = argparse.ArgumentParser(description="UrbanTrack GIS Projection CLI (Phase 1)")
    parser.add_argument("--test-camera", type=str, default="S01/c001",
                        help="Camera identifier to test (e.g. S01/c001 or CAM_S01_C001)")
    parser.add_argument("--calibration-file", type=str, default=None,
                        help="Direct path to calibration.txt (overrides manifest)")
    parser.add_argument("--generate-fov", action="store_true",
                        help="Generate GeoJSON FOV footprint for the test camera")
    parser.add_argument("--generate-trajectories", action="store_true",
                        help="Convert existing UrbanTrack trajectories to GeoJSON for the camera")
    parser.add_argument("--all-fov", action="store_true",
                        help="Generate GeoJSON FOV footprint for all cameras in manifest")

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    manifest_path = project_root / "data" / "config" / "aicity_manifest.json"

    # Default fallback path for S01/c001 if manifest path is not found directly
    default_s01_c001_calib = Path(r"C:\Users\kanis\Downloads\AICity22_Track1_MTMC_Tracking\train\S01\c001\calibration.txt")

    print("==================================================")
    print("URBANTRACK GIS PROJECTION UTILITY (PHASE 1)")
    print("==================================================")

    # 1. Resolve camera
    cam_info = None
    calib_path = None

    if args.calibration_file:
        calib_path = Path(args.calibration_file)
        cam_info = {
            "camera_id": args.test_camera,
            "scene": "S01",
            "camera": "c001",
            "calibration_path": str(calib_path)
        }
    else:
        cam_info = resolve_manifest_camera(args.test_camera, manifest_path)
        if cam_info:
            calib_path = Path(cam_info["calibration_path"])
        elif default_s01_c001_calib.exists():
            calib_path = default_s01_c001_calib
            cam_info = {
                "camera_id": "CAM_S01_C001",
                "scene": "S01",
                "camera": "c001",
                "calibration_path": str(calib_path)
            }

    if not calib_path or not calib_path.exists():
        print(f"Error: Calibration file could not be resolved for {args.test_camera}")
        sys.exit(1)

    print(f"Camera ID:        {cam_info['camera_id']}")
    print(f"Scene:            {cam_info['scene']}")
    print(f"Calibration File: {calib_path}")
    print("--------------------------------------------------")

    # Load projector
    projector = GISProjector(calibration_path=calib_path)
    print("Loaded 3x3 Homography Matrix (H):")
    print(projector.H)
    print("\nComputed Inverse Homography (H^-1):")
    print(projector.H_inv)
    print("--------------------------------------------------")

    # TASK 6 Execution: Test standard points
    test_points = [
        (960, 540),
        (960, 1080),
        (0, 1080),
        (1920, 1080)
    ]

    print("TASK 6: Projecting Test Pixel Coordinates (u, v) -> (lat, lon):")
    for u, v in test_points:
        coords = projector.project_pixel(u, v)
        print(f"Pixel ({u:>4}, {v:>4})")
        print(f"    -> lat={coords['latitude']:.6f}")
        print(f"    -> lon={coords['longitude']:.6f}")

    print("--------------------------------------------------")

    # TASK 3 Execution: Image boundary footprint
    boundary_poly = projector.project_image_boundary(width=1920, height=1080)
    print("TASK 3: Image Boundary Ground Footprint Polygon [lat, lon]:")
    for idx, pt in enumerate(boundary_poly):
        corner_names = ["Top-Left (0,0)", "Top-Right (W,0)", "Bottom-Right (W,H)", "Bottom-Left (0,H)"]
        print(f"  Corner {idx+1} [{corner_names[idx]}]: lat={pt[0]:.6f}, lon={pt[1]:.6f}")

    print("--------------------------------------------------")

    # TASK 4 Execution: GeoJSON FOV generation
    if args.generate_fov or not (args.generate_trajectories or args.all_fov):
        out_fov_path = project_root / "data" / "gis" / "fov" / f"{cam_info['camera_id']}.geojson"
        fov_geojson = generate_camera_fov_geojson(
            camera_id=cam_info["camera_id"],
            scene=cam_info["scene"],
            calibration_path=calib_path,
            output_path=out_fov_path
        )
        print(f"TASK 4: Generated Camera FOV GeoJSON -> {out_fov_path}")
        print(f"  First 2 coordinates [lon, lat]: {fov_geojson['geometry']['coordinates'][0][:2]}")

    # TASK 5 Execution: Convert existing trajectories to GeoJSON
    if args.generate_trajectories:
        traj_input_file = project_root / "data" / "output" / cam_info["camera_id"] / "trajectories.json"
        if traj_input_file.exists():
            with open(traj_input_file, "r", encoding="utf-8") as f:
                traj_list = json.load(f)
            converted_count = 0
            for item in traj_list:
                track_id = item.get("track_id")
                pts = item.get("trajectory", [])
                start_f = item.get("start_frame", 0)
                out_traj_path = project_root / "data" / "gis" / "trajectories" / cam_info["camera_id"] / f"{track_id}.geojson"
                convert_trajectory_to_geojson(
                    camera_id=cam_info["camera_id"],
                    track_id=track_id,
                    trajectory_points=pts,
                    calibration_path=calib_path,
                    start_frame=start_f,
                    output_path=out_traj_path
                )
                converted_count += 1
            print(f"TASK 5: Converted {converted_count} tracks to GeoJSON in: data/gis/trajectories/{cam_info['camera_id']}/")
        else:
            print(f"Note: No existing trajectories.json found at {traj_input_file}")

    # All FOV generation
    if args.all_fov and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)
        ds_root = Path(manifest_data.get("dataset_root", ""))
        generated = 0
        for cam in manifest_data.get("cameras", []):
            cid = cam.get("camera_id")
            sc = cam.get("scene")
            c_rel = cam.get("calibration_path_relative")
            c_full = ds_root / c_rel
            if c_full.exists():
                out_p = project_root / "data" / "gis" / "fov" / f"{cid}.geojson"
                try:
                    generate_camera_fov_geojson(cid, sc, c_full, output_path=out_p)
                    generated += 1
                except Exception as e:
                    print(f"Warning: Failed to generate FOV for {cid}: {e}")
        print(f"Generated FOV GeoJSON for {generated} cameras in data/gis/fov/")


if __name__ == "__main__":
    main()
