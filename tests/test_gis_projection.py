"""
Unit tests for UrbanTrack GIS Projection Utility (Phase 1)
"""

import math
import sys
from pathlib import Path
import numpy as np
import pytest

# Ensure scripts/ directory is on sys.path
scripts_dir = Path(__file__).resolve().parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from project_to_gis import (
    GISProjector,
    GISProjectionError,
    load_calibration_homography,
    project_pixel_to_gis,
    project_trajectory_to_gis,
    project_image_boundary_to_gis,
    generate_camera_fov_geojson,
    convert_trajectory_to_geojson
)


@pytest.fixture
def sample_s01_calibration(tmp_path):
    """Fixture providing a mock S01/c001 calibration file."""
    calib_file = tmp_path / "calibration.txt"
    content = (
        "Homography matrix: -33.391316152634239 -24.645266281592367 -815.911776622374418;"
        "-3.187500816161974 0.400210599253526 171.861512099508360;"
        "-0.016515000037136 0.003281226883010 1.000000000000000\n"
        "Reprojection error: 5.499888905989678\n"
    )
    calib_file.write_text(content, encoding="utf-8")
    return calib_file


def test_homography_loading_and_inversion(sample_s01_calibration):
    projector = GISProjector(calibration_path=sample_s01_calibration)
    assert projector.H is not None
    assert projector.H.shape == (3, 3)
    assert projector.H_inv is not None
    assert projector.H_inv.shape == (3, 3)
    # Check that H @ H_inv is identity
    eye = projector.H @ projector.H_inv
    assert np.allclose(eye, np.eye(3), atol=1e-7)


def test_pixel_projection_coordinate_order_and_bounds(sample_s01_calibration):
    projector = GISProjector(calibration_path=sample_s01_calibration)
    res = projector.project_pixel(960, 540)

    assert "latitude" in res
    assert "longitude" in res
    lat = res["latitude"]
    lon = res["longitude"]

    # WGS84 bounds
    assert -90.0 <= lat <= 90.0
    assert -180.0 <= lon <= 180.0

    # Plausible for Dubuque, IA S01 (approx 42.525, -90.723)
    assert math.isclose(lat, 42.525655, abs_tol=1e-3)
    assert math.isclose(lon, -90.723457, abs_tol=1e-3)


def test_singular_homography_error():
    singular_H = np.zeros((3, 3))
    with pytest.raises(GISProjectionError):
        GISProjector(H=singular_H)


def test_invalid_pixel_error(sample_s01_calibration):
    projector = GISProjector(calibration_path=sample_s01_calibration)
    with pytest.raises(GISProjectionError):
        projector.project_pixel(float("nan"), 540)
    with pytest.raises(GISProjectionError):
        projector.project_pixel(960, float("inf"))


def test_trajectory_bottom_center_projection(sample_s01_calibration):
    projector = GISProjector(calibration_path=sample_s01_calibration)

    # Observation with [x, y, w, h] format
    # Bottom center is (x + w/2, y + h) = (1000 + 100/2, 500 + 40) = (1050, 540)
    obs_list = [
        {"frame": 1, "bbox": [1000, 500, 100, 40]}
    ]
    traj = projector.project_trajectory(obs_list)
    assert len(traj) == 1
    assert traj[0]["frame"] == 1

    expected_coords = projector.project_pixel(1050, 540)
    assert math.isclose(traj[0]["latitude"], expected_coords["latitude"], abs_tol=1e-7)
    assert math.isclose(traj[0]["longitude"], expected_coords["longitude"], abs_tol=1e-7)


def test_image_boundary_polygon(sample_s01_calibration):
    projector = GISProjector(calibration_path=sample_s01_calibration)
    poly = projector.project_image_boundary(1920, 1080)
    assert len(poly) == 4
    for pt in poly:
        assert len(pt) == 2
        lat, lon = pt
        assert -90.0 <= lat <= 90.0
        assert -180.0 <= lon <= 180.0


def test_geojson_rfc7946_ordering(tmp_path, sample_s01_calibration):
    out_fov = tmp_path / "test_fov.geojson"
    geojson = generate_camera_fov_geojson(
        camera_id="CAM_S01_C001",
        scene="S01",
        calibration_path=sample_s01_calibration,
        output_path=out_fov
    )

    assert geojson["type"] == "Feature"
    assert geojson["geometry"]["type"] == "Polygon"
    coords = geojson["geometry"]["coordinates"][0]

    # Linear ring must close (first == last)
    assert coords[0] == coords[-1]
    assert len(coords) == 5

    # RFC 7946: [longitude, latitude]
    for lon, lat in coords:
        assert -180.0 <= lon <= 180.0
        assert -90.0 <= lat <= 90.0
        # For Dubuque: lon is negative (~ -90.7), lat is positive (~ 42.5)
        assert lon < 0.0
        assert lat > 0.0

    assert geojson["properties"]["coordinate_system"] == "WGS84"
    assert geojson["properties"]["projection_method"] == "inverse_homography"
