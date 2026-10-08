"""
UrbanTrack AI - GIS Projection Visualization (Phase 1)
======================================================
Generates an interactive Folium HTML map visualizing:
1. Scenario S01 approximate center GPS anchor (ReadMe.txt)
2. Camera S01/c001 projected ground-plane FOV polygon
3. Projected sample pixel points on the road surface
4. Sample vehicle ground-plane trajectory line

IMPORTANT CONSTRAINTS:
- Does NOT add fake camera pole markers.
- Does NOT pretend scenario anchor is camera position.
- Uses WGS84 coordinates directly derived from inverse homography.
"""

from pathlib import Path
import json
import folium
from folium import plugins

from project_to_gis import GISProjector, project_image_boundary_to_gis


def create_s01_visualization(output_html: Path):
    # Scenario S01 reference anchor from dataset ReadMe.txt
    s01_anchor = {"latitude": 42.525678, "longitude": -90.723601}

    # Calibration path for S01/c001
    project_root = Path(__file__).resolve().parent.parent
    calib_file = Path(r"C:\Users\kanis\Downloads\AICity22_Track1_MTMC_Tracking\train\S01\c001\calibration.txt")

    if not calib_file.exists():
        # Try manifest relative
        manifest_p = project_root / "data" / "config" / "aicity_manifest.json"
        if manifest_p.exists():
            with open(manifest_p, "r", encoding="utf-8") as f:
                m = json.load(f)
            calib_file = Path(m["dataset_root"]) / "train" / "S01" / "c001" / "calibration.txt"

    projector = GISProjector(calibration_path=calib_file)

    # 1. Initialize Map centered on Scenario S01 Anchor
    m = folium.Map(
        location=[s01_anchor["latitude"], s01_anchor["longitude"]],
        zoom_start=18,
        tiles="OpenStreetMap",
        control_scale=True
    )

    # Optional high-res tile layer
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery",
        name="Esri Satellite"
    ).add_to(m)

    # 2. Add Scenario Anchor Marker (Strictly labeled as Scenario Center, NOT Camera Position)
    folium.Marker(
        location=[s01_anchor["latitude"], s01_anchor["longitude"]],
        popup=folium.Popup(
            "<b>Scenario S01 Approximate Center Anchor</b><br>"
            "Source: ReadMe.txt line 42<br>"
            f"Lat: {s01_anchor['latitude']:.6f}<br>Lon: {s01_anchor['longitude']:.6f}<br>"
            "<i>(Note: This is an intersection anchor, NOT a camera pole location)</i>",
            max_width=300
        ),
        tooltip="Scenario S01 GPS Anchor (Center)",
        icon=folium.Icon(color="darkblue", icon="info-sign")
    ).add_to(m)

    # 3. Add Projected FOV Footprint Polygon
    # Folium expects [ [lat, lon], [lat, lon], ... ]
    boundary_lat_lon = projector.project_image_boundary(width=1920, height=1080)

    folium.Polygon(
        locations=boundary_lat_lon,
        color="#2b82cb",
        weight=2,
        fill=True,
        fill_color="#3388ff",
        fill_opacity=0.25,
        popup=folium.Popup(
            "<b>CAM_S01_C001 Projected Ground-Plane FOV</b><br>"
            "Method: H^-1 projection of 4 image corners<br>"
            "<i>Footprint projected on road surface plane</i>",
            max_width=300
        ),
        tooltip="CAM_S01_C001 Ground FOV Footprint"
    ).add_to(m)

    # 4. Add Sample Projected Points
    sample_points = [
        {"name": "Image Center (960, 540)", "u": 960, "v": 540, "color": "green"},
        {"name": "Image Bottom-Center (960, 1080)", "u": 960, "v": 1080, "color": "orange"},
        {"name": "Image Bottom-Left (0, 1080)", "u": 0, "v": 1080, "color": "purple"},
        {"name": "Image Bottom-Right (1920, 1080)", "u": 1920, "v": 1080, "color": "cadetblue"},
    ]

    for pt in sample_points:
        coords = projector.project_pixel(pt["u"], pt["v"])
        folium.CircleMarker(
            location=[coords["latitude"], coords["longitude"]],
            radius=6,
            color=pt["color"],
            fill=True,
            fill_color=pt["color"],
            fill_opacity=0.8,
            popup=folium.Popup(
                f"<b>{pt['name']}</b><br>"
                f"Pixel: ({pt['u']}, {pt['v']})<br>"
                f"Lat: {coords['latitude']:.6f}<br>Lon: {coords['longitude']:.6f}",
                max_width=250
            ),
            tooltip=f"{pt['name']}: [{coords['latitude']:.5f}, {coords['longitude']:.5f}]"
        ).add_to(m)

    # 5. Add Sample Vehicle Trajectory (Track 1) if available
    traj_geojson_path = project_root / "data" / "gis" / "trajectories" / "CAM_S01_C001" / "1.geojson"
    if traj_geojson_path.exists():
        with open(traj_geojson_path, "r", encoding="utf-8") as f:
            traj_data = json.load(f)
        # GeoJSON is [lon, lat], folium expects [lat, lon]
        traj_coords = [[p[1], p[0]] for p in traj_data["geometry"]["coordinates"]]
        if traj_coords:
            folium.PolyLine(
                locations=traj_coords,
                color="red",
                weight=4,
                opacity=0.85,
                tooltip="Vehicle Track #1 Ground Trajectory",
                popup="Vehicle Track #1 (Projected bottom-center contacts)"
            ).add_to(m)

    folium.LayerControl().add_to(m)

    output_html.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_html))
    print(f"Visualization map successfully saved to: {output_html}")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    vis_output = project_dir / "data" / "gis" / "visualizations" / "s01_c001_map.html"
    create_s01_visualization(vis_output)
