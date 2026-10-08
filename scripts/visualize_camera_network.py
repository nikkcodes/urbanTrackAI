"""
UrbanTrack AI - Camera Network Visualizer & Validation Studio
=============================================================
Generates comprehensive visual validation artifacts for Member 3 and Member 2:
1. Interactive Leaflet/Folium map with OpenStreetMap tiles:
   data/gis/visualizations/camera_network_map.html
2. Side-by-side comparison plots (Original cam_loc Map vs GPS Network):
   - data/gis/visualizations/comparison_S01.png
   - data/gis/visualizations/comparison_S02.png
   - data/gis/visualizations/comparison_S0345.png
   - data/gis/visualizations/comparison_S06.png
3. Standalone scenario plots:
   - data/gis/visualizations/scenario_S01.png
   - data/gis/visualizations/scenario_S02.png
   - data/gis/visualizations/scenario_S0345.png
   - data/gis/visualizations/scenario_S06.png
4. City-wide overview plot:
   - data/gis/visualizations/camera_network_all.png
"""

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import folium
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


SCENARIO_COLORS = {
    "S01": "#D62828",  # Crimson Red
    "S02": "#0077B6",  # Blue
    "S03": "#2A9D8F",  # Persian Green
    "S04": "#E76F51",  # Burnt Orange
    "S05": "#9B5DE5",  # Purple
    "S06": "#F4A261",  # Amber
}


def create_interactive_map(locations: Dict[str, Dict[str, Any]], graph: Dict[str, Any], output_path: Path) -> None:
    """Create an interactive Leaflet/Folium map with layer controls, popups, and direction arrows."""
    map_center = [42.502, -90.710]
    m = folium.Map(
        location=map_center,
        zoom_start=13,
        tiles="OpenStreetMap",
        control_scale=True
    )

    scenario_groups: Dict[str, folium.FeatureGroup] = {}
    for scn in ["S01", "S02", "S03", "S04", "S05", "S06"]:
        scenario_groups[scn] = folium.FeatureGroup(name=f"Scenario {scn} (Cameras & Topology)", show=True)

    # 1. Edges
    for edge in graph.get("edges", []):
        src_id, tgt_id = edge["source"], edge["target"]
        if src_id not in locations or tgt_id not in locations:
            continue
        p1 = locations[src_id]["location"]
        p2 = locations[tgt_id]["location"]
        scn = locations[src_id]["scenario"]

        coords = [[p1["latitude"], p1["longitude"]], [p2["latitude"], p2["longitude"]]]
        color = SCENARIO_COLORS.get(scn, "#444444")

        tooltip_text = (
            f"<b>Transition:</b> {src_id} &rarr; {tgt_id}<br>"
            f"<b>Distance:</b> {edge['distance_m']} m<br>"
            f"<b>Bearing:</b> {edge['bearing_deg']}&deg;<br>"
            f"<b>Confidence:</b> {edge['confidence']}"
        )

        folium.PolyLine(
            locations=coords,
            color=color,
            weight=2.5,
            opacity=0.7,
            dash_array="5, 5" if edge["confidence"] != "HIGH" else None,
            tooltip=tooltip_text
        ).add_to(scenario_groups[scn])

    # 2. Camera Nodes
    for cid, cam in locations.items():
        lat = cam["location"]["latitude"]
        lon = cam["location"]["longitude"]
        scn = cam["scenario"]
        cnum = cam["camera_number"]
        color = SCENARIO_COLORS.get(scn, "#333333")

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; font-size: 13px; width: 280px;">
            <h4 style="margin: 0 0 6px 0; color: {color}; border-bottom: 2px solid {color}; padding-bottom: 4px;">
                {cid} ({cnum})
            </h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 12px;">
                <tr><td><b>Scenario:</b></td><td>{scn}</td></tr>
                <tr><td><b>Coordinates:</b></td><td>{lat:.6f}, {lon:.6f}</td></tr>
                <tr><td><b>Road:</b></td><td>{cam['road_context']['road']}</td></tr>
                <tr><td><b>Intersection:</b></td><td>{cam['road_context']['intersection']}</td></tr>
                <tr><td><b>Direction:</b></td><td>{cam['direction']['description']}</td></tr>
                <tr><td><b>Bearing:</b></td><td>{cam['direction']['bearing_deg']}&deg;</td></tr>
                <tr><td><b>Confidence:</b></td><td><span style="font-weight:bold; color:{'green' if cam['confidence']=='HIGH' else 'orange'};">{cam['confidence']}</span></td></tr>
                <tr><td><b>Source:</b></td><td>{cam['location_source']}</td></tr>
            </table>
            <p style="margin: 6px 0 0 0; font-size: 11px; color: #555;"><i>{cam['notes']}</i></p>
        </div>
        """

        folium.CircleMarker(
            location=[lat, lon],
            radius=6.5,
            color="#111111",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{cid} ({cnum}) - {cam['road_context']['road']}"
        ).add_to(scenario_groups[scn])

        # Heading vector line
        bearing = cam["direction"].get("bearing_deg")
        if bearing is not None:
            d_lat = 0.00015 * math.cos(math.radians(bearing))
            d_lon = (0.00015 / math.cos(math.radians(lat))) * math.sin(math.radians(bearing))
            folium.PolyLine(
                locations=[[lat, lon], [lat + d_lat, lon + d_lon]],
                color="#000000",
                weight=2,
                opacity=0.85
            ).add_to(scenario_groups[scn])

    for scn in scenario_groups:
        scenario_groups[scn].add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m.save(str(output_path))
    print(f"      Saved interactive HTML map: {output_path}")


def draw_network_on_axis(
    ax: plt.Axes,
    target_scenarios: List[str],
    locations: Dict[str, Dict[str, Any]],
    graph: Dict[str, Any],
    title: str,
    aspect_equal: bool = True
) -> None:
    """Render camera nodes, heading vectors, and graph edges onto a given Matplotlib axis."""
    sub_cams = {k: v for k, v in locations.items() if v["scenario"] in target_scenarios}
    sub_cam_ids = set(sub_cams.keys())

    # Edges
    for edge in graph.get("edges", []):
        src, tgt = edge["source"], edge["target"]
        if src in sub_cam_ids and tgt in sub_cam_ids:
            p1 = sub_cams[src]["location"]
            p2 = sub_cams[tgt]["location"]
            color = SCENARIO_COLORS.get(sub_cams[src]["scenario"], "#555555")

            ax.annotate(
                "",
                xy=(p2["longitude"], p2["latitude"]),
                xytext=(p1["longitude"], p1["latitude"]),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.5,
                    alpha=0.65,
                    mutation_scale=14,
                    shrinkA=7,
                    shrinkB=7,
                ),
                zorder=2
            )

    # Nodes & Vectors
    for cid, cam in sub_cams.items():
        lat = cam["location"]["latitude"]
        lon = cam["location"]["longitude"]
        scn = cam["scenario"]
        cnum = cam["camera_number"]
        color = SCENARIO_COLORS.get(scn, "#333333")

        # Camera node point
        ax.scatter(lon, lat, c=color, s=120, edgecolors="black", linewidth=1.2, zorder=5)

        # Orientation vector (approx 15-20 meters)
        bearing = cam["direction"].get("bearing_deg")
        if bearing is not None:
            v_len = 0.00012  # in degrees (~13m)
            dx = v_len * math.sin(math.radians(bearing)) / math.cos(math.radians(lat))
            dy = v_len * math.cos(math.radians(bearing))
            ax.annotate(
                "",
                xy=(lon + dx, lat + dy),
                xytext=(lon, lat),
                arrowprops=dict(
                    arrowstyle="->",
                    color="black",
                    lw=2.0,
                    mutation_scale=10,
                ),
                zorder=4
            )

        # Label box
        label_text = cnum if len(target_scenarios) <= 2 else f"{cnum}\n({scn})"
        ax.text(
            lon + 0.00004, lat + 0.00004, label_text,
            fontsize=8.5, fontweight="bold", color="#111111",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", alpha=0.9, edgecolor="#CCCCCC"),
            zorder=6
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Longitude (°W)", fontsize=10)
    ax.set_ylabel("Latitude (°N)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5, zorder=1)
    if aspect_equal:
        ax.set_aspect("equal", adjustable="datalim")


def create_side_by_side_comparison(
    ref_image_path: Path,
    target_scenarios: List[str],
    locations: Dict[str, Dict[str, Any]],
    graph: Dict[str, Any],
    output_path: Path,
    scenario_title: str
) -> None:
    """Generate a 2-panel comparison figure: original cam_loc reference map vs georeferenced network."""
    fig, (ax_ref, ax_geo) = plt.subplots(1, 2, figsize=(18, 8), dpi=200)

    # Left panel: Original reference image
    if ref_image_path.exists():
        img = mpimg.imread(str(ref_image_path))
        ax_ref.imshow(img)
        ax_ref.set_title(f"Original Reference: {ref_image_path.name}", fontsize=12, fontweight="bold", pad=10)
        ax_ref.axis("off")
    else:
        ax_ref.text(0.5, 0.5, f"Missing: {ref_image_path}", ha="center", va="center")

    # Right panel: Georeferenced GPS network
    draw_network_on_axis(
        ax_geo,
        target_scenarios,
        locations,
        graph,
        f"Georeferenced GPS Network: {scenario_title}",
        aspect_equal=True
    )

    plt.suptitle(f"UrbanTrack AI: Visual Verification - {scenario_title}", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"      Saved comparison plot: {output_path}")


def create_standalone_plot(
    title: str,
    target_scenarios: List[str],
    locations: Dict[str, Dict[str, Any]],
    graph: Dict[str, Any],
    output_path: Path,
    aspect_equal: bool = True
) -> None:
    """Generate a standalone high-resolution plot for a scenario or full city."""
    fig, ax = plt.subplots(figsize=(12, 9), dpi=200)
    draw_network_on_axis(ax, target_scenarios, locations, graph, title, aspect_equal=aspect_equal)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"      Saved standalone plot: {output_path}")


def main():
    print("[1/4] Loading camera network data...")
    loc_file = Path("data/config/camera_locations.json")
    graph_file = Path("data/config/camera_graph.json")

    with open(loc_file, "r", encoding="utf-8") as f:
        locations = json.load(f)
    with open(graph_file, "r", encoding="utf-8") as f:
        graph = json.load(f)

    vis_dir = Path("data/gis/visualizations")
    vis_dir.mkdir(parents=True, exist_ok=True)

    print("[2/4] Generating interactive Leaflet/Folium map with OpenStreetMap...")
    create_interactive_map(locations, graph, vis_dir / "camera_network_map.html")

    print("[3/4] Generating side-by-side verification comparisons with cam_loc maps...")
    create_side_by_side_comparison(
        Path("cam_loc/S01.png"),
        ["S01"],
        locations,
        graph,
        vis_dir / "comparison_S01.png",
        "Scenario S01 (NW Arterial & JFK Rd)"
    )
    create_side_by_side_comparison(
        Path("cam_loc/S02.png"),
        ["S02"],
        locations,
        graph,
        vis_dir / "comparison_S02.png",
        "Scenario S02 (US-20 & Century Dr)"
    )
    create_side_by_side_comparison(
        Path("cam_loc/S0345.png"),
        ["S03", "S04", "S05"],
        locations,
        graph,
        vis_dir / "comparison_S0345.png",
        "Scenarios S03, S04, S05 (University Ave & Hill St)"
    )
    create_side_by_side_comparison(
        Path("cam_loc/S06.png"),
        ["S06"],
        locations,
        graph,
        vis_dir / "comparison_S06.png",
        "Scenario S06 (US-20 Dodge St Corridor)"
    )

    print("[4/4] Generating standalone scenario and city-wide overview plots...")
    create_standalone_plot(
        "UrbanTrack AI - Scenario S01: Northwest Arterial & JFK Rd Junction",
        ["S01"], locations, graph, vis_dir / "scenario_S01.png"
    )
    create_standalone_plot(
        "UrbanTrack AI - Scenario S02: US-20 (Dodge St) & Century Dr Junction",
        ["S02"], locations, graph, vis_dir / "scenario_S02.png"
    )
    create_standalone_plot(
        "UrbanTrack AI - Scenarios S03, S04, S05: University Ave Corridor & Hill St",
        ["S03", "S04", "S05"], locations, graph, vis_dir / "scenario_S0345.png"
    )
    create_standalone_plot(
        "UrbanTrack AI - Scenario S06: US-20 (Dodge St) Highway Corridor",
        ["S06"], locations, graph, vis_dir / "scenario_S06.png"
    )
    create_standalone_plot(
        "UrbanTrack AI - City-Wide CCTV Surveillance Network (Dubuque, IA)\nAll 65 Camera Streams & Topological Transitions",
        ["S01", "S02", "S03", "S04", "S05", "S06"],
        locations, graph, vis_dir / "camera_network_all.png",
        aspect_equal=False
    )

    print("All visualizations created successfully!")


if __name__ == "__main__":
    main()
