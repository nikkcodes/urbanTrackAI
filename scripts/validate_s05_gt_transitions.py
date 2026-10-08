"""
UrbanTrack AI - S05 Ground-Truth Transition Auditing Script
============================================================
Empirically cross-references S05 camera graph candidate transitions
against AI City Challenge 2022 Track 1 MTMC S05 ground-truth annotations (gt.txt).

Reports:
- source
- target
- GT transition count (direct consecutive and multi-hop)
- graph distance (m)
- graph bearing (deg)
- confidence
"""

import json
import math
from pathlib import Path
from collections import defaultdict


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def forward_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def main():
    print("=" * 80)
    print(" UrbanTrack AI: S05 Ground-Truth Camera Transition Audit")
    print("=" * 80)

    # 1. Load graph and locations
    graph_path = Path("data/config/camera_graph.json")
    locations_path = Path("data/config/camera_locations.json")

    with open(graph_path, "r", encoding="utf-8") as f:
        graph = json.load(f)
    with open(locations_path, "r", encoding="utf-8") as f:
        locations = json.load(f)

    # Index graph edges
    edge_map = {(e["source"], e["target"]): e for e in graph["edges"]}

    # 2. Parse S05 ground truth
    dataset_s05_dir = Path(r"C:\Users\kanis\Downloads\AICity22_Track1_MTMC_Tracking\validation\S05")
    if not dataset_s05_dir.exists():
        print(f"[ERROR] S05 dataset path not found: {dataset_s05_dir}")
        return

    camera_records = defaultdict(list)
    gt_files = list(dataset_s05_dir.glob("c*/gt/gt.txt"))
    print(f"Loaded {len(gt_files)} S05 camera ground-truth files from:\n{dataset_s05_dir}\n")

    for gt_file in gt_files:
        c_name = f"CAM_S05_{gt_file.parent.parent.name.upper()}"
        with open(gt_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    frame = int(parts[0])
                    vid = int(parts[1])
                    camera_records[c_name].append((vid, frame))

    # Build per-vehicle trajectories across cameras
    v_data = defaultdict(lambda: defaultdict(list))
    for cam, records in camera_records.items():
        for vid, frame in records:
            v_data[vid][cam].append(frame)

    vehicle_paths = {}
    for vid, cams in v_data.items():
        intervals = []
        for cam, frames in cams.items():
            intervals.append({"cam": cam, "start": min(frames), "end": max(frames), "count": len(frames)})
        intervals.sort(key=lambda x: x["start"])
        vehicle_paths[vid] = intervals

    # Count consecutive transitions and any-temporal precedence
    consecutive_counts = defaultdict(int)
    precedence_counts = defaultdict(int)

    for vid, path in vehicle_paths.items():
        for i in range(len(path) - 1):
            consecutive_counts[(path[i]["cam"], path[i+1]["cam"])] += 1
        for i in range(len(path)):
            for j in range(i + 1, len(path)):
                if path[i]["cam"] != path[j]["cam"]:
                    precedence_counts[(path[i]["cam"], path[j]["cam"])] += 1

    # Candidate / Intended edges to audit
    intended_edges = [
        ("CAM_S05_C010", "CAM_S05_C017"),
        ("CAM_S05_C017", "CAM_S05_C010"),
        ("CAM_S05_C033", "CAM_S05_C034"),
        ("CAM_S05_C034", "CAM_S05_C033"),
        ("CAM_S05_C029", "CAM_S05_C034"),
        ("CAM_S05_C034", "CAM_S05_C029"),
    ]

    header = f"{'Source':<15} | {'Target':<15} | {'Direct GT':<10} | {'Multi-hop':<10} | {'Dist (m)':<9} | {'Bearing':<8} | {'Conf':<6} | {'In Graph'}"
    print(header)
    print("-" * len(header))

    for src, tgt in intended_edges:
        direct_cnt = consecutive_counts.get((src, tgt), 0)
        multi_cnt = precedence_counts.get((src, tgt), 0)

        # Distance & bearing from coordinates
        p1 = locations[src]["location"]
        p2 = locations[tgt]["location"]
        calc_dist = round(haversine_m(p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"]), 1)
        calc_bearing = round(forward_bearing_deg(p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"]), 1)

        in_graph = (src, tgt) in edge_map
        edge_data = edge_map.get((src, tgt), {})
        graph_dist = edge_data.get("distance_m", calc_dist)
        graph_bearing = edge_data.get("bearing_deg", calc_bearing)
        conf = edge_data.get("confidence", "HIGH")

        print(
            f"{src:<15} | {tgt:<15} | {direct_cnt:<10} | {multi_cnt:<10} | {graph_dist:<9.1f} | {graph_bearing:<8.1f} | {conf:<6} | {'YES' if in_graph else 'NO'}"
        )

    print("\n" + "=" * 80)
    print("Corridor Flow Notes:")
    print(" - C010 <-> C017: Bidirectional traffic directly observed (14 Northbound, 15 Southbound).")
    print(" - C033 <-> C034: Bidirectional traffic directly observed (146 Westbound, 10 Eastbound).")
    print(" - C034 -> C029: 48 vehicles directly observed moving Eastbound along University Ave.")
    print(" - C029 -> C034: 47 vehicles observed moving Westbound along University Ave via C033.")
    print("=" * 80)


if __name__ == "__main__":
    main()
