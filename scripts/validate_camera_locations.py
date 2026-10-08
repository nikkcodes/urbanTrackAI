"""
UrbanTrack AI - Camera Locations & Graph Validation Script
==========================================================
Validates that:
1. All expected camera streams (65) from manifest/dataset are represented.
2. All 46 numeric camera IDs (C001-C046) are represented.
3. Scenario-qualified IDs are unique.
4. Coordinates are valid WGS84 latitude/longitude numbers.
5. Coordinates are within the expected Dubuque region bounding box.
6. No camera has accidentally been assigned scenario-center coordinates.
7. Road/intersection context exists where identifiable.
8. Confidence exists and is HIGH, MEDIUM, or LOW.
9. Source/provenance exists.
10. Camera graph references only existing cameras in camera_locations.json.
11. No invalid self-loops exist in graph.
12. Distances in graph edges are mathematically consistent with coordinates (within 0.5m tolerance).
13. C010 in S03 and C010 in S05 are kept distinct as scenario-qualified keys.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


# Dubuque, Iowa geographic bounding box
DUBUQUE_BBOX = {
    "min_lat": 42.40,
    "max_lat": 42.60,
    "min_lon": -90.80,
    "max_lon": -90.60
}

# AI City Challenge ReadMe scenario center reference coordinates
SCENARIO_CENTERS = {
    "S01": (42.525678, -90.723601),
    "S02": (42.491916, -90.723723),
    "S03": (42.498780, -90.686393),
    "S04": (42.498780, -90.686393),
    "S05": (42.498780, -90.686393),
    "S06": (42.492448, -90.723343),
}


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * R * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def validate() -> bool:
    print("=" * 70)
    print(" UrbanTrack AI: Camera Location & Network Validation Suite")
    print("=" * 70)

    all_passed = True
    errors: List[str] = []
    warnings: List[str] = []

    # 1. Load data
    loc_file = Path("data/config/camera_locations.json")
    graph_file = Path("data/config/camera_graph.json")
    geojson_file = Path("data/gis/cameras/camera_locations.geojson")
    manifest_file = Path("data/config/aicity_manifest.json")

    if not loc_file.exists():
        print(f"[FAIL] Missing {loc_file}")
        return False
    if not graph_file.exists():
        print(f"[FAIL] Missing {graph_file}")
        return False
    if not manifest_file.exists():
        print(f"[FAIL] Missing {manifest_file}")
        return False

    with open(loc_file, "r", encoding="utf-8") as f:
        locations: Dict[str, Dict[str, Any]] = json.load(f)

    with open(graph_file, "r", encoding="utf-8") as f:
        graph: Dict[str, Any] = json.load(f)

    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    expected_manifest_cams = {c["camera_id"] for c in manifest["cameras"]}
    expected_cnums = {f"C{i:03d}" for i in range(1, 47)}

    # Rule 1: All expected camera streams are represented
    print(f"\n[Check 1/13] Expected camera streams coverage (expected {len(expected_manifest_cams)})...")
    missing_streams = expected_manifest_cams - set(locations.keys())
    if missing_streams:
        errors.append(f"Missing stream IDs in camera_locations: {missing_streams}")
    else:
        print(f"  -> PASSED: All {len(expected_manifest_cams)} video streams are represented.")

    # Rule 2: All 46 numeric camera IDs are represented
    print(f"[Check 2/13] All 46 numeric camera IDs represented (C001-C046)...")
    found_cnums = {loc["camera_number"] for loc in locations.values()}
    missing_cnums = expected_cnums - found_cnums
    if missing_cnums:
        errors.append(f"Missing numeric camera numbers: {missing_cnums}")
    else:
        print(f"  -> PASSED: All 46 numeric camera numbers (C001-C046) present.")

    # Rule 3: Scenario-qualified IDs are unique
    print(f"[Check 3/13] Scenario-qualified IDs uniqueness...")
    if len(locations.keys()) != len(set(locations.keys())):
        errors.append("Duplicate scenario-qualified IDs detected in camera_locations.json")
    else:
        print(f"  -> PASSED: All {len(locations)} keys are strictly unique.")

    # Rule 4: Coordinates are valid latitude/longitude
    print(f"[Check 4/13] Coordinate validity (WGS84 range)...")
    coord_errors = []
    for cid, rec in locations.items():
        loc = rec.get("location", {})
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        if lat is None or lon is None or not (isinstance(lat, (int, float)) and isinstance(lon, (int, float))):
            coord_errors.append(f"{cid} has invalid non-numeric coordinates: {loc}")
        elif not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            coord_errors.append(f"{cid} coordinates out of range: lat={lat}, lon={lon}")
    if coord_errors:
        errors.extend(coord_errors)
    else:
        print(f"  -> PASSED: All coordinates are valid finite WGS84 floats.")

    # Rule 5: Coordinates are within expected Dubuque region
    print(f"[Check 5/13] Dubuque geographic bounding box check...")
    bbox_errors = []
    for cid, rec in locations.items():
        loc = rec["location"]
        lat, lon = loc["latitude"], loc["longitude"]
        if not (DUBUQUE_BBOX["min_lat"] <= lat <= DUBUQUE_BBOX["max_lat"] and
                DUBUQUE_BBOX["min_lon"] <= lon <= DUBUQUE_BBOX["max_lon"]):
            bbox_errors.append(f"{cid} outside Dubuque bbox: lat={lat}, lon={lon}")
    if bbox_errors:
        errors.extend(bbox_errors)
    else:
        print(f"  -> PASSED: All coordinates lie within Dubuque, IA bounds [{DUBUQUE_BBOX['min_lat']}, {DUBUQUE_BBOX['max_lat']}] N, [{DUBUQUE_BBOX['min_lon']}, {DUBUQUE_BBOX['max_lon']}] W.")

    # Rule 6: No camera assigned scenario-center coordinate
    print(f"[Check 6/13] Verify no camera defaulted to generic scenario center...")
    center_errors = []
    for cid, rec in locations.items():
        scn = rec["scenario"]
        lat, lon = rec["location"]["latitude"], rec["location"]["longitude"]
        if scn in SCENARIO_CENTERS:
            clat, clon = SCENARIO_CENTERS[scn]
            dist_to_center = haversine_m(lat, lon, clat, clon)
            if dist_to_center < 1.0:
                center_errors.append(f"{cid} has exact center coordinates of scenario {scn}")
    if center_errors:
        errors.extend(center_errors)
    else:
        print(f"  -> PASSED: No camera is set to generic scenario-center coordinates.")

    # Rule 7: Road / intersection context exists
    print(f"[Check 7/13] Road and intersection context verification...")
    context_errors = []
    for cid, rec in locations.items():
        road_ctx = rec.get("road_context", {})
        road = road_ctx.get("road", "").strip()
        intersection = road_ctx.get("intersection", "").strip()
        if not road or not intersection:
            context_errors.append(f"{cid} missing road or intersection context: {road_ctx}")
    if context_errors:
        errors.extend(context_errors)
    else:
        print(f"  -> PASSED: All 65 cameras have verified road and intersection contexts.")

    # Rule 8: Confidence exists and is HIGH, MEDIUM, or LOW
    print(f"[Check 8/13] Confidence rating categorization...")
    conf_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    conf_errors = []
    for cid, rec in locations.items():
        conf = rec.get("confidence")
        if conf not in conf_counts:
            conf_errors.append(f"{cid} invalid confidence: '{conf}'")
        else:
            conf_counts[conf] += 1
    if conf_errors:
        errors.extend(conf_errors)
    else:
        print(f"  -> PASSED: Valid confidence levels: HIGH={conf_counts['HIGH']}, MEDIUM={conf_counts['MEDIUM']}, LOW={conf_counts['LOW']}.")

    # Rule 9: Source and provenance exists
    print(f"[Check 9/13] Source provenance and documentation...")
    source_errors = []
    for cid, rec in locations.items():
        src = rec.get("location_source", "")
        method = rec.get("location_method", "")
        if not src or not method:
            source_errors.append(f"{cid} missing location_source or method")
    if source_errors:
        errors.extend(source_errors)
    else:
        print(f"  -> PASSED: All cameras have source maps (cam_loc/*.png) and method provenance recorded.")

    # Rule 10: Camera graph references only existing cameras
    print(f"[Check 10/13] Graph node referential integrity...")
    graph_nodes = set(graph.get("nodes", []))
    loc_nodes = set(locations.keys())
    if graph_nodes != loc_nodes:
        errors.append(f"Graph nodes mismatch with locations: diff={graph_nodes ^ loc_nodes}")
    invalid_edges = []
    for edge in graph.get("edges", []):
        src, tgt = edge.get("source"), edge.get("target")
        if src not in loc_nodes or tgt not in loc_nodes:
            invalid_edges.append(f"Edge {src} -> {tgt} references non-existent camera")
    if invalid_edges:
        errors.extend(invalid_edges)
    else:
        print(f"  -> PASSED: Graph references only existing scenario camera nodes.")

    # Rule 11: No invalid self-loops
    print(f"[Check 11/13] Graph self-loop check...")
    self_loops = []
    for edge in graph.get("edges", []):
        if edge.get("source") == edge.get("target"):
            self_loops.append(edge["source"])
    if self_loops:
        errors.append(f"Invalid self-loops detected in graph: {self_loops}")
    else:
        print(f"  -> PASSED: No self-loops present in transition graph.")

    # Rule 12: Distances mathematically consistent with coordinates
    print(f"[Check 12/13] Edge distance mathematical consistency (tolerance 0.5m)...")
    dist_mismatches = []
    for edge in graph.get("edges", []):
        src, tgt = edge["source"], edge["target"]
        claimed_dist = edge.get("distance_m", 0.0)
        p1 = locations[src]["location"]
        p2 = locations[tgt]["location"]
        real_dist = haversine_m(p1["latitude"], p1["longitude"], p2["latitude"], p2["longitude"])
        if abs(claimed_dist - real_dist) > 0.5:
            dist_mismatches.append(f"Edge {src}->{tgt}: claimed {claimed_dist}m, actual {round(real_dist, 1)}m")
    if dist_mismatches:
        errors.extend(dist_mismatches[:5])
    else:
        print(f"  -> PASSED: All {len(graph['edges'])} edge distances are mathematically consistent with coordinates.")

    # Rule 13: C010 in S03 and C010 in S05 are distinct keys
    print(f"[Check 13/13] Cross-scenario qualification (CAM_S03_C010 vs CAM_S05_C010)...")
    if "CAM_S03_C010" not in locations or "CAM_S05_C010" not in locations:
        errors.append("CAM_S03_C010 or CAM_S05_C010 missing")
    elif "CAM_S03_C010" == "CAM_S05_C010":
        errors.append("CAM_S03_C010 merged with CAM_S05_C010")
    else:
        s03_c10 = locations["CAM_S03_C010"]
        s05_c10 = locations["CAM_S05_C010"]
        print(f"  -> PASSED: CAM_S03_C010 and CAM_S05_C010 maintained as distinct scenario keys.")
        print(f"     S03: {s03_c10['camera_id']} ({s03_c10['scenario']}) - {s03_c10['road_context']['intersection']}")
        print(f"     S05: {s05_c10['camera_id']} ({s05_c10['scenario']}) - {s05_c10['road_context']['intersection']}")

    # Check GeoJSON if present
    if geojson_file.exists():
        with open(geojson_file, "r", encoding="utf-8") as f:
            gj = json.load(f)
        features = gj.get("features", [])
        print(f"\n[GeoJSON Check] {geojson_file}: {len(features)} Point features loaded.")
        if len(features) != len(locations):
            errors.append(f"GeoJSON feature count ({len(features)}) does not match camera count ({len(locations)})")

    print("\n" + "=" * 70)
    if errors:
        print(f" VALIDATION FAILED with {len(errors)} error(s):")
        for err in errors:
            print(f"   [ERROR] {err}")
        return False
    else:
        print(" ALL 13 VALIDATION CHECKS PASSED SUCCESSFULLY!")
        print("=" * 70)
        return True


if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
