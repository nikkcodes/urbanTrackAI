# UrbanTrack AI — Camera Location & Network Architecture

This document specifies the **Camera Location and Camera Network** spatial foundation for **UrbanTrack AI**. It serves as the authoritative reference for:
- **Member 2 (Multi-Camera Tracking & Spatial Inference)**: Probable next-camera inference, travel-time windowing, spatial distance, bearing calculation, and tracklet linking.
- **Member 3 (GIS Dashboard & City-Wide Mapping)**: GIS visualization, interactive map rendering, GeoJSON ingestion, and camera-to-camera topological display.

---

## 1. Overview & Source of Camera Locations

UrbanTrack AI operates on the **AI City Challenge 2022 Track 1 Multi-Target Multi-Camera (MTMC) Tracking benchmark** (CityFlowV2), captured in the city of **Dubuque, Iowa, USA**.

The physical surveillance network is documented in 4 official reference map documents located in `cam_loc/`:

| Map File | Scenarios Covered | Numeric Camera IDs | Video Streams | Region / Physical Corridor in Dubuque, IA |
|---|---|---|---|---|
| [`cam_loc/S01.png`](file:///c:/UrbanTrack/cam_loc/S01.png) | `S01` | `C001` – `C005` (5 cams) | 5 | Intersection of **Northwest Arterial (IA-32) & John F. Kennedy Rd** |
| [`cam_loc/S02.png`](file:///c:/UrbanTrack/cam_loc/S02.png) | `S02` | `C006` – `C009` (4 cams) | 4 | Intersection of **US-20 (Dodge St) & Century Dr** |
| [`cam_loc/S0345.png`](file:///c:/UrbanTrack/cam_loc/S0345.png) | `S03`, `S04`, `S05` | `C010` – `C040` (31 cams) | 50 | Corridor along **University Avenue** and **Hill St / 5th St / 3rd St** |
| [`cam_loc/S06.png`](file:///c:/UrbanTrack/cam_loc/S06.png) | `S06` | `C041` – `C046` (6 cams) | 6 | Expressway arterial corridor along **US-20 (Dodge St)** from Old Hwy Rd to Center Grove |
| **Total** | **6 Scenarios** | **46 Numeric IDs** | **65 Streams** | **Dubuque, Iowa Metropolitan Traffic Grid** |

---

## 2. Georeferencing Methodology

### Ground Truth vs. Scenario Center
The dataset ReadMe provides scenario-level reference coordinates (e.g., $42.525678, -90.723601$ for S01). **Under no circumstances are scenario-center coordinates reused for individual cameras.** Doing so would collapse multi-camera spatial reasoning into a single point.

Each physical camera location was georeferenced using a rigorous three-tier methodology:

1. **Visual Map Triangulation**:
   - Each red numbered camera circle on `cam_loc/*.png` was located and correlated with surrounding street layout, roadway geometry, curb medians, and intersection quadrants.
   - Street names (e.g., *NW Arterial*, *JFK Rd*, *Dodge St*, *Century Dr*, *University Ave*, *Loras Blvd*, *Hill St*, *Cedar Cross Rd*) and key landmarks (*UnityPoint Health Finley Hospital*, *Allison-Henderson Park*, *Flora Park*, *Kennedy Mall*, *Walmart Supercenter*, *Dubuque Museum of Art*) were identified.

2. **Ground-Plane Calibration & Orientation Analysis**:
   - Ground homography matrices from `calibration.txt` for all 65 cameras were projected to WGS84 coordinates using `scripts/project_to_gis.py`.
   - The field-of-view vector and viewing direction were cross-validated against the red directional arrows drawn on the `cam_loc` maps.

3. **OpenStreetMap Ground Network Georeferencing**:
   - Camera mounting positions (poles, mast arms, median posts) were positioned on the correct roadway edge/quadrant in OpenStreetMap coordinates in Dubuque, Iowa ($42.45^\circ\text{N} - 42.55^\circ\text{N}$, $-90.76^\circ\text{W} - -90.65^\circ\text{W}$).
   - Positions were preserved as **MAP-DERIVED and APPROXIMATE** (sub-10 meter precision).

---

## 3. Camera ID Convention

Although there are **46 physical numeric camera IDs** (`C001` through `C046`), there are **65 active video streams** across the 6 scenarios.

To ensure deterministic, unambiguous data pipelines:
- **Scenario-qualified IDs are mandatory**: `CAM_<SCENARIO>_<CAMERA>`, e.g.:
  - `CAM_S01_C001`, `CAM_S01_C002`
  - `CAM_S03_C010` (recorded in S03)
  - `CAM_S05_C010` (recorded in S05)
  - `CAM_S06_C046`
- **Distinct physical reuse**: `CAM_S03_C010` and `CAM_S05_C010` monitor the same physical intersection (Hill St & W 5th St), but represent distinct video recording sessions. They are maintained as separate keys in [`data/config/camera_locations.json`](file:///c:/UrbanTrack/data/config/camera_locations.json), explicitly documenting their sensor provenance.

---

## 4. Camera Location Schema (`data/config/camera_locations.json`)

The primary configuration artifact is [`data/config/camera_locations.json`](file:///c:/UrbanTrack/data/config/camera_locations.json), containing 65 scenario camera nodes:

```json
{
  "CAM_S01_C001": {
    "camera_id": "CAM_S01_C001",
    "camera_number": "C001",
    "scenario": "S01",
    "location": {
      "latitude": 42.525540,
      "longitude": -90.723480
    },
    "location_source": "cam_loc/S01.png",
    "location_method": "map_georeferenced",
    "location_precision": "approximate",
    "road_context": {
      "road": "John F. Kennedy Road",
      "intersection": "Northwest Arterial & John F. Kennedy Road (Southeast corner)"
    },
    "direction": {
      "description": "North-Northwest viewing northbound traffic entering intersection",
      "bearing_deg": 335.0
    },
    "confidence": "HIGH",
    "notes": "Mounted on SE corner mast arm along JFK Rd; views northbound traffic entering Northwest Arterial junction."
  }
}
```

---

## 5. Camera Graph Topology (`data/config/camera_graph.json`)

The camera network is modeled as a directed spatial graph $G = (V, E)$, saved in [`data/config/camera_graph.json`](file:///c:/UrbanTrack/data/config/camera_graph.json):
- **Nodes ($|V| = 65$)**: Scenario-qualified cameras.
- **Edges ($|E| = 164$)**: Feasible vehicular transitions based on physical road connectivity, permitted turn maneuvers, and travel direction.

### Edge Structure
```json
{
  "source": "CAM_S01_C001",
  "target": "CAM_S01_C002",
  "relationship": "possible_transition",
  "distance_m": 33.4,
  "bearing_deg": 320.2,
  "evidence": [
    "cam_loc/S01.png"
  ],
  "confidence": "HIGH"
}
```

### Topological Characteristics by Scenario:
1. **Scenario S01 (Intersection Grid, 18 directed edges)**:
   - Full 4-way signalized intersection movements: straight through, right turns, left turns across NW Arterial & JFK Rd.
2. **Scenario S02 (Divided Highway Signal, 10 directed edges)**:
   - Divided highway transition topology across Dodge St (US-20) and Century Dr, including median crossover storage.
3. **Scenario S03 (Historic Hillside Cluster, 14 directed edges)**:
   - Steep urban hillside transitions between Hill St, W 5th St, W 3rd St, and Alpine St.
4. **Scenario S04 (University Avenue Corridor, 58 directed edges)**:
   - Arterial corridor transitions spanning ~2.8 km from Downtown / Loras Blvd west to Flora Park / Pennsylvania Ave.
5. **Scenario S05 (University Ave & Hill St Subset, 52 directed edges)**:
   - 19 active cameras matching the physical topology of S03 and S04.
6. **Scenario S06 (US-20 Highway Progression, 12 directed edges)**:
   - Linear expressway corridor connecting Old Hwy Rd ($C046$) $\leftrightarrow$ Walmart ($C045$) $\leftrightarrow$ NW Arterial ($C044$) $\leftrightarrow$ Kennedy Mall ($C043$) $\leftrightarrow$ Cedar Cross Rd ($C042$) $\leftrightarrow$ Center Grove ($C041$).

---

## 6. Confidence Classification

All 65 cameras are assigned an explicit confidence grade:

| Confidence | Count | Criteria | Cameras |
|---|---|---|---|
| **HIGH** | **61** | Unambiguous intersection or roadway corridor location where visual map markers, street names, landmarks, and ground calibration agree within sub-10m precision. | $C001-C023$, $C025-C028$, $C030-C046$ |
| **MEDIUM** | **4** | Cameras positioned at complex multi-leg splits, dense diagonal street intersections, or long-range fields of view where pole position has $\approx 10-25\text{m}$ placement variance. | `C024` (Nevada St campus approach), `C029` (Custer St extended FOV) in S04 & S05 |
| **LOW** | **0** | No cameras have indeterminate or unidentifiable coordinates. | None |

---

## 7. GIS Output (`data/gis/cameras/camera_locations.geojson`)

To support direct ingestion by standard GIS software (QGIS, Mapbox, Leaflet, ArcGIS):
- File: [`data/gis/cameras/camera_locations.geojson`](file:///c:/UrbanTrack/data/gis/cameras/camera_locations.geojson)
- Compliant with **RFC 7946** (`[longitude, latitude]` coordinate ordering).
- Properties include: `camera_id`, `camera_number`, `scenario`, `confidence`, `direction`, `bearing_deg`, `road`, `intersection`, `location_source`, `location_method`.

---

## 8. Consumer Workflows

### Member 2: Spatial Reasoning & Cross-Camera Tracking
Member 2 consumes [`data/config/camera_graph.json`](file:///c:/UrbanTrack/data/config/camera_graph.json) and [`data/config/camera_locations.json`](file:///c:/UrbanTrack/data/config/camera_locations.json) to perform multi-camera trajectory linking:

```
Vehicle Observation at C_i (Timestamp T_i, Bearing Theta_i, Re-ID E_i, Plate P_i)
                        │
                        ▼
         Query Outgoing Edges from C_i in Camera Graph
                        │
                        ▼
            Candidate Cameras {C_j, C_k, ...}
                        │
                        ▼
        Multi-Factor Spatial-Temporal Filter & Ranking:
        1. Physical Connectivity: Edge existence in camera_graph.json
        2. Geographic Distance: edge.distance_m
        3. Bearing / Flow Alignment: delta(Theta_i, edge.bearing_deg)
        4. Travel-Time Feasibility: (distance_m / v_max) <= (T_j - T_i) <= (distance_m / v_min)
        5. Visual Re-ID: Cosine similarity(E_i, E_j)
        6. ANPR Agreement: Levenshtein distance(P_i, P_j)
                        │
                        ▼
            Most Probable Next Camera & Trajectory Link
```

### Member 3: GIS Dashboard & Visual Mapping
Member 3 loads [`data/gis/cameras/camera_locations.geojson`](file:///c:/UrbanTrack/data/gis/cameras/camera_locations.geojson) or [`data/config/camera_locations.json`](file:///c:/UrbanTrack/data/config/camera_locations.json) to render:
- CCTV camera icons placed at true physical coordinates across Dubuque, IA.
- Camera frustum cones or orientation arrows pointing in the direction of surveillance.
- Interactive popups displaying camera ID, scenario, road context, and confidence.
- Topological transition lines between cameras color-coded by scenario.

---

## 9. Camera GPS vs. Vehicle GPS

A critical distinction must be maintained across the system:

| Attribute | Camera GPS (Static) | Vehicle GPS (Dynamic) |
|---|---|---|
| **Definition** | Physical mounting location of the CCTV pole/mast arm at the roadway edge. | Projected location of the vehicle on the ground plane within the camera FOV. |
| **Variability** | Constant (fixed invariant geographic coordinate). | Time-varying $(lat(t), lon(t))$ trajectory per frame. |
| **Derivation** | Derived from `cam_loc/*.png` and georeferenced road network. | Derived from bounding box bottom-center projected through `calibration.txt` homography. |
| **Storage** | [`data/config/camera_locations.json`](file:///c:/UrbanTrack/data/config/camera_locations.json) | [`data/gis/trajectories/`](file:///c:/UrbanTrack/data/gis/trajectories/) or `trajectories.json` |

---

## 10. Verification & Reproduction Commands

All camera locations, graphs, and visual validation artifacts can be regenerated and validated with the following commands:

```bash
# 1. Regenerate camera locations, graph, and GeoJSON:
python scripts/generate_camera_network.py

# 2. Run the 13-rule automated validation suite:
python scripts/validate_camera_locations.py

# 3. Generate interactive HTML map and side-by-side comparison plots:
python scripts/visualize_camera_network.py
```

### Generated Artifacts
- **Interactive Map**: [`data/gis/visualizations/camera_network_map.html`](file:///c:/UrbanTrack/data/gis/visualizations/camera_network_map.html)
- **Side-by-Side Verification Plots**:
  - [`data/gis/visualizations/comparison_S01.png`](file:///c:/UrbanTrack/data/gis/visualizations/comparison_S01.png)
  - [`data/gis/visualizations/comparison_S02.png`](file:///c:/UrbanTrack/data/gis/visualizations/comparison_S02.png)
  - [`data/gis/visualizations/comparison_S0345.png`](file:///c:/UrbanTrack/data/gis/visualizations/comparison_S0345.png)
  - [`data/gis/visualizations/comparison_S06.png`](file:///c:/UrbanTrack/data/gis/visualizations/comparison_S06.png)
- **City-Wide Plot**: [`data/gis/visualizations/camera_network_all.png`](file:///c:/UrbanTrack/data/gis/visualizations/camera_network_all.png)
- **GeoJSON**: [`data/gis/cameras/camera_locations.geojson`](file:///c:/UrbanTrack/data/gis/cameras/camera_locations.geojson)
- **JSON Configs**:
  - [`data/config/camera_locations.json`](file:///c:/UrbanTrack/data/config/camera_locations.json)
  - [`data/config/camera_graph.json`](file:///c:/UrbanTrack/data/config/camera_graph.json)
