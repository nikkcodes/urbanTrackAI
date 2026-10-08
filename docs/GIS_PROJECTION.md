# UrbanTrack AI — GIS Projection Architecture (Phase 1)

This document describes the mathematical and software architecture of UrbanTrack AI's GIS projection layer. The GIS layer transforms pixel-space detections and trajectories from traffic surveillance cameras into georeferenced coordinates on the Earth's surface.

---

## 1. Dataset Calibration Overview

The underlying dataset is the **AI City Challenge 2022 Track 1 MTMC** (part of the CityFlow/CityFlowV2 family). It features real-world traffic surveillance environments across 6 scenarios (`S01`–`S06`) captured by 46 unique cameras.

Each camera subfolder provides a `calibration.txt` file containing:
- `Homography matrix:` A $3 \times 3$ matrix $H$
- `Reprojection error:` Calibration reprojection error in pixels
- *(Optional, 6 cameras only)* `Intrinsic parameter matrix` and `Distortion coefficients`

No extrinsic matrices (3D rotation $R$ and 3D translation $t$), projection matrices, or physical camera pole coordinates are provided in the raw dataset.

---

## 2. Homography Matrix Mathematics

A planar homography is a projective transformation that relates points between two planes. In the context of traffic surveillance:
- **Plane 1:** The camera image sensor plane, parameterized by pixel coordinates $(u, v)$ where $u \in [0, W]$ and $v \in [0, H]$.
- **Plane 2:** The ground/road surface plane, parameterized by spatial coordinates.

The matrix $H$ stored in each camera's `calibration.txt` relates a 2D ground-plane point to an image pixel point:

$$\begin{bmatrix} u \cdot w \\ v \cdot w \\ w \end{bmatrix} = H \begin{bmatrix} X_{ground} \\ Y_{ground} \\ 1 \end{bmatrix}$$

where $H \in \mathbb{R}^{3 \times 3}$:

$$H = \begin{bmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & h_{33} \end{bmatrix}$$

---

## 3. Pixel to Ground-Plane Transformation ($H^{-1}$)

To project an observed pixel coordinate $(u, v)$ (such as a vehicle's tire-contact patch) onto the ground plane, we must apply the **inverse** transformation $H^{-1}$:

$$\begin{bmatrix} \tilde{X} \\ \tilde{Y} \\ \tilde{W} \end{bmatrix} = H^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}$$

Normalizing by the homogeneous scale factor $\tilde{W}$:

$$X_{ground} = \frac{\tilde{X}}{\tilde{W}}, \quad Y_{ground} = \frac{\tilde{Y}}{\tilde{W}}$$

### Why $H^{-1}$ is Used
- $H$ maps from ground coordinates to image pixel coordinates (forward camera projection).
- $H^{-1}$ back-projects image pixels along camera rays onto the intersection with the calibrated ground plane ($Z = 0$).

---

## 4. WGS84 Geographic Interpretation

Empirical validation across all 65 camera calibration files confirmed that the ground-plane coordinate system $[X_{ground}, Y_{ground}]$ used to calibrate $H$ **is already geographic WGS84 coordinates in decimal degrees**:
- $X_{ground} \equiv \text{Latitude}$ (degrees North, $[-90.0, 90.0]$)
- $Y_{ground} \equiv \text{Longitude}$ (degrees East/West, $[-180.0, 180.0]$)

For example, evaluating the center pixel $(960, 540)$ for camera `S01/c001` yields:
- $\text{Latitude} = 42.525655^\circ \text{ N}$
- $\text{Longitude} = -90.723457^\circ \text{ W}$

This precisely matches the real-world intersection of **NW Arterial and Kennedy Road in Dubuque, Iowa**, which corresponds to the official scenario center anchor provided in `ReadMe.txt` (`42.525678, -90.723601`).

---

## 5. GeoJSON Coordinate Ordering (RFC 7946)

There is a critical difference between Python/GIS mathematical conventions and standard GeoJSON serialization:

| Context | Coordinate Order | Example |
| :--- | :--- | :--- |
| **Internal Python API** (`project_to_gis.py`) | `latitude, longitude` | `{"latitude": 42.525655, "longitude": -90.723457}` |
| **Folium Map Library** | `[latitude, longitude]` | `[42.525655, -90.723457]` |
| **GeoJSON Standard (RFC 7946)** | `[longitude, latitude]` | `[-90.723457, 42.525655]` |

The UrbanTrack GIS projection utility explicitly ensures that all generated GeoJSON files (`data/gis/fov/*.geojson` and `data/gis/trajectories/*/*.geojson`) strictly adhere to RFC 7946 by converting internal `[lat, lon]` pairs into `[lon, lat]` tuples.

---

## 6. Distinguishing Spatial Concepts

To prevent architectural errors, three spatial concepts must be strictly separated:

```
[ Physical Camera Pole ]  (elevated, e.g. 8m high)
          \
           \  Camera optical ray
            \
             v
   [ Vehicle Bounding Box Bottom-Center ]  --> Projects to [ Ground-Plane Vehicle Position ]
             |
             +-------------------------------> Inside [ Camera FOV Footprint Polygon ]
```

1. **Physical Camera Position:**
   - The 3D location $(X_c, Y_c, Z_c)$ where the camera hardware is mounted on a physical pole or gantry.
   - **Status in Dataset:** *Unavailable as numeric data.* Cannot be computed from a 2D homography without known 3D extrinsics or full intrinsic matrix decomposition.
2. **Ground-Plane Vehicle Position:**
   - The point $(lat, lon)$ where a vehicle's wheels touch the road surface plane.
   - Calculated by computing the bottom-center of the vehicle's 2D detection box:
     $$u = x + \frac{\text{width}}{2}, \quad v = y + \text{height}$$
     and applying $H^{-1}$.
3. **Camera FOV Ground Footprint:**
   - The 4-corner polygon formed by back-projecting the image sensor boundaries $(0,0)$, $(W,0)$, $(W,H)$, $(0,H)$ via $H^{-1}$.
   - Represents the planar ground patch observed by the camera optics. Because the homography is fitted to the road plane, areas near or above the horizon can produce extreme or unstable coordinates.

---

## 7. Current Dataset Limitations

1. **Exact Camera Pole GPS Coordinates:**
   - The challenge creators explicitly state in `ReadMe.txt`:
     > *"Since we do not have access to the exact GPS location of each camera, the GPS location for the approximate center of each scenario is provided."*
   - Placing exact camera pole markers on a GIS map requires separate manual or semi-automated digitization from the raster maps in `cam_loc/*.png` (`S01.png`, `S02.png`, `S0345.png`, `S06.png`).
2. **Horizon Instability:**
   - Planar homography assumes all projected rays intersect the ground plane $Z = 0$. For cameras with shallow tilt angles, top image corners $(0, 0)$ and $(W, 0)$ project far down the road or near the horizon line where projective distortion is large. The utility validates all coordinates against WGS84 limits and rejects non-finite results.
3. **Elevated Objects:**
   - Bounding box centers for tall vehicles (trucks, buses) do not touch the ground. For accurate ground-plane GIS projection, the **bottom-center** $(x + w/2, y + h)$ must always be used.
