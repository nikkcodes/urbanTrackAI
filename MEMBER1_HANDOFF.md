# UrbanTrack AI — Member 1 → Member 2 Handoff

## Status

**Layer 1 — Perception, Data Processing, Validation & GIS Foundation: COMPLETE**

Member 1 has completed the perception and GIS foundation and pushed the work to:

`origin/member-1`

The perception baseline has been processed and validated across **65 camera streams**.

---

## 1. Layer 1 Responsibilities Completed

The Layer 1 pipeline provides:

- YOLOv8 vehicle detection
- ByteTrack vehicle tracking
- Vehicle trajectory generation
- License plate detection and OCR
- OSNet vehicle Re-ID
- 512-dimensional appearance embeddings
- Camera-level perception metrics
- Camera reliability information
- Camera network / topology graph
- Pixel-to-geographic GIS projection using AI City calibration homographies
- Structured JSON outputs
- Dataset manifest and validation tooling

---

## 2. Main Per-Camera Outputs

Each processed camera is located under:

`data/output/<CAMERA_ID>/`

The primary files are:

### `observations.json`

Frame-level vehicle observations.

Important information includes:

- `camera_id`
- `frame_number`
- `timestamp`
- `track_id`
- `bbox`
- `centroid`
- `vehicle_type`
- detection confidence
- velocity information
- direction
- license plate information when available
- Re-ID metadata

This file should be used when Member 2 needs frame-level temporal or spatial information.

---

### `trajectories.json`

Track-level vehicle information.

Important fields include:

- `track_id`
- `vehicle_type`
- `start_frame`
- `end_frame`
- `duration_frames`
- `trajectory`
- `trajectory_length`
- `average_velocity_px`
- `appearance_embedding`
- `embedding_quality`
- `embedding_dim`
- `reid_model`

This is the **primary file for cross-camera Re-ID association**.

---

## 3. Re-ID Embeddings

`trajectories.json` contains the actual Re-ID appearance vector in:

`appearance_embedding`

The embedding is a **512-dimensional OSNet feature vector**.

It is **not** the same as `embedding_id`.

### Verified example

Camera:

`CAM_S05_C033`

Track:

`2840`

Embedding dimension:

`512`

Embedding quality:

`0.6307`

Re-ID model:

`osnet_x0_25_aicity`

Example structure:

```text
track_id: 2840
appearance_embedding: [512 floating-point values]
embedding_quality: 0.6307
embedding_dim: 512
reid_model: osnet_x0_25_aicity