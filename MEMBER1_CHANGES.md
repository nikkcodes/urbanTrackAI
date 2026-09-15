# MEMBER1 CHANGES — Observation Schema + Provenance

Ticket A: strengthen the observation schema and provenance layer without
changing any existing perception behaviour (YOLO, ByteTrack, OCR, OSNet
Re-ID all untouched; no existing JSON field names changed).

## Files changed

| File | Change |
| --- | --- |
| `perception/config.py` | Added factual provenance model-name constants (`DETECTOR_MODEL`, `TRACKER_MODEL`, `OCR_MODEL`, `FPS_SOURCE`). |
| `perception/pipeline.py` | Imported the new constants and appended a `provenance` block to every frame observation in `observations.json`. Existing observation keys are preserved. |
| `MEMBER1_CHANGES.md` | This document. |

## Provenance block (`observations.json` -> `frames[].provenance`)

Added to every frame observation. All values are factual, observed values
taken from the running pipeline:

```json
"provenance": {
  "source_video": "<input video filename>",
  "camera_id": "CAM_001",
  "frame_number": <int>,
  "detector_model": "yolov8s.pt",
  "tracker_model": "bytetrack",
  "ocr_model": "easyocr",
  "reid_model": "osnet_x0_25_msmt17",
  "processing_device": "cuda" | "cpu",
  "fps_source": "video_metadata"
}
```

`source_video`, `camera_id`, `frame_number`, `processing_device` and
`fps_source` are observed directly. `detector_model`, `tracker_model`,
`ocr_model` and `reid_model` name the concrete artefacts actually loaded
during the run; they are not inferred from configuration.

## Field semantics

### `timestamp`
Observed. Formatted as `HH:MM:SS.mmm` derived from
`frame_number / fps` (see `PerceptionPipeline._format_timestamp`).
Zero-padded to milliseconds. Not wall-clock time.

### `track_id`
Observed. Assigned by ByteTrack (`VehicleDetector._model.track` with
`perception/bytetrack_urban.yaml`). Integers are stable across frames for a
single physical object while it remains tracked; a new physical object
receives a new id. Re-use of an id for a different object after a long
gap is a ByteTrack behaviour, not a guarantee of identity.

### `trajectory_point` / `centroid`
Observed. `[x, y]` in source-frame pixel coordinates, where `x` is the
horizontal centre of the vehicle bounding box and `y` is the bottom edge
(`y2`). Recorded once per frame, de-duplicated so consecutive identical
points are not repeated, and capped at `TRAIL_LENGTH` (120) points.

### `velocity_px`
Inferred. Euclidean distance in pixels between the current frame centroid
and the previous frame centroid for the same `track_id`:
`sqrt(dx^2 + dy^2)`, rounded to 2 decimals. First appearance of a track is
`0.0` (no previous centroid). This is a pixel-space displacement, not a
physical speed; it scales with frame resolution and camera geometry.

### `direction`
Inferred. Cardinal direction of travel derived from the sign of the
centroid delta between consecutive frames:

- `stationary` — `velocity_px < 2.0`
- `southbound` / `northbound` — `|dy| > |dx|`, sign of `dy`
- `eastbound` / `westbound` — `|dx| >= |dy|`, sign of `dx`

Pixel-space only; it does not equal compass direction on the ground.

### `appearance_embedding`
Observed when `EXPORT_TRACK_EMBEDDINGS` is enabled, otherwise absent
(`null`). An L2-normalised 512-dim float vector produced by OSNet
(`osnet_x0_25` trained on `msmt17`) via TorchReID, extracted from the
first frame in which the track is seen. `null` when the crop is too small
(`< MIN_REID_CROP_SIZE`), the model failed to load, or extraction raised.
`embedding_quality` is an observed proxy score in `[0.01, 1.0]`
combining crop resolution and Laplacian sharpness.

### `plate_bbox` / `plate_confidence`
Observed. `plate_bbox` is the license-plate rectangle (from the plate YOLO
model) mapped back into source-frame coordinates; `null` when no plate
met `PLATE_CONF_THRESHOLD` (0.40) in the vehicle crop. `plate_confidence`
is the plate detector's confidence for that box, or `null`.

### `plate_number` / `plate_text_confidence`
Observed. `plate_number` is the cleaned, validated Indian-format plate text
(`[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{4}`) from EasyOCR, or `null` when OCR did not
succeed or the cleaned text was invalid/shorter than 6 characters.
`plate_text_confidence` is the EasyOCR confidence for a successful read,
or `null`. Raw OCR text is never exposed; only the validated, cleaned form.

## Coordinate System Semantics

All spatial fields below are expressed in image pixel coordinates of the
original video frame, with the origin `(0, 0)` at the top-left corner. They
are camera-frame quantities only; no ground-plane, world, or GPS conversion
is performed unless camera calibration is available.

* **bbox**: `[x1, y1, x2, y2]` are image pixel coordinates in the original
  video frame, with origin `(0,0)` at the top-left.
* **centroid**: Bottom-center pixel of the detected vehicle bounding box,
  used for local tracking and trajectory generation.
* **trajectory_point**: Image-space pixel coordinate derived from the
  tracked centroid. It represents the vehicle's location within the camera
  image only, **not** GPS, ground-plane, or world coordinates.
* **velocity_px**: Estimated motion in **pixels per frame** computed from
  successive trajectory points. It is **not** real-world speed (km/h or
  m/s) because camera calibration is unavailable.
* **direction**: Image-plane movement direction derived from the motion
  vector within the camera frame. It is **not** a compass heading unless
  camera calibration and orientation are available.

## Observed vs inferred vs unavailable

| Classification | Fields |
| --- | --- |
| **Observed** (measured or directly reported by a component) | `track_id`, `vehicle_type`, `confidence`, `bbox`, `centroid`, `plate_bbox`, `plate_confidence`, `plate_number`, `plate_text_confidence`, `appearance_embedding`, `embedding_quality`, `timestamp`, `fps`, `frame_number`, `vehicle_count`, plus the provenance block. |
| **Inferred** (derived from observed values) | `velocity_px`, `velocity_vector`, `direction`, `trajectory` points (aggregated centroids). |
| **Unavailable / null** | `ground_plane_position` when the camera is not calibrated (`CAMERA_METADATA` empty); `appearance_embedding` and `embedding_quality` when Re-ID is disabled or extraction fails; `plate_bbox` / `plate_confidence` / `plate_number` / `plate_text_confidence` when no plate was detected or OCR did not succeed. |

`null` is used wherever a value is genuinely unavailable; no fabricated
placeholders are emitted.

## trajectories.json

Unchanged. The provenance additions only touch `observations.json`. This
file references trajectory fields (`trajectory`, `average_velocity_px`,
`appearance_embedding`, `embedding_quality`) for documentation only.

## Camera Metadata

A standalone camera metadata layer lives in
`data/config/camera_metadata.json`, describing the six prototype cameras
(`CAM_001` … `CAM_006`) without mixing camera information into vehicle
observations.

### Field sources

| Field | Source | Value |
| --- | --- | --- |
| `camera_id` / `camera_name` | Prototype identity | Factual identifier for each prototype camera. |
| `metadata_source` | Provenance tag | `video_metadata` when the camera's measured video properties come from a real source video; `prototype_configuration` when the camera is defined in the prototype configuration but has no real video telemetry. |
| `fps` / `resolution_width` / `resolution_height` | Video metadata | Observed from the source video (`traffics.mp4` is 3840x2160 @ 30 fps for `CAM_001`, so its `metadata_source` is `video_metadata`). `null` for `CAM_002`…`CAM_006`, whose `metadata_source` is `prototype_configuration`. |
| `calibrated` | Implementation truth | `false` for all cameras: `perception/config.py` ships an empty `CAMERA_METADATA` dict, so `CameraCalibration` loads no homography and `ground_plane_position` is `null` in observations. |

### Unavailable (`null`)

`latitude`, `longitude`, `camera_heading`, `field_of_view_deg`,
`camera_height_m`, `synchronization_source`,
  `synchronization_accuracy_ms`, and `neighboring_cameras` are `null` for
  every camera. These are genuinely unavailable — no GPS, no mounting
  orientation, no calibration parameters, no sync infrastructure, and no
  camera adjacency graph has been provided. They are intentionally `null`,
  not fabricated. Their `metadata_source` is `prototype_configuration`
  because no measured telemetry exists for them.

## Multi-Camera Support

The perception layer now supports observations from six prototype cameras
(`CAM_001` … `CAM_006`) while remaining camera-local: no cross-camera
matching, global identities, or synthetic multi-camera detections are
implemented.

### Supported camera IDs

`CAM_001`, `CAM_002`, `CAM_003`, `CAM_004`, `CAM_005`, `CAM_006` — the six
cameras defined in `data/config/camera_metadata.json`. Any other id is
rejected at load time.

### Camera-aware input pipeline

The camera id is now an independent input, decoupled from the video path:

```
python -m perception.pipeline --camera-id CAM_001
python -m perception.pipeline --camera-id CAM_003
```

If no `--camera-id` is supplied, the pipeline defaults to `CAM_001`. The
resolved id is validated against `camera_metadata.json`; an unknown camera
raises a clear `ValueError` and metadata is never silently created. The id
propagates into every exported observation.

### Camera-local track IDs

`track_id` remains local to a single camera. ByteTrack assigns ids
independently per video, so two cameras may both have `track_id = 17` and
they are independent observations — there is no global id and no attempt
to match tracks across cameras. Each camera's tracking state
(`track_history`, `previous_centroids`, `ocr_cache`, `track_colors`,
`inactive_track_age`, `_track_embeddings`) is scoped to its own run.

### Camera-Local Identity Rule

* `track_id` is a **camera-local ByteTrack identifier** generated
  independently for each camera.
* The pair **(`camera_id`, `track_id`)** uniquely identifies a vehicle
  track within the perception layer.
* Two different cameras may legitimately have the same `track_id`; they
  are **not** the same vehicle.
* Member 1 exports only local observations and local trajectories.
* Member 1 does **not** create global vehicle identities, candidate
  matches, or cross-camera associations.
* Cross-camera identity fusion, Bayesian matching, and global trajectory
  reconstruction are performed entirely by **Member 2**.

### Folder structure for multiple cameras

Videos may be organized per camera:

```
data/input/
  CAM_001/
    traffics.mp4
  CAM_002/
    camera2.mp4
  CAM_003/
    camera3.mp4
```

The pipeline looks inside `data/input/<camera_id>/` first and falls back to
the flat `data/input/` directory for backward compatibility. Only `CAM_001`
is required; `CAM_002`–`CAM_006` need not have videos. Cameras without a
video simply have no input to process — no observations are fabricated for
them.

### Export schema

Every observation already includes `camera_id` (top-level), `frame_number`,
`timestamp`, per-vehicle `track_id`, and `provenance.camera_id`. No schema
changes were needed beyond camera-awareness.

### Default camera behaviour

With no `--camera-id`, the pipeline defaults to `CAM_001` and loads the
newest supported video from `data/input/`, preserving the original
behaviour. CAM_001 output is otherwise equivalent to the previous run except
for camera-aware loading and validation.

### `metadata_source` provenance

`metadata_source` tags the provenance of each camera's metadata values:

- `video_metadata` — measured from a real source video. Only `CAM_001`
  carries this; its `fps`, `resolution_width`, and `resolution_height`
  were observed from `data/input/traffics.mp4` (3840x2160 @ 30 fps).
- `prototype_configuration` — the camera is defined in the prototype
  configuration but has no real video telemetry. Applies to `CAM_002`
  through `CAM_006`, and to every unavailable field on all cameras.

### Why separate from perception observations

Camera metadata is a static, per-camera configuration artefact, whereas
perception observations are per-frame, per-track measurements. Keeping
them in separate files means:

- Camera parameters can be authored/updated without touching the
  observation or trajectory JSON schemas.
- Observations stay free of static camera facts that would otherwise be
  repeated on every frame.
- Downstream consumers can load camera context once, independently of
  the streaming observation data.

## Synthetic Benchmark Dataset

A separate synthetic degradation generator
(`perception/synthetic_degradation.py`) produces degraded copies of the
real perception observations for benchmarking downstream identity fusion.
It is **only for benchmarking** and is completely isolated from the real
perception outputs.

### Purpose

Member 1 exports only local, camera-local observations. To stress-test
Member 2's identity fusion, this module applies controlled, reversible
degradations to copies of the real data -- it never fabricates detections,
plates, embeddings, or GPS values, and it only removes or weakens existing
observed information.

### Separation from real data

- Real outputs live in `data/output/` (`observations.json`,
  `trajectories.json`, `camera_metrics.json`, `perception_summary.json`).
- Synthetic outputs live in `data/synthetic_output/`
  (`observations_degraded.json`, `trajectories_degraded.json`,
  `degradation_summary.json`).
- The generator never writes to `data/output/` and never modifies the real
  perception pipeline, YOLO, ByteTrack, OCR, Re-ID, trajectory generation,
  or camera metrics.

### Supported degradation types

- **Missing plate** — `plate_number` and `plate_text_confidence` set to
  `null`; `plate_bbox` may remain.
- **OCR failure** — `plate_number` and `ocr_confidence` set to `null`.
- **Low OCR confidence** — detected text preserved, confidence reduced.
- **Missing Re-ID embedding** — `appearance_embedding` and
  `embedding_quality` set to `null` for selected tracks.
- **Occluded vehicle** — `occluded: true`; detection confidence reduced to
  a configurable bound.
- **Detection confidence degradation** — confidence lowered, always kept
  within `[0, 1]`.
- **Track fragmentation** — selected observations removed from a track
  while preserving `track_id`, making it discontinuous.
- **Dropped frames** — complete frame observations removed by a
  configurable probability.
- **Camera outage** — all observations removed for configurable frame
  ranges, recorded in the summary.

### Provenance

Every degraded observation carries `synthetic: true` and a
`degradation_tags` list (e.g. `["missing_plate", "occlusion"]`). Multiple
degradations may coexist. Real observations never contain these fields.

### Reproducibility using seeds

Generation is driven by `random.Random(seed)`. Running twice with the same
seed produces byte-identical degraded outputs; only the summary's
`generation_timestamp` differs. CLI:

```
python -m perception.synthetic_degradation --seed 42
```

### Configuration

Degradation probabilities live in `perception/config.py` under
`SYNTHETIC_*`. `SYNTHETIC_ENABLE` defaults to `True`; the rates are
conservative (`0.02`–`0.10`) so generation is lightweight by default.
Synthetic outputs are **not** real observations.