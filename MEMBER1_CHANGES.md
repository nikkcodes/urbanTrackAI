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