# UrbanTrack AI — Member 2 Data Semantics, Coordinate Contracts & Score Terminology

**Document Version**: 1.0.0 (Hardened Production)  
**Author / Responsibility**: Member 2 (Vivek) — Data Integrity, Contract Verification, Semantically Honest Terminology  
**Status**: Verified & Auditable  

---

## 1. Absolute Data Integrity & Non-Fabrication Rules

The Member 2 subsystem strictly enforces zero fabrication across all real perception feeds (Rule 4):
1. **Never Fabricate Missing Data**: If a physical attribute (license plate, OCR confidence, ReID embedding, GPS, camera reliability, calibration, or ground truth identity) is absent in real sensor feeds, it must remain `None` / `null`.
2. **No Silent Synthetic Substitution**: Synthetic or imputed values must never be mixed into real perception feeds.
3. **Dataset Namespace Separation**:
   - Real sensor data lives strictly under `data/member1_perception/` or `data/real_data/` and carries `dataset_type = "real_perception"`.
   - Controlled synthetic benchmark data lives strictly under `data/benchmarks/` and carries `dataset_type = "synthetic_controlled"`.
4. **Single-Camera Real Data Boundary**: Real perception from CAM_001 is designated as `REAL PERCEPTION INTEGRATION`. It is not represented as "real-world multi-camera identity accuracy" because it lacks independent cross-camera identity ground truth (Rule 20).

---

## 2. Coordinate & Kinematic Systems

| Field | Coordinate System | Measurement Unit | Context / Semantics | Usage Constraints |
|---|---|---|---|---|
| `trajectory_point` | `image` | Pixels $[x, y]$ | 2D bounding box bottom-center in camera optical frame ($3840 	imes 2160$) | Never compute geographic distance or physical km/h speed from image pixels. |
| `latitude`, `longitude` | `gps` | WGS-84 Decimal Degrees | Calibrated camera pole or vehicle GPS location | Used for Haversine geographic distance $D(A, B)$ and road network node projection. |
| `timestamp_seconds` | `video_relative` | Seconds from video start | Local frame timeline ($t = 	ext{frame\_id} / 	ext{fps}$) | Cannot be compared across different cameras without shared time reference. |
| `timestamp_seconds` | `synchronized` | Seconds (Unix / NTP) | City-wide synchronized reference (`time_reference_id = "city_network_sync"`) | Valid for cross-camera $\Delta t$ calculation and kinematic speed bounding. |

---

## 3. Sensor & Model Confidence Semantics

To prevent semantic conflation of unrelated uncertainty signals, all confidence fields are isolated:

```json
{
  "detection_confidence": 0.942,
  "frame_detection_confidence_mean": 0.887,
  "ocr_confidence": 0.910,
  "plate_confidence": 0.910,
  "camera_reliability": 0.850
}
```

1. **`detection_confidence`**: Instantaneous YOLOv8 object detector confidence for the current bounding box.
2. **`frame_detection_confidence_mean`**: Temporal mean detection confidence across all frames of the local camera tracklet.
3. **`ocr_confidence` / `plate_confidence`**: Verified confidence emitted by the OCR character recognition model for the license plate text.
4. **`camera_reliability`**: Operational health score of the camera sensor based on environmental telemetry (blur, exposure, weather, optical occlusion).
5. **Separation Rule**: Detector confidence is never substituted for OCR confidence or camera reliability.

---

## 4. Score Semantics & Terminology Honesty

### 4.1 Uncalibrated Score Designation (Rule 5)
- **Terminology**: The primary pairwise output is designated as **`same_vehicle_score`** (aliased as `same_vehicle_probability` solely for backward compatibility).
- **Semantics**: The score is a deterministic heuristic ranking metric in $[0.0, 1.0]$ formed by combining normalized appearance cosine similarity, license plate edit distance, and kinematic feasibility.
- **Scientific Claim Boundary**:
  - The system does **NOT** claim to output a calibrated Bayesian posterior probability, probability distribution, or statistically valid confidence interval.
  - The score represents relative match plausibility for ranking candidate hypothesis edges.
  - In documentation and reports, scores are described as **"match score"**, **"estimated match likelihood"**, or **"relative route likelihood"**.

### 4.2 Evidence Ledger & Availability States (Rule 11)
`IdentityFusion` explicitly distinguishes between four fundamental evidence states:

```
                  ┌─────────────────────────────────────────┐
                  │          Evidence State Enum            │
                  ├─────────────────────────────────────────┤
                  │ AVAILABLE AND HIGH  (Score >= 0.80)     │
                  │ AVAILABLE AND LOW   (Score < 0.35)      │
                  │ MISSING             (Attribute is None) │
                  │ CONTRADICTORY       (Physical Veto)     │
                  │ NOT APPLICABLE      (Out of Domain)     │
                  └─────────────────────────────────────────┘
```

Structure exposed by `IdentityFusion`:
```json
{
  "appearance": {
    "status": "available",
    "score": 0.9412,
    "vector_dim": 512,
    "quality": 0.92
  },
  "plate": {
    "status": "missing",
    "score": null,
    "confidence": null
  },
  "temporal": {
    "status": "comparable",
    "delta_seconds": 45.2,
    "feasible": true
  },
  "spatial": {
    "status": "feasible",
    "distance_meters": 566.0,
    "required_speed_kmh": 45.1,
    "max_allowed_kmh": 120.0
  }
}
```

---

## 5. Three-Way Decision States (Rule 12)

Identity reasoning rejects binary match/non-match forcing. Every evaluated pair is classified into one of three distinct operational states:

1. **`CONFIRMED`**:
   - `same_vehicle_score` $\ge 	au_{	ext{confirmed}}$ ($0.75$).
   - Positive identity evidence present (verified plate or appearance embedding).
   - Spatio-temporal and vehicle type feasibility verified.
   - Zero physical contradictions.
2. **`AMBIGUOUS`**:
   - `same_vehicle_score` $\in [0.40, 0.75)$, OR
   - Kinematically feasible but identity evidence is missing or partially degraded, OR
   - Conflicting evidence detected without definitive contradiction (e.g. minor OCR variation on similar vehicles).
   - **Operational Rule**: Ambiguous pairs are preserved in uncertainty ledgers but **never merged into confirmed identity clusters**.
3. **`REJECTED`**:
   - `same_vehicle_score` $< 0.40$, OR
   - Hard physical contradiction (incompatible vehicle types, speed $> 120	ext{ km/h}$, plate contradiction $< 0.35$ with verified OCR).
