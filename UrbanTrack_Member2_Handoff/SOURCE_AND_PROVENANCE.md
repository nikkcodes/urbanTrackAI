# UrbanTrack AI — Source & Provenance Specification

**Document Version**: 1.0.0  
**Producer**: Member 2 Engineering  
**Consumer**: Member 3 Analytics & Governance  
**Scope**: Complete data lineage, origin tracing, and epistemological taxonomy  

---

## 1. Epistemological Categories

To guarantee scientific defensibility and audit compliance, Member 2 strictly categorizes all data into four mutually exclusive classes:

```
┌─────────────────┐  Raw sensor & algorithmic signals emitted by upstream camera
│    OBSERVED     │  pipelines (YOLOv8, DeepSORT, OSNet, video telemetry).
└────────┬────────┘
         │
         ▼
┌─────────────────┐  Probabilistic, geometric, and graph outputs derived by Member 2
│    INFERRED     │  (clustering, probabilistic evidence fusion, homography projection, calibration).
└────────┬────────┘
         │
         ▼
┌─────────────────┐  Academic benchmark annotations from AI City Challenge 2022.
│  GROUND TRUTH   │  STRICTLY ISOLATED TO VALIDATION AND AUDIT. ZERO RUNTIME EXPOSURE.
└────────┬────────┘
         │
         ▼
┌─────────────────┐  Synthetic perturbations generated during robustness & stress testing.
│    SIMULATED    │  EXPLICITLY LABELED. NEVER TREATED AS REAL OBSERVATIONS.
└─────────────────┘
```

---

## 2. Upstream Layer: Member 1 Perception Feeds

Member 1 processes raw video feeds from three cameras (`CAM_S01_C001`, `CAM_S01_C002`, `CAM_S01_C003`) and produces structured observation batches.

| Component / Field | Nature | Upstream Tool / Model | Physical Description | Artifact Source |
| :--- | :--- | :--- | :--- | :--- |
| **Bounding Boxes** | `OBSERVED` | YOLOv8x | Pixel coordinates $[x, y, w, h]$ in 1080p frame | `UrbanTrack_Member1_Handoff/output/*/tracklets.json` |
| **Local Tracklets** | `OBSERVED` | DeepSORT (camera-local) | Temporal association within single camera | `UrbanTrack_Member1_Handoff/output/*/tracklets.json` |
| **Appearance Embeddings** | `OBSERVED` | OSNet x0.25 | 512-dimensional unit-normalized feature vector | `UrbanTrack_Member1_Handoff/output/*/embeddings.npy` |
| **Telemetry (Blur/Brightness)** | `OBSERVED` | OpenCV Laplacian / Hist | Sensor quality indicators per frame | `UrbanTrack_Member1_Handoff/output/*/telemetry.json` |
| **Detection Confidence** | `OBSERVED` | YOLOv8x Softmax | Class confidence score for vehicle | `UrbanTrack_Member1_Handoff/output/*/tracklets.json` |
| **License Plates** | `UNAVAILABLE` | Censored at Video Source | Intentionally blurred in AI City 2022 dataset | N/A |

### Critical Upstream Discrepancy Note:
- `CAM_S01_C001` and `CAM_S01_C003` embeddings were extracted using checkpoint `osnet_x0_25_aicity.pth`.
- `CAM_S01_C002` embeddings were extracted using checkpoint `osnet_x0_25_msmt17.pth`.
- Member 2 dynamically detects this model mismatch via `aicity_manifest.json` and suppresses cross-model cosine comparison to prevent false merges.

---

## 3. Reference Layer: AI City Challenge 2022 Benchmark

External benchmark metadata provided by NVIDIA AI City Challenge 2022 Track 1 / CityFlowV2.

| Asset | Nature | Reference File / Standard | Provenance & Usage |
| :--- | :--- | :--- | :--- |
| **Video Timing & FPS** | `OBSERVED` | Video container metadata | Fixed 10.0 FPS across all three camera feeds |
| **Camera Synchronization** | `OBSERVED` | Official `cam_timestamp.txt` | C001: 0.00s, C002: +1.64s, C003: +2.049s relative to sequence base |
| **Homography Matrices** | `OBSERVED` | Official `calibration.txt` | 3x3 planar projection matrices mapping image pixels to metric coordinates |
| **Ground Truth Tracklets** | `GROUND TRUTH` | `gt/gt.txt` | Official 2D bounding boxes and global vehicle identities for benchmark validation |
| **CityFlow World Frame** | `OBSERVED` | CityFlowV2 coordinate origin | Local Euclidean planar coordinates in meters (**NOT** WGS84 GPS latitude/longitude) |

---

## 4. Processing Layer: Member 2 Intelligence Engine

All Member 2 algorithms transform `OBSERVED` feeds into calibrated `INFERRED` intelligence.

| Output Entity | Nature | Generating Algorithm / Module | Lineage Path |
| :--- | :--- | :--- | :--- |
| **Synchronized Timestamps** | `INFERRED` | `TimeSyncManager` | Frame number $\div$ 10.0 $+$ AI City offset |
| **World Coordinates $[X, Y]$** | `INFERRED` | `HomographyProjector` | Image bottom-center $\to$ $H \cdot [u, v, 1]^T$ (with $|W| \ge 0.50$ safeguard) |
| **Candidate Pairs** | `INFERRED` | `CandidateGenerator` | $O(N \log N)$ spatiotemporal search window |
| **Re-ID Similarity** | `INFERRED` | `CosineSimilarityCalculator` | Dot product of 512-D vectors (guarded against model mismatch) |
| **Pairwise Fusion Score** | `INFERRED` | `IdentityFusion` | Multimodal probabilistic evidence fusion |
| **Calibrated Confidence** | `INFERRED` | `PlattCalibrator` | Logistic transformation $1 / (1 + \exp(-(As + B)))$ fitted on holdout |
| **Identity Clusters** | `INFERRED` | `IdentityGraph` | Connected components with transitive contradiction resolution |
| **Trajectories** | `INFERRED` | `TrajectoryEngine` | Chronological waypoint sequencing and velocity calculation |
| **Camera Reliability Priors** | `INFERRED` | `CameraReliabilityModel` | Beta distribution parameterized by sensor telemetry |

---

## 5. Audit & Validation Layer: Disjoint Ground-Truth Linkage

Ground-truth evaluation is strictly isolated from inference logic.

1. **Pre-Inference Leakage Audit**: Prior to running inference, an automated audit verifies that no `gt_vehicle_id`, label, or evaluation metric is accessible in the inference memory space or configuration.
2. **Post-Hoc Spatial IoU Linkage**: Member 1 tracklets are linked to AI City GT vehicles post-hoc using spatial bounding-box IoU $\ge 0.50$ on synchronized frames.
   - Total tracklets: 384
   - Successfully linked tracklets: 294 (76.6%)
   - Distinct GT vehicles captured: 95
   - Distinct cross-camera GT vehicles: 79
   - Ground-truth positive cross-camera pairs: 308
   - Ground-truth negative cross-camera pairs: 28,380
3. **Disjoint Split Evaluation**: Validation is partitioned across disjoint vehicle IDs (57 training vehicles, 38 holdout vehicles) ensuring zero data leakage during calibration.

---

## 6. Simulation & Robustness Layer: Stress Testing

All stress testing and synthetic ablation scenarios are categorized as `SIMULATED`:
- **Drop-Camera Ablations**: Simulating loss of camera C002 or C003 (`SIMULATED`).
- **Telemetry Noise Ingestion**: Perturbing blur and lighting by up to $+50\%$ (`SIMULATED`).
- **Timestamp Drift Testing**: Introducing synthetic clock skew of $\pm 5.0$ seconds (`SIMULATED`).

**Strict Rule for Member 3**: Any downstream counterfactual or digital-twin scenario (e.g. simulated road closures, signal timing alterations, or demand surges) must be tagged `SIMULATED` and never mingled with live historical inference artifacts.
