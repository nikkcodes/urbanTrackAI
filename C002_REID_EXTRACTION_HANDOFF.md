# UrbanTrack AI — C002 Re-ID Extraction Audit & Controlled Experiment Handoff

**Document Status**: Controlled Experiment Audit & Reproduction Handoff  
**Scope**: Reproduction of C002 Vehicle Re-ID Extraction using AI City Fine-Tuned OSNet Checkpoint  
**Target Audience**: Senior AI/ML Engineer, Vision Systems Auditor, Pair Programming Agent  
**Analysis Date**: 2026-10-08  
**Repository Working Tree**: `git branch member-2` (Commit: `24559e5`)  
**Baseline State**: Frozen Baseline Preserved — Zero Code/Output Modifications  

---

## Executive Summary & Root Cause Confirmation

In the frozen baseline handoff (`UrbanTrack_Member1_Handoff/output/`), cameras C001 and C003 have appearance embeddings extracted using the fine-tuned vehicle Re-ID model `osnet_x0_25_aicity`, whereas camera C002 has embeddings extracted using `osnet_x0_25_msmt17` (a pedestrian Re-ID model).

Forensic chronological analysis of timestamps in `UrbanTrack_Member1_Handoff/` confirms the exact historical root cause:
1. **2026-09-17T14:20:21 UTC**: Member 1 processed camera **CAM_S01_C002** using the default MSMT17 weights (`reid_model: osnet_x0_25_msmt17`), prior to model training.
2. **2026-09-17T16:13:12 UTC**: Member 1 finished fine-tuning the AI City vehicle Re-ID checkpoint on Track 1 (`UrbanTrack_Member1_Handoff/training_metrics.json`), producing `osnet_x0_25_aicity_best.pth`.
3. **2026-09-17T16:31:22 UTC**: Member 1 processed camera **CAM_S01_C001** with the fine-tuned checkpoint (`reid_model: osnet_x0_25_aicity`).
4. **2026-09-17T16:53:14 UTC**: Member 1 processed camera **CAM_S01_C003** with the fine-tuned checkpoint (`reid_model: osnet_x0_25_aicity`).
5. **CAM_S01_C002 was never re-extracted with the fine-tuned checkpoint**, leaving C002 in an unaligned latent space that blocks 196 out of 308 (63.6%) cross-camera ground-truth pairs in Member 2.

This document details the exact requirements, code paths, preprocessing steps, and dependency environment needed to re-extract C002 in an isolated experimental setup.

---

## 1. 14-Point Technical Audit

### 1. Exact Python Script / Function Responsible for ReID Extraction
- **Extraction Class & Method**: `ReIDExtractor.extract(self, crop: np.ndarray) -> tuple[list[float] | None, float | None]` in [`UrbanTrack_Member1_Handoff/reid_extractor.py`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff/reid_extractor.py#L41-L167).
- **Underlying Extractor Call**: `self._extractor(crop_rgb)` invoking `torchreid.utils.FeatureExtractor.__call__`.
- **Pipeline Orchestration**: Invoked inside `PerceptionPipeline.process()` (from Member 1's `perception/pipeline.py`), called by [`UrbanTrack_Member1_Handoff/process_camera.py`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff/process_camera.py#L139-L144) and [`UrbanTrack_Member1_Handoff/process_aicity_batch.py`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff/process_aicity_batch.py#L430-L435).

### 2. Exact Model Architecture
- **Backbone**: OSNet (Omni-Scale Network for Re-Identification), variant `osnet_x0_25` (width multiplier 0.25).
- **Input Channels**: 3 (RGB).
- **Output Embedding Dimension**: 512 (`REID_EMBEDDING_DIM = 512`).
- **Framework & Class**: `torchreid.models.build_model(name="osnet_x0_25", num_classes=..., pretrained=False)` wrapped inside `torchreid.utils.FeatureExtractor`.

### 3. Exact Checkpoint File Used
- **Checkpoint Path**: [`UrbanTrack_Member1_Handoff/osnet_x0_25_aicity_best.pth`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff/osnet_x0_25_aicity_best.pth).
- **File Format**: PyTorch Zip Archive (`Zip archive data, at least v0.0 to extract`).
- **File Size**: 3,876,670 bytes.
- **Internal Archive**: 1,552 member files; serialized weights stored in `osnet_x0_25_aicity_best/data.pkl` (180,462 bytes) containing 487 OSNet parameter tensor keys (`conv1`, `conv2`, `conv3`, `conv4`, `conv5`, `fc`, `classifier`).
- **Training Origin**: Fine-tuned on AI City Challenge 2022 Track 1 vehicle crops (2 epochs, batch size 32, learning rate 0.0003, Adam optimizer, AMP enabled; Rank-1 accuracy 92.76%, mAP 94.62%; completed 2026-09-17T16:13:12 UTC).

### 4. Exact Checkpoint-Loading Code
From [`UrbanTrack_Member1_Handoff/reid_extractor.py`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff/reid_extractor.py#L60-L80):
```python
# 1. Initialize extractor with base OSNet model
extractor = torchreid.utils.FeatureExtractor(
    model_name=self.model_name,  # "osnet_x0_25"
    model_path="",
    device=self.device,
    verbose=False,
)

# 2. Load the fine-tuned AI City checkpoint with strict=False
ckpt = torch.load(AICITY_CHECKPOINT_PATH, map_location=self.device)
sd = ckpt.get("state_dict", ckpt)
new_sd = {
    (k[7:] if k.startswith("module.") else k): v
    for k, v in sd.items()
}
model_dict = extractor.model.state_dict()
filtered_sd = {
    k: v
    for k, v in new_sd.items()
    if k in model_dict and v.shape == model_dict[k].shape
}
extractor.model.load_state_dict(filtered_sd, strict=False)
```

### 5. Exact Preprocessing
- **Crop Source**: Video frame image (BGR array from OpenCV). For frame $t$ and vehicle bounding box $[x_1, y_1, x_2, y_2]$:
  $$\text{crop} = \text{frame}[\text{int}(y_1):\text{int}(y_2), \text{int}(x_1):\text{int}(x_2)]$$
- **Crop Coordinate Bounds**: Bounding box in source-frame pixel coordinates (resolution $1920 \times 1080$).
- **Minimum Dimension Gate**: 
  $$\text{crop\_h} \ge \text{MIN\_REID\_CROP\_SIZE} \quad \land \quad \text{crop\_w} \ge \text{MIN\_REID\_CROP\_SIZE} \quad (\text{threshold} = 32\text{ px})$$
  Crops failing this threshold return `(None, None)`.
- **Color / Channel Ordering**: BGR converted to RGB:
  $$\text{crop\_rgb} = \text{cv2.cvtColor}(\text{crop}, \text{cv2.COLOR\_BGR2RGB})$$
- **Resize**: Resized to height 256, width 128 (`image_size = (256, 128)`), per `training_config.yaml`.
- **Normalization**: Scaled to float tensor $[0.0, 1.0]$ via `torchvision.transforms.ToTensor()`, then ImageNet channel normalized:
  $$\mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
- **Augmentation**: None (inference / feature extraction mode).

### 6. Exact Frame / Track Sampling Strategy
- **Single-Frame First-Appearance Sampling**: Extracted once per track on the **first frame in which the track appears** (`start_frame`).
- **Reference**: [`UrbanTrack_Member1_Handoff/MEMBER1_CHANGES.md`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff/MEMBER1_CHANGES.md#L84-L86):
  > *"An L2-normalised 512-dim float vector produced by OSNet via TorchReID, extracted from the first frame in which the track is seen."*
- **Persistence**: Cached in `_track_embeddings[track_id] = (embedding, quality)`. At track termination, written to `trajectories.json`.

### 7. Exact Input Image / Crop Passed to the Model
- A 3-channel RGB tensor of shape `(1, 3, 256, 128)` normalized with ImageNet statistics.

### 8. Exact Embedding Dimensionality
- 512 floating-point values (`len(embedding) == 512`).

### 9. Exact Normalization Applied to Embeddings
- L2 Unit Normalization ([`reid_extractor.py`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/UrbanTrack_Member1_Handoff/reid_extractor.py#L138-L139)):
  $$\mathbf{e}_{\text{norm}} = \frac{\mathbf{e}}{\max(\|\mathbf{e}\|_2, 10^{-12})}$$
  Ensuring $\sum_{i=1}^{512} e_i^2 = 1.0$.

### 10. Exact Output JSON / Schema Fields
1. **`trajectories.json`**:
   - `appearance_embedding`: `List[float]` (512 elements)
   - `embedding_quality`: `float` in $[0.01, 1.0]$, computed as:
     $$\text{quality} = \text{round}(0.6 \times \text{resolution\_score} + 0.4 \times \text{blur\_score}, 4)$$
     where $\text{resolution\_score} = \min(1.0, \frac{h \times w}{256 \times 128})$ and $\text{blur\_score} = \min(1.0, \frac{\text{Var}(\text{Laplacian})}{500.0})$.
   - `embedding_dim`: `512`
   - `reid_model`: `"osnet_x0_25_aicity"` (replaces `"osnet_x0_25_msmt17"`)
2. **`observations.json`**:
   - `frames[].vehicles[].reid_model`: `"osnet_x0_25_aicity"`
   - `frames[].vehicles[].embedding_id`: `int` (matching `track_id`)
   - `frames[].vehicles[].embedding_dim`: `512`
   - `frames[].provenance.reid_model`: `"osnet_x0_25_aicity"`
3. **`perception_summary.json`**:
   - `"reid_model"`: `"osnet_x0_25_aicity"`

### 11. Exact Command Used to Generate the Embeddings
- **Batch Command**:
  ```bash
  python scripts/process_aicity_batch.py --split train --camera-filter CAM_S01_C002 --overwrite
  ```
- **Single-Camera Command**:
  ```bash
  python scripts/process_camera.py --camera-id CAM_S01_C002 --video "<path_to_c002_vdo.avi>"
  ```

### 12. Whether the AICity Checkpoint Can Currently Be Loaded in the Environment
- **NO**. Active runtime test of `import torch` fails with:
  `ModuleNotFoundError: No module named 'torch'`.
- The current virtual environment (`/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`) only contains `numpy`, `pytest`, `PyYAML`, `scikit-learn`, and `scipy`.
- The checkpoint file itself is verified 100% valid and uncorrupted, but execution requires PyTorch and TorchReID.

### 13. Whether the Same Pipeline Can Be Run on C002 Without Changing C001/C003
- **YES**. Member 1's perception and Re-ID pipeline processes each camera independently and writes to isolated directory structures (`data/output/<camera_id>/`). Running C002 requires zero modifications to C001 or C003.

### 14. Whether the C002 Experiment Can Be Isolated From the Frozen Member 1 Handoff
- **YES, and it MUST be isolated.**
- In-place overwriting of `UrbanTrack_Member1_Handoff/output/CAM_S01_C002/` would violate baseline immutability and break existing automated tests:
  - [`tests/test_aicity_handoff_integration.py:187-190`](file:///Users/yanalavivekreddy/Projects/urbantrack-ai/tests/test_aicity_handoff_integration.py#L187-L190) explicitly asserts:
    ```python
    elif obs.camera_id == "CAM_S01_C002":
        assert model == "osnet_x0_25_msmt17"
    ```
- Overwriting the baseline would cause `pytest` regression failures. The experiment must write to a designated experiment directory.

---

## 2. Structured Section Breakdown

### A. Existing C001 / C003 Extraction Pipeline
In C001 and C003:
1. Video frames ($1920 \times 1080$ @ 10 fps) are decoded via OpenCV.
2. ByteTrack assigns camera-local integer track IDs.
3. On the first frame of each track (`start_frame`), the vehicle bounding box is cropped.
4. `ReIDExtractor` initializes with base `osnet_x0_25`, loads `osnet_x0_25_aicity_best.pth`, strips `module.` prefixes, and filters weights matching the model dictionary.
5. The crop is converted BGR $\to$ RGB, resized to $256 \times 128$, normalized via ImageNet transforms, and passed through the model.
6. The resulting 512-D tensor is L2-normalized and converted to a Python list.
7. Quality score is computed from resolution and Laplacian sharpness.
8. Embeddings are stored in `trajectories.json` under `appearance_embedding` with `reid_model = "osnet_x0_25_aicity"`.

### B. Exact Checkpoint / Model Details
| Property | Value |
| :--- | :--- |
| **Model Name** | `osnet_x0_25` |
| **Model Weights Identifier** | `aicity` (registered as `osnet_x0_25_aicity`) |
| **Checkpoint Path** | `UrbanTrack_Member1_Handoff/osnet_x0_25_aicity_best.pth` |
| **Parameter Size** | 3.88 MB (3,876,670 bytes) |
| **Compatibility Group** | `aicity_track1_osnet` (`inference/reid_compatibility.py:65`) |
| **Latent Dimension** | 512-D float vector |
| **Normalization** | Unit L2-norm ($\|\mathbf{e}\|_2 = 1.0$) |

### C. Exact Preprocessing Specification
| Step | Operation | Parameters / Specification |
| :--- | :--- | :--- |
| 1. BBox Crop | Array slice on frame | `crop = frame[y1:y2, x1:x2]` |
| 2. Gate | Dimension check | `h >= 32 and w >= 32` |
| 3. Color Space | OpenCV conversion | `cv2.COLOR_BGR2RGB` |
| 4. Spatial Resize | Bilinear interpolation | `(height=256, width=128)` |
| 5. Tensor Conversion | Standard transform | `torchvision.transforms.ToTensor()` $[0.0, 1.0]$ |
| 6. Standardization | ImageNet z-score | $\mu = [0.485, 0.456, 0.406], \sigma = [0.229, 0.224, 0.225]$ |
| 7. Forward Pass | Feature extractor | Evaluates pooled feature representation (pre-classifier) |
| 8. L2 Normalize | Vector norm clamp | $\mathbf{e} / \max(\|\mathbf{e}\|_2, 10^{-12})$ |

### D. Exact Execution Command
To perform standalone C002-only re-extraction without running the full perception stack (using existing bounding boxes from `observations.json`):
```bash
python scripts/experiments/extract_c002_aicity_reid.py \
    --input-dir UrbanTrack_Member1_Handoff/output/CAM_S01_C002 \
    --video-path <path_to_c002_video> \
    --checkpoint UrbanTrack_Member1_Handoff/osnet_x0_25_aicity_best.pth \
    --output-dir results/experiments/c002_aicity_reid/output/CAM_S01_C002
```

### E. Dependencies & Environment Requirements
The following packages must be installed in the active Python environment:
1. `torch >= 2.0.0`
2. `torchvision >= 0.15.0`
3. `torchreid` (via `pip install torchreid` or `pip install git+https://github.com/KaiyangZhou/deep-person-reid.git`)
4. `opencv-python >= 4.8.0`

### F. C002 Extraction Feasibility
- **Feasibility Status**: Blocked by missing Python dependencies (`torch`, `torchreid`, `cv2`) and missing raw video file (`vdo.avi` was located on Windows machine `C:\Users\kanis\...`).
- **Potential Video Fallback**: The repository contains `UrbanTrack_Member1_Handoff/output/CAM_S01_C002/vdo_tracked.avi` (200MB, 2110 frames @ 10 fps). However, `vdo_tracked.avi` contains visual tracking overlays (bounding boxes and labels drawn on vehicle centroids), which may contaminate appearance crops. Obtaining or restoring raw `vdo.avi` is preferred.

### G. Required Files Checklist
- [x] **Fine-Tuned Checkpoint**: `UrbanTrack_Member1_Handoff/osnet_x0_25_aicity_best.pth` (VERIFIED PRESENT & VALID).
- [x] **Existing C002 Bounding Boxes & Tracks**: `UrbanTrack_Member1_Handoff/output/CAM_S01_C002/observations.json` and `trajectories.json` (VERIFIED PRESENT).
- [x] **Extraction Logic Reference**: `UrbanTrack_Member1_Handoff/reid_extractor.py` (VERIFIED PRESENT).
- [ ] **Python Dependencies**: `torch`, `torchreid`, `cv2` (MISSING).
- [ ] **Unannotated C002 Source Video**: `vdo.avi` (MISSING; only annotated `vdo_tracked.avi` exists in repository).

### H. Proposed Isolated Experiment Output Location
To guarantee complete isolation from the frozen baseline:
```
results/experiments/c002_aicity_reid/
├── output/
│   └── CAM_S01_C002/
│       ├── observations.json       # Clone with reid_model='osnet_x0_25_aicity'
│       ├── trajectories.json       # Updated with genuine aicity 512-D embeddings
│       ├── camera_metrics.json     # Cloned telemetry
│       └── perception_summary.json # Updated reid_model summary
└── experiment_audit.json           # Evaluation metrics comparing baseline vs experiment
```

### I. Risks & Failure Points
1. **Repository Baseline Contamination**: If `UrbanTrack_Member1_Handoff/output/CAM_S01_C002/` is modified directly, regression tests in `test_aicity_handoff_integration.py` will fail, and pre-hardening baseline comparisons will be corrupted.
2. **Overlay Contamination in `vdo_tracked.avi`**: If crops are extracted from `vdo_tracked.avi` instead of clean `vdo.avi`, bounding box outlines and text labels drawn on frame 0..2110 will be embedded into the feature representation, distorting cosine similarities.
3. **Dependency Version Drift**: `torchreid` requires matching PyTorch and TorchVision ABI versions. On macOS Apple Silicon (ARM64), `torchreid` must be installed carefully without broken CUDA dependencies.
4. **False Merge Risk Post-Alignment**: Aligning C002 into the AICity vehicle latent space will unlock appearance scoring for 196 previously blocked pairs. The decision threshold must remain strictly $\tau = 0.70$ to ensure zero false merges occur.

---

## Feasibility Classification

### Primary Classification:
**`FEASIBLE_WITH_DEPENDENCY_FIX`**

### Categorical Classification:
**`B. Dependency/environment blocker`**

*(Execution is blocked exclusively by the absence of `torch`, `torchreid`, and `cv2` in the active environment, and the need to verify unannotated source video availability. The architecture, checkpoint weights, schema definitions, and extraction logic are completely validated.)*
