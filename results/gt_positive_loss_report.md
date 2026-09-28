# Forensic Audit Report: AI City / CityFlowV2 Ground-Truth Positive Association Losses

**Evaluation Dataset**: AI City Challenge 2022 / CityFlowV2 Scenario S01 (C001, C002, C003)  
**Tracklets Evaluated**: 384 local tracklets across 3 cameras  
**Ground-Truth Multi-Camera Cross Pairs**: 308 pairs  
**Retrieved Candidates**: 251 pairs (81.49%)  
**Missed Candidates**: 57 pairs (18.51%)  
**Audit Timestamp**: 2026-09-28T16:21:00Z  

---

## 1. Executive Summary & Dominant Root Cause Analysis

A systematic forensic examination of all 308 ground-truth cross-camera positive pairs revealed that **100% of the missed pairs (57 of 57)** result from a single compound phenomenon:

$$\text{Missed Positive} = (\text{Vehicle Type Confusion: } \texttt{car} \leftrightarrow \texttt{truck}) \;\land\; (\text{Cross-Model Re-ID Incompatibility: } \texttt{msmt17} \neq \texttt{aicity})$$

1. **Intra-Class Perception Noise**: The upstream Member-1 perception detector (YOLOv8s) exhibits systematic labeling variance on mid-sized vehicles (e.g., SUVs, pickups, vans) between viewpoints: a vehicle labeled `car` in C002 is frequently labeled `truck` in C003 (or vice-versa).
2. **Re-ID Model Incompatibility Gate**: Under the candidate generation architecture, when vehicle types disagree, candidates are only admitted if independent appearance evidence is available (i.e. both cameras share a compatible Re-ID feature space). Because C002 uses `osnet_x0_25_msmt17` while C001 and C003 use `osnet_x0_25_aicity`, cross-model comparison is strictly blocked to avoid random false merges. Consequently, the candidate gate prunes these pairs.
3. **No Spatiotemporal Rejections**: Not a single GT-positive pair was pruned by the temporal horizon gate ($\Delta t \le 7200\text{ s}$), simultaneous camera filter ($\Delta t = 0\text{ s}$), or the physical speed ceiling ($v \le 120\text{ km/h}$).

---

## 2. Failure Category Breakdown

| Failure Category | Pruned Pairs Count | % of Missed (N=57) | % of Total GT (N=308) | Finding & Remediation |
| :--- | :---: | :---: | :---: | :--- |
| **Compound Vehicle-Type / Re-ID** | **57** | **100.0%** | **18.51%** | Upstream detector noise (`car` vs `truck`) across unaligned Re-ID spaces. |
| **Vehicle Type Only** | 0 | 0.0% | 0.0% | No pairs with compatible Re-ID were blocked solely by vehicle type. |
| **Temporal Horizon Exceeded** | 0 | 0.0% | 0.0% | $7200\text{ s}$ window is fully sufficient for all true cross-camera transitions. |
| **Simultaneous Different Cameras** | 0 | 0.0% | 0.0% | Synchronized timestamps properly order all camera crossings. |
| **Speed Gate ($> 120\text{ km/h}$)** | 0 | 0.0% | 0.0% | No real vehicle exceeded the physical speed boundary between cameras. |
| **Missing World Coordinates** | 0 | 0.0% | 0.0% | Homography calibration is 100% available across C001, C002, and C003. |
| **Topology Gate** | 0 | 0.0% | 0.0% | All 3 cameras reside in the connected S01 corridor. |
| **Strong Plate Contradiction** | 0 | 0.0% | 0.0% | CityFlowV2 has 0 readable plate characters; no false plate contradiction occurred. |
| **Tracklet Fragmentation** | 0 | 0.0% | 0.0% | GT mapping links all fragmented local tracks to their true vehicle IDs. |
| **Other / Unclassified** | 0 | 0.0% | 0.0% | Every single missed pair is completely accounted for. |

---

## 3. Detailed Audit Table (Sample of Missed GT-Positive Pairs)

Below is an excerpt of the 57 audited missed pairs from `results/gt_positive_loss_audit.json`:

| GT ID | Camera A | Camera B | Tracklet A | Tracklet B | Types (A / B) | Re-ID Models (A / B) | $\Delta t$ (s) | Distance (m) | Implied Speed | Candidate Gate Decision |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GT_24** | C002 | C003 | `trk_368` | `trk_150` | `car` / `truck` | msmt17 / aicity | 3.91 | 42.1 | 38.8 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_24** | C002 | C003 | `trk_372` | `trk_150` | `car` / `truck` | msmt17 / aicity | 0.39 | 5.2 | 48.0 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_25** | C002 | C003 | `trk_380` | `trk_138` | `truck` / `car` | msmt17 / aicity | 9.69 | 114.5 | 42.5 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_28** | C002 | C003 | `trk_387` | `trk_201` | `car` / `truck` | msmt17 / aicity | 18.11 | 210.8 | 41.9 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_27** | C002 | C003 | `trk_394` | `trk_171` | `truck` / `car` | msmt17 / aicity | 1.79 | 24.3 | 48.9 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_29** | C002 | C003 | `trk_396` | `trk_177` | `truck` / `car` | msmt17 / aicity | 0.99 | 12.8 | 46.5 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_31** | C002 | C003 | `trk_405` | `trk_192` | `truck` / `car` | msmt17 / aicity | 0.59 | 7.9 | 48.2 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_32** | C002 | C003 | `trk_412` | `trk_198` | `truck` / `car` | msmt17 / aicity | 0.59 | 8.1 | 49.4 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_33** | C002 | C003 | `trk_427` | `trk_214` | `car` / `truck` | msmt17 / aicity | 0.19 | 2.1 | 39.8 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_47** | C002 | C003 | `trk_446` | `trk_164` | `truck` / `car` | msmt17 / aicity | 42.99 | 482.0 | 40.4 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_66** | C002 | C003 | `trk_510` | `trk_305` | `truck` / `car` | msmt17 / aicity | 2.29 | 31.4 | 49.3 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_85** | C002 | C003 | `trk_552` | `trk_376` | `car` / `truck` | msmt17 / aicity | 1.71 | 22.0 | 46.3 km/h | **REJECTED** (vtype + reid mismatch) |
| **GT_93** | C002 | C003 | `trk_599` | `trk_412` | `car` / `truck` | msmt17 / aicity | 11.59 | 142.1 | 44.1 km/h | **REJECTED** (vtype + reid mismatch) |

*(Complete 308-pair itemized data available in `results/gt_positive_loss_audit.json`)*

---

## 4. Key Scientific Insights

1. **Why `car` vs `truck` mismatch occurs**: In surveillance video, bounding box detectors struggle to differentiate pickup trucks, large SUVs, and delivery vans from passenger cars when viewing angles change from front-facing (C002) to side/rear-facing (C003).
2. **Impact on Hierarchical Evidence (Phase 6)**:
   - A hard rejection of `car` vs `truck` is scientifically unjustified when detection confidence is noisy.
   - Genuine hard contradictions should be reserved for mutually exclusive physical geometries, e.g., `motorcycle` vs `bus` or `bicycle` vs `truck`.
   - `car` vs `truck` should be treated as `WEAK_NEGATIVE` rather than `IMPOSSIBLE`.
3. **Impact on Re-ID Feature Alignment**:
   - Because C002 lacks CityFlow-aligned weights, direct Re-ID comparison cannot resolve the ambiguity.
   - However, spatiotemporal kinematic plausibility (vehicles traveling 38-50 km/h between adjacent cameras) is highly informative.
