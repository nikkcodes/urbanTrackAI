# Forensic Audit: Layer 2 Candidate Generation Retrieval Stage

**Stage**: Layer 2 Cross-Camera Association — Candidate Generation / Retrieval Audit  
**Date**: October 9, 2026  
**Audit Status**: COMPLETED  
**Overall Verdict**: `LAYER2_CANDIDATE_AUDIT = C` (**RECALL RISK**)  
**Production Artifacts Inspected**:
- `results/layer2_candidates/candidate_pairs.json` (5.2 GB, 6,863,626 candidates)
- `results/layer2_candidates/candidate_rejections.json` (707 MB, 2,297,286 rejections)
- `results/layer2_candidates/candidate_summary.json` (4.3 KB)
- `results/layer2_candidates/candidate_validation.json` (1.7 KB)
- `results/gt_positive_loss_audit.json` / `results/baseline_before_hardening/tracklet_to_gt.json` (CityFlow S01 Ground Truth)

---

## 1. Candidate Accounting Verification

Every pair in the complete theoretical combinatorial space of $N = 7,448$ canonical tracklets was audited for integer-level balance:

$$\begin{aligned}
\text{Total Canonical Tracklets } (N) &= 7,448 \\
\text{Global Theoretical Combinatorial Pairs } \binom{N}{2} &= \frac{7,448 \times 7,447}{2} = \mathbf{27,732,628}
\end{aligned}$$

### Identity 1: Global Space Partitioning
$$\text{Global Theoretical Pairs} = \text{Cross-Scenario Eliminated} + \text{Intra-Scenario Pairs}$$
$$27,732,628 = 18,571,716 + 9,160,912 \quad (\mathbf{EXACT\ MATCH})$$

### Identity 2: Intra-Scenario Gating Exhaustion
$$\begin{aligned}
\text{Intra-Scenario Theoretical} &= \text{Same-Camera Rejected} + \text{Negative-Time Rejected} \\
&\quad + \text{Zero-Time Rejected} + \text{Excessive-Speed Rejected} + \text{Candidate Pairs}
\end{aligned}$$
$$\begin{aligned}
9,160,912 &= 741,750 + 712,742 + 4,019 + 838,775 + 6,863,626 \\
9,160,912 &= 9,160,912 \quad (\mathbf{EXACT\ MATCH})
\end{aligned}$$

| Accounting Bucket | Pair Count | % of Global Space | % of Intra-Scenario | Verification Status |
| :--- | :---: | :---: | :---: | :---: |
| **Cross-Scenario Eliminated (Partitioning)** | $18,571,716$ | $66.97\%$ | — | Verified |
| **Same-Camera Rejected (`REJECT_SAME_CAMERA`)** | $741,750$ | $2.67\%$ | $8.10\%$ | Verified |
| **Negative Time Rejected (`REJECT_NEGATIVE_TIME`)** | $712,742$ | $2.57\%$ | $7.78\%$ | Verified |
| **Zero Time Rejected (`REJECT_ZERO_OR_INVALID_TIME`)** | $4,019$ | $0.01\%$ | $0.04\%$ | Verified |
| **Excessive Speed Rejected (`REJECT_EXCESSIVE_SPEED`)** | $838,775$ | $3.02\%$ | $9.16\%$ | Verified |
| **Accepted Candidates (`CANDIDATE`)** | $\mathbf{6,863,626}$ | $\mathbf{24.75\%}$ | $\mathbf{74.92\%}$ | Verified |
| **Total Accounted Pairs** | $\mathbf{27,732,628}$ | $\mathbf{100.00\%}$ | — | **Zero Discrepancy** |

---

## 2. Scenario-by-Scenario Breakdown

The dataset comprises 6 scenarios varying dramatically in spatial topology, tracklet density, and camera layout:

| Scenario | Cameras | Tracklets | Theoretical Intra Pairs | Same-Camera Rejections | Negative-Time Rejections | Zero-Time Rejections | Excessive-Speed Rejections | Candidates | Candidate % of Intra |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **S01** | 5 | 625 | 195,000 | 39,309 | 12,394 | 0 | 938 | 142,359 | 73.00% |
| **S02** | 4 | 730 | 266,085 | 67,444 | 13,105 | 0 | 1,082 | 184,454 | 69.32% |
| **S03** | 6 | 219 | 23,871 | 4,182 | 5,983 | 0 | 184 | 13,522 | 56.65% |
| **S04** | 25 | 781 | 304,590 | 14,272 | 36,315 | 0 | 19,185 | 234,818 | 77.09% |
| **S05** | 19 | 3,921 | 7,685,160 | 498,620 | 598,701 | 3,452 | 643,935 | 5,940,452 | 77.30% |
| **S06** | 6 | 1,172 | 686,206 | 117,923 | 46,244 | 567 | 173,451 | 348,021 | 50.72% |
| **Total** | **65** | **7,448** | **9,160,912** | **741,750** | **712,742** | **4,019** | **838,775** | **6,863,626** | **74.92%** |

### Key Observations
1. **S05 Scale Dominance**: Scenario S05 alone contributes **86.55%** ($5,940,452 / 6,863,626$) of the entire candidate pool. With 3,921 tracklets and 19 cameras, a permissive speed gate allows 77.3% of all intra-scenario pairs to survive.
2. **S06 Expressway Speed Filtering**: In S06, 173,451 pairs (25.28% of intra-scenario pairs) are eliminated by `REJECT_EXCESSIVE_SPEED` because long inter-camera distances (up to 3,766m) expose impossible transit times.
3. **Zero-Time Frequency**: `REJECT_ZERO_OR_INVALID_TIME` only triggers in S05 ($3,452$) and S06 ($567$) where high tracklet volume leads to simultaneous integer millisecond start/end timestamps across distinct cameras.

---

## 3. Camera-Pair Breakdown

Across the 65 cameras, a total of **856 distinct directed camera transitions** generate candidates.

### Top 20 Directed Transitions by Candidate Count

| Rank | Source Camera | Destination Camera | Scenario | Candidate Count | Rejections | Candidate Rate | Distance (m) | Graph Edge? | Min $\Delta t$ (s) | Med $\Delta t$ (s) | Max $\Delta t$ (s) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | `CAM_S05_C036` | `CAM_S05_C033` | S05 | **67,891** | 13,007 | 83.92% | 169.2m | **No** | 3.84s | 87.2s | 393.5s |
| 2 | `CAM_S05_C036` | `CAM_S05_C029` | S05 | **62,081** | 12,042 | 83.75% | 286.1m | **No** | 6.44s | 85.9s | 392.4s |
| 3 | `CAM_S05_C033` | `CAM_S05_C029` | S05 | **61,929** | 11,885 | 83.89% | 119.3m | **No** | 2.74s | 87.0s | 393.0s |
| 4 | `CAM_S05_C029` | `CAM_S05_C033` | S05 | **61,734** | 12,476 | 83.19% | 119.3m | **No** | 2.75s | 86.8s | 392.5s |
| 5 | `CAM_S05_C033` | `CAM_S05_C036` | S05 | **59,578** | 13,678 | 81.33% | 169.2m | **No** | 3.85s | 86.6s | 392.7s |
| 6 | `CAM_S05_C029` | `CAM_S05_C036` | S05 | **58,908** | 13,293 | 81.59% | 286.1m | **No** | 6.45s | 86.3s | 392.2s |
| 7 | `CAM_S05_C036` | `CAM_S05_C034` | S05 | **58,071** | 10,753 | 84.38% | 254.5m | **No** | 5.74s | 88.0s | 393.5s |
| 8 | `CAM_S05_C034` | `CAM_S05_C036` | S05 | **55,758** | 12,238 | 82.00% | 254.5m | **No** | 5.75s | 87.8s | 393.0s |
| 9 | `CAM_S05_C033` | `CAM_S05_C034` | S05 | **54,163** | 10,488 | 83.78% | 98.4m | **Yes** | 2.24s | 88.1s | 393.3s |
| 10 | `CAM_S05_C034` | `CAM_S05_C033` | S05 | **53,952** | 10,875 | 83.22% | 98.4m | **Yes** | 2.25s | 88.0s | 393.1s |
| 11 | `CAM_S05_C034` | `CAM_S05_C029` | S05 | **52,431** | 10,266 | 83.63% | 48.2m | **No** | 1.15s | 87.9s | 393.2s |
| 12 | `CAM_S05_C029` | `CAM_S05_C034` | S05 | **52,248** | 10,622 | 83.11% | 48.2m | **No** | 1.14s | 88.1s | 393.0s |
| 13 | `CAM_S05_C036` | `CAM_S05_C035` | S05 | **44,692** | 7,651 | 85.38% | 412.3m | **No** | 9.24s | 90.1s | 393.5s |
| 14 | `CAM_S05_C035` | `CAM_S05_C036` | S05 | **43,842** | 8,241 | 84.18% | 412.3m | **No** | 9.25s | 89.8s | 393.0s |
| 15 | `CAM_S05_C033` | `CAM_S05_C035` | S05 | **42,678** | 7,492 | 85.07% | 243.6m | **No** | 5.54s | 90.3s | 393.2s |
| 16 | `CAM_S05_C035` | `CAM_S05_C033` | S05 | **42,395** | 7,789 | 84.48% | 243.6m | **No** | 5.55s | 90.0s | 393.0s |
| 17 | `CAM_S05_C034` | `CAM_S05_C035` | S05 | **40,517** | 7,125 | 85.04% | 158.4m | **No** | 3.64s | 90.5s | 393.1s |
| 18 | `CAM_S05_C035` | `CAM_S05_C034` | S05 | **40,248** | 7,377 | 84.51% | 158.4m | **No** | 3.65s | 90.4s | 392.9s |
| 19 | `CAM_S05_C029` | `CAM_S05_C035` | S05 | **39,874** | 7,088 | 84.91% | 204.8m | **No** | 4.65s | 90.4s | 393.0s |
| 20 | `CAM_S05_C035` | `CAM_S05_C029` | S05 | **39,620** | 7,352 | 84.35% | 204.8m | **No** | 4.64s | 90.5s | 392.8s |

> **Critical Finding**: 18 of the top 20 candidate-producing camera transitions in S05 have **NO DIRECT ROAD GRAPH EDGE** in `camera_graph.json`! Because distance is moderate ($48\text{m} - 412\text{m}$) and temporal gaps up to 393s are permitted, these non-edge transitions dominate the candidate explosion.

---

## 4. Temporal Transit Distribution

Distribution of $\Delta t = t_{\text{dest, start}} - t_{\text{orig, end}}$ across all $6,863,626$ candidates:

| Quantile | Global $\Delta t$ (s) | S01 $\Delta t$ (s) | S02 $\Delta t$ (s) | S03 $\Delta t$ (s) | S04 $\Delta t$ (s) | S05 $\Delta t$ (s) | S06 $\Delta t$ (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Minimum** | $0.19\text{ s}$ | $0.20\text{ s}$ | $0.34\text{ s}$ | $0.45\text{ s}$ | $0.21\text{ s}$ | $0.19\text{ s}$ | $10.60\text{ s}$ |
| **p01** | $7.10\text{ s}$ | $5.30\text{ s}$ | $5.10\text{ s}$ | $6.20\text{ s}$ | $6.40\text{ s}$ | $7.50\text{ s}$ | $16.80\text{ s}$ |
| **p05** | $21.00\text{ s}$ | $15.20\text{ s}$ | $14.80\text{ s}$ | $17.50\text{ s}$ | $18.60\text{ s}$ | $22.00\text{ s}$ | $24.70\text{ s}$ |
| **p25** | $62.30\text{ s}$ | $44.10\text{ s}$ | $43.20\text{ s}$ | $48.90\text{ s}$ | $52.80\text{ s}$ | $64.80\text{ s}$ | $48.20\text{ s}$ |
| **Median (p50)** | **114.90 s** | **83.50 s** | **81.40 s** | **88.20 s** | **94.70 s** | **119.20 s** | **81.30 s** |
| **p75** | $187.80\text{ s}$ | $132.80\text{ s}$ | $130.10\text{ s}$ | $139.60\text{ s}$ | $145.20\text{ s}$ | $193.50\text{ s}$ | $119.50\text{ s}$ |
| **p95** | $296.40\text{ s}$ | $187.40\text{ s}$ | $184.20\text{ s}$ | $205.10\text{ s}$ | $198.30\text{ s}$ | $304.80\text{ s}$ | $164.20\text{ s}$ |
| **p99** | $354.20\text{ s}$ | $203.10\text{ s}$ | $200.50\text{ s}$ | $228.40\text{ s}$ | $212.10\text{ s}$ | $361.20\text{ s}$ | $181.70\text{ s}$ |
| **Maximum** | $427.10\text{ s}$ | $211.50\text{ s}$ | $209.80\text{ s}$ | $244.20\text{ s}$ | $219.80\text{ s}$ | $427.10\text{ s}$ | $198.50\text{ s}$ |

### Long-Gap Audit Finding
- The global median transit gap is **114.9 seconds** ($\sim 2\text{ minutes}$).
- Fully **47.88%** ($3,286,075$ candidates) have $\Delta t > 120.0\text{s}$.
- Top 5% of pairs span $\ge 296.4\text{ seconds}$ ($5\text{ minutes}$).
- Because no maximum horizon was set, the candidate generator allows any tracklet occurring at the beginning of a video to pair with any tracklet at the end of the video, provided implied speed is $\le 45\text{ m/s}$.

---

## 5. Implied Speed Feasibility Distribution

Implied velocity $v = d / \Delta t$ across all $6,863,626$ candidates:

| Quantile | Speed (m/s) | Speed (km/h) | Operational Road Classification |
| :--- | :---: | :---: | :--- |
| **Minimum** | $0.03\text{ m/s}$ | $0.11\text{ km/h}$ | Stationary / extended traffic signal waiting |
| **p01** | $0.09\text{ m/s}$ | $0.32\text{ km/h}$ | Severe congestion / multi-cycle red light |
| **p05** | $0.22\text{ m/s}$ | $0.79\text{ km/h}$ | Traffic queue crawl |
| **p25** | $2.37\text{ m/s}$ | $8.53\text{ km/h}$ | Stop-and-go urban movement |
| **Median (p50)** | **6.42 m/s** | **23.11 km/h** | Normal city intersection transit |
| **p75** | $13.38\text{ m/s}$ | $48.17\text{ km/h}$ | Free-flow urban arterial driving |
| **p95** | $31.39\text{ m/s}$ | $113.00\text{ km/h}$ | Highway / arterial transit |
| **p99** | $41.45\text{ m/s}$ | $149.22\text{ km/h}$ | Upper physical highway speed envelope |
| **Maximum** | $45.00\text{ m/s}$ | $162.00\text{ km/h}$ | Enforced production threshold ceiling |

---

## 6. Topology Coverage: Soft Prior vs Non-Edge Pairs

How much of the $6.86\text{M}$ candidate volume is supported by directed camera graph edges versus arbitrary same-scenario camera pairs?

| Category | Candidate Count | Percentage of Candidate Pool |
| :--- | :---: | :---: |
| **Supported by Directed Graph Edge** | $1,432,765$ | **20.87%** |
| **Non-Edge Pairs (Spatial Fallback)** | $5,430,861$ | **79.13%** |
| **Total Candidates** | $\mathbf{6,863,626}$ | $\mathbf{100.00\%}$ |

### Topology Coverage by Scenario

| Scenario | Total Candidates | With Graph Edge | % With Edge | Without Graph Edge | % Without Edge |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **S01** | 142,359 | 131,048 | **92.05%** | 11,311 | 7.95% |
| **S02** | 184,454 | 163,892 | **88.85%** | 20,562 | 11.15% |
| **S03** | 13,522 | 8,241 | **60.94%** | 5,281 | 39.06% |
| **S04** | 234,818 | 91,482 | **38.96%** | 143,336 | 61.04% |
| **S05** | 5,940,452 | 967,318 | **16.28%** | 4,973,134 | **83.72%** |
| **S06** | 348,021 | 70,784 | **20.34%** | 277,237 | **79.66%** |

> **Audit Insight**: In small intersection clusters (S01, S02), camera graph edges cover ~90% of candidates. But in expansive network scenarios (S04, S05, S06), **up to 83.7% of candidates have no graph edge**. The soft prior properly admits them, but without topological constraints, candidate density scales quadratically with tracklet count.

---

## 7. Vehicle Type Distribution

Detector classification consistency across candidate endpoints:

| Agreement Category | Candidate Count | Percentage | Operational Meaning |
| :--- | :---: | :---: | :--- |
| **Exact Match (`EXACT_MATCH`)** | $5,529,187$ | **80.56%** | Origin and destination detectors assign identical class (`car-car`, `truck-truck`, etc.). |
| **Type Mismatch (`MISMATCH`)** | $1,334,439$ | **19.44%** | Detector class disagreement (`car-truck`, `car-bus`, etc.). Preserved for multimodal fusion. |
| **Unknown / Missing** | 0 | 0.00% | Zero null vehicle types in canonical dataset. |

---

## 8. Re-ID Embedding Availability & Model Compatibility

Feature space audit across candidate endpoints:

| Feature Space Category | Candidate Count | Percentage | Multimodal Association Implication |
| :--- | :---: | :---: | :--- |
| **Both Available & Compatible** | $5,966,759$ | **86.93%** | Both use `AICITY_VEHICLE_V1` (512-D). Admissible for cosine similarity. |
| **One Embedding Missing** | $802,968$ | **11.70%** | One endpoint lacks appearance vector. Downstream fusion must use non-ReID cues. |
| **Both Embeddings Missing** | $53,662$ | **0.78%** | Neither endpoint has embedding. Spatial/temporal/OCR cues only. |
| **Incompatible Feature Spaces** | $40,237$ | **0.59%** | Cross-model pairing (`MSMT17` vs `AICITY`). Cosine similarity prohibited. |
| **Total Candidates** | $\mathbf{6,863,626}$ | $\mathbf{100.00\%}$ | Zero candidates pruned due to missing/incompatible Re-ID. |

### CAM_S01_C002 Specific Audit
- Total candidates involving `CAM_S01_C002`: **57,329**.
- Cross-model pairs with AICity cameras: **40,237** (marked `compatible = false`).
- Intra-camera C002 pairs: $0$ (same-camera pairs excluded).

---

## 9. Automatic Number Plate Recognition (ANPR / OCR) Availability

| Endpoint Plate Status | Candidate Count | Percentage | Downstream Match Significance |
| :--- | :---: | :---: | :--- |
| **Neither Endpoint Available** | $6,850,667$ | **99.81%** | CityFlow resolution/angles yield low plate readability; Re-ID/motion must drive fusion. |
| **Origin Only Available** | $6,704$ | **0.10%** | Unilateral plate evidence; cannot compute edit distance. |
| **Destination Only Available** | $6,248$ | **0.09%** | Unilateral plate evidence; cannot compute edit distance. |
| **Both Endpoints Available** | **7** | **0.0001%** | Bilateral readable plates available for exact string comparison. |

---

## 10. Combined Evidence Completeness Tiers

Classification of the $6.86\text{M}$ candidate pool by multimodal evidence richness:

| Tier | Available Modalities | Candidate Count | Share | Fusion Strategy |
| :--- | :--- | :---: | :---: | :--- |
| **HIGH** | Compatible ReID + OCR + Topology Edge + Spatiotemporal | $2,625$ | **0.04%** | Full multimodal confirmation with plate + appearance + route. |
| **MEDIUM-HIGH** | Compatible ReID + Topology Edge + Spatiotemporal | $1,094,431$ | **15.95%** | Robust fusion: Re-ID cosine similarity along known road edges. |
| **MEDIUM** | Compatible ReID + Spatiotemporal (No Graph Edge) | $4,869,703$ | **70.95%** | Re-ID cosine similarity across non-adjacent camera transitions. |
| **LOW-MEDIUM** | Topology Edge + Spatiotemporal (No Compatible Re-ID) | $335,522$ | **4.89%** | Route-constrained tracking for un-embedded or C002 vehicles. |
| **LOW** | Spatiotemporal Only (No ReID, No Topology, No OCR) | $560,592$ | **8.17%** | Weakest candidate pool; relies solely on transit speed plausibility. |
| **OTHER** | Edge/OCR partials | $753$ | **0.01%** | Specialized edge cases. |

---

## 11. Candidate Explosion Root-Cause Analysis

Why do **6,863,626 candidates** survive from $9.16\text{M}$ intra-scenario pairs?

```
Intra-Scenario Theoretical Pairs (9,160,912)
│
├── Same Camera Filter (-741,750 / 8.1%)
├── Negative Time Filter (-712,742 / 7.8%)
├── Excessive Speed Filter (-838,775 / 9.2%)
└── Zero Time Filter (-4,019 / 0.04%)
    │
    ▼
Surviving Candidates: 6,863,626 (74.9% of intra-scenario)
├── 83.7% in Scenario S05 (dense arterial with 3,921 tracklets)
├── 79.1% on Non-Graph-Edge Camera Transitions (5,430,861 pairs)
└── 47.9% across Long Time Horizons delta_t > 120s (3,286,075 pairs)
```

### The Three Drivers of Explosion
1. **Unbounded Temporal Horizon**: Without a realistic temporal window ($T_{\max}$), $3.29\text{M}$ pairs with $\Delta t > 120\text{s}$ survive across video durations up to $430\text{s}$.
2. **Dense Non-Edge Combinatorics in S05**: With 19 cameras in S05, $19 \times 18 = 342$ directed camera pairs exist, but only $50$ have road edges. The remaining 292 pairs generate $4.97\text{M}$ candidates.
3. **Absence of Minimum Velocity Filter**: Because vehicles may wait at traffic lights, $v_{\min} = 0$ is permitted. Consequently, two tracklets separated by $100\text{m}$ occurring $400\text{s}$ apart yield $v = 0.25\text{ m/s}$ and pass the speed gate.

---

## 12. Ground-Truth Recall & False Negative Audit (S01 Ground Truth)

### Ground Truth Mapping Status
- **Scenario S01**: Authoritative ground-truth cross-camera identities exist from official CityFlowV2 annotations (`results/gt_positive_loss_audit.json` / `tracklet_to_gt.json`), comprising **308 true cross-camera positive pairs** across cameras C001, C002, and C003.
- **Scenarios S02–S06**: `CANDIDATE RECALL NOT MEASURABLE FROM CURRENT ARTIFACTS`. While raw video `gt.txt` files exist on disk, no spatiotemporal bounding box consensus mapping has been established to map predicted ByteTrack tracklets to GT vehicle IDs for these cameras.

### S01 Candidate Retrieval Performance

$$\text{S01 Candidate Recall} = \frac{109 \text{ Recovered}}{308 \text{ True Positives}} = \mathbf{35.39\%}$$
$$\text{S01 False Negative Rate} = \frac{199 \text{ Missed}}{308 \text{ True Positives}} = \mathbf{64.61\%}$$

### False Negative Gate Attribution

| Rejection Gate | Missed Pairs Count | % of Missed GT | Root Cause & Physical Mechanism |
| :--- | :---: | :---: | :--- |
| `REJECT_NEGATIVE_TIME` | **182** | **91.46%** | **Overlapping Fields of View**: Cameras C001, C002, and C003 observe the same JFK Rd intersection mast arms. A vehicle enters camera B before leaving camera A. Thus $t_{\text{dest, start}} < t_{\text{orig, end}} \implies \Delta t < 0$. The strict temporal gate unconditionally rejects these true positive transitions. |
| `REJECT_EXCESSIVE_SPEED` | **17** | **8.54%** | **Pole-to-Pole Distortion**: In an intersection, the FOV boundaries touch. A vehicle exits C001 and enters C002 within $0.1\text{s} - 0.4\text{s}$. Dividing the pole-to-pole camera distance ($33.2\text{m}$) by $\Delta t = 0.14\text{s}$ yields $v = 237\text{ m/s}$ ($853\text{ km/h}$), triggering false excessive speed rejection. |
| `REJECT_SAME_CAMERA` | 0 | 0.00% | No cross-camera GT pairs were classified as same-camera. |
| `REJECT_ZERO_OR_INVALID_TIME` | 0 | 0.00% | No GT pairs had exact $\Delta t = 0.0\text{s}$. |
| **Total Missed GT Positives** | **199** | **100.00%** | **Critical structural vulnerability in overlapping FOVs.** |

### Camera-Pair Recall Breakdown (S01)

| Transition Pair | Ground Truth Positives | Recovered Candidates | Missed Pairs | Candidate Recall | Negative Time Missed | Speed Missed |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `CAM_S01_C001 <-> CAM_S01_C002` | 86 | 29 | 57 | **33.72%** | 52 | 5 |
| `CAM_S01_C002 <-> CAM_S01_C003` | 134 | 47 | 87 | **35.07%** | 78 | 9 |
| `CAM_S01_C001 <-> CAM_S01_C003` | 88 | 33 | 55 | **37.50%** | 52 | 3 |
| **Total S01 Transitions** | **308** | **109** | **199** | **35.39%** | **182** | **17** |

---

## 13. Offline Sensitivity Experiments

To evaluate gating thresholds without altering production outputs, two offline experiments were executed across all $7,448$ tracklets:

### Experiment 1: Speed Ceiling Sensitivity ($v_{\max}$)

| Threshold ($v_{\max}$) | Candidate Count | Global Reduction % | S01 GT Recovered | S01 Candidate Recall % |
| :---: | :---: | :---: | :---: | :---: |
| **$20.0\text{ m/s}$ ($72\text{ km/h}$)** | $5,914,668$ | $78.67\%$ | $101 / 308$ | $32.79\%$ |
| **$30.0\text{ m/s}$ ($108\text{ km/h}$)** | $6,469,444$ | $76.67\%$ | $108 / 308$ | $35.06\%$ |
| **$40.0\text{ m/s}$ ($144\text{ km/h}$)** | $6,763,250$ | $75.61\%$ | $108 / 308$ | $35.06\%$ |
| **$45.0\text{ m/s}$ ($162\text{ km/h}$, Prod)** | $\mathbf{6,863,626}$ | $\mathbf{75.25\%}$ | $\mathbf{109 / 308}$ | $\mathbf{35.39\%}$ |
| **$50.0\text{ m/s}$ ($180\text{ km/h}$)** | $6,945,301$ | $74.96\%$ | $112 / 308$ | $36.36\%$ |
| **$60.0\text{ m/s}$ ($216\text{ km/h}$)** | $7,067,787$ | $74.51\%$ | $112 / 308$ | $36.36\%$ |

> **Finding**: Increasing the speed ceiling from $45\text{ m/s}$ to $60\text{ m/s}$ only recovers 3 additional GT pairs (raising recall from 35.39% to 36.36%) while adding $204,161$ candidates. The speed ceiling is **not** the primary cause of missed recall—the unhandled FOV overlap is.

### Experiment 2: Maximum Temporal Horizon Sensitivity ($T_{\max}$)

| Horizon ($T_{\max}$) | Candidate Count | Global Reduction % | S01 GT Recovered | S01 Candidate Recall % |
| :---: | :---: | :---: | :---: | :---: |
| **$30.0\text{ s}$** | $585,368$ | $97.89\%$ | $92 / 308$ | $29.87\%$ |
| **$60.0\text{ s}$** | $1,635,375$ | $94.10\%$ | $107 / 308$ | $34.74\%$ |
| **$120.0\text{ s}$** | $\mathbf{3,577,107}$ | $\mathbf{87.10\%}$ | $\mathbf{109 / 308}$ | $\mathbf{35.39\%}$ |
| **$300.0\text{ s}$** | $6,547,416$ | $76.39\%$ | $109 / 308$ | $35.39\%$ |
| **Unbounded (Current Prod)** | $\mathbf{6,863,626}$ | $\mathbf{75.25\%}$ | $\mathbf{109 / 308}$ | $\mathbf{35.39\%}$ |

> **Critical Discovery**:
> - Between $T_{\max} = 120.0\text{s}$ and Unbounded, **GT recall is 100% identical (109 / 308 = 35.39%)**.
> - Not a single true vehicle transition in S01 took longer than 120 seconds.
> - Allowing $\Delta t > 120\text{s}$ adds **3,286,519 candidates** (nearly 3.3 million pairs) with **ZERO recall benefit**.
> - Setting $T_{\max} = 120\text{s}$ would slash the candidate pool by **47.88%** ($6.86\text{M} \to 3.58\text{M}$) with zero loss of true positives.

---

## 14. Artifact Verification

All audit findings and raw metrics are permanently serialized in:
- `results/layer2_candidate_audit/candidate_retrieval_audit.json`
- `results/layer2_candidate_audit/candidate_retrieval_audit.md`

No production files in `results/layer2_candidates/` or `UrbanTrack_Member1_Handoff 2/` were modified.

---

## 15. Final Assessment & Verdict

The candidate generation stage must be classified as:

$$\mathbf{C = RECALL\ RISK}$$

### Scientific Rationale
1. **The Purpose of Retrieval**: In a two-stage association pipeline, the retrieval stage's cardinal virtue is **recall**. Any true identity pair pruned at Layer 2 is permanently lost to Layer 3 identity fusion, regardless of how strong the appearance, plate, or motion embeddings may be.
2. **Acute Recall Failure on Overlapping Networks**: On intersection clusters (S01), where cameras share fields of view, the current candidate generator incurs a **64.61% false negative rate** (199 / 308 ground-truth positives lost). The strict assumption that $\Delta t = t_{\text{dest, start}} - t_{\text{orig, end}}$ must be positive breaks down whenever cameras observe the same vehicle simultaneously ($\Delta t < 0$).
3. **Compound Permissiveness**: Simultaneously, the generator is over-inclusive on non-adjacent camera pairs and long time horizons, retaining $3.29\text{M}$ pairs with $\Delta t > 120\text{s}$ in S05 with zero recall gain.

While the engineering execution is sound (clean code, 100% unit tests passing, deterministic, 63s runtime, zero memory blowup), the underlying physical assumptions produce a severe **RECALL RISK** for multi-camera intersection tracking.

---

```
LAYER2_CANDIDATE_AUDIT = C
```
