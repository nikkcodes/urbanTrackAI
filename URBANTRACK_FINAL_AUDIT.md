# UrbanTrack AI — Final Evidence Audit

Generated: `2026-09-17T09:04:57.012166+00:00`  
Source command: `python scripts/reproduce_all.py`

This is a fact-only audit. It does not assign a hackathon score.

## Capability matrix

| Capability | Status | Evidence |
|---|---|---|
| `canonical_observation_contract` | **VERIFIED_BY_EXECUTION** | stage_2_semantic_contract |
| `native_cityflowv2_ingestion` | **VERIFIED_BY_EXECUTION** | stage_7c_cityflowv2_s01 |
| `ground_truth_isolation` | **VERIFIED_BY_EXECUTION** | cityflow metadata ground_truth_inference_leakage=false |
| `identity_fusion_and_identity_graph` | **VERIFIED_BY_EXECUTION** | stage_5_full_fusion_real_data, stage_7b_multicamera_benchmark |
| `osnet_512d_boundary` | **VERIFIED_BY_EXECUTION** | stage_3_real_member1_feed and CityFlow missing-evidence flags |
| `dev_fit_freeze_holdout_calibration` | **VERIFIED_BY_EXECUTION** | stage_7d_probability_calibration |
| `trajectory_and_missing_camera_reasoning` | **VERIFIED_BY_EXECUTION** | stage_11_trajectory_inference |
| `robustness_adversarial_counterfactual` | **VERIFIED_BY_EXECUTION** | stage_9_degradation_benchmark, stage_10_adversarial_suite |
| `scalability_1k_5k_10k` | **VERIFIED_BY_EXECUTION** | stage_8c_large_scale_candidate_pipeline |
| `reproducible_machine_json` | **VERIFIED_BY_EXECUTION** | benchmark_results.json and this audit |
| `frontend_map_visualization` | **NOT_IMPLEMENTED** | No frontend/map application present in supplied project. |
| `production_deployment` | **NOT_VERIFIED** | No deployment environment or live multi-camera service supplied. |

## Measured CityFlow S01 result

- Frame observations: **98180**
- Tracklet summaries: **1301**
- Cameras: **5**
- Identity pair F1: **0.0**
- Candidate positive recall: **0.0**
- Ground truth used for evaluation only; no identity labels entered inference.

## Calibration protocol

- DEV/FIT pairs: **9625**
- HOLDOUT pairs: **4125**
- Frozen parameters: **a=6.7645, b=-3.6465**
- HOLDOUT Brier score after calibration: **0.0563**

## Limitations

- The extracted CityFlow S01 tracker file contains no 512-D appearance vectors or license plates; its identity result is therefore a conservative no-edge baseline, not a Re-ID accuracy claim.
- CityFlow video files were not extracted because the archive is approximately 16 GB compressed; source video paths remain in provenance.
- The supplied repository has backend inference and audit tooling but no frontend/map implementation.
