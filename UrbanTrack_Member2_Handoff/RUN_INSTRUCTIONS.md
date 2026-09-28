# UrbanTrack AI — Member 2 Run & Reproduction Instructions

**Target Audience**: Member 3 Developers & Downstream Integrators  
**Scope**: Reproduction, execution, validation, and programmatic consumption of Member 2 outputs  

---

## 1. Environment Requirements

Member 2 is built with a lightweight, deterministic dependency footprint. It requires **no GPU**, **no Docker**, and **no heavyweight machine learning frameworks** (such as PyTorch or TensorFlow) at runtime or for downstream consumption.

### Verified Runtime Environment
- **Operating System**: macOS (tested on Darwin ARM64) / Linux (Ubuntu 20.04+)
- **Python Version**: Python 3.10, 3.11, 3.12, or 3.13 (verified on `Python 3.13.5`)
- **Required Python Packages**:
  - `numpy >= 1.24.0` (verified with `numpy 2.4.1`)
  - `scipy >= 1.10.0` (verified with `scipy 1.17.0`)
  - `pytest >= 7.0.0` (verified with `pytest 9.1.1` for running test suites)
- **Standard Library Modules**: `math`, `json`, `time`, `pathlib`, `typing`, `dataclasses`, `hashlib`, `collections`.

```bash
# Optional: Create and activate a clean virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the minimal required packages
pip install numpy scipy pytest
```

---

## 2. Reading Member 2 Outputs (Quick Start for Member 3)

Member 3 can consume the structured outputs directly via standard Python JSON loading without importing internal Member 2 algorithms:

```python
import json
from pathlib import Path

HANDOFF_DIR = Path("UrbanTrack_Member2_Handoff")

# 1. Load Inferred Vehicle Identities (355 clusters)
with open(HANDOFF_DIR / "outputs/inferred_identities.json") as f:
    identities = json.load(f)
print(f"Loaded {len(identities)} inferred vehicle hypotheses.")
# Access confirmed vehicles:
confirmed = [v for v in identities if v["admission_status"] == "confirmed"]
print(f"Confirmed multi-observation clusters: {len(confirmed)}")

# 2. Load Trajectories (355 trajectory objects)
with open(HANDOFF_DIR / "outputs/trajectories.json") as f:
    trajectories = json.load(f)
print(f"Loaded {len(trajectories)} trajectories.")
# NOTE: Coordinates are local planar meters (cityflow_world), NOT WGS84 GPS.

# 3. Load Camera States & Telemetry (C001, C002, C003)
with open(HANDOFF_DIR / "outputs/camera_states.json") as f:
    camera_states = json.load(f)
for cam_id, state in camera_states.items():
    print(f"{cam_id}: Reliability={state['camera_reliability']:.2f}, "
          f"Re-ID Guard={state['reid_compatibility_status']}")

# 4. Load Graph Topology (384 nodes, 36 edges)
with open(HANDOFF_DIR / "outputs/identity_graph.json") as f:
    graph = json.load(f)
print(f"Nodes: {len(graph['nodes'])}, Confirmed Edges: {len(graph['edges'])}")
```

---

## 3. Validating the Handoff Package

Run this automated validation command to verify schema consistency, numeric integrity (zero NaN / Inf), and file availability:

```bash
python3 -c "
import json
from pathlib import Path

handoff = Path('UrbanTrack_Member2_Handoff')
assert (handoff / 'HANDOFF_MANIFEST.json').exists(), 'Manifest missing!'

for json_file in handoff.glob('**/*.json'):
    with open(json_file) as f:
        data = json.load(f)
    text = json_file.read_text()
    assert 'NaN' not in text, f'NaN found in {json_file}'
    assert 'Infinity' not in text, f'Infinity found in {json_file}'
    print(f'✓ Validated: {json_file.relative_to(handoff)}')

print('\nALL HANDOFF ARTIFACTS VERIFIED AND NUMERICALLY SAFE.')
"
```

---

## 4. Running the Member 2 Test Suite

To verify that the entire repository baseline passes without regressions:

```bash
# Run all 399 repository tests
pytest tests/ -q

# Expected Output:
# ........................................................................................
# 399 passed in ~49s
```

To run individual sub-suites:
```bash
# Member 1 integration tests (11 tests)
pytest tests/test_aicity_member1_integration.py -v

# Scientific validation tests (8 tests)
pytest tests/test_aicity_validation.py -v
```

---

## 5. Regenerating Member 2 Outputs

If Member 3 wishes to regenerate the validation artifacts from scratch using the raw inputs:

### Input Dependencies Required in Repository
1. `UrbanTrack_Member1_Handoff/output/` (Member 1 perception feeds for C001, C002, C003)
2. `data/aicity_ground_truth/cam_timestamp/S01.txt` (Official camera synchronization)
3. `data/aicity_ground_truth/calibration/` (Official 3x3 homography matrices)
4. `data/aicity_ground_truth/gt/gt.txt` (Official AI City challenge ground truth)

### Execution Command
```bash
# Run the full end-to-end AI City validation and evaluation pipeline
python3 scripts/run_aicity_validation.py

# Generated Outputs:
# - results/aicity_validation/metrics.json
# - results/aicity_validation/validation_summary.json
# - results/aicity_validation/ablation_results.json
# - results/aicity_validation/robustness_results.json
# - results/aicity_validation/scalability_results.json
# - results/aicity_validation/leakage_audit_report.json
```

### Running the Live Demonstration Script
```bash
# Run the interactive end-to-end hackathon demonstration
python3 scripts/demo_hackathon_pipeline.py
```
This script demonstrates:
- Ground-plane projection with horizon safeguard ($|W| < 0.50$).
- Detection of C002 Re-ID model mismatch and dynamic guard activation.
- Trajectory reconstruction with calibrated confidence and explicit uncertainty fallbacks.

---

## 6. Directory Map of Outputs & Artifacts

```
UrbanTrack_Member2_Handoff/
├── HANDOFF_MANIFEST.json        <- Package index, hashes, data scope, and versioning
├── MEMBER2_HANDOFF.md           <- Executive architecture, boundaries, and guarantees
├── INTERFACE_CONTRACT.md        <- Field-by-field normative data contract
├── DATA_DICTIONARY.md           <- Semantic catalog answering the 8 canonical questions
├── SOURCE_AND_PROVENANCE.md     <- Lineage breakdown (OBSERVED, INFERRED, GT, SIMULATED)
├── VALIDATION_STATUS.md         <- Comprehensive empirical audit report & limitations
├── RUN_INSTRUCTIONS.md          <- Execution, reproduction, and test commands
├── schemas/                     <- Formal JSON Schemas (Draft-07)
│   ├── inferred_identity_schema.json
│   ├── trajectory_schema.json
│   ├── identity_graph_schema.json
│   ├── camera_state_schema.json
│   ├── uncertainty_schema.json
│   └── anomaly_input_schema.json
├── outputs/                     <- Standalone production-ready data artifacts
│   ├── inferred_identities.json (355 vehicle clusters)
│   ├── trajectories.json        (355 trajectory objects)
│   ├── camera_states.json       (C001, C002, C003 telemetry & state)
│   ├── identity_graph.json      (384 nodes, 36 edges)
│   ├── uncertainty.json         (Platt scaling parameters & metrics)
│   └── validation_summary.json  (Consolidated validation ledger)
├── examples/                    <- Isolated concrete examples from real outputs
│   ├── confirmed_vehicle.json
│   ├── single_camera_vehicle.json
│   ├── ambiguous_vehicle.json
│   ├── rejected_match.json
│   ├── multi_camera_vehicle.json
│   └── uncertain_trajectory.json
└── validation/                  <- Official validation artifacts
    ├── FINAL_METRICS.json
    ├── FINAL_VALIDATION_STATUS.md
    ├── ABLATION_RESULTS.json
    ├── ROBUSTNESS_RESULTS.json
    └── SCALABILITY_RESULTS.json
```
