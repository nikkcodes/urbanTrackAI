# UrbanTrackAI

UrbanTrackAI is an intelligent, city-scale vehicle mobility and decision-intelligence platform. It transforms distributed CCTV and ANPR sensor observations into situational awareness, probabilistic network flows, bottleneck forecasting, and counterfactual decision simulation.

---

## System Architecture & Team Roles

The UrbanTrackAI pipeline connects multi-camera computer vision to city-scale mobility intelligence:

```text
CCTV / ANPR Video Feeds
         ↓
Perception (Member 1: Vehicle Detection, Plate OCR, Appearance Re-ID)
         ↓
Inference & Fusion (Member 2: Tracklet Association, Probabilistic Trajectories)
         ↓
City Mobility Graph (Member 3: Road Network Topology, Dynamic Metrics, Routing)  <-- Current Focus
         ↓
Mobility Intelligence (OD Matrix, Bottleneck Forecasting, Network Anomalies)
         ↓
Counterfactual Simulation ("What-If" Road Closures, Incident Rerouting)
         ↓
GIS Dashboard (Spatial Visualization, Flow Heatmaps, Incident Controls)
```

### Member 3 Role: Urban Mobility & Decision Intelligence Engineer
- Maintains the directed road network topology and edge capacities.
- Computes baseline travel times, distances, and multi-candidate paths.
- Provides road closure and restoration abstractions for incident management.
- Consumes probabilistic trajectory distributions from Member 2 to compute flow aggregation (future phases).
- Powers counterfactual simulation and GIS mobility intelligence APIs (future phases).

---

## Phase 1: Mobility Engine Foundation

Phase 1 establishes a clean, tested, and extensible foundation for the city road network layer:

- **Data Models (`backend/mobility/models.py`)**:
  - `Node`: Typed representation of intersections and sensor locations with coordinate support.
  - `RoadSegment`: Directed edges with distance ($km$), speed limit ($km/h$), capacity ($vph$), free-flow travel time ($min$), and closure state.
  - Strict validation against negative lengths, zero speeds, empty IDs, NaN values, and self-loops.
- **Mobility Graph (`backend/mobility/graph.py`)**:
  - Encapsulates NetworkX `DiGraph` for directed graph operations.
  - Non-destructive `close_road()` and `restore_road()` operations.
  - Deep copy isolation for safe counterfactual scenario evaluations.
  - JSON serialization and deserialization.
- **Route Discovery (`backend/mobility/routes.py`)**:
  - Multi-candidate route discovery via `get_candidate_routes()`.
  - Route distance and free-flow travel-time calculation.
  - Dynamic exclusion of closed road segments.
- **Synthetic City (`data/synthetic/city_network.json`)**:
  - Deterministic 14-junction, 28-road network with multiple alternative corridors (expressways, arterials, urban collectors).
- **Comprehensive Unit Tests (`tests/mobility/`)**:
  - 28 test cases verifying graph creation, model validation, route discovery, closures, and integrity.
- **Smoke Test Demo (`run_demo.py`)**:
  - End-to-end executable demonstration.

---

## Phase 2: Probabilistic Flow Aggregation

Phase 2 bridges probabilistic vehicle tracking with network-level road traffic analytics:

- **Normalized Trajectory Layer (`backend/flow/models.py`)**:
  - `CandidateRoute`: Ordered node sequences with valid probability values $[0.0, 1.0]$.
  - `NormalizedTrajectory`: Decoupled internal representation carrying track IDs, OD endpoints, vehicle weights, time windows, and candidate routes.
  - Strict non-silent probability validation: sum of candidate route probabilities must equal $1.0 \pm 10^{-6}$.
  - `RoadFlow` & `FlowAggregationResult`: Structured flow output retaining underlying trajectories for Phase 4 OD analysis.
- **Adapter Boundary (`backend/flow/adapters.py`)**:
  - `BaseTrajectoryAdapter`: Abstract adapter interface for future Member 2 integration.
  - `MockTrajectoryAdapter`: Ingests synthetic mock trajectory payloads from JSON or memory without leaking mock details into aggregation logic.
- **Aggregation Engine (`backend/flow/aggregation.py`)**:
  - `ExpectedFlowAggregator`: Maps candidate route node sequences to directed `RoadSegment` IDs in the `MobilityGraph`.
  - Preserves uncertainty across candidate routes: $\text{expected flow} += \text{weight} \times \text{probability}$.
  - Duplicate-road traversal rule: loops in a candidate route count the road once per route for expected vehicle flow.
  - Supports partitioned aggregation by time window (`aggregate_by_time_window`).
- **Synthetic Mock Fixture (`data/synthetic/mock_trajectories.json`)**:
  - Isolated test fixture covering multi-vehicle road sharing, split-route probabilities, vehicle weights, and time windows.
- **Executable Demo (`run_phase2_demo.py`)**:
  - Runnable demonstration aggregating flows onto the synthetic city network.

---

## Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.14.0)

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run Demonstrations
Phase 1 Mobility Engine demo:
```powershell
python run_demo.py
```

Phase 2 Flow Aggregation Engine demo:
```powershell
python run_phase2_demo.py
```

### 3. Run Unit Tests
Execute the comprehensive test suite across all phases:
```powershell
python -m unittest discover -s tests -v
```

---

## Repository Structure

```text
urbanTrackAI/
├── backend/
│   ├── __init__.py
│   ├── api/
│   │   └── __init__.py            # API layer placeholder (future phase)
│   ├── flow/                      # Phase 2: Flow Aggregation Engine
│   │   ├── __init__.py            # Flow engine public exports
│   │   ├── adapters.py            # Base and Mock trajectory adapters
│   │   ├── aggregation.py         # ExpectedFlowAggregator
│   │   └── models.py              # CandidateRoute, NormalizedTrajectory, RoadFlow
│   └── mobility/                  # Phase 1: Mobility Engine
│       ├── __init__.py            # Mobility engine public exports
│       ├── config.py              # Routing parameters & unit constants
│       ├── graph.py               # Directed MobilityGraph with closure support
│       ├── models.py              # Validated Node and RoadSegment models
│       └── routes.py              # Candidate route discovery & travel metrics
│
├── data/
│   └── synthetic/
│       ├── city_network.json      # Deterministic 14-junction test network
│       └── mock_trajectories.json # Phase 2 synthetic trajectory test fixture
│
├── docs/
│   ├── member3_mobility.md        # Phase 1 technical architecture documentation
│   └── member3_phase2.md          # Phase 2 flow aggregation documentation
│
├── tests/
│   ├── __init__.py
│   ├── flow/                      # Phase 2 unit tests
│   │   ├── __init__.py
│   │   ├── test_adapters.py       # Trajectory adapter tests
│   │   ├── test_aggregation.py    # Flow aggregation engine tests (14 required tests)
│   │   └── test_models.py         # Trajectory and route model validation tests
│   └── mobility/                  # Phase 1 unit tests
│       ├── __init__.py
│       ├── test_graph.py          # Unit tests for graph, nodes, roads, closures
│       └── test_routes.py         # Unit tests for routing and travel-time metrics
│
├── run_demo.py                    # Phase 1 demonstration script
├── run_phase2_demo.py             # Phase 2 demonstration script
├── requirements.txt               # Dependencies (networkx)
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## Roadmap: Subsequent Development Phases

- **Phase 3: Traffic Volume, Utilization & Congestion Metrics**:
  - Capacity utilization ($V/C$).
  - Dynamic BPR (Bureau of Public Roads) travel times under congestion.
  - Congestion index & speed-flow metrics.
- **Phase 4: OD Analysis & Bottleneck Identification**:
  - Origin-Destination (OD) matrix generation from retained trajectory data.
  - Critical corridor analysis and bottleneck detection.
  - Counterfactual closure simulations.
- **Phase 5: Mobility APIs & GIS Dashboard**:
  - REST/WebSocket APIs for mobility telemetry.
  - React/Mapbox/Leaflet interactive dashboard integration.