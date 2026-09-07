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

## Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.14.0)

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Run the Demonstration
Run the executable smoke test to verify graph loading, candidate route discovery, road incident closure, and restoration:
```powershell
python run_demo.py
```

### 3. Run Unit Tests
Execute the comprehensive test suite:
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
│   │   └── __init__.py            # API layer placeholder
│   └── mobility/
│       ├── __init__.py            # Mobility engine public exports
│       ├── config.py              # Routing parameters & unit constants
│       ├── graph.py               # Directed MobilityGraph with closure support
│       ├── models.py              # Validated Node and RoadSegment models
│       └── routes.py              # Candidate route discovery & travel metrics
│
├── data/
│   └── synthetic/
│       └── city_network.json      # Deterministic 14-junction test network
│
├── docs/
│   └── member3_mobility.md        # Technical architecture documentation
│
├── tests/
│   ├── __init__.py
│   └── mobility/
│       ├── __init__.py
│       ├── test_graph.py          # Unit tests for graph, nodes, roads, closures
│       └── test_routes.py         # Unit tests for routing and travel-time metrics
│
├── run_demo.py                    # Smoke test & demonstration script
├── requirements.txt               # Dependencies (networkx)
├── .env.example                   # Environment configuration template
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

---

## Roadmap: Subsequent Development Phases

- **Phase 2: Probabilistic Flow Aggregation**:
  - Ingesting probabilistic trajectory candidates from Member 2.
  - BPR (Bureau of Public Roads) volume-delay functions for dynamic travel time under congestion.
- **Phase 3: Traffic Analytics & Bottleneck Forecasting**:
  - Origin-Destination (OD) matrix generation across temporal slices.
  - Centrality and spillback bottleneck detection.
- **Phase 4: Counterfactual Simulation Engine**:
  - Automated what-if scenario testing for planned road closures and emergency rerouting.
- **Phase 5: Mobility APIs & GIS Dashboard**:
  - REST/WebSocket APIs for mobility telemetry.
  - React/Mapbox/Leaflet interactive dashboard integration.