# Member 3 Phase 7: REST API Layer

Phase 7 is a thin FastAPI adapter over the existing Member 3 Phase 1–6 engines. It does not duplicate mobility, flow, traffic, analytics, anomaly, or simulation logic.

## Start

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start locally:

```powershell
uvicorn backend.api.app:app --reload
```

The API uses the existing synthetic city network and mock normalized trajectories until a real Member 2 trajectory source is connected.

## Endpoints

- `GET /api/health`: operational status.
- `GET /api/network`: actual nodes and registered road segments from `MobilityGraph`.
- `GET /api/traffic`: observed Phase 3 traffic metrics only. It does not fabricate zero-flow records for unobserved roads.
- `GET /api/analytics/od`: Phase 4 OD demand and probability-weighted route demand.
- `GET /api/analytics/bottlenecks`: Phase 4 bottleneck records.
- `GET /api/anomalies`: Phase 5 anomaly result, including missing-observation semantics.
- `GET /api/trajectories`: existing normalized synthetic trajectories.
- `GET /api/trajectories/{track_id}`: one trajectory or HTTP 404.
- `POST /api/simulation`: Phase 6 scenario simulation using the existing `Scenario` and `CounterfactualSimulationEngine` contracts.

## Simulation Request

```json
{
  "scenario_id": "close_R01",
  "name": "Close R01",
  "description": "Local counterfactual closure",
  "closed_road_ids": ["R01"],
  "capacity_modifications_vph": {},
  "speed_modifications_kmph": {}
}
```

PowerShell example:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/simulation `
  -ContentType 'application/json' `
  -Body '{"scenario_id":"close_R01","name":"Close R01","closed_road_ids":["R01"]}'
```

The response contains the existing Phase 6 scenario result: baseline and scenario summaries, road impacts, OD route impacts, unroutable demand, decision recommendation, and explanation.

## Units

The API preserves the existing semantics:

- `expected_flow`: accumulated expected vehicle/vehicle-weight count over an aggregation window.
- `hourly_flow`: vehicles/hour.
- `capacity_vph`: vehicles/hour.
- Phase 3 utilization and BPR metrics use hourly flow.

## Architecture and Limitations

The API loads one cached synthetic engine context for local demonstration. It is intentionally stateless and has no database, authentication, streaming, or deployment layer. Phase 8 can consume the JSON response shapes without changing the underlying Member 3 engine contracts. A future real trajectory adapter can replace the current mock source behind the API context.
