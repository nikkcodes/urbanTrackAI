"""Thin FastAPI adapter over the existing UrbanTrackAI Member 3 engines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException

from backend.analytics.network import UrbanMobilityAnalyzer
from backend.anomaly.detector import AnomalyDetector
from backend.anomaly.models import MobilitySnapshot
from backend.flow.adapters import MockTrajectoryAdapter
from backend.flow.aggregation import ExpectedFlowAggregator
from backend.flow.models import FlowAggregationResult, NormalizedTrajectory
from backend.mobility.graph import MobilityGraph
from backend.simulation.engine import CounterfactualSimulationEngine
from backend.simulation.models import Scenario
from backend.traffic.metrics import TrafficMetricsCalculator

from backend.api.models import SimulationRequest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CITY_NETWORK_PATH = PROJECT_ROOT / "data" / "synthetic" / "city_network.json"
TRAJECTORIES_PATH = PROJECT_ROOT / "data" / "synthetic" / "mock_trajectories.json"


@dataclass(frozen=True)
class EngineContext:
    graph: MobilityGraph
    trajectories: List[NormalizedTrajectory]
    flow_result: FlowAggregationResult
    metrics: list
    phase4_result: Any
    anomaly_result: Any


def _build_context() -> EngineContext:
    graph = MobilityGraph.load_from_json(CITY_NETWORK_PATH)
    trajectories = MockTrajectoryAdapter().adapt(TRAJECTORIES_PATH)
    flow_result = ExpectedFlowAggregator(graph).aggregate(trajectories, include_zero_flow_roads=False)
    metrics = TrafficMetricsCalculator(graph=graph).calculate_metrics(flow_result)
    phase4_result = UrbanMobilityAnalyzer().analyze(flow_result, metrics, graph, top_n=5)
    snapshot = MobilitySnapshot.from_phase4_result(
        snapshot_id="api-current",
        result=phase4_result,
        graph=graph,
        aggregation_duration_hours=1.0,
        metadata={"source": "synthetic", "api": True},
    )
    anomaly_result = AnomalyDetector().compare(snapshot, snapshot, graph)
    return EngineContext(graph, trajectories, flow_result, metrics, phase4_result, anomaly_result)


_context: Optional[EngineContext] = None


def get_context() -> EngineContext:
    global _context
    if _context is None:
        _context = _build_context()
    return _context


def _trajectory_dict(trajectory: NormalizedTrajectory) -> Dict[str, Any]:
    return trajectory.to_dict()


def _route_demand_dict(route: Any) -> Dict[str, Any]:
    return {
        "route_nodes": list(route.route_nodes),
        "road_ids": list(route.road_ids),
        "demand": route.demand,
        "time_window_start": route.time_window_start,
        "time_window_end": route.time_window_end,
    }


def _bottleneck_dict(bottleneck: Any) -> Dict[str, Any]:
    return {
        "road_id": bottleneck.road_id,
        "severity": bottleneck.severity,
        "utilization_ratio": bottleneck.utilization_ratio,
        "congestion_score": bottleneck.congestion_score,
        "hourly_flow": bottleneck.hourly_flow,
        "capacity_vph": bottleneck.capacity_vph,
        "delay_minutes": bottleneck.delay_minutes,
        "bottleneck_score": bottleneck.bottleneck_score,
        "signals": list(bottleneck.signals),
    }


def _anomaly_dict(result: Any) -> Dict[str, Any]:
    return result.to_dict()


def _simulation_dict(result: Any) -> Dict[str, Any]:
    return result.to_dict()


app = FastAPI(
    title="UrbanTrackAI Member 3 API",
    version="7.0.0",
    description="Thin REST adapter over the existing synthetic Member 3 mobility engines.",
)


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "engine": "operational"}


@app.get("/api/network")
def network() -> Dict[str, Any]:
    context = get_context()
    return {
        "name": context.graph.name,
        "nodes": [node.to_dict() for node in sorted(context.graph.all_nodes(), key=lambda item: item.node_id)],
        "roads": [road.to_dict() for road in sorted(context.graph.all_roads(include_closed=True), key=lambda item: item.road_id)],
    }


@app.get("/api/traffic")
def traffic() -> Dict[str, Any]:
    context = get_context()
    return {
        "metrics": [metric.to_dict() for metric in context.metrics],
        "evaluated_roads_count": len(context.metrics),
        "time_window_start": context.flow_result.time_window_start,
        "time_window_end": context.flow_result.time_window_end,
    }


@app.get("/api/analytics/od")
def od_analytics() -> Dict[str, Any]:
    context = get_context()
    od = context.phase4_result.od_analysis
    return {
        "total_demand": od.total_demand,
        "trajectory_count": od.trajectory_count,
        "pairs": [
            {
                "origin": pair.origin,
                "destination": pair.destination,
                "demand": pair.demand,
                "time_window_start": pair.time_window_start,
                "time_window_end": pair.time_window_end,
            }
            for pair in od.matrix.pairs
        ],
        "route_demands": [_route_demand_dict(route) for route in context.phase4_result.route_demands],
    }


@app.get("/api/analytics/bottlenecks")
def bottlenecks() -> Dict[str, Any]:
    context = get_context()
    return {"bottlenecks": [_bottleneck_dict(item) for item in context.phase4_result.bottlenecks]}


@app.get("/api/anomalies")
def anomalies() -> Dict[str, Any]:
    return _anomaly_dict(get_context().anomaly_result)


@app.get("/api/trajectories")
def trajectories() -> Dict[str, Any]:
    context = get_context()
    return {"trajectories": [_trajectory_dict(item) for item in context.trajectories]}


@app.get("/api/trajectories/{track_id}")
def trajectory(track_id: str) -> Dict[str, Any]:
    for item in get_context().trajectories:
        if item.track_id == track_id:
            return _trajectory_dict(item)
    raise HTTPException(status_code=404, detail=f"Trajectory '{track_id}' not found")


@app.post("/api/simulation")
def simulation(request: SimulationRequest) -> Dict[str, Any]:
    context = get_context()
    try:
        scenario = Scenario(
            scenario_id=request.scenario_id,
            name=request.name,
            description=request.description,
            closed_road_ids=tuple(request.closed_road_ids),
            capacity_modifications_vph=dict(request.capacity_modifications_vph),
            speed_modifications_kmph=dict(request.speed_modifications_kmph),
        )
        trajectories = [item for item in context.trajectories if item.time_window_start == "08:00:00"]
        result = CounterfactualSimulationEngine(
            aggregation_duration_hours=0.25,
            time_window_start="08:00:00",
            time_window_end="08:15:00",
        ).simulate(context.graph, trajectories, scenario)
        return _simulation_dict(result)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
