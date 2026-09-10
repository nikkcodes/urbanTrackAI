"""
UrbanTrack AI - Anomaly Detection & Investigation Subsystem (Day 7).

Provides modular, explainable anomaly detection for:
- Trajectory-level movement (travel-time, route deviation, physical consistency)
- Road-level flow (over-capacity, unexpected demand)
- Network-level patterns (structural bottlenecks, congestion spread)
"""

from .investigation_engine import InvestigationEngine
from .network_anomaly import NetworkAnomalyDetector
from .road_anomaly import RoadAnomalyDetector
from .trajectory_anomaly import TrajectoryAnomalyDetector

__all__ = [
    "InvestigationEngine",
    "TrajectoryAnomalyDetector",
    "RoadAnomalyDetector",
    "NetworkAnomalyDetector",
]
