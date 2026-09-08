"""Adapters for converting upstream or mock trajectory formats into normalized trajectories.

Maintains a strict boundary between external trajectory representations (e.g. mock
fixtures, future Member 2 pipeline outputs) and the normalized internal models used
by the expected flow aggregation engine.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Union

from backend.flow.models import CandidateRoute, NormalizedTrajectory


class BaseTrajectoryAdapter(ABC):
    """Abstract base class for trajectory adapters."""

    @abstractmethod
    def adapt(self, raw_data: Any) -> List[NormalizedTrajectory]:
        """Convert an external or raw dataset into a list of NormalizedTrajectory instances.

        Args:
            raw_data: External trajectory payload (e.g., dict, list, file path).

        Returns:
            List of normalized probabilistic trajectories.
        """
        pass

    @abstractmethod
    def adapt_one(self, raw_item: Dict[str, Any]) -> NormalizedTrajectory:
        """Convert a single external trajectory dictionary into a NormalizedTrajectory.

        Args:
            raw_item: Dictionary representing a single trajectory.

        Returns:
            A single NormalizedTrajectory instance.
        """
        pass


class MockTrajectoryAdapter(BaseTrajectoryAdapter):
    """Adapter for converting mock/synthetic trajectory payloads into NormalizedTrajectory instances.

    Supports reading directly from in-memory dict/list structures or JSON file paths.
    """

    def adapt(self, raw_data: Union[Dict[str, Any], List[Dict[str, Any]], str, Path]) -> List[NormalizedTrajectory]:
        """Convert mock trajectory data into a list of NormalizedTrajectory objects.

        Args:
            raw_data: May be a dictionary with a "trajectories" key, a list of trajectory dicts,
                      or a path to a JSON file.

        Returns:
            List of NormalizedTrajectory instances.
        """
        payload = raw_data

        # If file path is passed, read JSON
        if isinstance(raw_data, (str, Path)):
            file_path = Path(raw_data)
            if not file_path.exists():
                raise FileNotFoundError(f"Mock trajectory file not found: {file_path}")
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)

        items: List[Dict[str, Any]]
        if isinstance(payload, dict):
            if "trajectories" in payload:
                items = payload["trajectories"]
            else:
                # Could be a single trajectory dict
                if "track_id" in payload:
                    return [self.adapt_one(payload)]
                items = []
        elif isinstance(payload, list):
            items = payload
        else:
            raise TypeError(f"Unsupported payload type for MockTrajectoryAdapter: {type(payload)}")

        return [self.adapt_one(item) for item in items]

    def adapt_one(self, raw_item: Dict[str, Any]) -> NormalizedTrajectory:
        """Convert a single mock trajectory dictionary into a NormalizedTrajectory.

        Args:
            raw_item: Dictionary representing a mock vehicle trajectory.

        Returns:
            A validated NormalizedTrajectory instance.
        """
        if not isinstance(raw_item, dict):
            raise TypeError(f"Expected dict for trajectory item, got {type(raw_item)}")

        track_id = str(raw_item["track_id"])
        origin_node = str(raw_item["origin_node"])
        destination_node = str(raw_item["destination_node"])

        # Candidate routes conversion
        raw_routes = raw_item.get("candidate_routes", [])
        candidate_routes: List[CandidateRoute] = []
        for r in raw_routes:
            if isinstance(r, CandidateRoute):
                candidate_routes.append(r)
            elif isinstance(r, dict):
                # Support "path" or "nodes" keys in raw data
                nodes = r.get("nodes") or r.get("path")
                if nodes is None:
                    raise KeyError(f"Candidate route missing 'nodes' or 'path' in trajectory '{track_id}'")
                prob = float(r["probability"])
                meta = dict(r.get("metadata", {}))
                candidate_routes.append(
                    CandidateRoute(nodes=list(nodes), probability=prob, metadata=meta)
                )
            else:
                raise TypeError(f"Expected dict or CandidateRoute, got {type(r)}")

        # Vehicle weight / count support
        vehicle_weight = float(
            raw_item.get("vehicle_weight", raw_item.get("weight", raw_item.get("count", 1.0)))
        )

        timestamp = raw_item.get("timestamp")
        time_window_start = raw_item.get("time_window_start")
        time_window_end = raw_item.get("time_window_end")
        metadata = dict(raw_item.get("metadata", {}))

        return NormalizedTrajectory(
            track_id=track_id,
            origin_node=origin_node,
            destination_node=destination_node,
            candidate_routes=candidate_routes,
            vehicle_weight=vehicle_weight,
            timestamp=timestamp,
            time_window_start=time_window_start,
            time_window_end=time_window_end,
            metadata=metadata,
        )
