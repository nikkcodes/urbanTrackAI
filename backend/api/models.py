"""Pydantic request models for the thin Phase 7 API boundary."""

from __future__ import annotations

from typing import Dict, List

from pydantic import BaseModel, ConfigDict, Field


class SimulationRequest(BaseModel):
    """Request payload mapped directly to the existing Phase 6 Scenario model."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = ""
    closed_road_ids: List[str] = Field(default_factory=list)
    capacity_modifications_vph: Dict[str, float] = Field(default_factory=dict)
    speed_modifications_kmph: Dict[str, float] = Field(default_factory=dict)


class ApiError(BaseModel):
    detail: str
