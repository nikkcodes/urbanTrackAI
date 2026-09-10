"""
Observation module re-exporting the primary Observation data model.
"""

from schemas.observation_schema import CameraMetadata, Observation

__all__ = ["Observation", "CameraMetadata"]
