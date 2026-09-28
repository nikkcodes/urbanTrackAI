"""
UrbanTrack AI — Official AI City 2022 Track 1 Camera Synchronization Module.

Parses official camera start timestamps from CityFlowV2 / AI City 2022 cam_timestamp metadata.
Strictly preserves original video-relative time while providing explicit synchronized timeline fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import hashlib

from schemas.observation_schema import Observation


def normalize_camera_key(camera_id: str) -> str:
    """Normalize camera key to both short (c001) and long (CAM_S01_C001) lookup forms."""
    cam_clean = camera_id.strip().lower()
    if "_" in cam_clean:
        parts = cam_clean.split("_")
        return parts[-1]  # e.g. c001
    return cam_clean


class AICitySynchronizer:
    """
    Authoritative synchronization loader and processor for AI City 2022 Track 1.
    """

    def __init__(
        self,
        sync_file: Union[str, Path] = "data/aicity_ground_truth/cam_timestamp/S01.txt",
    ) -> None:
        self.sync_file = Path(sync_file)
        self.offsets: Dict[str, float] = {}
        self.file_sha256: Optional[str] = None
        self.scenario_id: str = "S01"
        self._load()

    def _load(self) -> None:
        if not self.sync_file.is_file():
            raise FileNotFoundError(f"AI City synchronization file not found: {self.sync_file}")

        content = self.sync_file.read_bytes()
        self.file_sha256 = hashlib.sha256(content).hexdigest()

        lines = content.decode("utf-8").strip().splitlines()
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                cam_name = parts[0].strip().lower()
                offset_sec = float(parts[1].strip())
                self.offsets[cam_name] = offset_sec
                # Also store canonical long form
                if cam_name.startswith("c"):
                    cam_num = cam_name[1:]
                    long_id = f"CAM_{self.scenario_id}_C{cam_num.upper()}"
                    self.offsets[long_id] = offset_sec

    def get_offset(self, camera_id: str) -> float:
        """
        Return the authoritative starting offset in seconds for a camera.
        Defaults to 0.0 if not found.
        """
        if camera_id in self.offsets:
            return self.offsets[camera_id]
        key = normalize_camera_key(camera_id)
        if key in self.offsets:
            return self.offsets[key]
        return 0.0

    def get_synchronized_time(self, camera_id: str, video_relative_seconds: float) -> float:
        """
        Compute synchronized elapsed seconds:
        t_sync = t_video_relative + offset
        """
        offset = self.get_offset(camera_id)
        return round(float(video_relative_seconds) + offset, 4)

    def attach_synchronization(
        self,
        observations: List[Observation],
    ) -> List[Observation]:
        """
        Attach official synchronization metadata to canonical Observation instances.

        Guarantees:
        - obs.timestamp_seconds remains the authoritative video-relative elapsed time.
        - obs.timestamp_semantics remains 'video_relative'.
        - obs.clock_offset_seconds is set to the official camera start offset.
        - obs.synchronized_timestamp_seconds stores video_relative + clock_offset.
        - obs.time_reference_id remains camera_id for video-relative context.
        - Synchronized timeline provenance is attached to source_provenance.
        """
        for obs in observations:
            offset = self.get_offset(obs.camera_id)
            sync_ts = round(float(obs.timestamp_seconds) + offset, 4)

            obs.clock_offset_seconds = offset
            setattr(obs, "synchronized_timestamp_seconds", sync_ts)
            setattr(obs, "synchronized_timestamp_semantics", "aicity_official_synchronized")
            setattr(obs, "synchronized_time_reference_id", f"AICity_{self.scenario_id}_Global")

            if obs.source_provenance is None:
                obs.source_provenance = {}
            obs.source_provenance["synchronization"] = {
                "source_file": str(self.sync_file),
                "file_sha256": self.file_sha256,
                "camera_offset_seconds": offset,
                "synchronized_timestamp_seconds": sync_ts,
                "status": "official_metadata_applied",
            }

        return observations

    def to_dict(self) -> Dict[str, Any]:
        """Export synchronization summary for diagnostics and provenance artifacts."""
        return {
            "source_file": str(self.sync_file),
            "file_sha256": self.file_sha256,
            "scenario": self.scenario_id,
            "offsets_by_camera": {
                k: v for k, v in self.offsets.items() if k.startswith("CAM_")
            },
            "raw_offsets": {
                k: v for k, v in self.offsets.items() if not k.startswith("CAM_")
            },
        }
