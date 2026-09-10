"""
Privacy Guard and Role-Based Access Views for UrbanTrack AI.

Enforces:
1. HMAC-SHA256 pseudonymization of license plates.
2. Separation between physical traffic analytics and personal identifiers.
3. Configurable retention window enforcement.
4. Structured immutable audit logging for identity queries.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
import hashlib
import hmac
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from schemas.observation_schema import Observation


class AccessRole(str, Enum):
    """Role definitions for privacy-preserving data access."""
    ANALYTICS = "analytics"  # Urban planners / Traffic engineers: aggregate PCU demand only, no plates
    AUDIT = "audit"          # Compliance auditors: pseudonymized plates, consistency ledgers
    ADMIN = "admin"          # Authorized security operators: raw identities with mandatory audit event


@dataclass
class AuditEvent:
    """Immutable audit record for identity queries."""
    event_id: str
    timestamp: float
    iso_timestamp: str
    requester_id: str
    role: str
    purpose: str
    target_identifier: str
    access_granted: bool
    fields_accessed: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def pseudonymize_plate(plate: Optional[str], salt: str = "urbantrack_default_salt_2026") -> Optional[str]:
    """
    Produce a deterministic, cryptographically keyed pseudonym for a license plate string.
    Uses HMAC-SHA256 with a secure salt.

    Returns:
        String of form 'PSEUDO_PLATE_<12-hex-chars>' or None if plate is None.
    """
    if plate is None:
        return None
    clean = "".join(c for c in str(plate).upper() if c.isalnum())
    if not clean:
        return None
    h = hmac.new(salt.encode("utf-8"), clean.encode("utf-8"), hashlib.sha256).hexdigest()
    return f"PSEUDO_PLATE_{h[:12].upper()}"


class PrivacyGuard:
    """
    Privacy governance engine implementing role-based filtering,
    pseudonymization, retention policies, and immutable query auditing.
    """

    def __init__(
        self,
        salt: str = "urbantrack_production_salt_secure_2026",
        retention_horizon_seconds: float = 2592000.0,  # 30 days default
    ) -> None:
        self.salt = salt
        self.retention_horizon_seconds = retention_horizon_seconds
        self._audit_log: List[AuditEvent] = []

    def log_query(
        self,
        requester_id: str,
        role: AccessRole,
        purpose: str,
        target_identifier: str,
        fields_accessed: List[str],
        access_granted: bool = True,
    ) -> AuditEvent:
        """Record an immutable audit event whenever sensitive identity data is queried."""
        now = time.time()
        event_id = f"AUDIT_{len(self._audit_log) + 1:06d}_{int(now)}"
        event = AuditEvent(
            event_id=event_id,
            timestamp=now,
            iso_timestamp=datetime.fromtimestamp(now).isoformat(),
            requester_id=requester_id,
            role=role.value,
            purpose=purpose,
            target_identifier=target_identifier,
            access_granted=access_granted,
            fields_accessed=fields_accessed,
        )
        self._audit_log.append(event)
        return event

    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Return the immutable query audit trail."""
        return [e.to_dict() for e in self._audit_log]

    def filter_observation(
        self,
        obs: Observation,
        role: AccessRole,
        requester_id: str = "anonymous",
        purpose: str = "general_processing",
    ) -> Dict[str, Any]:
        """
        Apply role-based privacy masking to an individual observation.

        - ANALYTICS: Plate completely stripped (None), embeddings stripped (None).
        - AUDIT: Plate pseudonymized, embeddings replaced with dimensionality hash.
        - ADMIN: Raw plate provided; mandatory audit log recorded.
        """
        base_dict = obs.to_dict()

        if role == AccessRole.ANALYTICS:
            base_dict["plate"] = None
            base_dict["plate_confidence"] = None
            base_dict["appearance_embedding"] = None
            base_dict["privacy_status"] = "anonymized_for_analytics"
            return base_dict

        elif role == AccessRole.AUDIT:
            raw_plate = base_dict.get("plate")
            base_dict["plate"] = pseudonymize_plate(raw_plate, salt=self.salt)
            if base_dict.get("appearance_embedding"):
                base_dict["appearance_embedding"] = f"<EMBEDDING_DIM_{len(base_dict['appearance_embedding'])}>"
            base_dict["privacy_status"] = "pseudonymized_for_audit"
            self.log_query(
                requester_id=requester_id,
                role=role,
                purpose=purpose,
                target_identifier=obs.observation_id,
                fields_accessed=["pseudonymized_plate", "temporal_spatial_data"],
            )
            return base_dict

        elif role == AccessRole.ADMIN:
            self.log_query(
                requester_id=requester_id,
                role=role,
                purpose=purpose,
                target_identifier=obs.observation_id,
                fields_accessed=["raw_plate", "raw_embedding", "full_observation"],
            )
            base_dict["privacy_status"] = "raw_authorized_admin_access"
            return base_dict

        raise ValueError(f"Unsupported access role: {role}")

    def filter_identity_cluster(
        self,
        cluster: Dict[str, Any],
        role: AccessRole,
        requester_id: str = "anonymous",
        purpose: str = "mobility_analytics",
    ) -> Dict[str, Any]:
        """
        Filter a candidate or final vehicle identity cluster according to the requesting role.
        """
        out = dict(cluster)
        members = out.get("member_observations", [])

        if role == AccessRole.ANALYTICS:
            # Strip all raw plate and appearance information
            out["member_observations"] = [
                {k: v for k, v in m.items() if k not in ("plate", "appearance_embedding")}
                for m in members
            ]
            out["privacy_level"] = "ANALYTICS_AGGREGATE"

        elif role == AccessRole.AUDIT:
            # Pseudonymize all plates
            out["member_observations"] = [
                {
                    **m,
                    "plate": pseudonymize_plate(m.get("plate"), salt=self.salt),
                }
                for m in members
            ]
            out["privacy_level"] = "AUDIT_PSEUDONYMIZED"
            self.log_query(
                requester_id=requester_id,
                role=role,
                purpose=purpose,
                target_identifier=out.get("identity_id", "UNKNOWN_CLUSTER"),
                fields_accessed=["cluster_trajectory", "pseudonymized_plates"],
            )

        elif role == AccessRole.ADMIN:
            out["privacy_level"] = "ADMIN_FULL_ACCESS"
            self.log_query(
                requester_id=requester_id,
                role=role,
                purpose=purpose,
                target_identifier=out.get("identity_id", "UNKNOWN_CLUSTER"),
                fields_accessed=["cluster_trajectory", "raw_plates", "full_evidence"],
            )

        return out

    def enforce_retention(
        self,
        observations: List[Observation],
        current_time_seconds: Optional[float] = None,
    ) -> Tuple[List[Observation], int]:
        """
        Purge raw plate identifiers and appearance embeddings from observations older
        than the retention horizon.

        Returns:
            Tuple of (retained_observations, purged_record_count).
        """
        if current_time_seconds is None:
            current_time_seconds = max((o.timestamp_seconds for o in observations), default=0.0)

        purged_count = 0
        retained: List[Observation] = []

        for o in observations:
            age = current_time_seconds - o.timestamp_seconds
            if age > self.retention_horizon_seconds:
                # Retention expired: create privacy-sanitized observation copy
                sanitized = Observation(
                    observation_id=o.observation_id,
                    camera_id=o.camera_id,
                    timestamp=o.timestamp,
                    timestamp_seconds=o.timestamp_seconds,
                    vehicle_type=o.vehicle_type,
                    plate=None,  # Purged
                    plate_confidence=None,
                    appearance_embedding=None,  # Purged
                    latitude=o.latitude,
                    longitude=o.longitude,
                    timestamp_semantics=o.timestamp_semantics,
                    time_reference_id=o.time_reference_id,
                )
                retained.append(sanitized)
                purged_count += 1
            else:
                retained.append(o)

        return retained, purged_count

    @staticmethod
    def get_privacy_limitations() -> Dict[str, str]:
        """
        Document honest privacy and legal boundaries.
        Explicitly states what this module DOES and DOES NOT guarantee.
        """
        return {
            "pseudonymization_scope": (
                "HMAC-SHA256 pseudonymization replaces raw plate strings with deterministic hashes. "
                "While this prevents casual reading of plates, linkability across camera feeds still "
                "exists as long as the secret salt key is held."
            ),
            "key_management": (
                "The current implementation utilizes a configured salt parameter. In a production deployment, "
                "this key MUST be managed via a hardware security module (HSM) or cloud KMS with key rotation."
            ),
            "regulatory_status": (
                "This privacy layer implements foundational technical safeguards (role-based separation, "
                "pseudonymization, retention schedules, and query auditing). It DOES NOT certify complete "
                "statutory compliance (such as GDPR, DPDP Act, or ISO 27001) without verified organizational "
                "and infrastructure controls."
            ),
        }
