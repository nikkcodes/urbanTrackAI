"""
UrbanTrack AI — Responsible Surveillance & Privacy Architecture.

Provides:
- Salted HMAC-SHA256 pseudonymization of license plate identifiers.
- Role-based data views (ANALYTICS vs AUDIT vs ADMIN).
- Configurable data retention horizon filtering.
- Structured audit event logging for all identity queries.
- Explicit documentation of security and privacy boundaries.
"""

from .privacy_guard import (
    AccessRole,
    AuditEvent,
    PrivacyGuard,
    pseudonymize_plate,
)

__all__ = [
    "AccessRole",
    "AuditEvent",
    "PrivacyGuard",
    "pseudonymize_plate",
]
