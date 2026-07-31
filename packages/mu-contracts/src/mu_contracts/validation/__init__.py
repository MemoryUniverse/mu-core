"""Cross-cutting validators shared by every plane/surface (pydantic-only, mu-contracts §1.3).

Home of the plane-gating validator (build-plan Stage B ruling 1) — see
:mod:`mu_contracts.validation.plane_gate`.
"""

from __future__ import annotations

from mu_contracts.validation.plane_gate import (
    PRIVATE_PLANE_FIELDS,
    SHARED_PLANE_FIELDS,
    validate_plane_fields,
)

__all__ = [
    "PRIVATE_PLANE_FIELDS",
    "SHARED_PLANE_FIELDS",
    "validate_plane_fields",
]
