"""The plane-gate validator (sdk-engine-server-design.md §2.5 "Unified verb surface"; build-plan
``2026-07-31-sdk-build-plan.md`` §0 ruling 1; build-queue §13 item 3).

**Why this lives in mu-contracts, not mu-local or mu-sdk-python.** The canonical superset
signature's plane-gating rule must be enforced on BOTH ``LocalMemory`` (mu-local, embedded) and
``MemoryClient`` (mu-sdk-python, wire) — one rule, no duplication. ``MemoryClient`` is bound by
``PACKAGING-v2.md`` §5's "sdk-has-no-engine" boundary to importing ONLY ``mu-contracts`` (it may
never import ``mu-engine``/``mu-local``), so the shared validator has exactly one legal home:
``mu-contracts`` (pydantic-only, ``.importlinter``'s ``contracts-imports-nothing-in-project``).

**What it does.** Design §2.5's canonical ``add``/``recall``/etc. superset signature splits its
plane-gated kwargs into two groups:

- PRIVATE-plane fields (``user``, ``session``, ``agent``) — active iff a private plane is
  configured (``mode="embedded"``/``"local_server"``, design §4).
- SHARED-plane fields (``visibility``, ``subject``, ``predicate``, ``object``) — active iff a
  shared plane is configured (``mode="remote"``, or a populated ``shared=`` sub-object, design §4).

"Supplying a field that doesn't apply to the currently-configured plane is a REJECTION, not a
silent no-op" (design §2.5) — :func:`validate_plane_fields` is that rejection, raising the named
:class:`~mu_contracts.domain.errors.PlaneFieldRejectedError` (never a silent drop, DEV-STANDARDS
rule 8) on the first offending field.

**Deliberately decoupled from any SDK-specific ``mode`` Literal.** ``SdkConfig.mode``
(``"embedded" | "local_server" | "remote"``) is a Stage-D / mu-sdk-python concept this
pydantic-only package cannot and should not know about. Callers (``LocalMemory``, ``MemoryClient``)
resolve their own ``mode``/``shared=`` configuration down to two plain booleans before calling
this — e.g. ``private_configured = mode in ("embedded", "local_server")``, ``shared_configured =
(mode == "remote") or (shared is not None)`` (design §4's worked dual-plane example).
``LocalMemory`` itself is private-plane-only BY CONSTRUCTION today (``mu-local/local_memory.py:
12-13``: "NO SHARED partition ... those are mu-server concepts") — it always calls this with
``shared_configured=False``, so every shared-plane field it is ever handed is rejected, which is
exactly the "LocalMemory gains acceptance of the shared-plane fields (rejected when no shared plane
is configured)" behavior design §2.5 asks for.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mu_contracts.domain.errors import PlaneFieldRejectedError

__all__ = [
    "PRIVATE_PLANE_FIELDS",
    "SHARED_PLANE_FIELDS",
    "validate_plane_fields",
]

#: design §2.5 canonical ``add`` superset signature — the private-plane-gated arg names.
PRIVATE_PLANE_FIELDS: frozenset[str] = frozenset({"user", "session", "agent"})

#: design §2.5 canonical ``add`` superset signature — the shared-plane-gated arg names.
SHARED_PLANE_FIELDS: frozenset[str] = frozenset({"visibility", "subject", "predicate", "object"})


def validate_plane_fields(
    fields: Mapping[str, Any],
    *,
    private_configured: bool,
    shared_configured: bool,
) -> None:
    """Reject any supplied field whose plane is not configured (design §2.5).

    ``fields`` is the caller's kwargs for one verb invocation. A value of ``None`` is read as
    "not supplied" — every plane-gated field defaults to ``None`` on both ``LocalMemory``'s and
    ``MemoryClient``'s canonical signatures (design §2.5's ``add`` example), so this matches the
    discipline those defaults already establish; there is no other way to distinguish "omitted"
    from "explicitly None" across a plain kwargs mapping, and design §2.5's own worked example
    ("passing ``visibility=`` while ``mode="embedded"`` and ``shared=None`` raises") only exercises
    the supplied-a-real-value case. A field name absent from BOTH :data:`PRIVATE_PLANE_FIELDS` and
    :data:`SHARED_PLANE_FIELDS` (e.g. ``tier``, ``metadata`` — "already-common across both surfaces
    today, unchanged" per design §2.5) is never plane-gated and always passes through untouched.

    Raises on the FIRST offending field, in ``fields`` iteration order — design §2.5 requires "a
    rejection, not a silent drop", not an exhaustive multi-field report; a caller that wants to
    surface every violation at once may catch-and-retry, but Python dict iteration order is
    insertion-stable so a caller building ``fields`` from a fixed kwarg list gets a deterministic
    first failure.

    :raises PlaneFieldRejectedError: a shared-plane field was supplied with ``shared_configured=
        False``, or a private-plane field was supplied with ``private_configured=False``.
    """
    for name, value in fields.items():
        if value is None:
            continue
        if name in SHARED_PLANE_FIELDS and not shared_configured:
            raise PlaneFieldRejectedError(
                name, plane="shared", reason="no shared plane is configured"
            )
        if name in PRIVATE_PLANE_FIELDS and not private_configured:
            raise PlaneFieldRejectedError(
                name, plane="private", reason="only a shared plane is configured"
            )
