"""DeviceRegistryPort — the SINGLE device registry port (sync-devices-gateway-spec.md §A.3).

The gateway does NOT define its own. ``touch()`` is unified and carries THREE things —
content-free ``ClientMetadata`` (heartbeat) AND ``last_synced_seq`` AND ``last_lamport`` — so a
two-signature split-brain is structurally impossible. All three hot-path mutables are HUB-advanced
only (a device cannot lie its cursor/VC forward).

⚠ **Every read and every write is ORG-SCOPED BY SIGNATURE — phase-3 spec D-13 (build step 2).**
The module docstring has always asserted *"Reads are scoped ``(workspace, principal[, id])``, never
a global ``device_id`` scan"*, and ``CANONICAL-CONTRACTS.md:247`` forbids *"any cross-tenant
``device_id`` scan"* — but until D-13 landed, ``resolve``/``touch``/``revoke``/``set_state`` took a
**bare ``device_id``** and ``devices_of`` a bare ``principal_id``. **The Protocol as written could
not express the invariant its own docstring asserted**, and project rule 4 (every store access is
namespace-scoped) forbids that shape. D's answer — assert ``TenancyGuard.assert_scope`` in
``DeviceService`` *above* the port — leaves the port itself un-scoped and re-invitable by the next
caller, and on a multi-tenant plane an un-org-scoped device query IS a cross-tenant read.

So **every method takes ``org_id: str`` as its first argument**, and ``DeviceService``
**additionally** runs ``TenancyGuard.assert_scope``.

⚠ **Every method that WRITES additionally takes ``principal_id`` — the OWNER — as its second
argument, and it is a predicate, not a stamp.** ``org_id`` alone scopes a device query to the
TENANT; a shared plane's tenant holds many principals, and ``DeviceRow``'s own comment calls
``principal_id`` *"the ONE owning principal"*. A security review reproduced the consequence against
real Postgres while ``set_state`` was org-only: any principal in an org could revoke any other
principal's device given its id — a same-org denial of service — and the ``DeviceRecord`` the call
returns handed back that other principal's ``public_key``, ``label`` and sync cursors. The 409/404
split made the same call a cross-principal existence oracle. Device ids are not secrets: they ride
``X-MU-Device-Id``, ``DeviceEnrolled``/``DeviceRevoked`` on the shared bus, and the enrolment log
line.

``resolve`` deliberately keeps its ``(org_id, device_id)`` face: it is a READ whose caller may not
yet have decided whose device it is, and it returns the record for the caller — not for a mutation
— which is the difference between "look this device up in my tenant" and "change this device". A
caller that must not learn about another principal's device asks ``devices_of``.

⚠ **``enroll`` is the ONE exception and it is a precedence rule, not an oversight — D-33.**
``enroll`` keeps its single-argument face because ``DeviceEnrollRequest`` is the port's own DTO and
carries ``org_id`` on itself, constructed SERVER-SIDE by ``DeviceService`` from the verified
``AuthContext``. One carrier per call, never both — two carriers for one hash input is how the id
and the scope silently disagree.

⚠ **Isolation here does NOT rest on ``to_prefix()``, and saying so would be false comfort.**
``CANONICAL-CONTRACTS.md:75`` names two instruments — ``to_prefix()`` prefixing AND
``TenancyGuard.assert_scope`` — and the device registry is deliberately **not**
``to_prefix()``-partitioned (a device spans sessions and visibilities and has no η; ``DeviceRow``'s
PK is ``device_id`` alone with tenancy as plain columns). Isolation therefore rests on (i) the
``org_id``-first argument below and (ii) the guard, which is what makes an adapter-level "no SQL
selects on ``device_id`` without ``org_id``" assertion load-bearing rather than belt-and-braces.

⚠ **``set_key_epoch`` is deliberately NOT declared here.** ``D:192``/``S:121`` pin it and the
phase-3 spec §3.2 schedules it, but it is the E2E tier's seam (RESERVED §7.18) and this Protocol is
``@runtime_checkable`` — method PRESENCE is the isinstance contract, so declaring it forces every
adapter to define a body for a capability with zero reachable call sites. It lands with the
acceptance gate that guards it (§10.4's zero-call-site AST test), not before: *a stub is worse than
an absence*.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from mu_contracts.domain.model.device import (
    DeviceEnrollRequest,
    DeviceRecord,
    DeviceState,
    RevokeReason,
)

__all__ = ["ClientMetadata", "DeviceRegistryPort"]


class ClientMetadata(BaseModel):
    """Content-free heartbeat metadata carried on ``touch()`` (gateway §7.1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_build: str = ""
    platform: str = ""


@runtime_checkable
class DeviceRegistryPort(Protocol):
    async def enroll(self, req: DeviceEnrollRequest) -> DeviceRecord:  # idempotent put-if-absent
        """``req.org_id`` carries the scope — see the module docstring's D-33 note."""
        ...

    async def resolve(self, org_id: str, device_id: str) -> DeviceRecord | None: ...

    async def devices_of(self, org_id: str, principal_id: str) -> list[DeviceRecord]:
        """The principal's device FLEET in ``org_id`` — every row whose ``state`` is not
        ``REVOKED``.

        ⚠ **The filter is ``state <> 'revoked'``, NOT ``state == 'active'``** (§4b.6). A THIN
        device may never leave ``PENDING`` — activation is *"first successful sync"* and a THIN
        device never performs the appender-A ingest that would trigger it (**O-30**) — so gating
        on ``ACTIVE`` would lock every THIN fleet off the plane. This matches the fan-out rule
        (§5.3 step 3 excludes only ``REVOKED``) deliberately: if O-30 later gives ``PENDING`` a
        meaning, this filter is where it lands.

        ⚠ **It returns the SET, never a membership boolean** (§4b.6/3). This is the same call the
        ``device_fan_out`` abuse signal counts (D-26), so ONE query serves both the gateway's
        stage-6 bind gate and the count that gate's traffic feeds. A cached boolean, or a second
        device-query path, would be free to diverge from the number it must agree with.
        """
        ...

    async def touch(
        self,
        org_id: str,
        principal_id: str,
        device_id: str,
        *,
        meta: ClientMetadata,
        last_seen_at: datetime,
        last_synced_seq: int,
        last_lamport: int,
    ) -> None: ...

    async def revoke(
        self, org_id: str, principal_id: str, device_id: str, *, reason: RevokeReason
    ) -> None:
        """``principal_id`` is the device's OWNER and belongs in the WHERE clause — see the module
        docstring's owner-predicate note.

        ``reason`` is the CLOSED :class:`~mu_contracts.domain.model.device.RevokeReason` (D-34),
        never free text: ``DeviceRevoked.reason`` reaches the SHARED bus and every trace of that
        event, and the bus's content-free guard is a field-NAME check that cannot see this class of
        leak."""
        ...

    async def set_state(
        self,
        org_id: str,
        principal_id: str,
        device_id: str,
        state: DeviceState,
        *,
        at: datetime,
        by: str | None = None,
    ) -> DeviceRecord:  # compare-and-set (PENDING→ACTIVE)
        """⚠ ``principal_id`` is the device's OWNER and must sit **inside** the compare-and-set
        predicate, alongside ``org_id`` and the expected state — never as a check layered above it,
        which a concurrent write can slip past. ``by`` is the ACTOR that caused the transition and
        is a STAMP written onto the row; the two are different roles and are equal only in the
        ordinary case where a principal revokes its own device."""
        ...
