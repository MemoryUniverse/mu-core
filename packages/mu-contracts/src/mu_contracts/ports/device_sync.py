"""PrivateSyncLogPort — ONE log, ONE seq, TWO appenders (sync-devices-gateway-spec.md §B.3).

``append`` is the SOLE seq authority (hub-assigned monotonic). ``read`` is the durable backfill
from a cursor (contiguity apply-rule, CANONICAL §7.6 X16). ``floor_seq`` drives the reseed
decision when a device has fallen behind the retention window.

⚠ **Every method is keyed ``(org_id, workspace_id, principal_id)`` — SPEC DECISION D-28b**
(mu-server-phase3-devices-sync-spec.md §4.4). Three things are true at once and the signature has
to hold all of them:

* The **stream key is ``(org_id, principal_id)``** — ADR 0026's rename (`workspace`->`org`,
  `namespace`->`workspace`) makes CANONICAL §7.14a's ``UNIQUE(workspace_id, principal_id,
  content_hash, occurrence_discriminator)`` read ``org_id`` today, and the shipped table agrees:
  ``PrimaryKeyConstraint("org_id", "principal_id", "seq")``. ``seq`` is monotonic and gap-free
  *within that pair* and within nothing else.
* ``PrivateSyncLogRow.workspace_id`` is ``String(128), nullable=False`` with **no default and no
  server default**, and ``PrivateDelta`` carries **no ``workspace_id``**. Without a third argument
  every INSERT raises ``NotNullViolation`` — D-28b exists because that is a trap the adapter
  author would otherwise discover at runtime.
* The column stays because it is the only record of WHICH workspace a delta was written in, and
  the callers both know it (appender A from the verified ``AuthContext``, appender B from
  ``Namespace.workspace``), so passing it costs nothing.

``workspace_id`` is therefore a **payload column, never part of the stream key**: it is written on
append and returned nowhere, and no read predicate filters on it. Adding it to a ``WHERE`` clause
would silently fork one user's ``seq`` line in two.

⚠ **CORRECTION — an earlier version of this docstring claimed the column stops a delta written in
a workspace the principal later leaves from replicating to their devices. It does not, and nothing
in this repo does.** Every read predicate omits ``workspace_id`` BY DESIGN (that is the paragraph
above), so the private stream IS behaviourally workspace-blind; the column is written and never
read. Stated plainly here because a reader who believed the old sentence would think a
leave-workspace boundary was being enforced on this path, and would build on a guarantee that has
no code behind it. Enforcing one means a *filtered* read, which forks the ``seq`` line and breaks
§7.6 contiguity — so it is a design question, not an oversight to patch.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.device_sync import PrivateDelta

__all__ = ["PrivateSyncLogPort"]


@runtime_checkable
class PrivateSyncLogPort(Protocol):
    async def append(
        self, org_id: str, workspace_id: str, principal_id: str, delta: PrivateDelta
    ) -> int:  # -> seq; sole seq authority
        """Assign this delta the next ``seq`` on the ``(org_id, principal_id)`` stream.

        **Contiguity is a CONTRACT** (CANONICAL §7.6): a replica applies a delta only when
        ``seq == last_synced_seq + 1``, and a monotonic high-water-mark is explicitly REJECTED. So
        the sequence must be gap-free, which is why D-21 pins in-transaction ``MAX(seq)+1`` and
        **forbids a Postgres ``SEQUENCE``** — a SEQUENCE is global and burns a number on every
        rollback, and every burnt number looks like a lost delta to every replica forever.

        **Idempotent by the 4-tuple, not by ``content_hash``** (CANONICAL §7.14a): re-appending an
        occurrence that already exists — same ``(content_hash, origin_device_id, lamport)`` —
        returns that occurrence's ORIGINAL ``seq`` and assigns no new one, while a genuine SECOND
        reinforcement of the same fact (a later ``lamport``) is a distinct occurrence and gets its
        own ``seq``. Bare ``content_hash`` was explicitly WRONG here: it collapses the second
        reinforcement so it never propagates.
        """
        ...

    async def append_many(
        self,
        org_id: str,
        workspace_id: str,
        principal_id: str,
        deltas: Sequence[PrivateDelta],
    ) -> list[int]:
        """The ordered batch — ONE transaction, ``seq`` in submitted order (D-40, §6.8).

        Declared alongside :meth:`append` rather than left to a caller-side loop because the
        property D-40 pins is **all-or-nothing**: *"the transaction rolls back and the device
        re-drives, which is safe because the 4-tuple collapses the replay"*. A loop over
        independently-transacted :meth:`append` calls cannot deliver that — a mid-batch failure
        would leave a prefix committed — so the batch boundary has to be inside the adapter that
        owns the transaction. ``append(delta)`` is exactly ``append_many([delta])[0]``.

        Returns one ``seq`` per input delta, positionally, INCLUDING for items that collapsed onto
        an existing occurrence (which return that occurrence's original ``seq``). That is what lets
        a device acknowledge a partly-deduped batch item by item.
        """
        ...

    async def read(
        self,
        org_id: str,
        workspace_id: str,
        principal_id: str,
        *,
        after_seq: int,
        limit: int,
    ) -> list[PrivateDelta]:
        """The durable backfill: deltas with ``seq > after_seq``, in ``seq`` order, at most
        ``limit`` of them. This is the run a replica applies under §7.6's contiguity rule."""
        ...

    async def live_payload_refs(
        self, org_id: str, workspace_id: str, principal_id: str, *, limit: int
    ) -> tuple[str, ...]:
        """The ``payload_ref``s a re-seed snapshot carries — ``PrivateSnapshot.items``.

        "Live" means: the most recent delta for each ``memory_id`` on the stream does not tombstone
        it, and it has a ``payload_ref``. This is a read for the reseed branch of the backfill and
        has exactly one caller.

        ⚠ **O-31 — the referent has no transport, and this method cannot fix that.** A
        ``payload_ref`` points into the principal's private-hosted partition; for an appender-A
        delta the body is on the DEVICE and was never uploaded, so many of a real fleet's deltas
        carry ``payload_ref=None`` and are simply absent from this tuple. That is O-31 showing
        through, not a bug here, and it is why the reseed path is dead-but-correct code rather than
        a working re-seed: it returns the refs that exist, honestly, and O-31 owns the rest.
        """
        ...

    async def head_seq(self, org_id: str, workspace_id: str, principal_id: str) -> int:
        """The highest ``seq`` on the stream, or ``0`` for an empty one."""
        ...

    async def floor_seq(
        self, org_id: str, workspace_id: str, principal_id: str
    ) -> int:  # retention floor → reseed decision
        """**The highest ``seq`` that is NO LONGER retained** — i.e. the last pruned one, ``0`` for
        a stream nothing has pruned.

        ⚠ Read that definition twice, because the intuitive one ("the lowest ``seq`` still here")
        is off by one and the difference is a **fleet-wide spurious re-seed**. The reseed predicate
        is ``after_seq < floor_seq``, and a brand-new device asks with ``after_seq=0``. Under the
        intuitive definition an unpruned log answers ``floor_seq() == 1``, so ``0 < 1`` is true and
        every device on every first contact is told to re-seed. Under this definition an unpruned
        log answers ``0``, ``0 < 0`` is false, and the device backfills normally — which is also
        what mu-server-phase3-devices-sync-spec.md §8 asserts in as many words: *"with no pruner
        ``floor_seq()`` is always 0"*.

        ⚠ **Nothing prunes this table in any repo (O-32)**, so today this always answers ``0`` and
        the reseed branch behind it is dead-but-correct code. It is implemented rather than stubbed
        because the day a pruner lands, the predicate above must already be the right one.
        """
        ...
