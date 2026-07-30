"""``LifecycleLeasePort`` — the user-prefix-grained lifecycle-sweep lease.

Spec: ``memory-lifecycle-manager-spec.md`` §4b (lines 173-192) + §17 item 8b (line 711).
Contract addition: ``PROPOSED-CANONICAL-ADDITIONS-mlm.md`` P8 (lines 171-199). ADR 0032 (F1 fix).

**Sibling of, and NOT the same Protocol as, ``mu_engine.pipelines.distill.WriterLeasePort``**
(CANONICAL §7.5, EXISTS, ``distill.py:133-137``). ``WriterLeasePort.acquire(ns: Namespace)`` is
keyed on the session-inclusive ``Namespace`` — its ``to_prefix()`` carries a ``session`` segment
(CANONICAL §1 rule 5), so two sessions of one user take two *different* ``to_prefix()`` values and
two distinct lease keys, both of which acquire successfully. That is exactly the F1 bug this port
exists to fix: a per-user sweep (session-spanning by definition, spec §1) cannot ride a
session-inclusive lease without silently reproducing the double-write hole.

``LifecycleLeasePort`` is therefore its **own** Protocol, keyed on ``UserPrefix`` (the
session-spanning grain minted in S0-01 — a truncation of ``Namespace.to_prefix()`` before the
session segment: ``mu/{org}/{workspace}/{visibility}/{user_slot}/``), NOT ``Namespace``. Same
acquire-or-defer, plane-qualified semantics as §7.5's distill lease; distinct key type, distinct
protocol object, distinct lease name. This is the identical distinguish-don't-merge resolution
already applied to ``LifecycleWorkflowRunnerPort`` vs the generic ``WorkflowRunnerPort``
(spec §17 item 2; GAPSWEEP BLOCKER 1).

This is an ADDITIVE second lease. §7.5's distill lease is untouched by this port:
``DistillPipeline``'s own ``WriterLeasePort`` field still acquires per-``Namespace`` exactly as
today (``distill.py:216``-area); ``to_prefix()`` is not reshaped (KEEP-AND-SCOPE).

Lease key shape (plane-qualified, following §7.5's naming convention so the same key cannot exist
in two Redis instances, but at a coarser grain and under a distinct lease name):
``lifecycle-sweep-lease:local:{device_id}:{user_prefix}`` (device) ·
``lifecycle-sweep-lease:hub:{user_prefix}`` (server).

Acquisition order is pinned system-wide: **lifecycle lease (outer) -> distill lease (inner)**.
There is no reverse-order acquirer (the capture-path reconcile takes only the distill lease), so no
deadlock. Cross-fact convergence remains CANONICAL §7.5 deterministic re-derivation over the delta
set, never a distributed lock — this lease guards exactly one lifecycle sweep per user across all
their sessions (and, in HYBRID, serializes a manual lifecycle verb against that sweep).

Two adapters implement this port (construction is out of scope for this contract-only module):
``SqliteWalLeaseAdapter`` (LOCAL — a ``BEGIN IMMEDIATE`` row lock on the same local job-log
database ``SqliteWalRunner`` already opens) and a Redis-``SETNX`` hub adapter (unchanged from the
original §7.5-style design).
"""

from __future__ import annotations

import contextlib
from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.lifecycle import UserPrefix

__all__ = ["LifecycleLeasePort"]


@runtime_checkable
class LifecycleLeasePort(Protocol):
    """Acquire-or-defer, plane-qualified — same semantics as ``WriterLeasePort`` (CANONICAL
    §7.5, ``mu_engine.pipelines.distill.WriterLeasePort``), but keyed on ``UserPrefix`` (the
    session-spanning grain, spec §1/§17a) instead of ``Namespace`` (session-inclusive). Deliberately
    NOT the same Protocol as ``WriterLeasePort`` — see module docstring for the F1 rationale.

    Two adapters: ``SqliteWalLeaseAdapter`` (LOCAL, below) · a Redis-``SETNX`` hub adapter
    (hub, EXISTS pattern per CANONICAL §7.5).
    """

    def acquire(self, prefix: UserPrefix) -> contextlib.AbstractAsyncContextManager[None]: ...
