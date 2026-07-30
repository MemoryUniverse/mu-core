"""``LifecycleLeasePort`` — spec §4b/§17 item 8b, ADR 0032 (F1 fix).

Acceptance obligations (S0-02):
1. ``acquire(prefix: UserPrefix) -> AbstractAsyncContextManager[None]`` — keyed on ``UserPrefix``
   (S0-01), never ``Namespace``.
2. A doc-comment cross-reference to ``mu_engine.pipelines.distill.WriterLeasePort`` is present —
   this port is deliberately its OWN ``runtime_checkable`` Protocol, not that one, and not a
   reshape of it. mu-contracts imports nothing in-project (``.importlinter``:
   ``contracts-imports-nothing-in-project``), so this file does NOT import ``mu_engine`` — the
   distinctness is a static (mypy) + documentation guarantee, not a runtime isinstance check
   (structural Protocols with an identically-shaped single-arg ``acquire`` method would satisfy
   ``isinstance`` either way regardless of the annotated key type; no isinstance-collision test is
   meaningful or required here).
3. ``mypy --strict`` clean.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import runtime_checkable

import pytest

from mu_contracts.domain.model.lifecycle import UserPrefix
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.ports.lifecycle_lease import LifecycleLeasePort

pytestmark = pytest.mark.unit


def _user_prefix() -> UserPrefix:
    ns = Namespace(
        org="acme", workspace="proj", user="alice", session="s1", visibility=Visibility.PRIVATE
    )
    return UserPrefix(ns)


class _FakeLifecycleLease:
    """A minimal structural adapter — proves the Protocol shape is satisfiable and callable,
    without depending on either concrete adapter (``SqliteWalLeaseAdapter`` / Redis-``SETNX``,
    both deferred to a later stage)."""

    def __init__(self) -> None:
        self.acquired_with: list[UserPrefix] = []

    @contextlib.asynccontextmanager
    async def acquire(self, prefix: UserPrefix) -> AsyncIterator[None]:
        self.acquired_with.append(prefix)
        yield


class _NotALease:
    """Deliberately missing ``acquire`` — the negative isinstance case."""


def test_is_runtime_checkable_protocol() -> None:
    assert runtime_checkable(LifecycleLeasePort) is LifecycleLeasePort  # already decorated
    assert getattr(LifecycleLeasePort, "_is_runtime_protocol", False) is True


def test_fake_adapter_satisfies_protocol_structurally() -> None:
    fake = _FakeLifecycleLease()
    assert isinstance(fake, LifecycleLeasePort)


def test_object_missing_acquire_does_not_satisfy_protocol() -> None:
    assert not isinstance(_NotALease(), LifecycleLeasePort)


@pytest.mark.asyncio
async def test_acquire_is_keyed_on_user_prefix_and_is_an_async_context_manager() -> None:
    fake = _FakeLifecycleLease()
    prefix = _user_prefix()
    assert prefix == "mu/acme/proj/private/alice/"

    async with fake.acquire(prefix):
        pass

    assert fake.acquired_with == [prefix]
    # the acquired grain is the session-spanning prefix, NOT the session-inclusive to_prefix()
    assert not prefix.endswith("s1")


def test_module_doc_cross_references_writer_lease_port_sibling() -> None:
    import mu_contracts.ports.lifecycle_lease as mod

    assert mod.__doc__ is not None
    assert "WriterLeasePort" in mod.__doc__
    assert "distill" in mod.__doc__
    assert LifecycleLeasePort.__doc__ is not None
    assert "WriterLeasePort" in LifecycleLeasePort.__doc__


def test_lifecycle_lease_port_is_a_distinct_object_not_an_alias() -> None:
    # A sibling port, never a rebind of an existing Protocol under a new name (spec §17 item 2's
    # distinguish-don't-merge resolution, applied identically here per §17 item 8b). Compared via
    # qualname (not `is not`, which mypy correctly flags as a non-overlapping identity check
    # between two statically-distinct classes).
    import mu_contracts.ports.workflow as workflow_mod

    assert LifecycleLeasePort.__qualname__ != workflow_mod.WorkflowRunnerPort.__qualname__
    assert LifecycleLeasePort.__name__ == "LifecycleLeasePort"
