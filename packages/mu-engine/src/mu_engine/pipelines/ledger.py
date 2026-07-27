"""``StageLedger`` — the per-stage idempotency substrate (PIPELINES-design.md §2.4).

Generalises ``workflows/sync.py``'s ``SyncLedger``/``RedisSyncLedger`` (the prototype
``/home/user/hackathon/memory_universe/...``): a stage records its completion marker AND its
events as ONE durable write (B4), so a replay/retry that hits the ledger re-publishes the recorded
events rather than returning empty — it is structurally impossible for a ledger-complete key to
carry no events.

Two adapters ship here: :class:`InMemoryStageLedger` (tests / minimal local mode) and
:class:`RedisStageLedger` (the durable backend, one ``HSET`` per key against the real cache). The
event codec round-trips a ``DomainEvent`` through the frozen ``mu_contracts.domain.events`` catalog
by class name — content-free payloads only (the catalog forbids body fields at class-definition).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from redis.asyncio import Redis

from mu_contracts.domain import events as _events
from mu_contracts.domain.events import DomainEvent
from mu_engine.platform.decorators import retry_io

__all__ = [
    "InMemoryStageLedger",
    "RedisStageLedger",
    "StageLedger",
    "StageLedgerRecord",
]

_LEDGER_IO_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class StageLedgerRecord:
    """(B4) the durably-recorded events for a completed stage key (PIPELINES §2.4)."""

    events: Sequence[DomainEvent]


class StageLedger(Protocol):
    """(B4) ``mark_completed`` persists the completion marker AND ``events`` as ONE durable write —
    never a bare flag followed by a separate publish. ``completed_with_events`` returns those events
    on a replay so they can be re-published (PIPELINES §2.4)."""

    async def completed_with_events(self, key: str) -> StageLedgerRecord | None: ...

    async def mark_completed(self, key: str, *, events: Sequence[DomainEvent] = ()) -> None: ...


# ----------------------------------------------------------------- DomainEvent JSON codec
# The frozen event catalog is content-free (CANONICAL §3.1), so a class-name tag + the
# pydantic json dump is a lossless, safe wire form. Unknown tag => fail loud (never silent).
_EVENT_TYPES: dict[str, type[DomainEvent]] = {
    name: obj
    for name, obj in vars(_events).items()
    if isinstance(obj, type) and issubclass(obj, DomainEvent) and obj is not DomainEvent
}


def _encode_events(events: Sequence[DomainEvent]) -> str:
    return json.dumps(
        [{"__type__": type(e).__name__, "data": e.model_dump(mode="json")} for e in events]
    )


def _decode_events(blob: str) -> list[DomainEvent]:
    out: list[DomainEvent] = []
    for row in json.loads(blob):
        tag = row["__type__"]
        cls = _EVENT_TYPES.get(tag)
        if cls is None:
            raise KeyError(f"unknown DomainEvent type in ledger: {tag!r}")
        out.append(cls.model_validate(row["data"]))
    return out


class InMemoryStageLedger:
    """A process-local ledger — tests + minimal single-process local mode (PIPELINES §2.4)."""

    def __init__(self) -> None:
        self._records: dict[str, StageLedgerRecord] = {}

    async def completed_with_events(self, key: str) -> StageLedgerRecord | None:
        return self._records.get(key)

    async def mark_completed(self, key: str, *, events: Sequence[DomainEvent] = ()) -> None:
        # ONE write: marker + events together (B4). Idempotent — a re-mark keeps the first record.
        self._records.setdefault(key, StageLedgerRecord(events=tuple(events)))


class RedisStageLedger:
    """Durable ledger over Redis/Valkey — same keyspace discipline as ``RedisSyncLedger``.

    Marker + events are ONE key (``SET`` with the encoded events as the value, B4). Every call is
    wrapped by :func:`retry_io` (transient-only retry + per-attempt timeout) so none is unbounded.
    """

    def __init__(self, client: Redis, *, key_prefix: str = "mu:pipeline-ledger") -> None:
        self._redis = client
        self._prefix = key_prefix

    def _key(self, key: str) -> str:
        return f"{self._prefix}:{key}"

    @retry_io(timeout_s=_LEDGER_IO_TIMEOUT_S)
    async def completed_with_events(self, key: str) -> StageLedgerRecord | None:
        raw = await self._redis.get(self._key(key))
        if raw is None:
            return None
        blob = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return StageLedgerRecord(events=_decode_events(blob))

    @retry_io(timeout_s=_LEDGER_IO_TIMEOUT_S)
    async def mark_completed(self, key: str, *, events: Sequence[DomainEvent] = ()) -> None:
        # NX: the first completion wins; a retry never overwrites the recorded events (B4).
        await self._redis.set(self._key(key), _encode_events(events), nx=True)
