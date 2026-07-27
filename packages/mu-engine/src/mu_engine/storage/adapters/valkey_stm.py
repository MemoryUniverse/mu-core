"""Valkey KV/STM adapter — registered EXPLICITLY under ``(kv, "valkey")`` (owner stage
2026-07-27, ``storage-pluggable-spec.md §3.2``: "``valkey`` | prod | Redis-compatible
OSS-licence deploys | wire-identical, no gap").

``redis-py`` is the standard client for Valkey too (RESP-wire compatible), so ALL the I/O
logic — pipeline, ``SETEX``/``ZADD``/``ZREVRANGE``, tenancy-prefixed keys — is inherited
verbatim from :class:`RedisStmAdapter` (DEV-STANDARDS rule 6, DRY: a wire-identical backend
gets a thin OOP subclass, not a copy-pasted adapter). The distinct class + registry key exist
so (a) the registry can name ``valkey`` as its own first-class backend rather than an alias of
``redis``, and (b) observability/tracing spans can attribute calls to the actual product in
use. ``mu-dev-cache`` was recreated from ``redis:7-alpine`` to ``valkey/valkey:8-alpine``
(``docker-compose.dev.yml``) precisely so this key names a REAL Valkey server end to end.
"""

from __future__ import annotations

from mu_engine.storage.adapters.redis_stm import RedisStmAdapter

__all__ = ["ValkeyStmAdapter"]


class ValkeyStmAdapter(RedisStmAdapter):
    """Implements ``StmTierRepository`` over a real Valkey connection (redis-py client).

    No overridden behavior — the wire protocol, key catalog, and pipeline semantics are
    identical to Redis (spec §3.2 "wire-identical, no gap"); this subclass exists purely as
    the distinct (kv, "valkey") registry identity.
    """
