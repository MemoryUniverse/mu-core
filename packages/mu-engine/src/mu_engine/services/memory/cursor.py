"""The composite ``enumerate`` cursor — one opaque token over three incompatible tier cursors.

**Unspecified by the design set, so the choice is recorded here rather than improvised.**
``MemoryRepository.enumerate`` returns a single ``str | None``, but the three tiers page on three
different mechanisms: a recency-ZSET RANK (Redis/in-process KV), an opaque Qdrant scroll OFFSET,
and a keyset ``m.id`` (FalkorDB). ``memory-health-pinning-spec.md`` §3.1 pins the signature and
says nothing about the token's shape.

Three properties are load-bearing:

1. **The cursor NEVER carries tenancy.** ``to_prefix()`` is re-derived from the authorized ``ns``
   on every leg of every call, so an attacker-supplied or replayed token can at most mis-position
   a walk INSIDE the caller's own partition. CANONICAL §1 rule 5 — *"a query-filter bug cannot
   leak across tenants. Not a filter."* — is satisfied structurally, by the cursor being unable to
   influence scope at all, rather than by validating it well.

2. **It is BOUND to the partition that minted it, and a mismatch is refused loud.** Binding is
   belt-and-braces on top of (1): a token minted for tenant A replayed against tenant B cannot
   read B's data (scope is re-derived), but it CAN silently resume B's walk from a meaningless
   position and hand back a page that looks like an answer. That is a wrong answer, so it raises
   ``NamespaceIsolationError`` instead — the non-enumerating denial, carrying neither prefix.

3. **It survives the DEGRADED RETRY.** ``MemoryHealthService._walk`` re-issues the SAME cursor
   with ``tiers={STM, MTM}`` after an LTM failure, so a leg's position must remain decodable when
   that leg is not walked — which is why positions live in a per-tier mapping keyed by tier value,
   not in a fixed-arity tuple. ``TieredMemoryRepository.enumerate`` carries such a position
   forward verbatim rather than rebuilding the token from the legs it walked; dropping it would
   end the walk ``next_cursor=None`` (i.e. EXHAUSTED) over a tier that was never read.

4. **It counts CONSECUTIVE STALLED pages, because surviving the degraded retry creates a way to
   not terminate.** Carrying an un-walked tier's position forward is what stops a blip from
   deleting that tier from the walk — but if the tier is down PERMANENTLY, every later page is
   narrowed away from it again, walks nothing, and mints the same token: an infinite sequence of
   empty pages. ``next_cursor=None`` is not available as an escape, because ``ports/memory.py``
   lines 78-80 define it as EXHAUSTED and the tier is not. So one stalled page is tolerated —
   that is the round trip in which the caller stops narrowing and the recovered tier resumes —
   and the second consecutive one raises. Bounded, and neither a loop nor a false completion.
"""

from __future__ import annotations

import base64
import binascii
import json
from hashlib import sha256
from typing import Any

from mu_contracts.domain.errors import NamespaceIsolationError
from mu_contracts.domain.model.memory import Namespace, Tier

__all__ = ["MAX_STALLED_PAGES", "TierCursor", "decode_cursor", "encode_cursor"]

#: Token format version. Present so a future change to the composition can be REFUSED rather than
#: mis-parsed: an old token arriving at new code is a resumable walk with a different meaning, and
#: reading it under the wrong rules would silently skip or repeat a caller's memories.
_FORMAT = 1

#: Length of the partition-binding digest. A truncated SHA-256 of ``to_prefix()`` — never the
#: prefix itself, which would put a tenant identifier into a token that travels through IPC, MCP
#: and client logs (content-free discipline, CLAUDE.md rule 3).
_BINDING_LEN = 16

#: A tier is present in the mapping while its walk has more to give. ABSENCE means EXHAUSTED —
#: which is what lets the token shrink as tiers finish, and what makes "all tiers absent" the
#: same thing as "the walk is over" (at which point no token is minted at all).
TierCursor = dict[Tier, str]

#: How many CONSECUTIVE pages may make no progress before the walk is refused. One: the single
#: round trip a caller needs to drop its narrowing and let a recovered tier resume. See the
#: module docstring's point 3.
MAX_STALLED_PAGES = 1


def _binding(ns: Namespace) -> str:
    return sha256(ns.to_prefix().encode("utf-8")).hexdigest()[:_BINDING_LEN]


def encode_cursor(ns: Namespace, positions: TierCursor, *, stalls: int = 0) -> str | None:
    """Compose the per-tier positions into one opaque token, or ``None`` when the walk is done.

    Returning ``None`` for an empty mapping is the ``enumerate`` contract itself: *"``next_cursor
    is None`` iff the walk is exhausted"* (``ports/memory.py`` lines 78-80). Minting a token that
    decodes to "nothing left to walk" would make every caller take one extra round trip to learn
    what this return already says.
    """
    if not positions:
        return None
    payload = {
        "v": _FORMAT,
        "b": _binding(ns),
        "t": {tier.value: position for tier, position in positions.items()},
        # Consecutive no-progress pages. Omitted from the docstring's "positions only" framing on
        # purpose: it is walk BOOKKEEPING, not a position, and like the positions it can only ever
        # mis-position a walk inside the caller's own partition — it names no tenant and no scope.
        "s": stalls,
    }
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(ns: Namespace, cursor: str | None) -> tuple[TierCursor, int]:
    """Read a token back into per-tier positions, refusing anything not minted for ``ns``.

    Returns ``(positions, stalls)``. ``None`` yields an empty mapping, which every leg reads as
    "start at the beginning" — the first page of a walk and the end of one are distinguished by
    the CALLER's ``next_cursor`` being ``None``, never by this function.

    A malformed token raises rather than silently restarting. The two failures look identical to
    a caller and are not: restarting a walk the caller believed it was continuing re-serves rows
    it has already seen and never reaches the ones it has not — a paginated read that silently
    loops. Refusing is the only answer that cannot be mistaken for progress.
    """
    if cursor is None:
        return {}, 0
    payload = _decode_payload(cursor)
    if payload.get("v") != _FORMAT:
        raise NamespaceIsolationError("enumerate cursor has an unsupported format")
    if payload.get("b") != _binding(ns):
        # Deliberately the same error and the same wording as any other cross-partition refusal,
        # naming neither namespace: a probe must not be able to use a rejected cursor to learn
        # which partition minted it.
        raise NamespaceIsolationError("enumerate cursor does not belong to this namespace")
    raw_positions = payload.get("t")
    if not isinstance(raw_positions, dict):
        raise NamespaceIsolationError("enumerate cursor is malformed")
    positions: TierCursor = {}
    for key, value in raw_positions.items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise NamespaceIsolationError("enumerate cursor is malformed")
        try:
            tier = Tier(key)
        except ValueError as exc:
            raise NamespaceIsolationError("enumerate cursor names an unknown tier") from exc
        positions[tier] = value
    raw_stalls = payload.get("s", 0)
    if not isinstance(raw_stalls, int) or isinstance(raw_stalls, bool) or raw_stalls < 0:
        raise NamespaceIsolationError("enumerate cursor is malformed")
    return positions, raw_stalls


def _decode_payload(cursor: str) -> dict[str, Any]:
    padding = "=" * (-len(cursor) % 4)
    try:
        raw = base64.urlsafe_b64decode(cursor + padding)
        payload = json.loads(raw)
    except (binascii.Error, ValueError, UnicodeDecodeError) as exc:
        raise NamespaceIsolationError("enumerate cursor is malformed") from exc
    if not isinstance(payload, dict):
        raise NamespaceIsolationError("enumerate cursor is malformed")
    return payload
