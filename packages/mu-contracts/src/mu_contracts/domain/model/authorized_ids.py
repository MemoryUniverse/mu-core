"""Model-A ``authorized_ids`` — the ONE stamp vocabulary, shared by the writer and the reader.

Authority: ``CANONICAL-CONTRACTS.md`` §7.4 (*"the exploded set of PRINCIPAL ids permitted to read
that item — and nothing else"*). §7.4 pins BOTH halves of one filter:

* the WRITE half — *"It is STAMPED at write/sync time from the session participant set +
  materialized ACL rows, resolving every role/session grant down to concrete principals"*;
* the READ half — a point is returned only while its stamp still contains the caller principal id
  (*"recall never returns a point whose ``authorized_ids`` list no longer contains the caller
  principal id"*).

Before this module the two halves were written independently at every site that needed them (the
Qdrant payload mapper reads ``metadata["authorized_ids"]``; the FalkorDB mapper mirrors it; the
``mu-server`` governance stamper re-derives its own subject validation). That is the shape
CANONICAL §7.4's CONTRACT-TEST OBLIGATION exists to prevent: *"no role id and no session id is ever
written into a point's ``authorized_ids`` at ANY stamp/sync/re-stamp site"* is a property of the
VOCABULARY, so the vocabulary — the key name, what a legal subject is, and what "permitted" means —
lives in one place that every plane imports.

**Why the predicate lives here and not only in a store's query compiler.** §7.4's mechanism
sentence (*"SERVER-SIDE inside filterable-HNSW BEFORE top-k truncation"*) is written for a vector
store. The STM tier is a Redis ZSET + one key per row: it has no filterable index and cannot
execute that sentence, which is exactly how the STM recency-floor arm fell out of Model-A entirely
and shipped a SHARED-plane read with NO authorization on it at all (ARCHITECTURE-DELTAS AD-128,
reproduced against real stores). :func:`model_a_permits` is the SAME predicate compiled in Python
over an ALREADY-BOUNDED window (``RecallSettings.recency_floor_limit``), never an over-fetch — see
the delta for the open question of whether §7.4's *"NEVER a Python post-filter"* prohibits the
mechanism or the pre-truncation property.

**Fail-closed is the whole point.** An item whose stamp is absent or empty is an item no
governance decision has been recorded for; :func:`model_a_permits` answers ``False`` for it. A
caller with no identity set is a caller nobody resolved; it answers ``False`` too. Neither is a
silent empty at the call site — the tier repositories raise
:class:`~mu_contracts.domain.errors.CallerIdentitySetRequiredError` when a SHARED read arrives with
no caller set, because that is a wiring bug, not a denial.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any, Final

from mu_contracts.domain.errors import StampSubjectError
from mu_contracts.domain.model.recall import CallerIdentitySet

__all__ = [
    "AUTHORIZED_IDS_KEY",
    "model_a_permits",
    "stamp_of",
    "validate_stamp_subjects",
]

#: The one key the stamp is carried under — in ``MemoryItem.metadata`` (STM/JSON), in the Qdrant
#: payload keyword index, and as the FalkorDB list property. Every reader and writer names it
#: through this constant so a rename cannot desynchronize the tiers.
AUTHORIZED_IDS_KEY: Final[str] = "authorized_ids"

#: The id kinds §7.4 forbids in a stamp, matched by the prefixes THIS system mints for them.
#: A NEGATIVE check, deliberately (the same choice ``mu-server``'s ``AuthorizedIdsStamper`` makes):
#: a positive allowlist of principal shapes would silently start DROPPING legitimate ids the day a
#: new principal-id scheme lands, and a stamp that silently drops an id is an access that silently
#: disappears. A role or session token in a stamp is an offboarding hole (leaving the role would
#: not drop access); a device id is an ACL bypass (§7.11: a device is a replica endpoint of ONE
#: principal, never a grantee).
_FORBIDDEN_SUBJECT_PREFIXES: Final[tuple[str, ...]] = (
    "dev_",
    "device_",
    "role_",
    "ses_",
    "sess_",
)

#: A stamped id may not carry a separator that would let two ids be read as one by a consumer
#: splitting a keyword list.
_SUBJECT_RE: Final = re.compile(r"^[A-Za-z0-9_.:\-]{1,128}$")


def validate_stamp_subjects(ids: Iterable[str]) -> frozenset[str]:
    """Return ``ids`` as a stampable principal set, or RAISE (§7.4 CONTRACT-TEST OBLIGATION 1).

    Refuses a role id, a session id and a device id — the three kinds §7.4 says are never written
    into a point's ``authorized_ids`` — and any id carrying a keyword-list separator. A refusal,
    not a filter: dropping a bad subject silently would turn a governance mistake into a quiet,
    unreviewable loss (or grant) of access.
    """
    out: set[str] = set()
    for raw in ids:
        subject = raw.strip()
        if not subject:
            raise StampSubjectError("an empty id is not a principal id (CANONICAL §7.4)")
        lowered = subject.lower()
        for prefix in _FORBIDDEN_SUBJECT_PREFIXES:
            if lowered.startswith(prefix):
                raise StampSubjectError(
                    f"authorized_ids holds EXPLODED PRINCIPAL ids only (CANONICAL §7.4); "
                    f"{prefix!r}-prefixed subjects (role/session/device) are refused — explode "
                    f"the grant to its member principals at stamp time"
                )
        if not _SUBJECT_RE.match(subject):
            raise StampSubjectError(
                "an authorized_ids subject must match [A-Za-z0-9_.:-]{1,128} (a separator in a "
                "keyword-list member lets two ids be read as one)"
            )
        out.add(subject)
    return frozenset(out)


def stamp_of(metadata: Mapping[str, Any] | None) -> frozenset[str]:
    """The Model-A stamp carried by an item's ``metadata``, as a set. Absent/empty → empty set.

    Tolerant of the shapes the stamp legitimately round-trips through (a JSON list out of Redis, a
    tuple/set out of a fixture) and of nothing else: a non-iterable or a bare string is a
    corrupted stamp and reads as EMPTY, which :func:`model_a_permits` denies — never as permissive.
    """
    if not metadata:
        return frozenset()
    raw = metadata.get(AUTHORIZED_IDS_KEY)
    if raw is None or isinstance(raw, str | bytes):
        return frozenset()
    if not isinstance(raw, Iterable):
        return frozenset()
    return frozenset(str(subject) for subject in raw)


def model_a_permits(
    *, stamp: frozenset[str], caller_identity_set: CallerIdentitySet | None
) -> bool:
    """``MatchAny(authorized_ids, caller_identity_set)`` — the §7.4 predicate, one implementation.

    ``True`` iff the item's exploded principal stamp INTERSECTS the caller principal set. Every
    other case denies, and each of them is a real state this system reaches:

    * ``caller_identity_set is None`` — nobody resolved a caller. Denied (the tier repos raise
      before reaching here on a SHARED read; this is the belt).
    * ``caller_identity_set == frozenset()`` — a caller resolved to nothing. Denied: an empty
      caller set authorizes NOTHING, which is what ``RecallService`` already relies on when it
      coerces a missing SHARED caller set to ``frozenset()`` (*"the safe direction, never an
      over-broad match"*).
    * ``stamp == frozenset()`` — the item carries no governance decision. Denied. This is the
      case that made AD-128 exploitable in the other direction: an unstamped SHARED row read as
      "unrestricted" is a room-wide leak, and it is the reading a naive ``if stamp:`` guard takes.
    """
    if not caller_identity_set or not stamp:
        return False
    return not stamp.isdisjoint(caller_identity_set)
