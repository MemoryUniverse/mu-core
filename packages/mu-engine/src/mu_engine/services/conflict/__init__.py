"""``services/conflict`` — the deterministic cross-device total order (CANONICAL §7.17 item 4a).

Only ``total_order_key`` lives here today (spec D-29). ``ConflictAdjudicator`` (the LLM-judged
single-replica supersession decision, ADR 0037) stays in ``mu_engine.lifecycle.conflict`` — a
DIFFERENT concern (one replica deciding a NEW winner) from this module's (every replica
re-deriving the SAME winner over an already-replicated delta set, with no coordination).
"""

from mu_engine.services.conflict.order import total_order_key

__all__ = ["total_order_key"]
