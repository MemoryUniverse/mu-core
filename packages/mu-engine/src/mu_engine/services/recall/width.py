"""Context-budget-derived recall width — ACCURACY-PLAN-0831.md item 4 / RETRIEVAL-EVAL-0829.md §13.

**Why this exists.** mem0 sends the answering model **60** memories, MemOS **40** — both hardcoded
constants in an evaluation harness, tuned once against one dataset on one model
(`ACCURACY-PLAN-0831.md` §5.4/§7). This engine sends **10** (`mu_contracts.contracts.defaults.
DEFAULT_RECALL_LIMIT`), also a constant. The owner's design, and the one that generalises past a
single benchmark: derive the width from the CONSUMING model's own context budget, so a caller on a
million-token model gets a wide window and a caller on a small local model still gets a WORKING one
— never tuned by a human per deployment, never a crippled FULL-LOCAL baseline (root `CLAUDE.md`
boundary rule).

**The formula.** ``available = max_input_tokens - prompt_reserve_tokens - answer_reserve_tokens``;
``raw = floor(available / tokens_per_memory)``; the result is clamped to
``[min_limit, max_limit]``. Every term is a named, overridable :class:`~mu_engine.services.recall.
dto.RecallSettings` field — nothing here is a bare literal (DEV-STANDARDS rule 3).

**The measured constants the DEFAULTS use** (`docs/tracking/MULTIHOP-AND-LLM-TESTS-0831.md` §7,
reconstructed from the real harness prompts + the harness's own tokenizer over real LoCoMo turns,
not guessed): the mem0 ``ANSWER_PROMPT`` scaffold costs **~370 tokens** before a single recalled
line is added, and each recalled context line costs **~45 tokens** — ``(823 measured total input
at 10 lines - 370 overhead) / 10 ≈ 45.3``, matching the doc's own "~45 tokens/context line" figure.
These are real per-line/per-prompt costs for the shape of context this engine actually renders
(one memory per line), not an invented number.

**Why the ceiling defaults to 30, not 60.** `RETRIEVAL-EVAL-0829.md` §13 measured the recall@k
curve out to k=60 (still rising — R@60 0.7043) but ONLY ran real end-to-end ANSWER-QUALITY at
k=10 and k=30; k=30 was validated to help (+16.2pt overall, every category +11.7pt or more, against
a <=3pt run-to-run noise floor, §13.2) and §13.4 states plainly: *"this pass found only that it had
not yet turned down by k=30 ... the obvious next run is k=60, to find the actual turning point."*
Defaulting the cap to a width whose ANSWER-quality effect was never measured would be exactly the
"width that buys nothing but costs tokens" regression this mechanism exists to avoid — capping at
the last point actually shown to help, not at the last point merely shown to raise recall, is the
honest reading of that evidence. Raise ``max_derived_limit`` explicitly (it is config, not code)
once a k=60 (or wider) answer-quality run reports where the curve actually turns.

**Why it is a floor division, not a round.** Overshooting the budget by a fraction of a memory line
is a token-budget regression the caller pays for on every single call; undershooting by the same
fraction is nothing (the caller already has slack, since ``answer_reserve_tokens`` is itself a
reservation, not a hard wall). Round-down is the conservative direction.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from mu_engine.providers.catalog import Task

__all__ = ["ContextBudgetPort", "derive_recall_limit"]


@runtime_checkable
class ContextBudgetPort(Protocol):
    """The ONE seam :class:`~mu_engine.services.recall.service.RecallService` needs to derive a
    width — deliberately narrower than the whole :class:`~mu_engine.providers.model_router.
    ModelRouter` surface (structural typing, no import of the concrete router required by callers
    that only need this). ``ModelRouter.context_window`` already satisfies this Protocol; a
    composition root that has no real model layer wired (heuristic mode) simply injects ``None``
    for :class:`~mu_engine.services.recall.dto.RecallSettings`'s ``context_budget`` seam and
    derivation is skipped in favour of the static wire default — "no invented numbers"
    (`mu_engine.providers.shipped_catalog` states the same rule for the catalog itself).

    ``task`` is always :attr:`~mu_engine.providers.catalog.Task.ANSWER` at today's one call site
    (RANKED recall has no synthesis step yet, but the width it derives is sized for whichever
    model eventually reads the rendered context — the ANSWER task's configured model-group is the
    one place that model is already named, model-layer-spec §2.3) — the Protocol takes ``task``
    rather than hardcoding it so a future INJECT-mode caller sizing against a DIFFERENT consumer
    is a call-site change, not a new port.
    """

    def context_window(self, task: Task) -> int: ...


def derive_recall_limit(
    *,
    max_input_tokens: int,
    prompt_reserve_tokens: int,
    answer_reserve_tokens: int,
    tokens_per_memory: float,
    min_limit: int,
    max_limit: int,
) -> int:
    """Pure derivation — no I/O, no settings object, so it is trivial to mutation-test in
    isolation from every composition-root wiring question. Callers pass already-resolved
    scalars (:meth:`~mu_engine.services.recall.service.RecallService._derive_limit` is the one
    call site that resolves them from :class:`~mu_engine.services.recall.dto.RecallSettings` +
    a :class:`~mu_engine.services.recall.width.ContextBudgetPort`).

    ``min_limit``/``max_limit`` ALWAYS win over the arithmetic result — a budget so tight the raw
    division floors to 0 (or goes negative) still returns a floor-clamped, WORKING width rather
    than an empty or negative one (the "FULL-LOCAL must stay good" boundary rule, applied at the
    one place a tiny local context window could otherwise starve recall to nothing). A caller that
    wants NO floor passes ``min_limit=0``; this function does not invent one.

    Raises ``ValueError`` on a non-positive ``tokens_per_memory`` (a zero or negative per-memory
    cost makes the division either undefined or a fabricated infinite width — a misconfiguration,
    never a silent default) and on ``min_limit > max_limit`` (an inverted clamp range, caught here
    rather than producing a silently-empty or silently-swapped result downstream).
    """
    if tokens_per_memory <= 0:
        raise ValueError(f"tokens_per_memory must be > 0, got {tokens_per_memory!r}")
    if min_limit > max_limit:
        raise ValueError(f"min_limit ({min_limit}) must be <= max_limit ({max_limit})")

    available = max_input_tokens - prompt_reserve_tokens - answer_reserve_tokens
    raw = int(available // tokens_per_memory) if available > 0 else 0
    return max(min_limit, min(max_limit, raw))
