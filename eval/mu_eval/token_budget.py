"""Token counting/truncation for prompts going to a HARD per-request token ceiling.

Built for one concrete problem: Azure Foundry's ``Ministral-3B`` deployment caps every request at
400 tokens, prompt + completion together (measured from the live response headers, not guessed —
see ``docs/tracking/STATE-AND-DEFECTS-0829.md`` / ``ARCHITECTURE-DELTAS.md`` AD- entries around
2026-08-30). A prompt that is merely "usually short" 429s the first time a user rambles; this
module makes truncation a stated, testable rule instead of a hope.

TOKENIZER CHOICE (documented assumption, not hidden): there is no public tokenizer checkpoint for
``Ministral-3B`` itself. Mistral released Ministral-3B and Ministral-8B together on 2024-10-16 as
one family sharing the same "Tekken v3" tokenizer (131072-token vocab); ``mistralai/Ministral-8B-
Instruct-2410`` publishes that tokenizer on the Hub and IS used here as the 3B model's stand-in.
This is an approximation for the pre-flight truncation decision only — the actual proof of fit is
the live response's own ``usage.prompt_tokens`` (see ``mu_eval judge-probe``), which is the exact
count from the real model, not this proxy.

FALLBACK, for when the tokenizer cannot be loaded (offline, no cached weights, transformers not
importable): a conservative characters-per-token ratio that OVER-counts rather than under-counts,
so a truncation decision made without the real tokenizer errs toward cutting more, never toward a
silent 429. English text on the Tekken tokenizer measured 3.5-4.3 chars/token across this project's
own LoCoMo prompts; the fallback uses 3.0 to stay on the safe (over-counting) side even for
token-dense text (dates, numbers, punctuation-heavy strings — exactly what this judge's inputs are).
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Protocol

__all__ = ["count_tokens", "truncate_to_tokens"]

_TOKENIZER_MODEL = "mistralai/Ministral-8B-Instruct-2410"
_FALLBACK_CHARS_PER_TOKEN = 3.0  # deliberately conservative (over-counts) -- see module docstring


class _Encoder(Protocol):
    def encode(self, text: str, add_special_tokens: bool = ...) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...


@lru_cache(maxsize=1)
def _tokenizer() -> _Encoder | None:
    """Best-effort load, cached once per process. Returns ``None`` on any failure so callers fall
    back to the conservative char-based estimate rather than crashing a measurement run."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(  # type: ignore[no-any-return,no-untyped-call]
            _TOKENIZER_MODEL
        )
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Exact count via the Ministral-family tokenizer when available, else the conservative
    char-based fallback (see module docstring)."""
    tok = _tokenizer()
    if tok is not None:
        return len(tok.encode(text, add_special_tokens=False))
    return math.ceil(len(text) / _FALLBACK_CHARS_PER_TOKEN)


def truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Cut ``text`` to at most ``max_tokens`` tokens, keeping the PREFIX (the rubric this judge
    applies only needs the topic/fact, which answers state up front; LoCoMo generated answers are
    themselves instructed to be "less than 5-6 words", so truncation is a safety rail for the rare
    case that ignores that, not the common path). Returns ``text`` unchanged when it already fits.
    """
    if max_tokens <= 0:
        return ""
    tok = _tokenizer()
    if tok is not None:
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return str(tok.decode(ids[:max_tokens]))
    # Fallback: same prefix rule, estimated in characters instead of tokens.
    max_chars = int(max_tokens * _FALLBACK_CHARS_PER_TOKEN)
    return text if len(text) <= max_chars else text[:max_chars]
