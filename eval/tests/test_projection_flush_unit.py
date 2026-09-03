"""The pre-run cost projection must reach the operator BEFORE the run spends anything.

This is not a formatting test. ``__main__._print`` writes to ``sys.stdout`` and never flushes;
CPython block-buffers stdout (8 KB) whenever it is not a TTY, and every real invocation of this
harness is exactly that — ``eval/vm_eval.sh`` runs the CLI over ``ssh`` and its callers redirect
the output to a log. MEASURED on ``mu-dev-vm`` during the verification of this feature: with the
projection unflushed, the whole ``PROJECTED COST`` block — including the line that says "abandon
now with Ctrl-C ... nothing has been spent yet" — did not appear until after the run's own ingest
logging, i.e. after the spending had already begun. A pre-spend warning delivered post-spend is
not a warning.

So the flush is asserted at the point that matters: it must have happened by the time
``_print_projected_cost`` RETURNS, because the very next thing its caller does is resolve the API
key and start making paid calls (``__main__._cmd_answer_quality``).
"""

from __future__ import annotations

import sys
from pathlib import Path

_EVAL_ROOT = str(Path(__file__).resolve().parents[1])
if _EVAL_ROOT not in sys.path:  # pragma: no cover - import shim, mirrors eval/conftest.py
    sys.path.insert(0, _EVAL_ROOT)

import pytest  # noqa: E402
from mu_eval.__main__ import _print_projected_cost  # noqa: E402
from mu_eval.usage import MeanQueryUsage  # noqa: E402


class _RecordingStdout:
    """A stdout stand-in that records the ORDER of writes and flushes, so a test can assert that
    a flush actually happened after the last line rather than merely that a flush happened."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str]] = []

    def write(self, text: str) -> int:
        self.events.append(("write", text))
        return len(text)

    def flush(self) -> None:
        self.events.append(("flush", ""))

    @property
    def text(self) -> str:
        return "".join(t for kind, t in self.events if kind == "write")


_MEANS = {
    "answer": MeanQueryUsage(
        label="answer", mean_prompt_tokens=880.5, mean_completion_tokens=400.0, source="test"
    ),
    "judge": MeanQueryUsage(
        label="judge", mean_prompt_tokens=116.7, mean_completion_tokens=76.0, source="test"
    ),
}


def _run(monkeypatch: pytest.MonkeyPatch) -> _RecordingStdout:
    out = _RecordingStdout()
    monkeypatch.setattr(sys, "stdout", out)
    _print_projected_cost(
        n_queries=1531,
        answer_model="gpt-5",
        judge_model="gpt-5",
        rate_card_path=None,
        means=_MEANS,
    )
    return out


def test_projection_is_flushed_before_the_function_returns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _run(monkeypatch)
    assert any(kind == "flush" for kind, _ in out.events), (
        "the projection was never flushed — under a pipe (vm_eval.sh over ssh, output "
        "redirected to a log) it would stay in stdout's 8 KB buffer past the point where the "
        "run starts spending"
    )


def test_the_flush_comes_after_the_last_projection_line_not_in_the_middle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    out = _run(monkeypatch)
    kinds = [kind for kind, _ in out.events]
    assert kinds[-1] == "flush", (
        f"the last event was {kinds[-1]!r}, so at least one projection line is still buffered "
        "when this function returns and its caller starts making paid calls"
    )


def test_the_projection_block_still_says_what_it_costs_and_that_nothing_is_spent_yet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    text = _run(monkeypatch).text
    assert "PROJECTED COST for 1531 eligible queries" in text
    assert "nothing has been spent yet" in text
    assert "abandon now with Ctrl-C" in text
    # 1531 * (880.5 * 1.25 + 400 * 10) / 1e6 + 1531 * (116.7 * 1.25 + 76 * 10) / 1e6
    assert "TOTAL projected: $9.20" in text
