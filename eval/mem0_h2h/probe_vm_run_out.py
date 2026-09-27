"""Smoke test for `vm_run.sh`'s own plumbing — the runner, not the engine. Seconds, $0, no store.

    H2H_OUT=/tmp/x.json H2H_LABEL=probe ./eval/mem0_h2h/vm_run.sh eval/mem0_h2h/probe_vm_run_out.py

Asserts, ON the VM, the three things every arm silently depends on and no test covered:

1. **`H2H_OUT` is writable from the remote shell.** It was not: every driver in this repo passes a
   LAPTOP path, `H2H_OUT` is opened on the VM, and `ours_arm.py` writes it LAST — so an arm ran
   correctly, printed its `gold_in_context`, and then died with `FileNotFoundError` and exit 1 with
   no artifact (AD-336; two arms of that pass lost their per-row export exactly so). `vm_run.sh` now
   stages the file VM-side and copies it back, and this probe is what proves the copy-back works
   where it runs rather than where it was written.
2. **`H2H_EVAL_DIR` is exported and importable**, so `sys.path` reaches `mu_eval`.
3. **The env forwarding arrived**, including anything named in `MU_FORWARD` — the mechanism whose
   absence once ran an arm under defaults while its label claimed otherwise.

Exits non-zero with a named reason on any failure; prints the JSON it wrote on success.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("H2H_EVAL_DIR", str(Path(__file__).resolve().parent.parent)))

from mem0_h2h import emit


def main() -> int:
    out = os.environ.get("H2H_OUT")
    if not out:
        emit("probe: H2H_OUT is unset — nothing to prove")
        return 2

    eval_dir = os.environ.get("H2H_EVAL_DIR")
    if not eval_dir or not Path(eval_dir).is_dir():
        emit(f"probe: H2H_EVAL_DIR unusable: {eval_dir!r}")
        return 3
    sys.path.insert(0, eval_dir)
    try:
        from mu_eval.runner import gold_answer_word_coverage
    except ImportError as exc:  # pragma: no cover - the failure this exists to name
        emit(f"probe: mu_eval not importable from H2H_EVAL_DIR: {exc}")
        return 4

    # A real call, so "importable" is not confused with "usable".
    coverage = gold_answer_word_coverage("Sweden", "- Caroline moved from Sweden in 2009.")
    if coverage != 1.0:
        emit(f"probe: gold_answer_word_coverage returned {coverage}, expected 1.0")
        return 5

    payload = {
        "probe": "vm_run_out",
        "on_vm": os.environ.get("MU_ON_VM"),
        "label": os.environ.get("H2H_LABEL"),
        "h2h_out_seen_by_the_remote_shell": out,
        "eval_dir": eval_dir,
        "coverage_call": coverage,
        # Enumerated BY PREFIX, never from `MU_FORWARD` — the first version of this probe read
        # the names out of `$MU_FORWARD`, which `vm_run.sh` does not itself forward, so the dict
        # came back `{}` and the assertion was vacuous while looking like a pass. Reporting what
        # the remote shell actually HAS cannot be vacuous in that way.
        "engine_env_seen_on_the_vm": {
            k: v for k, v in sorted(os.environ.items()) if k.startswith(("H2H_", "MU_"))
        },
    }
    if len(payload["engine_env_seen_on_the_vm"]) < 3:  # type: ignore[arg-type]
        emit(f"probe: env forwarding looks broken: {payload['engine_env_seen_on_the_vm']}")
        return 6
    Path(out).write_text(json.dumps(payload, indent=2))
    emit(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
