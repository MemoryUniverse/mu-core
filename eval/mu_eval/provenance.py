"""Full run provenance — everything needed to reproduce a reported number from its own artifact.

HOW-THEY-MEASURE-0901.md item 2 / F5: "Eval reports record no ranker configuration at all — no
channel_pool_size, no weight_*, no floor_protect_limit, no rerank_enabled. A run's provenance
lives in whatever shell script launched it, which is exactly how the false multi-hop claim was
constructed." A grep for ``git rev-parse|commit_hash|wandb|mlflow|json.dump(...config...)`` across
all seven competitor evaluation trees this project studied returned only ``logger.debug`` calls
(the same finding, F5: "not one of the seven is ahead of us" on provenance) — the best of them
(HippoRAG) dumps a config object to ``logger.debug`` at a level invisible by default and then
writes results without it. This module is what closes that gap for THIS harness: pure,
side-effect-free (besides reading files/git/env, never writing anything), so every function here
is independently unit-testable without a store or a network call.

Four things go in the artifact:
  1. The RANKER config actually in effect (``recall_settings_snapshot``) — see that function's own
     docstring for the one thing it can and cannot fix from inside ``eval/``.
  2. The MODEL actually served, read from the live response (``openai_chat.OpenAICompatChat.
     last_model``/``served_models``), not the string a caller asked for — a deployment can silently
     answer with something other than what was requested; ``model_provenance`` packages both.
  3. The DATASET's content hash (``dataset_sha256``) — "the same LoCoMo file" is a claim that
     should be checkable, not assumed from a shared filename.
  4. The CODE REVISION this ran under (``git_revision``), including whether the tree was dirty —
     an artifact from a dirty tree is still useful, but a reader deserves to know before trusting
     it as reproducible from git alone.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any

__all__ = [
    "build_provenance",
    "dataset_sha256",
    "git_revision",
    "model_provenance",
    "recall_settings_snapshot",
]


def dataset_sha256(path: str | Path) -> str:
    """SHA-256 of the dataset file's raw bytes. "Ran on locomo10.json" is not a reproducibility
    claim; "ran on the file whose content hashes to <this>" is."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_revision(repo_dir: str | Path | None = None) -> tuple[str | None, bool | None]:
    """``(commit_hash, is_dirty)`` for the repo containing ``repo_dir`` (default: this file's own
    repo — ``mu-core``, where ``eval/`` lives). Best-effort: ``(None, None)`` on any failure (no
    git binary, not a git checkout, timeout) rather than raising — a provenance gap is honestly
    reported as ``None``, never allowed to abort a real measurement run over a missing tool.
    """
    start = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent
    try:
        rev = subprocess.run(  # noqa: S603 -- fixed argv, no shell, no untrusted input
            ["git", "rev-parse", "HEAD"],  # noqa: S607 -- git resolved via PATH, deliberately
            cwd=start,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if rev.returncode != 0:
            return None, None
        commit = rev.stdout.strip()
        status = subprocess.run(  # noqa: S603 -- fixed argv, no shell, no untrusted input
            ["git", "status", "--porcelain"],  # noqa: S607 -- git resolved via PATH, deliberately
            cwd=start,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        return commit, dirty
    except (OSError, subprocess.SubprocessError):
        return None, None


def recall_settings_snapshot() -> dict[str, Any]:
    """The ``RecallSettings``/``IngestSettings`` actually in effect for a ``LocalMemory`` run,
    read from the SAME cached accessor the composition root itself reads
    (``mu_engine.config.get_engine_settings()``).

    THE HONEST LIMIT OF THIS FUNCTION, stated because the brief specifically warned about a config
    that "isn't merely unrecorded, it's ambient": ``LocalMemory.__init__``'s ``settings=`` parameter
    binds only ``mu_contracts.config.settings.Settings`` (storage connection info) — there is no
    ``engine_settings=``/``recall=`` override seam on ``LocalMemory`` or ``LocalContainer`` at all
    (``mu_local.composition.LocalContainer.__init__``, ``packages/mu-local/src/mu_local/
    composition.py:302-313``: the constructor takes ``engine_settings: EngineSettings | None``, but
    ``LocalMemory`` never forwards one). So the ranker config governing an eval run is, today,
    UNCONDITIONALLY whatever ``get_engine_settings()`` resolves from process environment — this
    harness (in ``eval/``, which owns none of ``mu-local``'s composition root) cannot select a
    different one, only observe and record the one that actually ran. That is still real progress
    on F2/F5: "unrecorded" becomes "recorded, and honestly labelled ambient" rather than staying
    invisible. Making it selectable is a ``mu-local`` change and is out of this file's ownership —
    left as a following item, not silently worked around here.
    """
    from mu_engine.config import get_engine_settings

    engine_settings = get_engine_settings()
    return {
        "recall": engine_settings.recall.model_dump(mode="json"),
        "ingest": engine_settings.ingest.model_dump(mode="json"),
        "ambient": True,  # see docstring: never selected by this harness, always env-resolved
    }


def model_provenance(chat: Any, *, label: str) -> dict[str, Any]:
    """Package ``{requested, served}`` model identity for one ``OpenAICompatChat``-shaped client.

    ``served`` is a SORTED LIST of every distinct ``body["model"]`` value that client's live
    responses actually carried (``OpenAICompatChat.served_models`` — a set, because a deployment
    can drift mid-run and that is itself worth surfacing, not silently overwritten by the last
    call). Empty when the client made zero calls (e.g. every row was skipped) — reported as an
    empty list, never omitted or defaulted to the requested string, so a reader cannot mistake "we
    never actually confirmed what served this" for "we confirmed it matched"."""
    return {
        "label": label,
        "requested": getattr(chat, "requested_model", None),
        "served": sorted(getattr(chat, "served_models", set()) or set()),
    }


def build_provenance(
    *,
    dataset_path: str | Path,
    chats: dict[str, Any] | None = None,
    repo_dir: str | Path | None = None,
    include_captions: bool | None = None,
) -> dict[str, Any]:
    """The one call site assembles: dataset hash, code revision, ranker config, and model
    provenance for every named chat client (``chats={"answer": answer_chat, "judge": judge_chat}``
    typically). Returns a plain dict (not a pydantic model) so it merges cleanly into whatever
    report dict is about to be written to ``--out`` (``__main__.py``'s own ``_write``), without
    requiring every report schema in this package to carry an identical nested field.

    ``include_captions`` (T7, ``TRACE-0923.md`` §7/§5.5): whether THIS run's ``load_locomo`` call
    included ``blip_caption`` text. Recorded here — not as a new field on ``RunReport``/
    ``AnswerQualityReport`` (both ``extra="forbid"``) — for the same reason ``recall_settings``
    already lives in this free-form dict rather than a strict nested model: reading a caption-arm
    artifact back with the OLDER pydantic schema must keep working, and a run's comparability to
    published mem0/MemOS numbers is exactly the kind of fact a reader needs at the SAME place they
    already look for "was this run reproducible", not a second place to remember to check.
    ``None`` (the default) means "the caller did not say" — reported honestly as ``None``, never
    silently defaulted to ``False``, so an artifact written before this fix cannot be misread as
    having positively confirmed captions were excluded.
    """
    commit, dirty = git_revision(repo_dir)
    provenance: dict[str, Any] = {
        "dataset_path": str(dataset_path),
        "dataset_sha256": dataset_sha256(dataset_path),
        "dataset_include_captions": include_captions,
        "code_revision": commit,
        "code_dirty": dirty,
        "recall_settings": recall_settings_snapshot(),
        "models": [model_provenance(chat, label=label) for label, chat in (chats or {}).items()],
    }
    return provenance
