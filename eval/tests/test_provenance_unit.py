"""Unit tests for ``provenance.py`` — item 2 (full provenance in the artifact).

Every assertion below is mutation-checkable: break the function on purpose (swap the hash
algorithm, drop the ``cwd=``, return the requested model instead of the served one) and the
matching test goes RED.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest
from mu_eval.provenance import (
    build_provenance,
    dataset_sha256,
    git_revision,
    model_provenance,
    recall_settings_snapshot,
)

pytestmark = pytest.mark.unit


def test_dataset_sha256_matches_a_direct_hash_of_the_bytes(tmp_path: Path) -> None:
    f = tmp_path / "locomo10.json"
    f.write_bytes(b'{"hello": "world"}')
    assert dataset_sha256(f) == hashlib.sha256(b'{"hello": "world"}').hexdigest()


def test_dataset_sha256_changes_when_content_changes(tmp_path: Path) -> None:
    f = tmp_path / "locomo10.json"
    f.write_bytes(b"version one")
    first = dataset_sha256(f)
    f.write_bytes(b"version two")
    second = dataset_sha256(f)
    assert first != second, "a content change must change the hash — else 'same file' is unproven"


def test_git_revision_reports_the_real_head_of_this_checkout() -> None:
    repo_root = Path(__file__).resolve().parents[2]  # eval/tests -> eval -> mu-core
    expected = subprocess.run(  # noqa: S603 -- fixed argv, test-only
        ["git", "rev-parse", "HEAD"],  # noqa: S607 -- git resolved via PATH, deliberately
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    commit, dirty = git_revision(repo_root)
    assert commit == expected
    assert dirty in (True, False)


def test_git_revision_is_none_none_outside_any_git_checkout(tmp_path: Path) -> None:
    # tmp_path is a fresh directory with no .git anywhere above it inside the sandbox — best
    # effort: if the CI temp dir happens to sit under a git checkout this assertion would need
    # revisiting, but the sandbox's own tmp root is not one.
    commit, dirty = git_revision(tmp_path)
    assert (commit, dirty) == (None, None)


def test_recall_settings_snapshot_carries_the_ranker_knobs_the_brief_named() -> None:
    """F5's own list: channel_pool_size, weight_*, floor_protect_limit, rerank_enabled — a report
    that omits any of these is exactly the defect this function exists to close."""
    snapshot = recall_settings_snapshot()
    recall = snapshot["recall"]
    for key in (
        "channel_pool_size",
        "weight_stm",
        "weight_mtm",
        "weight_ltm",
        "floor_protect_limit",
        "rerank_enabled",
        "rrf_k",
    ):
        assert key in recall, f"{key!r} missing from the recorded ranker config"
    assert snapshot["ambient"] is True


def test_recall_settings_snapshot_dumps_the_whole_model_not_a_hand_picked_subset() -> None:
    """AD-228 (ARCHITECTURE-DELTAS.md) leans on this: the reason a FUTURE recall knob (e.g. an
    S1b neighbour-expansion setting `services/recall/` has not written yet) will show up in every
    run's provenance with NO matching change needed here is that `recall_settings_snapshot`
    dumps `engine_settings.recall.model_dump(...)` WHOLE, never a hand-picked list of field names.
    Proved directly against `RecallSettings.model_fields` rather than merely asserted — a future
    edit that narrows the dump to a curated subset turns this test RED before it ships."""
    from mu_engine.services.recall.dto import RecallSettings

    snapshot = recall_settings_snapshot()
    assert set(snapshot["recall"]) == set(RecallSettings.model_fields)


class _StubChat:
    def __init__(self, *, requested: str, served: set[str]) -> None:
        self.requested_model = requested
        self.served_models = served


def test_model_provenance_reports_served_not_requested_when_they_differ() -> None:
    chat = _StubChat(requested="gpt-5", served={"gpt-5-2026-08-07"})
    info = model_provenance(chat, label="answer")
    assert info == {"label": "answer", "requested": "gpt-5", "served": ["gpt-5-2026-08-07"]}


def test_model_provenance_served_list_is_sorted_and_deduplicated_by_the_underlying_set() -> None:
    chat = _StubChat(requested="gpt-5", served={"b-model", "a-model"})
    info = model_provenance(chat, label="judge")
    assert info["served"] == ["a-model", "b-model"]


def test_model_provenance_empty_when_the_client_never_actually_served_a_call() -> None:
    chat = _StubChat(requested="gpt-5", served=set())
    info = model_provenance(chat, label="answer")
    assert (
        info["served"] == []
    ), "an unconfirmed model must be an empty list, never the requested string"


def test_build_provenance_merges_dataset_code_and_model_info(tmp_path: Path) -> None:
    f = tmp_path / "locomo10.json"
    f.write_bytes(b"[]")
    provenance = build_provenance(
        dataset_path=f,
        chats={"answer": _StubChat(requested="gpt-5", served={"gpt-5"})},
    )
    assert provenance["dataset_path"] == str(f)
    assert provenance["dataset_sha256"] == dataset_sha256(f)
    assert "recall_settings" in provenance
    assert provenance["models"] == [{"label": "answer", "requested": "gpt-5", "served": ["gpt-5"]}]


def test_build_provenance_with_no_chats_reports_an_empty_models_list(tmp_path: Path) -> None:
    f = tmp_path / "locomo10.json"
    f.write_bytes(b"[]")
    provenance = build_provenance(dataset_path=f)
    assert provenance["models"] == []


def test_build_provenance_records_include_captions_when_the_caller_says(tmp_path: Path) -> None:
    """T7 (ARCHITECTURE-DELTAS.md AD-227): a run's caption-arm must be readable from the SAME
    artifact a reader already checks for reproducibility, not a second place to remember."""
    f = tmp_path / "locomo10.json"
    f.write_bytes(b"[]")
    on = build_provenance(dataset_path=f, include_captions=True)
    off = build_provenance(dataset_path=f, include_captions=False)
    assert on["dataset_include_captions"] is True
    assert off["dataset_include_captions"] is False


def test_build_provenance_include_captions_defaults_to_none_not_false(tmp_path: Path) -> None:
    """`None` means "the caller did not say" — never silently defaulted to False, so an artifact
    from before this fix (or a caller that forgot to pass it) cannot be misread as having
    positively confirmed captions were excluded."""
    f = tmp_path / "locomo10.json"
    f.write_bytes(b"[]")
    provenance = build_provenance(dataset_path=f)
    assert provenance["dataset_include_captions"] is None
