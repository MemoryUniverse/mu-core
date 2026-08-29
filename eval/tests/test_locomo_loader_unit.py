"""Loader tests over a fixture with the EXACT shape verified by reading the real locomo10.json.

The fixture is not "a made-up corpus to evaluate on" — the evaluation itself never touches it.
It exists so the parsing of the real file's schema (session_N keys, dia_id, evidence, category)
is provable without shipping 2.8 MB of someone else's dataset into this repo.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mu_eval.locomo import ADVERSARIAL_CATEGORY, load_locomo

pytestmark = pytest.mark.unit

_SAMPLE = [
    {
        "sample_id": "conv-99",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "7 May 2023",
            "session_1": [
                {"speaker": "Caroline", "dia_id": "D1:1", "text": "Hey Mel!"},
                {"speaker": "Melanie", "dia_id": "D1:2", "text": "Hey Caroline!"},
                {"speaker": "Caroline", "dia_id": "D1:3", "text": "", "img_urls": ["x"]},
            ],
            "session_2_date_time": "9 May 2023",
            "session_2": [{"speaker": "Melanie", "dia_id": "D2:1", "text": "Back again."}],
        },
        "qa": [
            {"question": "Who said hi?", "answer": "Caroline", "evidence": ["D1:1"], "category": 4},
            {"question": "Nothing here", "adversarial_answer": "No info", "category": 5},
        ],
    }
]


def _write(tmp_path: Path) -> Path:
    path = tmp_path / "locomo_fixture.json"
    path.write_text(json.dumps(_SAMPLE), encoding="utf-8")
    return path


def test_sessions_are_ordered_numerically_not_lexically(tmp_path: Path) -> None:
    # "session_10" sorts before "session_2" as a STRING; the loader must order by the integer.
    payload = json.loads(json.dumps(_SAMPLE))
    conv = payload[0]["conversation"]
    conv["session_10"] = [{"speaker": "Melanie", "dia_id": "D10:1", "text": "Much later."}]
    conv["session_10_date_time"] = "1 Jan 2024"
    path = tmp_path / "ordered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    turns = load_locomo(path)[0].turns
    assert [t.dia_id for t in turns] == ["D1:1", "D1:2", "D2:1", "D10:1"]


def test_empty_bodied_turns_are_dropped(tmp_path: Path) -> None:
    turns = load_locomo(_write(tmp_path))[0].turns
    assert "D1:3" not in {t.dia_id for t in turns}


def test_gold_evidence_and_adversarial_flag_are_parsed(tmp_path: Path) -> None:
    queries = load_locomo(_write(tmp_path))[0].queries
    assert queries[0].evidence == ("D1:1",)
    assert not queries[0].is_adversarial
    assert queries[1].category == ADVERSARIAL_CATEGORY
    assert queries[1].is_adversarial


def test_missing_dataset_raises_rather_than_falling_back(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="never falls back to a synthetic corpus"):
        load_locomo(tmp_path / "absent.json")
