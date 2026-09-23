"""LoCoMo dataset loader — REAL labelled data, no synthetic corpus.

WHY LoCoMo and not a hand-rolled corpus: CLAUDE.md rule 1/2 (work from the actual reference
source, never guess) and DEV-STANDARDS "Evaluate retrieval/extraction quality on REAL data with a
repeatable harness ... use the dataset's OFFICIAL scorer where one exists". LoCoMo is the one
long-horizon conversational-memory dataset already provisioned on this machine
(`/home/user/D/abstract_project/mma/data/locomo/locomo10.json`, HF `snap-stanford/locomo`, MIT,
recorded in `mma/data/MANIFEST.md`), and it is the dataset whose official LLM-judge every
comparable system (mem0, MemOS, Zep) reports against.

WHY it can carry a RETRIEVAL metric at all: every LoCoMo QA row carries an ``evidence`` list of
dialogue-turn ids (``"D1:3"`` = session 1, turn 3). Those ids are the dataset's own gold
relevance labels over the dialogue turns, so recall@k / MRR / nDCG over the retrieved turn set are
computed against LABELS SHIPPED WITH THE DATASET — not against a relevance judgement this harness
invented. The IR metrics themselves are textbook (see ``metrics.py``); the LABELS are the
dataset's.

Schema, verified by reading the file (never assumed):

    locomo10.json = list[10] of samples, each:
        sample_id: str
        conversation: {speaker_a, speaker_b,
                       session_1: [ {speaker, dia_id, text, (img_urls...)}, ... ],
                       session_1_date_time: str, ... session_N ...}
        qa: [ {question, answer, evidence: ["D1:3", ...], category: int}, ... ]
        observation / event_summary / session_summary: derived summaries — NOT used here
        (they are a different system's intermediate output, not ground truth).

Category codes, from the LoCoMo paper's own mapping as recorded in the official MemOS scorer
(`other_repos/MemOS/evaluation/scripts/locomo/locomo_metric.py:40-45`):
    4 = single hop, 1 = multi hop, 2 = temporal reasoning, 3 = open domain.
Category 5 is the ADVERSARIAL / unanswerable split: those rows carry no usable evidence and are
EXCLUDED from the retrieval metrics (there is no gold turn to retrieve). They are counted and
reported, never silently dropped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ADVERSARIAL_CATEGORY",
    "CATEGORY_NAMES",
    "Conversation",
    "LabelledQuery",
    "Turn",
    "load_locomo",
    "normalize_text",
]

ADVERSARIAL_CATEGORY = 5

# `locomo_metric.py:40-45` — the official category mapping, copied verbatim (str keys there).
CATEGORY_NAMES: dict[int, str] = {
    4: "single hop",
    1: "multi hop",
    2: "temporal reasoning",
    3: "open domain",
    ADVERSARIAL_CATEGORY: "adversarial",
}

_SESSION_KEY = re.compile(r"^session_(\d+)$")


class Turn(BaseModel):
    """One dialogue turn — the unit of retrieval and the unit the gold ``evidence`` ids name."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dia_id: str
    session_index: int
    session_date: str
    speaker: str
    text: str

    @property
    def ingest_text(self) -> str:
        """What is written into the memory system.

        The speaker prefix is part of the fact ("Caroline went to a support group" vs "Melanie
        went ..."), and LoCoMo questions name the speakers, so dropping it would destroy
        answerability for a reason that has nothing to do with the ranker. The date is NOT
        prepended: MU stamps its own ``valid_at``/``created_at``, and injecting the dataset's
        date into the content would leak a lexical shortcut into every temporal-reasoning query.
        """
        return f"{self.speaker}: {self.text}"


class LabelledQuery(BaseModel):
    """One QA row = one query + its gold turn ids (the dataset's own relevance labels)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    query_id: str
    question: str
    answer: str
    evidence: tuple[str, ...]
    category: int

    @property
    def is_adversarial(self) -> bool:
        return self.category == ADVERSARIAL_CATEGORY


class Conversation(BaseModel):
    """One LoCoMo sample: the corpus (turns) plus the labelled queries over it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_id: str
    speaker_a: str
    speaker_b: str
    turns: tuple[Turn, ...]
    queries: tuple[LabelledQuery, ...] = Field(default_factory=tuple)

    def turn_by_dia_id(self) -> dict[str, Turn]:
        return {t.dia_id: t for t in self.turns}


def normalize_text(text: str) -> str:
    """Whitespace-collapsed, case-folded key used to map a recalled body back to its turn.

    The recall surface (`mu_contracts.contracts.recall.RecallItemView`) deliberately drops
    ``content_hash``, so the retrieved item cannot be joined back to its source turn by id alone
    across tiers (a promoted copy may carry a different tier-stable id than its STM original).
    Content text IS carried, and the ingest path stores it verbatim — so the join key is the
    normalized body. Verified rather than assumed: ``test_local_roundtrip_int.py`` asserts the
    original content round-trips through add → promote → recall unchanged.
    """
    return " ".join(text.split()).casefold()


def _parse_evidence(raw: object) -> tuple[str, ...]:
    """Gold turn ids. LoCoMo stores them as ``["D1:3", ...]``; a few rows carry a bare string.

    T3 (``TRACE-0923.md`` §7): at least one row (conv-26 qa[37]) carries a SINGLE evidence
    STRING naming TWO turn ids joined by ``"; "`` (``"D8:6; D9:17"``) rather than two separate
    list entries. Before this fix that joined string never matched a real ``dia_id`` (the ``;``
    is part of the id string), so ``gold = {e for e in query.evidence if e in known}`` in
    ``runner.py``/``answer_quality.py`` was always empty and the query was silently counted as
    "no gold in corpus" even though both evidence turns were ingested and indexed. Splitting on
    ``;`` here, once, at the loader, fixes every caller without duplicating the parsing.
    """
    if raw is None:
        return ()
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return ()
    out: list[str] = []
    for entry in raw:
        if not isinstance(entry, str | int):
            continue
        for part in str(entry).split(";"):
            part = part.strip()
            if part:
                out.append(part)
    return tuple(out)


def load_locomo(path: str | Path, *, samples: int | None = None) -> list[Conversation]:
    """Load LoCoMo conversations. ``samples`` caps how many of the 10 are returned (in file order).

    Raises ``FileNotFoundError`` rather than falling back to anything synthetic — a harness that
    quietly evaluates on a toy corpus is worse than one that refuses to run.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"LoCoMo not found at {data_path}. Provision it (HF snap-stanford/locomo, MIT) — "
            "this harness never falls back to a synthetic corpus."
        )
    raw = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{data_path} is not a LoCoMo sample list (got {type(raw).__name__})")

    out: list[Conversation] = []
    for sample in raw[: samples if samples is not None else len(raw)]:
        conv = sample["conversation"]
        session_indices = sorted(
            int(m.group(1)) for k in conv if (m := _SESSION_KEY.match(str(k))) is not None
        )
        turns: list[Turn] = []
        for idx in session_indices:
            date = str(conv.get(f"session_{idx}_date_time", ""))
            for turn in conv[f"session_{idx}"]:
                text = str(turn.get("text", "")).strip()
                if not text:
                    continue  # image-only turns carry no retrievable body
                turns.append(
                    Turn(
                        dia_id=str(turn["dia_id"]),
                        session_index=idx,
                        session_date=date,
                        speaker=str(turn["speaker"]),
                        text=text,
                    )
                )
        queries = tuple(
            LabelledQuery(
                query_id=f"{sample['sample_id']}::q{i}",
                question=str(qa["question"]),
                answer=str(qa.get("answer", qa.get("adversarial_answer", ""))),
                evidence=_parse_evidence(qa.get("evidence")),
                category=int(qa.get("category", 0)),
            )
            for i, qa in enumerate(sample.get("qa", []))
        )
        out.append(
            Conversation(
                sample_id=str(sample["sample_id"]),
                speaker_a=str(conv.get("speaker_a", "A")),
                speaker_b=str(conv.get("speaker_b", "B")),
                turns=tuple(turns),
                queries=queries,
            )
        )
    return out
