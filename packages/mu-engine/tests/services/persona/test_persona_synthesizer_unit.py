"""Stage 2 — ``memorybank_rollup_v1`` (``persona-design.md`` §2.3, lines 109-115).

Zero infra: the model seam is the narrow :class:`PersonaSynthesisPort`, so the two-level roll-up
is testable without a router, a provider or a key — which is also the property that keeps FULL
LOCAL honest (spec line 112 routes through ``Task.SUMMARIZE``, never a bespoke persona model).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime

import pytest

from mu_contracts.domain.model.persona import PersonaSlot, SlotValue
from mu_engine.providers._contracts import MessageRole
from mu_engine.providers.catalog import Task
from mu_engine.services.persona.settings import MEMORYBANK_ROLLUP_V1, PersonaSettings
from mu_engine.services.persona.synthesizer import (
    MemoryBankRollupV1Synthesizer,
    persona_synthesizer_registry,
)

from .conftest import T0, StubRouter

pytestmark = pytest.mark.unit

_LEVEL1 = json.dumps({"brief": "Terse data engineer.", "strategy": "Answer in one paragraph."})
_LEVEL2 = "Terse data engineer; answer in one short paragraph."


def _slot(value: str, *, confidence: float = 0.9, updated_at: datetime = T0) -> SlotValue:
    return SlotValue(
        value=value, confidence=confidence, support_ids=("mem_1",), updated_at=updated_at
    )


@pytest.fixture
def slots() -> dict[PersonaSlot, SlotValue]:
    return {
        PersonaSlot.OCCUPATION: _slot("data engineer"),
        PersonaSlot.RESPONSE_STYLE: _slot("terse", confidence=0.7),
    }


@pytest.fixture
def build() -> Callable[..., MemoryBankRollupV1Synthesizer]:
    def _build(
        replies: list[str], settings: PersonaSettings | None = None
    ) -> MemoryBankRollupV1Synthesizer:
        return MemoryBankRollupV1Synthesizer(
            router=StubRouter(replies), settings=settings or PersonaSettings()
        )

    return _build


# ----------------------------------------------------------------------- two-level roll-up
async def test_two_level_rollup_returns_overall_brief_and_strategy(
    build: Callable[..., MemoryBankRollupV1Synthesizer], slots: dict[PersonaSlot, SlotValue]
):
    """Level 1 (``summarize_person_prompt``) then level 2 (``summarize_overall_personality``) —
    the ported MemoryBank shape, ``summarize_memory.py:94`` then ``:87``, stored as ``:142``."""
    synth = build([_LEVEL1, _LEVEL2])
    overall, strategy = await synth.synthesize(slots)
    assert overall == _LEVEL2
    assert strategy == "Answer in one paragraph."
    assert len(synth._router.calls) == 2  # type: ignore[attr-defined]  # two levels, not one


async def test_every_call_routes_through_task_summarize(
    build: Callable[..., MemoryBankRollupV1Synthesizer], slots: dict[PersonaSlot, SlotValue]
):
    """Spec line 112 / CANONICAL §7.2: ``models.summarize_model``, never a bespoke persona model.
    ``Task.SUMMARIZE`` is what makes ``TaskClassMapper`` resolve that field, so asserting the Task
    IS asserting the model choice."""
    synth = build([_LEVEL1, _LEVEL2])
    await synth.synthesize(slots)
    settings = PersonaSettings()
    for task, messages, max_tokens, temperature, _fmt in synth._router.calls:  # type: ignore[attr-defined]
        assert task is Task.SUMMARIZE
        assert max_tokens == settings.synthesis_max_tokens
        assert temperature == settings.synthesis_temperature
        assert [m.role for m in messages] == [MessageRole.SYSTEM, MessageRole.USER]


async def test_level1_prompt_is_byte_stable_under_slot_insertion_order(
    build: Callable[..., MemoryBankRollupV1Synthesizer],
):
    """The prompt is sorted by slot key, so the same persona never produces two different prompts
    (and two different portraits) because a dict happened to be built in another order."""
    a = {PersonaSlot.OCCUPATION: _slot("data engineer"), PersonaSlot.HOBBY: _slot("climbing")}
    b = {PersonaSlot.HOBBY: _slot("climbing"), PersonaSlot.OCCUPATION: _slot("data engineer")}
    first, second = build([_LEVEL1, _LEVEL2]), build([_LEVEL1, _LEVEL2])
    await first.synthesize(a)
    await second.synthesize(b)
    assert first._router.calls[0][1][1].content == second._router.calls[0][1][1].content  # type: ignore[attr-defined]


# ---------------------------------------------------------------------- letta char cap (§2.3)
async def test_brief_is_capped_at_brief_char_limit(
    build: Callable[..., MemoryBankRollupV1Synthesizer], slots: dict[PersonaSlot, SlotValue]
):
    """letta's ``Block.limit`` ergonomic (``block.py:141``): the stored brief ALWAYS fits the
    system-prompt budget, whatever the model returned."""
    synth = build([_LEVEL1, "word " * 4000], PersonaSettings(brief_char_limit=80))
    overall, _ = await synth.synthesize(slots)
    assert len(overall) <= 80
    assert overall.endswith("…")  # truncation is VISIBLE, never a silent mid-sentence stop


async def test_a_short_brief_is_returned_untouched(
    build: Callable[..., MemoryBankRollupV1Synthesizer], slots: dict[PersonaSlot, SlotValue]
):
    synth = build([_LEVEL1, _LEVEL2], PersonaSettings(brief_char_limit=4000))
    overall, _ = await synth.synthesize(slots)
    assert overall == _LEVEL2


# ------------------------------------------------------------------------- fail-loud parsing
@pytest.mark.parametrize(
    "reply",
    [
        "not json at all",
        json.dumps(["brief", "strategy"]),
        json.dumps({"brief": "ok"}),
        json.dumps({"brief": "", "strategy": "s"}),
        json.dumps({"brief": "b", "strategy": 7}),
    ],
)
async def test_a_malformed_level1_reply_raises(
    build: Callable[..., MemoryBankRollupV1Synthesizer],
    slots: dict[PersonaSlot, SlotValue],
    reply: str,
):
    """Never salvage half a portrait (DEV-STANDARDS rule 8). ``PersonaService`` owns the NAMED
    degrade; the strategy itself refuses."""
    with pytest.raises(ValueError):
        await build([reply, _LEVEL2]).synthesize(slots)


async def test_an_empty_level2_reply_raises(
    build: Callable[..., MemoryBankRollupV1Synthesizer], slots: dict[PersonaSlot, SlotValue]
):
    with pytest.raises(ValueError):
        await build([_LEVEL1, "   "]).synthesize(slots)


async def test_an_empty_slot_table_is_refused_before_any_model_call(
    build: Callable[..., MemoryBankRollupV1Synthesizer],
):
    """No slots means no portrait to draw — refuse rather than spend a token asking a model to
    invent one."""
    synth = build([_LEVEL1, _LEVEL2])
    with pytest.raises(ValueError):
        await synth.synthesize({})
    assert synth._router.calls == []  # type: ignore[attr-defined]


# --------------------------------------------------------------------------- registry (§6)
def test_synthesizer_registry_registers_no_router_less_default():
    """Deliberate: ``Registry``'s ``Callable[[Settings], T]`` factory cannot supply a router, and a
    factory that fabricated one would open the new provider path spec line 112 forbids."""
    from mu_contracts.domain.errors import UnknownComponentError

    assert persona_synthesizer_registry.names() == ()
    with pytest.raises(UnknownComponentError):
        persona_synthesizer_registry.create(MEMORYBANK_ROLLUP_V1, object())  # type: ignore[arg-type]
