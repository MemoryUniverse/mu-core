"""Stage 2 — ``memorybank_rollup_v1``, the LLM portrait synthesizer (``persona-design.md`` §2.3).

Authority: spec §2.3 (lines 109-115) + §2.4 line 122. **SLEEP-TIME ONLY.** The single caller is
:meth:`~mu_engine.services.persona.service.PersonaService.rebuild`; nothing on the recall/ingest
hot path and nothing at warm/preload time may reach it (preload §4.6: *"Persona inference is a GPT
task and is not done at warm time"*). ``PersonaService.on_promoted`` — the §2.4 line 121
incremental path — does not hold a reference to it at all, which is what makes "no LLM on the
incremental path" structural rather than a comment.

PORT-FROM (CODE-ADOPTION-METHODOLOGY step 3), verified line-by-line against the actual vendored
source ``/home/user/D/abstract_project/mma/other_repos/MemoryBank/memory_bank/summarize_memory.py``
(MIT, © 2023 Wanjun Zhong; paper arXiv:2305.10250):

* ``summarize_memory.py:94`` ``def summarize_person_prompt(content, user_name, boot_name,
  language)`` — LEVEL 1. Its closing instruction (``:104``) asks for exactly three things:
  *"{user_name}'s personality traits, emotions, and {boot_name}'s response strategy"*. That is
  where spec line 112's "brief **plus a response-strategy line**" comes from.
* ``summarize_memory.py:137`` ``memory[user_name]['personality'][date] = person_summary`` — level
  1's output is stored per period.
* ``summarize_memory.py:87`` ``def summarize_overall_personality(content, language)`` — LEVEL 2,
  which folds the per-period analyses. Its instruction (``:90-91``) asks for a *"highly concise
  and general summary of the user's personality and the most appropriate response strategy"*.
* ``summarize_memory.py:142`` ``memory[user_name]['overall_personality'] = …`` — level 2's output
  is the single stored string, which is what SiliconFriend prepends at generation time. Ours is
  :attr:`~mu_contracts.domain.model.persona.PersonaProfile.overall_brief`.

DELIBERATE DEVIATIONS (methodology step 4):

1. **Level 1 compresses SLOTS, not raw dialogue.** Upstream summarises transcripts; we already
   ran the deterministic Stage 1, so level 1 is handed the structured slot table. This is the
   spec's own "deterministic-before-LLM" ordering (§2, line 74) and it means no raw conversation
   is ever put in the prompt.
2. **Char-capped, letta-style.** ``PersonaSettings.brief_char_limit`` is enforced on the stored
   brief (letta ``Block``'s ``limit`` ergonomic — ``OR/letta/letta/schemas/block.py:141``
   ``BlockUpdate.limit``, on the ``Persona`` block at ``:128``/``:131``), so the brief always fits
   a system-prompt budget. Upstream has no cap. Enforcement is a truncation at a word boundary
   rather than letta's rejection: this runs on a background worker where refusing the write would
   simply leave the user with a stale portrait.
3. **No language switch.** Upstream branches on ``language=='cn'``; the model is instructed in the
   prompt instead, and the user's own ``LANGUAGE``/``LANGUAGE_STYLE`` slots are in the slot table.

MODEL ROUTE (spec line 112, CANONICAL §7.2): ``Task.SUMMARIZE`` only
(``mu_engine/providers/catalog.py:41``), which ``TaskClassMapper`` resolves to
``ModelSettings.summarize_model`` (``providers/task_map.py:35``) → model group ``mu-summarize``
(``providers/shipped_catalog.py:134``, local-first, so FULL-LOCAL works with zero API keys).
Never a bespoke persona model, and never the HARD tier (which by ADR 0037 has no local row).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from mu_contracts.domain.model.persona import PersonaSlot, SlotValue
from mu_engine.platform.registry import Registry
from mu_engine.providers._contracts import Completion, Message, MessageRole
from mu_engine.providers.catalog import Task
from mu_engine.services.persona.settings import MEMORYBANK_ROLLUP_V1, PersonaSettings

__all__ = [
    "MemoryBankRollupV1Synthesizer",
    "PersonaSynthesisPort",
    "PortraitSynthesizer",
    "persona_synthesizer_registry",
]


@runtime_checkable
class PersonaSynthesisPort(Protocol):
    """The narrow slice of ``ModelRouter`` this synthesizer depends on (DEV-STANDARDS rule 6;
    the same shape ``ConflictAdjudicationPort`` declares at ``lifecycle/conflict.py:251-268``).
    Structurally satisfied by ``mu_engine.providers.model_router.ModelRouter.generate`` — so
    persona opens NO new provider path and adds no ``ModelSettings`` field."""

    async def generate(
        self,
        task: Task,
        messages: list[Message],
        *,
        override: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        response_format: str | None = None,
    ) -> Completion: ...


@runtime_checkable
class PortraitSynthesizer(Protocol):
    """Spec §6 line 217: ``synthesize(slots, *, models) -> tuple[str, str]`` (brief, strategy).

    **SIGNATURE DELTA (recorded, not silently made):** the ``models: ModelSettings`` parameter is
    dropped. Task→model resolution lives in ``TaskClassMapper.group_for``
    (``providers/task_map.py:44-55``) INSIDE the router, so a ``ModelSettings`` handed to the
    strategy would be either unused or a second, divergent resolution path — and spec line 112's
    requirement ("``summarize_model``, never a bespoke persona model") is exactly what routing by
    ``Task`` guarantees. The model seam is :class:`PersonaSynthesisPort`, injected.
    """

    async def synthesize(self, slots: Mapping[PersonaSlot, SlotValue]) -> tuple[str, str]: ...


_LEVEL1_SYSTEM = (
    "You build a compact user portrait from a structured trait table. "
    "Return ONLY a JSON object with exactly two string keys: "
    '"brief" (the user\'s personality traits and stable preferences) and '
    '"strategy" (one sentence: how an assistant should speak to this user). '
    "Write in the user's own language if the table names one. "
    "Never invent a trait the table does not support."
)

_LEVEL2_SYSTEM = (
    "Fold the portrait and the response strategy below into ONE highly concise paragraph "
    "covering the user's personality and the most appropriate response strategy. "
    "Return the paragraph as plain text, nothing else. "
    "Never invent a trait the input does not support."
)

_LEVEL1_KEY_BRIEF = "brief"
_LEVEL1_KEY_STRATEGY = "strategy"

#: Appended when the char cap truncates the brief, so a downstream reader can SEE that the
#: portrait was cut rather than silently reading a sentence that stops mid-thought.
_TRUNCATION_MARK = "…"


class MemoryBankRollupV1Synthesizer:
    """The default two-level portrait roll-up (spec §2.3)."""

    key = MEMORYBANK_ROLLUP_V1

    def __init__(self, *, router: PersonaSynthesisPort, settings: PersonaSettings) -> None:
        self._router = router
        self._settings = settings

    async def synthesize(self, slots: Mapping[PersonaSlot, SlotValue]) -> tuple[str, str]:
        """``(overall_brief, strategy)``.

        Raises on an empty slot table or a malformed model reply — fail-loud, never a guessed
        portrait (DEV-STANDARDS rule 8). ``PersonaService`` owns the NAMED degrade that turns such
        a raise into "structured slots, previous brief carried forward".

        **Where the second element goes — recorded spec gap.** ``PersonaProfile`` (§3.1 lines
        132-139, and the shipped class) has exactly ONE string field, ``overall_brief``, so the
        spec never says where the response-strategy line is persisted. It is folded into the
        level-2 prompt (upstream does the same: ``summarize_memory.py:90-91`` asks level 2 for
        personality AND response strategy, and ``:142`` stores the one result), so nothing is
        lost; the separate line is still returned because spec line 217 pins the tuple and the
        §5.1 ``PersonaAdapter`` wants the two apart. Either ``PersonaProfile`` gains a second
        field or §2.3's return type collapses to one string — an owner call, flagged.
        """
        if not slots:
            raise ValueError("PortraitSynthesizer.synthesize requires a non-empty slot table")
        brief, strategy = await self._level1(slots)
        overall = await self._level2(brief, strategy)
        return self._cap(overall), strategy

    async def _level1(self, slots: Mapping[PersonaSlot, SlotValue]) -> tuple[str, str]:
        """Slots -> (portrait, response strategy) — ``summarize_person_prompt`` (``:94``)."""
        completion = await self._generate(_LEVEL1_SYSTEM, _render_slots(slots), json_reply=True)
        return _parse_level1(completion.text)

    async def _level2(self, brief: str, strategy: str) -> str:
        """(portrait, strategy) -> ``overall_brief`` — ``summarize_overall_personality`` (``:87``),
        stored as ``overall_personality`` (``:142``)."""
        completion = await self._generate(
            _LEVEL2_SYSTEM, f"Portrait:\n{brief}\n\nResponse strategy:\n{strategy}"
        )
        text = completion.text.strip()
        if not text:
            raise ValueError("portrait roll-up returned an empty brief")
        return text

    async def _generate(self, system: str, user: str, *, json_reply: bool = False) -> Completion:
        return await self._router.generate(
            Task.SUMMARIZE,
            [
                Message(role=MessageRole.SYSTEM, content=system),
                Message(role=MessageRole.USER, content=user),
            ],
            max_tokens=self._settings.synthesis_max_tokens,
            temperature=self._settings.synthesis_temperature,
            response_format="json_object" if json_reply else None,
        )

    def _cap(self, brief: str) -> str:
        """letta's ``Block.limit`` char cap (``block.py:141``), applied at a word boundary."""
        limit = self._settings.brief_char_limit
        if len(brief) <= limit:
            return brief
        head = brief[: limit - len(_TRUNCATION_MARK)]
        cut = head.rsplit(" ", 1)[0] if " " in head else head
        return cut + _TRUNCATION_MARK


def _render_slots(slots: Mapping[PersonaSlot, SlotValue]) -> str:
    """The slot table as prompt text. Sorted by slot key so the prompt — and therefore a cached
    completion — is byte-stable for the same slots regardless of dict insertion order."""
    return "\n".join(
        f"{slot.value}: {slots[slot].value} (confidence {slots[slot].confidence:.2f})"
        for slot in sorted(slots, key=lambda s: s.value)
    )


def _parse_level1(text: str) -> tuple[str, str]:
    """Strict parse of level 1's JSON reply. A malformed reply RAISES — the caller degrades to
    "slots only, previous brief", it never salvages half a portrait."""
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("portrait level-1 reply was not JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("portrait level-1 reply was not a JSON object")
    brief = payload.get(_LEVEL1_KEY_BRIEF)
    strategy = payload.get(_LEVEL1_KEY_STRATEGY)
    if not isinstance(brief, str) or not isinstance(strategy, str):
        raise ValueError("portrait level-1 reply lacked string 'brief'/'strategy'")
    if not brief.strip() or not strategy.strip():
        raise ValueError("portrait level-1 reply carried an empty 'brief'/'strategy'")
    return brief.strip(), strategy.strip()


#: The Stage-2 substitution seam (spec lines 243, 245). Same fail-loud ``Registry`` and the same
#: ``AdapterRegistry`` naming delta recorded on ``persona_aggregator_registry``.
#:
#: Unlike the aggregator, this strategy needs a ROUTER, which ``Registry``'s
#: ``Callable[[Settings], T]`` factory signature cannot supply — so no default factory is
#: registered here. That is deliberate: a factory that fabricated a router would open the "new
#: provider path" spec line 112 forbids, and one that registered a router-less synthesizer would
#: be a silent stub. The composition root registers the built strategy (``register_factory``) or,
#: as everywhere else in this repo, injects it into ``PersonaService`` directly.
persona_synthesizer_registry: Registry[PortraitSynthesizer] = Registry("persona_synthesizer")
