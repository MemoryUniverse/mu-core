"""Task.CONFLICT_ADJUDICATION catalog entry (ADR 0037; PROPOSED-CANONICAL-ADDITIONS-mlm.md P6).

S0-05 acceptance:
  1. `Task.CONFLICT_ADJUDICATION` exists as an enum member.
  2. It resolves to `ModelSettings.adjudicate_model` (NOT classify/rerank) via the SAME
     `TaskClassMapper` table every other `Task` member uses — no second mapping mechanism.
  3. It is excluded from `ModelCatalogSettings.local_capable_tasks` (HARD tier only; ADR 0037
     explicitly excludes the local SLM from this judgment).
"""

from __future__ import annotations

from mu_engine.providers.catalog import Task
from mu_engine.providers.settings import ModelCatalogSettings, ModelSettings
from mu_engine.providers.task_map import TaskClassMapper


def test_conflict_adjudication_is_a_task_member() -> None:
    assert Task.CONFLICT_ADJUDICATION in set(Task)
    assert Task.CONFLICT_ADJUDICATION.value == "conflict_adjudication"


def test_conflict_adjudication_resolves_to_adjudicate_model_not_classify_or_rerank() -> None:
    m = ModelSettings(
        adjudicate_model="ADJUDICATE-GROUP",
        classify_model="CLASSIFY-GROUP",
        rerank_model="RERANK-GROUP",
    )
    tm = TaskClassMapper(m)

    resolved = tm.group_for(Task.CONFLICT_ADJUDICATION)

    assert resolved == m.adjudicate_model == "ADJUDICATE-GROUP"
    assert resolved != m.classify_model
    assert resolved != m.rerank_model


def test_conflict_adjudication_shares_the_same_field_as_adjudicate() -> None:
    """Two distinct Task enum members (`CONFLICT_ADJUDICATION` != `ADJUDICATE` by construction,
    since StrEnum forbids duplicate values), but ONE model field — no new adjudicator field was
    introduced (ADR 0037)."""
    m = ModelSettings()
    tm = TaskClassMapper(m)

    assert tm.group_for(Task.CONFLICT_ADJUDICATION) == tm.group_for(Task.ADJUDICATE)


def test_conflict_adjudication_uses_the_one_task_groups_table() -> None:
    """It must appear in the SAME `task_groups()` table other Task members use — no second
    mapping mechanism was invented for this task."""
    tm = TaskClassMapper(ModelSettings())
    table = tm.task_groups()

    assert Task.CONFLICT_ADJUDICATION in table
    assert table[Task.CONFLICT_ADJUDICATION] == tm._m.adjudicate_model
    # every Task member is covered by the one table (guards against a parallel/shadow map)
    assert set(table) == set(Task)


def test_conflict_adjudication_precedence_still_honours_overrides() -> None:
    m = ModelSettings(model_override="GLOBAL")
    tm = TaskClassMapper(m)

    assert tm.group_for(Task.CONFLICT_ADJUDICATION, override="PERCALL") == "PERCALL"
    assert tm.group_for(Task.CONFLICT_ADJUDICATION) == "GLOBAL"


def test_conflict_adjudication_excluded_from_local_capable_tasks() -> None:
    """ADR 0037: the local SLM is explicitly excluded from conflict adjudication — HARD tier
    only. `local_capable_tasks` is the default local-SLM-eligible set (model-layer-spec §4)."""
    catalog = ModelCatalogSettings()

    assert Task.CONFLICT_ADJUDICATION not in catalog.local_capable_tasks
    # sanity: the default set is non-empty and does contain genuinely local-capable tasks
    assert Task.CLASSIFY in catalog.local_capable_tasks
    assert Task.EMBED in catalog.local_capable_tasks
