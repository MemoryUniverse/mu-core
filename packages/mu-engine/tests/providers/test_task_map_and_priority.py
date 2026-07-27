"""L3 task→group resolution + L4 local-priority predicate (model-layer-spec §8 tests 2,3)."""

from __future__ import annotations

import pytest

from mu_engine.providers.catalog import Task
from mu_engine.providers.local_priority import LOCAL_ORDER, REMOTE_ORDER, LocalPriorityPolicy
from mu_engine.providers.settings import ModelSettings
from mu_engine.providers.task_map import TaskClassMapper

pytestmark = pytest.mark.unit


def test_task_maps_to_its_field() -> None:
    m = ModelSettings()
    tm = TaskClassMapper(m)
    assert tm.group_for(Task.ANSWER) == m.answer_model
    assert tm.group_for(Task.HARD_EXTRACT) == m.hard_extract_model
    assert tm.group_for(Task.CLASSIFY) == m.classify_model
    assert tm.group_for(Task.EMBED) == m.embed_model


def test_precedence_percall_override_beats_global_beats_field() -> None:
    m = ModelSettings(model_override="GLOBAL")
    tm = TaskClassMapper(m)
    # per-call override wins over global override
    assert tm.group_for(Task.ANSWER, override="PERCALL") == "PERCALL"
    # global override wins over the task field
    assert tm.group_for(Task.ANSWER) == "GLOBAL"


def test_unmapped_task_falls_back_to_default_class_never_none() -> None:
    m = ModelSettings()
    tm = TaskClassMapper(m, default_task=Task.ROUTINE_EXTRACT)
    # task_groups always cover every Task; force a hole to prove the default path returns a value
    tm._m = ModelSettings(routine_extract_model="DEFCLASS")
    got = tm._group_for_default()
    assert got == "DEFCLASS"
    assert got is not None


def test_task_groups_table_covers_all_tasks() -> None:
    tm = TaskClassMapper(ModelSettings())
    assert set(tm.task_groups()) == set(Task)


def test_local_priority_predicate() -> None:
    pol = LocalPriorityPolicy(
        local_capable_tasks=frozenset({Task.CLASSIFY, Task.EMBED}), enabled=True
    )
    assert pol.prefers_local(Task.CLASSIFY) is True
    assert pol.prefers_local(Task.ANSWER) is False


def test_local_priority_order_for() -> None:
    pol = LocalPriorityPolicy(local_capable_tasks=frozenset({Task.CLASSIFY}), enabled=True)
    assert pol.order_for(is_local=True, task=Task.CLASSIFY) == LOCAL_ORDER
    assert pol.order_for(is_local=False, task=Task.CLASSIFY) == REMOTE_ORDER
    assert pol.order_for(is_local=True, task=Task.ANSWER) == REMOTE_ORDER  # not local-capable


def test_local_priority_disabled_never_prefers_local() -> None:
    pol = LocalPriorityPolicy(local_capable_tasks=frozenset({Task.CLASSIFY}), enabled=False)
    assert pol.prefers_local(Task.CLASSIFY) is False
    assert pol.order_for(is_local=True, task=Task.CLASSIFY) == REMOTE_ORDER
