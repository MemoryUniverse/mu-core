"""LifecycleJob/JobHandle/JobResult/LifecycleJobKind/JobStatus/UserPrefix invariants (spec §17a)
+ ``LifecycleWorkflowRunnerPort`` distinctness from
``mu_contracts.ports.workflow.WorkflowRunnerPort`` (GAPSWEEP BLOCKER 1 /
PROPOSED-CANONICAL-ADDITIONS-mlm.md P2).

NOTE: ``mu_contracts.domain.model.lifecycle`` / ``mu_contracts.ports.lifecycle_workflow`` are not
yet re-exported from the package ``__init__.py`` files (out of this task's owned_paths — the
integrate phase must wire that export). Imports below go straight at the module.
"""

from __future__ import annotations

import inspect
import itertools
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from mu_contracts.domain.model.lifecycle import (
    JobHandle,
    JobResult,
    JobStatus,
    LifecycleJob,
    LifecycleJobKind,
    UserPrefix,
)
from mu_contracts.domain.model.memory import Namespace, Visibility
from mu_contracts.ports import workflow as generic_workflow_port
from mu_contracts.ports.lifecycle_workflow import LifecycleWorkflowRunnerPort

pytestmark = pytest.mark.unit


def _ns(
    *, org: str = "acme", workspace: str = "proj", user: str = "alice", session: str = "s1"
) -> Namespace:
    return Namespace(
        org=org, workspace=workspace, user=user, session=session, visibility=Visibility.PRIVATE
    )


def _shared_ns(*, org: str = "acme", workspace: str = "proj", session: str = "room1") -> Namespace:
    return Namespace(
        org=org, workspace=workspace, user="*", session=session, visibility=Visibility.SHARED
    )


# ---------------------------------------------------------------------------
# BLOCKER 1 fix: LifecycleWorkflowRunnerPort is a DISTINCT Protocol, never an alias.
# ---------------------------------------------------------------------------


def test_lifecycle_runner_port_is_not_the_generic_workflow_runner_port() -> None:
    lifecycle_cls = LifecycleWorkflowRunnerPort
    generic_cls = generic_workflow_port.WorkflowRunnerPort
    lifecycle_port: object = lifecycle_cls
    generic_port: object = generic_cls
    assert lifecycle_port is not generic_port
    same_name = lifecycle_cls.__name__ == generic_cls.__name__
    same_module = lifecycle_cls.__module__ == generic_cls.__module__
    assert not (same_name and same_module)
    # No module-level alias binding one name to the other object.
    import mu_contracts.ports.lifecycle_workflow as lifecycle_mod
    import mu_contracts.ports.workflow as workflow_mod

    assert getattr(lifecycle_mod, "WorkflowRunnerPort", None) is None
    assert getattr(workflow_mod, "LifecycleWorkflowRunnerPort", None) is None


def test_lifecycle_runner_port_shape_is_submit_await_resume_not_start_execute() -> None:
    lifecycle_members = inspect.getmembers(LifecycleWorkflowRunnerPort)
    lifecycle_methods = {name for name, _ in lifecycle_members if not name.startswith("_")}
    generic_methods = {
        name
        for name, _ in inspect.getmembers(generic_workflow_port.WorkflowRunnerPort)
        if not name.startswith("_")
    }
    assert lifecycle_methods == {"submit", "await_result", "resume_pending"}
    assert generic_methods == {"start", "execute", "readiness"}
    assert lifecycle_methods.isdisjoint(generic_methods)


# ---------------------------------------------------------------------------
# UserPrefix — derivation, validation, and the property-based truncation contract.
# ---------------------------------------------------------------------------


def test_user_prefix_is_truncation_of_to_prefix_before_session_segment() -> None:
    ns = _ns()
    prefix = UserPrefix(ns)
    assert prefix == ns.to_prefix().rsplit("/", 1)[0] + "/"
    assert prefix == "mu/acme/proj/private/alice/"
    assert isinstance(prefix, str)


def test_user_prefix_shared_zeroes_user_slot_same_as_to_prefix() -> None:
    ns = _shared_ns()
    prefix = UserPrefix(ns)
    assert prefix == ns.to_prefix().rsplit("/", 1)[0] + "/"
    assert prefix == "mu/acme/proj/shared/*/"


def _namespace_universe() -> list[Namespace]:
    """A property-style sweep over the Namespace input space (no hypothesis dependency in
    mu-contracts' dev group): every combination of org/workspace/session strings crossed with
    both visibilities, PRIVATE getting a distinct user id and SHARED forced to the "*" sentinel
    Namespace itself requires (CANONICAL §1 rule 4)."""
    orgs = ["acme", "o2", "org-3"]
    workspaces = ["proj", "ws2"]
    sessions = ["s1", "sess-2"]
    users = ["alice", "u2"]
    namespaces: list[Namespace] = []
    for org, workspace, session, user in itertools.product(orgs, workspaces, sessions, users):
        namespaces.append(
            Namespace(
                org=org,
                workspace=workspace,
                user=user,
                session=session,
                visibility=Visibility.PRIVATE,
            )
        )
        namespaces.append(
            Namespace(
                org=org,
                workspace=workspace,
                user="*",
                session=session,
                visibility=Visibility.SHARED,
            )
        )
    return namespaces


@pytest.mark.parametrize("ns", _namespace_universe())
def test_user_prefix_property_for_any_namespace(ns: Namespace) -> None:
    """Acceptance: for any Namespace, UserPrefix(ns) == ns.to_prefix().rsplit('/',1)[0] + '/'."""
    assert UserPrefix(ns) == ns.to_prefix().rsplit("/", 1)[0] + "/"


def test_user_prefix_rejects_malformed_string_on_pydantic_validation() -> None:
    with pytest.raises(ValidationError):
        LifecycleJob(
            job_id="j1",
            kind=LifecycleJobKind.SWEEP_USER,
            user_prefix="not-a-valid-prefix",  # type: ignore[arg-type]
            submitted_at=datetime.now(UTC),
            config_version="v1",
            policy_version="v1",
        )


def test_user_prefix_round_trips_through_model_dump_json() -> None:
    job = LifecycleJob(
        job_id="j1",
        kind=LifecycleJobKind.SWEEP_USER,
        user_prefix=UserPrefix(_ns()),
        submitted_at=datetime.now(UTC),
        config_version="v1",
        policy_version="v1",
    )
    replayed = LifecycleJob.model_validate_json(job.model_dump_json())
    assert replayed == job
    assert isinstance(replayed.user_prefix, UserPrefix)


# ---------------------------------------------------------------------------
# LifecycleJob / JobHandle / JobResult — field-complete, frozen, extra="forbid".
# ---------------------------------------------------------------------------


def test_lifecycle_job_all_fields_and_defaults() -> None:
    job = LifecycleJob(
        job_id="j1",
        kind=LifecycleJobKind.PROMOTE,
        user_prefix=UserPrefix(_ns()),
        namespace=_ns().to_prefix(),
        memory_id="m1",
        submitted_at=datetime.now(UTC),
        config_version="v1",
        policy_version="v1",
    )
    assert job.after_offset is None
    assert job.kind is LifecycleJobKind.PROMOTE


def test_lifecycle_job_frozen_and_forbids_extra() -> None:
    job = LifecycleJob(
        job_id="j1",
        kind=LifecycleJobKind.SWEEP_USER,
        user_prefix=UserPrefix(_ns()),
        submitted_at=datetime.now(UTC),
        config_version="v1",
        policy_version="v1",
    )
    with pytest.raises(ValidationError):
        job.job_id = "other"
    with pytest.raises(ValidationError):
        LifecycleJob(
            job_id="j1",
            kind=LifecycleJobKind.SWEEP_USER,
            user_prefix=UserPrefix(_ns()),
            submitted_at=datetime.now(UTC),
            config_version="v1",
            policy_version="v1",
            extra_field="nope",  # type: ignore[call-arg]
        )


def test_lifecycle_job_kind_has_all_four_verbs() -> None:
    assert {k.value for k in LifecycleJobKind} == {"sweep_user", "promote", "demote", "consolidate"}


def test_job_status_has_all_four_states() -> None:
    assert {s.value for s in JobStatus} == {"pending", "running", "succeeded", "failed"}


def test_job_handle_field_complete_frozen_forbids_extra() -> None:
    now = datetime.now(UTC)
    handle = JobHandle(job_id="j1", submitted_at=now)
    assert handle.job_id == "j1"
    assert handle.submitted_at == now
    with pytest.raises(ValidationError):
        handle.job_id = "other"
    with pytest.raises(ValidationError):
        JobHandle(job_id="j1", submitted_at=now, extra="nope")  # type: ignore[call-arg]


def test_job_result_field_complete_frozen_forbids_extra_and_content_free_error() -> None:
    result = JobResult(job_id="j1", status=JobStatus.FAILED, error="engine_unavailable")
    assert result.completed_at is None
    with pytest.raises(ValidationError):
        result.status = JobStatus.SUCCEEDED
    with pytest.raises(ValidationError):
        JobResult(job_id="j1", status=JobStatus.SUCCEEDED, bogus=1)  # type: ignore[call-arg]


def test_job_result_success_has_completed_at() -> None:
    now = datetime.now(UTC)
    result = JobResult(job_id="j1", status=JobStatus.SUCCEEDED, completed_at=now)
    assert result.error is None
    assert result.completed_at == now
