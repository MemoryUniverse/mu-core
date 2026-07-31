"""Lifecycle job/handle/result DTOs + ``UserPrefix`` — spec §17a field-level closure (S-2).

Authority: ``memory-lifecycle-manager-spec.md`` §5 (lines 222-254, the
``LifecycleWorkflowRunnerPort`` shape this DTO set backs) and §17/§17a (lines 625-664, "Field-level
DTO/port definitions (S-2 closure)" — every type named-but-not-field-defined in the port signature
gets a concrete shape here, grounded against real code where an EXISTS type applies);
``PROPOSED-CANONICAL-ADDITIONS-mlm.md`` P2 (lines 38-77, port half — this module is the DTO half);
``memory-lifecycle-manager-GAPSWEEP.md`` BLOCKER 1 (lines 17-45 — the port-name collision this
module's sibling port ``mu_contracts.ports.lifecycle_workflow`` resolves; this module is pure data
and carries none of that collision itself).

Every DTO below is a frozen, ``extra="forbid"`` pydantic v2 model — no bare-named-but-undefined
type remains from the port signature.

``UserPrefix`` is a validated **truncation** of ``Namespace.to_prefix()`` before the trailing
session segment (§17a) — ``mu/{org}/{workspace}/{visibility}/{user_slot}/`` (note the trailing
``/``; unlike ``to_prefix()``, which has no trailing separator and includes the session segment).
It is NOT a new physical key shape (CANONICAL §1 rule 5: ``to_prefix()``'s byte-shape is untouched
by this truncation) and carries no fields beyond the same four segments CANONICAL §1 already names
(org/workspace/visibility/user_slot). The sanctioned constructor path is ``UserPrefix(namespace)``
— derived from a live ``Namespace``, "never constructed independently" (§17a) by application code
from an arbitrary string. Round-tripping an already-serialized prefix string (Temporal history
replay / SQLite-WAL log rehydration, pydantic (de)serialization) is handled by the pydantic core
schema below, which re-validates the exact byte-shape before accepting it — it is not a second
public application-facing constructor.
"""

from __future__ import annotations

import re
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema

from mu_contracts.domain.model.memory import Namespace

__all__ = [
    "JobHandle",
    "JobResult",
    "JobStatus",
    "LifecycleJob",
    "LifecycleJobKind",
    "UserPrefix",
]

# mu/{org}/{workspace}/{private|shared}/{user_slot}/ — UserPrefix's validated byte-shape:
# Namespace.to_prefix() truncated before its trailing session segment (CANONICAL §1/§7.3),
# with the trailing "/" kept so the prefix reads as a directory-style lease grain (§4b).
_USER_PREFIX_PATTERN = re.compile(r"^mu/[^/]+/[^/]+/(?:private|shared)/[^/]+/$")


class UserPrefix(str):
    """The session-spanning lease-grain prefix ``mu/{org}/{workspace}/{visibility}/{user_slot}/``
    (spec §1/§4b/§17.8/§17a) — see module docstring for the full derivation contract."""

    __slots__ = ()

    def __new__(cls, namespace: Namespace) -> UserPrefix:
        text = namespace.to_prefix().rsplit("/", 1)[0] + "/"
        return cls._from_validated_str(text)

    @classmethod
    def _from_validated_str(cls, value: str) -> UserPrefix:
        """Internal rehydration path (pydantic (de)serialization only) — validates the exact
        byte-shape before accepting it. Not a sanctioned application-facing constructor; use
        ``UserPrefix(namespace)`` to derive one."""
        if not _USER_PREFIX_PATTERN.match(value):
            raise ValueError(f"not a valid UserPrefix byte-shape: {value!r}")
        return str.__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        def _validate(value: object) -> UserPrefix:
            if isinstance(value, UserPrefix):
                return value
            if isinstance(value, Namespace):
                return cls(value)
            if isinstance(value, str):
                return cls._from_validated_str(value)
            raise TypeError(f"UserPrefix cannot be built from {type(value)!r}")

        return core_schema.no_info_plain_validator_function(
            _validate,
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        # JSON-schema projection only — no-info plain-validator core schemas have no
        # default JSON-schema mapping, so pydantic's generator raises unless a type
        # supplies one explicitly (see module docstring: runtime validation/wire byte-shape
        # is untouched by this method).
        return {
            "type": "string",
            "format": "mu-user-prefix",
            "pattern": _USER_PREFIX_PATTERN.pattern,
            "examples": ["mu/acme/ws1/private/user-42/"],
        }


class LifecycleJobKind(StrEnum):
    """The four durable-write verbs §17's API submits through ``LifecycleWorkflowRunnerPort``."""

    SWEEP_USER = "sweep_user"
    PROMOTE = "promote"
    DEMOTE = "demote"
    CONSOLIDATE = "consolidate"


class JobStatus(StrEnum):
    """Terminal/non-terminal states of a submitted ``LifecycleJob`` (§17a)."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class LifecycleJob(BaseModel):
    """A durable-write job submitted to ``LifecycleWorkflowRunnerPort.submit`` (spec §5/§17a).

    ``job_id`` is the idempotency key — the Temporal ``workflow_id`` on the server runner, the
    SQLite-WAL row id on the client runner. ``namespace``/``memory_id`` are set for the
    single-memory verbs (``PROMOTE``/``DEMOTE``); ``after_offset`` is ``CONSOLIDATE``'s cursor
    (§13.1); all three are ``None`` for ``SWEEP_USER``. ``config_version``/``policy_version`` pin
    the ``LifecycleSettings``/``ManagerModeSettings`` (or ``RetentionSettings``) generation active
    at submit time (S6, §16/§20) so a replayed job re-applies the policy it was submitted under.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(min_length=1)
    kind: LifecycleJobKind
    user_prefix: UserPrefix  # the lease grain this job runs under (§4b)
    namespace: str | None = None  # to_prefix(); set for promote/demote (single memory), else None
    memory_id: str | None = None  # set for promote/demote, else None
    after_offset: int | None = None  # consolidate's cursor (§13.1); None for other kinds
    submitted_at: datetime  # Clock.now() (§19) at submit — not the decision instant
    config_version: str = Field(min_length=1)  # LifecycleSettings generation active at submit
    policy_version: str = Field(min_length=1)  # ManagerModeSettings/RetentionSettings generation


class JobHandle(BaseModel):
    """Returned fast by ``submit()`` — never carries the result (spec §5/§17a)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(min_length=1)
    submitted_at: datetime


class JobResult(BaseModel):
    """Returned by the optional ``await_result()`` — reads never call this (spec §5/§17a).

    ``error`` is content-free (SafeTraceFields discipline, no raw fact text) — a short
    machine-safe reason, never the memory content that triggered the failure.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(min_length=1)
    status: JobStatus
    error: str | None = None
    completed_at: datetime | None = None
