"""ADR 0070 — the containerized ``mode="local_server"`` product could not store a memory.

**What happened (MEASURED, on a real ``make up`` stack on ``mu-dev-vm``, 2026-09-24).** Every
write to `mu-engine-server` returned HTTP 500::

    StageExecutionError: stage 'persist_raw_artifact' failed:
      [Errno 30] Read-only file system:
      '/home/mu-engine-server/.memory-universe/artifacts/mu/default/local/private/f3-auth-user'

`PersistRawArtifactStage` is a MANDATORY ingest stage. `ArtifactFsSettings.content_root`'s default
moved that day from the CWD-relative ``"./.mu_data/artifacts"`` to the deterministic, user-scoped
``"~/.memory-universe/artifacts"`` (FAULT-HUNT-0924.md F4c — ten scattered plaintext blob trees).
That default is correct on a laptop and lands, in THIS container, inside the ``:ro`` bind mount
``docker-compose.yml`` has carried since 2026-08-27 for the bearer token ("the container only ever
verifies, never mints" — a posture that is right, and not the thing to change). Two separately
sound decisions, composed, produced a server that 500s on every write.

**Why this test and not only the acceptance suite.** Stage F caught it — its four WRITING criteria
(F1, F2a, F2b, and F3's one 201) all failed while its never-writing auth/404 criteria passed — but
Stage F needs an operator to ``make up`` a real stack first, so it cannot gate a laptop or a CI
lane that has no docker. This test reads the two shipped deployment artifacts as TEXT and asserts
the composition that failed cannot come back, in milliseconds, anywhere.

**MUTATION CHECK (run, red):** delete the ``MU_STORAGE__ARTIFACT__CONTENT_ROOT`` line from
``docker-compose.yml`` and ``test_the_container_artifact_root_is_not_inside_a_read_only_mount``
fails, naming the ro mount the default falls into; delete the ``mkdir -p
/var/lib/mu-engine-server/artifacts`` line from the ``Dockerfile`` and
``test_the_artifact_root_exists_in_the_image_so_a_fresh_volume_inherits_ownership`` fails.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from mu_contracts.config.settings import ArtifactFsSettings

_PACKAGE_DIR: Final = Path(__file__).resolve().parents[1]
_COMPOSE: Final = _PACKAGE_DIR / "docker-compose.yml"
_DOCKERFILE: Final = _PACKAGE_DIR / "Dockerfile"

#: The container's HOME (``Dockerfile``'s own ``ENV HOME=``) — what ``~`` expands to in there,
#: which is NOT what it expands to on the machine running this test. Read from the Dockerfile
#: rather than hardcoded, so a change to either file keeps this check honest.
_HOME_RE: Final = re.compile(r"^\s*HOME=(\S+)", re.MULTILINE)
#: Read-only bind/volume mounts: a compose ``- <src>:<dst>:ro`` entry.
_RO_MOUNT_RE: Final = re.compile(r"^\s*-\s*\S+?:(/\S+?):ro\s*$", re.MULTILINE)
#: The central-Settings override the server's artifact factory actually reads.
_CONTENT_ROOT_RE: Final = re.compile(
    r"^\s*MU_STORAGE__ARTIFACT__CONTENT_ROOT:\s*(\S+)", re.MULTILINE
)


def _container_home() -> str:
    match = _HOME_RE.search(_DOCKERFILE.read_text())
    assert match is not None, "Dockerfile no longer sets HOME — this test's ~ expansion is blind"
    return match.group(1)


def _effective_content_root() -> str:
    """What ``_build_artifact_fs`` will resolve INSIDE the container: the compose override if
    present, else the central default — with ``~`` expanded to the CONTAINER's HOME, never this
    machine's."""
    compose = _COMPOSE.read_text()
    override = _CONTENT_ROOT_RE.search(compose)
    raw = override.group(1) if override else ArtifactFsSettings().content_root
    return raw.replace("~", _container_home(), 1) if raw.startswith("~") else raw


def test_the_container_artifact_root_is_not_inside_a_read_only_mount() -> None:
    root = _effective_content_root()
    ro_targets = _RO_MOUNT_RE.findall(_COMPOSE.read_text())
    assert ro_targets, "no :ro mount found — the regex stopped matching the compose file's shape"
    offenders = [t for t in ro_targets if root == t or root.startswith(t.rstrip("/") + "/")]
    assert not offenders, (
        f"the artifact blob root the container will resolve ({root!r}) is inside the READ-ONLY "
        f"mount(s) {offenders!r}. `PersistRawArtifactStage` is a mandatory ingest stage, so every "
        f"write to this server returns HTTP 500 with `[Errno 30] Read-only file system` (ADR 0070, "
        f"measured on a real `make up` stack). Give the artifact root its own WRITABLE volume and "
        f"point MU_STORAGE__ARTIFACT__CONTENT_ROOT at it — do NOT make the token mount writable."
    )


def test_the_artifact_root_exists_in_the_image_so_a_fresh_volume_inherits_ownership() -> None:
    """Docker seeds a fresh named volume from the image's content AND ownership at the mount
    point — but only when that path exists in the image. Mount one at a path the image lacks and
    docker creates it ``root:root``, which the ``USER mu-engine-server`` process cannot write: the
    same 500, one errno over (13, not 30)."""
    root = _effective_content_root()
    dockerfile = _DOCKERFILE.read_text()
    assert f"mkdir -p {root}" in dockerfile, (
        f"{root!r} is mounted as a named volume but is never created in the Dockerfile, so a fresh "
        f"volume is seeded root:root and the app user gets `[Errno 13] Permission denied` on the "
        f"first ingest"
    )
    chown_block = dockerfile.split("RUN chown -R mu-engine-server:mu-engine-server", 1)
    assert len(chown_block) == 2, "the Dockerfile's chown line moved — this check is blind"
    assert (
        "/var/lib/mu-engine-server" in chown_block[1].split("\n\n", 1)[0]
    ), "the artifact root's parent is not chowned to the app user"
