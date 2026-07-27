"""Layer-0 infra adapters. Importing this package runs the registration decorators (spec §7):
the ``inproc`` bus and the ``inline`` workflow runner become available on their registries."""

from __future__ import annotations

from mu_engine.platform.adapters import bus_inproc, workflow_inline

__all__ = ["bus_inproc", "workflow_inline"]
