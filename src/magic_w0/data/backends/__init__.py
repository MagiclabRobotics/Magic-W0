# Copyright 2026 MagicVLA authors. (Ported from MagicMix, Apache-2.0.)
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Data backend selection by source ``type``.

MagicMix keyed backends by lerobot version; standalone MagicW0 keys them by
source ``type`` so heterogeneous sources (a spec-driven ``synthetic`` source,
a ``lerobot`` source that dispatches on per-source ``lerobot_version`` to our
standalone v2.1 / v3.0 readers (no lerobot package import) -- and, once wired,
``vlm_jsonl``) can co-exist in one mix. Register new backends with
:func:`register_backend`.

Backends here never import :mod:`models`: a source's schema is a contract with
the model's processor/``forward``, but it is expressed as *config* (see
:mod:`data.backends.synthetic`), so the data layer stays model-agnostic.
"""

from __future__ import annotations

from magic_w0.data.backends.base import DataBackend
from magic_w0.data.backends.lerobot import LeRobotBackend
from magic_w0.data.backends.synthetic import SyntheticBackend
from magic_w0.data.backends.vlm_jsonl import VlmJsonlBackend

_BACKENDS: dict[str, type[DataBackend]] = {
    "synthetic": SyntheticBackend,
    "lerobot": LeRobotBackend,
    "vlm_jsonl": VlmJsonlBackend,
}


def register_backend(source_type: str, backend_cls: type[DataBackend]) -> None:
    """Register a :class:`DataBackend` under a source ``type`` name."""
    _BACKENDS[source_type] = backend_cls


def get_data_backend(source_type: str = "synthetic") -> DataBackend:
    """Return the backend instance matching a source ``type``."""
    if source_type not in _BACKENDS:
        raise NotImplementedError(
            f"No data backend registered for source type {source_type!r}. "
            f"Available: {sorted(_BACKENDS)}. Register one with "
            "magic_w0.data.backends.register_backend(...) when adapting a new source."
        )
    return _BACKENDS[source_type]()


__all__ = [
    "DataBackend",
    "SyntheticBackend",
    "VlmJsonlBackend",
    "get_data_backend",
    "register_backend",
]
