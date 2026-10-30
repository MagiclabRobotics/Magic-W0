# Copyright 2026 MagicVLA authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Spec-driven synthetic backend -- random tensors in any schema, from config.

What a source yields is a contract with the *policy's* ``forward``, so a
synthetic source must be able to speak whatever schema the policy under test
reads. Rather than one module per policy, the schema is declared as data in
``sub.params["spec"]``: a mapping of batch key -> field spec. The data layer
therefore stays policy-agnostic -- it never imports :mod:`policies`, and adding
a policy adds a config block, not a backend.

A field spec is ``{"shape": [...], "dtype": ..., "dist": ..., <dist args>}``:

===========  ==========================================  =======================
``dist``     draws                                       extra keys
===========  ==========================================  =======================
``normal``   standard normal                             --
``uniform``  uniform over ``[low, high)``                ``low`` (0.0), ``high`` (1.0)
``randint``  integers over ``[low, high)``               ``low`` (0), ``high`` (required)
``full``     a constant                                  ``value`` (required)
===========  ==========================================  =======================

``dtype`` defaults to ``float32``; ``dist`` defaults by dtype (``normal`` for
floats, ``randint`` for ints, ``full`` for bool -- so a bool mask needs only a
shape and a ``value``). Samples are deterministic: the generator is seeded by
sample index, and fields are drawn in spec order.

Example -- a generic ``{observation, action}`` source (the default spec)::

    params: {num_samples: 256, spec: {observation: {shape: [32]},
                                      action: {shape: [8, 7]}}}

Example -- a pi05-shaped robotics source (keys owned by the policy, not us)::

    params:
      spec:
        observation.images.base_0_rgb:       {shape: [3, 224, 224], dist: uniform}
        observation.language.tokens:         {shape: [12], dtype: int64, high: 1000}
        observation.language.attention_mask: {shape: [12], dtype: bool, value: true}
        real_action:                         {shape: [4, 7]}

The data is random, so a synthetic source validates wiring, not learning.
"""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from magic_w0.data.backends.base import DataBackend
from magic_w0.data.config import SubDatasetConfig

_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

# Per-dtype default draw, so the common cases spell out only a shape.
_DEFAULT_DIST = {"float": "normal", "int": "randint", "bool": "full"}

_PARAM_KEYS = frozenset({"num_samples", "spec"})
_FIELD_KEYS = frozenset({"shape", "dtype", "dist", "low", "high", "value"})

_DEFAULT_NUM_SAMPLES = 256
_DEFAULT_SPEC: dict[str, dict[str, Any]] = {
    "observation": {"shape": [32]},
    "action": {"shape": [8, 7]},
}


def _default_dist(dtype: torch.dtype) -> str:
    if dtype is torch.bool:
        return _DEFAULT_DIST["bool"]
    return _DEFAULT_DIST["int" if not dtype.is_floating_point else "float"]


def _require(field: dict[str, Any], key: str, key_name: str, dist: str) -> Any:
    if key not in field:
        raise ValueError(f"spec[{key_name!r}]: dist {dist!r} requires a {key!r}")
    return field[key]


def _draw(key: str, field: dict[str, Any], gen: torch.Generator) -> torch.Tensor:
    """Draw one field of one sample from its spec."""
    unknown = set(field) - _FIELD_KEYS
    if unknown:
        raise ValueError(
            f"spec[{key!r}]: unknown keys {sorted(unknown)}; allowed {sorted(_FIELD_KEYS)}"
        )
    if "shape" not in field:
        raise ValueError(f"spec[{key!r}]: missing required 'shape'")
    shape = tuple(int(dim) for dim in field["shape"])

    dtype_name = str(field.get("dtype", "float32"))
    if dtype_name not in _DTYPES:
        raise ValueError(
            f"spec[{key!r}]: unknown dtype {dtype_name!r}; allowed {sorted(_DTYPES)}"
        )
    dtype = _DTYPES[dtype_name]

    dist = str(field.get("dist") or _default_dist(dtype))
    if dist == "normal":
        drawn = torch.randn(shape, generator=gen)
    elif dist == "uniform":
        low, high = float(field.get("low", 0.0)), float(field.get("high", 1.0))
        drawn = torch.rand(shape, generator=gen) * (high - low) + low
    elif dist == "randint":
        low = int(field.get("low", 0))
        high = int(_require(field, "high", key, dist))
        drawn = torch.randint(low, high, shape, generator=gen)
    elif dist == "full":
        drawn = torch.full(shape, float(_require(field, "value", key, dist)))
    else:
        raise ValueError(
            f"spec[{key!r}]: unknown dist {dist!r}; allowed ['normal', 'uniform', 'randint', 'full']"
        )

    return drawn.to(dtype)


class _SyntheticDataset(Dataset):
    """Deterministic (seed-by-index) dict-of-tensors dataset over a field spec."""

    def __init__(self, num_samples: int, spec: dict[str, dict[str, Any]]):
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}")
        if not spec:
            raise ValueError("spec must declare at least one field")
        self._n = num_samples
        self._spec = spec
        self._draw(0)  # fail at construction, not in a worker process

    def _draw(self, index: int) -> dict[str, torch.Tensor]:
        gen = torch.Generator().manual_seed(index)
        return {key: _draw(key, field, gen) for key, field in self._spec.items()}

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self._draw(index)


class SyntheticBackend(DataBackend):
    """Backend for ``type: synthetic`` sources; the schema comes from ``sub.params``."""

    version = "synthetic"

    def make_dataset(self, sub: SubDatasetConfig) -> _SyntheticDataset:
        params = sub.params or {}
        unknown = set(params) - _PARAM_KEYS
        if unknown:
            raise ValueError(
                f"synthetic source: unknown params {sorted(unknown)}; allowed {sorted(_PARAM_KEYS)}. "
                "Field shapes are declared under params['spec'], not as top-level knobs."
            )
        # An absent spec falls back to the default schema; an explicitly empty or
        # null one is a config mistake, not a request for the default.
        return _SyntheticDataset(
            num_samples=int(params.get("num_samples", _DEFAULT_NUM_SAMPLES)),
            spec=params["spec"] if "spec" in params else _DEFAULT_SPEC,
        )
