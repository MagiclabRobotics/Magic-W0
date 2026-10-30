# Copyright 2026 MagicVLA authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Lazy construction of the released MagicW0 training model and data processor.

Import targets are resolved only when building the model or processor. Importing
configuration and data tools therefore does not instantiate a pretrained model.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

COTRAIN = "cotrain"
_KINDS = (COTRAIN,)


class UnknownModelError(KeyError):
    """Raised when a model name is not registered. Message lists what is."""

    def __str__(self) -> str:  # KeyError reprs its arg; we want the plain message
        return self.args[0] if self.args else ""


def _resolve(target: str) -> type:
    """Import ``"pkg.module:Attr"`` and return ``Attr``."""
    module_name, sep, attr = target.partition(":")
    if not sep or not attr:
        raise ValueError(
            f"import target must look like 'pkg.module:Attr', got {target!r}"
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ImportError(
            f"{module_name!r} has no attribute {attr!r} (target {target!r})"
        ) from exc


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """What the registry knows about a model before anything is imported."""

    name: str
    kind: str
    config_target: str  # Lazy import target for the MagicW0 configuration.
    policy_target: str  # Lazy import target for the MagicW0 policy.
    data_processor_target: str | None = None
    description: str = ""

    def config_cls(self) -> type:
        """Import and return the config class (first call pays the import)."""
        return _resolve(self.config_target)

    def policy_cls(self) -> type:
        """Import and return the policy class (first call pays the import)."""
        return _resolve(self.policy_target)

    def data_processor_cls(self) -> type | None:
        """Return the optional model-owned data processor class."""
        return (
            _resolve(self.data_processor_target) if self.data_processor_target else None
        )


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(
    name: str,
    *,
    kind: str,
    config_target: str,
    policy_target: str,
    data_processor_target: str | None = None,
    description: str = "",
    override: bool = False,
) -> ModelSpec:
    """Register a model under ``name``.

    Targets are ``"pkg.module:Attr"`` strings, imported on demand -- so
    registering costs no import. Pass ``override=True`` to replace an existing
    entry; without it, a duplicate name is an error rather than a silent
    shadowing of the policy someone else registered.
    """
    if kind not in _KINDS:
        raise ValueError(f"unknown model kind {kind!r}; allowed {list(_KINDS)}")
    if name in _REGISTRY and not override:
        raise ValueError(
            f"model {name!r} is already registered; pass override=True to replace it"
        )
    spec = ModelSpec(
        name=name,
        kind=kind,
        config_target=config_target,
        policy_target=policy_target,
        data_processor_target=data_processor_target,
        description=description,
    )
    _REGISTRY[name] = spec
    return spec


def list_models(kind: str | None = None) -> list[str]:
    """Registered model names, sorted; optionally filtered to one ``kind``."""
    if kind is not None and kind not in _KINDS:
        raise ValueError(f"unknown model kind {kind!r}; allowed {list(_KINDS)}")
    return sorted(n for n, s in _REGISTRY.items() if kind is None or s.kind == kind)


def get_model_spec(name: str) -> ModelSpec:
    """Look up a spec by name. Nothing is imported."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise UnknownModelError(
            f"unknown model {name!r}. Registered: {list_models()}. "
            "Add one with models.register_model(...)."
        ) from None


def build_model(
    name: str,
    params: dict[str, Any] | None = None,
    *,
    device: str | None = None,
    kind: str | None = None,
    checkpoint: str | None = None,
    local_files_only: bool = True,
    strict: bool = True,
):
    """Build a model by name from a plain (YAML-shaped) ``params`` dict.

    ``kind`` is an assertion by the caller about what it can drive: the
    co-training loop passes ``kind=COTRAIN`` so that naming a scaffold policy
    fails here, with a list of the trainable ones, rather than deep inside a
    training step. ``device`` is applied only if the config declares the field;
    asking for a device a config cannot hold is an error, not a no-op.
    """
    spec = get_model_spec(name)
    if kind is not None and spec.kind != kind:
        raise ValueError(
            f"model {name!r} is a {spec.kind!r} model, not {kind!r}. "
            f"{kind!r} models: {list_models(kind=kind)}"
        )

    config_cls = spec.config_cls()
    from_dict = getattr(config_cls, "from_dict", None)
    if from_dict is None:
        raise TypeError(
            f"{spec.config_target} must expose a from_dict(payload) classmethod "
            "to be built by the registry"
        )
    config = from_dict(dict(params or {}))

    if device:
        if not hasattr(config, "device"):
            raise ValueError(
                f"model {name!r} config ({type(config).__name__}) has no 'device' field, "
                f"so device={device!r} cannot be applied"
            )
        config.device = device

    model_cls = spec.policy_cls()
    if checkpoint is None:
        return model_cls(config)
    from_pretrained = getattr(model_cls, "from_pretrained", None)
    if from_pretrained is None:
        raise TypeError(f"model {name!r} does not support checkpoint loading")
    return from_pretrained(
        checkpoint,
        config=config,
        local_files_only=local_files_only,
        strict=strict,
    )


def build_data_processor(
    name: str,
    params: dict[str, Any] | None,
    *,
    model_params: dict[str, Any] | None = None,
):
    """Build the optional model-owned adapter from raw source samples."""
    spec = get_model_spec(name)
    processor_cls = spec.data_processor_cls()
    if processor_cls is None:
        if params:
            raise ValueError(f"model {name!r} does not define a data processor")
        return None

    model_config = spec.config_cls().from_dict(dict(model_params or {}))
    from_dict = getattr(processor_cls, "from_dict", None)
    if from_dict is None:
        raise TypeError(
            f"{spec.data_processor_target} must expose "
            "from_dict(payload, model_config=...)"
        )
    return from_dict(dict(params or {}), model_config=model_config)


register_model(
    "magic_w0",
    kind=COTRAIN,
    config_target="magic_w0.models.world_action:MagicW0TrainingConfig",
    policy_target="magic_w0.models.world_action:MagicW0TrainingPolicy",
    data_processor_target="magic_w0.models.world_action:MagicW0TrainingDataProcessor",
    description="MagicW0 released three-stream world-action architecture.",
)
