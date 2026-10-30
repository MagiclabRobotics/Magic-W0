"""Portable YAML recipes with relative includes and explicit environment roots."""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml

_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def _expand(value):
    if isinstance(value, str):

        def replace(match):
            name, default = match.groups()
            result = os.environ.get(name) or default
            if result is None:
                raise ValueError(f"Required environment variable {name} is unset")
            return result

        return _ENV.sub(replace, value)
    if isinstance(value, list):
        return [_expand(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand(item) for key, item in value.items()}
    return value


def _merge(base, override):
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def load_recipe(path: str | Path, *, _stack=()) -> dict:
    """Merge includes beside each YAML; child values override included values.

    Lists replace rather than append. Runtime data/model paths are relative to
    the repository working directory, while include paths are YAML-relative.
    """
    path = Path(path).resolve()
    if path in _stack:
        raise ValueError(f"Cyclic recipe include: {path}")
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise TypeError(f"Recipe must contain a mapping: {path}")
    includes = payload.pop("include", [])
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list) or any(not isinstance(x, str) for x in includes):
        raise TypeError("include must be a path or list of paths")
    result = {}
    for item in includes:
        result = _merge(result, load_recipe(path.parent / item, _stack=(*_stack, path)))
    return _expand(_merge(result, payload))
