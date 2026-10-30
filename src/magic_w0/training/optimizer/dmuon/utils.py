"""DMuon detection helpers.

The repository does not require DMuon, but keeping this boundary lets trainer
code test for it without importing optional dependencies at module import time.
"""

from __future__ import annotations

import importlib.util


def is_dmuon_available() -> bool:
    """Return whether the optional dmuon package can be imported."""
    return importlib.util.find_spec("dmuon") is not None


def is_dmuon_model(model) -> bool:
    """Return whether a model exposes DMuon-dedicated parameters."""
    if not is_dmuon_available():
        return False
    import dmuon

    return bool(dmuon.get_dedicated_params(model))
