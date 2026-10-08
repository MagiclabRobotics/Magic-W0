"""Installable Magic-W0 inference runtime.

Configuration and checkpoint helpers do not import PyTorch. Model and image
processing components are loaded only when requested.
"""

from .config import (
    MagicW0Config,
    validate_checkpoint_modes,
    SUPPORTED_MODES,
    STATE,
    ACTION_DIM_MASK,
    CAMERA_VALID,
    TASK,
    EMBODIMENT,
    STATE_DIM_MASK,
    VIEW_VALID_MASK,
    WORLD_3D_VALID_MASK,
    HISTORY_FRAMES,
    HISTORY_LEN,
    HISTORY_IMAGES,
    HISTORY_PLACEHOLDER_TOKEN,
)

__all__ = [
    "MagicW0",
    "MagicW0Config",
    "validate_checkpoint_modes",
    "SUPPORTED_MODES",
    "STATE",
    "ACTION_DIM_MASK",
    "CAMERA_VALID",
    "TASK",
    "EMBODIMENT",
    "STATE_DIM_MASK",
    "VIEW_VALID_MASK",
    "WORLD_3D_VALID_MASK",
    "HISTORY_FRAMES",
    "HISTORY_LEN",
    "HISTORY_IMAGES",
    "HISTORY_PLACEHOLDER_TOKEN",
    "_prepare_image",
]


def __getattr__(name):
    if name == "MagicW0":
        from .modeling import MagicW0

        return MagicW0
    if name == "_prepare_image":
        from .processing import _prepare_image

        return _prepare_image
    raise AttributeError(name)
