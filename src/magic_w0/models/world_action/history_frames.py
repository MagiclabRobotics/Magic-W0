"""MagicW0 visual history for ``magicw0`` (off unless ``history_enabled``).

``history_frames`` STRICTLY PAST frames of the history camera, one every
``history_interval`` action steps, are encoded by the VLM's own vision tower;
each frame's merged 8x8 grid is adaptive-average-pooled to ``pool x pool``
tokens and scattered into ``<|vision_pad|>`` placeholders.  The user turn is
laid out as in MagicW0::

    Robot task: ...\\n
    History images: <16 slots of frame 1>\\n<16 slots of frame 2>\\n...
    Head image: <current head> Left wrist image: <...> Right wrist image: <...>
    <state text><closing sentence>

No timestamps: every frame run has the same length and the current views follow
the block directly, so the relative RoPE distance from "k slots ago" to the
current views is the same at every episode phase.

MagicW0 keeps a fixed slot grid and masks the empty slots (``<unused1>``:
embedding zeroed, attention masked, position skipped).  That is exact on its
all-softmax Gemma3 backbone but not here: 18 of Qwen3.5's 24 layers are Gated
DeltaNet, where a masked token in the middle of a row still decays the carried
state.  So an empty slot is not rendered at all -- the same sequence MagicW0's
masking produces -- and batching only ever pads on the left, before any real
token, where the recurrent state is still exactly zero.
"""

from __future__ import annotations


HISTORY_FRAMES = "magicw0.history_frames"
HISTORY_LEN = "magicw0.history_len"
#: Per-sample dropout outcome, collated for diagnostics only.
HISTORY_MODE = "magicw0.history_mode"
HISTORY_MODE_KEEP = 0
HISTORY_MODE_PARTIAL = 1
HISTORY_MODE_FULL = 2
HISTORY_MODE_EXEMPT = 3
#: vlm_inputs key holding the image-processor output of the past frames.
HISTORY_IMAGES = "history_images"
#: Reserved by Qwen3.5 and never emitted by its image processor, so it can play
#: MagicW0's ``<unused0>``.  ``<|image_pad|>`` must NOT be used: the processor's
#: grid bookkeeping would count it as a current image.
HISTORY_PLACEHOLDER_TOKEN = "<|vision_pad|>"

_IMAGE_LABELS = {
    "cam_high": "Head",
    "cam_left_wrist": "Left wrist",
    "cam_right_wrist": "Right wrist",
}


def history_frame_stride(history_interval: int, action_upsample_rate: int) -> int:
    """Video frames per history slot for an interval given in action steps."""
    interval, rate = int(history_interval), int(action_upsample_rate)
    if interval < 1 or rate < 1:
        raise ValueError(
            f"interval and upsample rate must be >= 1, got {interval}, {rate}"
        )
    if interval % rate:
        raise ValueError(
            f"history_interval={interval} action steps is not a whole number of "
            f"video frames at action_upsample_rate={rate}"
        )
    return interval // rate


def past_frame_indices(frame_index: int, n_frames: int, stride: int) -> list[int]:
    """Episode-local strictly past frames, oldest first; none before frame 0."""
    frame = int(frame_index)
    if frame < 0:
        raise ValueError(f"frame_index must be non-negative, got {frame}")
    if n_frames < 1 or stride < 1:
        raise ValueError(f"n_frames and stride must be >= 1, got {n_frames}, {stride}")
    return [
        frame - back * stride
        for back in range(n_frames, 0, -1)
        if frame - back * stride >= 0
    ]


# Use the same history prompt and pooling implementation as public inference.
from magic_w0.processing import (
    history_block_text as history_block_text,
    pool_frame_features as pool_frame_features,
    image_prompt_label as image_prompt_label,
)
