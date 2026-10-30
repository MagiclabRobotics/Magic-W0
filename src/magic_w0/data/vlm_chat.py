"""Shared Qwen chat-message construction for VLM data and training."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any


def build_vlm_chat_messages(
    turns: Sequence[dict[str, str]],
    n_images: int,
    *,
    image_at: Callable[[int], Any] | None = None,
) -> list[dict[str, Any]]:
    """Build the exact multimodal message structure consumed by Qwen.

    ``image_at=None`` emits image placeholders only.  This is sufficient for
    ``apply_chat_template(..., tokenize=False)`` and lets the data scanner price
    the final token sequence without decoding any referenced images.
    """
    if n_images < 0:
        raise ValueError("n_images must be non-negative")

    def image_content(index: int) -> dict[str, Any]:
        content: dict[str, Any] = {"type": "image"}
        if image_at is not None:
            content["image"] = image_at(index)
        return content

    messages: list[dict[str, Any]] = []
    image_index = 0
    for turn in turns:
        role = str(turn["role"])
        text = str(turn["text"])
        if role == "assistant":
            messages.append(
                {"role": "assistant", "content": [{"type": "text", "text": text}]}
            )
            continue
        if role != "user":
            raise ValueError(f"unsupported VLM chat role {role!r}")
        content: list[dict[str, Any]] = []
        segments = text.split("<image>")
        for part_index, segment in enumerate(segments):
            segment = segment.strip()
            if segment:
                content.append({"type": "text", "text": segment})
            if part_index < len(segments) - 1 and image_index < n_images:
                content.append(image_content(image_index))
                image_index += 1
        messages.append(
            {"role": "user", "content": content or [{"type": "text", "text": " "}]}
        )

    while image_index < n_images:
        for turn in reversed(messages):
            if turn["role"] == "user":
                turn["content"].append(image_content(image_index))
                break
        else:
            messages.append({"role": "user", "content": [image_content(image_index)]})
        image_index += 1
    return messages
