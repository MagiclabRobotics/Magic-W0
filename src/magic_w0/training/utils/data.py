"""Data helpers shared by trainer implementations."""

from __future__ import annotations


def move_to_device(value, device: str | None):
    """Recursively move tensors in a nested structure to a device."""
    if device is None:
        return value
    if hasattr(value, "to"):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    return value


def batch_size(batch) -> int:
    """Best-effort batch-size extraction for supported batch containers."""
    if hasattr(batch, "batch_size"):
        return int(batch.batch_size)
    if isinstance(batch, (tuple, list)):
        return len(batch)
    if isinstance(batch, dict):
        for value in batch.values():
            if hasattr(value, "shape") and value.shape:
                return int(value.shape[0])
    raise ValueError(f"Cannot infer batch size from {type(batch).__name__}.")
