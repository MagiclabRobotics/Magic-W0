"""Bounded, deterministic recovery from corrupt dataset samples.

The wrapper lives outside the train/eval split. A failed training index is
therefore replaced only by another index in that same view, and every request
still returns exactly one sample. That fixed cardinality is the important DDP
contract: filtering failures in ``collate_fn`` would let ranks build different
batch sizes or even empty batches.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import Dataset, Subset, get_worker_info

logger = logging.getLogger(__name__)

DATA_RETRY_ATTEMPTS = "__magic_w0_data_retry_attempts"
DATA_RETRY_REPLACED = "__magic_w0_data_retry_replaced"


def _sum_count(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.detach().sum().cpu().item())
    if isinstance(value, (list, tuple)):
        return sum(int(item) for item in value)
    return int(value)


def preserve_retry_metadata(
    source: Mapping[str, Any], target: dict[str, Any]
) -> dict[str, Any]:
    """Copy collated retry counts through model-owned batch preparation.

    Counts become Python integers here so the host-to-device traversal does not
    create a GPU synchronization merely to log two tiny diagnostic scalars.
    """
    for key in (DATA_RETRY_ATTEMPTS, DATA_RETRY_REPLACED):
        if key in source:
            target[key] = _sum_count(source[key])
    return target


class RetryDataset(Dataset):
    """Retry failed reads with a deterministic sequence of replacement indices.

    ``max_retries`` counts replacements after the original read, so a value of 3
    performs at most four reads. Replacement offsets are a seed/index-dependent
    permutation of ``1..len(dataset)-1``. Thus attempts are reproducible across
    workers and restarts, spread across the same dataset view, and do not repeat
    a candidate until every other index has been tried (unless the retry bound
    exceeds the dataset size).
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        max_retries: int,
        seed: int,
        label: str,
        log_limit: int = 20,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if log_limit < 0:
            raise ValueError("log_limit must be non-negative")
        if len(dataset) <= 0:
            raise ValueError("RetryDataset requires a non-empty dataset")
        self.dataset = dataset
        self.max_retries = int(max_retries)
        self.seed = int(seed)
        self.label = str(label)
        self.log_limit = int(log_limit)
        # With persistent workers this counter is per worker, which is exactly
        # where log throttling is needed. Global successful counts travel in the
        # returned sample and are reduced by the trainer.
        self._failures_seen = 0

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _mix64(value: int) -> int:
        """Stable SplitMix64 finalizer; unlike hash(), independent of PYTHONHASHSEED."""
        mask = (1 << 64) - 1
        value = (value + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return value ^ (value >> 31)

    def replacement_index(self, requested_index: int, retry_number: int) -> int:
        """Return the deterministic candidate for retry numbers starting at one."""
        size = len(self)
        if retry_number <= 0:
            return int(requested_index)
        if size == 1:
            return 0
        modulus = size - 1
        key = (int(requested_index) & ((1 << 64) - 1)) ^ (self.seed & ((1 << 64) - 1))
        start = self._mix64(key) % modulus
        stride = 1 + (self._mix64(key ^ 0xD1B54A32D192ED03) % modulus)
        while math.gcd(stride, modulus) != 1:
            stride = 1 if stride == modulus else stride + 1
        slot = (start + (retry_number - 1) * stride) % modulus
        offset = 1 + slot
        return (int(requested_index) + offset) % size

    def _index_trace(self, index: int) -> tuple[int, ...]:
        """Expose wrapper/subset/raw-anchor indices to make a bad file findable."""
        trace = [int(index)]
        node: Any = self.dataset
        current = int(index)
        seen: set[int] = set()
        while node is not None and id(node) not in seen:
            seen.add(id(node))
            try:
                if isinstance(node, Subset):
                    current = int(node.indices[current])
                    trace.append(current)
                else:
                    indices = getattr(node, "indices", None)
                    if indices is not None:
                        current = int(indices[current])
                        trace.append(current)
                    sample_indices = getattr(node, "_sample_indices", None)
                    if sample_indices is not None:
                        current = int(sample_indices[current])
                        trace.append(current)
            except (IndexError, KeyError, TypeError, ValueError):
                break
            node = getattr(node, "dataset", None)
        return tuple(trace)

    def _should_log(self) -> bool:
        return self._failures_seen <= self.log_limit or (
            self._failures_seen % 1000 == 0
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        size = len(self)
        requested = int(index)
        if requested < 0:
            requested += size
        if requested < 0 or requested >= size:
            raise IndexError(index)

        failures: list[str] = []
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            candidate = (
                requested
                if attempt == 0
                else self.replacement_index(requested, attempt)
            )
            trace = self._index_trace(candidate)
            try:
                item = self.dataset[candidate]
                if not isinstance(item, Mapping):
                    raise TypeError(
                        "retry-wrapped datasets must return a mapping, got "
                        f"{type(item).__name__}"
                    )
                output = dict(item)
                output[DATA_RETRY_ATTEMPTS] = attempt
                output[DATA_RETRY_REPLACED] = int(attempt > 0)
                if attempt > 0 and self._should_log():
                    logger.warning(
                        "data retry recovered source=%s requested=%d "
                        "resolved=%d trace=%s failed_attempts=%d worker=%s",
                        self.label,
                        requested,
                        candidate,
                        trace,
                        attempt,
                        self._worker_id(),
                    )
                return output
            except MemoryError:
                # Retrying another sample under real process/cgroup exhaustion
                # worsens the failure and can hide the actual resource problem.
                raise
            except Exception as exc:  # noqa: BLE001 - decoder types vary
                last_error = exc
                self._failures_seen += 1
                summary = (
                    f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:500]}"
                )
                failures.append(
                    f"attempt={attempt} index={candidate} trace={trace} error={summary}"
                )
                if self._should_log():
                    logger.warning(
                        "data sample failed source=%s requested=%d attempt=%d/%d "
                        "candidate=%d trace=%s worker=%s error=%s",
                        self.label,
                        requested,
                        attempt,
                        self.max_retries,
                        candidate,
                        trace,
                        self._worker_id(),
                        summary,
                    )

        details = "; ".join(failures)
        message = (
            f"data retries exhausted for source={self.label!r}, requested={requested}, "
            f"max_retries={self.max_retries}: {details}"
        )
        logger.error(message)
        raise RuntimeError(message) from last_error

    @staticmethod
    def _worker_id() -> str:
        worker = get_worker_info()
        return "main" if worker is None else str(worker.id)


__all__ = [
    "DATA_RETRY_ATTEMPTS",
    "DATA_RETRY_REPLACED",
    "RetryDataset",
    "preserve_retry_metadata",
]
