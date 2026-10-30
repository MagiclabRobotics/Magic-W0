# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Move the model's CPU batch preparation into the DataLoader workers.

`prepare_batch` (chat templating, the Qwen processor's image resize/normalize/patchify,
FAST action tokenization) is pure CPU work, but it used to run in the training loop's
main process between fetching a batch and the forward pass -- single-threaded, with the
GPU idle and the `num_workers` dataloader workers already finished and waiting.

Measured on the eo cotrain run (8xA800, robot batch 32, 3 cameras): `t_prep` was 1.95 s
of a 6.63 s step -- 29%, about as much as the forward pass -- while `t_data` was 0.000 s,
i.e. the workers had nothing left to do. Running it as the loader's `collate_fn` puts it
in those workers instead: `num_workers`-way parallel AND overlapped with GPU compute via
prefetch, so a step only pays for it when the workers cannot keep up.

Two details make this correct rather than merely faster:

* **Modality must be baked in per source.** `InterleavedLoader` tags the modality in the
  MAIN process, after collation (see `interleaved.py`), so a worker-side preparer would
  never see it and would silently route every batch to the native modality. Each source's
  collator therefore carries its own tag and injects it before preparing.
* **Workers must not touch CUDA.** The preparer keeps everything on CPU (`device=None`);
  the training loop still performs the host-to-device move.

The collator holds a reference to the model, which is free under the default `fork` start
method (workers inherit it copy-on-write, and the GPU tensors are never read). Under
`spawn` it would have to pickle the model, so this raises a clear error instead.
"""

from __future__ import annotations

import ctypes
import gc
from typing import Any

from torch.utils.data._utils.collate import default_collate

from magic_w0.data.retry import preserve_retry_metadata

# Collect once per this many prepared samples. Counted in SAMPLES, not batches,
# so the cadence does not move when a recipe changes batch size or when a budget
# sampler hands over batches of varying size.
#
# WHY ANY OF THIS.  prepare_batch renders a chat template per batch
# (modeling_magicw0_base.py:1188), and rendering leaves a Jinja reference cycle
# behind -- Macro -> cell -> Context -> Macro, one set per sample. Trapped inside
# it is the message content built just above the render, which holds one
# ``{"type": "image", "image": <tensor>}`` per camera. Refcounting cannot free a
# cycle, so those image tensors wait for the cyclic collector, and CPython
# schedules that by object COUNT (gen0 at 700 allocations): a cycle contributing
# a few dozen objects while holding megabytes of pixels barely moves the counter,
# so it survives for a long time. Measured on RoboTwin2.0, 12,000 samples, with
# nothing reclaiming: +52.6 KB per sample, linear, no saturation. EgoDex is flat
# -- it has one camera, so its cycles hold a third as much.
#
# It is not allocator retention: malloc_trim(0) after EVERY batch still leaves
# +230 MB, because the memory is still reachable from Python's point of view.
# gc.collect() takes it to zero. The trim after it is what hands the freed pages
# back to the OS rather than leaving them in a glibc arena.
#
# This is per dataloader worker, and a node runs 256 of them, which is how the
# 48-GPU pretrain was OOMKilled (EIP 7370, step 44760) with the main process RSS
# flat at 17 GB the whole time.
#
# WHAT THE INTERVAL BUYS, measured end to end through this collator on the source
# that leaks (RoboTwin2.0, 3 cameras, 9,600 samples, nothing else reclaiming):
#
#   off        +304.9 -> +405.8 -> +522.9 MB   still climbing   43.8 ms/batch
#   every 512  +195.7 -> +207.7 -> +138.1 MB   bounded          46.4 ms/batch
#
# So this bounds the growth rather than removing it: a worker settles into a
# sawtooth around ~170 MB instead of climbing without limit, which is the
# difference that matters -- the run died of the climb, not of a plateau. The
# +5.8% is collator time, not step time: t_data measures 0.0006 s, i.e. the
# loader is entirely hidden behind GPU compute, so the cost lands in spare worker
# CPU. Shrinking the interval trades that CPU for a lower plateau and vice versa;
# this is the knob if a future recipe needs one end or the other.
_COLLECT_EVERY_SAMPLES = 512


def _load_malloc_trim():
    """glibc's malloc_trim, or a no-op where it does not exist (musl, macOS)."""
    try:
        return ctypes.CDLL("libc.so.6").malloc_trim
    except OSError:
        return None


_MALLOC_TRIM = _load_malloc_trim()


def _release_cycles() -> None:
    """Break the render's cycles, then return the freed pages to the OS."""
    gc.collect()
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


class PreparedBatchCollator:
    """Collate samples, tag the modality, then run the model's CPU-side preparation."""

    def __init__(
        self,
        model: Any,
        modality: str,
        modality_key: str = "modality",
        collect_every_samples: int = _COLLECT_EVERY_SAMPLES,
    ) -> None:
        prepare = getattr(model, "prepare_batch", None)
        if prepare is None:
            raise ValueError(
                "model must expose prepare_batch() to offload batch preparation"
            )
        # 0 disables the periodic collection, for a caller that has its own policy.
        self._collect_every = max(0, int(collect_every_samples))
        self._samples_since_collect = 0
        self._model = model
        self._modality = modality
        self._modality_key = modality_key

    def __call__(self, samples):
        batch = default_collate(samples)
        if isinstance(batch, dict):
            batch[self._modality_key] = self._modality
        # device=None keeps every tensor on CPU: the worker must not touch CUDA.
        prepared = self._model.prepare_batch(batch, None)
        preserve_retry_metadata(batch, prepared)
        # After preparing, not before: the cycles this clears are the ones the
        # call just created, and the prepared batch is about to leave for the
        # main process, so nothing here is holding it.
        self._samples_since_collect += len(samples)
        if self._collect_every and self._samples_since_collect >= self._collect_every:
            self._samples_since_collect = 0
            _release_cycles()
        return prepared

    def __reduce__(self):
        raise RuntimeError(
            "PreparedBatchCollator cannot be pickled -- DataLoader workers must inherit it "
            "via fork. If the start method is 'spawn', pass collate_factory=None to keep "
            "batch preparation in the training loop."
        )


def make_collate_factory(model: Any, modality_key: str = "modality"):
    """Return ``modality -> collate_fn``, or ``None`` when the model cannot offload.

    A model opts in simply by tolerating ``prepare_batch(batch, None)``; one that still
    requires a device returns tensors on that device and would break the contract, so it
    is expected to declare support via ``supports_cpu_prepare = True``.
    """
    if not getattr(model, "supports_cpu_prepare", False):
        return None
    return lambda modality: PreparedBatchCollator(model, modality, modality_key)
