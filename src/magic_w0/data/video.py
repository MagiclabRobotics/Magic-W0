"""Small video-decoder compatibility layer for standalone LeRobot readers."""

from __future__ import annotations

import logging
from pathlib import Path

import torch

_TORCHCODEC_FAILURE: str | None = None


def make_video_decoder(path: str | Path, *, backend: str, fps: float):
    """Build a decoder with the subset of TorchCodec's API used by the readers."""
    if backend not in {"auto", "torchcodec", "pyav"}:
        raise ValueError("video_backend must be 'auto', 'torchcodec', or 'pyav'")
    if backend in {"auto", "torchcodec"} and _torchcodec_usable():
        try:
            from torchcodec.decoders import VideoDecoder

            return VideoDecoder(str(path), dimension_order="NCHW")
        except Exception as exc:
            if backend == "torchcodec":
                raise
            _remember_torchcodec_failure(exc)
    return PyAVVideoDecoder(path, fps=fps)


def _torchcodec_usable() -> bool:
    return _TORCHCODEC_FAILURE is None


def _remember_torchcodec_failure(exc: Exception) -> None:
    global _TORCHCODEC_FAILURE
    if _TORCHCODEC_FAILURE is None:
        _TORCHCODEC_FAILURE = str(exc)
        logging.warning("TorchCodec unavailable; falling back to PyAV: %s", exc)


class PyAVVideoDecoder:
    """Seekable PyAV decoder exposing TorchCodec-like frame selection methods."""

    def __init__(self, path: str | Path, *, fps: float) -> None:
        self.path = str(path)
        self.fps = float(fps)
        import av

        with av.open(self.path) as container:
            stream = container.streams.video[0]
            self.num_frames = int(stream.frames or 0)
            self.duration_seconds = (
                float(stream.duration * stream.time_base)
                if stream.duration is not None
                else 0.0
            )
        self.metadata = _VideoMetadata(self.num_frames, self.duration_seconds)

    def get_frames_at(self, *, indices: list[int]):
        seconds = [(index + 0.5) / self.fps for index in indices]
        return _FrameBatch(self._decode_nearest(seconds))

    def get_frames_played_at(self, *, seconds: list[float]):
        return _FrameBatch(self._decode_nearest(seconds))

    def _decode_nearest(self, targets: list[float]) -> torch.Tensor:
        if not targets:
            return torch.empty((0, 3, 0, 0), dtype=torch.uint8)
        import av

        ordered = sorted(
            enumerate(float(value) for value in targets), key=lambda pair: pair[1]
        )
        # ffmpeg's default thread_count is 0 = auto, and auto sizes the decode
        # thread pool from the cores it can SEE, not the cores this process may
        # USE.  Inside a container those differ: EIP's ml.a800.4 slice reports
        # nproc=255 while /sys/fs/cgroup/cpu.max grants 3100000/100000 = 31.
        # A batch of 64 samples across 3 cameras is 192 av.open() calls, each in
        # one of 8 dataloader workers per rank -- with 255-wide pools that blows
        # past the quota, and CFS throttling then freezes the WHOLE cgroup,
        # including the main process that feeds the GPUs.  Measured on 4xA800
        # (EIP 7555 -> 7560), num_workers=8 either side of this one line:
        #   t_fwd  2.10 -> 0.97   t_bwd 3.59 -> 2.05   3.43 s/step vs 6.66
        # t_data stays 0.00 in both, i.e. one thread per decode is still ample
        # -- the workers were never short of throughput, only of quota.
        # NEAREST-FRAME SELECTION STREAMS; IT DOES NOT BUFFER THE WINDOW.
        # Decoded frames arrive in non-decreasing timestamp order and the targets
        # are sorted, so the nearest frame to a target is always one of the two
        # that straddle it -- holding a single pending frame is enough.  The
        # earlier version appended every frame in [min_target - 1 s, max_target]
        # and argmin'd at the end, i.e. it held ONE SECOND of frames at the
        # video's native resolution (fps frames; ~0.9 MB each at 640x480, ~6.2 MB
        # at 1080p) to return one of them.  Resizing to image_size happens later,
        # so nothing shrinks the buffer first.
        #
        # That mattered once MagicW0 added world supervision: the future frame is
        # fetched by _load_future_item, so a sample went from 3 decode calls
        # (one per camera) to 5, in 256 dataloader workers per node.  Repeatedly
        # allocating and freeing multi-MB blocks of VARYING size is precisely the
        # pattern that ratchets an allocator -- see vlm_jsonl.py:358 for the same
        # failure mode on pyarrow, where the retained amount reached 4-5x the
        # largest single block and never came back down.
        #
        # TIES BREAK TOWARD THE EARLIER FRAME.  get_frames_at asks for
        # (index + 0.5) / fps, so a target sits exactly midway between two frames
        # whose timestamps are multiples of 1/fps, and the distances are exactly
        # equal.  min() returned the first minimum in decode order, i.e. the
        # requested index; "<=" below reproduces that.  ">" would silently return
        # frame i+1 for every exact-index fetch.
        selected: list[torch.Tensor | None] = [None] * len(targets)
        first_target = max(ordered[0][1] - 1.0, 0.0)
        tolerance = 1.0 / self.fps
        pending = iter(ordered)
        current: tuple[int, float] | None = next(pending, None)
        previous: tuple[float, torch.Tensor] | None = None
        accepted = 0
        with av.open(self.path) as container:
            stream = container.streams.video[0]
            stream.thread_count = 1
            if stream.time_base is not None:
                container.seek(
                    int(first_target / float(stream.time_base)),
                    stream=stream,
                    backward=True,
                    any_frame=False,
                )
            for frame in container.decode(stream):
                timestamp = (
                    float(frame.pts * stream.time_base)
                    if frame.pts is not None and stream.time_base is not None
                    else accepted / self.fps
                )
                if timestamp + tolerance < first_target:
                    continue
                image = torch.from_numpy(frame.to_ndarray(format="rgb24")).permute(
                    2, 0, 1
                )
                accepted += 1
                while current is not None and timestamp >= current[1]:
                    original_index, target = current
                    if (
                        previous is not None
                        and target - previous[0] <= timestamp - target
                    ):
                        selected[original_index] = previous[1]
                    else:
                        selected[original_index] = image
                    current = next(pending, None)
                previous = (timestamp, image)
                if current is None:
                    break
        if previous is None:
            raise RuntimeError(f"PyAV decoded no frames from {self.path}")
        # Targets past the last decoded frame resolve to it, which is what the
        # argmin over the collected window did.
        while current is not None:
            selected[current[0]] = previous[1]
            current = next(pending, None)
        return torch.stack([image for image in selected if image is not None])


class _VideoMetadata:
    def __init__(self, num_frames: int, duration_seconds: float) -> None:
        self.num_frames = num_frames
        self.duration_seconds = duration_seconds


class _FrameBatch:
    def __init__(self, data: torch.Tensor) -> None:
        self.data = data


__all__ = ["PyAVVideoDecoder", "make_video_decoder"]
