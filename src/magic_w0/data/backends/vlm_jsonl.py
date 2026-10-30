# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Indexed JSONL text source for VLM co-training."""

from __future__ import annotations

import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from magic_w0.data.backends.base import DataBackend
from magic_w0.data.config import IMAGE_MODES, VlmJsonlSourceConfig
from magic_w0.data.vlm_chat import build_vlm_chat_messages

logger = logging.getLogger(__name__)


class VlmJsonlBackend(DataBackend):
    """Read a JSONL file or an ``out/meta/index.jsonl`` sharded directory."""

    version = "jsonl"

    def make_dataset(self, sub: VlmJsonlSourceConfig):
        return VlmJsonlDataset(
            sub.root,
            splits=sub.splits,
            capabilities=sub.capabilities,
            subsets=sub.subsets,
            max_token_len=sub.max_token_len,
            include_action=sub.include_action,
            include_video=sub.include_video,
            text_field=sub.text_field,
            max_samples=sub.max_samples,
            image_mode=sub.image_mode,
            image_column=sub.image_column,
            image_size=sub.image_size,
            image_resize_mode=sub.image_resize_mode,
            max_images=sub.max_images,
            max_multimodal_tokens=sub.max_multimodal_tokens,
            vision_token_stride=sub.vision_token_stride,
            image_token_cost=sub.image_token_cost,
            sequence_length_processor=sub.sequence_length_processor,
            parquet_path_remap=sub.parquet_path_remap,
            exclude_flags=sub.exclude_flags,
        )


class VlmJsonlDataset(Dataset):
    """Map-style text dataset backed by byte offsets into one or more JSONL files."""

    def __init__(
        self,
        root: str | Path,
        *,
        splits: tuple[str, ...] = ("train",),
        capabilities: list[str] | None = None,
        subsets: list[str] | None = None,
        max_token_len: int | None = None,
        include_action: bool = False,
        include_video: bool = False,
        text_field: str | None = None,
        max_samples: int | None = None,
        image_mode: str | None = None,
        image_column: str = "image",
        image_size: int | None = 256,
        image_resize_mode: str = "square",
        max_images: int = 8,
        max_multimodal_tokens: int | None = None,
        vision_token_stride: int = 28,
        image_token_cost: int | None = None,
        sequence_length_processor: str | None = None,
        parquet_path_remap: list[str] | None = None,
        exclude_flags: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.splits = frozenset(splits)
        self.capabilities = frozenset(capabilities or ())
        self.subsets = frozenset(subsets or ())
        self.max_token_len = max_token_len
        self.include_action = include_action
        self.include_video = include_video
        self.text_field = text_field
        # image_mode enables loading the sample's image(s) so the VLM source is a
        # real image-text VQA sample (not just flattened text). Supports N images per
        # sample (interleave / multiview); samples with more than max_images are skipped.
        # An unrecognised value must not fall through to the text path: that failure is
        # silent, and it turns a whole VQA corpus into unanswerable text-only questions.
        if image_mode is not None and image_mode not in IMAGE_MODES:
            raise ValueError(
                f"unknown image_mode {image_mode!r}; expected one of {IMAGE_MODES} or None"
            )
        self.image_mode = image_mode
        # Row-level flags marking weak supervision (e.g. magiclabel's `label_suspect`,
        # `phase_derived`). A row is dropped when any named field is truthy.
        self.exclude_flags = tuple(exclude_flags or ())
        self.image_column = image_column
        self.image_size = image_size
        if image_resize_mode not in {"square", "resize_with_pad"}:
            raise ValueError(
                "image_resize_mode must be 'square' or 'resize_with_pad', "
                f"got {image_resize_mode!r}"
            )
        self.image_resize_mode = image_resize_mode
        self.max_images = max_images
        # Per-sample multimodal-cost guard: skip (don't downscale) samples whose
        # estimated text+vision token budget is exceeded, so one huge sample can't OOM
        # the VLM or blow up the padded batch length.
        self.max_multimodal_tokens = max_multimodal_tokens
        self.vision_token_stride = max(1, vision_token_stride)
        self.image_token_cost = int(
            image_token_cost
            or (max(1, (image_size or 0) // self.vision_token_stride) ** 2)
        )
        self.sequence_length_processor = sequence_length_processor
        self._cost_processor = None
        self._cost_tokenizer = None
        self._image_pad_token_id: int | None = None
        if sequence_length_processor is not None:
            self._load_sequence_length_processor(sequence_length_processor)
        # Rewrite recorded images_ref.parquet paths when the source image dataset
        # was relocated on shared storage. Expect [old_substring, new_substring].
        if parquet_path_remap is not None and len(parquet_path_remap) != 2:
            raise ValueError("parquet_path_remap must be [old, new]")
        self.parquet_path_remap = (
            tuple(parquet_path_remap) if parquet_path_remap else None
        )
        self._skipped_oversized: list[str] = []
        self._records: list[tuple[Path, int]] = []
        # (images, final_sequence_tokens) per accepted record, captured during the
        # scan so the sampler can bound the padded Qwen language-model sequence.
        # Images are truncated at max_images because _load_reference_images() truncates too.
        self._n_images: list[int] = []
        self._text_tokens: list[int] = []
        self._handles: OrderedDict[Path, Any] = OrderedDict()
        self._parquet_cache: OrderedDict[str, Any] = OrderedDict()
        self._row_group_cache: dict[str, Any] = {}

        if max_samples is not None and max_samples <= 0:
            raise ValueError("max_samples must be positive when set")
        for path in _resolve_shards(self.root):
            with path.open("rb") as handle:
                while True:
                    offset = handle.tell()
                    line = handle.readline()
                    if not line:
                        break
                    if not line.strip():
                        continue
                    sample = json.loads(line)
                    if self._accept(sample) and _extract_text(sample, text_field):
                        n_images = (
                            min(self._sample_n_images(sample), self.max_images)
                            if self.image_mode
                            else 0
                        )
                        sequence_tokens = self._sample_sequence_tokens(sample, n_images)
                        if (
                            self.max_multimodal_tokens is not None
                            and sequence_tokens > self.max_multimodal_tokens
                        ):
                            if len(self._skipped_oversized) < 1000:
                                self._skipped_oversized.append(
                                    str(sample.get("id", "?"))
                                )
                            continue
                        self._records.append((path, offset))
                        self._n_images.append(n_images)
                        self._text_tokens.append(sequence_tokens)
                        if (
                            max_samples is not None
                            and len(self._records) >= max_samples
                        ):
                            break
            if max_samples is not None and len(self._records) >= max_samples:
                break
        if not self._records:
            raise ValueError(f"no usable VLM text samples found under {self.root}")
        if self._skipped_oversized:
            logger.warning(
                "vlm_jsonl cost guard skipped %d oversized sample(s) (> %d final sequence tokens); "
                "first ids: %s",
                len(self._skipped_oversized),
                self.max_multimodal_tokens,
                self._skipped_oversized[:10],
            )

    def batch_costs(self) -> list[tuple[int, int]]:
        """Per-sample ``(n_images, final_sequence_tokens)`` for cost batching.

        The second value is the complete per-sample Qwen input sequence, including
        image placeholders expanded to their real vision-token count. The sampler
        therefore prices language-model padding against the longest *full* sequence,
        not merely the longest text string.

        Consumed by :class:`data.batching.MultimodalBudgetBatchSampler`. Neither
        term needs a tokenizer or an image decode -- both come off the scan.
        """
        return list(zip(self._n_images, self._text_tokens, strict=True))

    def _load_sequence_length_processor(self, path: str) -> None:
        try:
            from transformers import AutoProcessor
        except ImportError as exc:
            raise ImportError(
                "sequence_length_processor requires transformers with the matching Qwen processor"
            ) from exc
        processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise ValueError(f"{path}: processor does not expose a tokenizer")
        image_pad = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if image_pad is None or image_pad == getattr(tokenizer, "unk_token_id", None):
            raise ValueError(f"{path}: tokenizer does not recognise <|image_pad|>")
        self._cost_processor = processor
        self._cost_tokenizer = tokenizer
        self._image_pad_token_id = int(image_pad)

    def _sample_sequence_tokens(self, sample: dict[str, Any], n_images: int) -> int:
        """Return the final Qwen input length without decoding referenced images."""
        if self._cost_processor is None or self._cost_tokenizer is None:
            # Legacy fallback. This is still a full-sequence estimate (unlike the
            # old text-only value), but chars/4 remains only a proxy for Qwen text.
            return self._sample_text_tokens(sample) + n_images * self.image_token_cost

        turns = _normalize_turns(sample.get("messages", ()))
        messages = build_vlm_chat_messages(turns, n_images)
        rendered = self._cost_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        ids = self._cost_tokenizer.encode(rendered, add_special_tokens=False)
        image_pad_count = ids.count(self._image_pad_token_id)
        if image_pad_count != n_images:
            raise ValueError(
                "Qwen sequence-cost template/image mismatch for "
                f"sample {sample.get('id', '?')!r}: expected {n_images} image pads, "
                f"got {image_pad_count}"
            )
        return len(ids) + n_images * (self.image_token_cost - 1)

    @staticmethod
    def _sample_n_images(sample: dict[str, Any]) -> int:
        """Image count for a reference sample.

        Order: explicit ``n_images`` -> ``images_ref.n`` -> the number of ``<image>``
        markers in the chat. The EO-Data1.5M ``eo/`` subsets (qa-multiview, free_chat,
        ...) carry NO count field, but their parquet cell is a list whose length equals
        the ``<image>`` marker count, so the markers are the correct, decode-free count.
        """
        n = sample.get("n_images")
        if n is None:
            n = (sample.get("images_ref") or {}).get("n")
        if n is None:
            n = _count_image_markers(sample.get("messages"))
        return int(n or 0)

    @staticmethod
    def _sample_text_tokens(sample: dict[str, Any]) -> int:
        """Text token count for a sample, without running a tokenizer."""
        text_tokens = int(sample.get("token_len", 0) or 0)
        if not text_tokens:
            # EO JSONLs omit token_len -> approximate text from the chat length
            # (~4 chars/token) so the guard still counts long open-ended answers.
            chars = sum(
                len(_message_text(m.get("content", m.get("value", ""))))
                for m in (sample.get("messages") or [])
                if isinstance(m, dict)
            )
            text_tokens = chars // 4
        return text_tokens

    def _estimate_multimodal_tokens(self, sample: dict[str, Any]) -> int:
        """Cheap per-sample budget estimate: text token_len + vision tokens.

        Vision tokens ~= n_images * (image_size / stride)^2 (square resize -> a fixed
        grid; Qwen patch14 x merge2 => stride ~= 28). Uses the JSONL ``token_len`` for
        text (0 when absent), so no tokenizer / image decode is needed at scan time.
        """
        text_tokens = self._sample_text_tokens(sample)
        n = self._sample_n_images(sample)
        size = self.image_size or 0
        per_image = (max(1, size // self.vision_token_stride) ** 2) if size else 0
        return text_tokens + n * per_image

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> dict[str, str]:
        path, offset = self._records[index]
        handle = self._handles.get(path)
        if handle is None or handle.closed:
            handle = path.open("rb")
            self._handles[path] = handle
            if len(self._handles) > 8:
                _, old_handle = self._handles.popitem(last=False)
                old_handle.close()
        handle.seek(offset)
        sample = json.loads(handle.readline())
        if self.image_mode:
            return self._vqa_image_item(sample)
        return {"text": _extract_text(sample, self.text_field)}

    def _vqa_image_item(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Load the referenced image(s) + the full (multi-turn) chat for supervised VQA.

        Returns ``vqa_images`` as a fixed ``[max_images, C, H, W]`` uint8 tensor
        (zero-padded, default-collate-safe) + the real count ``vqa_n_images``, and
        ``vqa_chat`` = a JSON list of ``{role, text}`` turns (user turns KEEP their
        ``<image>`` markers so the policy can interleave images per turn). Multi-turn
        dialogues are preserved; every assistant turn is supervised downstream.
        """
        if self.image_mode == "extract":
            images, n = self._load_extracted_images(
                sample.get("images") or (), str(sample.get("id", "?"))
            )
        else:
            images, n = self._load_reference_images(sample.get("images_ref") or {})
        turns = _normalize_turns(sample.get("messages", ()))
        return {
            "vqa_images": images,
            "vqa_n_images": n,
            "vqa_chat": json.dumps(turns),
        }

    def _stack_images(self, images):
        """Resize decoded PIL images -> (uint8 [max_images, C, H, W], real_count)."""
        import numpy as np
        import torch

        size = self.image_size or 256
        tensors = []
        for image in images:
            if self.image_size:
                if self.image_resize_mode == "resize_with_pad":
                    image = _resize_image_with_pad(image, size)
                else:
                    image = image.resize((size, size))
            array = np.array(image, dtype=np.uint8)  # HWC (copy -> writable)
            tensors.append(
                torch.from_numpy(array).permute(2, 0, 1).contiguous()
            )  # CHW uint8

        n = len(tensors)
        if n == 0:
            raise ValueError("image_mode VLM sample resolved to zero images")
        c, h, w = tensors[0].shape
        stacked = torch.zeros(
            self.max_images, c, h, w, dtype=torch.uint8
        )  # zero-pad to fixed N
        for i, tensor in enumerate(tensors):
            stacked[i] = tensor
        return stacked, n

    def _load_extracted_images(self, relatives, sample_id: str = "?"):
        """Decode on-disk image files -> (uint8 [max_images, C, H, W], real_count).

        ``extract`` corpora write each image once under the dataset root and carry
        root-relative paths, so there is no parquet cell and no Arrow pool to hand
        back (see :meth:`_load_reference_images`).

        A missing file raises rather than yielding a short sample: the JSONL row and
        its ``<image>`` markers already promised that image to the chat template.
        """
        from PIL import Image

        images = []
        for relative in list(relatives)[: self.max_images]:
            path = self._resolve_image_path(relative, sample_id)
            if not path.is_file():
                raise FileNotFoundError(
                    f"VLM sample {sample_id!r} references a missing image: {relative}"
                )
            images.append(Image.open(path).convert("RGB"))
        return self._stack_images(images)

    def _resolve_image_path(self, relative: Any, sample_id: str) -> Path:
        """Resolve a root-relative image path, refusing anything outside the root.

        The paths come from the corpus, not from us, so ``..`` and absolute paths
        would let a JSONL shard read arbitrary files off the host.
        """
        candidate = (self.root / str(relative)).resolve()
        root = self.root.resolve()
        if not candidate.is_relative_to(root):
            raise ValueError(
                f"VLM sample {sample_id!r} image path escapes the dataset root: {relative}"
            )
        return candidate

    def _parquet(self, path: str):
        pf = self._parquet_cache.get(path)
        if pf is None:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            self._parquet_cache[path] = pf
            if len(self._parquet_cache) > 4:
                self._parquet_cache.popitem(last=False)
        else:
            self._parquet_cache.move_to_end(path)
        return pf

    def _load_reference_images(self, ref: dict[str, Any]):
        """Decode all referenced images -> (uint8 [max_images, C, H, W], real_count)."""
        import io

        import pyarrow as pa
        from PIL import Image

        parquet = ref.get("parquet")
        row_group = int(ref.get("row_group", 0))
        row = int(ref.get("row", 0))
        if not parquet:
            raise ValueError("image_mode VLM sample is missing images_ref.parquet")
        if self.parquet_path_remap:
            parquet = parquet.replace(*self.parquet_path_remap)
        key = f"{parquet}::{row_group}"
        table = self._row_group_cache.get(key)
        if table is None:
            table = self._parquet(parquet).read_row_group(
                row_group, columns=[self.image_column]
            )
            # Keep only the most recent row group (streams read rows in-order per group).
            self._row_group_cache = {key: table}
            # Hand the freed buffers back to the OS. Arrow's default pool is
            # mimalloc, which keeps what it frees, and this read hands it a
            # different multi-MB size every time: the EO source stores 100 rows and
            # 0.5-75 MB of image bytes per group (p50 10.5, p90 33.2 over the 2,283
            # groups it references), while the sampler draws uniformly over 1.4M
            # records -- so the cache above never hits (0 hits in 2,100 measured
            # draws) and every sample allocates a whole group to serve one row.
            # The retained total is not one block but an accumulation across the
            # differing sizes: reading one group over and over is flat
            # (-0.18 KB/read), reading different groups out of a *single* file with
            # one ParquetFile open for the whole run still costs +456 MB, four times
            # that file's largest group. So this is not handle churn either --
            # growing the cache in _parquet() to 256 entries does not help (+672 MB).
            # Measured over 1,200-1,500 random reads with nothing reclaiming, as a
            # dataloader worker runs: +215 to +631 MB without this line, +7.5 to
            # +19 MB with it. 16 VLM workers x 8 ranks share one node, which is how
            # a flat-looking main process still ran a node out of memory
            # (EIP 7370, OOMKilled at step 44760).
            # Switching the pool to glibc instead does NOT work (+266 MB): glibc
            # retains the blocks too. The memory has to be handed back explicitly.
            pa.default_memory_pool().release_unused()
        cell = table.column(self.image_column)[
            row
        ].as_py()  # list[{"bytes"}] / {"bytes"} / bytes
        entries = list(cell) if isinstance(cell, (list, tuple)) else [cell]
        entries = entries[: self.max_images]

        images = []
        for entry in entries:
            image_bytes = entry["bytes"] if isinstance(entry, dict) else entry
            images.append(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        return self._stack_images(images)

    def _accept(self, sample: dict[str, Any]) -> bool:
        if any(sample.get(flag) for flag in self.exclude_flags):
            return False
        split = sample.get("split")
        if split is not None and self.splits and split not in self.splits:
            return False
        capability = sample.get("capability")
        if self.capabilities and capability not in self.capabilities:
            return False
        subset = sample.get("subset")
        if self.subsets and subset not in self.subsets:
            return False
        if (
            self.max_token_len is not None
            and sample.get("token_len", 0) > self.max_token_len
        ):
            return False
        # has_action matters only for the (unimplemented) action-fusion path; image VQA
        # uses image+text and ignores the action array, so don't filter it out here.
        if (
            not self.image_mode
            and not self.include_action
            and sample.get("has_action", False)
        ):
            return False
        if not self.include_video and sample.get("is_video", False):
            return False
        if self.image_mode:
            # Need image(s); accept 1..max_images (skip huge multi-image).
            n = self._sample_n_images(sample)
            if n < 1 or n > self.max_images:
                return False
            located = (
                sample.get("images")
                if self.image_mode == "extract"
                else sample.get("images_ref")
            )
            if not located:
                return False
        return True


def _resize_image_with_pad(image, output_size: int):
    """Fit an RGB PIL image in the configured square and center-pad it."""
    from PIL import Image

    if output_size <= 0:
        raise ValueError(f"output_size must be positive, got {output_size}")
    fit_size = output_size
    width, height = image.size
    scale = min(fit_size / height, fit_size / width)
    resized = image.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        resample=getattr(Image, "Resampling", Image).BILINEAR,
    )
    canvas = Image.new("RGB", (output_size, output_size), (0, 0, 0))
    canvas.paste(
        resized,
        ((output_size - resized.width) // 2, (output_size - resized.height) // 2),
    )
    return canvas


def _resolve_shards(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    index_path = root / "meta" / "index.jsonl"
    if index_path.is_file():
        paths = []
        with index_path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    relative = json.loads(line).get("path")
                    if relative:
                        paths.append(root / relative)
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"VLM index references missing shards: {missing[:3]}"
            )
        return paths
    paths = sorted(path for path in root.rglob("*.jsonl") if "meta" not in path.parts)
    if not paths:
        raise FileNotFoundError(f"no JSONL files found under {root}")
    return paths


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            str(item.get("text", ""))
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _count_image_markers(messages: Any) -> int:
    """Count ``<image>`` markers across a chat (content may be str or list)."""
    total = 0
    for message in messages or ():
        if not isinstance(message, dict):
            continue
        content = message.get("content", message.get("value", ""))
        if isinstance(content, str):
            total += content.count("<image>")
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    total += str(item.get("text", "")).count("<image>")
    return total


def _normalize_turns(messages: Any) -> list[dict[str, str]]:
    """Normalize a chat into ``[{role, text}]`` turns (role in {user, assistant}).

    User turns KEEP their ``<image>`` markers (the policy interleaves images at those
    positions); assistant turns are stripped of any ``<image>``. Consecutive same-role
    turns are merged so the result strictly alternates where possible.
    """
    turns: list[dict[str, str]] = []
    for message in messages or ():
        if not isinstance(message, dict):
            continue
        raw_role = message.get("role") or message.get("from")
        text = _message_text(message.get("content", message.get("value", "")))
        if raw_role in ("user", "human"):
            role, text = "user", text.strip()
        elif raw_role in ("assistant", "gpt"):
            role, text = "assistant", text.replace("<image>", " ").strip()
        else:
            continue
        if turns and turns[-1]["role"] == role:  # merge consecutive same-role turns
            turns[-1]["text"] = (turns[-1]["text"] + "\n" + text).strip()
        else:
            turns.append({"role": role, "text": text})
    return turns


def _extract_text(sample: dict[str, Any], text_field: str | None = None) -> str:
    if text_field:
        return str(sample.get(text_field, "")).strip()
    for field in ("text", "caption", "prompt"):
        value = sample.get(field)
        if isinstance(value, str) and value.strip():
            return value.replace("<image>", " ").strip()
    parts: list[str] = []
    for message in sample.get("messages", ()):
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            parts.extend(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict) and item.get("type") == "text"
            )
    return "\n".join(parts).replace("<image>", " ").strip()


__all__ = ["VlmJsonlBackend", "VlmJsonlDataset"]
