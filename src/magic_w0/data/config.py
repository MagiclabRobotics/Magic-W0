# Copyright 2026 MagicVLA authors. (Ported from MagicMix, Apache-2.0.)
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Config dataclasses for co-training data.

Standalone variant of MagicMix's ``data/config.py``. ``SubDatasetConfig.type``
selects the data backend (see :mod:`data.backends`): ``"synthetic"`` (built-in,
torch-only, for smoke tests), and ``"lerobot"``. Source-specific fields are
retained here so readers can adapt their storage format without depending on a
model implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# How a VLM sample points at its image(s). ``reference`` = bytes in a parquet cell
# named by ``images_ref``; ``extract`` = files on disk named by root-relative
# ``images``. ``None`` is the text-only path and loads no image at all.
IMAGE_MODES = ("reference", "extract")


@dataclass
class SubDatasetConfig:
    """One source in the co-training mix.

    Each source is loaded as an independent (homogeneous) dataset and
    interleaved at the batch level -- so sources may have completely different
    feature schemas (e.g. robot actions vs. image-caption pairs).
    """

    repo_id: str | None = (
        None  # dataset id/path for external backends; None for "synthetic"
    )
    type: str = "synthetic"  # discriminator selecting the backend (see data.backends)
    modality: str = "robotics"  # routes to the matching loss handler in the policy
    weight: float = 1.0  # relative sampling probability
    root: str | None = None
    episodes: list[int] | None = None
    # Optional LeRobot task selector. v2.1 accepts either the integer
    # task_index from meta/tasks.jsonl or its exact task text. The backend
    # resolves it to an episode subset before building the frame index.
    task_filter: str | int | None = None
    batch_size: int | None = (
        None  # per-source override; falls back to the global batch_size
    )
    delta_timestamps: dict[str, list[float]] | None = None
    # Temporal observations. The reader obtains fps from LeRobot meta/info.json
    # and converts this wall-clock interval to an integer frame stride.
    obs_steps: int = 1
    obs_interval_seconds: float = 0.1
    video_backend: str | None = (
        None  # e.g. "pyav" | "torchcodec"; backend default when None
    )
    # Per-source version for a future lerobot backend (None -> inherit the global).
    lerobot_version: str | None = None
    # Backend-specific knobs (e.g. the SyntheticBackend field spec). Kept generic
    # so backends can read what they need without new top-level fields each time.
    params: dict[str, Any] | None = None


@dataclass
class VlmJsonlSourceConfig:
    """One VLM-SFT source (sharded JSONL of Qwen-VL chat samples).

    Interleaved at the batch level alongside robot sources. The concrete
    backend for this source type is deferred (it needs the Qwen processor);
    the config is carried so the mix can already express it.
    """

    root: str  # the pipeline `out/` directory
    type: str = "vlm_jsonl"
    modality: str = "vlm_sft"  # the per-batch tag routed to the VLM loss
    weight: float = 1.0
    qwen_model: str | None = None  # retained for non-PI0.5 processors
    capabilities: list[str] | None = None  # None -> all
    subsets: list[str] | None = None  # None -> all
    splits: tuple[str, ...] = ("train",)  # never include "test"/held-out by default
    max_token_len: int | None = None  # drop samples whose `token_len` exceeds this
    include_action: bool = False  # skip has_action samples (need action fusion)
    include_video: bool = False  # skip is_video samples
    # How samples point at their images. ``reference`` = bytes in a parquet cell named
    # by each row's ``images_ref``; ``extract`` = JPEG/PNG files on disk named by each
    # row's root-relative ``images`` list (what the pipeline's extract writer emits).
    # None keeps the text-only path, which loads no image at all.
    image_mode: str | None = None
    image_column: str = "image"  # parquet image column for `reference` mode
    image_size: int | None = 256  # final canvas; controlled by image_resize_mode
    # ``square`` preserves the legacy behavior; ``resize_with_pad`` preserves
    # aspect ratio and center zero-pads the configured square canvas.
    image_resize_mode: str = "square"
    max_images: int = (
        8  # cap images/sample (multi-image interleave/multiview); skip beyond
    )
    # Per-sample final-Qwen-sequence guard (bounds VLM peak memory / batch length).
    # With ``sequence_length_processor`` this is the exact input_ids length after
    # image placeholders expand into vision tokens. A sample exceeding it is SKIPPED
    # at scan time (its id is logged) -- resolution is never reduced. None disables
    # the guard.
    max_multimodal_tokens: int | None = None
    # Pixels per vision token. 32 = Qwen3.5's patch16 x merge2, measured against the
    # checkpoint's own processor: a 256x256 image yields grid 16x16 = 256 patches,
    # /merge^2 = 64 tokens, and (256/32)^2 = 64. (The old 28 assumed patch14.)
    vision_token_stride: int = 32
    batch_size: int | None = None  # per-source override
    # Cost-budgeted batching (image_mode sources only). When set, batch_size is
    # ignored and samples are packed until the batch's modelled cost would exceed
    # this budget, in token-equivalents:
    #     sum(n_images) * image_token_cost + n_samples * max(sequence_tokens)
    # ``sequence_tokens`` is the full Qwen sequence, including expanded image
    # placeholders. Vision-tower work is additive while the language model pads
    # full sequences, so a long/multi-image chat re-prices every batch sample (see
    # data.batching.MultimodalBudgetBatchSampler). None -> fixed batch_size.
    batch_token_budget: int | None = None
    # Vision tokens per image. None -> (image_size / vision_token_stride)^2, which
    # matches the measured processor output (64 for a 256px image). Override only if
    # a different checkpoint's processor disagrees.
    image_token_cost: int | None = None
    # Optional processor used at dataset-scan time to compute the exact final Qwen
    # sequence length. It must match the model checkpoint. None keeps the legacy
    # chars/4 proxy, which is unsuitable for a memory-bounded image-VLM run.
    sequence_length_processor: str | None = None
    # Hard cap on samples per batch. Backstop for whatever the cost model misses;
    # keep it low enough to actually bind.
    max_batch_samples: int = 8
    text_field: str | None = None  # explicit scalar text field; auto-detect when None
    max_samples: int | None = None  # optional bounded scan for smoke/debug runs
    # [old, new] substring rewrite applied to each images_ref.parquet absolute path
    # before opening. Used when the referenced image dataset was moved/renamed on
    # shared storage (e.g. keplerccc/ -> Robo2VLM-1/) and /pfs is read-only so a
    # symlink is not possible. None -> paths used verbatim.
    parquet_path_remap: list[str] | None = None
    # Row-level flags marking weak supervision; a row is dropped when any named field
    # is truthy. Lets a corpus ship its own quality markers (e.g. magiclabel's
    # `label_suspect`, `phase_derived`) and each run decide what to keep.
    exclude_flags: list[str] | None = None

    def __post_init__(self):
        if self.image_resize_mode not in {"square", "resize_with_pad"}:
            raise ValueError(
                "VlmJsonlSourceConfig.image_resize_mode must be 'square' or "
                f"'resize_with_pad', got {self.image_resize_mode!r}"
            )
        # Catch a misspelled mode here rather than at the first batch: an unknown
        # value used to fall through to the text-only path, silently turning an
        # image corpus into unanswerable text questions.
        if self.image_mode is not None and self.image_mode not in IMAGE_MODES:
            raise ValueError(
                f"VlmJsonlSourceConfig.image_mode must be one of {IMAGE_MODES} or None, "
                f"got {self.image_mode!r}"
            )


@dataclass
class DatasetGroupConfig:
    """A virtual source whose children are mixed at sample level.

    The group itself is interleaved with other top-level sources at batch level.
    Children must therefore share one modality and one post-transform schema.
    """

    type: str = "dataset_group"
    name: str = "dataset_group"
    modality: str = "robotics"
    weight: float = 1.0  # probability of this virtual source in the outer mix
    mixing_level: str = "sample"
    batch_size: int | None = None
    samples_per_epoch: int | None = None
    # ``weighted_replacement`` draws source/sample pairs with replacement and
    # supports arbitrary child weights. ``concat_shuffle`` uniformly shuffles the
    # physical concatenation without replacement; it therefore requires
    # dataset-size weighting, unit child weights, and a natural (uncapped) epoch.
    sampling_mode: str = "weighted_replacement"
    # ``dataset_size`` matches uniform sampling from a physical ConcatDataset;
    # ``explicit`` uses each child source's weight independent of its length.
    weight_mode: str = "explicit"
    datasets: list[Any] = field(default_factory=list)

    def __post_init__(self):
        if self.type != "dataset_group":
            raise ValueError(
                f"DatasetGroupConfig.type must be 'dataset_group', got {self.type!r}"
            )
        if self.mixing_level != "sample":
            raise ValueError(
                "dataset_group currently supports only mixing_level='sample'"
            )
        if self.sampling_mode not in {"weighted_replacement", "concat_shuffle"}:
            raise ValueError(
                f"dataset group {self.name!r} sampling_mode must be "
                "'weighted_replacement' or 'concat_shuffle'"
            )
        if self.weight_mode not in {"explicit", "dataset_size"}:
            raise ValueError(
                f"dataset group {self.name!r} weight_mode must be 'explicit' or "
                f"'dataset_size', got {self.weight_mode!r}"
            )
        if self.weight <= 0:
            raise ValueError(f"dataset group {self.name!r} weight must be positive")
        if not self.datasets:
            raise ValueError(
                f"dataset group {self.name!r} must contain at least one dataset"
            )
        if self.sampling_mode == "concat_shuffle":
            if self.weight_mode != "dataset_size":
                raise ValueError(
                    f"dataset group {self.name!r} concat_shuffle requires "
                    "weight_mode='dataset_size'"
                )
            if self.samples_per_epoch is not None:
                raise ValueError(
                    f"dataset group {self.name!r} concat_shuffle does not support "
                    "samples_per_epoch; it traverses the physical concatenation"
                )
            non_unit_weights = [
                source.weight for source in self.datasets if source.weight != 1.0
            ]
            if non_unit_weights:
                raise ValueError(
                    f"dataset group {self.name!r} concat_shuffle requires every child "
                    f"weight to be 1.0; got {sorted(set(non_unit_weights))}"
                )
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError(f"dataset group {self.name!r} batch_size must be positive")
        if self.samples_per_epoch is not None and self.samples_per_epoch <= 0:
            raise ValueError(
                f"dataset group {self.name!r} samples_per_epoch must be positive"
            )
        modalities = {source.modality for source in self.datasets}
        if modalities != {self.modality}:
            raise ValueError(
                f"dataset group {self.name!r} requires all children to use modality "
                f"{self.modality!r}; got {sorted(modalities)}"
            )
        if any(getattr(source, "batch_token_budget", None) for source in self.datasets):
            raise ValueError(
                f"dataset group {self.name!r} does not support child batch_token_budget; "
                "use a fixed shared batch size"
            )
        if any(isinstance(source, DatasetGroupConfig) for source in self.datasets):
            raise ValueError("nested dataset_group entries are not supported")


DatasetSourceConfig = SubDatasetConfig | VlmJsonlSourceConfig | DatasetGroupConfig


@dataclass
class CoTrainDatasetConfig:
    """Top-level co-training data config."""

    datasets: list[DatasetSourceConfig] = field(default_factory=list)
    lerobot_version: str = "v2.1"  # default version for a future lerobot backend
    batch_size: int = 8  # global default; overridable per source
    num_workers: int = 4
    # Optional lower worker count for the independent evaluation loader. None
    # preserves the historical behavior of using num_workers for both loaders.
    eval_num_workers: int | None = None
    # Number of deterministic replacement reads after a sample raises. Zero
    # preserves fail-fast behavior. A value of 3 means at most four total reads.
    sample_max_retries: int = 0
    # Per DataLoader worker, log the first N failures/recoveries; aggregate counts
    # still reach the trainer for every successful batch.
    sample_retry_log_limit: int = 20
    seed: int = 0  # shared across ranks (source draw + train/eval split)
    modality_key: str = "modality"
    # "batch": choose one homogeneous source for each batch (legacy co-training).
    # "sample": draw weighted sources for individual samples, then collate them
    # into one batch. Sample mixing requires all sources to share one modality and
    # a collatable post-transform schema.
    mixing_level: str = "batch"
    # Per-rank virtual epoch length for replacement-based sample mixing. None uses
    # the combined source length divided across ranks.
    samples_per_epoch: int | None = None
    # Fraction of EACH source held out for evaluation (per-source, so the eval
    # mix keeps the same modality composition as train). 0.3 = the "三成" split;
    # 0.0 disables eval. The split is deterministic (seeded by `seed`) and
    # identical on every rank, so train/eval stay disjoint under DDP.
    eval_fraction: float = 0.3
    # Independently shuffle/evaluate the full training data without holding out
    # samples. This monitors train-set forward error; it is not validation.
    eval_on_train: bool = False
    # How that fraction is cut. "episode" holds out whole trajectories, so no
    # held-out frame has its 33 ms neighbour sitting in train -- without it val
    # loss reads optimistically and cannot separate memorization from fit.
    # "sample" is the older per-frame cut, kept for sources that have no
    # trajectories at all (a VQA jsonl), which "episode" falls back to anyway.
    eval_split: str = "episode"

    def __post_init__(self):
        if not self.datasets:
            raise ValueError(
                "CoTrainDatasetConfig.datasets must list at least one source"
            )
        if not 0.0 <= self.eval_fraction < 1.0:
            raise ValueError("eval_fraction must be in [0.0, 1.0)")
        if self.eval_on_train and self.eval_fraction > 0.0:
            raise ValueError("eval_on_train requires eval_fraction=0.0")
        if self.eval_num_workers is not None and self.eval_num_workers < 0:
            raise ValueError("eval_num_workers must be non-negative when set")
        if self.sample_max_retries < 0:
            raise ValueError("sample_max_retries must be non-negative")
        if self.sample_retry_log_limit < 0:
            raise ValueError("sample_retry_log_limit must be non-negative")
        if self.eval_split not in {"episode", "sample"}:
            raise ValueError("eval_split must be 'episode' or 'sample'")
        if self.mixing_level not in {"batch", "sample"}:
            raise ValueError("mixing_level must be 'batch' or 'sample'")
        if self.samples_per_epoch is not None and self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive when set")
        for source in self.datasets:
            if source.weight <= 0:
                raise ValueError(
                    f"dataset source {source.modality!r} weight must be positive"
                )
            children = (
                source.datasets if isinstance(source, DatasetGroupConfig) else [source]
            )
            for child in children:
                if isinstance(child, SubDatasetConfig) and child.type == "lerobot":
                    child.lerobot_version = (
                        child.lerobot_version or self.lerobot_version
                    )


def source_config_from_dict(
    payload: dict[str, Any],
) -> DatasetSourceConfig | None:
    """Parse one YAML source, returning ``None`` for ``enabled: false`` entries."""
    source = dict(payload)
    enabled = source.pop("enabled", True)
    if not isinstance(enabled, bool):
        raise TypeError("dataset source enabled must be a boolean")
    if not enabled:
        return None
    if source.get("type") == "dataset_group":
        if "datasets" in source and "sources" in source:
            raise TypeError(
                "dataset_group must use either datasets or sources, not both"
            )
        child_payloads = source.pop("datasets", None)
        if child_payloads is None:
            child_payloads = source.pop("sources", None)
        if not isinstance(child_payloads, list):
            raise TypeError("dataset_group.datasets must be a list")
        children = [source_config_from_dict(child) for child in child_payloads]
        children = [child for child in children if child is not None]
        source["datasets"] = children
        config_cls = DatasetGroupConfig
    else:
        config_cls = (
            VlmJsonlSourceConfig
            if source.get("type") == "vlm_jsonl"
            else SubDatasetConfig
        )
    try:
        return config_cls(**source)
    except TypeError as exc:
        raise TypeError(
            f"invalid {source.get('type', 'synthetic')!r} dataset source: {exc}"
        ) from exc
