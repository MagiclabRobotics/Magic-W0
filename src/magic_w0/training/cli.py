"""Command-line entrypoints for framework-native distributed training."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import yaml


def _build_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps for a bounded verification run.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override training.output_dir without modifying the source YAML.",
    )
    parser.add_argument(
        "--skip-checkpoints",
        action="store_true",
        help="Disable checkpoint writes for a bounded verification run.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Restore a complete training checkpoint or run directory (strict).",
    )
    parser.add_argument(
        "--task-filter",
        "--task",
        dest="task_filter",
        default=None,
        help=(
            "Restrict the only enabled LeRobot source to one v2.1 task. "
            "Accepts a task_index (for example 30) or exact task text."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Initialize model weights with fresh optimizer state",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved recipe without allocating a GPU",
    )
    return parser


def train_main() -> None:
    """Assemble data, model, and DistributedTrainer from one YAML file."""
    args = _build_parser("MagicW0 distributed train runner.").parse_args()
    from .config import load_recipe

    payload = load_recipe(args.config)
    if args.init_checkpoint is not None and args.resume is not None:
        raise ValueError("--init-checkpoint and --resume are mutually exclusive")
    if args.init_checkpoint is not None:
        payload["model"]["checkpoint"] = str(args.init_checkpoint)
    if args.dry_run:
        from magic_w0.models import get_model_spec

        get_model_spec(payload["model"]["name"]).config_cls().from_dict(
            payload["model"].get("params", {})
        )
        from magic_w0.data.config import source_config_from_dict

        for source in payload["data"]["datasets"]:
            source_config_from_dict(source)
        _apply_train_cli_overrides(payload["training"], args)
        if payload["training"].get("resume_from"):
            payload["model"]["checkpoint"] = payload["training"]["resume_from"]
        print(yaml.safe_dump(payload, sort_keys=False))
        return
    if not isinstance(payload, dict):
        raise TypeError("training config must contain a YAML mapping")

    from magic_w0.data.collate import make_collate_factory
    from magic_w0.data.config import CoTrainDatasetConfig, source_config_from_dict
    from magic_w0.data.factory import make_interleaved_loaders
    from magic_w0.models import COTRAIN, build_data_processor, build_model

    run_config = copy.deepcopy(payload)

    from magic_w0.training.dist_trainer import (
        DistributedTrainer,
        DistributionStrategyConfig,
    )
    from magic_w0.training.dist_trainer.distribution_strategy import cleanup_distributed
    from magic_w0.training.dist_trainer.metrics_logger import WandbConfig
    from magic_w0.training.hooks import build_hooks
    from magic_w0.training.optimizer import optimizer_config_from_dict
    from magic_w0.training.scheduler import SchedulerConfig

    model_cfg = _mapping(payload, "model")
    data_cfg = _mapping(payload, "data")
    training_cfg = _mapping(payload, "training")
    _reject_unknown("top-level", payload)
    _apply_train_cli_overrides(training_cfg, args)
    if training_cfg.get("resume_from"):
        model_cfg["checkpoint"] = training_cfg["resume_from"]
        from .resume import check_resume_checkpoint

        check_resume_checkpoint(training_cfg["resume_from"], model_cfg.get("params", {}))
    _apply_data_cli_overrides(data_cfg, args)
    # Record the effective CLI-overridden training settings in W&B/run metadata.
    run_config["training"] = copy.deepcopy(training_cfg)
    run_config["data"] = copy.deepcopy(data_cfg)
    run_config["model"] = copy.deepcopy(model_cfg)

    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    from .resources import check_resources

    check_resources({"model": model_cfg}, check_imports=True, verbose=(rank == 0))
    torch.cuda.set_device(local_rank)
    from .metadata import write_run_metadata

    write_run_metadata(run_config, rank=rank)

    name = model_cfg.pop("name")
    checkpoint = model_cfg.pop("checkpoint", None)
    params = model_cfg.pop("params", {})
    processor = build_data_processor(
        name,
        model_cfg.pop("processor", None),
        model_params=params,
    )
    local_files_only = bool(model_cfg.pop("local_files_only", True))
    strict = bool(model_cfg.pop("strict", True))
    _reject_unknown("model", model_cfg)

    source_payloads = data_cfg.pop("datasets")
    if not isinstance(source_payloads, list):
        raise TypeError("config.data.datasets must be a list")
    sources = [source_config_from_dict(source) for source in source_payloads]
    sources = [source for source in sources if source is not None]
    if not sources:
        raise ValueError(
            "config.data.datasets must contain at least one enabled source"
        )
    cotrain_cfg = CoTrainDatasetConfig(datasets=sources, **data_cfg)

    # The model is built BEFORE the loaders so its CPU batch preparation can be handed to
    # the dataloader workers (see data.collate). Models that do not opt in get
    # collate_factory None -> torch's default collate -> preparation stays in the loop.
    model = build_model(
        name,
        params,
        device=f"cuda:{local_rank}",
        kind=COTRAIN,
        checkpoint=checkpoint,
        local_files_only=local_files_only,
        strict=strict,
    )

    train_loader, val_loader = make_interleaved_loaders(
        cotrain_cfg,
        rank=rank,
        world_size=world_size,
        pin_memory=bool(training_cfg.pop("pin_memory", True)),
        dataset_transform=(
            processor.transform_dataset if processor is not None else None
        ),
        collate_factory=make_collate_factory(model, cotrain_cfg.modality_key),
    )

    strategy = DistributionStrategyConfig(**training_cfg.pop("strategy", {}))
    scheduler_payload = training_cfg.pop("scheduler", None)
    scheduler = SchedulerConfig(**scheduler_payload) if scheduler_payload else None
    learning_rate = float(training_cfg.get("learning_rate", 1e-4))
    weight_decay = float(training_cfg.get("weight_decay", 1e-4))
    optimizer = optimizer_config_from_dict(
        training_cfg.pop("optimizer", None),
        default_learning_rate=learning_rate,
        default_weight_decay=weight_decay,
    )
    hooks = build_hooks(training_cfg.pop("hooks", []))
    wandb_config = WandbConfig.from_dict(training_cfg.pop("wandb", None))
    try:
        trainer = DistributedTrainer(
            model,
            strategy_config=strategy,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            scheduler_config=scheduler,
            optimizer_config=optimizer,
            hooks=hooks,
            wandb_config=wandb_config,
            run_config=run_config,
            seed=cotrain_cfg.seed,
            **training_cfg,
        )
        trainer.fit()
    finally:
        cleanup_distributed()


def _mapping(payload: dict, key: str) -> dict:
    value = payload.pop(key, None)
    if not isinstance(value, dict):
        raise TypeError(f"config.{key} must be a mapping")
    return dict(value)


def _apply_train_cli_overrides(training_cfg: dict, args: argparse.Namespace) -> None:
    if args.output_dir is not None:
        training_cfg["output_dir"] = str(args.output_dir)
    if args.resume is not None:
        from .dist_trainer.checkpoint_io import resolve_checkpoint

        training_cfg["resume_from"] = str(resolve_checkpoint(args.resume))
    if args.max_steps is not None:
        training_cfg["max_steps"] = args.max_steps
    if args.skip_checkpoints:
        training_cfg["save_every_epochs"] = None
        training_cfg["save_every_steps"] = None
        training_cfg["save_milestone_every_steps"] = None


def _apply_data_cli_overrides(data_cfg: dict, args: argparse.Namespace) -> None:
    """Apply global data overrides that are unambiguous for a single source."""

    if args.task_filter is None:
        return
    source_payloads = data_cfg.get("datasets")
    if not isinstance(source_payloads, list):
        raise TypeError("config.data.datasets must be a list")

    lerobot_sources: list[dict] = []

    def collect(sources: list) -> None:
        for source in sources:
            if not isinstance(source, dict) or source.get("enabled", True) is False:
                continue
            if source.get("type") == "dataset_group":
                children = source.get("datasets", source.get("sources", []))
                if isinstance(children, list):
                    collect(children)
            elif source.get("type") == "lerobot":
                lerobot_sources.append(source)

    collect(source_payloads)
    if len(lerobot_sources) != 1:
        raise ValueError(
            "--task-filter requires exactly one enabled LeRobot source, found "
            f"{len(lerobot_sources)}; configure task_filter per source in YAML "
            "for a multi-source run"
        )
    lerobot_sources[0]["task_filter"] = args.task_filter


def _reject_unknown(section: str, payload: dict) -> None:
    if payload:
        raise ValueError(f"unknown config.{section} keys: {sorted(payload)}")
