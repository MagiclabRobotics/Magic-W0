"""Preflight user-provided resources before allocating a training model."""

import importlib
from pathlib import Path
import sys
from magic_w0.models import get_model_spec


def check_resources(recipe, *, check_imports=False, verbose=True):
    config = (
        get_model_spec(recipe["model"]["name"])
        .config_cls()
        .from_dict(recipe["model"].get("params", {}))
    )
    required = {"Qwen processor/config": Path(config.vlm_checkpoint)}
    init = recipe["model"].get("checkpoint")
    if init:
        required["initial policy checkpoint"] = Path(init)
    if config.enable_world_teacher_supervision:
        teacher = Path(config.track4world_project_root)
        required.update(
            {
                "DINOv3 directory": Path(config.dinov3_checkpoint),
                "Track4World source": teacher,
                "Track4World config": teacher / config.track4world_config_path,
                "Track4World checkpoint": Path(config.track4world_ckpt),
                "DA3 directory": Path(config.track4world_da3_backbone_path),
                "ConvNeXt weights": Path(config.track4world_convnext_tiny_weights_path),
            }
        )
    if config.use_fast_action_tokens:
        fast_root = Path(config.fast_tokenizer_path).expanduser()
        fast_bpe = fast_root / "bpe_tokenizer/tokenizer.json"
        if not fast_bpe.is_file():
            fast_bpe = fast_root / "tokenizer.json"
        required.update(
            {
                "FAST tokenizer": fast_root,
                "FAST processor config": fast_root / "processor_config.json",
                "FAST BPE tokenizer": fast_bpe,
            }
        )
    directories = {
        "Qwen processor/config",
        "DINOv3 directory",
        "Track4World source",
        "DA3 directory",
        "FAST tokenizer",
    }
    missing = []
    for label, path in required.items():
        path = path.expanduser()
        valid = (
            path.is_dir()
            if label in directories
            else path.is_file() and path.stat().st_size > 0
        )
        if verbose:
            print(f"{'OK' if valid else 'MISSING'} {label}: {path}")
        if not valid:
            missing.append(label)
    if missing:
        raise FileNotFoundError("Provide required resources: " + ", ".join(missing))
    config.validate_checkpoint_layout()
    if (
        not init
        and not any(Path(config.vlm_checkpoint).glob("*.safetensors"))
        and not any(Path(config.vlm_checkpoint).glob("pytorch_model*.bin"))
    ):
        raise FileNotFoundError(
            "Fresh pretraining requires Qwen base weights in vlm_checkpoint"
        )
    if check_imports and config.use_fast_action_tokens:
        from magic_w0.models.action_backbone.fast_action_tokenizer import (
            load_fast_action_tokenizer,
        )

        fast = load_fast_action_tokenizer(required["FAST tokenizer"])
        if verbose:
            print(f"FAST implementation/assets load OK (vocab_size={fast.vocab_size})")
    if check_imports and config.enable_world_teacher_supervision:
        sys.path.insert(0, str(Path(config.track4world_project_root).resolve()))
        cls = importlib.import_module("track4world.nets.model").Track4World
        import inspect

        parameters = inspect.signature(cls.infer_pair).parameters
        if (
            not {"return_intermediate_features", "stop_after_motion_latent"}
            <= parameters.keys()
        ):
            raise ValueError(
                "Track4World feature patch is missing; run scripts/prepare_track4world.py"
            )
        # DA3 loads this dependency dynamically while building its nested backbone.
        # Check it before allocating teacher weights.
        importlib.import_module("evo.core.trajectory")
        if verbose:
            print("Teacher feature interface and pose alignment imports OK")
