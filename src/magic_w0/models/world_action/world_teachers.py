"""Frozen DINOv3 and Track4World target generation for MagicW0."""

from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .processor_magicw0 import (
    GEOMETRY_3D_TARGET,
    MOTION_3D_TARGET,
    WORLD_3D_VALID_MASK,
)

TRACK4WORLD_TEACHER_SECONDS = "track4world_teacher_seconds"
TRACK4WORLD_MOTION_RETRY_ROWS = "track4world_motion_retry_rows"
TRACK4WORLD_MOTION_RECOVERED_ROWS = "track4world_motion_recovered_rows"
TRACK4WORLD_MOTION_DROPPED_ROWS = "track4world_motion_dropped_rows"
TRACK4WORLD_TENSORRT_RELEASED_GIB = "track4world_tensorrt_released_gib"
_TRACK4WORLD_TEACHER_EVENTS = "_track4world_teacher_events"

logger = logging.getLogger(__name__)


class WorldTeacherMixin:
    """Methods shared by the MagicW0 policy without registering Track4World eagerly."""

    config: Any
    dinov3_teacher: nn.Module | None

    def _prepare_dino_pixel_values(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim == 6:
            images = images[:, :, -1]
        if images.ndim != 5:
            raise ValueError(
                "future 2D images must be [B,V,C,H,W] or [B,V,T,C,H,W], got "
                f"{tuple(images.shape)}"
            )
        batch_size, views = images.shape[:2]
        input_dtype = images.dtype
        images = images.reshape(batch_size * views, *images.shape[2:])
        if images.shape[1] not in {1, 3} and images.shape[-1] in {1, 3}:
            images = images.permute(0, 3, 1, 2)
        if images.shape[1] == 1:
            images = images.expand(-1, 3, -1, -1)
        if images.shape[1] != 3:
            raise ValueError(
                f"DINOv3 expects RGB images, got {images.shape[1]} channels"
            )
        # Keep scaling a property of the input contract, never of other images
        # that happened to land in the same batch.  The old batch-global amax
        # heuristic made a very dark uint8 frame use 0..255 values when alone
        # but 0..1 values when batched with a brighter frame.  LeRobot decoders
        # emit uint8, while callers that deliberately provide floating-point
        # images must provide an already scaled [0, 1] tensor.
        if input_dtype == torch.uint8:
            images = images.float().div_(255.0)
        elif input_dtype.is_floating_point:
            # Floating-point callers own the [0, 1] contract.  Do not scan image
            # values here: besides reintroducing data-dependent control flow,
            # per-step finite/min/max reductions over the full image batch add
            # avoidable CPU work on the training input path.
            images = images.float()
        else:
            raise TypeError(
                "DINOv3 images must be uint8 in [0, 255] or floating point in "
                f"[0, 1], got dtype={input_dtype}"
            )
        expected = int(self.config.image_size)
        if tuple(images.shape[-2:]) != (expected, expected):
            raise ValueError(
                f"DINOv3 future images must be {expected}x{expected}, got "
                f"{tuple(images.shape[-2:])}"
            )
        mean = images.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = images.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        return (images - mean) / std

    def _dino_patch_targets(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if self.dinov3_teacher is None:
            raise RuntimeError("DINOv3 teacher is not initialized")
        with torch.no_grad():
            outputs = self.dinov3_teacher(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        registers = int(getattr(self.dinov3_teacher.config, "num_register_tokens", 0))
        patches = hidden[:, 1 + registers :]
        patch_size = int(getattr(self.dinov3_teacher.config, "patch_size", 16))
        input_h, input_w = pixel_values.shape[-2:]
        if input_h % patch_size or input_w % patch_size:
            raise RuntimeError("DINO input size must be divisible by its patch size")
        dense_grid = (input_h // patch_size, input_w // patch_size)
        if patches.shape[1] != dense_grid[0] * dense_grid[1]:
            raise RuntimeError(
                f"DINO produced {patches.shape[1]} patches, expected "
                f"{dense_grid[0] * dense_grid[1]}"
            )
        if dense_grid != tuple(self.config.gen_2d_grid_size):
            raise RuntimeError(
                f"DINO grid {dense_grid} does not match MagicW0 2D grid "
                f"{self.config.gen_2d_grid_size}"
            )
        batch_size = pixel_values.shape[0] // self.config.gen_2d_max_views
        return patches.reshape(batch_size, -1, patches.shape[-1]).detach()

    def _track4world_teacher_targets(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
        valid_mask: torch.Tensor,
        device: torch.device,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        if valid_mask.ndim != 1:
            raise ValueError(
                f"Track4World valid mask must be [B], got {valid_mask.shape}"
            )
        batch_size = int(valid_mask.shape[0])
        if (
            current_images.shape[0] != batch_size
            or future_images.shape[0] != batch_size
        ):
            raise ValueError("Track4World image and validity batch sizes differ")
        valid_mask = valid_mask.bool().clone()
        valid_indices = torch.nonzero(valid_mask, as_tuple=False).flatten()
        target_shape = (batch_size, self.config.gen_3d_num_query_tokens, 1024)
        if valid_indices.numel() == 0:
            dtype = self._torch_dtype(self.config.dtype)
            return {
                GEOMETRY_3D_TARGET: torch.zeros(
                    target_shape, dtype=dtype, device=device
                ),
                MOTION_3D_TARGET: torch.zeros(target_shape, dtype=dtype, device=device),
                WORLD_3D_VALID_MASK: valid_mask.to(device),
                TRACK4WORLD_TEACHER_SECONDS: time.perf_counter() - started,
                TRACK4WORLD_MOTION_RETRY_ROWS: 0,
                TRACK4WORLD_MOTION_RECOVERED_ROWS: 0,
                TRACK4WORLD_MOTION_DROPPED_ROWS: 0,
            }

        current = current_images.index_select(0, valid_indices)
        future = future_images.index_select(0, valid_indices)
        # DISABLED: CUDA Events for memory leak investigation
        # teacher_events = None
        # if device.type == "cuda":
        #     teacher_events = (
        #         torch.cuda.Event(enable_timing=True),
        #         torch.cuda.Event(enable_timing=True),
        #     )
        #     teacher_events[0].record()
        frame_pair = self._prepare_track4world_frame_pair(current, future).to(device)
        teacher = self._get_track4world_teacher(device)
        outputs = self._run_track4world_teacher_pair(teacher, frame_pair, device)
        valid_geometry = self._compress_geometry_target(
            outputs["geometry_latent_t"].detach()
        )
        valid_motion = self._compress_motion_target(
            outputs["motion_latent_t_to_target"].detach()
        )
        expected_valid = (
            int(valid_indices.numel()),
            self.config.gen_3d_num_query_tokens,
            1024,
        )
        if (
            valid_geometry.shape != expected_valid
            or valid_motion.shape != expected_valid
        ):
            raise RuntimeError(
                "Track4World compressed target mismatch: "
                f"geometry={tuple(valid_geometry.shape)}, motion={tuple(valid_motion.shape)}, "
                f"expected={expected_valid}"
            )
        bad_motion_rows = torch.nonzero(
            ~torch.isfinite(valid_motion).flatten(1).all(dim=1),
            as_tuple=False,
        ).flatten()
        retry_rows = int(bad_motion_rows.numel())
        recovered_rows = 0
        dropped_rows = 0
        if retry_rows:
            packed_rows = [int(row) for row in bad_motion_rows.detach().cpu().tolist()]
            original_rows = [
                int(valid_indices[row].detach().cpu().item()) for row in packed_rows
            ]
            logger.warning(
                "Track4World produced non-finite motion targets for batch rows %s; "
                "retrying each row with B=1",
                original_rows,
            )
            # Compression is view-like. Clone before dropping the full-batch output
            # so recovered rows no longer retain the poisoned latent allocation.
            valid_motion = valid_motion.clone()
            del outputs
            for packed_row, original_row in zip(
                packed_rows, original_rows, strict=True
            ):
                retry_motion = None
                try:
                    retry_outputs = self._run_track4world_teacher_pair(
                        teacher,
                        frame_pair[packed_row : packed_row + 1],
                        device,
                    )
                    retry_motion = self._compress_motion_target(
                        retry_outputs["motion_latent_t_to_target"].detach()
                    )
                    retry_finite = bool(torch.isfinite(retry_motion).all().item())
                except Exception:  # keep every DDP rank progressing
                    retry_finite = False
                    logger.exception(
                        "Track4World B=1 retry raised for batch row %d; "
                        "dropping this row's 3D supervision",
                        original_row,
                    )
                if retry_finite and retry_motion is not None:
                    valid_motion[packed_row].copy_(retry_motion[0])
                    recovered_rows += 1
                else:
                    # Masking alone is insufficient because NaN * False is NaN.
                    valid_motion[packed_row].zero_()
                    valid_mask[original_row] = False
                    dropped_rows += 1
            logger.warning(
                "Track4World B=1 motion retry finished: requested=%d recovered=%d "
                "dropped=%d original_batch_rows=%s",
                retry_rows,
                recovered_rows,
                dropped_rows,
                original_rows,
            )
        geometry = valid_geometry.new_zeros(target_shape)
        motion = valid_motion.new_zeros(target_shape)
        geometry.index_copy_(0, valid_indices, valid_geometry)
        motion.index_copy_(0, valid_indices, valid_motion)
        targets = {
            GEOMETRY_3D_TARGET: geometry.detach(),
            MOTION_3D_TARGET: motion.detach(),
            WORLD_3D_VALID_MASK: valid_mask.to(device),
            TRACK4WORLD_MOTION_RETRY_ROWS: retry_rows,
            TRACK4WORLD_MOTION_RECOVERED_ROWS: recovered_rows,
            TRACK4WORLD_MOTION_DROPPED_ROWS: dropped_rows,
        }
        release_execution_memory = getattr(teacher, "release_execution_memory", None)
        if callable(release_execution_memory):
            released_bytes = int(release_execution_memory())
            targets[TRACK4WORLD_TENSORRT_RELEASED_GIB] = released_bytes / 2**30
        # DISABLED: CUDA Events for memory leak investigation
        # Always use CPU time now instead of GPU Events
        # if teacher_events is None:
        #     targets[TRACK4WORLD_TEACHER_SECONDS] = time.perf_counter() - started
        # else:
        #     teacher_events[1].record()
        #     targets[_TRACK4WORLD_TEACHER_EVENTS] = teacher_events
        targets[TRACK4WORLD_TEACHER_SECONDS] = time.perf_counter() - started
        return targets

    def _run_track4world_teacher_pair(
        self,
        teacher: nn.Module,
        frame_pair: torch.Tensor,
        device: torch.device,
    ) -> dict[str, Any]:
        autocast_dtype = self._track4world_autocast_dtype()
        autocast = (
            torch.autocast(device_type="cuda", dtype=autocast_dtype)
            if device.type == "cuda" and autocast_dtype is not torch.float32
            else nullcontext()
        )
        with torch.no_grad(), autocast:
            return teacher.infer_pair(
                frame_pair,
                iters=self.config.track4world_inference_iters,
                sw=None,
                is_training=False,
                tracking3d=True,
                force_projection=True,
                return_intermediate_features=True,
                stop_after_motion_latent=True,
                profile_intermediate_features=False,
            )

    def _get_track4world_teacher(self, device: torch.device) -> nn.Module:
        if not self.config.enable_world_teacher_supervision:
            raise RuntimeError(
                "Track4World teacher supervision is disabled for inference"
            )
        teacher = getattr(self, "_track4world_teacher", None)
        if teacher is not None:
            return teacher
        root = _resolve_magicw0_path(self.config.track4world_project_root)
        if self.config.track4world_inference_backend == "tensorrt":
            from .track4world_tensorrt import Track4WorldTensorRT

            teacher = Track4WorldTensorRT(
                str(
                    _resolve_magicw0_path(self.config.track4world_tensorrt_engine_dir)
                ),
                str(root),
                keep_execution_memory=(
                    self.config.track4world_tensorrt_keep_execution_memory
                ),
                expected_engine_batch_sizes=(
                    self.config.track4world_tensorrt_expected_engine_batch_sizes
                ),
            )
            object.__setattr__(self, "_track4world_teacher", teacher)
            return teacher
        config_path = _resolve_under_root(root, self.config.track4world_config_path)
        checkpoint_path = _resolve_magicw0_path(self.config.track4world_ckpt)
        backbone_path = _resolve_magicw0_path(
            self.config.track4world_da3_backbone_path
        )
        convnext_path = _resolve_magicw0_path(
            self.config.track4world_convnext_tiny_weights_path
        )
        if self.config.track4world_local_files_only:
            missing = [
                str(path)
                for path in (config_path, checkpoint_path, backbone_path, convnext_path)
                if not path.exists()
            ]
            if missing:
                raise FileNotFoundError(f"missing local Track4World assets: {missing}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from track4world.nets.model import Track4World

        with config_path.open("r", encoding="utf-8") as handle:
            track_config = json.load(handle)
        teacher = Track4World(
            **_track4world_constructor_kwargs(
                track_config,
                backbone_path=backbone_path,
                local_files_only=self.config.track4world_local_files_only,
                convnext_path=convnext_path,
                motion_chunk_size=self.config.track4world_motion_chunk_size,
            )
        )
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        teacher.load_pretrained_with_remap(state_dict)
        teacher.to(device).requires_grad_(False).eval()
        # Lazy teacher should not enter optimizer/checkpoint/DDP parameter scans.
        object.__setattr__(self, "_track4world_teacher", teacher)
        return teacher

    def _prepare_track4world_frame_pair(
        self,
        current_images: torch.Tensor,
        future_images: torch.Tensor,
    ) -> torch.Tensor:
        current = _as_chw_batch(current_images)
        future = _as_chw_batch(future_images)
        if current.shape != future.shape:
            raise ValueError("Track4World current/future image shapes must match")
        images = torch.stack((current, future), dim=1).float()
        batch, pair, channels, height, width = images.shape
        if channels == 1:
            images = images.expand(-1, -1, 3, -1, -1)
        if images.shape[2] != 3:
            raise ValueError("Track4World expects RGB images")
        if images.numel() and images.detach().amax() <= 2.0:
            images = images * 255.0
        if self.config.track4world_preprocess_mode == "already_letterboxed_256":
            expected = self.config.track4world_image_size
            if (height, width) != (expected, expected):
                raise ValueError(
                    f"Track4World expects {expected}x{expected}, got {(height, width)}"
                )
            return images

        scale = min(
            self.config.track4world_image_size / height,
            self.config.track4world_image_size / width,
        )
        if self.config.track4world_preprocess_mode == "demo_resize":
            output_h = max(
                int(height * scale)
                // self.config.track4world_multiple
                * self.config.track4world_multiple,
                self.config.track4world_multiple,
            )
            output_w = max(
                int(width * scale)
                // self.config.track4world_multiple
                * self.config.track4world_multiple,
                self.config.track4world_multiple,
            )
            flat = images.reshape(batch * pair, *images.shape[2:])
            resized = F.interpolate(
                flat, size=(output_h, output_w), mode="bilinear", align_corners=False
            )
            return resized.reshape(batch, pair, 3, output_h, output_w)

        resized_h, resized_w = round(height * scale), round(width * scale)
        model_h = _ceil_to_multiple(resized_h, self.config.track4world_multiple)
        model_w = _ceil_to_multiple(resized_w, self.config.track4world_multiple)
        flat = images.reshape(batch * pair, *images.shape[2:])
        resized = F.interpolate(
            flat, size=(resized_h, resized_w), mode="bilinear", align_corners=False
        )
        pad_h, pad_w = model_h - resized_h, model_w - resized_w
        padded = F.pad(
            resized,
            (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        )
        return padded.reshape(batch, pair, 3, model_h, model_w)

    def _track4world_autocast_dtype(self) -> torch.dtype:
        return {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }[self.config.track4world_autocast_dtype]

    def _compress_geometry_target(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or latent.shape[1:] != (324, 1024):
            raise RuntimeError(
                f"Track4World geometry must be [B,324,1024], got {tuple(latent.shape)}"
            )
        grid = latent.reshape(latent.shape[0], 18, 18, 1024).permute(0, 3, 1, 2)
        pooled = F.interpolate(
            grid.float(), size=self.config.gen_3d_grid_size, mode="area"
        ).to(grid.dtype)
        return pooled.flatten(2).transpose(1, 2).detach()

    def _compress_motion_target(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1:] != (256, 32, 32):
            raise RuntimeError(
                f"Track4World motion must be [B,256,32,32], got {tuple(latent.shape)}"
            )
        grid = latent.permute(0, 2, 3, 1)
        target_h, target_w = self.config.gen_3d_grid_size
        if grid.shape[1:3] != (target_h * 2, target_w * 2):
            raise RuntimeError("Track4World motion grid must be 2x the 3D query grid")
        batch, _, _, channels = grid.shape
        return (
            grid.reshape(batch, target_h, 2, target_w, 2, channels)
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(batch, target_h * target_w, 4 * channels)
            .detach()
        )


def _as_chw_batch(images: torch.Tensor) -> torch.Tensor:
    if images.ndim == 5:
        images = images[:, -1]
    if images.ndim != 4:
        raise ValueError(f"expected [B,C,H,W] images, got {tuple(images.shape)}")
    if images.shape[1] not in {1, 3} and images.shape[-1] in {1, 3}:
        images = images.permute(0, 3, 1, 2)
    return images


def _ceil_to_multiple(value: int, multiple: int) -> int:
    return ((int(value) + multiple - 1) // multiple) * multiple


def _resolve_under_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def _resolve_magicw0_path(value: str | Path) -> Path:
    """Resolve portable config paths relative to the MagicW0 repository root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    repo_root = Path(__file__).resolve().parents[4]
    return (repo_root / path).resolve()


def _track4world_constructor_kwargs(
    track_config: dict[str, Any],
    *,
    backbone_path: Path,
    local_files_only: bool,
    convnext_path: Path,
    motion_chunk_size: int,
) -> dict[str, Any]:
    """Merge the external JSON with MagicW0-owned runtime overrides.

    Track4World's v1.json may already carry DA3 path/offline keys. Passing that
    mapping with the same explicit keyword arguments raises ``TypeError`` before
    the teacher is constructed. Build one mapping and let the training YAML's
    validated paths override stale machine-specific values from the external JSON.
    """
    model = track_config.get("model")
    if not isinstance(model, dict):
        raise TypeError("Track4World config must contain a 'model' mapping")
    kwargs = dict(model)
    kwargs.update(
        {
            "seqlen": 16,
            "use_3d": True,
            "use_model": "depthanythingv3",
            "depthanythingv3_model_path": str(backbone_path),
            "depthanythingv3_local_files_only": bool(local_files_only),
            "convnext_tiny_weights_path": str(convnext_path),
            "motion_chunk_size": int(motion_chunk_size),
        }
    )
    return kwargs
