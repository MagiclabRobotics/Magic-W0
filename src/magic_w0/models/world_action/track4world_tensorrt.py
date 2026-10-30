"""TensorRT-backed Track4World latent inference."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


class _TensorRTEngine:
    def __init__(self, path: Path) -> None:
        try:
            import tensorrt as trt
        except ImportError as exc:
            raise ImportError(
                "TensorRT Python bindings are required for "
                "track4world_inference_backend='tensorrt'"
            ) from exc

        if not path.is_file():
            raise FileNotFoundError(f"missing TensorRT engine: {path}")
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.ERROR)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"failed to deserialize TensorRT engine: {path}")
        self._path = path
        self._context = None
        self._context_device: torch.device | None = None
        self._workspace: _TensorRTWorkspace | None = None
        self.inputs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        self.outputs: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        for index in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(index)
            shape = tuple(int(value) for value in self._engine.get_tensor_shape(name))
            if any(value < 0 for value in shape):
                raise ValueError(
                    f"dynamic TensorRT binding is not supported: {name}={shape}"
                )
            trt_dtype = self._engine.get_tensor_dtype(name)
            dtype = {
                trt.float32: torch.float32,
                trt.float16: torch.float16,
                trt.int32: torch.int32,
                trt.int8: torch.int8,
                trt.bool: torch.bool,
            }.get(trt_dtype)
            if dtype is None:
                raise TypeError(f"unsupported TensorRT dtype for {name}: {trt_dtype}")
            destination = (
                self.inputs
                if self._engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                else self.outputs
            )
            destination[name] = (shape, dtype)

    @property
    def batch_size(self) -> int:
        shapes = [shape for shape, _ in self.inputs.values()]
        return shapes[0][0]

    def __call__(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if set(inputs) != set(self.inputs):
            raise ValueError(
                f"TensorRT inputs mismatch: got={sorted(inputs)}, "
                f"expected={sorted(self.inputs)}"
            )
        device = next(iter(inputs.values())).device
        if device.type != "cuda":
            raise ValueError("TensorRT Track4World inference requires a CUDA device")
        if self._context is None:
            if self._workspace is None:
                raise RuntimeError("TensorRT engine workspace is not configured")
            self._context = (
                self._engine.create_execution_context_without_device_memory()
            )
            if self._context is None:
                raise RuntimeError(
                    f"failed to create TensorRT context: {self._path}; "
                    "check available GPU memory"
                )
            self._context.device_memory = self._workspace.pointer(device)
            self._context_device = device
        elif self._context_device != device:
            raise RuntimeError(
                f"TensorRT context was created on {self._context_device}, got {device}"
            )
        bound_inputs: dict[str, torch.Tensor] = {}
        for name, value in inputs.items():
            expected_shape, expected_dtype = self.inputs[name]
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"TensorRT input {name} has shape {tuple(value.shape)}, "
                    f"expected {expected_shape}"
                )
            value = value.to(device=device, dtype=expected_dtype).contiguous()
            bound_inputs[name] = value
            self._context.set_tensor_address(name, value.data_ptr())
        outputs = {
            name: torch.empty(shape, dtype=dtype, device=device)
            for name, (shape, dtype) in self.outputs.items()
        }
        for name, value in outputs.items():
            self._context.set_tensor_address(name, value.data_ptr())
        stream = torch.cuda.current_stream(device).cuda_stream
        if not self._context.execute_async_v3(stream_handle=stream):
            raise RuntimeError("TensorRT execution failed")
        return outputs

    def release_context(self) -> None:
        self._context = None
        self._context_device = None


class _TensorRTWorkspace:
    def __init__(self, size_bytes: int) -> None:
        self.size_bytes = int(size_bytes)
        self._allocation: torch.Tensor | None = None
        self._device: torch.device | None = None

    def pointer(self, device: torch.device) -> int:
        if self._allocation is None:
            self._allocation = torch.empty(
                self.size_bytes, dtype=torch.uint8, device=device
            )
            self._device = device
        elif self._device != device:
            raise RuntimeError(
                f"TensorRT workspace was allocated on {self._device}, got {device}"
            )
        return self._allocation.data_ptr()

    def release(self) -> int:
        released_bytes = self.size_bytes if self._allocation is not None else 0
        self._allocation = None
        self._device = None
        return released_bytes


class Track4WorldTensorRT(nn.Module):
    """Reproduce Track4World pair inference around four TensorRT components."""

    def __init__(
        self,
        engine_dir: str | Path,
        project_root: str | Path,
        *,
        keep_execution_memory: bool = False,
        expected_engine_batch_sizes: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        engine_dir = Path(engine_dir).expanduser().resolve()
        self.project_root = Path(project_root).expanduser().resolve()
        self.keep_execution_memory = bool(keep_execution_memory)
        self.da3_anyview = _TensorRTEngine(engine_dir / "da3_anyview.engine")
        self.da3_metric = _TensorRTEngine(engine_dir / "da3_metric.engine")
        self.flow_feature = _TensorRTEngine(engine_dir / "flow_feature.engine")
        self.motion_iteration = _TensorRTEngine(engine_dir / "motion_iteration.engine")
        engines = (
            self.da3_anyview,
            self.da3_metric,
            self.flow_feature,
            self.motion_iteration,
        )
        self.engine_batch_sizes = {
            "da3_anyview": self.da3_anyview.batch_size,
            "da3_metric": self.da3_metric.batch_size,
            "flow_feature": self.flow_feature.batch_size,
            "motion_iteration": self.motion_iteration.batch_size,
        }
        if (
            expected_engine_batch_sizes is not None
            and self.engine_batch_sizes != expected_engine_batch_sizes
        ):
            raise ValueError(
                "TensorRT engine batch sizes do not match configuration: "
                f"actual={self.engine_batch_sizes}, "
                f"expected={expected_engine_batch_sizes}"
            )
        self.engine_batch_size = max(self.engine_batch_sizes.values())
        self.workspace = _TensorRTWorkspace(
            max(engine._engine.device_memory_size_v2 for engine in engines)
        )
        for engine in engines:
            engine._workspace = self.workspace

    @torch.inference_mode()
    def release_execution_memory(self) -> int:
        """Release transient TensorRT context/workspace memory before backward."""
        if self.keep_execution_memory:
            return 0
        allocation = self.workspace._allocation
        if allocation is None:
            return 0
        torch.cuda.current_stream(allocation.device).synchronize()
        for engine in (
            self.da3_anyview,
            self.da3_metric,
            self.flow_feature,
            self.motion_iteration,
        ):
            engine.release_context()
        return self.workspace.release()

    @torch.inference_mode()
    def diagnostic_motion_iteration(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run one motion engine call for same-input numerical attribution."""
        return self._run_padded(self.motion_iteration, **inputs)

    @torch.inference_mode()
    def diagnostic_flow_feature(
        self, inputs: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Run the flow engine once for same-input numerical attribution."""
        return self._run_padded(self.flow_feature, **inputs)

    @torch.inference_mode()
    def infer_pair(
        self,
        images: torch.Tensor,
        *,
        iters: int = 4,
        return_intermediate_features: bool = False,
        stop_after_motion_latent: bool = False,
        tracking3d: bool = False,
        diagnostic_intermediates: bool = False,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        if not return_intermediate_features or not stop_after_motion_latent:
            raise ValueError(
                "TensorRT Track4World supports only intermediate latent inference"
            )
        if not tracking3d:
            raise ValueError("TensorRT Track4World requires tracking3d=True")
        if images.ndim != 5 or images.shape[1:] != (2, 3, 256, 256):
            raise ValueError(
                "TensorRT Track4World expects [B,2,3,256,256], got "
                f"{tuple(images.shape)}"
            )
        batch_size = images.shape[0]

        normalized = images.to(torch.float16).div(255.0)
        mean = torch.tensor(
            (0.485, 0.456, 0.406), device=images.device, dtype=torch.float32
        ).view(1, 1, 3, 1, 1)
        std = torch.tensor(
            (0.229, 0.224, 0.225), device=images.device, dtype=torch.float32
        ).view(1, 1, 3, 1, 1)
        normalized = ((normalized - mean) / std).contiguous()
        images_14 = F.interpolate(
            normalized.flatten(0, 1).float(),
            size=(252, 252),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        ).reshape(batch_size, 2, 3, 252, 252)

        anyview = self._run_padded(self.da3_anyview, images_14=images_14)
        metric = self._run_padded(self.da3_metric, images_14=images_14)
        depth, extrinsics, intrinsics = self._postprocess_da3(anyview, metric)
        points = self._camera_points(depth, intrinsics)
        scale = torch.linalg.vector_norm(points.float(), dim=2).mean((1, 2, 3))
        points = points.float().div(scale[:, None, None, None, None].clamp_min(1e-6))
        points = F.interpolate(
            points.flatten(0, 1),
            size=(256, 256),
            mode="bilinear",
            align_corners=False,
        ).reshape(batch_size, 2, 3, 256, 256)

        flow = self._run_padded(
            self.flow_feature,
            geometry_tokens=anyview["geometry_tokens"],
            images_256_flat=normalized.flatten(0, 1).float(),
        )
        flow_maps = self._tokens_to_map(flow["flow_tokens"], 128)
        detail_maps = self._tokens_to_map(flow["flow3d_tokens"], 256)
        context_maps = (
            flow["context_features"].float().reshape(batch_size, 2, 128, 32, 32)
        )
        point_maps = F.interpolate(
            points.flatten(0, 1), size=(32, 32), mode="nearest"
        ).reshape(batch_size, 2, 3, 32, 32)
        motion_latent, motion_diagnostics = self._run_motion(
            flow_maps,
            detail_maps,
            context_maps,
            point_maps,
            int(iters),
            diagnostic_intermediates=diagnostic_intermediates,
        )
        result = {
            "geometry_latent_t": anyview["geometry_tokens"][:, 0],
            "motion_latent_t_to_target": motion_latent,
            "intermediate_metadata": {
                "backend": "tensorrt",
                "num_iterations": int(iters),
                "engine_batch_size": self.engine_batch_size,
                "engine_batch_sizes": self.engine_batch_sizes,
            },
        }
        if diagnostic_intermediates:
            result["diagnostic_intermediates"] = {
                "flow_tokens": flow["flow_tokens"],
                "detail_tokens": flow["flow3d_tokens"],
                "context_features": flow["context_features"],
                "motion_iterations": motion_diagnostics,
            }
        return result

    def _run_padded(
        self, engine: _TensorRTEngine, **inputs: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        input_rows_per_sample = {
            name: shape[0] // engine.batch_size
            for name, (shape, _) in engine.inputs.items()
        }
        if any(engine.inputs[name][0][0] % engine.batch_size for name in engine.inputs):
            raise ValueError("TensorRT input batch is not divisible by logical batch")
        logical_batch_sizes = {
            value.shape[0] // input_rows_per_sample[name]
            for name, value in inputs.items()
        }
        if any(
            value.shape[0] % input_rows_per_sample[name]
            for name, value in inputs.items()
        ):
            raise ValueError("TensorRT input rows do not describe whole samples")
        if len(logical_batch_sizes) != 1:
            raise ValueError(
                "TensorRT inputs do not describe one logical batch: "
                f"{sorted(logical_batch_sizes)}"
            )
        logical_batch_size = logical_batch_sizes.pop()
        output_rows_per_sample = {
            name: shape[0] // engine.batch_size
            for name, (shape, _) in engine.outputs.items()
        }
        if any(
            engine.outputs[name][0][0] % engine.batch_size for name in engine.outputs
        ):
            raise ValueError("TensorRT output batch is not divisible by logical batch")
        chunks: dict[str, list[torch.Tensor]] = {name: [] for name in engine.outputs}
        for start in range(0, logical_batch_size, engine.batch_size):
            chunk_size = min(engine.batch_size, logical_batch_size - start)
            padded: dict[str, torch.Tensor] = {}
            for name, value in inputs.items():
                rows_per_sample = input_rows_per_sample[name]
                chunk = value[
                    start * rows_per_sample : (start + chunk_size) * rows_per_sample
                ]
                expected_rows = engine.inputs[name][0][0]
                if chunk.shape[0] < expected_rows:
                    padding = chunk.new_zeros(
                        (expected_rows - chunk.shape[0], *chunk.shape[1:])
                    )
                    chunk = torch.cat((chunk, padding), dim=0)
                padded[name] = chunk
            outputs = engine(**padded)
            for name, value in outputs.items():
                output_rows = chunk_size * output_rows_per_sample[name]
                chunks[name].append(value[:output_rows])
        return {name: torch.cat(values, dim=0) for name, values in chunks.items()}

    def _postprocess_da3(
        self,
        anyview: dict[str, torch.Tensor],
        metric: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        depth = anyview["depth"]
        confidence = anyview["depth_conf"]
        c2w, intrinsics = _decode_pose(anyview["pose_encoding"], (252, 252))
        extrinsics = _affine_inverse(c2w)
        focal = (intrinsics[..., 0, 0] + intrinsics[..., 1, 1]) / 2
        metric_depth = metric["metric_depth"] * focal[..., None, None] / 300.0
        non_sky = metric["metric_sky"] < 0.3
        median_conf = _masked_quantile_per_sample(
            confidence, non_sky, 0.5, max_samples=100000
        )
        threshold = median_conf.reshape(-1, 1, 1, 1)
        align_mask = (
            (confidence >= threshold) & non_sky & (metric_depth > 1e-2) & (depth > 1e-3)
        )
        valid = align_mask.flatten(1)
        metric_values = torch.where(valid, metric_depth.float().flatten(1), 0.0)
        depth_values = torch.where(valid, depth.float().flatten(1), 0.0)
        scale = (metric_values * depth_values).sum(1) / depth_values.square().sum(
            1
        ).clamp_min(1e-12)
        depth = depth * scale[:, None, None, None].to(depth)
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] *= scale[:, None, None].to(extrinsics)
        sky_depth = _masked_quantile_per_sample(
            depth, non_sky, 0.99, max_samples=100000
        ).clamp_max(200.0)
        depth = torch.where(non_sky, depth, sky_depth[:, None, None, None].to(depth))
        return depth, extrinsics, intrinsics

    @staticmethod
    def _camera_points(depth: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        batch, frames, height, width = depth.shape
        y, x = torch.meshgrid(
            torch.arange(height, device=depth.device, dtype=torch.float32),
            torch.arange(width, device=depth.device, dtype=torch.float32),
            indexing="ij",
        )
        z = depth.float()
        x_points = (
            (x - intrinsics[..., 0, 2, None, None])
            * z
            / intrinsics[..., 0, 0, None, None]
        )
        y_points = (
            (y - intrinsics[..., 1, 2, None, None])
            * z
            / intrinsics[..., 1, 1, None, None]
        )
        return torch.stack((x_points, y_points, z), dim=2).reshape(
            batch, frames, 3, height, width
        )

    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor, channels: int) -> torch.Tensor:
        batch = tokens.shape[0]
        maps = tokens.reshape(batch * 2, 18, 18, channels).permute(0, 3, 1, 2)
        return F.interpolate(maps, size=(32, 32), mode="area").reshape(
            batch, 2, channels, 32, 32
        )

    def _run_motion(
        self,
        flow_maps: torch.Tensor,
        detail_maps: torch.Tensor,
        context_maps: torch.Tensor,
        point_maps: torch.Tensor,
        iters: int,
        *,
        diagnostic_intermediates: bool = False,
    ) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
        batch = flow_maps.shape[0]
        flow_anchor, flow_target = flow_maps[:, 0], flow_maps[:, 1]
        context_anchor, context_target = context_maps[:, 0], context_maps[:, 1]
        detail_anchor, detail_target = detail_maps[:, 0], detail_maps[:, 1]
        point_anchor, point_target = point_maps[:, 0], point_maps[:, 1]
        feature_corr = _Correlation(flow_anchor, flow_target)
        context_corr = _Correlation(context_anchor, context_target)
        flow_feature = flow_anchor.clone()
        context_feature = context_anchor + _time_embedding(128, flow_anchor.device)[
            None, :, None, None
        ].to(context_anchor)
        detail_3d = detail_anchor + _time_embedding(256, flow_anchor.device)[
            None, :, None, None
        ].to(detail_anchor)
        flow_2d = flow_anchor.new_zeros((batch, 2, 32, 32))
        flow_3d = flow_anchor.new_zeros((batch, 3, 32, 32))
        visibility = flow_anchor.new_zeros((batch, 2, 32, 32))
        coords = _coords_grid(batch, 32, 32, flow_anchor.device, flow_anchor.dtype)
        motion_latent = None
        diagnostics: list[dict[str, torch.Tensor]] = []
        for _ in range(iters):
            sample_coords = (coords + flow_2d).detach()
            feature_correlation = feature_corr(sample_coords)
            context_correlation = context_corr(sample_coords)
            motion_2d = _posenc(flow_2d, 10)
            motion_3d = _posenc(flow_3d, 10)
            pre_grid = _normalized_grid(sample_coords)
            sampled_feature_pre = F.grid_sample(
                detail_target,
                pre_grid,
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            )
            sampled_point_pre = F.grid_sample(
                point_target,
                pre_grid,
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            )
            point_corr = _Correlation(
                (point_anchor + flow_3d).detach(), point_target, mode="nearest"
            )(sample_coords)
            probe = self._run_padded(
                self.motion_iteration,
                flow_feature=flow_feature,
                context_feature=context_feature,
                visibility_confidence=visibility,
                feature_correlation=feature_correlation,
                context_correlation=context_correlation,
                motion_2d=motion_2d,
                detail_3d=detail_3d,
                sampled_feature_pre=sampled_feature_pre,
                sampled_feature=sampled_feature_pre,
                flow_3d_prior=sampled_point_pre - sampled_point_pre,
                point_correlation=point_corr,
                motion_3d=motion_3d,
            )
            next_flow_2d = flow_2d + probe["flow_2d_update"]
            post_grid = _normalized_grid(coords + next_flow_2d)
            sampled_feature = F.grid_sample(
                detail_target,
                post_grid,
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            )
            sampled_point = F.grid_sample(
                point_target,
                post_grid,
                mode="bilinear",
                align_corners=True,
                padding_mode="border",
            )
            outputs = self._run_padded(
                self.motion_iteration,
                flow_feature=flow_feature,
                context_feature=context_feature,
                visibility_confidence=visibility,
                feature_correlation=feature_correlation,
                context_correlation=context_correlation,
                motion_2d=motion_2d,
                detail_3d=detail_3d,
                sampled_feature_pre=sampled_feature_pre,
                sampled_feature=sampled_feature,
                flow_3d_prior=sampled_point - sampled_point_pre,
                point_correlation=point_corr,
                motion_3d=motion_3d,
            )
            if diagnostic_intermediates:
                diagnostics.append(
                    {
                        "flow_feature": flow_feature,
                        "context_feature": context_feature,
                        "visibility_confidence": visibility,
                        "feature_correlation": feature_correlation,
                        "context_correlation": context_correlation,
                        "motion_2d": motion_2d,
                        "detail_3d": detail_3d,
                        "sampled_feature_pre": sampled_feature_pre,
                        "sampled_feature": sampled_feature,
                        "flow_3d_prior": sampled_point - sampled_point_pre,
                        "point_correlation": point_corr,
                        "motion_3d": motion_3d,
                        "updated_flow_feature": outputs["updated_flow_feature"],
                        "flow_2d_update": outputs["flow_2d_update"],
                        "flow_3d_update": outputs["flow_3d_update"],
                        "updated_detail_3d": outputs["updated_detail_3d"],
                        "motion_latent": outputs["motion_latent"],
                    }
                )
            flow_feature = outputs["updated_flow_feature"]
            flow_2d = next_flow_2d
            detail_3d = outputs["updated_detail_3d"]
            flow_3d = flow_3d + outputs["flow_3d_update"]
            visibility = visibility + outputs["visibility_update"]
            motion_latent = outputs["motion_latent"]
        if motion_latent is None:
            raise ValueError("track4world_inference_iters must be positive")
        return motion_latent, diagnostics


def _coords_grid(
    batch: int, height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack((x, y), dim=0)[None].repeat(batch, 1, 1, 1)


def _normalized_grid(coords: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        (
            coords[:, 0] * (2.0 / (coords.shape[-1] - 1)) - 1.0,
            coords[:, 1] * (2.0 / (coords.shape[-2] - 1)) - 1.0,
        ),
        dim=-1,
    ).detach()


def _posenc(motion: torch.Tensor, max_degree: int) -> torch.Tensor:
    values = motion.permute(0, 2, 3, 1)
    scales = values.new_tensor([2**degree for degree in range(max_degree)])
    encoded = (values[..., None, :] * scales[:, None]).flatten(-2)
    encoded = torch.sin(torch.cat((encoded, encoded + 0.5 * torch.pi), dim=-1))
    return torch.cat((values, encoded), dim=-1).permute(0, 3, 1, 2).contiguous()


def _time_embedding(channels: int, device: torch.device) -> torch.Tensor:
    omega = torch.arange(channels // 2, device=device, dtype=torch.float64)
    omega = 1.0 / 10000 ** (omega / (channels / 2.0))
    values = omega
    return torch.cat((values.sin(), values.cos())).float()


class _Correlation:
    def __init__(
        self, first: torch.Tensor, second: torch.Tensor, mode: str = "area"
    ) -> None:
        self.radius = 4
        self.pyramid = []
        for _ in range(5):
            batch, channels, height, width = first.shape
            target_h, target_w = second.shape[-2:]
            correlation = first.flatten(2).transpose(1, 2) @ second.flatten(2)
            correlation = (
                correlation.reshape(batch * height * width, 1, target_h, target_w)
                / channels**0.5
            )
            self.pyramid.append(correlation)
            second = F.interpolate(second, scale_factor=0.5, mode=mode)

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        coords = coords.permute(0, 2, 3, 1)
        batch, height, width, _ = coords.shape
        delta_y, delta_x = torch.meshgrid(
            torch.linspace(-4, 4, 9, device=coords.device),
            torch.linspace(-4, 4, 9, device=coords.device),
            indexing="ij",
        )
        delta = torch.stack((delta_y, delta_x), dim=-1).view(1, 9, 9, 2)
        sampled = []
        for level, correlation in enumerate(self.pyramid):
            centroid = coords.reshape(batch * height * width, 1, 1, 2) / 2**level
            grid = (centroid + delta.to(centroid)).to(correlation.dtype)
            grid[..., 0] = grid[..., 0] * (2.0 / max(correlation.shape[-1] - 1, 1)) - 1
            grid[..., 1] = grid[..., 1] * (2.0 / max(correlation.shape[-2] - 1, 1)) - 1
            values = F.grid_sample(
                correlation, grid, align_corners=True, padding_mode="border"
            )
            sampled.append(values.reshape(batch, height, width, -1))
        return torch.cat(sampled, dim=-1).permute(0, 3, 1, 2).contiguous()


def _decode_pose(
    pose: torch.Tensor, image_size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    translation = pose[..., :3]
    quaternion = pose[..., 3:7]
    i, j, k, real = quaternion.unbind(-1)
    two_scale = 2.0 / quaternion.square().sum(-1)
    rotation = torch.stack(
        (
            1 - two_scale * (j * j + k * k),
            two_scale * (i * j - k * real),
            two_scale * (i * k + j * real),
            two_scale * (i * j + k * real),
            1 - two_scale * (i * i + k * k),
            two_scale * (j * k - i * real),
            two_scale * (i * k - j * real),
            two_scale * (j * k + i * real),
            1 - two_scale * (i * i + j * j),
        ),
        dim=-1,
    ).reshape(*pose.shape[:-1], 3, 3)
    c2w = (
        torch.eye(4, device=pose.device, dtype=pose.dtype)
        .expand(*pose.shape[:-1], 4, 4)
        .clone()
    )
    c2w[..., :3, :3] = rotation
    c2w[..., :3, 3] = translation
    height, width = image_size
    intrinsics = pose.new_zeros((*pose.shape[:-1], 3, 3))
    intrinsics[..., 0, 0] = (width / 2.0) / torch.tan(pose[..., 8] / 2.0).clamp_min(
        1e-6
    )
    intrinsics[..., 1, 1] = (height / 2.0) / torch.tan(pose[..., 7] / 2.0).clamp_min(
        1e-6
    )
    intrinsics[..., 0, 2] = width / 2.0
    intrinsics[..., 1, 2] = height / 2.0
    intrinsics[..., 2, 2] = 1.0
    return c2w, intrinsics


def _affine_inverse(matrix: torch.Tensor) -> torch.Tensor:
    rotation = matrix[..., :3, :3]
    translation = matrix[..., :3, 3:]
    bottom = matrix[..., 3:, :]
    return torch.cat(
        (torch.cat((rotation.mT, -rotation.mT @ translation), dim=-1), bottom),
        dim=-2,
    )


def _masked_quantile_per_sample(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    quantile: float,
    *,
    max_samples: int,
) -> torch.Tensor:
    values = []
    for batch_index in range(tensor.shape[0]):
        selected = tensor[batch_index][mask[batch_index]]
        if selected.numel() == 0:
            raise ValueError(f"sample {batch_index} has no valid depth values")
        if selected.numel() > max_samples:
            indices = (
                torch.arange(max_samples, device=selected.device)
                * selected.numel()
                // max_samples
            )
            selected = selected.flatten().index_select(0, indices)
        values.append(torch.quantile(selected.float(), quantile))
    return torch.stack(values)
