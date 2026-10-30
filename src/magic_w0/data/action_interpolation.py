# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Semantics-aware temporal interpolation for robot action chunks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ActionInterpolationSpec:
    """Dimension groups in the post-selection/post-padding action space."""

    joint_indices: tuple[int, ...]
    gripper_indices: tuple[int, ...]
    eef_position_indices: tuple[int, ...]
    rotation_groups: tuple[tuple[int, ...], ...]
    rotation_gaussian_sigma: float = 0.22


def parse_action_interpolation_spec(
    payload: Mapping[str, Any] | None,
    *,
    source_action_dim: int,
    target_action_dim: int,
    active_action_indices: Sequence[int] | None = None,
) -> ActionInterpolationSpec:
    """Validate a semantic interpolation schema.

    Indices refer to the action vector after source selection and projection
    into the target layout. Every real source dimension must be assigned exactly
    once; padded or intentionally inserted target dimensions are omitted.
    """
    if not isinstance(payload, Mapping):
        raise ValueError(
            "action_upsample_mode='semantic' with rate > 1 requires an "
            "action_interpolation mapping"
        )
    allowed = {
        "joint_indices",
        "gripper_indices",
        "eef_position_indices",
        "rotation_groups",
        "rotation_gaussian_sigma",
    }
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"action_interpolation has unknown keys: {sorted(unknown)}")

    def _indices(name: str) -> tuple[int, ...]:
        raw = payload.get(name, ())
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise TypeError(f"action_interpolation.{name} must be a sequence")
        return tuple(int(index) for index in raw)

    joint_indices = _indices("joint_indices")
    gripper_indices = _indices("gripper_indices")
    eef_position_indices = _indices("eef_position_indices")
    raw_groups = payload.get("rotation_groups", ())
    if not isinstance(raw_groups, Sequence) or isinstance(raw_groups, (str, bytes)):
        raise TypeError("action_interpolation.rotation_groups must be a sequence")
    rotation_groups = tuple(
        tuple(int(index) for index in group) for group in raw_groups
    )
    if any(len(group) != 6 for group in rotation_groups):
        raise ValueError(
            "every action_interpolation.rotation_groups entry must contain 6 indices"
        )

    flat = [
        *joint_indices,
        *gripper_indices,
        *eef_position_indices,
        *(index for group in rotation_groups for index in group),
    ]
    if len(flat) != len(set(flat)):
        raise ValueError("action_interpolation dimension groups must not overlap")
    if flat and (min(flat) < 0 or max(flat) >= target_action_dim):
        raise ValueError(
            "action_interpolation indices must be inside the post-projection "
            f"target action dimension [0, {target_action_dim})"
        )
    expected = (
        set(range(source_action_dim))
        if active_action_indices is None
        else {int(index) for index in active_action_indices}
    )
    if len(expected) != source_action_dim:
        raise ValueError(
            "active_action_indices must contain one unique target slot per "
            f"selected source action dimension; expected {source_action_dim}, "
            f"got {len(expected)}"
        )
    if expected and (min(expected) < 0 or max(expected) >= target_action_dim):
        raise ValueError(
            "active_action_indices must be inside the post-projection target "
            f"action dimension [0, {target_action_dim})"
        )
    assigned = set(flat)
    if assigned != expected:
        raise ValueError(
            "action_interpolation must assign every selected source action dimension "
            f"exactly once; missing={sorted(expected - assigned)}, "
            f"unexpected={sorted(assigned - expected)}"
        )
    sigma = float(payload.get("rotation_gaussian_sigma", 0.22))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError(
            "action_interpolation.rotation_gaussian_sigma must be positive"
        )
    return ActionInterpolationSpec(
        joint_indices=joint_indices,
        gripper_indices=gripper_indices,
        eef_position_indices=eef_position_indices,
        rotation_groups=rotation_groups,
        rotation_gaussian_sigma=sigma,
    )


def semantic_upsample_actions(
    values: np.ndarray,
    value_dim_mask: np.ndarray,
    quality_valid: np.ndarray,
    *,
    output_size: int,
    rate: int,
    spec: ActionInterpolationSpec,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Upsample actions on the grid ``source_time = output_index / rate``.

    Joint positions use piecewise quintic Hermite interpolation with finite-
    difference velocity and acceleration estimates. Gripper and EEF position
    dimensions use linear interpolation. Each 6D rotation is projected to SO(3),
    interpolated by shortest-path quaternion SLERP with a normalized Gaussian-CDF
    time weight, and converted back to the 6D representation.

    The returned row-valid mask includes quality validity. Interpolated rows depend
    on both source endpoints and are valid only when both endpoints are valid.
    """
    values = np.asarray(values, dtype=np.float32)
    value_dim_mask = np.asarray(value_dim_mask, dtype=np.bool_)
    quality_valid = np.asarray(quality_valid, dtype=np.bool_)
    if values.ndim != 2 or value_dim_mask.shape != values.shape:
        raise ValueError(
            "values and value_dim_mask must have matching [time, dim] shapes"
        )
    if quality_valid.shape != (len(values),):
        raise ValueError("quality_valid must have shape [time]")
    if rate <= 0 or output_size <= 0 or len(values) == 0:
        raise ValueError("rate, output_size, and source action length must be positive")

    produced_rows = min(output_size, len(values) * rate)
    coordinates = np.arange(produced_rows, dtype=np.float64) / float(rate)
    left = np.floor(coordinates).astype(np.int64)
    right = np.minimum(left + 1, len(values) - 1)
    alpha = (coordinates - left).astype(np.float32)
    has_right = right != left

    output = values[left].copy()
    output_dim_mask = value_dim_mask[left].copy()
    output_quality = quality_valid[left].copy()
    interpolated = has_right & (alpha > 0)
    if interpolated.any():
        output_dim_mask[interpolated] &= value_dim_mask[right[interpolated]]
        output_quality[interpolated] &= quality_valid[right[interpolated]]

    linear_indices = (*spec.gripper_indices, *spec.eef_position_indices)
    if linear_indices:
        dims = list(linear_indices)
        output[:, dims] = (
            values[left][:, dims] * (1.0 - alpha[:, None])
            + values[right][:, dims] * alpha[:, None]
        )

    if spec.joint_indices:
        dims = list(spec.joint_indices)
        joint_output = _quintic_hermite(
            values[:, dims],
            value_dim_mask[:, dims],
            left,
            right,
            alpha,
        )
        output[np.ix_(has_right, dims)] = joint_output[has_right]

    for group in spec.rotation_groups:
        dims = list(group)
        group_valid = output_dim_mask[:, dims].all(axis=1)
        output_dim_mask[:, dims] = group_valid[:, None]
        for output_index, (i0, i1, fraction) in enumerate(
            zip(left, right, alpha, strict=True)
        ):
            if i0 == i1 or fraction == 0:
                output[output_index, dims] = _project_rotation_6d(values[i0, dims])
                continue
            gaussian_fraction = _gaussian_cdf_weight(
                float(fraction), spec.rotation_gaussian_sigma
            )
            output[output_index, dims] = _slerp_rotation_6d(
                values[i0, dims], values[i1, dims], gaussian_fraction
            )

    return output, output_quality, output_dim_mask


def _quintic_hermite(
    values: np.ndarray,
    valid: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Piecewise C2 quintic interpolation with mask-aware finite differences."""
    n_rows, n_dims = values.shape
    velocity = np.zeros_like(values, dtype=np.float64)
    acceleration = np.zeros_like(values, dtype=np.float64)
    values64 = values.astype(np.float64, copy=False)
    for dim in range(n_dims):
        for index in range(n_rows):
            if not valid[index, dim]:
                continue
            before = index > 0 and valid[index - 1, dim]
            after = index + 1 < n_rows and valid[index + 1, dim]
            if before and after:
                velocity[index, dim] = 0.5 * (
                    values64[index + 1, dim] - values64[index - 1, dim]
                )
                acceleration[index, dim] = (
                    values64[index + 1, dim]
                    - 2.0 * values64[index, dim]
                    + values64[index - 1, dim]
                )
            elif after:
                velocity[index, dim] = values64[index + 1, dim] - values64[index, dim]
            elif before:
                velocity[index, dim] = values64[index, dim] - values64[index - 1, dim]

    p0, p1 = values64[left], values64[right]
    v0, v1 = velocity[left], velocity[right]
    a0, a1 = acceleration[left], acceleration[right]
    c0 = p0
    c1 = v0
    c2 = 0.5 * a0
    delta_p = p1 - c0 - c1 - c2
    delta_v = v1 - c1 - 2.0 * c2
    delta_a = a1 - 2.0 * c2
    c3 = 10.0 * delta_p - 4.0 * delta_v + 0.5 * delta_a
    c4 = -15.0 * delta_p + 7.0 * delta_v - delta_a
    c5 = 6.0 * delta_p - 3.0 * delta_v + 0.5 * delta_a
    u = alpha.astype(np.float64)[:, None]
    result = c0 + u * (c1 + u * (c2 + u * (c3 + u * (c4 + u * c5))))
    return result.astype(np.float32)


def _gaussian_cdf_weight(fraction: float, sigma: float) -> float:
    """Endpoint-preserving Gaussian-CDF time warp on [0, 1]."""
    scale = sigma * math.sqrt(2.0)

    def cdf(value: float) -> float:
        return 0.5 * (1.0 + math.erf((value - 0.5) / scale))

    low, high = cdf(0.0), cdf(1.0)
    return float(np.clip((cdf(fraction) - low) / (high - low), 0.0, 1.0))


def _project_rotation_6d(rotation: np.ndarray) -> np.ndarray:
    return _matrix_to_rotation_6d(_rotation_6d_to_matrix(rotation))


def _rotation_6d_to_matrix(rotation: np.ndarray) -> np.ndarray:
    first = np.asarray(rotation[:3], dtype=np.float64)
    second = np.asarray(rotation[3:6], dtype=np.float64)
    first_norm = np.linalg.norm(first)
    if first_norm < 1e-8:
        return np.eye(3, dtype=np.float64)
    b1 = first / first_norm
    second = second - np.dot(b1, second) * b1
    second_norm = np.linalg.norm(second)
    if second_norm < 1e-8:
        candidate = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(candidate, b1)) > 0.9:
            candidate = np.array([0.0, 1.0, 0.0])
        second = candidate - np.dot(b1, candidate) * b1
        second_norm = np.linalg.norm(second)
    b2 = second / second_norm
    b3 = np.cross(b1, b2)
    return np.stack((b1, b2, b3), axis=0)


def _matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix[:2], dtype=np.float32).reshape(6)


def _slerp_rotation_6d(
    start: np.ndarray, end: np.ndarray, fraction: float
) -> np.ndarray:
    q0 = _matrix_to_quaternion(_rotation_6d_to_matrix(start))
    q1 = _matrix_to_quaternion(_rotation_6d_to_matrix(end))
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:
        quaternion = q0 + fraction * (q1 - q0)
        quaternion /= np.linalg.norm(quaternion)
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        quaternion = (
            math.sin((1.0 - fraction) * theta) / sin_theta * q0
            + math.sin(fraction * theta) / sin_theta * q1
        )
    return _matrix_to_rotation_6d(_quaternion_to_matrix(quaternion))


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a proper rotation matrix to a normalized wxyz quaternion."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(quaternion, dtype=np.float64)
    norm = w * w + x * x + y * y + z * z
    scale = 2.0 / max(norm, 1e-12)
    return np.array(
        [
            [
                1.0 - scale * (y * y + z * z),
                scale * (x * y - z * w),
                scale * (x * z + y * w),
            ],
            [
                scale * (x * y + z * w),
                1.0 - scale * (x * x + z * z),
                scale * (y * z - x * w),
            ],
            [
                scale * (x * z - y * w),
                scale * (y * z + x * w),
                1.0 - scale * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )
