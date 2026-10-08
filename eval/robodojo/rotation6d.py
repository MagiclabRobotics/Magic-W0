# Copyright 2026 Magic_W0 authors. (Apache-2.0.)
"""6D rotation conversion and relative rotation composition.

The 6D representation concatenates the first two columns of a rotation matrix.
Relative rotations use ``R_target @ R_reference.T``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compose_relative_rot6d",
    "matrix_to_rot6d",
    "rot6d_to_matrix",
]


def rot6d_to_matrix(rot6d: np.ndarray) -> np.ndarray:
    """``[..., 6] -> [..., 3, 3]`` via Gram-Schmidt (Zhou et al., 2019).

    Orthonormalize the first two columns; their cross product supplies the third.
    """
    rot6d = np.asarray(rot6d, dtype=np.float64)
    if rot6d.shape[-1] != 6:
        raise ValueError(
            f"rot6d must have a trailing dimension of 6, got {rot6d.shape}"
        )

    first = rot6d[..., 0:3]
    second = rot6d[..., 3:6]

    column1 = _normalize(first)
    projection = np.sum(column1 * second, axis=-1, keepdims=True) * column1
    column2 = _normalize(second - projection)
    column3 = np.cross(column1, column2)
    return np.stack((column1, column2, column3), axis=-1)


def matrix_to_rot6d(matrix: np.ndarray) -> np.ndarray:
    """``[..., 3, 3] -> [..., 6]``; the inverse of :func:`rot6d_to_matrix`."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape[-2:] != (3, 3):
        raise ValueError(f"expected a [..., 3, 3] matrix, got {matrix.shape}")
    return np.concatenate((matrix[..., :, 0], matrix[..., :, 1]), axis=-1)


def compose_relative_rot6d(target: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """The rotation carrying ``reference`` to ``target``, as rot6d.

    ``R_rel = R_target @ R_reference.T``.  Broadcasts, so a single reference
    pose can be applied to a whole action chunk.
    """
    target_matrix = rot6d_to_matrix(target)
    reference_matrix = rot6d_to_matrix(reference)
    relative = target_matrix @ np.swapaxes(reference_matrix, -1, -2)
    return matrix_to_rot6d(relative)


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    # A zero row is a masked/absent pose rather than a rotation; leaving it at
    # zero keeps it distinguishable instead of turning it into a NaN that
    # propagates through the whole chunk.
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 1e-12)
