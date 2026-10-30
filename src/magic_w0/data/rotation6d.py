# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""6D rotation representation, and the composition that a subtraction is not.

The corpus stores end-effector orientation as ``rot6d``: the first two columns
of the rotation matrix, flattened.  Verified on all eight training sources --
both 3-vectors have unit norm and their dot product is 0.0000, so the stored
values are already an orthonormal frame and the Gram-Schmidt decode below is
exact rather than a projection.

Delta action representations used to subtract these six numbers elementwise.
That is not the difference of two rotations.  The relative rotation taking pose
A to pose B is ``R_rel = R_B @ R_A.T``, and elementwise subtraction differs from
it in three ways that matter:

1. **The result leaves the manifold.**  Two orthonormal vectors minus two other
   orthonormal vectors is generally not an orthonormal frame, so the target
   decodes to no rotation at all.

2. **It is not frame-invariant.**  A relative rotation must not depend on the
   coordinate system both poses are expressed in: rotating both by ``R0`` leaves
   ``R_rel`` unchanged, because ``(R0 R_B)(R0 R_A)^T = R0 R_B R_A^T R0^T`` --
   conjugation, which preserves the rotation angle.  The elementwise difference
   instead maps to ``R0 (R_B - R_A)``, so the same physical wrist motion yields
   a different target depending on where the arm happens to be.  The model has
   to spend capacity unlearning a dependence the representation was chosen to
   remove.

3. **Its scale is pose-dependent.**  Rotating 5 degrees about a fixed axis
   changes the six components by wildly different amounts depending on the
   starting pose, because the components are sines and cosines evaluated at
   different operating points.  Normalization statistics are computed over the
   whole dataset, so that non-uniformity is baked straight into the loss
   weighting.

None of this raises: the loss still falls, because the network can learn a map
from the odd difference vector to the action.  It just pays for something the
representation was supposed to give away for free.
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

    The two 3-vectors are the first two columns of ``R``; the third is their
    cross product.  Stored data is already orthonormal, so this reproduces the
    original frame exactly, but the orthonormalization is kept because it is
    what makes the inverse well defined for any input.
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
