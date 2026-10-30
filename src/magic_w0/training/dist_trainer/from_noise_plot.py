# Copyright 2026 MagicVLA authors.
#
# Licensed under the Apache License, Version 2.0 (the "License").
"""Render the from-noise open-loop error curve as a PNG.

One aggregate number hides the thing worth seeing.  Integration error grows
along the action chunk, so a policy can look healthy on mean MSE while the tail
of every chunk is far enough off to miss; the horizon-normalizer work found
exactly that (end-of-chunk error a quarter high under a normalizer shared across
all 50 horizon positions).  The curve makes it a glance instead of an audit.

Colours follow ``scripts/eval_openloop_magicvla_vqa.py`` so the offline and
in-training figures read as the same family.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

_GT_BLUE = "#1f6feb"
_PRED_ORANGE = "#e0603a"
_GRID_GREY = "#9aa5b1"
_DPI = 115


def _pyplot():
    """Import pyplot on demand, or ``None`` if it is not installed.

    matplotlib is in the training image but is not a package dependency, and a
    figure is never worth taking a training run down for.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display on a compute node
        from matplotlib import pyplot as plt

        return plt
    except Exception as error:  # pragma: no cover - depends on the environment
        logger.warning("from-noise PNG skipped: matplotlib unavailable (%s)", error)
        return None


def render_per_horizon_png(
    path: str | Path,
    history: list[tuple[int, np.ndarray]],
    *,
    title: str = "from-noise open-loop error",
) -> Path | None:
    """Draw per-horizon MSE for each recorded eval, newest curve highlighted.

    ``history`` is ``(global_step, per_horizon_mse)`` in step order.  Returns the
    written path, or ``None`` when there is nothing to draw or no matplotlib.

    The arrays are dumped next to the figure: re-styling a plot should never cost
    another pass of flow-matching sampling.
    """
    if not history:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = [int(step) for step, _ in history]
    curves = [np.asarray(curve, dtype=np.float64) for _, curve in history]
    np.savez_compressed(
        path.with_suffix(".npz"),
        steps=np.asarray(steps, dtype=np.int64),
        per_horizon=np.stack(curves),
    )

    plt = _pyplot()
    if plt is None:
        try:
            from PIL import Image, ImageDraw

            image = Image.new("RGB", (900, 360), "white")
            draw = ImageDraw.Draw(image)
            left, top, right, bottom = 55, 25, 865, 315
            draw.rectangle((left, top, right, bottom), outline=_GRID_GREY, width=1)
            curve = curves[-1]
            if curve.size:
                low, high = float(curve.min()), float(curve.max())
                span = max(high - low, 1e-12)
                points = [
                    (
                        left + (right - left) * index / max(len(curve) - 1, 1),
                        bottom - (bottom - top) * (float(value) - low) / span,
                    )
                    for index, value in enumerate(curve)
                ]
                if len(points) == 1:
                    x, y = points[0]
                    draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=_PRED_ORANGE)
                else:
                    draw.line(points, fill=_PRED_ORANGE, width=3)
            image.save(path, format="PNG")
            return path
        except Exception as error:  # pragma: no cover - optional fallback
            logger.warning("from-noise fallback PNG skipped: %s", error)
            return None

    horizon = np.arange(len(curves[-1]))

    fig, (ax_curve, ax_trend) = plt.subplots(
        1, 2, figsize=(11.5, 4.0), gridspec_kw={"width_ratios": [1.6, 1.0]}
    )

    # Left: error along the chunk.  Older evals fade so the shape of the most
    # recent one stays readable when a long run stacks dozens of curves.
    for index, (step, curve) in enumerate(zip(steps, curves, strict=True)):
        is_last = index == len(curves) - 1
        ax_curve.plot(
            np.arange(len(curve)),
            curve,
            color=_PRED_ORANGE if is_last else _GRID_GREY,
            alpha=1.0 if is_last else max(0.12, 0.5 * (index + 1) / len(curves)),
            lw=2.0 if is_last else 1.0,
            label=f"step {step}" if is_last else None,
            zorder=3 if is_last else 2,
        )
    ax_curve.set_xlabel("horizon position h")
    ax_curve.set_ylabel("normalized MSE")
    ax_curve.set_title("error along the action chunk", fontsize=10)
    ax_curve.grid(alpha=0.25, lw=0.6)
    ax_curve.margins(x=0.01)
    if len(horizon):
        ax_curve.set_xlim(0, len(horizon) - 1)
    ax_curve.legend(fontsize=8, loc="best")

    # Right: how the aggregate and the two ends of the chunk move over training.
    mean_curve = [float(curve.mean()) for curve in curves]
    first_h = [float(curve[0]) for curve in curves]
    last_h = [float(curve[-1]) for curve in curves]
    ax_trend.plot(
        steps, mean_curve, color=_PRED_ORANGE, lw=1.8, marker="o", ms=3, label="mean"
    )
    ax_trend.plot(
        steps, first_h, color=_GT_BLUE, lw=1.4, ls="--", marker="s", ms=3, label="h=0"
    )
    ax_trend.plot(
        steps,
        last_h,
        color=_GRID_GREY,
        lw=1.4,
        ls=":",
        marker="^",
        ms=3,
        label=f"h={len(curves[-1]) - 1}",
    )
    ax_trend.set_xlabel("global step")
    ax_trend.set_ylabel("normalized MSE")
    ax_trend.set_title("over training", fontsize=10)
    ax_trend.grid(alpha=0.25, lw=0.6)
    ax_trend.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"{title} — {len(curves)} eval(s), latest step {steps[-1]}, "
        f"mean {mean_curve[-1]:.4f}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=_DPI)
    plt.close(fig)

    logger.info("from-noise curve written to %s", path)
    return path


__all__ = ["render_per_horizon_png"]
