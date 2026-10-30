# Copyright 2026 MagicVLA authors. (Apache-2.0.)
"""Stable embodiment IDs and robot prompt descriptions. Append new IDs; never renumber existing ones."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "Embodiment",
    "EMBODIMENTS",
    "EMBODIMENT_BY_NAME",
    "EmbodimentSpec",
    "UNKNOWN_EMBODIMENT",
    "decode_embodiment_id",
    "describe",
    "resolve_spec",
]


@dataclass(frozen=True)
class Embodiment:
    """One row of the registry.

    ``description`` is what reaches the VLM, so it is written for a reader rather
    than a lookup: the make and model, the arm count, and the gripper, because
    those are what distinguish the actions the policy has to produce.
    """

    index: int
    name: str
    description: str


#: The registry.  Indices are permanent -- append only, never renumber.
EMBODIMENTS: tuple[Embodiment, ...] = (
    Embodiment(0, "arx_x5", "ARX X5 dual-arm robot with parallel grippers"),
    Embodiment(1, "piper", "AgileX PiPER dual-arm robot with parallel grippers"),
    Embodiment(2, "piper_x", "AgileX PiPER-X dual-arm robot with parallel grippers"),
    Embodiment(3, "aloha_agilex", "AgileX ALOHA dual-arm robot with parallel grippers"),
    Embodiment(4, "franka", "Franka Emika Panda dual-arm setup with parallel grippers"),
    Embodiment(5, "ur5", "Universal Robots UR5 dual-arm setup with parallel grippers"),
    Embodiment(
        6,
        "human_hands",
        "human hands recorded egocentrically, thumb-index pinch as the gripper",
    ),
    Embodiment(7, "agibot_a2d", "AgiBot A2-D dual-arm robot with parallel grippers"),
    Embodiment(
        8,
        "galaxea_r1lite",
        "Galaxea R1 Lite dual-arm robot on a static base with parallel grippers",
    ),
    # Index 9 is ours, not the pipeline's.  The pipeline deliberately gives real
    # and simulated arx_x5 the same index 0 ("同一物理机器人跨数据集共享同一 id"),
    # because it identifies the *robot*.  What reaches the VLM is not a robot id
    # though, it is a sentence, and "this is simulation" is worth saying: the
    # RoboDojo sim source is 3,489 episodes against RoboDojo real's own arx_x5
    # split, and the two differ in dynamics and rendering, not in kinematics.
    #
    # Safe because it appends.  Note the asymmetry it creates: nothing on disk
    # writes 9, so a dataset carrying observation.embodiment_id=0 still decodes
    # to plain arx_x5.  The sim distinction exists only where a source config
    # asserts it -- which is exactly the four column-less sources.
    Embodiment(
        9, "arx_x5_sim", "simulated ARX X5 dual-arm robot with parallel grippers"
    ),
    # Same reasoning as index 9, for the other simulated arm in the mix:
    # InternData-A1's lift2 corpus is a dual AgileX PiPER on a lift column,
    # rendered in simulation, and it carries no observation.embodiment_id
    # column at all -- the source config asserts the name. Reusing index 1
    # ("piper") would tell the VLM a real arm produced 458 hours of rendered
    # frames; nothing on disk writes 10, so real PiPER data still decodes to
    # plain piper.
    Embodiment(
        10, "piper_sim", "simulated AgileX PiPER dual-arm robot with parallel grippers"
    ),
)

EMBODIMENT_BY_INDEX: Mapping[int, Embodiment] = {e.index: e for e in EMBODIMENTS}
EMBODIMENT_BY_NAME: Mapping[str, Embodiment] = {e.name: e for e in EMBODIMENTS}

#: What a sample carries when the source cannot say which robot produced it.
#: It is a sentinel rather than a guess, and it suppresses the prompt line
#: entirely -- an absent claim is recoverable, a wrong one is not.
UNKNOWN_EMBODIMENT = "unknown"


def _check_registry() -> None:
    """A renumbering is a silent corpus-wide relabel, so refuse to load with one."""
    if [e.index for e in EMBODIMENTS] != list(range(len(EMBODIMENTS))):
        raise RuntimeError("embodiment indices must be contiguous and start at 0")
    if len(EMBODIMENT_BY_NAME) != len(EMBODIMENTS):
        raise RuntimeError("embodiment names must be unique")


_check_registry()


def describe(name: str) -> str | None:
    """The prompt phrasing for a registry name, or ``None`` if it has none."""
    entry = EMBODIMENT_BY_NAME.get(name)
    return entry.description if entry is not None else None


def decode_embodiment_id(value: Any) -> str:
    """Map a raw ``observation.embodiment_id`` cell to a registry name.

    Accepts both encodings.  A width-1 cell is the index; a wider cell is the
    one-hot and its hot position is the index.  Anything else -- an all-zero
    one-hot, a multi-hot, an index outside the registry -- returns
    :data:`UNKNOWN_EMBODIMENT` rather than guessing, because the alternative is
    labelling frames with a robot that did not produce them.
    """
    values = [float(v) for v in _flatten(value)]
    if not values:
        return UNKNOWN_EMBODIMENT

    if len(values) == 1:
        index = values[0]
    else:
        hot = [position for position, v in enumerate(values) if v != 0.0]
        if len(hot) != 1 or values[hot[0]] != 1.0:
            return UNKNOWN_EMBODIMENT
        index = float(hot[0])

    if index != int(index):
        return UNKNOWN_EMBODIMENT
    entry = EMBODIMENT_BY_INDEX.get(int(index))
    return entry.name if entry is not None else UNKNOWN_EMBODIMENT


def _as_path(value: Any) -> tuple[str, ...] | None:
    """Normalize a field path to segments.

    A list is taken verbatim, which is the only unambiguous form when a key can
    contain a dot.  A string is split, kept for paths that genuinely have no
    dotted keys.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(value.split("."))
    return tuple(str(part) for part in value)


def _flatten(value: Any) -> list[Any]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        out: list[Any] = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


@dataclass(frozen=True)
class EmbodimentSpec:
    """How one source answers "which robot produced this frame?".

    Exactly one of the three strategies applies, checked in the order they are
    listed here, because a source that has the column does not need a rule and a
    source with a constant robot does not need either.

    ``constant``
        The whole dataset is one robot.  Covers the four training sources with no
        column: EgoDex (human_hands), Hy-Embodied table_000 (arx_x5) and
        RoboDojo-3cam (arx_x5) are single-robot; the fourth, RoboTwin2.0, is not.

    ``column``
        Read it per frame from the parquet.  This is the default when the column
        exists, and handles both on-disk encodings.

    ``episode_meta``
        Derive it per episode from ``meta/episodes.jsonl`` by matching patterns
        against one field path, given as a LIST of segments.  A dotted string is
        accepted but ambiguous and should not be used here: the field this rule
        targets sits under a key that itself contains dots
        (``per_camera["observation.images.cam_high"]``), so splitting on "."
        walks into three levels that do not exist and every episode silently
        resolves to unknown -- which is exactly what happened to all 73,216
        RoboTwin episodes before the path became a list.  RoboTwin2.0 needs this: it mixes
        aloha_agilex / arx_x5 / piper in one dataset with no column, and the only
        field that names the robot for all 73,216 episodes is the source hdf5
        path recorded under image-quality metrics.
    """

    constant: str | None = None
    column: str | None = None
    episode_meta_path: tuple[str, ...] | None = None
    episode_meta_patterns: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        chosen = [
            bool(self.constant),
            bool(self.column),
            bool(self.episode_meta_path),
        ]
        if sum(chosen) > 1:
            raise ValueError(
                "embodiment: give one of constant / column / episode_meta_path, not several"
            )
        for name in filter(
            None, [self.constant, *(n for _, n in self.episode_meta_patterns)]
        ):
            if name not in EMBODIMENT_BY_NAME and name != UNKNOWN_EMBODIMENT:
                raise ValueError(
                    f"embodiment {name!r} is not in the registry; known names are "
                    f"{sorted(EMBODIMENT_BY_NAME)}"
                )
        if self.episode_meta_path and not self.episode_meta_patterns:
            raise ValueError("embodiment.episode_meta_path needs at least one pattern")

    def match_episode_meta(self, text: str) -> str:
        """First matching pattern wins, so order the config longest-prefix first.

        ``piper_x`` sources also contain ``piper``; the patterns are matched in
        the order written rather than by dict iteration for exactly that reason.
        """
        lowered = text.lower()
        for pattern, name in self.episode_meta_patterns:
            if re.search(pattern, lowered):
                return name
        return UNKNOWN_EMBODIMENT


def resolve_spec(value: Any) -> EmbodimentSpec | None:
    """Build a spec from config.

    ``None`` means "decide at read time": use the column if the dataset has one,
    otherwise report unknown.  A bare string is the constant form, which is what
    six of the eight training sources want.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return EmbodimentSpec(constant=value)
    if isinstance(value, int) and not isinstance(value, bool):
        entry = EMBODIMENT_BY_INDEX.get(int(value))
        if entry is None:
            raise ValueError(f"embodiment index {value} is not in the registry")
        return EmbodimentSpec(constant=entry.name)
    if not isinstance(value, Mapping):
        raise TypeError(
            f"embodiment must be a string, an index or a mapping, got {type(value)!r}"
        )

    patterns = value.get("patterns") or value.get("episode_meta_patterns") or {}
    if isinstance(patterns, Mapping):
        pattern_items = tuple((str(k), str(v)) for k, v in patterns.items())
    else:
        pattern_items = tuple(
            (str(entry["pattern"]), str(entry["name"])) for entry in patterns
        )
    return EmbodimentSpec(
        constant=value.get("constant") or value.get("name"),
        column=value.get("column"),
        episode_meta_path=_as_path(value.get("episode_meta_path") or value.get("path")),
        episode_meta_patterns=pattern_items,
    )
