"""Typed replay data containers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np


EpisodeFormat = Literal["raw", "delivered"]


@dataclass(frozen=True)
class EpisodeStreams:
    """Loaded episode streams used to build a replay."""

    episode_dir: Path
    actions_left: np.ndarray
    ts_left: np.ndarray
    actions_right: np.ndarray
    ts_right: np.ndarray
    camera_frames: dict[str, np.ndarray]
    camera_ts: dict[str, np.ndarray]


@dataclass(frozen=True)
class EpisodeContext:
    """Episode data plus sim-specific metadata."""

    streams: EpisodeStreams
    episode_format: EpisodeFormat
    scene: str
    task: str
    instruction: str | None
    rand_state: Any | None
    raw_sim_states: np.ndarray | None
    raw_sim_timestamps: np.ndarray | None


@dataclass(frozen=True)
class ReplayTimeline:
    """Aligned replay arrays on a single playback clock."""

    actions: np.ndarray
    grid_ts: np.ndarray
    sim_states: np.ndarray | None
