"""Replay timeline alignment helpers."""

from __future__ import annotations

import numpy as np

from xdof_sim.rendering.replay.types import EpisodeContext, ReplayTimeline


def sample_hold_align(values: np.ndarray, timestamps: np.ndarray, query_ts: np.ndarray) -> np.ndarray:
    """Align samples to query timestamps using zero-order hold."""
    if len(values) != len(timestamps):
        raise ValueError("values and timestamps must have the same length")
    if len(query_ts) == 0:
        return np.empty((0, *values.shape[1:]), dtype=values.dtype)
    idx = np.searchsorted(timestamps, query_ts, side="right") - 1
    idx = np.clip(idx, 0, len(values) - 1)
    return values[idx]


def build_action_timeline(
    actions_left: np.ndarray,
    ts_left: np.ndarray,
    actions_right: np.ndarray,
    ts_right: np.ndarray,
    control_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample async left/right action streams onto a regular control grid."""
    t_start = min(ts_left[0], ts_right[0])
    t_end = min(ts_left[-1], ts_right[-1])
    grid_ts = np.arange(t_start, t_end, 1.0 / control_hz)

    left_aligned = sample_hold_align(actions_left, ts_left, grid_ts)
    right_aligned = sample_hold_align(actions_right, ts_right, grid_ts)
    actions = np.concatenate([left_aligned.astype(np.float32), right_aligned.astype(np.float32)], axis=1)
    return actions, grid_ts


def align_sim_states(
    context: EpisodeContext,
    grid_ts: np.ndarray,
) -> np.ndarray | None:
    """Align raw qpos samples to the replay clock."""
    raw_qposes = context.raw_sim_states
    raw_timestamps = context.raw_sim_timestamps
    if raw_qposes is None or raw_timestamps is None:
        return None
    aligned_ts = raw_timestamps
    # Delivered episodes use a relative action clock in output.mcap but absolute
    # Unix timestamps in sim_state.mcap. Shift the qpos stream onto the same
    # zero-based timebase before sample-and-hold alignment.
    if (
        context.episode_format == "delivered"
        and len(aligned_ts) > 0
        and len(grid_ts) > 0
        and aligned_ts[0] > 1e9
        and grid_ts[0] < 1e6
    ):
        aligned_ts = aligned_ts - aligned_ts[0]
    return sample_hold_align(raw_qposes, aligned_ts, grid_ts)


def build_replay_timeline(context: EpisodeContext, control_hz: float) -> ReplayTimeline:
    """Build the aligned action/qpos timeline used by the viewer."""
    if context.replay_actions is not None and context.replay_timestamps is not None:
        return ReplayTimeline(
            actions=context.replay_actions.astype(np.float32, copy=False),
            grid_ts=np.asarray(context.replay_timestamps, dtype=np.float64),
            sim_states=context.raw_sim_states,
            sim_state_kind=context.replay_state_kind,
            sim_state_alignment=context.replay_state_alignment,
        )

    actions, grid_ts = build_action_timeline(
        context.streams.actions_left,
        context.streams.ts_left,
        context.streams.actions_right,
        context.streams.ts_right,
        control_hz=control_hz,
    )
    sim_states = align_sim_states(context, grid_ts)
    return ReplayTimeline(
        actions=actions,
        grid_ts=grid_ts,
        sim_states=sim_states,
        sim_state_kind=context.replay_state_kind,
        sim_state_alignment=context.replay_state_alignment,
    )
