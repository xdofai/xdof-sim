"""Shared replay state machine for viewers and headless exporters."""

from __future__ import annotations

from typing import Literal

import mujoco
import numpy as np


ReplayMode = Literal["auto", "physics", "qpos"]


class ReplaySession:
    """Owns replay stepping, reset behavior, and progress state."""

    def __init__(
        self,
        env,
        actions: np.ndarray,
        grid_ts: np.ndarray,
        *,
        sim_states: np.ndarray | None = None,
        rand_state=None,
        task: str = "",
        mode: ReplayMode = "auto",
    ):
        self.env = env
        self.model = env.model
        self.data = env.data
        self.actions = actions
        self.grid_ts = grid_ts
        self.sim_states = sim_states
        self.rand_state = rand_state
        self.task = task
        self._mode: Literal["physics", "qpos"] = self._resolve_mode(mode)
        self._step_idx = 0
        self.reset()

    def _resolve_mode(self, mode: ReplayMode) -> Literal["physics", "qpos"]:
        if mode == "auto":
            return "qpos" if self.sim_states is not None else "physics"
        if mode == "qpos" and self.sim_states is None:
            raise ValueError("qpos replay mode requested but no aligned sim states are available")
        return mode

    @property
    def mode(self) -> Literal["physics", "qpos"]:
        return self._mode

    @property
    def step_idx(self) -> int:
        return self._step_idx

    @property
    def has_exact_qpos(self) -> bool:
        return self.sim_states is not None

    @property
    def current_frame_idx(self) -> int:
        if len(self.grid_ts) == 0:
            return 0
        return min(self._step_idx, len(self.grid_ts) - 1)

    @property
    def current_timestamp(self) -> float:
        if len(self.grid_ts) == 0:
            return 0.0
        return float(self.grid_ts[self.current_frame_idx])

    @property
    def step_dt(self) -> float:
        if len(self.grid_ts) > 1:
            return float(self.grid_ts[1] - self.grid_ts[0])
        return 1.0 / 30.0

    @property
    def duration_s(self) -> float:
        if len(self.grid_ts) > 1:
            return float(self.grid_ts[-1] - self.grid_ts[0])
        return 0.0

    @property
    def elapsed_s(self) -> float:
        if len(self.grid_ts) == 0:
            return 0.0
        return float(self.grid_ts[self.current_frame_idx] - self.grid_ts[0])

    @property
    def total_steps(self) -> int:
        if self.mode == "qpos" and self.sim_states is not None:
            return max(0, len(self.sim_states) - 1)
        return len(self.actions)

    @property
    def is_done(self) -> bool:
        return self._step_idx >= self.total_steps

    def set_mode(self, mode: ReplayMode) -> None:
        self._mode = self._resolve_mode(mode)
        self.reset()

    def _apply_randomization(self) -> None:
        if self.rand_state is None or not self.task:
            return
        randomizer = getattr(self.env, "_task_randomizer", None)
        if randomizer is not None:
            randomizer.apply(self.model, self.data, self.rand_state)
            self.model = self.env.model
            self.data = self.env.data

    def _apply_qpos_frame(self, frame_idx: int) -> None:
        if self.sim_states is None or len(self.sim_states) == 0:
            return
        qpos = self.sim_states[frame_idx]
        self.data.qpos[: len(qpos)] = qpos
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        q0 = self.env.get_init_q()
        self.data.qpos[: len(q0)] = q0
        mujoco.mj_forward(self.model, self.data)
        self._apply_randomization()
        mujoco.mj_forward(self.model, self.data)
        self._step_idx = 0
        if self.mode == "qpos":
            self._apply_qpos_frame(0)

    def step(self) -> bool:
        if self.mode == "qpos":
            next_idx = self._step_idx + 1
            if self.sim_states is None or next_idx >= len(self.sim_states):
                return False
            self._step_idx = next_idx
            self._apply_qpos_frame(self._step_idx)
            return True

        if self._step_idx >= len(self.actions):
            return False
        action = self.actions[self._step_idx].astype(np.float32)
        self.env._step_single(action)
        self._step_idx += 1
        return True
