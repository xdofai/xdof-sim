"""Batched GPU-accelerated simulation environment for policy deployment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from xdof_sim.env import (
    MuJoCoYAMEnv,
    _GRIPPER_CTRL_MAX,
    project_policy_state_batch,
)
from xdof_sim.rendering.replay.renderer import RendererWrapper, WarpReplayRuntime
from xdof_sim.task_eval import TaskEvalResult, make_task_evaluator


@dataclass(frozen=True)
class WorldResetInfo:
    seed: int | None
    randomization: Any


class BatchedWarpYAMEnv:
    """Run multiple identical YAM worlds in parallel with Warp physics."""

    def __init__(
        self,
        base_env: MuJoCoYAMEnv,
        *,
        num_worlds: int,
        camera_backend: Literal["mjwarp", "madrona"],
        camera_gpu_id: int | None = None,
        render_cameras: bool = True,
    ) -> None:
        if num_worlds < 1:
            raise ValueError(f"num_worlds must be >= 1, got {num_worlds}")
        if camera_backend == "mujoco":
            raise ValueError("BatchedWarpYAMEnv only supports 'mjwarp' or 'madrona' camera backends.")

        self.base_env = base_env
        self.model = base_env.model
        self.data = base_env.data
        self.config = base_env.config
        self.prompt = base_env.prompt
        self.camera_names = list(base_env.camera_names)
        self.robot_names = list(base_env.robot_names)
        self.num_worlds = num_worlds
        self.chunk_dim = base_env.chunk_dim
        self.single_timestep_action_dim = base_env.single_timestep_action_dim
        self._render_cameras = render_cameras
        self._camera_backend = camera_backend
        self._camera_gpu_id = camera_gpu_id
        self._camera_height = base_env._camera_height
        self._camera_width = base_env._camera_width
        self._control_decimation = base_env._control_decimation
        self._qpos_indices = list(base_env._qpos_indices)
        self._ctrl_indices = list(base_env._ctrl_indices)
        self._gripper_indices = list(base_env._gripper_indices)
        self._gripper_set = set(self._gripper_indices)
        self._task = base_env._task
        self._task_spec = base_env._task_spec
        self._task_evaluator = make_task_evaluator(self.model, self._task_spec)

        self._runtime = WarpReplayRuntime(
            self.model,
            self.data,
            nworld=num_worlds,
            gpu_id=camera_gpu_id,
            nconmax=max(512, int(getattr(self.model, "nconmax", 64)) * num_worlds * 16),
            njmax=max(4096, int(getattr(self.model, "njmax", 128)) * num_worlds * 64),
        )
        self._renderer = None
        if render_cameras:
            self._renderer = RendererWrapper(
                backend=camera_backend,
                runtime=self._runtime,
                cam_res=(self._camera_width, self._camera_height),
                gpu_id=camera_gpu_id,
            )

        self._camera_index = {
            self.model.cam(i).name: i for i in range(self.model.ncam)
        }
        self._needs_renderer_reset = True
        self._step_count = 0
        self._episode_index = 0
        self._world_reset_info: list[WorldResetInfo] = []

    def _world_seed(self, *, base_seed: int | None, world_index: int) -> int | None:
        if base_seed is None:
            return None
        return int(base_seed) + world_index

    def _snapshot_cpu_state(self) -> dict[str, np.ndarray | None]:
        act = None
        if self.model.na > 0 and hasattr(self.data, "act"):
            act = np.asarray(self.data.act, dtype=np.float32).copy()
        mocap_pos = None
        mocap_quat = None
        if self.model.nmocap > 0:
            mocap_pos = np.asarray(self.data.mocap_pos, dtype=np.float32).copy()
            mocap_quat = np.asarray(self.data.mocap_quat, dtype=np.float32).copy()
        return {
            "qpos": np.asarray(self.data.qpos, dtype=np.float32).copy(),
            "qvel": np.asarray(self.data.qvel, dtype=np.float32).copy(),
            "ctrl": np.asarray(self.data.ctrl, dtype=np.float32).copy(),
            "act": act,
            "mocap_pos": mocap_pos,
            "mocap_quat": mocap_quat,
            "time": np.float32(self.data.time),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
        randomize: bool = True,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        snapshots: list[dict[str, np.ndarray | None]] = []
        reset_info: list[WorldResetInfo] = []

        for world_idx in range(self.num_worlds):
            world_seed = self._world_seed(base_seed=seed, world_index=world_idx)
            self.base_env.reset(seed=world_seed, options=options, randomize=randomize)
            snapshots.append(self._snapshot_cpu_state())
            reset_info.append(
                WorldResetInfo(
                    seed=world_seed,
                    randomization=self.base_env._last_randomization,
                )
            )

        qpos = np.stack([snap["qpos"] for snap in snapshots], axis=0)
        qvel = np.stack([snap["qvel"] for snap in snapshots], axis=0)
        ctrl = np.stack([snap["ctrl"] for snap in snapshots], axis=0)
        act = None
        if snapshots[0]["act"] is not None:
            act = np.stack([snap["act"] for snap in snapshots], axis=0)
        mocap_pos = None
        mocap_quat = None
        if snapshots[0]["mocap_pos"] is not None:
            mocap_pos = np.stack([snap["mocap_pos"] for snap in snapshots], axis=0)
            mocap_quat = np.stack([snap["mocap_quat"] for snap in snapshots], axis=0)
        time_arr = np.asarray([snap["time"] for snap in snapshots], dtype=np.float32)

        self._runtime.reset_from_mujoco()
        self._runtime.load_state_batch(
            qpos=qpos,
            qvel=qvel,
            ctrl=ctrl,
            act=act,
            mocap_pos=mocap_pos,
            mocap_quat=mocap_quat,
            time=time_arr,
        )
        self._runtime.forward()
        self._needs_renderer_reset = True
        self._step_count = 0
        self._episode_index += 1
        self._world_reset_info = reset_info
        if self._task_evaluator is not None:
            self._task_evaluator.reset(nworld=self.num_worlds)
        obs = self.get_obs()
        info = {
            "world_seeds": [item.seed for item in reset_info],
            "world_randomization": [item.randomization for item in reset_info],
        }
        return obs, info

    def _controls_from_actions(self, action_batch: np.ndarray) -> np.ndarray:
        action_batch = np.asarray(action_batch, dtype=np.float32)
        expected = (self.num_worlds, self.single_timestep_action_dim)
        if action_batch.shape != expected:
            raise ValueError(f"Expected batched actions with shape {expected}, got {action_batch.shape}")

        scaled = action_batch.copy()
        if self._gripper_indices:
            scaled[:, self._gripper_indices] *= _GRIPPER_CTRL_MAX

        ctrl = np.zeros((self.num_worlds, self.model.nu), dtype=np.float32)
        ctrl[:, self._ctrl_indices] = scaled
        return ctrl

    def step(self, action_batch: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        ctrl = self._controls_from_actions(action_batch)
        self._runtime.set_ctrl_batch(ctrl)
        self._runtime.step(nstep=self._control_decimation)
        self._step_count += 1

        info = {
            "sim_time": self._runtime.d_warp.time.numpy()[: self.num_worlds].copy()
            if hasattr(self._runtime.d_warp, "time")
            else None,
            "sim_step": self._step_count,
            "task_name": self._task,
        }
        task_eval = self.evaluate_task()
        reward = 0.0
        if task_eval is not None:
            reward = float(task_eval.reward.mean())
            info["task_reward"] = task_eval.reward.copy()
            info["task_success"] = task_eval.success.copy()
            info["task_eval"] = task_eval.to_info(squeeze=False)
        return self.get_obs(), reward, False, False, info

    def _state_batch(self) -> np.ndarray:
        qpos = self._runtime.d_warp.qpos.numpy()[: self.num_worlds].copy()
        return project_policy_state_batch(
            qpos,
            self._qpos_indices,
            self._gripper_indices,
            dtype=np.float32,
        )

    def _render_image_batch(self) -> dict[str, np.ndarray]:
        if not self._render_cameras or self._renderer is None:
            return {
                name: np.zeros(
                    (self.num_worlds, 3, self._camera_height, self._camera_width),
                    dtype=np.uint8,
                )
                for name in self.camera_names
            }

        if self._needs_renderer_reset:
            images = self._renderer.reset_numpy(actual_batch=self.num_worlds)
            self._needs_renderer_reset = False
        else:
            images = self._renderer.render_numpy(actual_batch=self.num_worlds)

        return {
            name: images[:, self._camera_index[name]].transpose(0, 3, 1, 2).copy()
            for name in self.camera_names
            if name in self._camera_index
        }

    def get_obs(self) -> dict[str, Any]:
        state = self._state_batch()
        images = self._render_image_batch()
        masks = {
            name: np.ones((self.num_worlds,), dtype=bool)
            for name in self.camera_names
        }
        if hasattr(self._runtime.d_warp, "time"):
            ts = self._runtime.d_warp.time.numpy()[: self.num_worlds].copy()
        else:
            ts = np.zeros((self.num_worlds,), dtype=np.float32)
        timestamps = {name: ts.copy() for name in self.camera_names}
        return {
            "images": images,
            "masks": masks,
            "state": state,
            "prompt": [self.prompt] * self.num_worlds,
            "camera_timestamps": timestamps,
        }

    def evaluate_task(self) -> TaskEvalResult | None:
        """Compute task reward/success for all live worlds."""

        if self._task_evaluator is None:
            return None
        qpos = self._runtime.d_warp.qpos.numpy()[: self.num_worlds].copy()
        return self._task_evaluator.evaluate_qpos_batch(qpos)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
        self.base_env.close()

    def rollout_metadata(self) -> dict[str, Any]:
        return {
            "task": self._task,
            "prompt": self.prompt,
            "camera_names": list(self.camera_names),
            "camera_height": self._camera_height,
            "camera_width": self._camera_width,
            "num_worlds": self.num_worlds,
            "step_count": self._step_count,
            "camera_backend": self._camera_backend,
            "camera_gpu_id": self._camera_gpu_id,
            "world_seeds": [item.seed for item in self._world_reset_info],
        }
