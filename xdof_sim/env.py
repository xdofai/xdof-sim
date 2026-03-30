"""MuJoCo simulation environment for bimanual YAM robot."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

from xdof_sim.config import RobotSystemConfig

# Path to scene XMLs (override with MUJOCO_SCENE_XML env var)
_MODELS_DIR = Path(__file__).parent / "models"
_SCENE_XMLS = {
    "bottles": _MODELS_DIR / "yam_bimanual_scene.xml",
    "marker": _MODELS_DIR / "yam_marker_scene.xml",
    "ball_sorting": _MODELS_DIR / "yam_ball_sorting_scene.xml",
    "empty": _MODELS_DIR / "yam_bimanual_empty.xml",
    "dishrack": _MODELS_DIR / "yam_dishwasher_scene.xml",
    "chess": _MODELS_DIR / "yam_chess_scene.xml",
    "chess2": _MODELS_DIR / "yam_chess2_scene.xml",
    "blocks": _MODELS_DIR / "yam_blocks_scene.xml",
    "mug_tree": _MODELS_DIR / "yam_mug_tree_scene.xml",
    "mug_flip": _MODELS_DIR / "yam_mug_flip_scene.xml",
    "pour": _MODELS_DIR / "yam_pour_screw_scene.xml",
    "spelling": _MODELS_DIR / "yam_block_spelling_scene.xml",
    "drawer": _MODELS_DIR / "yam_drawer.xml"
}
_DEFAULT_SCENE_XML = Path(
    os.environ.get("MUJOCO_SCENE_XML", str(_SCENE_XMLS["bottles"]))
)

# Gripper actuator ctrl range max (from menagerie model)
_GRIPPER_CTRL_MAX = 0.0475


class MuJoCoYAMEnv(gym.Env):
    """MuJoCo-based simulation of the bimanual YAM robot.

    Actions and observations use the same 14D format as the real robot:
    [left_j1..6, left_grip, right_j1..6, right_grip].
    """

    def __init__(
        self,
        config: RobotSystemConfig,
        chunk_dim: int = 30,
        prompt: str = "fold the towel",
        render_cameras: bool = True,
        camera_height: int = 480,
        camera_width: int = 640,
        physics_dt: float = 0.002,
        control_decimation: int = 17,  # 0.002 * 17 ≈ 0.034s ≈ 30Hz
        scene_xml: str | Path | None = None,
    ):
        super().__init__()
        self.config = config
        self.chunk_dim = chunk_dim
        self.prompt = prompt
        self._render_cameras_flag = render_cameras
        self._camera_height = camera_height
        self._camera_width = camera_width
        self._physics_dt = physics_dt
        self._control_decimation = control_decimation
        self._scene_xml = Path(scene_xml) if scene_xml else _DEFAULT_SCENE_XML
        self._task: str = ""  # set by make_env after construction

        # Populated by reset() when randomize=True; readable by callers.
        self._last_randomization: Any = None

        self.camera_names = list(config.cameras.keys())
        self.robot_names = list(config.robots.keys())
        self.single_timestep_action_dim = 7 * len(config.robots)  # 14
        self.action_dim = self.single_timestep_action_dim * self.chunk_dim

        # Observation space
        self.observation_space = spaces.Dict(
            {
                "images": spaces.Dict(
                    {
                        name: spaces.Box(
                            low=0,
                            high=255,
                            shape=(
                                config.cameras[name].height,
                                config.cameras[name].width,
                                3,
                            ),
                            dtype=np.uint8,
                        )
                        for name in self.camera_names
                    }
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.single_timestep_action_dim,),
                    dtype=np.float32,
                ),
            }
        )

        # Action space: (chunk_dim, 14)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.chunk_dim, self.single_timestep_action_dim),
            dtype=np.float32,
        )

        self.setup_model()
        self.cur_step = 0

    # ------------------------------------------------------------------
    # MuJoCo model setup
    # ------------------------------------------------------------------

    def setup_model(self):
        self.model = mujoco.MjModel.from_xml_path(str(self._scene_xml))
        self.model.opt.timestep = self._physics_dt
        self.data = mujoco.MjData(self.model)

        if self._render_cameras_flag:
            self.renderer = mujoco.Renderer(
                self.model,
                height=self._camera_height,
                width=self._camera_width,
            )

        self._build_index_maps()

    def _build_index_maps(self):
        """Build mappings from 14D state/action to MuJoCo qpos/ctrl indices."""
        self._qpos_indices: list[int] = []
        self._ctrl_indices: list[int] = []
        self._gripper_indices: list[int] = []  # which of the 14 are grippers

        idx = 0
        for robot_name in self.robot_names:
            # 6 arm joints
            for j in range(1, 7):
                joint_name = f"{robot_name}_joint{j}"
                jnt_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                self._qpos_indices.append(self.model.jnt_qposadr[jnt_id])

                act_name = f"{robot_name}_joint{j}"
                act_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name
                )
                self._ctrl_indices.append(act_id)
                idx += 1

            # Gripper (left_finger joint, gripper actuator)
            finger_name = f"{robot_name}_left_finger"
            finger_jnt_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, finger_name
            )
            self._qpos_indices.append(self.model.jnt_qposadr[finger_jnt_id])

            grip_act_name = f"{robot_name}_gripper"
            grip_act_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, grip_act_name
            )
            self._ctrl_indices.append(grip_act_id)
            self._gripper_indices.append(idx)
            idx += 1

        # Convert to sets for fast lookup
        self._gripper_set = set(self._gripper_indices)

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
        randomize: bool = True,
    ):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        init_q = self.get_init_q()  # 14D
        self._set_qpos_from_state(init_q)
        mujoco.mj_forward(self.model, self.data)

        # Object randomization: perturb free-jointed scene objects.
        self._last_randomization = None
        if randomize and self._task:
            from xdof_sim.randomization import TASK_RANDOMIZERS
            randomizer = TASK_RANDOMIZERS.get(self._task)
            if randomizer is not None:
                self._last_randomization = randomizer.randomize(
                    self.model, self.data, seed=seed
                )

        self.cur_step = 0
        return self.get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(
            self.chunk_dim, self.single_timestep_action_dim
        )
        all_obs = []
        for i in range(self.chunk_dim):
            self._step_single(action[i])
            all_obs.append(self.get_obs())

        final_obs = all_obs[-1]
        chunk_history = self._stack_obs(all_obs)
        return final_obs, chunk_history, 0.0, False, False, {}

    def close(self):
        if hasattr(self, "renderer"):
            self.renderer.close()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_qpos_from_state(self, state: np.ndarray):
        """Write a 14D state vector into MuJoCo qpos."""
        for i, qpos_idx in enumerate(self._qpos_indices):
            val = state[i]
            if i in self._gripper_set:
                # Scale [0, 1] policy space → MuJoCo joint space
                val = val * _GRIPPER_CTRL_MAX
            self.data.qpos[qpos_idx] = val

    def _step_single(self, action_14d: np.ndarray):
        """Apply a single 14D action and step physics."""
        ctrl = np.zeros(self.model.nu)
        for i in range(self.single_timestep_action_dim):
            val = action_14d[i]
            if i in self._gripper_set:
                # Scale [0, 1] policy space → actuator ctrl range
                val = val * _GRIPPER_CTRL_MAX
            ctrl[self._ctrl_indices[i]] = val
        self.data.ctrl[:] = ctrl

        for _ in range(self._control_decimation):
            mujoco.mj_step(self.model, self.data)

    def get_obs(self) -> dict[str, Any]:
        """Return observation dict with state, images, and metadata."""
        # State: read qpos for the 14 controlled joints
        state = np.zeros(self.single_timestep_action_dim, dtype=np.float32)
        for i, qpos_idx in enumerate(self._qpos_indices):
            val = self.data.qpos[qpos_idx]
            if i in self._gripper_set:
                # MuJoCo joint space → [0, 1] policy space
                val = np.clip(val / _GRIPPER_CTRL_MAX, 0.0, 1.0)
            state[i] = val

        # Images
        if self._render_cameras_flag:
            images = self._render_cameras()
        else:
            images = {
                name: np.zeros(
                    (3, self._camera_height, self._camera_width), dtype=np.uint8
                )
                for name in self.camera_names
            }

        masks = {name: True for name in self.camera_names}
        sim_time = self.data.time
        timestamps = {name: sim_time for name in self.camera_names}

        return {
            "images": images,
            "masks": masks,
            "state": state,
            "prompt": self.prompt,
            "camera_timestamps": timestamps,
        }

    def _render_cameras(self) -> dict[str, np.ndarray]:
        """Render all cameras and return (3, H, W) uint8 images."""
        images = {}
        for name in self.camera_names:
            self.renderer.update_scene(self.data, camera=name)
            img = self.renderer.render()  # (H, W, 3) uint8
            images[name] = img.transpose(2, 0, 1).copy()  # → (3, H, W)
        return images

    def get_init_q(self) -> np.ndarray:
        """Return concatenated init_q across all robots as a flat 14D array."""
        return np.concatenate(
            [np.array(self.config.robots[name].init_q) for name in self.robot_names]
        )

    def _stack_obs(self, obses: list[dict]) -> dict:
        """Stack a list of observations into chunked arrays."""
        stacked: dict[str, Any] = {}
        for key in obses[0]:
            if isinstance(obses[0][key], str):
                stacked[key] = obses[0][key]
            elif isinstance(obses[0][key], dict):
                stacked[key] = {
                    k: np.stack([obs[key][k] for obs in obses])
                    for k in obses[0][key]
                }
            else:
                stacked[key] = np.stack([obs[key] for obs in obses])
        return stacked
