"""Bottle-in-bin task evaluator."""

from __future__ import annotations

import re
from typing import Any

import mujoco
import numpy as np

from xdof_sim.task_eval.base import TaskEvalResult
from xdof_sim.task_eval.debug_spec import EvalDebugSpec, PlotSpec, ThresholdSpec
from xdof_sim.task_specs import SimTaskSpec


def _quat_to_rotmat_batch(quat_batch: np.ndarray) -> np.ndarray:
    """Convert wxyz quaternions into rotation matrices."""

    quat_batch = np.asarray(quat_batch, dtype=np.float32)
    if quat_batch.ndim != 2 or quat_batch.shape[1] != 4:
        raise ValueError(f"Expected quaternion batch shape (B, 4), got {quat_batch.shape}")

    norm = np.linalg.norm(quat_batch, axis=1, keepdims=True)
    norm = np.where(norm > 0.0, norm, 1.0)
    q = quat_batch / norm
    w, x, y, z = q.T

    return np.stack(
        [
            np.stack([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)], axis=-1),
            np.stack([2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)], axis=-1),
            np.stack([2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)], axis=-1),
        ],
        axis=1,
    ).astype(np.float32)


class BottlesInBinEvaluator:
    """Score the bottles task from freejoint positions in the bin's local frame."""

    def __init__(
        self,
        *,
        model: mujoco.MjModel,
        spec: SimTaskSpec,
        success_count: int = 2,
    ) -> None:
        self.model = model
        self.spec = spec
        self.success_count = int(success_count)
        if self.success_count < 1:
            raise ValueError(f"success_count must be >= 1, got {self.success_count}")

        self.bottle_names, self._bottle_qpos_addrs = self._resolve_bottle_qpos_addrs(model)
        self._bin_qpos_adr = self._resolve_joint_qpos_adr(model, "bin_joint")
        self._bin_apothem = self._resolve_bin_apothem(model)
        self._bin_bottom_z = self._resolve_bin_bottom_z(model)
        self._bin_top_z = self._resolve_bin_top_z(model)
        self._nworld = 1
        self._max_bottles_in_bin = np.zeros((1,), dtype=np.int32)
        self._ever_success = np.zeros((1,), dtype=bool)

    def reset(self, *, nworld: int = 1) -> None:
        self._nworld = int(nworld)
        self._max_bottles_in_bin = np.zeros((self._nworld,), dtype=np.int32)
        self._ever_success = np.zeros((self._nworld,), dtype=bool)

    def debug_spec(self) -> dict[str, Any] | None:
        return EvalDebugSpec(
            plots=[
                PlotSpec(
                    key="num_bottles_in_bin",
                    title="Bottles In Bin",
                    color="#4fc3f7",
                    thresholds=[
                        ThresholdSpec(
                            value=float(self.success_count),
                            label=f"success >= {self.success_count}",
                            direction="gt",
                        )
                    ],
                ),
                PlotSpec(
                    key="max_bottles_in_bin_so_far",
                    title="Max Bottles So Far",
                    color="#7dd3fc",
                    thresholds=[
                        ThresholdSpec(
                            value=float(self.success_count),
                            label=f"ever hit {self.success_count}",
                            direction="gt",
                        )
                    ],
                ),
                PlotSpec(
                    key="closest_radial_margin",
                    title="Closest Radial Margin (m)",
                    color="#f59e0b",
                    thresholds=[
                        ThresholdSpec(value=0.0, label="inside footprint", direction="gt")
                    ],
                ),
                PlotSpec(
                    key="closest_height_margin",
                    title="Closest Height Margin (m)",
                    color="#a78bfa",
                    thresholds=[
                        ThresholdSpec(value=0.0, label="inside height band", direction="gt")
                    ],
                ),
                PlotSpec(
                    key="reward",
                    title="Reward",
                    color="#22c55e",
                    thresholds=[
                        ThresholdSpec(value=1.0, label="full reward", direction="gt")
                    ],
                ),
                PlotSpec(
                    key="ever_success",
                    title="Ever Success",
                    color="#34d399",
                    kind="bool",
                ),
                PlotSpec(
                    key="success",
                    title="Current Success",
                    color="#fb7185",
                    kind="bool",
                ),
            ]
        ).to_dict()

    @staticmethod
    def _resolve_joint_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint {joint_name!r} not found in model")
        return int(model.jnt_qposadr[joint_id])

    @staticmethod
    def _resolve_bottle_qpos_addrs(model: mujoco.MjModel) -> tuple[list[str], np.ndarray]:
        entries: list[tuple[int, str, int]] = []
        for joint_id in range(model.njnt):
            name = model.jnt(joint_id).name
            if not name:
                continue
            match = re.fullmatch(r"bottle_(\d+)_joint", name)
            if match is None:
                continue
            bottle_index = int(match.group(1))
            entries.append(
                (
                    bottle_index,
                    f"bottle_{bottle_index}",
                    int(model.jnt_qposadr[joint_id]),
                )
            )

        if not entries:
            raise ValueError("No bottle_*_joint freejoints found in model")

        entries.sort(key=lambda item: item[0])
        bottle_names = [name for _, name, _ in entries]
        addrs = np.asarray([adr for _, _, adr in entries], dtype=np.int32)
        return bottle_names, addrs

    @staticmethod
    def _resolve_bin_apothem(model: mujoco.MjModel) -> float:
        wall_distances: list[float] = []
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name
            if name and name.startswith("bin_wall_"):
                wall_distances.append(float(np.linalg.norm(model.geom_pos[geom_id][:2])))
        if not wall_distances:
            raise ValueError("No bin wall geoms found in model")
        return float(max(wall_distances))

    @staticmethod
    def _resolve_bin_bottom_z(model: mujoco.MjModel) -> float:
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bin_bottom")
        if geom_id < 0:
            return 0.0
        return float(model.geom_pos[geom_id][2])

    @staticmethod
    def _resolve_bin_top_z(model: mujoco.MjModel) -> float:
        top_values: list[float] = []
        for geom_id in range(model.ngeom):
            name = model.geom(geom_id).name
            if not name or not name.startswith("bin_wall_"):
                continue
            geom_type = int(model.geom_type[geom_id])
            if geom_type != mujoco.mjtGeom.mjGEOM_BOX:
                continue
            top_values.append(float(model.geom_pos[geom_id][2] + model.geom_size[geom_id][2]))

        if not top_values:
            raise ValueError("No bin wall box geoms found in model")
        return float(max(top_values))

    def evaluate_qpos_batch(self, qpos_batch: np.ndarray) -> TaskEvalResult:
        qpos_batch = np.asarray(qpos_batch, dtype=np.float32)
        if qpos_batch.ndim != 2 or qpos_batch.shape[1] != self.model.nq:
            raise ValueError(
                f"Expected qpos batch shape (B, {self.model.nq}), got {qpos_batch.shape}"
            )
        if qpos_batch.shape[0] != self._nworld:
            self.reset(nworld=qpos_batch.shape[0])

        bin_pos = qpos_batch[:, self._bin_qpos_adr : self._bin_qpos_adr + 3]
        bin_quat = qpos_batch[:, self._bin_qpos_adr + 3 : self._bin_qpos_adr + 7]

        bottle_positions = np.stack(
            [qpos_batch[:, adr : adr + 3] for adr in self._bottle_qpos_addrs],
            axis=1,
        )
        rel_world = bottle_positions - bin_pos[:, None, :]
        rot_world_from_local = _quat_to_rotmat_batch(bin_quat)
        rel_local = np.einsum("bij,bnj->bni", np.swapaxes(rot_world_from_local, 1, 2), rel_world)

        radial_xy = np.linalg.norm(rel_local[..., :2], axis=-1)
        local_z = rel_local[..., 2]
        radial_margin = self._bin_apothem - radial_xy
        lower_height_margin = local_z - self._bin_bottom_z
        upper_height_margin = self._bin_top_z - local_z
        height_margin = np.minimum(lower_height_margin, upper_height_margin)
        in_bin_mask = (
            (radial_margin >= 0.0)
            & (lower_height_margin >= 0.0)
            & (upper_height_margin >= 0.0)
        )

        num_bottles_in_bin = in_bin_mask.sum(axis=1).astype(np.int32)
        self._max_bottles_in_bin = np.maximum(self._max_bottles_in_bin, num_bottles_in_bin)
        reward = np.clip(
            num_bottles_in_bin.astype(np.float32) / float(self.success_count),
            0.0,
            1.0,
        )
        success = num_bottles_in_bin >= self.success_count
        self._ever_success |= success
        bottles_in_bin = [
            [name for name, in_bin in zip(self.bottle_names, world_mask) if bool(in_bin)]
            for world_mask in in_bin_mask
        ]

        metrics: dict[str, Any] = {
            "num_bottles_in_bin": num_bottles_in_bin,
            "max_bottles_in_bin_so_far": self._max_bottles_in_bin.copy(),
            "ever_success": self._ever_success.copy(),
            "closest_radial_margin": radial_margin.max(axis=1).astype(np.float32),
            "closest_height_margin": height_margin.max(axis=1).astype(np.float32),
            "bottle_in_bin_mask": in_bin_mask,
            "bottles_in_bin": bottles_in_bin,
            "bottle_names": list(self.bottle_names),
            "success_count": self.success_count,
            "bin_apothem": self._bin_apothem,
            "bin_bottom_z": self._bin_bottom_z,
            "bin_top_z": self._bin_top_z,
        }
        return TaskEvalResult(reward=reward, success=success, metrics=metrics)
