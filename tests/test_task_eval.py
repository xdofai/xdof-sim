from __future__ import annotations

import unittest

import mujoco
import numpy as np

import xdof_sim
from xdof_sim.task_eval import make_task_evaluator


class BottlesTaskEvalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.env = xdof_sim.make_env(task="bottles", render_cameras=False)
        cls.env.reset(seed=0, randomize=False)
        cls.evaluator = make_task_evaluator(cls.env.model, "bottles")
        if cls.evaluator is None:
            raise RuntimeError("Expected bottles task evaluator to be registered")

    @classmethod
    def tearDownClass(cls) -> None:
        cls.env.close()

    def _joint_adr(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        self.assertGreaterEqual(joint_id, 0, msg=f"Joint {joint_name!r} should exist")
        return int(self.env.model.jnt_qposadr[joint_id])

    def _set_joint_pose(
        self,
        qpos: np.ndarray,
        joint_name: str,
        *,
        pos: tuple[float, float, float],
        quat: tuple[float, float, float, float] | None = None,
    ) -> None:
        adr = self._joint_adr(joint_name)
        qpos[adr : adr + 3] = np.asarray(pos, dtype=np.float32)
        if quat is not None:
            qpos[adr + 3 : adr + 7] = np.asarray(quat, dtype=np.float32)

    def test_bottles_evaluator_counts_partial_and_full_success(self) -> None:
        self.evaluator.reset(nworld=1)
        qpos_batch = np.asarray(self.env.data.qpos, dtype=np.float32)[None, :].copy()
        self._set_joint_pose(qpos_batch[0], "bin_joint", pos=(0.70, 0.0, 0.76), quat=(1.0, 0.0, 0.0, 0.0))
        self._set_joint_pose(qpos_batch[0], "bottle_1_joint", pos=(0.75, 0.00, 0.82))
        self._set_joint_pose(qpos_batch[0], "bottle_2_joint", pos=(0.66, -0.02, 0.80))
        self._set_joint_pose(qpos_batch[0], "bottle_3_joint", pos=(0.30, 0.40, 0.80))
        self._set_joint_pose(qpos_batch[0], "bottle_4_joint", pos=(0.32, -0.45, 0.80))

        result = self.evaluator.evaluate_qpos_batch(qpos_batch)
        self.assertAlmostEqual(result.scalar_reward(), 1.0)
        self.assertTrue(result.scalar_success())

        info = result.to_info(squeeze=True)
        self.assertEqual(info["num_bottles_in_bin"], 2)
        self.assertEqual(set(info["bottles_in_bin"]), {"bottle_1", "bottle_2"})

        qpos_batch[0, self._joint_adr("bottle_2_joint") : self._joint_adr("bottle_2_joint") + 3] = np.asarray(
            (0.20, 0.20, 0.80),
            dtype=np.float32,
        )
        result = self.evaluator.evaluate_qpos_batch(qpos_batch)
        self.assertAlmostEqual(result.scalar_reward(), 0.5)
        self.assertFalse(result.scalar_success())
        self.assertEqual(result.to_info(squeeze=True)["num_bottles_in_bin"], 1)
        self.assertEqual(result.to_info(squeeze=True)["max_bottles_in_bin_so_far"], 2)
        self.assertTrue(result.to_info(squeeze=True)["ever_success"])

    def test_bottles_evaluator_exposes_curated_debug_spec(self) -> None:
        debug_spec = self.evaluator.debug_spec()
        self.assertIsNotNone(debug_spec)
        assert debug_spec is not None
        self.assertEqual(debug_spec["x_key"], "step")
        plot_keys = [plot["key"] for plot in debug_spec["plots"]]
        self.assertEqual(
            plot_keys,
            [
                "num_bottles_in_bin",
                "max_bottles_in_bin_so_far",
                "closest_radial_margin",
                "closest_height_margin",
                "reward",
                "ever_success",
                "success",
            ],
        )
        self.assertEqual(debug_spec["plots"][-1]["kind"], "bool")

    def test_bottles_evaluator_uses_bin_local_frame(self) -> None:
        qpos_batch = np.asarray(self.env.data.qpos, dtype=np.float32)[None, :].copy()
        yaw_90 = (np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5))
        bin_pos = np.asarray((0.70, 0.05, 0.76), dtype=np.float32)
        self._set_joint_pose(qpos_batch[0], "bin_joint", pos=tuple(bin_pos), quat=yaw_90)

        # Local offset (0.05, 0.00, 0.04) becomes world offset (0.00, 0.05, 0.04) under +90deg yaw.
        local_inside = np.asarray((0.05, 0.00, 0.04), dtype=np.float32)
        world_inside = bin_pos + np.asarray((0.00, 0.05, 0.04), dtype=np.float32)
        self._set_joint_pose(qpos_batch[0], "bottle_1_joint", pos=tuple(world_inside))
        self._set_joint_pose(qpos_batch[0], "bottle_2_joint", pos=(0.20, -0.20, 0.80))
        self._set_joint_pose(qpos_batch[0], "bottle_3_joint", pos=(0.25, 0.25, 0.80))
        self._set_joint_pose(qpos_batch[0], "bottle_4_joint", pos=(0.30, -0.30, 0.80))

        result = self.evaluator.evaluate_qpos_batch(qpos_batch)
        info = result.to_info(squeeze=True)
        self.assertEqual(info["num_bottles_in_bin"], 1)
        self.assertEqual(info["bottles_in_bin"], ["bottle_1"])

    def test_env_exposes_task_eval_from_current_state(self) -> None:
        self.env.reset(seed=0, randomize=False)
        self._set_joint_pose(self.env.data.qpos, "bin_joint", pos=(0.70, 0.0, 0.76), quat=(1.0, 0.0, 0.0, 0.0))
        self._set_joint_pose(self.env.data.qpos, "bottle_1_joint", pos=(0.75, 0.00, 0.82))
        self._set_joint_pose(self.env.data.qpos, "bottle_2_joint", pos=(0.66, -0.02, 0.80))
        mujoco.mj_forward(self.env.model, self.env.data)

        result = self.env.evaluate_task()
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.to_info(squeeze=True)["num_bottles_in_bin"], 2)

    def test_batched_qpos_evaluation_returns_per_world_results(self) -> None:
        qpos_batch = np.repeat(
            np.asarray(self.env.data.qpos, dtype=np.float32)[None, :],
            2,
            axis=0,
        )
        self._set_joint_pose(qpos_batch[0], "bin_joint", pos=(0.70, 0.0, 0.76), quat=(1.0, 0.0, 0.0, 0.0))
        self._set_joint_pose(qpos_batch[1], "bin_joint", pos=(0.70, 0.0, 0.76), quat=(1.0, 0.0, 0.0, 0.0))
        self._set_joint_pose(qpos_batch[0], "bottle_1_joint", pos=(0.75, 0.00, 0.82))
        self._set_joint_pose(qpos_batch[0], "bottle_2_joint", pos=(0.66, -0.02, 0.80))
        self._set_joint_pose(qpos_batch[1], "bottle_1_joint", pos=(0.75, 0.00, 0.82))

        result = self.evaluator.evaluate_qpos_batch(qpos_batch)
        self.assertEqual(result.reward.shape, (2,))
        self.assertTrue(bool(result.success[0]))
        self.assertFalse(bool(result.success[1]))
        self.assertEqual(result.to_info(squeeze=False)["num_bottles_in_bin"], [2, 1])


if __name__ == "__main__":
    unittest.main()
