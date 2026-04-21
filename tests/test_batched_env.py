from __future__ import annotations

import unittest

import numpy as np

from xdof_sim import make_batched_env
from xdof_sim.env import project_policy_state_batch


class BatchedEnvHelperTests(unittest.TestCase):
    def test_project_policy_state_batch_scales_grippers(self) -> None:
        qpos_batch = np.array(
            [
                [0.1, 0.2, 0.0475],
                [-0.3, 0.4, 0.02375],
            ],
            dtype=np.float32,
        )
        states = project_policy_state_batch(
            qpos_batch,
            qpos_indices=[0, 1, 2],
            gripper_indices=[2],
        )
        np.testing.assert_allclose(
            states,
            np.array(
                [
                    [0.1, 0.2, 1.0],
                    [-0.3, 0.4, 0.5],
                ],
                dtype=np.float32,
            ),
        )

    def test_make_batched_env_rejects_mujoco_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "only supports camera_backend='mjwarp' or 'madrona'"):
            make_batched_env(camera_backend="mujoco", num_worlds=2)


if __name__ == "__main__":
    unittest.main()
