from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from xdof_sim.dataset_export.metadata import (
    data_date_from_timestamps,
    episode_id_from_dir,
    normalize_task_name,
)
from xdof_sim.dataset_export.trajectory import (
    build_export_grid,
    build_export_trajectory,
    normalize_integration_timestamps,
    normalize_sim_timestamps,
)
from xdof_sim.env import project_policy_state
from xdof_sim.rendering.replay.types import EpisodeContext, EpisodeStreams


class FakeEnv:
    camera_names = ("top", "left", "right")

    def project_state_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return qpos[:4].astype(np.float32) * 2.0


class FakeIntegrationEnv(FakeEnv):
    def __init__(self) -> None:
        self.model = SimpleNamespace(nq=5)
        self.data = SimpleNamespace(qpos=np.zeros(5, dtype=np.float64))


def make_context() -> EpisodeContext:
    streams = EpisodeStreams(
        episode_dir=Path("/tmp/sim_spell_cat/episode_123"),
        actions_left=np.array(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
            ],
            dtype=np.float64,
        ),
        ts_left=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        actions_right=np.array(
            [
                [4.0, 40.0],
                [5.0, 50.0],
                [6.0, 60.0],
            ],
            dtype=np.float64,
        ),
        ts_right=np.array([0.0, 0.5, 1.0], dtype=np.float64),
        camera_frames={},
        camera_ts={},
    )
    return EpisodeContext(
        streams=streams,
        episode_format="delivered",
        scene="hybrid",
        task="blocks",
        instruction="sim_spell_cat",
        rand_state=None,
        raw_sim_states=np.array(
            [
                [0.1, 0.2, 0.3, 0.4, 9.0],
                [0.5, 0.6, 0.7, 0.8, 9.0],
                [0.9, 1.0, 1.1, 1.2, 9.0],
            ],
            dtype=np.float64,
        ),
        raw_sim_timestamps=np.array(
            [1_700_000_000.0, 1_700_000_000.5, 1_700_000_001.0],
            dtype=np.float64,
        ),
    )


class ProjectPolicyStateTests(unittest.TestCase):
    def test_project_policy_state_scales_grippers(self) -> None:
        qpos = np.array([1.0, 2.0, 0.02375, 0.0475], dtype=np.float64)
        state = project_policy_state(qpos, [0, 1, 2, 3], [2, 3])
        np.testing.assert_allclose(
            state,
            np.array([1.0, 2.0, 0.5, 1.0], dtype=np.float32),
        )


class MetadataHelperTests(unittest.TestCase):
    def test_episode_id_from_dir_preserves_existing_prefix(self) -> None:
        self.assertEqual(episode_id_from_dir(Path("/tmp/episode_abc")), "episode_abc")
        self.assertEqual(episode_id_from_dir(Path("/tmp/abc")), "episode_abc")

    def test_normalize_task_name(self) -> None:
        self.assertEqual(
            normalize_task_name("Sim Spell Cat", "blocks"),
            "sim_spell_cat",
        )
        self.assertEqual(normalize_task_name(None, "blocks"), "blocks")

    def test_data_date_from_unix_seconds(self) -> None:
        self.assertEqual(
            data_date_from_timestamps([1_700_000_000.0]),
            "nov_14",
        )
        self.assertEqual(data_date_from_timestamps([0.1]), "unknown")


class ExportTrajectoryTests(unittest.TestCase):
    def test_normalize_sim_timestamps_shifts_absolute_delivered_times(self) -> None:
        context = make_context()
        shifted = normalize_sim_timestamps(context)
        np.testing.assert_allclose(shifted, np.array([0.0, 0.5, 1.0]))

    def test_build_export_grid_uses_overlap_window(self) -> None:
        grid = build_export_grid(starts=[0.0, 0.1, 0.2], ends=[1.1, 0.8, 0.9], fps=4.0)
        np.testing.assert_allclose(grid, np.array([0.2, 0.45]))

    def test_build_export_trajectory_requires_integration_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires integration_state.npy"):
            build_export_trajectory(make_context(), FakeEnv(), fps=4.0)

    @mock.patch("xdof_sim.dataset_export.trajectory.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.dataset_export.trajectory.mujoco.mj_setState", autospec=True)
    @mock.patch("xdof_sim.dataset_export.trajectory.mujoco.mj_stateSize", autospec=True)
    def test_build_export_trajectory_prefers_integration_state_source(
        self,
        mj_state_size_mock,
        mj_set_state_mock,
        _mj_forward_mock,
    ) -> None:
        context = make_context()
        context = EpisodeContext(
            **{
                **context.__dict__,
                "raw_sim_integration_states": np.array(
                    [
                        [10.0, 100.0, 1000.0],
                        [20.0, 200.0, 2000.0],
                        [30.0, 300.0, 3000.0],
                    ],
                    dtype=np.float64,
                ),
                "raw_sim_integration_timestamps": np.array([0.918, 1.418, 1.918], dtype=np.float64),
                "raw_sim_integration_wallclock_timestamps": np.array(
                    [1_700_000_000.0, 1_700_000_000.5, 1_700_000_001.0],
                    dtype=np.float64,
                ),
                "raw_sim_state_spec": 16383,
            }
        )
        env = FakeIntegrationEnv()
        mj_state_size_mock.return_value = 3

        def _fake_set_state(_model, data, state, _spec):
            data.qpos[:] = np.array(
                [state[0], state[1], state[2], state[0] + 1.0, 9.0],
                dtype=np.float64,
            )

        mj_set_state_mock.side_effect = _fake_set_state

        traj = build_export_trajectory(context, env, fps=4.0)

        self.assertEqual(traj.state_source, "integration_state.npy")
        np.testing.assert_allclose(traj.timestamps, np.array([0.0, 0.25, 0.5, 0.75]))
        np.testing.assert_allclose(
            traj.qpos,
            np.array(
                [
                    [10.0, 100.0, 1000.0, 11.0, 9.0],
                    [10.0, 100.0, 1000.0, 11.0, 9.0],
                    [20.0, 200.0, 2000.0, 21.0, 9.0],
                    [20.0, 200.0, 2000.0, 21.0, 9.0],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(mj_set_state_mock.call_count, 4)

    def test_normalize_integration_timestamps_zero_bases_subsecond_delivered_sim_time(self) -> None:
        context = make_context()
        context = EpisodeContext(
            **{
                **context.__dict__,
                "raw_sim_integration_states": np.zeros((3, 3), dtype=np.float64),
                "raw_sim_integration_timestamps": np.array([0.918, 1.418, 1.918], dtype=np.float64),
                "raw_sim_integration_wallclock_timestamps": np.array(
                    [1_700_000_000.0, 1_700_000_000.5, 1_700_000_001.0],
                    dtype=np.float64,
                ),
                "raw_sim_state_spec": 16383,
            }
        )

        np.testing.assert_allclose(
            normalize_integration_timestamps(context),
            np.array([0.0, 0.5, 1.0], dtype=np.float64),
        )

    def test_delivered_integration_timestamps_reject_unrepaired_wallclock_stall(self) -> None:
        context = make_context()
        sim_timestamps = np.arange(12, dtype=np.float64) * 0.034
        wallclock_elapsed = sim_timestamps.copy()
        wallclock_elapsed[1:] += 1.2
        context = EpisodeContext(
            **{
                **context.__dict__,
                "raw_sim_integration_states": np.zeros((len(sim_timestamps), 3), dtype=np.float64),
                "raw_sim_integration_timestamps": sim_timestamps + 30.0,
                "raw_sim_integration_wallclock_timestamps": wallclock_elapsed + 1_700_000_000.0,
                "raw_sim_state_spec": 16383,
            }
        )

        with self.assertRaisesRegex(ValueError, "sim-time and wallclock elapsed clocks disagree"):
            normalize_integration_timestamps(context)

    def test_build_export_trajectory_rejects_integration_state_without_timestamps(self) -> None:
        context = make_context()
        context = EpisodeContext(
            **{
                **context.__dict__,
                "raw_sim_integration_states": np.zeros((3, 3), dtype=np.float64),
                "raw_sim_integration_timestamps": None,
                "raw_sim_state_spec": 16383,
            }
        )

        with self.assertRaisesRegex(ValueError, "requires integration_state_sim_time.npy"):
            build_export_trajectory(context, FakeIntegrationEnv(), fps=4.0)

    def test_delivered_integration_timestamps_require_wallclock_validation(self) -> None:
        context = make_context()
        context = EpisodeContext(
            **{
                **context.__dict__,
                "raw_sim_integration_states": np.zeros((3, 3), dtype=np.float64),
                "raw_sim_integration_timestamps": np.array([0.918, 1.418, 1.918], dtype=np.float64),
                "raw_sim_integration_wallclock_timestamps": None,
                "raw_sim_state_spec": 16383,
            }
        )

        with self.assertRaisesRegex(ValueError, "requires integration_state_wallclock.npy"):
            normalize_integration_timestamps(context)

    def test_delivered_integration_timestamps_reject_wallclock_length_mismatch(self) -> None:
        context = make_context()
        context = EpisodeContext(
            **{
                **context.__dict__,
                "raw_sim_integration_states": np.zeros((3, 3), dtype=np.float64),
                "raw_sim_integration_timestamps": np.array([0.918, 1.418, 1.918], dtype=np.float64),
                "raw_sim_integration_wallclock_timestamps": np.array(
                    [1_700_000_000.0, 1_700_000_000.5],
                    dtype=np.float64,
                ),
                "raw_sim_state_spec": 16383,
            }
        )

        with self.assertRaisesRegex(ValueError, "wallclock length mismatch"):
            normalize_integration_timestamps(context)


if __name__ == "__main__":
    unittest.main()
