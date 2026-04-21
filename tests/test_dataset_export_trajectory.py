from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from xdof_sim.dataset_export.metadata import (
    data_date_from_timestamps,
    episode_id_from_dir,
    normalize_task_name,
)
from xdof_sim.dataset_export.trajectory import (
    build_export_grid,
    build_export_trajectory,
    normalize_sim_timestamps,
)
from xdof_sim.env import project_policy_state
from xdof_sim.rendering.replay.types import EpisodeContext, EpisodeStreams


class FakeEnv:
    camera_names = ("top", "left", "right")

    def project_state_from_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return qpos[:4].astype(np.float32) * 2.0


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

    def test_build_export_trajectory_aligns_actions_and_qpos(self) -> None:
        traj = build_export_trajectory(make_context(), FakeEnv(), fps=4.0)

        self.assertEqual(traj.episode_id, "episode_123")
        self.assertEqual(traj.source_delivery, "sim_spell_cat")
        self.assertEqual(traj.task_name, "sim_spell_cat")
        self.assertEqual(traj.camera_names, ("top", "left", "right"))
        np.testing.assert_allclose(traj.timestamps, np.array([0.0, 0.25, 0.5, 0.75]))
        np.testing.assert_allclose(
            traj.actions,
            np.array(
                [
                    [1.0, 10.0, 4.0, 40.0],
                    [1.0, 10.0, 4.0, 40.0],
                    [2.0, 20.0, 5.0, 50.0],
                    [2.0, 20.0, 5.0, 50.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            traj.qpos,
            np.array(
                [
                    [0.1, 0.2, 0.3, 0.4, 9.0],
                    [0.1, 0.2, 0.3, 0.4, 9.0],
                    [0.5, 0.6, 0.7, 0.8, 9.0],
                    [0.5, 0.6, 0.7, 0.8, 9.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            traj.states,
            np.array(
                [
                    [0.2, 0.4, 0.6, 0.8],
                    [0.2, 0.4, 0.6, 0.8],
                    [1.0, 1.2, 1.4, 1.6],
                    [1.0, 1.2, 1.4, 1.6],
                ],
                dtype=np.float32,
            ),
        )


if __name__ == "__main__":
    unittest.main()
