from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from xdof_sim.rendering.replay.episode import (
    _load_direct_actions,
    _load_delivered_episode_streams,
    detect_episode_format,
    load_integration_states,
    load_episode_context,
)
from xdof_sim.rendering.replay.runtime import create_replay_env
from xdof_sim.rendering.replay.types import EpisodeContext, EpisodeStreams
from xdof_sim.rendering.replay.session import ReplaySession
from xdof_sim.rendering.replay.timeline import build_replay_timeline
from xdof_sim.rendering.replay.video import (
    _actions_to_ctrl_batch,
    _infer_physics_substeps_per_action,
    _qpos_frames_from_integration_states,
    _sample_qpos_frames_for_video,
    collect_physics_rollout_qpos_frames,
)
from xdof_sim.teleop.episode_recorder import TeleopEpisodeRecorder


def _ts(seconds: float) -> SimpleNamespace:
    whole = int(seconds)
    nanos = int(round((seconds - whole) * 1e9))
    return SimpleNamespace(seconds=whole, nanos=nanos)


def _topic_msg(topic: str, *, position: list[float] | None = None, data: str | None = None, timestamp: float | None = None):
    decoded_fields: dict[str, object] = {}
    if position is not None:
        decoded_fields["position"] = position
    if data is not None:
        decoded_fields["data"] = data
    if timestamp is not None:
        decoded_fields["timestamp"] = _ts(timestamp)
    decoded = SimpleNamespace(**decoded_fields)
    return (None, SimpleNamespace(topic=topic), None, decoded)


class _FakeDecodedReader:
    def __init__(self, messages) -> None:
        self._messages = list(messages)

    def iter_decoded_messages(self):
        return iter(self._messages)


class DatasetEpisodeContextTests(unittest.TestCase):
    def _fake_delivered_streams(self, episode_dir: Path) -> EpisodeStreams:
        return EpisodeStreams(
            episode_dir=episode_dir,
            actions_left=np.zeros((2, 7), dtype=np.float64),
            ts_left=np.array([0.0, 0.1], dtype=np.float64),
            actions_right=np.zeros((2, 7), dtype=np.float64),
            ts_right=np.array([0.0, 0.1], dtype=np.float64),
            camera_frames={},
            camera_ts={},
        )

    @mock.patch("xdof_sim.rendering.replay.episode.load_randomization", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.episode.read_sim_physics_overrides", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.episode.read_sim_config", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.episode.load_sim_states", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.episode.load_episode_streams", autospec=True)
    def test_load_raw_episode_context_aligns_actions_to_recorded_sim_steps(
        self,
        load_streams_mock,
        load_sim_states_mock,
        read_sim_config_mock,
        read_sim_physics_mock,
        load_randomization_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "raw_episode"
            episode_dir.mkdir(parents=True)

            streams = EpisodeStreams(
                episode_dir=episode_dir,
                actions_left=np.array(
                    [
                        [1, 2, 3, 4, 5, 6, 7],
                        [11, 12, 13, 14, 15, 16, 17],
                    ],
                    dtype=np.float64,
                ),
                ts_left=np.array([0.0, 0.1], dtype=np.float64),
                actions_right=np.array(
                    [
                        [21, 22, 23, 24, 25, 26, 27],
                        [31, 32, 33, 34, 35, 36, 37],
                    ],
                    dtype=np.float64,
                ),
                ts_right=np.array([0.0, 0.2], dtype=np.float64),
                camera_frames={},
                camera_ts={},
            )
            raw_qpos = np.array(
                [
                    np.arange(86, dtype=np.float64),
                    np.arange(86, dtype=np.float64) + 100.0,
                    np.arange(86, dtype=np.float64) + 200.0,
                ]
            )
            raw_ts = np.array([0.05, 0.10, 0.15], dtype=np.float64)

            load_streams_mock.return_value = streams
            load_sim_states_mock.return_value = (raw_qpos, raw_ts)
            read_sim_config_mock.return_value = {"scene": "hybrid", "task": "sweep"}
            read_sim_physics_mock.return_value = {"physics_dt": 0.0001, "control_decimation": 17}
            load_randomization_mock.return_value = None

            context = load_episode_context(episode_dir, load_recorded_cameras=False)

            self.assertEqual(context.episode_format, "raw")
            self.assertEqual(context.replay_state_alignment, "initial")
            np.testing.assert_allclose(context.initial_scene_qpos, raw_qpos[0].astype(np.float32))
            self.assertEqual(context.physics_overrides, {"physics_dt": 0.0001, "control_decimation": 17})
            np.testing.assert_allclose(context.raw_sim_states, raw_qpos)
            np.testing.assert_allclose(context.replay_timestamps, raw_ts)
            np.testing.assert_allclose(
                context.replay_actions,
                np.array(
                    [
                        [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27],
                        [11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25, 26, 27],
                    ],
                    dtype=np.float32,
                ),
            )

            timeline = build_replay_timeline(context, control_hz=999.0)
            np.testing.assert_allclose(timeline.grid_ts, raw_ts)
            np.testing.assert_allclose(timeline.actions, context.replay_actions)
            np.testing.assert_allclose(timeline.sim_states, raw_qpos)
            self.assertEqual(timeline.sim_state_alignment, "initial")

    def test_load_dataset_episode_context_uses_states_actions_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "sim_tasks_test_224" / "episode_abc"
            episode_dir.mkdir(parents=True)

            arr = np.array(
                [
                    list(range(14)) + list(range(100, 114)),
                    list(range(14, 28)) + list(range(114, 128)),
                ],
                dtype=np.float64,
            )
            (episode_dir / "states_actions.bin").write_bytes(arr.tobytes())
            np.save(episode_dir / "initial_qpos.npy", np.array([9.0, 8.0, 7.0], dtype=np.float32))
            (episode_dir / "episode_metadata.json").write_text(
                json.dumps(
                    {
                        "scene": "training",
                        "task": "bottles",
                        "instruction": "throw plastic bottles in bin",
                        "fps": 5.0,
                        "cameras": ["top", "left", "right"],
                        "initial_qpos_file": "initial_qpos.npy",
                    }
                )
            )

            self.assertEqual(detect_episode_format(episode_dir), "dataset")

            context = load_episode_context(episode_dir, load_recorded_cameras=False)
            self.assertEqual(context.episode_format, "dataset")
            self.assertEqual(context.scene, "training")
            self.assertEqual(context.task, "bottles")
            self.assertEqual(context.instruction, "throw plastic bottles in bin")
            self.assertEqual(context.replay_state_kind, "policy_state")
            np.testing.assert_allclose(context.initial_scene_qpos, np.array([9.0, 8.0, 7.0], dtype=np.float32))
            np.testing.assert_allclose(context.replay_timestamps, np.array([0.0, 0.2]))
            np.testing.assert_allclose(context.raw_sim_states, arr[:, :14].astype(np.float32))
            np.testing.assert_allclose(context.replay_actions, arr[:, 14:].astype(np.float32))
            np.testing.assert_allclose(context.streams.actions_left, arr[:, 14:21].astype(np.float32))
            np.testing.assert_allclose(context.streams.actions_right, arr[:, 21:28].astype(np.float32))

            timeline = build_replay_timeline(context, control_hz=123.0)
            np.testing.assert_allclose(timeline.grid_ts, np.array([0.0, 0.2]))
            np.testing.assert_allclose(timeline.actions, arr[:, 14:].astype(np.float32))
            np.testing.assert_allclose(timeline.sim_states, arr[:, :14].astype(np.float32))
            self.assertEqual(timeline.sim_state_kind, "policy_state")

    def test_load_recorded_episode_context_uses_exact_actions_and_qpos(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "recorded_episode"
            episode_dir.mkdir(parents=True)

            actions = np.array(
                [
                    np.arange(14, dtype=np.float32),
                    np.arange(14, dtype=np.float32) + 10.0,
                ],
                dtype=np.float32,
            )
            qpos = np.array(
                [
                    np.arange(32, dtype=np.float32),
                    np.arange(32, dtype=np.float32) + 100.0,
                    np.arange(32, dtype=np.float32) + 200.0,
                ],
                dtype=np.float32,
            )
            np.save(episode_dir / "actions_full.npy", actions)
            np.save(episode_dir / "qpos.npy", qpos)
            np.save(episode_dir / "initial_qpos.npy", qpos[0])
            np.save(episode_dir / "action_timestamps.npy", np.array([0.1, 0.2], dtype=np.float64))
            np.save(episode_dir / "qpos_timestamps.npy", np.array([0.0, 0.1, 0.2], dtype=np.float64))
            (episode_dir / "config.json").write_text(
                json.dumps(
                    {
                        "scene": "hybrid",
                        "task": "bottles",
                        "prompt": "throw plastic bottles in bin",
                        "control_rate": 10.0,
                    }
                )
            )

            self.assertEqual(detect_episode_format(episode_dir), "recorded")

            context = load_episode_context(episode_dir, load_recorded_cameras=False)
            self.assertEqual(context.episode_format, "recorded")
            self.assertEqual(context.scene, "hybrid")
            self.assertEqual(context.task, "bottles")
            self.assertEqual(context.instruction, "throw plastic bottles in bin")
            self.assertEqual(context.replay_state_kind, "qpos")
            np.testing.assert_allclose(context.initial_scene_qpos, qpos[0])
            np.testing.assert_allclose(context.raw_sim_states, qpos)
            np.testing.assert_allclose(context.raw_sim_timestamps, np.array([0.0, 0.1, 0.2]))
            self.assertIsNone(context.raw_sim_ctrls)
            self.assertIsNone(context.raw_sim_qvels)
            self.assertIsNone(context.initial_scene_qvel)
            self.assertIsNone(context.replay_ctrls)
            np.testing.assert_allclose(context.replay_actions, actions)
            np.testing.assert_allclose(context.replay_timestamps, np.array([0.0, 0.1, 0.2]))
            np.testing.assert_allclose(context.streams.actions_left, actions[:, :7])
            np.testing.assert_allclose(context.streams.actions_right, actions[:, 7:])
            np.testing.assert_allclose(context.streams.ts_left, np.array([0.1, 0.2]))

            timeline = build_replay_timeline(context, control_hz=123.0)
            np.testing.assert_allclose(timeline.actions, actions)
            np.testing.assert_allclose(timeline.grid_ts, np.array([0.0, 0.1, 0.2]))
            np.testing.assert_allclose(timeline.sim_states, qpos)
            self.assertEqual(timeline.sim_state_kind, "qpos")

    def test_load_recorded_episode_context_loads_optional_dynamic_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "recorded_episode_dynamic"
            episode_dir.mkdir(parents=True)

            actions = np.array(
                [
                    np.arange(14, dtype=np.float32),
                    np.arange(14, dtype=np.float32) + 10.0,
                ],
                dtype=np.float32,
            )
            qpos = np.array(
                [
                    np.arange(32, dtype=np.float64),
                    np.arange(32, dtype=np.float64) + 100.0,
                    np.arange(32, dtype=np.float64) + 200.0,
                ],
                dtype=np.float64,
            )
            qvel = np.array(
                [
                    np.arange(24, dtype=np.float64) + 1.0,
                    np.arange(24, dtype=np.float64) + 11.0,
                    np.arange(24, dtype=np.float64) + 21.0,
                ],
                dtype=np.float64,
            )
            ctrl = np.array(
                [
                    np.arange(14, dtype=np.float64) + 1000.0,
                    np.arange(14, dtype=np.float64) + 1010.0,
                    np.arange(14, dtype=np.float64) + 1020.0,
                ],
                dtype=np.float64,
            )
            act = np.array(
                [
                    np.array([0.1, 0.2, 0.3], dtype=np.float64),
                    np.array([0.4, 0.5, 0.6], dtype=np.float64),
                    np.array([0.7, 0.8, 0.9], dtype=np.float64),
                ],
                dtype=np.float64,
            )
            mocap_pos = np.array(
                [
                    [[1.0, 2.0, 3.0]],
                    [[4.0, 5.0, 6.0]],
                    [[7.0, 8.0, 9.0]],
                ],
                dtype=np.float64,
            )
            mocap_quat = np.array(
                [
                    [[1.0, 0.0, 0.0, 0.0]],
                    [[0.0, 1.0, 0.0, 0.0]],
                    [[0.0, 0.0, 1.0, 0.0]],
                ],
                dtype=np.float64,
            )
            integration_state = np.array(
                [
                    np.arange(9, dtype=np.float64),
                    np.arange(9, dtype=np.float64) + 1000.0,
                    np.arange(9, dtype=np.float64) + 2000.0,
                ]
            )

            np.save(episode_dir / "actions_full.npy", actions)
            np.save(episode_dir / "integration_state.npy", integration_state)
            np.save(episode_dir / "qpos.npy", qpos)
            np.save(episode_dir / "ctrl.npy", ctrl)
            np.save(episode_dir / "qvel.npy", qvel)
            np.save(episode_dir / "act.npy", act)
            np.save(episode_dir / "mocap_pos.npy", mocap_pos)
            np.save(episode_dir / "mocap_quat.npy", mocap_quat)
            np.save(episode_dir / "initial_qpos.npy", qpos[0])
            np.save(episode_dir / "initial_qvel.npy", qvel[0])
            np.save(episode_dir / "initial_act.npy", act[0])
            np.save(episode_dir / "initial_mocap_pos.npy", mocap_pos[0])
            np.save(episode_dir / "initial_mocap_quat.npy", mocap_quat[0])
            np.save(episode_dir / "action_timestamps.npy", np.array([0.1, 0.2], dtype=np.float64))
            np.save(episode_dir / "qpos_timestamps.npy", np.array([0.0, 0.1, 0.2], dtype=np.float64))
            (episode_dir / "config.json").write_text(
                json.dumps(
                    {
                        "scene": "hybrid",
                        "task": "bottles",
                        "prompt": "throw plastic bottles in bin",
                        "control_rate": 10.0,
                        "mj_state_spec": 16383,
                    }
                )
            )

            context = load_episode_context(episode_dir, load_recorded_cameras=False)
            np.testing.assert_allclose(context.raw_sim_integration_states, integration_state)
            self.assertEqual(context.raw_sim_state_spec, 16383)
            np.testing.assert_allclose(context.raw_sim_ctrls, ctrl)
            np.testing.assert_allclose(context.raw_sim_qvels, qvel)
            np.testing.assert_allclose(context.raw_sim_acts, act)
            np.testing.assert_allclose(context.raw_sim_mocap_pos, mocap_pos)
            np.testing.assert_allclose(context.raw_sim_mocap_quat, mocap_quat)
            np.testing.assert_allclose(context.initial_scene_integration_state, integration_state[0])
            np.testing.assert_allclose(context.replay_ctrls, ctrl[1:])
            self.assertEqual(context.raw_sim_states.dtype, np.float64)
            self.assertEqual(context.initial_scene_qpos.dtype, np.float64)
            self.assertEqual(context.raw_sim_ctrls.dtype, np.float64)
            self.assertEqual(context.raw_sim_qvels.dtype, np.float64)
            self.assertEqual(context.replay_ctrls.dtype, np.float64)
            np.testing.assert_allclose(context.initial_scene_qvel, qvel[0])
            np.testing.assert_allclose(context.initial_scene_act, act[0])
            np.testing.assert_allclose(context.initial_scene_mocap_pos, mocap_pos[0])
            np.testing.assert_allclose(context.initial_scene_mocap_quat, mocap_quat[0])

    @mock.patch("xdof_sim.teleop.episode_recorder.mujoco.mj_getState", autospec=True)
    @mock.patch("xdof_sim.teleop.episode_recorder.mujoco.mj_stateSize", autospec=True)
    def test_teleop_episode_recorder_preserves_native_sim_state_precision(
        self,
        mj_state_size_mock,
        mj_get_state_mock,
    ) -> None:
        class _FakeRecorderData:
            def __init__(self) -> None:
                self.qpos = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
                self.qvel = np.array([0.1, 0.2, 0.3], dtype=np.float64)
                self.ctrl = np.array([0.4, 0.5], dtype=np.float64)
                self.act = np.array([0.6], dtype=np.float64)
                self.mocap_pos = np.array([[0.7, 0.8, 0.9]], dtype=np.float64)
                self.mocap_quat = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64)
                self.time = 0.0

        class _FakeRecorderEnv:
            def __init__(self) -> None:
                self.data = _FakeRecorderData()
                self.model = SimpleNamespace(
                    nq=4,
                    nv=3,
                    nu=2,
                    na=1,
                    nmocap=1,
                    opt=SimpleNamespace(timestep=0.002),
                )
                self.single_timestep_action_dim = 14

        with tempfile.TemporaryDirectory() as tmpdir:
            mj_state_size_mock.return_value = 6

            def _fake_get_state(model, data, state, spec):
                state[:] = np.array(
                    [
                        float(data.time),
                        float(data.qpos[0]),
                        float(data.qvel[0]),
                        float(data.ctrl[0]),
                        float(data.act[0]),
                        float(data.mocap_pos[0, 0]),
                    ],
                    dtype=np.float64,
                )

            mj_get_state_mock.side_effect = _fake_get_state
            env = _FakeRecorderEnv()
            recorder = TeleopEpisodeRecorder(
                Path(tmpdir),
                task="sweep",
                scene="hybrid",
                prompt="sweep paper scraps",
                control_rate=29.4,
            )
            recorder.start(env)
            env.data.qpos = np.array([10.0, 11.0, 12.0, 13.0], dtype=np.float64)
            env.data.qvel = np.array([1.1, 1.2, 1.3], dtype=np.float64)
            env.data.ctrl = np.array([1.4, 1.5], dtype=np.float64)
            env.data.act = np.array([1.6], dtype=np.float64)
            env.data.mocap_pos = np.array([[1.7, 1.8, 1.9]], dtype=np.float64)
            env.data.mocap_quat = np.array([[0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
            env.data.time = 0.034
            recorder.record_step(np.arange(14, dtype=np.float32), env)
            episode_dir = recorder.close()

            self.assertIsNotNone(episode_dir)
            assert episode_dir is not None
            self.assertEqual(np.load(episode_dir / "integration_state.npy").dtype, np.float64)
            self.assertEqual(np.load(episode_dir / "qpos.npy").dtype, np.float64)
            self.assertEqual(np.load(episode_dir / "qvel.npy").dtype, np.float64)
            self.assertEqual(np.load(episode_dir / "ctrl.npy").dtype, np.float64)
            self.assertEqual(np.load(episode_dir / "act.npy").dtype, np.float64)
            self.assertEqual(np.load(episode_dir / "mocap_pos.npy").dtype, np.float64)
            self.assertEqual(np.load(episode_dir / "mocap_quat.npy").dtype, np.float64)
            payload = json.loads((episode_dir / "config.json").read_text())
            self.assertEqual(payload["integration_state_dtype"], "float64")
            self.assertEqual(payload["qpos_dtype"], "float64")
            self.assertEqual(payload["qvel_dtype"], "float64")
            self.assertEqual(payload["ctrl_dtype"], "float64")

    def test_load_recorded_sweep_episode_upgrades_old_79d_qpos_to_current_86d_layout(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "recorded_sweep_episode"
            episode_dir.mkdir(parents=True)

            actions = np.array(
                [
                    np.arange(14, dtype=np.float32),
                    np.arange(14, dtype=np.float32) + 10.0,
                ],
                dtype=np.float32,
            )
            qpos = np.array(
                [
                    np.arange(79, dtype=np.float32),
                    np.arange(79, dtype=np.float32) + 100.0,
                    np.arange(79, dtype=np.float32) + 200.0,
                ],
                dtype=np.float32,
            )
            np.save(episode_dir / "actions_full.npy", actions)
            np.save(episode_dir / "qpos.npy", qpos)
            np.save(episode_dir / "initial_qpos.npy", qpos[0])
            np.save(episode_dir / "action_timestamps.npy", np.array([0.1, 0.2], dtype=np.float64))
            np.save(episode_dir / "qpos_timestamps.npy", np.array([0.0, 0.1, 0.2], dtype=np.float64))
            (episode_dir / "config.json").write_text(
                json.dumps(
                    {
                        "scene": "hybrid",
                        "task": "sweep",
                        "prompt": "sweep away paper scraps from the table",
                        "control_rate": 10.0,
                    }
                )
            )

            context = load_episode_context(episode_dir, load_recorded_cameras=False)

            self.assertEqual(context.task, "sweep")
            self.assertEqual(context.raw_sim_states.shape, (3, 86))
            self.assertEqual(context.initial_scene_qpos.shape, (86,))
            np.testing.assert_allclose(context.raw_sim_states[:, :63], qpos[:, :63])
            np.testing.assert_allclose(context.raw_sim_states[:, 70:], qpos[:, 63:])
            np.testing.assert_allclose(
                context.raw_sim_states[:, 63:70],
                np.tile(np.array([[-1.5, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (3, 1)),
            )
            np.testing.assert_allclose(context.initial_scene_qpos[:63], qpos[0, :63])
            np.testing.assert_allclose(context.initial_scene_qpos[70:], qpos[0, 63:])
            np.testing.assert_allclose(
                context.initial_scene_qpos[63:70],
                np.array([-1.5, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            )

    @mock.patch("mcap_protobuf.decoder.DecoderFactory", autospec=True)
    @mock.patch("mcap.reader.make_reader", autospec=True)
    def test_delivered_episode_uses_command_state_actions(
        self,
        make_reader_mock,
        _decoder_factory_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_delivered"
            episode_dir.mkdir(parents=True)
            (episode_dir / "output.mcap").write_bytes(b"fake")

            make_reader_mock.return_value = _FakeDecodedReader(
                [
                    _topic_msg("/instruction", data="sim_throw plastic bottles in bin"),
                    _topic_msg("/left-arm-leader", position=[101, 102, 103, 104, 105, 106], timestamp=0.1),
                    _topic_msg("/right-arm-leader", position=[113, 114, 115, 116, 117, 118], timestamp=0.1),
                    _topic_msg("/left-command-state", position=[1, 2, 3, 4, 5, 6, 0.25], timestamp=0.1),
                    _topic_msg("/left-command-state", position=[7, 8, 9, 10, 11, 12, 0.75], timestamp=0.2),
                    _topic_msg("/right-command-state", position=[13, 14, 15, 16, 17, 18, 0.5], timestamp=0.1),
                    _topic_msg("/right-command-state", position=[19, 20, 21, 22, 23, 24, 0.9], timestamp=0.2),
                ]
            )

            streams, instruction = _load_delivered_episode_streams(episode_dir, load_recorded_cameras=False)

            self.assertEqual(instruction, "sim_throw plastic bottles in bin")
            np.testing.assert_allclose(
                streams.actions_left,
                np.array([[1, 2, 3, 4, 5, 6, 0.25], [7, 8, 9, 10, 11, 12, 0.75]], dtype=np.float64),
            )
            np.testing.assert_allclose(
                streams.actions_right,
                np.array([[13, 14, 15, 16, 17, 18, 0.5], [19, 20, 21, 22, 23, 24, 0.9]], dtype=np.float64),
            )

    @mock.patch("mcap_protobuf.decoder.DecoderFactory", autospec=True)
    @mock.patch("mcap.reader.make_reader", autospec=True)
    def test_delivered_episode_requires_command_state_streams(
        self,
        make_reader_mock,
        _decoder_factory_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_delivered"
            episode_dir.mkdir(parents=True)
            (episode_dir / "output.mcap").write_bytes(b"fake")

            make_reader_mock.return_value = _FakeDecodedReader(
                [
                    _topic_msg("/instruction", data="sim_throw plastic bottles in bin"),
                    _topic_msg("/left-arm-leader", position=[1, 2, 3, 4, 5, 6], timestamp=0.1),
                    _topic_msg("/right-arm-leader", position=[13, 14, 15, 16, 17, 18], timestamp=0.1),
                    _topic_msg("/left-arm-proprio", position=[0, 0, 0, 0, 0, 0, 0.25], timestamp=0.1),
                    _topic_msg("/right-arm-proprio", position=[0, 0, 0, 0, 0, 0, 0.5], timestamp=0.1),
                ]
            )

            with self.assertRaisesRegex(ValueError, "missing command-state action streams"):
                _load_delivered_episode_streams(episode_dir, load_recorded_cameras=False)

    @mock.patch("xdof_sim.rendering.replay.episode.load_randomization", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.episode.load_sim_states", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.episode._load_delivered_episode_streams", autospec=True)
    def test_delivered_episode_prefers_recorded_scene_xml_and_loads_integration_state(
        self,
        load_streams_mock,
        load_sim_states_mock,
        load_randomization_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "sim_load_the_plates_into_the_dish_rack" / "episode_delivered"
            episode_dir.mkdir(parents=True)
            (episode_dir / "output.mcap").write_bytes(b"fake")
            (episode_dir / "scene_assembled.xml").write_text(
                """
<mujoco>
  <compiler meshdir="/old/checkout/xdof_sim/models/assets/" texturedir="/old/checkout/xdof_sim/models/assets/"/>
  <asset>
    <mesh name="rack" file="/old/checkout/xdof_sim/models/assets/task_dishrack/dish_rack/DishRack028/visual/DishRack028.obj"/>
    <mesh name="plate" file="/old/checkout/xdof_sim/models/assets/task_dishrack/plate/current/model.obj"/>
  </asset>
</mujoco>
""".strip()
            )
            integration_state = np.arange(12, dtype=np.float64).reshape(3, 4)
            np.save(episode_dir / "integration_state.npy", integration_state)
            np.save(episode_dir / "integration_state_sim_time.npy", np.array([0.0, 0.1, 0.2], dtype=np.float64))
            np.save(
                episode_dir / "integration_state_wallclock.npy",
                np.array([1_700_000_000.0, 1_700_000_000.1, 1_700_000_000.2], dtype=np.float64),
            )

            load_streams_mock.return_value = (
                self._fake_delivered_streams(episode_dir),
                "load the plates into the dish rack",
            )
            qpos = np.zeros((2, 30), dtype=np.float64)
            qpos_ts = np.array([10.0, 10.1], dtype=np.float64)
            load_sim_states_mock.return_value = (qpos, qpos_ts)
            rand_state = object()
            load_randomization_mock.return_value = rand_state

            context = load_episode_context(episode_dir, load_recorded_cameras=False)

            self.assertEqual(context.episode_format, "delivered")
            self.assertEqual(context.task, "dishrack")
            self.assertEqual(context.scene_source, "scene_assembled.xml")
            self.assertIs(context.rand_state, rand_state)
            self.assertIsNotNone(context.scene_xml_string)
            assert context.scene_xml_string is not None
            self.assertNotIn("/old/checkout", context.scene_xml_string)
            self.assertIn("task_dishrack/dish_rack/dish_rack_3/visual/DishRack028.obj", context.scene_xml_string)
            self.assertIn("task_dishrack/plate/plate_0/model.obj", context.scene_xml_string)
            np.testing.assert_allclose(context.raw_sim_integration_states, integration_state)
            np.testing.assert_allclose(
                context.raw_sim_integration_timestamps,
                np.array([0.0, 0.1, 0.2], dtype=np.float64),
            )
            np.testing.assert_allclose(
                context.raw_sim_integration_wallclock_timestamps,
                np.array([1_700_000_000.0, 1_700_000_000.1, 1_700_000_000.2], dtype=np.float64),
            )
            np.testing.assert_allclose(context.initial_scene_integration_state, integration_state[0])
            self.assertEqual(context.raw_sim_state_spec, 16383)

    def test_load_integration_states_preserves_recorded_frames(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "episode_delivered"
            episode_dir.mkdir()
            integration_state = np.arange(16, dtype=np.float64).reshape(4, 4)
            np.save(episode_dir / "integration_state.npy", integration_state)
            np.save(
                episode_dir / "integration_state_sim_time.npy",
                np.array([3.0, 0.1, 0.2, 0.3], dtype=np.float64),
            )
            np.save(
                episode_dir / "integration_state_wallclock.npy",
                np.array([100.0, 100.1, 100.2, 100.3], dtype=np.float64),
            )

            states, timestamps, wallclock_timestamps, state_spec = load_integration_states(episode_dir)

            np.testing.assert_allclose(states, integration_state)
            np.testing.assert_allclose(timestamps, np.array([3.0, 0.1, 0.2, 0.3], dtype=np.float64))
            np.testing.assert_allclose(
                wallclock_timestamps,
                np.array([100.0, 100.1, 100.2, 100.3], dtype=np.float64),
            )
            self.assertEqual(state_spec, 16383)

    def test_delivered_integration_states_are_aligned_to_replay_grid(self) -> None:
        episode_dir = Path("/tmp/delivered_episode")
        streams = EpisodeStreams(
            episode_dir=episode_dir,
            actions_left=np.zeros((4, 7), dtype=np.float64),
            ts_left=np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
            actions_right=np.zeros((4, 7), dtype=np.float64),
            ts_right=np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float64),
            camera_frames={},
            camera_ts={},
        )
        context = EpisodeContext(
            streams=streams,
            episode_format="delivered",
            scene="hybrid",
            task="dishrack",
            instruction="load plates into tabletop dish rack",
            rand_state=None,
            raw_sim_states=np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float64),
            raw_sim_timestamps=np.array([1_777_951_600.0, 1_777_951_600.05, 1_777_951_600.15, 1_777_951_600.25]),
            raw_sim_integration_states=np.array([[10.0], [20.0], [30.0], [40.0], [50.0]], dtype=np.float64),
            raw_sim_integration_timestamps=np.array([132.226, 132.276, 132.376, 132.476, 132.576], dtype=np.float64),
            raw_sim_integration_wallclock_timestamps=np.array(
                [1_777_951_600.0, 1_777_951_600.05, 1_777_951_600.15, 1_777_951_600.25, 1_777_951_600.35],
                dtype=np.float64,
            ),
            raw_sim_state_spec=16383,
        )

        timeline = build_replay_timeline(context, control_hz=10.0)

        np.testing.assert_allclose(timeline.grid_ts, np.array([0.0, 0.1, 0.2]))
        np.testing.assert_allclose(timeline.sim_states[:, 0], np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(timeline.sim_integration_states[:, 0], np.array([20.0, 30.0, 40.0]))

    @mock.patch("xdof_sim.rendering.replay.episode.load_randomization", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.episode.load_sim_states", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.episode._load_delivered_episode_streams", autospec=True)
    def test_delivered_episode_falls_back_to_randomization_when_scene_xml_missing(
        self,
        load_streams_mock,
        load_sim_states_mock,
        load_randomization_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "sim_throw_plastic_bottles_in_bin" / "episode_delivered"
            episode_dir.mkdir(parents=True)
            (episode_dir / "output.mcap").write_bytes(b"fake")
            rand_state = object()
            load_streams_mock.return_value = (
                self._fake_delivered_streams(episode_dir),
                "throw plastic bottles in bin",
            )
            load_sim_states_mock.return_value = (np.zeros((2, 32), dtype=np.float64), np.array([0.0, 0.1]))
            load_randomization_mock.return_value = rand_state

            context = load_episode_context(episode_dir, load_recorded_cameras=False)

            self.assertEqual(context.scene_source, "randomization.json")
            self.assertIs(context.rand_state, rand_state)
            self.assertIsNone(context.scene_xml_string)

    @mock.patch("xdof_sim.rendering.replay.episode.load_randomization", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.episode.load_sim_states", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.episode._load_delivered_episode_streams", autospec=True)
    def test_delivered_episode_requires_scene_xml_or_randomization(
        self,
        load_streams_mock,
        load_sim_states_mock,
        load_randomization_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir) / "sim_throw_plastic_bottles_in_bin" / "episode_delivered"
            episode_dir.mkdir(parents=True)
            (episode_dir / "output.mcap").write_bytes(b"fake")
            load_streams_mock.return_value = (
                self._fake_delivered_streams(episode_dir),
                "throw plastic bottles in bin",
            )
            load_sim_states_mock.return_value = (np.zeros((2, 32), dtype=np.float64), np.array([0.0, 0.1]))
            load_randomization_mock.return_value = None

            with self.assertRaisesRegex(ValueError, "missing scene_assembled.xml and randomization.json"):
                load_episode_context(episode_dir, load_recorded_cameras=False)

    @mock.patch("xdof_sim.make_env", autospec=True)
    def test_create_replay_env_uses_scene_xml_without_reapplying_randomization(self, make_env_mock) -> None:
        apply_mock = mock.Mock()
        env = SimpleNamespace(
            reset=mock.Mock(),
            model=object(),
            data=object(),
            _task_randomizer=SimpleNamespace(apply=apply_mock),
        )
        make_env_mock.return_value = env
        context = SimpleNamespace(
            scene="hybrid",
            task="dishrack",
            scene_xml_string="<mujoco/>",
            rand_state=object(),
            physics_overrides=None,
        )

        returned = create_replay_env(context)

        self.assertIs(returned, env)
        self.assertEqual(make_env_mock.call_args.kwargs["scene_xml_string"], "<mujoco/>")
        self.assertIs(make_env_mock.call_args.kwargs["enable_task_randomizer"], False)
        env.reset.assert_called_once_with(randomize=False)
        apply_mock.assert_not_called()

    @mock.patch("xdof_sim.make_env", autospec=True)
    def test_create_replay_env_refuses_randomization_fallback_when_scene_xml_fails(self, make_env_mock) -> None:
        apply_mock = mock.Mock()
        rand_state = object()
        make_env_mock.side_effect = ValueError("Error opening file 'missing.obj'")
        context = SimpleNamespace(
            streams=SimpleNamespace(episode_dir=Path("/tmp/episode")),
            scene="hybrid",
            task="dishrack",
            scene_xml_string="<mujoco/>",
            rand_state=rand_state,
            physics_overrides=None,
        )

        with self.assertRaisesRegex(ValueError, "missing.obj"):
            create_replay_env(context)

        self.assertEqual(make_env_mock.call_count, 1)
        self.assertEqual(make_env_mock.call_args.kwargs["scene_xml_string"], "<mujoco/>")
        self.assertIs(make_env_mock.call_args.kwargs["enable_task_randomizer"], False)
        apply_mock.assert_not_called()


class DirectActionLoadingTests(unittest.TestCase):
    @mock.patch("xdof_sim.rendering.replay.episode._load_direct_topic_positions", autospec=True)
    def test_load_direct_actions_prefers_command_state_topics(
        self,
        load_topic_positions_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir)
            (episode_dir / "left.mcap").write_bytes(b"fake")
            (episode_dir / "action-left.mcap").write_bytes(b"fake")

            def _fake_load(path, *, topic):
                path = Path(path)
                if path.name == "left.mcap" and topic == "/left-command-state":
                    return (
                        np.array(
                            [
                                [1, 2, 3, 4, 5, 6, 0.1],
                                [7, 8, 9, 10, 11, 12, 0.2],
                            ],
                            dtype=np.float64,
                        ),
                        np.array([0.0, 0.1], dtype=np.float64),
                    )
                if path.name == "action-left.mcap":
                    raise AssertionError("command-state path should short-circuit legacy action loading")
                return None

            load_topic_positions_mock.side_effect = _fake_load
            actions, ts = _load_direct_actions(episode_dir, "left")

            np.testing.assert_allclose(
                actions,
                np.array(
                    [
                        [1, 2, 3, 4, 5, 6, 0.1],
                        [7, 8, 9, 10, 11, 12, 0.2],
                    ],
                    dtype=np.float64,
                ),
            )
            np.testing.assert_allclose(ts, np.array([0.0, 0.1], dtype=np.float64))

    @mock.patch("xdof_sim.rendering.replay.episode._load_direct_topic_positions", autospec=True)
    def test_load_direct_actions_uses_explicit_action_gripper_topic(
        self,
        load_topic_positions_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir)
            (episode_dir / "action-left.mcap").write_bytes(b"fake")

            def _fake_load(path, *, topic):
                path = Path(path)
                if path.name == "action-left.mcap" and topic == "/action-left-robot-state":
                    return (
                        np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.float64),
                        np.array([0.0, 0.1], dtype=np.float64),
                    )
                if path.name == "action-left.mcap" and topic == "/action-left-gripper-state":
                    return (
                        np.array([[0.1], [0.2]], dtype=np.float64),
                        np.array([0.0, 0.1], dtype=np.float64),
                    )
                return None

            load_topic_positions_mock.side_effect = _fake_load
            actions, ts = _load_direct_actions(episode_dir, "left")

            np.testing.assert_allclose(
                actions,
                np.array(
                    [
                        [1, 2, 3, 4, 5, 6, 0.1],
                        [7, 8, 9, 10, 11, 12, 0.2],
                    ],
                    dtype=np.float64,
                ),
            )
            np.testing.assert_allclose(ts, np.array([0.0, 0.1], dtype=np.float64))

    @mock.patch("xdof_sim.rendering.replay.episode._load_direct_topic_positions", autospec=True)
    def test_load_direct_actions_rejects_missing_action_gripper_topic(
        self,
        load_topic_positions_mock,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            episode_dir = Path(tmpdir)
            (episode_dir / "left.mcap").write_bytes(b"fake")
            (episode_dir / "action-left.mcap").write_bytes(b"fake")

            def _fake_load(path, *, topic):
                path = Path(path)
                if path.name == "action-left.mcap" and topic == "/action-left-robot-state":
                    return (
                        np.array([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], dtype=np.float64),
                        np.array([0.0, 0.1], dtype=np.float64),
                    )
                if path.name == "left.mcap" and topic in {"/left-robot-state", "/left-gripper-state"}:
                    return (
                        np.array([[0.9], [0.8]], dtype=np.float64),
                        np.array([0.0, 0.1], dtype=np.float64),
                    )
                return None

            load_topic_positions_mock.side_effect = _fake_load

            with self.assertRaisesRegex(ValueError, "missing explicit action gripper stream"):
                _load_direct_actions(episode_dir, "left")

    def test_actions_to_ctrl_batch_scales_grippers_and_places_ctrl_indices(self) -> None:
        actions = np.array(
            [
                [1, 2, 3, 0.5],
                [4, 5, 6, 0.25],
            ],
            dtype=np.float32,
        )
        ctrl = _actions_to_ctrl_batch(
            actions,
            nu=6,
            ctrl_indices=[2, 4, 1, 5],
            gripper_indices=[3],
            gripper_ctrl_max=0.1,
        )
        expected = np.array(
            [
                [0.0, 3.0, 1.0, 0.0, 2.0, 0.05],
                [0.0, 6.0, 4.0, 0.0, 5.0, 0.025],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(ctrl, expected)

    def test_sample_qpos_frames_for_video_uses_timestamps_not_raw_step_count(self) -> None:
        qpos = np.arange(11, dtype=np.float32)[:, None]
        ts = np.arange(11, dtype=np.float64) * 0.1
        sampled = _sample_qpos_frames_for_video(
            qpos,
            ts,
            fps=2.0,
            max_output_frames=3,
        )
        np.testing.assert_allclose(sampled[:, 0], np.array([0.0, 5.0, 10.0], dtype=np.float32))

    def test_infer_physics_substeps_per_action_uses_single_steps_for_high_rate_grid(self) -> None:
        env = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002)), _control_decimation=17)
        session = SimpleNamespace(env=env, grid_ts=np.arange(100, dtype=np.float64) * 0.0007)
        self.assertEqual(_infer_physics_substeps_per_action(session), 1)


class FakeData:
    def __init__(self) -> None:
        self.qpos = np.zeros(14, dtype=np.float32)
        self.ctrl = np.zeros(14, dtype=np.float32)
        self.qvel = np.zeros(14, dtype=np.float32)
        self.act = np.zeros(0, dtype=np.float32)
        self.mocap_pos = np.zeros((0, 3), dtype=np.float32)
        self.mocap_quat = np.zeros((0, 4), dtype=np.float32)


class FakeEnv:
    def __init__(self) -> None:
        self.model = object()
        self.data = FakeData()
        self.state_writes: list[np.ndarray] = []
        self.applied_actions: list[np.ndarray] = []
        self._control_decimation = 3

    def get_init_q(self) -> np.ndarray:
        return np.zeros(14, dtype=np.float32)

    def _set_qpos_from_state(self, state: np.ndarray) -> None:
        self.state_writes.append(np.asarray(state, dtype=np.float32).copy())

    def _step_single(self, action: np.ndarray) -> None:
        self.applied_actions.append(np.asarray(action, dtype=np.float32).copy())


class FakeFullQposData(FakeData):
    def __init__(self) -> None:
        self.qpos = np.zeros(32, dtype=np.float32)
        self.ctrl = np.zeros(14, dtype=np.float32)
        self.qvel = np.zeros(24, dtype=np.float32)
        self.act = np.zeros(3, dtype=np.float32)
        self.mocap_pos = np.zeros((2, 3), dtype=np.float32)
        self.mocap_quat = np.zeros((2, 4), dtype=np.float32)


class FakePhysicsEnv(FakeEnv):
    def _step_single(self, action: np.ndarray) -> None:
        super()._step_single(action)
        self.data.qpos[: len(action)] = np.asarray(action, dtype=np.float32)


class ReplaySessionStateModeTests(unittest.TestCase):
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_policy_state_replay_uses_env_state_writer(self, _reset_mock, _forward_mock) -> None:
        env = FakeEnv()
        sim_states = np.array(
            [
                np.full(14, 1.0, dtype=np.float32),
                np.full(14, 2.0, dtype=np.float32),
            ]
        )
        session = ReplaySession(
            env,
            actions=np.zeros((2, 14), dtype=np.float32),
            grid_ts=np.array([0.0, 0.1], dtype=np.float64),
            sim_states=sim_states,
            sim_state_kind="policy_state",
            mode="qpos",
        )

        self.assertFalse(session.has_exact_qpos)
        self.assertTrue(session.has_state_replay)
        self.assertEqual(session.state_replay_label, "state (direct)")
        np.testing.assert_allclose(env.state_writes[0], sim_states[0])

        stepped = session.step()
        self.assertTrue(stepped)
        np.testing.assert_allclose(env.state_writes[-1], sim_states[1])

    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_physics_replay_starts_from_first_replay_state(self, _reset_mock, _forward_mock) -> None:
        env = FakeEnv()
        sim_states = np.array(
            [
                np.full(14, 3.0, dtype=np.float32),
                np.full(14, 4.0, dtype=np.float32),
            ]
        )
        actions = np.array(
            [
                np.full(14, 10.0, dtype=np.float32),
                np.full(14, 11.0, dtype=np.float32),
            ]
        )
        session = ReplaySession(
            env,
            actions=actions,
            grid_ts=np.array([0.0, 0.1], dtype=np.float64),
            sim_states=sim_states,
            sim_state_kind="policy_state",
            mode="physics",
        )

        np.testing.assert_allclose(env.state_writes[0], sim_states[0])
        self.assertEqual(session.step_idx, 0)

        stepped = session.step()
        self.assertTrue(stepped)
        np.testing.assert_allclose(env.applied_actions[0], actions[0])

    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_physics_replay_prefers_full_initial_qpos_when_available(self, _reset_mock, _forward_mock) -> None:
        env = FakeEnv()
        env.data = FakeFullQposData()
        sim_states = np.array(
            [
                np.full(14, 3.0, dtype=np.float32),
                np.full(14, 4.0, dtype=np.float32),
            ]
        )
        initial_scene_qpos = np.arange(20, dtype=np.float32) + 1.0
        session = ReplaySession(
            env,
            actions=np.zeros((2, 14), dtype=np.float32),
            grid_ts=np.array([0.0, 0.1], dtype=np.float64),
            sim_states=sim_states,
            initial_scene_qpos=initial_scene_qpos,
            sim_state_kind="policy_state",
            mode="physics",
        )

        self.assertEqual(len(env.state_writes), 0)
        np.testing.assert_allclose(env.data.qpos[: len(initial_scene_qpos)], initial_scene_qpos)

    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_physics_replay_applies_full_initial_dynamic_state(self, _reset_mock, _forward_mock) -> None:
        env = FakeEnv()
        env.data = FakeFullQposData()
        initial_scene_qpos = np.arange(20, dtype=np.float32) + 1.0
        initial_scene_qvel = np.arange(12, dtype=np.float32) + 2.0
        initial_scene_act = np.array([0.2, 0.4, 0.6], dtype=np.float32)
        initial_scene_mocap_pos = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        initial_scene_mocap_quat = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        ReplaySession(
            env,
            actions=np.zeros((2, 14), dtype=np.float32),
            grid_ts=np.array([0.0, 0.1], dtype=np.float64),
            sim_states=None,
            initial_scene_qpos=initial_scene_qpos,
            initial_scene_qvel=initial_scene_qvel,
            initial_scene_act=initial_scene_act,
            initial_scene_mocap_pos=initial_scene_mocap_pos,
            initial_scene_mocap_quat=initial_scene_mocap_quat,
            mode="physics",
        )

        np.testing.assert_allclose(env.data.qpos[: len(initial_scene_qpos)], initial_scene_qpos)
        np.testing.assert_allclose(env.data.qvel[: len(initial_scene_qvel)], initial_scene_qvel)
        np.testing.assert_allclose(env.data.act[: len(initial_scene_act)], initial_scene_act)
        np.testing.assert_allclose(
            env.data.mocap_pos[: len(initial_scene_mocap_pos)],
            initial_scene_mocap_pos,
        )
        np.testing.assert_allclose(
            env.data.mocap_quat[: len(initial_scene_mocap_quat)],
            initial_scene_mocap_quat,
        )

    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_post_step_qpos_replay_defers_first_state_until_first_step(self, _reset_mock, _forward_mock) -> None:
        env = FakeEnv()
        sim_states = np.array(
            [
                np.full(14, 5.0, dtype=np.float32),
                np.full(14, 6.0, dtype=np.float32),
            ]
        )
        session = ReplaySession(
            env,
            actions=np.zeros((2, 14), dtype=np.float32),
            grid_ts=np.array([0.1, 0.2], dtype=np.float64),
            sim_states=sim_states,
            sim_state_kind="policy_state",
            sim_state_alignment="post_step",
            mode="qpos",
        )

        self.assertEqual(len(env.state_writes), 0)
        self.assertEqual(session.total_steps, 2)
        self.assertEqual(session.current_frame_idx, 0)

        self.assertTrue(session.step())
        np.testing.assert_allclose(env.state_writes[-1], sim_states[0])
        self.assertEqual(session.step_idx, 1)
        self.assertEqual(session.current_frame_idx, 0)
        self.assertAlmostEqual(session.current_timestamp, 0.1)

        self.assertTrue(session.step())
        np.testing.assert_allclose(env.state_writes[-1], sim_states[1])
        self.assertEqual(session.step_idx, 2)
        self.assertEqual(session.current_frame_idx, 1)
        self.assertAlmostEqual(session.current_timestamp, 0.2)

    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_qpos_replay_applies_optional_dynamic_state_frames(self, _reset_mock, _forward_mock) -> None:
        env = FakeEnv()
        env.data = FakeFullQposData()
        sim_states = np.array(
            [
                np.arange(32, dtype=np.float32),
                np.arange(32, dtype=np.float32) + 100.0,
            ]
        )
        sim_qvels = np.array(
            [
                np.arange(24, dtype=np.float32) + 1.0,
                np.arange(24, dtype=np.float32) + 11.0,
            ]
        )
        sim_acts = np.array(
            [
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                np.array([0.4, 0.5, 0.6], dtype=np.float32),
            ]
        )
        sim_mocap_pos = np.array(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ],
            dtype=np.float32,
        )
        sim_mocap_quat = np.array(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                [[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        session = ReplaySession(
            env,
            actions=np.zeros((1, 14), dtype=np.float32),
            grid_ts=np.array([0.0, 0.1], dtype=np.float64),
            sim_states=sim_states,
            sim_qvels=sim_qvels,
            sim_acts=sim_acts,
            sim_mocap_pos=sim_mocap_pos,
            sim_mocap_quat=sim_mocap_quat,
            mode="qpos",
        )

        np.testing.assert_allclose(env.data.qpos[:32], sim_states[0])
        np.testing.assert_allclose(env.data.qvel[:24], sim_qvels[0])
        np.testing.assert_allclose(env.data.act[:3], sim_acts[0])
        np.testing.assert_allclose(env.data.mocap_pos[:2], sim_mocap_pos[0])
        np.testing.assert_allclose(env.data.mocap_quat[:2], sim_mocap_quat[0])

        self.assertTrue(session.step())
        np.testing.assert_allclose(env.data.qpos[:32], sim_states[1])
        np.testing.assert_allclose(env.data.qvel[:24], sim_qvels[1])
        np.testing.assert_allclose(env.data.act[:3], sim_acts[1])
        np.testing.assert_allclose(env.data.mocap_pos[:2], sim_mocap_pos[1])
        np.testing.assert_allclose(env.data.mocap_quat[:2], sim_mocap_quat[1])

    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_setState", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_qpos_replay_prefers_recorded_integration_state_frames(
        self,
        _reset_mock,
        mj_set_state_mock,
        _forward_mock,
    ) -> None:
        env = FakeFullQposData()
        fake_env = FakeEnv()
        fake_env.data = env
        sim_states = np.array(
            [
                np.arange(32, dtype=np.float32),
                np.arange(32, dtype=np.float32) + 100.0,
            ]
        )
        integration_states = np.array(
            [
                np.arange(6, dtype=np.float64),
                np.arange(6, dtype=np.float64) + 10.0,
            ]
        )
        session = ReplaySession(
            fake_env,
            actions=np.zeros((1, 14), dtype=np.float32),
            grid_ts=np.array([0.0, 0.1], dtype=np.float64),
            sim_states=sim_states,
            sim_integration_states=integration_states,
            sim_state_spec=16383,
            mode="qpos",
        )

        np.testing.assert_allclose(mj_set_state_mock.call_args_list[0].args[2], integration_states[0])
        self.assertTrue(session.step())
        np.testing.assert_allclose(mj_set_state_mock.call_args_list[-1].args[2], integration_states[1])

    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_setState", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_qpos_replay_accepts_validated_integration_states_without_raw_qpos(
        self,
        _reset_mock,
        mj_set_state_mock,
        _forward_mock,
    ) -> None:
        env = FakeEnv()
        integration_states = np.array(
            [
                np.arange(6, dtype=np.float64),
                np.arange(6, dtype=np.float64) + 10.0,
            ]
        )
        session = ReplaySession(
            env,
            actions=np.zeros((1, 14), dtype=np.float32),
            grid_ts=np.array([0.0, 0.1], dtype=np.float64),
            sim_states=None,
            sim_integration_states=integration_states,
            sim_state_spec=16383,
            mode="auto",
        )

        self.assertEqual(session.mode, "qpos")
        self.assertTrue(session.has_exact_qpos)
        self.assertEqual(session.state_replay_label, "integration state (exact)")
        np.testing.assert_allclose(mj_set_state_mock.call_args_list[0].args[2], integration_states[0])
        self.assertTrue(session.step())
        np.testing.assert_allclose(mj_set_state_mock.call_args_list[-1].args[2], integration_states[1])

    @mock.patch("xdof_sim.rendering.replay.video.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.video.mujoco.mj_setState", autospec=True)
    def test_batched_qpos_export_materializes_qpos_from_integration_states(
        self,
        mj_set_state_mock,
        _forward_mock,
    ) -> None:
        data = SimpleNamespace(qpos=np.zeros(5, dtype=np.float64))

        def _fake_set_state(_model, data_arg, state, _spec):
            data_arg.qpos[:] = np.array([state[0], state[1], state[2], state[0] + 1.0, 9.0])

        mj_set_state_mock.side_effect = _fake_set_state
        session = SimpleNamespace(
            model=SimpleNamespace(nq=5),
            data=data,
            sim_state_spec=16383,
        )
        integration_states = np.array(
            [
                [10.0, 100.0, 1000.0],
                [20.0, 200.0, 2000.0],
            ],
            dtype=np.float64,
        )

        qpos = _qpos_frames_from_integration_states(session, integration_states)

        np.testing.assert_allclose(
            qpos,
            np.array(
                [
                    [10.0, 100.0, 1000.0, 11.0, 9.0],
                    [20.0, 200.0, 2000.0, 21.0, 9.0],
                ],
                dtype=np.float32,
            ),
        )
        self.assertEqual(mj_set_state_mock.call_count, 2)

    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_setState", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_physics_replay_prefers_recorded_initial_integration_state(
        self,
        _reset_mock,
        mj_set_state_mock,
        _forward_mock,
    ) -> None:
        env = FakeEnv()
        initial_integration_state = np.arange(7, dtype=np.float64)
        ReplaySession(
            env,
            actions=np.zeros((1, 14), dtype=np.float32),
            grid_ts=np.array([0.0], dtype=np.float64),
            initial_scene_integration_state=initial_integration_state,
            sim_state_spec=16383,
            mode="physics",
        )

        np.testing.assert_allclose(mj_set_state_mock.call_args.args[2], initial_integration_state)

    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_post_step_physics_replay_uses_first_recorded_timestamp_after_first_action(
        self,
        _reset_mock,
        _forward_mock,
    ) -> None:
        env = FakeEnv()
        actions = np.array(
            [
                np.full(14, 10.0, dtype=np.float32),
                np.full(14, 11.0, dtype=np.float32),
            ]
        )
        sim_states = np.array(
            [
                np.full(14, 3.0, dtype=np.float32),
                np.full(14, 4.0, dtype=np.float32),
            ]
        )
        session = ReplaySession(
            env,
            actions=actions,
            grid_ts=np.array([0.1, 0.2], dtype=np.float64),
            sim_states=sim_states,
            sim_state_kind="policy_state",
            sim_state_alignment="post_step",
            mode="physics",
        )

        self.assertEqual(len(env.state_writes), 0)
        self.assertTrue(session.step())
        np.testing.assert_allclose(env.applied_actions[-1], actions[0])
        self.assertEqual(session.current_frame_idx, 0)
        self.assertAlmostEqual(session.current_timestamp, 0.1)

    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_step", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_physics_replay_prefers_recorded_ctrl_when_available(
        self,
        _reset_mock,
        _forward_mock,
        mj_step_mock,
    ) -> None:
        env = FakeEnv()
        actions = np.array([np.full(14, 10.0, dtype=np.float32)], dtype=np.float32)
        replay_ctrls = np.array([np.full(14, 123.0, dtype=np.float32)], dtype=np.float32)
        session = ReplaySession(
            env,
            actions=actions,
            grid_ts=np.array([0.0], dtype=np.float64),
            replay_ctrls=replay_ctrls,
            mode="physics",
        )

        self.assertTrue(session.step())
        self.assertEqual(len(env.applied_actions), 0)
        np.testing.assert_allclose(env.data.ctrl, replay_ctrls[0])
        self.assertEqual(mj_step_mock.call_count, env._control_decimation)

    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_forward", autospec=True)
    @mock.patch("xdof_sim.rendering.replay.session.mujoco.mj_resetData", autospec=True)
    def test_collect_physics_rollout_qpos_frames_uses_initial_scene_qpos(
        self,
        _reset_mock,
        _forward_mock,
    ) -> None:
        env = FakePhysicsEnv()
        env.data = FakeFullQposData()
        initial_scene_qpos = np.arange(20, dtype=np.float32) + 1.0
        actions = np.array(
            [
                np.full(14, 10.0, dtype=np.float32),
                np.full(14, 11.0, dtype=np.float32),
            ]
        )
        session = ReplaySession(
            env,
            actions=actions,
            grid_ts=np.array([0.0, 0.1], dtype=np.float64),
            sim_states=None,
            initial_scene_qpos=initial_scene_qpos,
            mode="physics",
        )

        frames = collect_physics_rollout_qpos_frames(session)

        self.assertEqual(frames.shape, (3, 32))
        np.testing.assert_allclose(frames[0, :20], initial_scene_qpos)
        np.testing.assert_allclose(frames[1, :14], actions[0])
        np.testing.assert_allclose(frames[2, :14], actions[1])


if __name__ == "__main__":
    unittest.main()
