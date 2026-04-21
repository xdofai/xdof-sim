from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from xdof_sim.dataset_export.metadata import (
    build_collected_entry,
    finalize_dataset_metadata,
)
from xdof_sim.dataset_export.types import ExportConfig
from xdof_sim.dataset_export.types import ExportTrajectory
from xdof_sim.dataset_export.video_io import probe_video_frame_count, write_rgb_video
from xdof_sim.dataset_export.writer import (
    camera_video_name,
    combined_video_name,
    validate_exported_episode,
    write_episode_metadata,
    write_states_actions,
)


FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def make_trajectory() -> ExportTrajectory:
    return ExportTrajectory(
        episode_id="episode_test",
        source_episode_dir=Path("/tmp/source/episode_test"),
        source_delivery="sim_spell_cat",
        scene="hybrid",
        task="blocks",
        task_name="sim_spell_cat",
        instruction="sim_spell_cat",
        data_date="apr_11",
        camera_names=("top", "left", "right"),
        timestamps=np.array([0.0, 1.0 / 30.0], dtype=np.float64),
        states=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        actions=np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        qpos=np.array([[0.1], [0.2]], dtype=np.float32),
    )


@unittest.skipUnless(FFMPEG_AVAILABLE, "ffmpeg/ffprobe are required for video IO tests")
class DatasetExportIoTests(unittest.TestCase):
    def test_write_rgb_video_exports_frame_mappings(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = Path(tmpdir) / "top_camera-images-rgb.mp4"
            frames = np.array(
                [
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    np.full((4, 4, 3), 255, dtype=np.uint8),
                ]
            )
            write_rgb_video(video_path, frames, fps=30.0)

            self.assertTrue(video_path.exists())
            self.assertEqual(probe_video_frame_count(video_path), 2)
            mapping_path = video_path.with_name(video_path.stem + "_frame_mappings.json")
            mapping = json.loads(mapping_path.read_text())
            self.assertEqual(len(mapping["frames"]), 2)

    def test_write_episode_files_and_finalize_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir)
            episode_dir = dataset_root / "data" / "episode_test"
            episode_dir.mkdir(parents=True)
            trajectory = make_trajectory()
            config = ExportConfig(batch_name="batch_a", fps=30.0, image_width=4, image_height=4)

            frames = np.array(
                [
                    np.zeros((4, 4, 3), dtype=np.uint8),
                    np.full((4, 4, 3), 64, dtype=np.uint8),
                ]
            )
            video_paths = {}
            for camera_name, pixel in zip(trajectory.camera_names, (0, 64, 128), strict=True):
                cam_frames = np.array(
                    [
                        np.full((4, 4, 3), pixel, dtype=np.uint8),
                        np.full((4, 4, 3), pixel + 1, dtype=np.uint8),
                    ]
                )
                video_paths[camera_name] = episode_dir / camera_video_name(camera_name)
                write_rgb_video(video_paths[camera_name], cam_frames, fps=config.fps)
            combined = np.concatenate([frames, frames, frames], axis=1)
            video_paths["combined"] = episode_dir / combined_video_name()
            write_rgb_video(video_paths["combined"], combined, fps=config.fps)

            write_states_actions(episode_dir, states=trajectory.states, actions=trajectory.actions)
            write_episode_metadata(episode_dir, trajectory, config=config, video_paths=video_paths)
            validate_exported_episode(
                episode_dir,
                expected_steps=2,
                camera_names=trajectory.camera_names,
            )

            collected = {
                trajectory.episode_id: build_collected_entry(trajectory, config=config),
                "episode_other": {
                    **build_collected_entry(
                        ExportTrajectory(
                            **{
                                **trajectory.__dict__,
                                "episode_id": "episode_other",
                                "states": np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
                                "actions": np.array([[13.0, 14.0], [15.0, 16.0]], dtype=np.float32),
                            }
                        ),
                        config=config,
                    )
                },
            }
            other_dir = dataset_root / "data" / "episode_other"
            other_dir.mkdir(parents=True)
            write_states_actions(
                other_dir,
                states=np.array([[9.0, 10.0], [11.0, 12.0]], dtype=np.float32),
                actions=np.array([[13.0, 14.0], [15.0, 16.0]], dtype=np.float32),
            )

            outputs = finalize_dataset_metadata(
                dataset_root,
                collected=collected,
                n_states=2,
                n_actions=2,
                state_pad_size=4,
                action_pad_size=4,
                val_ratio=0.5,
                seed=0,
            )

            self.assertTrue(outputs["collected"].exists())
            self.assertTrue(outputs["norm_stats"].exists())
            self.assertTrue(outputs["train_episodes"].exists())
            self.assertTrue(outputs["val_episodes"].exists())

            norm_stats = json.loads(outputs["norm_stats"].read_text())
            self.assertEqual(len(norm_stats["state"]["mean"]), 4)
            self.assertEqual(len(norm_stats["actions"]["std"]), 4)

            train_ids = json.loads(outputs["train_episodes"].read_text())
            val_ids = json.loads(outputs["val_episodes"].read_text())
            self.assertEqual(sorted(train_ids + val_ids), ["episode_other", "episode_test"])


if __name__ == "__main__":
    unittest.main()
