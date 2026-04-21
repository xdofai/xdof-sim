from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from xdof_sim.dataset_export.s3_pipeline import export_s3_shard, finalize_s3_dataset
from xdof_sim.dataset_export.s3_source import S3EpisodeSource
from xdof_sim.dataset_export.s3_utils import S3ObjectInfo, parse_s3_uri
from xdof_sim.dataset_export.types import ExportConfig


class S3PipelineTests(unittest.TestCase):
    def test_export_s3_shard_uploads_episode_and_shard_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scratch_dir = Path(tmpdir)
            staged_dir = scratch_dir / "stage" / "task_a" / "episode_1"
            staged_dir.mkdir(parents=True)
            exported_dir = scratch_dir / "dataset" / "data" / "episode_1"
            exported_dir.mkdir(parents=True)
            (exported_dir / "states_actions.npy").write_bytes(b"test")
            source = S3EpisodeSource(
                source_root=parse_s3_uri("s3://source-bucket/deliveries"),
                episode_prefix="deliveries/task_a/episode_1",
                relative_episode_prefix="task_a/episode_1",
                files=(
                    S3ObjectInfo("source-bucket", "deliveries/task_a/episode_1/output.mcap", 10),
                    S3ObjectInfo("source-bucket", "deliveries/task_a/episode_1/sim_state.mcap", 20),
                ),
            )
            artifacts = mock.Mock(episode_dir=exported_dir)
            metadata = {
                "episode_id": "episode_1",
                "task_name": "sim_spell_cat",
                "batch_name": "batch_a",
                "source_delivery": "task_a",
                "index_name": "collected",
                "data_date": "apr_11",
                "size": 10,
                "operator_id": "",
                "cameras": ["top", "left", "right"],
                "camera_profile_id": "profile_1",
                "_camera_profile": {"backend": "madrona"},
            }

            with (
                mock.patch(
                    "xdof_sim.dataset_export.s3_pipeline.discover_episode_sources",
                    return_value=[source],
                ),
                mock.patch(
                    "xdof_sim.dataset_export.s3_pipeline.shard_episode_sources",
                    return_value=[source],
                ),
                mock.patch(
                    "xdof_sim.dataset_export.s3_pipeline.stage_episode_locally",
                    return_value=staged_dir,
                ),
                mock.patch(
                    "xdof_sim.dataset_export.s3_pipeline.export_episode_with_backend_lifecycle",
                    return_value=(artifacts, metadata.copy()),
                ) as export_episode,
                mock.patch("xdof_sim.dataset_export.s3_pipeline.copy_local_dir_to_s3") as upload_episode_dir,
                mock.patch("xdof_sim.dataset_export.s3_pipeline.copy_local_file_to_s3") as upload_file,
                mock.patch("xdof_sim.dataset_export.s3_pipeline.cleanup_local_tree"),
            ):
                summary = export_s3_shard(
                    "s3://source-bucket/deliveries/",
                    "s3://dest-bucket/datasets/test_export",
                    scratch_dir,
                    config=ExportConfig(batch_name="batch_a"),
                    shard_index=0,
                    num_shards=1,
                )

            self.assertEqual(summary["exported_episodes"], 1)
            export_episode.assert_called_once_with(
                staged_dir,
                scratch_dir / "shard_00" / "dataset",
                config=ExportConfig(batch_name="batch_a"),
                source_delivery="task_a",
            )
            upload_episode_dir.assert_called_once_with(
                exported_dir,
                "s3://dest-bucket/datasets/test_export/data/episode_1",
                aws_profile=None,
                aws_region=None,
            )

            collected_path = Path(summary["collected_path"])
            self.assertTrue(collected_path.exists())
            collected = json.loads(collected_path.read_text())
            self.assertEqual(sorted(collected), ["episode_1"])

            camera_profiles_path = Path(summary["camera_profiles_path"])
            self.assertTrue(camera_profiles_path.exists())
            uploaded_names = sorted(Path(call.args[1]).name for call in upload_file.call_args_list)
            self.assertEqual(uploaded_names, ["camera_profiles_shard_00.json", "collected_shard_00.json"])

    def test_finalize_s3_dataset_merges_shards_downloads_numpy_and_uploads_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            scratch_dir = Path(tmpdir)
            metadata_objects = [
                S3ObjectInfo("dest-bucket", "datasets/test_export/metadata/collected_shard_00.json", 10),
                S3ObjectInfo("dest-bucket", "datasets/test_export/metadata/collected_shard_01.json", 10),
                S3ObjectInfo("dest-bucket", "datasets/test_export/metadata/camera_profiles_shard_00.json", 10),
            ]

            def fake_download(s3_uri, local_path, *, aws_profile=None, aws_region=None):
                local_path = Path(local_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                name = local_path.name
                if name == "collected_shard_00.json":
                    local_path.write_text(
                        json.dumps(
                            {
                                "episode_1": {
                                    "episode_id": "episode_1",
                                    "task_name": "sim_spell_cat",
                                    "batch_name": "batch_a",
                                    "source_delivery": "task_a",
                                    "index_name": "collected",
                                    "data_date": "apr_11",
                                    "size": 2,
                                    "operator_id": "",
                                    "cameras": ["top", "left", "right"],
                                    "camera_profile_id": "profile_1",
                                }
                            }
                        )
                    )
                elif name == "collected_shard_01.json":
                    local_path.write_text(
                        json.dumps(
                            {
                                "episode_2": {
                                    "episode_id": "episode_2",
                                    "task_name": "sim_spell_cat",
                                    "batch_name": "batch_a",
                                    "source_delivery": "task_a",
                                    "index_name": "collected",
                                    "data_date": "apr_11",
                                    "size": 2,
                                    "operator_id": "",
                                    "cameras": ["top", "left", "right"],
                                    "camera_profile_id": "profile_1",
                                }
                            }
                        )
                    )
                elif name == "camera_profiles_shard_00.json":
                    local_path.write_text(json.dumps({"profile_1": {"backend": "madrona"}}))
                elif name == "states_actions.npy":
                    episode_id = local_path.parent.name
                    value = 1.0 if episode_id == "episode_1" else 3.0
                    np.save(local_path, np.full((2, 28), value, dtype=np.float64))
                else:
                    raise AssertionError(f"Unexpected download target {name}")
                return local_path

            with (
                mock.patch(
                    "xdof_sim.dataset_export.s3_pipeline.list_s3_objects",
                    return_value=metadata_objects,
                ),
                mock.patch(
                    "xdof_sim.dataset_export.s3_pipeline.copy_s3_object_to_local",
                    side_effect=fake_download,
                ),
                mock.patch("xdof_sim.dataset_export.s3_pipeline.sync_local_dir_to_s3") as sync_dir,
            ):
                outputs = finalize_s3_dataset(
                    "s3://dest-bucket/datasets/test_export",
                    scratch_dir,
                    n_states=14,
                    n_actions=14,
                    val_ratio=0.5,
                    seed=0,
                )

            self.assertIn("norm_stats", outputs)
            self.assertIn("camera_profiles", outputs)
            collected = json.loads(Path(outputs["collected"]).read_text())
            self.assertEqual(sorted(collected), ["episode_1", "episode_2"])
            camera_profiles = json.loads(Path(outputs["camera_profiles"]).read_text())
            self.assertEqual(camera_profiles, {"profile_1": {"backend": "madrona"}})

            train_ids = json.loads(Path(outputs["train_episodes"]).read_text())
            val_ids = json.loads(Path(outputs["val_episodes"]).read_text())
            self.assertEqual(sorted(train_ids + val_ids), ["episode_1", "episode_2"])
            sync_dir.assert_called_once_with(
                Path(outputs["collected"]).parent,
                "s3://dest-bucket/datasets/test_export/metadata",
                aws_profile=None,
                aws_region=None,
            )


if __name__ == "__main__":
    unittest.main()
