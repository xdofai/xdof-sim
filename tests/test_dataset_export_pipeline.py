from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from xdof_sim.dataset_export.pipeline import export_dataset, find_episode_dirs
from xdof_sim.dataset_export.types import ExportConfig


class DatasetExportPipelineTests(unittest.TestCase):
    def test_find_episode_dirs_deduplicates_nested_copies(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "task_a" / "episode_1").mkdir(parents=True)
            (root / "task_a" / "episode_1" / "output.mcap").write_bytes(b"")
            (root / "task_a" / "episode_1" / "episode_1").mkdir(parents=True)
            (root / "task_a" / "episode_1" / "episode_1" / "output.mcap").write_bytes(b"")
            (root / "task_b" / "episode_2").mkdir(parents=True)
            (root / "task_b" / "episode_2" / "output.mcap").write_bytes(b"")

            episode_dirs = find_episode_dirs(root)
            self.assertEqual(
                episode_dirs,
                [root / "task_a" / "episode_1", root / "task_b" / "episode_2"],
            )

    def test_export_dataset_writes_collected_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            input_root = Path(tmpdir) / "input"
            output_root = Path(tmpdir) / "output"
            (input_root / "task_a" / "episode_1").mkdir(parents=True)
            (input_root / "task_a" / "episode_1" / "output.mcap").write_bytes(b"")
            (input_root / "task_b" / "episode_2").mkdir(parents=True)
            (input_root / "task_b" / "episode_2" / "output.mcap").write_bytes(b"")
            config = ExportConfig(batch_name="batch_a")

            def fake_export_episode(episode_dir, output_root_arg, *, config, source_delivery=None):
                episode_id = episode_dir.name
                return None, None, {
                    "episode_id": episode_id,
                    "task_name": "sim_spell_cat",
                    "batch_name": config.batch_name,
                    "source_delivery": episode_dir.parent.name,
                    "index_name": "collected",
                    "data_date": "apr_11",
                    "size": 10,
                    "operator_id": "",
                    "cameras": ["top", "left", "right"],
                    "camera_profile_id": "profile_1",
                    "_camera_profile": {"backend": "mjwarp"},
                }

            with mock.patch("xdof_sim.dataset_export.pipeline.export_episode", side_effect=fake_export_episode) as export_one:
                collected = export_dataset(input_root, output_root, config=config)

            self.assertEqual(export_one.call_count, 2)
            self.assertEqual(sorted(collected), ["episode_1", "episode_2"])
            collected_path = output_root / "metadata" / "collected.json"
            camera_profiles_path = output_root / "metadata" / "camera_profiles.json"
            self.assertTrue(collected_path.exists())
            self.assertTrue(camera_profiles_path.exists())
            self.assertEqual(sorted(json.loads(collected_path.read_text())), ["episode_1", "episode_2"])


if __name__ == "__main__":
    unittest.main()
