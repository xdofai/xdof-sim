from __future__ import annotations

import unittest
from unittest import mock

from xdof_sim.dataset_export.s3_source import discover_episode_sources, shard_episode_sources
from xdof_sim.dataset_export.s3_utils import S3ObjectInfo, parse_s3_uri


class S3UtilsTests(unittest.TestCase):
    def test_parse_s3_uri_and_child(self) -> None:
        uri = parse_s3_uri("s3://example-bucket/path/to/root/")
        self.assertEqual(uri.bucket, "example-bucket")
        self.assertEqual(uri.key, "path/to/root")
        self.assertEqual(uri.child("data", "episode_1").uri, "s3://example-bucket/path/to/root/data/episode_1")


class S3SourceTests(unittest.TestCase):
    def test_discover_episode_sources_deduplicates_nested_prefixes(self) -> None:
        objects = [
            S3ObjectInfo("bucket", "deliveries/task_a/episode_1/output.mcap", 10),
            S3ObjectInfo("bucket", "deliveries/task_a/episode_1/sim_state.mcap", 20),
            S3ObjectInfo("bucket", "deliveries/task_a/episode_1/randomization.json", 30),
            S3ObjectInfo("bucket", "deliveries/task_a/episode_1/episode_1/output.mcap", 11),
            S3ObjectInfo("bucket", "deliveries/task_a/episode_1/episode_1/sim_state.mcap", 21),
            S3ObjectInfo("bucket", "deliveries/task_b/episode_2/output.mcap", 12),
            S3ObjectInfo("bucket", "deliveries/task_b/episode_2/sim_state.mcap", 22),
        ]

        with mock.patch(
            "xdof_sim.dataset_export.s3_source.list_s3_objects",
            return_value=objects,
        ):
            sources = discover_episode_sources("s3://bucket/deliveries/")

        self.assertEqual([source.relative_episode_prefix for source in sources], ["task_a/episode_1", "task_b/episode_2"])
        file_names = sorted(sources[0].file_map())
        self.assertEqual(file_names, ["output.mcap", "randomization.json", "sim_state.mcap"])
        self.assertEqual(sources[0].source_delivery, "task_a")

    def test_shard_episode_sources_assigns_round_robin(self) -> None:
        with mock.patch(
            "xdof_sim.dataset_export.s3_source.list_s3_objects",
            return_value=[
                S3ObjectInfo("bucket", f"deliveries/task/episode_{idx}/output.mcap", 10 + idx)
                for idx in range(5)
            ]
            + [
                S3ObjectInfo("bucket", f"deliveries/task/episode_{idx}/sim_state.mcap", 20 + idx)
                for idx in range(5)
            ],
        ):
            sources = discover_episode_sources("s3://bucket/deliveries/")

        shard = shard_episode_sources(sources, shard_index=1, num_shards=3)
        self.assertEqual(
            [source.episode_name for source in shard],
            ["episode_1", "episode_4"],
        )

    def test_source_delivery_uses_source_prefix_basename_for_task_scoped_inputs(self) -> None:
        objects = [
            S3ObjectInfo("bucket", "deliveries/sim_spell_cat/episode_1/output.mcap", 10),
            S3ObjectInfo("bucket", "deliveries/sim_spell_cat/episode_1/sim_state.mcap", 20),
        ]
        with mock.patch(
            "xdof_sim.dataset_export.s3_source.list_s3_objects",
            return_value=objects,
        ):
            sources = discover_episode_sources("s3://bucket/deliveries/sim_spell_cat/")
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0].relative_episode_prefix, "episode_1")
        self.assertEqual(sources[0].source_delivery, "sim_spell_cat")


if __name__ == "__main__":
    unittest.main()
