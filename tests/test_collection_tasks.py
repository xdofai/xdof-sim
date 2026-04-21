from __future__ import annotations

import unittest

from xdof_sim.collection_tasks import (
    DATA_COLLECTION_TASKS,
    get_data_collection_task,
    list_data_collection_tasks,
    maybe_get_data_collection_task,
)


class CollectionTaskTests(unittest.TestCase):
    def test_declares_expected_ten_collection_task_families(self) -> None:
        names = [task.name for task in list_data_collection_tasks()]
        self.assertEqual(len(names), 10)
        self.assertEqual(
            names,
            [
                "chess",
                "spelling_with_blocks",
                "marker_in_drawer",
                "dish_rack",
                "sweep",
                "place_bottle_in_bin",
                "flip_mug",
                "mug_on_tree",
                "pouring_beads",
                "random_object_handover",
            ],
        )
        self.assertEqual(DATA_COLLECTION_TASKS, tuple(list_data_collection_tasks()))

    def test_resolves_aliases_and_grouped_sim_tasks(self) -> None:
        spelling = get_data_collection_task("spelling with blocks")
        self.assertEqual(spelling.env_task, "blocks")
        self.assertIn("spell_cat", spelling.sim_task_names)
        self.assertIn("spell_agi", spelling.sim_task_names)
        self.assertTrue(spelling.has_randomization)
        self.assertEqual(spelling.randomization_mode, "registry")

        drawer = get_data_collection_task("marker in drawer")
        self.assertEqual(drawer.env_task, "drawer")
        self.assertEqual(
            drawer.sim_task_names,
            (
                "put_markers_in_top_drawer",
                "put_markers_in_middle_drawer",
                "put_markers_in_bottom_drawer",
            ),
        )

    def test_tracks_missing_and_special_randomization_paths(self) -> None:
        sweep = get_data_collection_task("sweep")
        self.assertTrue(sweep.has_randomization)
        self.assertEqual(sweep.randomization_mode, "registry")
        self.assertEqual(sweep.scene_xml.name, "yam_sweep_scene.xml")

        handover = get_data_collection_task("random object handover")
        self.assertTrue(handover.has_randomization)
        self.assertEqual(handover.randomization_mode, "special")
        self.assertEqual(handover.env_task, "inhand_transfer")
        self.assertEqual(handover.sim_task_names, ())

    def test_maybe_get_returns_none_for_unknown_name(self) -> None:
        self.assertIsNone(maybe_get_data_collection_task("spelling"))
        self.assertIsNone(maybe_get_data_collection_task("definitely_not_real"))


if __name__ == "__main__":
    unittest.main()
