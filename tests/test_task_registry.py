from __future__ import annotations

import unittest

import xdof_sim
from xdof_sim.task_registry import (
    get_task_scene_xml,
    maybe_get_scene_task_spec,
    resolve_env_task_name,
    resolve_task,
)


class TaskRegistryTests(unittest.TestCase):
    def test_resolve_prompt_alias_to_scene_task(self) -> None:
        resolved = resolve_task("sim_spell_cat")
        self.assertEqual(resolved.env_task, "blocks")
        self.assertIsNotNone(resolved.scene_task)
        self.assertEqual(resolved.scene_task.name, "blocks")
        self.assertIsNotNone(resolved.task_spec)
        self.assertEqual(resolved.task_spec.prompt, "spell cat")

    def test_legacy_spelling_scene_alias_is_gone(self) -> None:
        self.assertIsNone(maybe_get_scene_task_spec("spelling"))
        self.assertEqual(resolve_env_task_name("put_markers_in_top_drawer"), "drawer")
        self.assertEqual(get_task_scene_xml("put_markers_in_top_drawer"), get_task_scene_xml("drawer"))

    def test_make_env_accepts_prompt_alias_task_names(self) -> None:
        env = xdof_sim.make_env(task="throw plastic bottles in bin", render_cameras=False)
        try:
            self.assertEqual(env._task, "bottles")
            self.assertEqual(env.prompt, "throw plastic bottles in bin")
            self.assertIsNotNone(env._task_spec)
            self.assertIsNotNone(env._task_evaluator)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
