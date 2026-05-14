from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import mujoco

import xdof_sim
from xdof_sim.examples.vr_streamer import (
    _parse_visible_groups_arg,
    _should_export_scene_after_reset,
    export_body_glbs,
)


class VisibleGroupsArgTests(unittest.TestCase):
    def test_defaults(self) -> None:
        self.assertEqual(_parse_visible_groups_arg(None), (0, 1, 2))

    def test_space_and_comma_separated_values(self) -> None:
        self.assertEqual(_parse_visible_groups_arg(["0", "2,4", "2"]), (0, 2, 4))

    def test_all_keyword(self) -> None:
        self.assertEqual(_parse_visible_groups_arg(["all"]), (0, 1, 2, 3, 4, 5))

    def test_invalid_group_raises(self) -> None:
        with self.assertRaises(ValueError):
            _parse_visible_groups_arg(["8"])


class SceneReloadDecisionTests(unittest.TestCase):
    def test_model_change_requires_scene_export(self) -> None:
        previous_model = object()
        current_model = object()
        self.assertTrue(
            _should_export_scene_after_reset(
                previous_model=previous_model,  # type: ignore[arg-type]
                current_model=current_model,  # type: ignore[arg-type]
                task="chess",
                previous_metadata=None,
                current_metadata=None,
            )
        )

    def test_chess_tin_visibility_change_requires_scene_export(self) -> None:
        model = object()
        self.assertTrue(
            _should_export_scene_after_reset(
                previous_model=model,  # type: ignore[arg-type]
                current_model=model,  # type: ignore[arg-type]
                task="chess",
                previous_metadata={"tin_active": False},
                current_metadata={"tin_active": True},
            )
        )

    def test_non_chess_metadata_change_does_not_force_scene_export(self) -> None:
        model = object()
        self.assertFalse(
            _should_export_scene_after_reset(
                previous_model=model,  # type: ignore[arg-type]
                current_model=model,  # type: ignore[arg-type]
                task="dishrack",
                previous_metadata={"tin_active": False},
                current_metadata={"tin_active": True},
            )
        )


class VrMeshExportTests(unittest.TestCase):
    def test_hidden_chess_tin_is_not_exported_for_non_tin_setup(self) -> None:
        env = xdof_sim.make_env(task="chess", render_cameras=False)
        try:
            env.reset(
                seed=10,
                randomize=True,
                options={
                    "randomization": {
                        "scenario": "table_setup",
                        "target_count": 12,
                        "randomize_scales": False,
                    }
                },
            )
            tin_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "tin_box")
            self.assertLess(float(env.data.xpos[tin_body][2]), -1.0)
            with tempfile.TemporaryDirectory() as tmp_dir:
                body_info, _stats = export_body_glbs(env.model, env.data, Path(tmp_dir))
            self.assertNotIn("tin_box", body_info)

            env.reset(
                seed=11,
                randomize=True,
                options={
                    "randomization": {
                        "scenario": "tin_setup",
                        "target_count": 12,
                        "randomize_scales": False,
                    }
                },
            )
            tin_body = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "tin_box")
            self.assertGreater(float(env.data.xpos[tin_body][2]), 0.7)
            with tempfile.TemporaryDirectory() as tmp_dir:
                body_info, _stats = export_body_glbs(env.model, env.data, Path(tmp_dir))
            self.assertIn("tin_box", body_info)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
