from __future__ import annotations

from pathlib import Path
import unittest

import mujoco
import numpy as np

import xdof_sim
from xdof_sim.randomization import (
    DishRackRandomizer,
    SweepRandomizer,
    _build_dishrack_scene_xml,
    _dishrack_geom_world_bounds,
    _dishrack_plate_body_name,
    _dishrack_plate_joint_name,
    _dishrack_variant_names,
)


def _body_in_subtree(model: mujoco.MjModel, body_id: int, root_body_id: int) -> bool:
    current = int(body_id)
    while current >= 0:
        if current == root_body_id:
            return True
        parent = int(model.body_parentid[current])
        if parent == current:
            break
        current = parent
    return False


class SweepRandomizerTests(unittest.TestCase):
    def test_sweep_randomizer_uses_pose_randomization_with_clustered_trash(self) -> None:
        scene_path = Path(__file__).resolve().parents[1] / "xdof_sim" / "models" / "yam_sweep_scene.xml"
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        randomizer = SweepRandomizer()
        state = randomizer.randomize(model, data, seed=123)

        self.assertEqual(
            set(state.object_states),
            {
                "brush_jnt",
                "bin_joint",
                "dustpan_jnt",
                "trash_1_jnt",
                "trash_2_jnt",
                "trash_3_jnt",
                "trash_4_jnt",
                "trash_5_jnt",
                "trash_6_jnt",
                "trash_7_jnt",
            },
        )
        trash_count = int(state.metadata["trash_count"])
        active_trash_joints = list(state.metadata["trash_joints"])
        inactive_trash_joints = [
            joint_name
            for joint_name in SweepRandomizer._all_trash_joints
            if joint_name not in active_trash_joints
        ]
        self.assertGreaterEqual(trash_count, SweepRandomizer.min_trash_count)
        self.assertLessEqual(trash_count, SweepRandomizer.max_trash_count)
        self.assertEqual(len(active_trash_joints), trash_count)

        brush_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "brush_jnt")
        brush_qadr = int(model.jnt_qposadr[brush_jnt_id])
        brush_nominal = np.array([0.50, 0.20, 0.77])
        brush_randomized = data.qpos[brush_qadr: brush_qadr + 3].copy()
        self.assertFalse(np.allclose(brush_randomized, brush_nominal))

        trash_xy = np.array(
            [state.object_states[joint_name]["pos"][:2] for joint_name in active_trash_joints],
            dtype=np.float64,
        )
        span = trash_xy.max(axis=0) - trash_xy.min(axis=0)
        self.assertLess(float(span[0]), 0.35)
        self.assertLess(float(span[1]), 0.35)

        trash_quats = np.array(
            [state.object_states[joint_name]["quat"] for joint_name in active_trash_joints],
            dtype=np.float64,
        )
        self.assertTrue(
            np.any(np.abs(trash_quats[:, 1]) > 1e-3) or np.any(np.abs(trash_quats[:, 2]) > 1e-3),
            msg="expected at least one trash object to receive non-planar orientation randomization",
        )

        for joint_name in ("brush_jnt", "bin_joint", "dustpan_jnt", *active_trash_joints):
            obj_state = state.object_states[joint_name]
            self.assertGreaterEqual(obj_state["pos"][0], 0.36)
            self.assertLessEqual(obj_state["pos"][0], 0.82)
            self.assertGreaterEqual(obj_state["pos"][1], -0.55)
            self.assertLessEqual(obj_state["pos"][1], 0.55)

        for joint_name in inactive_trash_joints:
            self.assertLess(state.object_states[joint_name]["pos"][2], 0.0)

    def test_sweep_randomizer_keeps_trash_outside_tool_keepout_radius(self) -> None:
        scene_path = Path(__file__).resolve().parents[1] / "xdof_sim" / "models" / "yam_sweep_scene.xml"
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        randomizer = SweepRandomizer()

        for seed in range(10):
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            state = randomizer.randomize(model, data, seed=seed)

            for trash_name in state.metadata["trash_joints"]:
                trash_xy = np.asarray(state.object_states[trash_name]["pos"][:2], dtype=np.float64)
                self.assertTrue(
                    randomizer._tool_keepout_ok(
                        trash_xy,
                        states=state.object_states,
                    ),
                    msg=f"{trash_name} landed inside the sampled brush/dustpan keep-out footprint for seed {seed}",
                )

    def test_sweep_randomizer_randomizes_trash_count_between_two_and_four(self) -> None:
        scene_path = Path(__file__).resolve().parents[1] / "xdof_sim" / "models" / "yam_sweep_scene.xml"
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)
        randomizer = SweepRandomizer()

        observed_counts: set[int] = set()
        for seed in range(24):
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)
            state = randomizer.randomize(model, data, seed=seed)
            observed_counts.add(int(state.metadata["trash_count"]))

        self.assertTrue(observed_counts.issubset({2, 3, 4}))
        self.assertGreaterEqual(len(observed_counts), 2)


class DishRackRandomizerBoundsTests(unittest.TestCase):
    def test_bounds_check_keeps_rack_footprint_off_table_edges(self) -> None:
        randomizer = DishRackRandomizer()
        randomizer._current_rack_half_extents_xy = (0.20, 0.10)

        x_min, _x_max, _y_min, _y_max = randomizer.rack_table_edge_bounds
        margin = randomizer.rack_table_margin_m
        quat = [1.0, 0.0, 0.0, 0.0]

        inside = {
            "dishrack": {
                "pos": [x_min + 0.20 + margin + 0.001, 0.0, 0.75],
                "quat": quat,
            }
        }
        too_close = {
            "dishrack": {
                "pos": [x_min + 0.20 + margin - 0.001, 0.0, 0.75],
                "quat": quat,
            }
        }

        self.assertTrue(randomizer._bounds_ok(inside))
        self.assertFalse(randomizer._bounds_ok(too_close))

    def test_bounds_check_keeps_plate_footprint_off_table_edges(self) -> None:
        randomizer = DishRackRandomizer()
        randomizer._current_plate_variants = ["plate_0"]
        randomizer._current_plate_collision_radii = [0.05]

        x_min, _x_max, _y_min, _y_max = randomizer.rack_table_edge_bounds
        margin = 0.05 + randomizer.plate_table_margin_m
        quat = [1.0, 0.0, 0.0, 0.0]

        inside = {
            "plate_joint": {
                "pos": [x_min + margin + 0.001, 0.0, 0.75],
                "quat": quat,
            }
        }
        too_close = {
            "plate_joint": {
                "pos": [x_min + margin - 0.001, 0.0, 0.75],
                "quat": quat,
            }
        }

        self.assertTrue(randomizer._bounds_ok(inside))
        self.assertFalse(randomizer._bounds_ok(too_close))


class DishRackVariantRandomizationTests(unittest.TestCase):
    def test_plate_variant_assets_compile_with_capped_collision_hulls(self) -> None:
        plate_root = Path(__file__).resolve().parents[1] / "xdof_sim" / "models" / "assets" / "task_dishrack" / "plate"

        for variant_name in _dishrack_variant_names("plate"):
            variant_dir = plate_root / variant_name
            model_path = variant_dir / "model.xml"
            mujoco.MjModel.from_xml_path(str(model_path))

            collision_meshes = sorted((variant_dir / "collision").glob("model_collision_*.obj"))
            if not collision_meshes:
                collision_meshes = sorted(variant_dir.glob("model_collision_*.obj"))

            self.assertGreater(
                len(collision_meshes),
                0,
                msg=f"{variant_name} is missing collision meshes",
            )
            self.assertLessEqual(
                len(collision_meshes),
                16,
                msg=f"{variant_name} exceeds the 16-hull collision cap",
            )

    def test_imported_plate_variants_spawn_from_bottom_origin_across_scale_range(self) -> None:
        base_dir = Path(__file__).resolve().parents[1] / "xdof_sim" / "models"
        base_xml = (base_dir / "yam_dishrack_base.xml").read_text()
        table_z = 0.75

        for variant_name in _dishrack_variant_names("plate"):
            for scale_factor in (0.95, 1.0, 1.05):
                xml = _build_dishrack_scene_xml(
                    dish_rack_variant="dish_rack_0",
                    plate_variant=variant_name,
                    scale_states={"plate_joint": scale_factor},
                    base_scene_xml=base_xml,
                    base_scene_dir=base_dir,
                )
                model = mujoco.MjModel.from_xml_string(xml)
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)

                plate_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate")
                min_z = min(
                    float(_dishrack_geom_world_bounds(model, data, geom_id)[0][2])
                    for geom_id in range(model.ngeom)
                    if _body_in_subtree(model, int(model.geom_bodyid[geom_id]), plate_body_id)
                )
                self.assertGreaterEqual(
                    min_z,
                    table_z - 1e-6,
                    msg=(
                        f"{variant_name} at scale {scale_factor:.2f} bottomed at {min_z:.6f} "
                        f"below table plane {table_z:.6f}"
                    ),
                )

    def test_imported_dishrack_variants_spawn_from_bottom_origin(self) -> None:
        base_xml = (
            Path(__file__).resolve().parents[1] / "xdof_sim" / "models" / "yam_dishrack_base.xml"
        ).read_text()
        table_z = 0.75

        for variant_name in _dishrack_variant_names("dish_rack"):
            xml = _build_dishrack_scene_xml(
                dish_rack_variant=variant_name,
                plate_variant="plate_0",
                scale_states={"plate_joint": 1.0},
                base_scene_xml=base_xml,
                base_scene_dir=Path(__file__).resolve().parents[1] / "xdof_sim" / "models",
            )
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            dishrack_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dishrack")
            min_z = min(
                float(_dishrack_geom_world_bounds(model, data, geom_id)[0][2])
                for geom_id in range(model.ngeom)
                if _body_in_subtree(model, int(model.geom_bodyid[geom_id]), dishrack_body_id)
            )
            self.assertGreaterEqual(
                min_z,
                table_z - 1e-6,
                msg=f"{variant_name} bottomed at {min_z:.6f} below table plane {table_z:.6f}",
            )

    def test_build_dishrack_scene_xml_loads_default_variants(self) -> None:
        xml = _build_dishrack_scene_xml(
            dish_rack_variant="dish_rack_0",
            plate_variant="plate_0",
            scale_states={"plate_joint": 1.0},
            base_scene_xml=None,
            base_scene_dir=None,
        )
        model = mujoco.MjModel.from_xml_string(xml)

        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dishrack"), 0)
        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate"), 0)
        self.assertIn("dish_rack_0", xml)
        self.assertIn("plate_0", xml)

    def test_build_dishrack_scene_xml_loads_selected_variants(self) -> None:
        xml = _build_dishrack_scene_xml(
            dish_rack_variant="dish_rack_7",
            plate_variant="plate_1",
            scale_states={"plate_joint": 1.0},
            base_scene_xml=None,
            base_scene_dir=None,
        )
        model = mujoco.MjModel.from_xml_string(xml)

        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dishrack"), 0)
        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "plate"), 0)
        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint"), 0)

    def test_build_dishrack_scene_xml_supports_multiple_plate_instances(self) -> None:
        plate_variants = ["plate_0", "plate_1", "plate_2", "plate_0"]
        xml = _build_dishrack_scene_xml(
            dish_rack_variant="dish_rack_7",
            plate_variants=plate_variants,
            scale_states={
                "plate_joint": 1.0,
                "plate_joint_1": 0.97,
                "plate_joint_2": 1.03,
                "plate_joint_3": 1.01,
            },
            base_scene_xml=None,
            base_scene_dir=None,
        )
        model = mujoco.MjModel.from_xml_string(xml)

        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "dishrack"), 0)
        for index in range(len(plate_variants)):
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, _dishrack_plate_body_name(index)),
                0,
            )
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, _dishrack_plate_joint_name(index)),
                0,
            )

    def test_make_env_randomizes_dishrack_mesh_variants(self) -> None:
        env = xdof_sim.make_env(task="dishrack", render_cameras=False)
        try:
            self.assertEqual(env._scene_xml.name, "yam_dishrack_base.xml")
            self.assertGreaterEqual(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "dishrack"), 0)
            self.assertGreaterEqual(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "plate_joint"), 0)

            for seed in range(8):
                env.reset(seed=seed, randomize=True)
                state = env._last_randomization
                self.assertIsNotNone(state)
                assert state is not None

                plate_count = int(state.metadata["plate_count"])
                plate_variants = list(state.metadata["plate_variants"])
                plate_joint_names = {
                    name for name in state.object_states if name == "plate_joint" or name.startswith("plate_joint_")
                }
                expected_joint_names = {_dishrack_plate_joint_name(index) for index in range(plate_count)}

                self.assertIn("dishrack", state.object_states)
                self.assertEqual(plate_joint_names, expected_joint_names)
                self.assertGreaterEqual(plate_count, 1)
                self.assertLessEqual(plate_count, 4)
                self.assertEqual(len(plate_variants), plate_count)
                self.assertIn(state.metadata["plate_variant"], set(_dishrack_variant_names("plate")))
                self.assertIn(state.metadata["dish_rack_variant"], set(_dishrack_variant_names("dish_rack")))

                for index in range(plate_count):
                    self.assertGreaterEqual(
                        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, _dishrack_plate_body_name(index)),
                        0,
                    )
                    self.assertGreaterEqual(
                        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, _dishrack_plate_joint_name(index)),
                        0,
                    )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
