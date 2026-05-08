from __future__ import annotations

from itertools import product
from pathlib import Path
import unittest

import mujoco
import numpy as np

import xdof_sim
from xdof_sim.randomization import (
    DishRackRandomizer,
    SweepRandomizer,
    _build_dishrack_scene_xml,
    _build_mug_scene_xml,
    _dishrack_geom_world_bounds,
    _dishrack_plate_body_name,
    _dishrack_plate_joint_name,
    _dishrack_variant_names,
    _MUG_FLIP_SPAWN_CLEARANCE_M,
    _MUG_FLIP_TRAY_FLOOR_Z_OFFSET,
    _MUG_FLIP_TRAY_INNER_HALF_XY,
    _mug_plain_color_material_names,
    _mug_variant_names,
)
from xdof_sim.env import project_policy_state


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


def _subtree_collision_bounds_in_body_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    root_body_id: int,
    frame_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    frame_pos = np.asarray(data.xpos[frame_body_id], dtype=np.float64)
    frame_rot = np.asarray(data.xmat[frame_body_id], dtype=np.float64).reshape(3, 3)
    points: list[np.ndarray] = []

    for geom_id in range(model.ngeom):
        if not _body_in_subtree(model, int(model.geom_bodyid[geom_id]), root_body_id):
            continue
        if not (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id])):
            continue

        geom_type = int(model.geom_type[geom_id])
        if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = int(model.geom_dataid[geom_id])
            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            geom_rot = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            geom_pos = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            local_points = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)
            world_points = local_points @ geom_rot.T + geom_pos
        else:
            lower, upper = _dishrack_geom_world_bounds(model, data, geom_id)
            world_points = np.asarray(
                list(product(*zip(lower.tolist(), upper.tolist()))),
                dtype=np.float64,
            )

        points.append((world_points - frame_pos) @ frame_rot)

    if not points:
        raise AssertionError("Expected at least one collision geom in body subtree")

    stacked = np.vstack(points)
    return stacked.min(axis=0), stacked.max(axis=0)


def _has_mug_mug_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    mug_roots: set[int] = set()
    for index in range(1, 5):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"mug_{index}")
        if body_id >= 0:
            mug_roots.add(int(model.body_rootid[body_id]))

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        root_1 = int(model.body_rootid[int(model.geom_bodyid[contact.geom1])])
        root_2 = int(model.body_rootid[int(model.geom_bodyid[contact.geom2])])
        if root_1 != root_2 and root_1 in mug_roots and root_2 in mug_roots:
            return True
    return False


def _has_mug_tree_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    tree_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug_tree")
    if tree_body_id < 0:
        return False

    mug_roots: set[int] = set()
    for index in range(1, 4):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"mug_{index}")
        if body_id >= 0:
            mug_roots.add(int(model.body_rootid[body_id]))

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        body_1 = int(model.geom_bodyid[contact.geom1])
        body_2 = int(model.geom_bodyid[contact.geom2])
        root_1 = int(model.body_rootid[body_1])
        root_2 = int(model.body_rootid[body_2])
        if (
            root_1 in mug_roots
            and _body_in_subtree(model, body_2, tree_body_id)
        ) or (
            root_2 in mug_roots
            and _body_in_subtree(model, body_1, tree_body_id)
        ):
            return True
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


class ResetArmPoseTests(unittest.TestCase):
    def _set_policy_state(self, env, state: np.ndarray) -> np.ndarray:
        env._set_qpos_from_state(np.asarray(state, dtype=np.float32))
        mujoco.mj_forward(env.model, env.data)
        return project_policy_state(
            np.asarray(env.data.qpos, dtype=np.float64),
            env._qpos_indices,
            env._gripper_indices,
        )

    def test_reset_preserves_current_arm_state_without_randomization(self) -> None:
        env = xdof_sim.make_env(task="bottles", render_cameras=False)
        try:
            env.reset(seed=0, randomize=False)
            target = env.get_init_q().astype(np.float32, copy=True)
            target[:6] += np.array([0.10, -0.05, 0.08, -0.04, 0.06, -0.03], dtype=np.float32)
            target[7:13] += np.array([-0.07, 0.05, -0.06, 0.04, -0.03, 0.02], dtype=np.float32)
            target[6] = 0.25
            target[13] = 0.75
            expected = self._set_policy_state(env, target)

            env.reset(seed=1, randomize=False)

            actual = project_policy_state(
                np.asarray(env.data.qpos, dtype=np.float64),
                env._qpos_indices,
                env._gripper_indices,
            )
            np.testing.assert_allclose(actual, expected, atol=1e-6)
        finally:
            env.close()

    def test_reset_preserves_current_arm_state_across_dishrack_scene_reload(self) -> None:
        env = xdof_sim.make_env(task="dishrack", scene="hybrid", render_cameras=False)
        try:
            env.reset(seed=0, randomize=True)
            target = env.get_init_q().astype(np.float32, copy=True)
            target[:6] += np.array([-0.09, 0.04, -0.07, 0.03, -0.05, 0.02], dtype=np.float32)
            target[7:13] += np.array([0.06, -0.03, 0.05, -0.02, 0.04, -0.01], dtype=np.float32)
            target[6] = 0.40
            target[13] = 0.60
            expected = self._set_policy_state(env, target)

            obs, info = env.reset(seed=1, randomize=True)
            self.assertIn("randomization", info)
            self.assertEqual(obs["state"].shape[0], env.single_timestep_action_dim)

            actual = project_policy_state(
                np.asarray(env.data.qpos, dtype=np.float64),
                env._qpos_indices,
                env._gripper_indices,
            )
            np.testing.assert_allclose(actual, expected, atol=1e-6)
        finally:
            env.close()


class DishRackVariantRandomizationTests(unittest.TestCase):
    def test_removed_dishrack_variant_is_not_enumerated(self) -> None:
        self.assertNotIn("dish_rack_10", _dishrack_variant_names("dish_rack"))

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


class MugVariantRandomizationTests(unittest.TestCase):
    def test_mug_variant_names_include_copied_plain_base_first(self) -> None:
        for task_name in ("mug_flip", "mug_tree"):
            variants = _mug_variant_names(task_name)
            self.assertEqual(variants[0], "mug_0")
            self.assertGreaterEqual(len(variants), 2)
            self.assertIn("mug_1", variants)
            self.assertTrue(_mug_plain_color_material_names(task_name, 0))

    def test_build_mug_scene_xml_loads_selected_variants(self) -> None:
        for task_name in ("mug_flip", "mug_tree"):
            for variant_name in ("mug_0", "mug_1"):
                xml = _build_mug_scene_xml(
                    task_name=task_name,
                    mug_variant=variant_name,
                    scale_states={},
                    base_scene_xml=None,
                    base_scene_dir=None,
                )
                self.assertIn(f"task_{task_name}/mug/{variant_name}", xml)
                model = mujoco.MjModel.from_xml_string(xml)
                self.assertGreaterEqual(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug_1"),
                    0,
                )
                self.assertGreaterEqual(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "mug_2"),
                    0,
                )

    def test_build_mug_flip_scene_xml_supports_variable_mug_count(self) -> None:
        for mug_count in range(1, 5):
            mug_variants = [f"mug_{index}" for index in range(mug_count)]
            xml = _build_mug_scene_xml(
                task_name="mug_flip",
                mug_variant="mug_0",
                mug_variants=mug_variants,
                scale_states={},
                base_scene_xml=None,
                base_scene_dir=None,
            )
            model = mujoco.MjModel.from_xml_string(xml)

            for index in range(1, mug_count + 1):
                self.assertGreaterEqual(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"mug_{index}"),
                    0,
                )
                self.assertGreaterEqual(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"mug_{index}_jnt"),
                    0,
                )
            for index in range(mug_count + 1, 5):
                self.assertLess(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"mug_{index}"),
                    0,
                )

    def test_build_mug_tree_scene_xml_supports_variable_mug_count(self) -> None:
        for mug_count in range(1, 4):
            mug_variants = [f"mug_{index}" for index in range(mug_count)]
            xml = _build_mug_scene_xml(
                task_name="mug_tree",
                mug_variant="mug_0",
                mug_variants=mug_variants,
                scale_states={},
                base_scene_xml=None,
                base_scene_dir=None,
            )
            model = mujoco.MjModel.from_xml_string(xml)

            for index in range(1, mug_count + 1):
                self.assertGreaterEqual(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"mug_{index}"),
                    0,
                )
                self.assertGreaterEqual(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"mug_{index}_jnt"),
                    0,
                )
            for index in range(mug_count + 1, 4):
                self.assertLess(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"mug_{index}"),
                    0,
                )

    def test_make_env_randomizes_mug_flip_count_and_instance_variants(self) -> None:
        env = xdof_sim.make_env(task="mug_flip", render_cameras=False)
        try:
            observed_counts: set[int] = set()
            observed_independent_sample = False
            for seed in range(12):
                env.reset(seed=seed, randomize=True)
                state = env._last_randomization
                self.assertIsNotNone(state)
                assert state is not None

                mug_count = int(state.metadata["mug_count"])
                mug_variants = list(state.metadata["mug_variants"])
                observed_counts.add(mug_count)
                observed_independent_sample = observed_independent_sample or len(set(mug_variants)) > 1

                self.assertGreaterEqual(mug_count, 1)
                self.assertLessEqual(mug_count, 4)
                self.assertEqual(len(mug_variants), mug_count)
                self.assertEqual(
                    {name for name in state.object_states if name.startswith("mug_")},
                    {f"mug_{index}_jnt" for index in range(1, mug_count + 1)},
                )
                self.assertIn("tray", state.object_states)

                for index in range(1, mug_count + 1):
                    self.assertGreaterEqual(
                        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"mug_{index}"),
                        0,
                    )
                for index in range(mug_count + 1, 5):
                    self.assertLess(
                        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"mug_{index}"),
                        0,
                    )

            self.assertGreaterEqual(len(observed_counts), 2)
            self.assertTrue(observed_independent_sample)
        finally:
            env.close()

    def test_make_env_randomizes_mug_tree_count_and_instance_variants(self) -> None:
        env = xdof_sim.make_env(task="mug_tree", render_cameras=False)
        try:
            observed_counts: set[int] = set()
            observed_independent_sample = False
            for seed in range(12):
                env.reset(seed=seed, randomize=True)
                state = env._last_randomization
                self.assertIsNotNone(state)
                assert state is not None

                mug_count = int(state.metadata["mug_count"])
                mug_variants = list(state.metadata["mug_variants"])
                observed_counts.add(mug_count)
                observed_independent_sample = observed_independent_sample or len(set(mug_variants)) > 1

                self.assertGreaterEqual(mug_count, 1)
                self.assertLessEqual(mug_count, 3)
                self.assertEqual(len(mug_variants), mug_count)
                self.assertEqual(
                    {name for name in state.object_states if name.startswith("mug_")},
                    {"mug_tree", *{f"mug_{index}_jnt" for index in range(1, mug_count + 1)}},
                )
                self.assertFalse(_has_mug_mug_contact(env.model, env.data))
                self.assertFalse(_has_mug_tree_contact(env.model, env.data))

                for index in range(1, mug_count + 1):
                    self.assertGreaterEqual(
                        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"mug_{index}"),
                        0,
                    )
                for index in range(mug_count + 1, 4):
                    self.assertLess(
                        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"mug_{index}"),
                        0,
                    )

            self.assertGreaterEqual(len(observed_counts), 2)
            self.assertTrue(observed_independent_sample)
        finally:
            env.close()

    def test_make_env_randomizes_mug_task_scales_by_default(self) -> None:
        cases = (
            ("mug_flip", ["mug_0", "mug_0"]),
            ("mug_tree", ["mug_0", "mug_0"]),
        )
        for task_name, mug_variants in cases:
            env = xdof_sim.make_env(task=task_name, render_cameras=False)
            try:
                env.reset(
                    seed=4,
                    randomize=True,
                    options={"randomization": {"mug_variants": mug_variants}},
                )
                state = env._last_randomization
                self.assertIsNotNone(state)
                assert state is not None

                expected_keys = {f"mug_{index}_jnt" for index in range(1, len(mug_variants) + 1)}
                self.assertEqual(set(state.scale_states), expected_keys)
                self.assertTrue(any(abs(scale - 1.0) > 1e-6 for scale in state.scale_states.values()))
                for scale in state.scale_states.values():
                    self.assertGreaterEqual(scale, 0.90)
                    self.assertLessEqual(scale, 1.10)
            finally:
                env.close()

    def test_mug_flip_clamps_sampled_mug_bounds_inside_tray(self) -> None:
        env = xdof_sim.make_env(task="mug_flip", render_cameras=False)
        try:
            env.reset(
                seed=2,
                randomize=True,
                options={
                    "randomization": {
                        "mug_variants": ["mug_0", "mug_0", "mug_0", "mug_0"],
                        "randomize_scales": False,
                    }
                },
            )
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.metadata["mug_count"], 4)

            tray_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "tray")
            self.assertGreaterEqual(tray_body_id, 0)
            for index in range(1, 5):
                mug_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, f"mug_{index}")
                self.assertGreaterEqual(mug_body_id, 0)
                lower, upper = _subtree_collision_bounds_in_body_frame(
                    env.model,
                    env.data,
                    root_body_id=mug_body_id,
                    frame_body_id=tray_body_id,
                )
                self.assertGreaterEqual(lower[0], -_MUG_FLIP_TRAY_INNER_HALF_XY[0] - 1e-6)
                self.assertLessEqual(upper[0], _MUG_FLIP_TRAY_INNER_HALF_XY[0] + 1e-6)
                self.assertGreaterEqual(lower[1], -_MUG_FLIP_TRAY_INNER_HALF_XY[1] - 1e-6)
                self.assertLessEqual(upper[1], _MUG_FLIP_TRAY_INNER_HALF_XY[1] + 1e-6)
                self.assertGreaterEqual(
                    lower[2],
                    _MUG_FLIP_TRAY_FLOOR_Z_OFFSET + _MUG_FLIP_SPAWN_CLEARANCE_M - 1e-6,
                )
        finally:
            env.close()

    def test_mug_flip_reduces_large_mug_count_instead_of_accepting_overlap(self) -> None:
        env = xdof_sim.make_env(task="mug_flip", render_cameras=False)
        try:
            env.reset(
                seed=0,
                randomize=True,
                options={
                    "randomization": {
                        "mug_variants": ["mug_4", "mug_4", "mug_4", "mug_4"],
                        "randomize_scales": False,
                    }
                },
            )
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.metadata["requested_mug_count"], 4)
            self.assertTrue(state.metadata["mug_count_reduced"])
            self.assertLess(state.metadata["mug_count"], 4)
            self.assertEqual(
                {name for name in state.object_states if name.startswith("mug_")},
                {f"mug_{index}_jnt" for index in range(1, int(state.metadata["mug_count"]) + 1)},
            )
            self.assertFalse(_has_mug_mug_contact(env.model, env.data))
        finally:
            env.close()

    def test_mug_flip_keeps_four_mugs_when_the_sampled_assets_fit(self) -> None:
        env = xdof_sim.make_env(task="mug_flip", render_cameras=False)
        try:
            env.reset(
                seed=0,
                randomize=True,
                options={
                    "randomization": {
                        "mug_variants": ["mug_0", "mug_0", "mug_0", "mug_0"],
                        "randomize_scales": False,
                    }
                },
            )
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.metadata["mug_count"], 4)
            self.assertNotIn("mug_count_reduced", state.metadata)
            self.assertFalse(_has_mug_mug_contact(env.model, env.data))
        finally:
            env.close()

    def test_mug_flip_randomizes_tray_color_and_replays_it(self) -> None:
        env = xdof_sim.make_env(task="mug_flip", render_cameras=False)
        try:
            env.reset(
                seed=0,
                randomize=True,
                options={
                    "randomization": {
                        "mug_variants": ["mug_0"],
                        "randomize_scales": False,
                    }
                },
            )
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None

            tray_color = np.asarray(state.metadata["tray_color"], dtype=np.float64)
            self.assertEqual(tray_color.shape, (4,))
            mat_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_MATERIAL, "tray_blue")
            self.assertGreaterEqual(mat_id, 0)
            np.testing.assert_allclose(env.model.mat_rgba[mat_id], tray_color)

            env.model.mat_rgba[mat_id] = np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
            env._task_randomizer.apply(env.model, env.data, state)
            mat_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_MATERIAL, "tray_blue")
            np.testing.assert_allclose(env.model.mat_rgba[mat_id], tray_color)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
