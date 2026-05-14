from __future__ import annotations

from itertools import product
from pathlib import Path
import unittest

import mujoco
import numpy as np

import xdof_sim
from xdof_sim.randomization import (
    ChessRandomizer,
    DishRackRandomizer,
    SweepRandomizer,
    WaterBottleRandomizer,
    _CHESS_PIECE_JOINTS,
    _build_chess_tin_scene_xml,
    _build_dishrack_scene_xml,
    _build_mug_scene_xml,
    _build_water_bottle_scene_xml,
    _chess_tin_variant_names,
    _dishrack_geom_world_bounds,
    _dishrack_plate_body_name,
    _dishrack_plate_joint_name,
    _dishrack_variant_names,
    _MUG_FLIP_SPAWN_CLEARANCE_M,
    _MUG_FLIP_TRAY_FLOOR_Z_OFFSET,
    _MUG_FLIP_TRAY_INNER_HALF_XY,
    _mug_plain_color_material_names,
    _mug_variant_names,
    _water_bottle_variant_names,
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


def _has_bottle_bottle_contact(model: mujoco.MjModel, data: mujoco.MjData, bottle_count: int) -> bool:
    bottle_roots: set[int] = set()
    for index in range(1, bottle_count + 1):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bottle_{index}")
        if body_id >= 0:
            bottle_roots.add(int(model.body_rootid[body_id]))

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        root_1 = int(model.body_rootid[int(model.geom_bodyid[contact.geom1])])
        root_2 = int(model.body_rootid[int(model.geom_bodyid[contact.geom2])])
        if root_1 != root_2 and root_1 in bottle_roots and root_2 in bottle_roots:
            return True
    return False


def _has_bottle_bin_contact(model: mujoco.MjModel, data: mujoco.MjData, bottle_count: int) -> bool:
    bin_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin_container")
    if bin_body_id < 0:
        return False
    bin_root = int(model.body_rootid[bin_body_id])

    bottle_roots: set[int] = set()
    for index in range(1, bottle_count + 1):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bottle_{index}")
        if body_id >= 0:
            bottle_roots.add(int(model.body_rootid[body_id]))

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        root_1 = int(model.body_rootid[int(model.geom_bodyid[contact.geom1])])
        root_2 = int(model.body_rootid[int(model.geom_bodyid[contact.geom2])])
        if (
            root_1 in bottle_roots
            and root_2 == bin_root
        ) or (
            root_2 in bottle_roots
            and root_1 == bin_root
        ):
            return True
    return False


def _rotated_local_z_axis(quat: list[float]) -> np.ndarray:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return np.array(
        [
            2.0 * (x * z + w * y),
            2.0 * (y * z - w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
        dtype=np.float64,
    )


def _quat_dot_abs(quat_a: list[float], quat_b: list[float]) -> float:
    return abs(float(np.dot(np.asarray(quat_a, dtype=np.float64), np.asarray(quat_b, dtype=np.float64))))


def _yaw_from_quat(quat: list[float] | np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _rotate_xy(vec: np.ndarray, yaw: float) -> np.ndarray:
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=np.float64)


def _inside_chess_board(state: dict[str, dict[str, list[float]]], xy: list[float], margin: float = 0.025) -> bool:
    board = state["chessboard"]
    center = np.asarray(board["pos"][:2], dtype=np.float64)
    local = _rotate_xy(np.asarray(xy, dtype=np.float64) - center, -_yaw_from_quat(board["quat"]))
    half = ChessRandomizer._board_half + margin
    return abs(float(local[0])) < half and abs(float(local[1])) < half


def _inside_chess_tin(
    state: dict[str, dict[str, list[float]]],
    xy: list[float],
    margin: float = 0.0,
    inner_half_xy: list[float] | tuple[float, float] | None = None,
) -> bool:
    tin = state["tin_box_joint"]
    center = np.asarray(tin["pos"][:2], dtype=np.float64)
    local = _rotate_xy(np.asarray(xy, dtype=np.float64) - center, -_yaw_from_quat(tin["quat"]))
    half_xy = inner_half_xy or ChessRandomizer._tin_inner_half_xy
    return (
        abs(float(local[0])) <= float(half_xy[0]) + margin
        and abs(float(local[1])) <= float(half_xy[1]) + margin
    )


def _has_chess_piece_piece_contact(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    piece_roots: set[int] = set()
    for joint_name in _CHESS_PIECE_JOINTS:
        jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jnt_id >= 0:
            piece_roots.add(int(model.body_rootid[int(model.jnt_bodyid[jnt_id])]))

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        root_1 = int(model.body_rootid[int(model.geom_bodyid[contact.geom1])])
        root_2 = int(model.body_rootid[int(model.geom_bodyid[contact.geom2])])
        if root_1 != root_2 and root_1 in piece_roots and root_2 in piece_roots:
            return True
    return False


def _chess_tin_collision_flags(model: mujoco.MjModel) -> list[tuple[int, int]]:
    flags: list[tuple[int, int]] = []
    for geom_name in (
        "tin_box_floor",
        "tin_box_wall_y_pos",
        "tin_box_wall_y_neg",
        "tin_box_wall_x_pos",
        "tin_box_wall_x_neg",
    ):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
        if geom_id >= 0:
            flags.append((int(model.geom_contype[geom_id]), int(model.geom_conaffinity[geom_id])))
    return flags


class ChessSetupRandomizationTests(unittest.TestCase):
    def _assert_chess_steps_stable(self, env, *, steps: int = 30) -> None:
        for _ in range(steps):
            mujoco.mj_step(env.model, env.data)
            self.assertTrue(np.isfinite(env.data.qpos).all())
            self.assertTrue(np.isfinite(env.data.qvel).all())
            self.assertTrue(np.isfinite(env.data.qacc).all())
            self.assertLess(float(np.nanmax(np.abs(env.data.qacc))), 1e8)

    def _reset_chess(
        self,
        *,
        scenario: str,
        target_count: int,
        color_mode: str = "mixed_partial",
        seed: int = 3,
    ):
        env = xdof_sim.make_env(task="chess", render_cameras=False)
        env.reset(
            seed=seed,
            randomize=True,
            options={
                "randomization": {
                    "scenario": scenario,
                    "target_count": target_count,
                    "color_mode": color_mode,
                    "randomize_scales": False,
                }
            },
        )
        state = env._last_randomization
        self.assertIsNotNone(state)
        assert state is not None
        return env, state

    def test_chess_table_setup_places_targets_off_board_and_records_metadata(self) -> None:
        env, state = self._reset_chess(
            scenario="table_setup",
            target_count=12,
            color_mode="white_only",
            seed=10,
        )
        try:
            metadata = state.metadata
            self.assertEqual(metadata["scenario"], "table_setup")
            self.assertEqual(metadata["target_count"], 12)
            self.assertEqual(metadata["target_colors"], ["white"])
            self.assertEqual(len(metadata["target_poses"]), 12)
            self.assertEqual(len(metadata["board_target_poses"]), len(_CHESS_PIECE_JOINTS))
            self.assertFalse(metadata["tin_active"])
            self.assertNotIn("tin_pose", metadata)
            self.assertIn("tin_box_joint", state.object_states)
            self.assertLess(state.object_states["tin_box_joint"]["pos"][2], -1.0)
            self.assertEqual(set(_chess_tin_collision_flags(env.model)), {(0, 0)})

            board_joint_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "chessboard")
            self.assertGreaterEqual(board_joint_id, 0)
            self.assertEqual(env.model.jnt_type[board_joint_id], mujoco.mjtJoint.mjJNT_FREE)

            target_joints = set(metadata["target_joints"])
            staged_abs_y = []
            for joint_name in target_joints:
                piece_state = state.object_states[joint_name]
                staged_abs_y.append(abs(piece_state["pos"][1]))
                self.assertGreater(abs(piece_state["pos"][1]), 0.28)
                self.assertFalse(
                    _inside_chess_board(
                        state.object_states,
                        piece_state["pos"][:2],
                        margin=ChessRandomizer._table_stage_board_margin,
                    )
                )
            self.assertLess(min(staged_abs_y), 0.42)

            for joint_name in metadata["non_target_pieces"]:
                piece_state = state.object_states[joint_name]
                board_target_pose = metadata["board_target_poses"][joint_name]
                distance = np.linalg.norm(
                    np.asarray(piece_state["pos"][:2], dtype=np.float64)
                    - np.asarray(board_target_pose["pos"][:2], dtype=np.float64)
                )
                self.assertLess(distance, 1e-9)
                self.assertTrue(_inside_chess_board(state.object_states, piece_state["pos"][:2], margin=0.05))

            self.assertFalse(_has_chess_piece_piece_contact(env.model, env.data))
            self._assert_chess_steps_stable(env)
        finally:
            env.close()

    def test_chess_knocked_setup_places_targets_near_board_squares_non_upright(self) -> None:
        env, state = self._reset_chess(
            scenario="knocked_setup",
            target_count=8,
            color_mode="black_only",
            seed=11,
        )
        try:
            metadata = state.metadata
            self.assertEqual(metadata["scenario"], "knocked_setup")
            self.assertEqual(metadata["target_count"], 8)
            self.assertEqual(metadata["target_colors"], ["black"])
            self.assertFalse(metadata["tin_active"])
            self.assertNotIn("tin_pose", metadata)
            self.assertLess(state.object_states["tin_box_joint"]["pos"][2], -1.0)
            self.assertEqual(set(_chess_tin_collision_flags(env.model)), {(0, 0)})

            board_center_xy = np.mean(
                [
                    np.asarray(pose["pos"][:2], dtype=np.float64)
                    for pose in metadata["board_target_poses"].values()
                ],
                axis=0,
            )
            for joint_name in metadata["target_joints"]:
                piece_state = state.object_states[joint_name]
                target_pose = metadata["target_poses"][joint_name]
                distance_to_target = np.linalg.norm(
                    np.asarray(piece_state["pos"][:2], dtype=np.float64)
                    - np.asarray(target_pose["pos"][:2], dtype=np.float64)
                )
                distance_to_center = np.linalg.norm(
                    np.asarray(piece_state["pos"][:2], dtype=np.float64) - board_center_xy
                )
                self.assertGreater(distance_to_target, 0.035)
                self.assertLess(distance_to_center, 0.18)
                self.assertLess(_quat_dot_abs(piece_state["quat"], target_pose["quat"]), 0.97)
                self.assertTrue(_inside_chess_board(state.object_states, piece_state["pos"][:2], margin=0.05))

            self._assert_chess_steps_stable(env)
        finally:
            env.close()

    def test_chess_tin_setup_places_targets_inside_free_jointed_tin(self) -> None:
        env, state = self._reset_chess(
            scenario="tin_setup",
            target_count=16,
            color_mode="both_broad",
            seed=12,
        )
        try:
            metadata = state.metadata
            self.assertEqual(metadata["scenario"], "tin_setup")
            self.assertEqual(metadata["target_count"], 16)
            self.assertEqual(set(metadata["target_colors"]), {"black", "white"})
            self.assertTrue(metadata["tin_active"])
            self.assertIn("tin_pose", metadata)
            self.assertEqual(_chess_tin_variant_names(), ["tin_2"])
            self.assertEqual(metadata["tin_variant"], "tin_2")
            self.assertGreater(metadata["tin_scale"], 1.0)
            self.assertIn("tin_outer_half_xy", metadata)
            self.assertEqual(set(state.scale_states), set())
            self.assertEqual(set(_chess_tin_collision_flags(env.model)), {(1, 1)})

            tin_z = state.object_states["tin_box_joint"]["pos"][2]
            self.assertGreater(tin_z - ChessRandomizer._tin_floor_half_z, 0.75)
            for joint_name in metadata["target_joints"]:
                piece_state = state.object_states[joint_name]
                self.assertTrue(
                    _inside_chess_tin(
                        state.object_states,
                        piece_state["pos"][:2],
                        margin=0.01,
                        inner_half_xy=metadata["tin_inner_half_xy"],
                    )
                )
                self.assertGreater(piece_state["pos"][2], metadata["tin_floor_z"])

            self.assertFalse(_has_chess_piece_piece_contact(env.model, env.data))
        finally:
            env.close()

    def test_chess_tin_scene_builder_uses_tin_2_body_variant(self) -> None:
        self.assertEqual(_chess_tin_variant_names(), ["tin_2"])
        for variant in _chess_tin_variant_names():
            xml = _build_chess_tin_scene_xml(
                tin_variant=variant,
                scale_states={},
                base_scene_xml=None,
                base_scene_dir=None,
            )
            self.assertIn(f"/{variant}/body/", xml)
            model = mujoco.MjModel.from_xml_string(xml)
            tin_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tin_box")
            tin_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tin_box_joint")
            self.assertGreaterEqual(tin_body, 0)
            self.assertGreaterEqual(tin_joint, 0)
            mesh_names = [model.mesh(index).name for index in range(model.nmesh)]
            self.assertTrue(
                any(name.startswith(f"chess_tin_{variant}") for name in mesh_names),
                msg=f"missing prefixed mesh for {variant}",
            )

    def test_chess_scale_randomization_is_off_by_default(self) -> None:
        env = xdof_sim.make_env(task="chess", render_cameras=False)
        try:
            env.reset(
                seed=13,
                randomize=True,
                options={
                    "randomization": {
                        "scenario": "table_setup",
                        "target_count": 6,
                        "color_mode": "mixed_partial",
                    }
                },
            )
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.scale_states, {})
        finally:
            env.close()

    def test_chess_scale_randomization_can_be_enabled_explicitly(self) -> None:
        env = xdof_sim.make_env(task="chess", render_cameras=False)
        try:
            env.reset(
                seed=14,
                randomize=True,
                options={
                    "randomization": {
                        "scenario": "table_setup",
                        "target_count": 6,
                        "color_mode": "mixed_partial",
                        "randomize_scales": True,
                    }
                },
            )
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(set(state.scale_states), set(_CHESS_PIECE_JOINTS))
            self.assertTrue(any(abs(scale - 1.0) > 1e-6 for scale in state.scale_states.values()))
        finally:
            env.close()

    def test_chess_asset_debug_cycles_scenario_color_mode_and_tin(self) -> None:
        env = xdof_sim.make_env(task="chess", render_cameras=False)
        try:
            debug_options = {
                "randomization": {
                    "randomize_variants": False,
                    "randomize_scales": False,
                }
            }
            env.reset(seed=20, randomize=True, options=debug_options)
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.metadata["scenario"], "table_setup")
            self.assertEqual(state.metadata["color_mode"], "white_only")
            self.assertEqual(state.metadata["target_count"], ChessRandomizer.target_count_range[0])
            self.assertEqual(state.metadata["tin_variant"], "tin_2")
            self.assertEqual(state.scale_states, {})

            env.reset(
                seed=21,
                randomize=True,
                options={
                    "randomization": {
                        "randomize_variants": False,
                        "randomize_scales": False,
                        "cycle_scenario": 1,
                    }
                },
            )
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.metadata["scenario"], "knocked_setup")
            self.assertEqual(state.metadata["color_mode"], "white_only")
            self.assertEqual(state.metadata["target_count"], ChessRandomizer.target_count_range[0])
            self.assertEqual(state.metadata["tin_variant"], "tin_2")

            env.reset(
                seed=22,
                randomize=True,
                options={
                    "randomization": {
                        "randomize_variants": False,
                        "randomize_scales": False,
                        "cycle_color_mode": 1,
                    }
                },
            )
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.metadata["scenario"], "knocked_setup")
            self.assertEqual(state.metadata["color_mode"], "black_only")
            self.assertEqual(state.metadata["target_count"], ChessRandomizer.target_count_range[0])
            self.assertEqual(state.metadata["tin_variant"], "tin_2")
        finally:
            env.close()

    def test_chess_reuses_model_when_asset_selection_is_unchanged(self) -> None:
        env = xdof_sim.make_env(task="chess", render_cameras=False)
        try:
            reset_options = {
                "randomization": {
                    "randomize_variants": False,
                    "randomize_scales": False,
                }
            }
            initial_model = env.model
            env.reset(seed=30, randomize=True, options=reset_options)
            self.assertIs(env.model, initial_model)
            env.reset(seed=31, randomize=True, options=reset_options)
            self.assertIs(env.model, initial_model)
            env.reset(
                seed=32,
                randomize=True,
                options={
                    "randomization": {
                        "randomize_variants": False,
                        "randomize_scales": False,
                        "cycle_scenario": 1,
                    }
                },
            )
            self.assertIs(env.model, initial_model)
        finally:
            env.close()

    def test_chess2_keeps_legacy_scatter_randomizer(self) -> None:
        env = xdof_sim.make_env(task="chess2", render_cameras=False)
        try:
            env.reset(seed=4, randomize=True)
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.metadata, {})
            self.assertNotIn("tin_box_joint", state.object_states)
        finally:
            env.close()


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

    def test_dishrack_scene_reload_preserves_mocap_transform(self) -> None:
        from xdof_sim.scene_xml import SceneXmlTransformOptions, build_scene_xml
        from xdof_sim.task_registry import get_task_scene_xml

        transform_options = SceneXmlTransformOptions(clean=True, mocap=True)
        xml, _ = build_scene_xml(get_task_scene_xml("dishrack"), options=transform_options)
        env = xdof_sim.make_env(
            task="dishrack",
            scene="hybrid",
            render_cameras=False,
            scene_xml_string=xml,
            scene_xml_transform_options=transform_options,
        )
        try:
            env.reset(seed=0, randomize=True)

            self.assertEqual(env.model.nmocap, 2)
            self.assertGreaterEqual(
                mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "left_mocap_weld"),
                0,
            )
            self.assertGreaterEqual(
                mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_EQUALITY, "right_mocap_weld"),
                0,
            )
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
        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dishrack"), 0)
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
        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dishrack"), 0)
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
        self.assertGreaterEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "dishrack"), 0)
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
            self.assertGreaterEqual(mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, "dishrack"), 0)
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


class WaterBottleVariantRandomizationTests(unittest.TestCase):
    def test_water_bottle_variant_names_include_combined_robocasa_assets(self) -> None:
        variants = _water_bottle_variant_names()
        self.assertEqual(variants[0], "bottle_0")
        self.assertEqual(len(variants), 19)
        self.assertIn("bottle_18", variants)

    def test_build_water_bottle_scene_xml_supports_variable_bottle_count(self) -> None:
        variants = _water_bottle_variant_names()
        for bottle_count in (2, 6):
            bottle_variants = variants[:bottle_count]
            xml = _build_water_bottle_scene_xml(
                bottle_variants=bottle_variants,
                scale_states={
                    f"bottle_{index}_joint": 1.0
                    for index in range(1, bottle_count + 1)
                },
                base_scene_xml=None,
                base_scene_dir=None,
            )
            model = mujoco.MjModel.from_xml_string(xml)

            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bin_container"),
                0,
            )
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "bin_joint"),
                0,
            )
            for index in range(1, bottle_count + 1):
                self.assertGreaterEqual(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bottle_{index}"),
                    0,
                )
                self.assertGreaterEqual(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"bottle_{index}_joint"),
                    0,
                )

            self.assertLess(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"bottle_{bottle_count + 1}"),
                0,
            )

    def test_water_bottle_variants_have_consistent_mass_and_contact_params(self) -> None:
        for variant_name in _water_bottle_variant_names():
            xml = _build_water_bottle_scene_xml(
                bottle_variants=[variant_name, variant_name],
                scale_states={},
                base_scene_xml=None,
                base_scene_dir=None,
            )
            model = mujoco.MjModel.from_xml_string(xml)
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)

            bottle_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "bottle_1")
            nonzero_body_ids = [
                body_id
                for body_id in range(model.nbody)
                if _body_in_subtree(model, body_id, bottle_body_id)
                and float(model.body_mass[body_id]) > 0.0
            ]
            self.assertEqual(len(nonzero_body_ids), 1, msg=variant_name)
            body_id = nonzero_body_ids[0]
            self.assertAlmostEqual(float(model.body_mass[body_id]), 0.05, places=6, msg=variant_name)
            self.assertGreater(float(model.body_inertia[body_id][0]), 9.0e-5, msg=variant_name)
            self.assertGreater(float(model.body_inertia[body_id][1]), 9.0e-5, msg=variant_name)
            self.assertGreater(float(model.body_inertia[body_id][2]), 1.3e-5, msg=variant_name)

            collision_geom_ids = [
                geom_id
                for geom_id in range(model.ngeom)
                if _body_in_subtree(model, int(model.geom_bodyid[geom_id]), bottle_body_id)
                and int(model.geom_group[geom_id]) == 3
            ]
            self.assertGreater(len(collision_geom_ids), 0, msg=variant_name)
            self.assertEqual(
                {
                    tuple(float(value) for value in model.geom_friction[geom_id])
                    for geom_id in collision_geom_ids
                },
                {(3.0, 0.03, 0.003)},
                msg=variant_name,
            )
            self.assertEqual(
                {int(model.geom_condim[geom_id]) for geom_id in collision_geom_ids},
                {6},
                msg=variant_name,
            )
            self.assertEqual(
                {int(model.geom_priority[geom_id]) for geom_id in collision_geom_ids},
                {1},
                msg=variant_name,
            )

    def test_make_env_randomizes_water_bottle_count_variants_scales_and_contacts(self) -> None:
        env = xdof_sim.make_env(task="put_bottles", render_cameras=False)
        try:
            observed_counts: set[int] = set()
            observed_independent_sample = False
            observed_bin_positions: set[tuple[float, float]] = set()
            observed_bin_colors: set[tuple[float, float, float, float]] = set()
            observed_bin_scales: set[float] = set()
            for seed in range(6):
                env.reset(seed=seed, randomize=True)
                state = env._last_randomization
                self.assertIsNotNone(state)
                assert state is not None

                bottle_count = int(state.metadata["bottle_count"])
                bottle_variants = list(state.metadata["bottle_variants"])
                observed_counts.add(bottle_count)
                observed_independent_sample = observed_independent_sample or len(set(bottle_variants)) > 1

                self.assertGreaterEqual(bottle_count, WaterBottleRandomizer.min_bottle_count)
                self.assertLessEqual(bottle_count, WaterBottleRandomizer.max_bottle_count)
                self.assertEqual(len(bottle_variants), bottle_count)
                self.assertEqual(
                    set(state.object_states),
                    {
                        *(f"bottle_{index}_joint" for index in range(1, bottle_count + 1)),
                        "bin_joint",
                    },
                )
                self.assertEqual(set(state.scale_states), set(state.object_states))
                for joint_name, scale in state.scale_states.items():
                    if joint_name == "bin_joint":
                        self.assertGreaterEqual(scale, 1.00)
                        self.assertLessEqual(scale, 1.40)
                        continue
                    self.assertGreaterEqual(scale, 0.90)
                    self.assertLessEqual(scale, 1.10)
                self.assertIn("bin_color", state.metadata)
                self.assertEqual(len(state.metadata["bin_color"]), 4)
                self.assertGreater(float(state.object_states["bin_joint"]["pos"][2]), 0.75)
                observed_bin_positions.add(
                    tuple(round(float(value), 3) for value in state.object_states["bin_joint"]["pos"][:2])
                )
                observed_bin_colors.add(tuple(float(value) for value in state.metadata["bin_color"]))
                observed_bin_scales.add(round(float(state.scale_states["bin_joint"]), 3))
                for joint_name, object_state in state.object_states.items():
                    if joint_name == "bin_joint":
                        continue
                    long_axis = _rotated_local_z_axis(object_state["quat"])
                    self.assertLess(abs(float(long_axis[2])), 1.0e-6)
                    self.assertGreater(float(object_state["pos"][2]), 0.75)

                self.assertFalse(_has_bottle_bottle_contact(env.model, env.data, bottle_count))
                self.assertFalse(_has_bottle_bin_contact(env.model, env.data, bottle_count))

            self.assertGreaterEqual(len(observed_counts), 2)
            self.assertTrue(observed_independent_sample)
            self.assertGreaterEqual(len(observed_bin_positions), 2)
            self.assertGreaterEqual(len(observed_bin_colors), 2)
            self.assertGreaterEqual(len(observed_bin_scales), 2)
        finally:
            env.close()

    def test_water_bottle_asset_debug_cycles_pinned_bottle_variant(self) -> None:
        env = xdof_sim.make_env(task="put_bottles", render_cameras=False)
        try:
            options = {
                "randomization": {
                    "randomize_variants": False,
                    "randomize_scales": False,
                }
            }
            env.reset(seed=1, randomize=True, options=options)
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.metadata["bottle_variants"], ["bottle_0", "bottle_0"])
            self.assertEqual(state.scale_states, {})

            env.reset(
                seed=2,
                randomize=True,
                options={
                    "randomization": {
                        "randomize_variants": False,
                        "randomize_scales": False,
                        "cycle_bottle": 1,
                    }
                },
            )
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.metadata["bottle_variants"], ["bottle_1", "bottle_1"])
            self.assertEqual(state.scale_states, {})

            env.reset(
                seed=3,
                randomize=True,
                options={
                    "randomization": {
                        "randomize_variants": False,
                        "randomize_scales": False,
                        "cycle_bottle": -1,
                    }
                },
            )
            state = env._last_randomization
            self.assertIsNotNone(state)
            assert state is not None
            self.assertEqual(state.metadata["bottle_variants"], ["bottle_0", "bottle_0"])
            self.assertEqual(state.scale_states, {})
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
