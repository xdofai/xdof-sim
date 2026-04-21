from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco

from xdof_sim.scene_xml import SceneXmlTransformOptions, build_scene_xml, transform_scene_xml
from xdof_sim.randomization import (
    RandomizationState,
    _apply_object_scales_to_scene_xml,
    _inhand_build_xml,
)


class SceneXmlTests(unittest.TestCase):
    def _models_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "xdof_sim" / "models"

    def _parse_xml(self, name: str) -> ET.Element:
        return ET.fromstring((self._models_dir() / name).read_text())

    def _find_body(self, root: ET.Element, name: str) -> ET.Element:
        for body in root.iter("body"):
            if body.get("name") == name:
                return body
        raise AssertionError(f"Missing body {name}")

    def _find_default(self, root: ET.Element, cls: str) -> ET.Element:
        for default in root.iter("default"):
            if default.get("class") == cls:
                return default
        raise AssertionError(f"Missing default {cls}")

    def _canonical_xml(self, elem: ET.Element) -> str:
        elem = deepcopy(elem)

        def normalize(node: ET.Element) -> None:
            node.text = None
            node.tail = None
            node.attrib = dict(sorted(node.attrib.items()))
            for child in node:
                normalize(child)

        normalize(elem)
        return ET.tostring(elem, encoding="unicode")

    def test_clean_transform_removes_visual_clutter(self) -> None:
        scene_path = self._models_dir() / "yam_bottles_scene.xml"
        xml, edits = build_scene_xml(
            scene_path,
            options=SceneXmlTransformOptions(clean=True),
        )
        self.assertEqual(edits, ("clean",))
        self.assertNotIn('name="floor"', xml)
        self.assertNotIn('name="back_wall"', xml)
        self.assertNotIn('name="top_camera_d405"', xml)
        self.assertIn('name="floor_collision"', xml)
        self.assertIn('name="back_wall_collision"', xml)
        self.assertIn('name="gate_collision"', xml)

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            dir=scene_path.parent,
            delete=False,
        ) as handle:
            handle.write(xml)
            temp_scene = Path(handle.name)
        try:
            model = mujoco.MjModel.from_xml_path(str(temp_scene))
            self.assertGreater(model.ngeom, 0)
            geom_names = [model.geom(i).name for i in range(model.ngeom)]
            self.assertIn("floor_collision", geom_names)
            self.assertIn("back_wall_collision", geom_names)
            self.assertNotIn("floor", geom_names)
            self.assertNotIn("back_wall", geom_names)
        finally:
            temp_scene.unlink(missing_ok=True)

    def test_mocap_transform_adds_targets_and_welds(self) -> None:
        scene_path = self._models_dir() / "yam_bimanual_empty.xml"
        xml, edits = build_scene_xml(
            scene_path,
            options=SceneXmlTransformOptions(mocap=True, debug=True),
        )
        self.assertEqual(edits, ("mocap",))
        self.assertIn('name="left_mocap"', xml)
        self.assertIn('name="right_mocap_weld"', xml)
        self.assertIn('rgba="0.2 0.9 0.2 0.3"', xml)

    def test_flexible_gripper_transform_swaps_flexible_bodies(self) -> None:
        scene_path = self._models_dir() / "yam_chess_scene.xml"
        xml, edits = build_scene_xml(
            scene_path,
            options=SceneXmlTransformOptions(flexible_gripper=True),
        )
        self.assertEqual(edits, ("flexible_gripper",))
        self.assertIn('name="left_flex_gripper"', xml)
        self.assertIn('name="right_flex_gripper"', xml)
        self.assertIn('name="flexible_base"', xml)

    def test_inhand_transfer_base_xml_loads_standalone(self) -> None:
        scene_path = self._models_dir() / "yam_inhand_transfer_base.xml"
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.assertGreater(model.nbody, 0)

    def test_inhand_transfer_generated_xml_loads_from_string(self) -> None:
        mesh_path = self._models_dir() / "assets" / "i2rt_yam" / "assets" / "base_visual_gate.stl"
        with tempfile.TemporaryDirectory() as tmp_dir:
            variant_dir = Path(tmp_dir)
            (variant_dir / "model.xml").write_text(
                "\n".join(
                    (
                        '<mujoco model="test_object">',
                        "  <asset>",
                        f'    <mesh name="proxy_mesh" file="{mesh_path}"/>',
                        "  </asset>",
                        "  <worldbody>",
                        '    <geom class="visual" mesh="proxy_mesh"/>',
                        '    <geom class="collision" mesh="proxy_mesh"/>',
                        "  </worldbody>",
                        "</mujoco>",
                    )
                )
            )
            xml = _inhand_build_xml("dish_brush", variant_dir, x=0.6, y=0.2, z=0.8, yaw=0.0)

        self.assertNotIn("TASK_ASSETS_PLACEHOLDER", xml)
        self.assertNotIn("TASK_BODY_PLACEHOLDER", xml)

        model = mujoco.MjModel.from_xml_string(xml)
        self.assertGreater(model.nbody, 0)

    def test_inhand_transfer_generated_xml_supports_vr_scene_transforms(self) -> None:
        mesh_path = self._models_dir() / "assets" / "i2rt_yam" / "assets" / "base_visual_gate.stl"
        with tempfile.TemporaryDirectory() as tmp_dir:
            variant_dir = Path(tmp_dir)
            (variant_dir / "model.xml").write_text(
                "\n".join(
                    (
                        '<mujoco model="test_object">',
                        "  <asset>",
                        f'    <mesh name="proxy_mesh" file="{mesh_path}"/>',
                        "  </asset>",
                        "  <worldbody>",
                        '    <geom class="visual" mesh="proxy_mesh"/>',
                        '    <geom class="collision" mesh="proxy_mesh"/>',
                        "  </worldbody>",
                        "</mujoco>",
                    )
                )
            )
            xml = _inhand_build_xml("dish_brush", variant_dir, x=0.6, y=0.2, z=0.8, yaw=0.0)

        transformed_xml, edits = transform_scene_xml(
            xml,
            options=SceneXmlTransformOptions(clean=True, mocap=True, flexible_gripper=True, debug=True),
        )

        self.assertEqual(edits, ("flexible_gripper", "clean", "mocap"))
        self.assertIn('name="left_mocap"', transformed_xml)
        self.assertIn('name="left_mocap_weld"', transformed_xml)
        self.assertIn('name="left_flex_gripper"', transformed_xml)
        self.assertNotIn('name="floor"', transformed_xml)
        self.assertNotIn('name="back_wall"', transformed_xml)

        model = mujoco.MjModel.from_xml_string(transformed_xml)
        self.assertGreater(model.nbody, 0)

    def test_object_scale_transform_only_changes_target_body(self) -> None:
        scene_path = self._models_dir() / "yam_blocks_scene.xml"
        scene_xml = scene_path.read_text()
        scaled_xml = _apply_object_scales_to_scene_xml(scene_xml, {"block_A_jnt": 1.05})
        root = ET.fromstring(scaled_xml)

        block_a = self._find_body(root, "block_A")
        block_b = self._find_body(root, "block_B")
        block_a_vis = next(geom for geom in block_a.findall("geom") if geom.get("name") == "block_A_vis")
        block_a_col = next(geom for geom in block_a.findall("geom") if geom.get("name") == "block_A_col")
        block_b_vis = next(geom for geom in block_b.findall("geom") if geom.get("name") == "block_B_vis")

        self.assertEqual(block_a_col.get("size"), "0.01575 0.01575 0.01575")
        self.assertEqual(block_b_vis.get("mesh"), "block_B")
        self.assertNotEqual(block_a_vis.get("mesh"), "block_A")

        mesh_lookup = {
            mesh.get("name"): mesh
            for mesh in root.find("asset").findall("mesh")
            if mesh.get("name")
        }
        self.assertEqual(mesh_lookup["block_A"].get("scale"), "0.15 0.15 0.15")
        self.assertEqual(mesh_lookup[block_a_vis.get("mesh")].get("scale"), "0.1575 0.1575 0.1575")

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            dir=scene_path.parent,
            delete=False,
        ) as handle:
            handle.write(scaled_xml)
            temp_scene = Path(handle.name)
        try:
            model = mujoco.MjModel.from_xml_path(str(temp_scene))
            self.assertGreater(model.ngeom, 0)
        finally:
            temp_scene.unlink(missing_ok=True)

    def test_randomization_state_serializes_scale_and_metadata(self) -> None:
        state = RandomizationState(
            seed=7,
            object_states={"block_A_jnt": {"pos": [1.0, 2.0, 3.0], "quat": [1.0, 0.0, 0.0, 0.0]}},
            scale_states={"block_A_jnt": 1.03},
            metadata={"variant": "demo"},
        )
        restored = RandomizationState.from_dict(state.to_dict())
        self.assertEqual(restored.seed, 7)
        self.assertEqual(restored.scale_states, {"block_A_jnt": 1.03})
        self.assertEqual(restored.metadata, {"variant": "demo"})

    def test_inhand_transfer_generated_xml_supports_scale_factor(self) -> None:
        mesh_path = self._models_dir() / "assets" / "i2rt_yam" / "assets" / "base_visual_gate.stl"
        with tempfile.TemporaryDirectory() as tmp_dir:
            variant_dir = Path(tmp_dir)
            (variant_dir / "model.xml").write_text(
                "\n".join(
                    (
                        '<mujoco model="test_object">',
                        "  <asset>",
                        f'    <mesh name="proxy_mesh" file="{mesh_path}" scale="2 3 4"/>',
                        "  </asset>",
                        "  <worldbody>",
                        '    <geom class="visual" mesh="proxy_mesh"/>',
                        '    <geom class="collision" mesh="proxy_mesh"/>',
                        "  </worldbody>",
                        "</mujoco>",
                    )
                )
            )
            xml = _inhand_build_xml(
                "dish_brush",
                variant_dir,
                x=0.6,
                y=0.2,
                z=0.8,
                yaw=0.0,
                scale_factor=1.1,
            )

        self.assertIn('scale="2.2 3.3 4.4"', xml)
        model = mujoco.MjModel.from_xml_string(xml)
        self.assertGreater(model.nbody, 0)

    def test_inhand_drawer_and_pour_share_standard_arm_variant(self) -> None:
        reference = self._parse_xml("yam_chess_scene.xml")
        reference_left_arm = self._canonical_xml(self._find_body(reference, "left_arm"))
        reference_right_arm = self._canonical_xml(self._find_body(reference, "right_arm"))
        reference_collision = self._canonical_xml(self._find_default(reference, "collision"))
        reference_finger = self._canonical_xml(self._find_default(reference, "finger"))

        for scene_name in (
            "yam_inhand_transfer_base.xml",
            "yam_drawer_scene.xml",
            "yam_pour_screw_scene.xml",
        ):
            with self.subTest(scene=scene_name):
                root = self._parse_xml(scene_name)
                self.assertEqual(
                    self._canonical_xml(self._find_body(root, "left_arm")),
                    reference_left_arm,
                )
                self.assertEqual(
                    self._canonical_xml(self._find_body(root, "right_arm")),
                    reference_right_arm,
                )
                self.assertEqual(
                    self._canonical_xml(self._find_default(root, "collision")),
                    reference_collision,
                )
                self.assertEqual(
                    self._canonical_xml(self._find_default(root, "finger")),
                    reference_finger,
                )


if __name__ == "__main__":
    unittest.main()
