from __future__ import annotations

import unittest

from xdof_sim.examples.visualize_dishrack_variants import (
    _format_plate_variant_label,
    _format_rack_variant_label,
    _label_text,
    _validate_variant,
)


class VisualizeDishrackVariantsTests(unittest.TestCase):
    def test_format_rack_variant_label_uses_numeric_suffix(self) -> None:
        self.assertEqual(_format_rack_variant_label("dish_rack_7"), "7")
        self.assertEqual(_format_rack_variant_label("dish_rack_0"), "0")

    def test_format_plate_variant_label_uses_numeric_suffix(self) -> None:
        self.assertEqual(_format_plate_variant_label("plate_11"), "11")
        self.assertEqual(_format_plate_variant_label("plate_0"), "0")

    def test_label_text_includes_variant_names(self) -> None:
        label = _label_text(
            frame_idx=1,
            total_frames=22,
            camera_name="top",
            sweep_kind="dish_rack",
            rack_variant="dish_rack_7",
            plate_variant="plate_0",
        )

        self.assertIn("dishrack [2/22]", label)
        self.assertIn("cam=top", label)
        self.assertIn("rack=7", label)
        self.assertIn("plate=0", label)

    def test_label_text_formats_plate_sweep(self) -> None:
        label = _label_text(
            frame_idx=2,
            total_frames=19,
            camera_name="top",
            sweep_kind="plate",
            rack_variant="dish_rack_0",
            plate_variant="plate_11",
        )

        self.assertIn("plate [3/19]", label)
        self.assertIn("cam=top", label)
        self.assertIn("rack=0", label)
        self.assertIn("plate=11", label)

    def test_validate_variant_rejects_unknown_name(self) -> None:
        self.assertEqual(_validate_variant("plate", "current"), "plate_0")
        with self.assertRaises(ValueError):
            _validate_variant("dish_rack", "does_not_exist")


if __name__ == "__main__":
    unittest.main()
