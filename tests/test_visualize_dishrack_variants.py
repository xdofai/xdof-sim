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
        self.assertEqual(_format_rack_variant_label("DishRack040"), "040")
        self.assertEqual(_format_rack_variant_label("current"), "current")

    def test_format_plate_variant_label_uses_numeric_suffix(self) -> None:
        self.assertEqual(_format_plate_variant_label("plate_11"), "11")
        self.assertEqual(_format_plate_variant_label("current"), "current")

    def test_label_text_includes_variant_names(self) -> None:
        label = _label_text(
            frame_idx=1,
            total_frames=22,
            camera_name="top",
            sweep_kind="dish_rack",
            rack_variant="DishRack040",
            plate_variant="current",
        )

        self.assertIn("dishrack [2/22]", label)
        self.assertIn("cam=top", label)
        self.assertIn("rack=040", label)
        self.assertIn("plate=current", label)

    def test_label_text_formats_plate_sweep(self) -> None:
        label = _label_text(
            frame_idx=2,
            total_frames=19,
            camera_name="top",
            sweep_kind="plate",
            rack_variant="current",
            plate_variant="plate_11",
        )

        self.assertIn("plate [3/19]", label)
        self.assertIn("cam=top", label)
        self.assertIn("rack=current", label)
        self.assertIn("plate=11", label)

    def test_validate_variant_rejects_unknown_name(self) -> None:
        self.assertEqual(_validate_variant("plate", "current"), "current")
        with self.assertRaises(ValueError):
            _validate_variant("dish_rack", "does_not_exist")


if __name__ == "__main__":
    unittest.main()
