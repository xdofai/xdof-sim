from __future__ import annotations

import unittest

import numpy as np

from xdof_sim.examples.visualize_randomization import (
    _format_rack_variant_label,
    build_contact_sheet,
)


class VisualizeRandomizationTests(unittest.TestCase):
    def test_format_rack_variant_label_uses_numeric_suffix(self) -> None:
        self.assertEqual(_format_rack_variant_label("DishRack040"), "040")
        self.assertEqual(_format_rack_variant_label("current"), "current")

    def test_build_contact_sheet_tiles_frames_row_major(self) -> None:
        red = np.full((4, 6, 3), (255, 0, 0), dtype=np.uint8)
        green = np.full((4, 6, 3), (0, 255, 0), dtype=np.uint8)
        blue = np.full((4, 6, 3), (0, 0, 255), dtype=np.uint8)

        sheet = build_contact_sheet([red, green, blue], cols=2)

        self.assertEqual(sheet.shape, (8, 12, 3))
        np.testing.assert_array_equal(sheet[:4, :6], red)
        np.testing.assert_array_equal(sheet[:4, 6:12], green)
        np.testing.assert_array_equal(sheet[4:8, :6], blue)
        np.testing.assert_array_equal(sheet[4:8, 6:12], np.zeros((4, 6, 3), dtype=np.uint8))

    def test_build_contact_sheet_rejects_empty_inputs(self) -> None:
        with self.assertRaises(ValueError):
            build_contact_sheet([], cols=4)


if __name__ == "__main__":
    unittest.main()
