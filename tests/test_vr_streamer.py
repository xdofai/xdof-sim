from __future__ import annotations

import unittest

from xdof_sim.examples.vr_streamer import _parse_visible_groups_arg


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


if __name__ == "__main__":
    unittest.main()
