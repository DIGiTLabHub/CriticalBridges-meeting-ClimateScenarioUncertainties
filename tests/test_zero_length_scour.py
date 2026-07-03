import unittest
from collections import Counter

from BridgeModeling.ZeroLengthElement import (
    depths,
    effective_scour_depth_mm,
    get_soil_depth,
    soil_spring_removed_by_scour,
    zero_length_defs,
)


def skipped_material_counts(scour_depth):
    effective_depth = effective_scour_depth_mm(scour_depth)
    skipped = Counter()

    for _, _, _, mat_list, _ in zero_length_defs:
        should_skip = False
        for mat_tag in mat_list:
            depth = get_soil_depth(mat_tag)
            if depth is not None and soil_spring_removed_by_scour(depth, effective_depth):
                should_skip = True
                break

        if should_skip:
            for mat_tag in set(mat_list):
                skipped[mat_tag] += 1

    return skipped


class ZeroLengthScourTests(unittest.TestCase):
    def test_scour_depth_rounds_up_to_next_soil_spring_level(self):
        self.assertEqual(effective_scour_depth_mm(0), 0.0)
        self.assertEqual(effective_scour_depth_mm(1), 500.0)
        self.assertEqual(effective_scour_depth_mm(499), 500.0)
        self.assertEqual(effective_scour_depth_mm(500), 500.0)
        self.assertEqual(effective_scour_depth_mm(501), 1000.0)

    def test_scour_depth_is_capped_at_deepest_modeled_soil_spring(self):
        max_effective_depth = max(depths) - min(depths)
        self.assertEqual(effective_scour_depth_mm(50000), max_effective_depth)

    def test_zero_scour_removes_no_soil_springs(self):
        for depth in depths:
            self.assertFalse(soil_spring_removed_by_scour(depth, effective_scour_depth_mm(0)))

    def test_rounded_scour_removes_same_levels_across_all_piles(self):
        skipped = skipped_material_counts(500)

        self.assertEqual(skipped[101], 18)
        self.assertEqual(skipped[102], 18)
        self.assertEqual(skipped[201], 18)
        self.assertEqual(skipped[202], 18)

        for mat_tag in [103, 203]:
            self.assertEqual(skipped[mat_tag], 0)


if __name__ == "__main__":
    unittest.main()
