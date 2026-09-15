"""A fused ND25FN upper must not be offered as a regular crescent swap."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from lx521_baffle.tweeter_options import ND25FW, BMR, ND25FN, TWEETER_FAMILIES, compatible_tweeters
from lx521_baffle.h2c.dayton import BODY, RETAINED_SOURCES, body_authority, mesh_name


class TweeterOptionsTests(unittest.TestCase):
    def test_common_lm_accepts_all_three_but_regular_um_does_not(self):
        self.assertEqual(set(compatible_tweeters('h2c_obiwan_core_lm', 'obiwan', 'lm_bottom')),
                         {ND25FW, BMR, ND25FN})
        self.assertNotIn(ND25FN, compatible_tweeters('h2c_obiwan_um', 'obiwan', 'um'))
        self.assertNotIn(ND25FN, compatible_tweeters('h2c_obiwan_wing', 'obiwan', 'regular_wing'))
        self.assertEqual(compatible_tweeters('h2c_dayton_nd25fn4_body', ND25FN, 'crescent_body'), [ND25FN])

    def test_bmr_is_one_family_with_multiple_mounts(self):
        self.assertEqual(len(TWEETER_FAMILIES), 3)
        for name in ('h2c_stock_upper_bmr', 'h2c_obiwan_bmr_crescent_opposed_TEBM35C10-4'):
            self.assertEqual(compatible_tweeters(name, 'obiwan', 'um'), [BMR])

    def test_stock_and_slim_bmr_magnets_have_no_standard_perimeter_mate(self):
        for family in ('stock', 'slim'):
            self.assertEqual(compatible_tweeters(f'h2c_{family}_wing', family, 'regular_wing'), [ND25FW])
            self.assertEqual(compatible_tweeters(f'h2c_{family}_lm', family, 'lm_bottom'), [ND25FW, BMR])

    def test_body_rename_preserves_transform_and_input_identity(self):
        import json
        source = RETAINED_SOURCES[BODY]
        original = json.loads(source.with_suffix('.print.json').read_text())
        renamed = body_authority()
        self.assertEqual(mesh_name(source), BODY)
        self.assertEqual(renamed['stl'], BODY)
        for key in ('source_to_stl_matrix', 'stl_sha256', 'approved_source_sha256'):
            self.assertEqual(renamed[key], original[key])
        with self.assertRaises(ValueError):
            mesh_name(ROOT / 'unrecognized_body.stl')


if __name__ == '__main__':
    unittest.main()
