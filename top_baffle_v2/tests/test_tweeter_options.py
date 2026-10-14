"""ND25FN-4 vases replace whole uppers; they are never offered as a regular crescent swap."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src')]
from lx521_baffle.tweeter_options import ND25FW, BMR, ND25FN, PT2522, AMT17, TWEETER_FAMILIES, compatible_tweeters


class TweeterOptionsTests(unittest.TestCase):
    def test_nd25fn4_parts_are_their_own_family(self):
        self.assertEqual(set(compatible_tweeters('h2c_stock_lm_no_floor_stand', 'stock', 'lm_bottom')),
                         {ND25FW, BMR, ND25FN})
        self.assertNotIn(ND25FN, compatible_tweeters('h2c_slim_lm_no_floor_stand', 'slim', 'lm_bottom'))
        # The Obiwan vase top needs its keyed LM bottom, not the one-piece carrier.
        self.assertNotIn(ND25FN, compatible_tweeters('h2c_obiwan_core_lm_carrier_floor_stand', 'obiwan', 'lm_bottom'))
        # The keyed LM bottom takes every Obiwan upper; the vase top replaces the keyed LM top.
        for state in ('floor_stand', 'no_floor_stand'):
            self.assertEqual(set(compatible_tweeters(f'h2c_obiwan_lm_keyed_bottom_{state}', 'obiwan', 'lm_bottom')),
                             {ND25FW, BMR, ND25FN, PT2522, AMT17})
        self.assertEqual(compatible_tweeters('h2c_obiwan_lm_keyed_top', 'obiwan', 'lm_top'), [ND25FW, BMR, PT2522, AMT17])
        # The NL8 lid closes the floor-stand bay under every Obiwan upper, the ND25FN-4 vase top included.
        self.assertEqual(set(compatible_tweeters('h2c_obiwan_addon_nl8_service_lid', 'obiwan', 'lm_bottom')),
                         {ND25FW, BMR, ND25FN, PT2522, AMT17})
        self.assertNotIn(ND25FN, compatible_tweeters('h2c_obiwan_um', 'obiwan', 'um'))
        self.assertNotIn(ND25FN, compatible_tweeters('h2c_obiwan_wing', 'obiwan', 'regular_wing'))
        for name, role in (('h2c_dayton_nd25fn4_obiwan_top_floor_stand', 'lm_top'),
                           ('h2c_dayton_nd25fn4_stock_upper', 'um')):
            self.assertEqual(compatible_tweeters(name, ND25FN, role), [ND25FN])
        products = next(r['products'] for r in TWEETER_FAMILIES if r['id'] == ND25FN)
        self.assertEqual(products, ['stock', 'obiwan'])

    def test_bmr_is_one_family_with_multiple_mounts(self):
        self.assertEqual(len(TWEETER_FAMILIES), 5)
        for name in ('h2c_stock_upper_bmr', 'h2c_obiwan_bmr_crescent_opposed_TEBM35C10-4'):
            self.assertEqual(compatible_tweeters(name, 'obiwan', 'um'), [BMR])

    def test_dipole_parts_are_obiwan_only_and_their_own_family(self):
        for fam in (PT2522, AMT17):
            self.assertEqual(next(r['products'] for r in TWEETER_FAMILIES if r['id'] == fam), ['obiwan'])
            self.assertEqual(compatible_tweeters(f'h2c_{fam}_obiwan_t_above_upright', fam, 'um'), [fam])
            for name, role in (('h2c_obiwan_core_lm_carrier_no_floor_stand', 'lm_bottom'),
                               ('h2c_obiwan_lm_keyed_bottom_floor_stand', 'lm_bottom'), ('h2c_obiwan_lm_keyed_top', 'lm_top')):
                self.assertIn(fam, compatible_tweeters(name, 'obiwan', role))
            for name, role in (('h2c_obiwan_core_2_of_2_um_carrier', 'um'), ('h2c_obiwan_wing_flat_left', 'regular_wing'),
                               ('h2c_obiwan_addon_tweeter_crescent', 'regular_tweeter'), ('h2c_stock_lm_no_floor_stand', 'lm_bottom')):
                self.assertNotIn(fam, compatible_tweeters(name, name.split('_')[1], role))

    def test_stock_and_slim_bmr_magnets_have_no_standard_perimeter_mate(self):
        for family in ('stock', 'slim'):
            self.assertEqual(compatible_tweeters(f'h2c_{family}_wing', family, 'regular_wing'), [ND25FW])
        self.assertEqual(compatible_tweeters('h2c_slim_lm', 'slim', 'lm_bottom'), [ND25FW, BMR])


if __name__ == '__main__':
    unittest.main()
