from copy import deepcopy
import unittest

from lx521_baffle.print_policy import (
    normalize_material_mapping, validate_material_mapping, validate_object_infill,
    native_material_values,
    selected_filament_value,
)


class PrintPolicyTests(unittest.TestCase):
    def test_native_variant_columns_preserve_the_PLA_recipe(self):
        self.assertEqual(native_material_values(['20','20','25','25'], '20', ['25','25']),
                         ['20','20','25','25'])
        self.assertEqual(native_material_values(['260','220'], ['260'], ['220']), ['260','220'])
        self.assertEqual(selected_filament_value(['21','40'], 'High Flow'), '40')
        self.assertEqual(selected_filament_value(['21','40'], 'Standard'), '21')
        self.assertEqual(native_material_values(['12','40'], ['12','12'], ['21','40']), ['12','40'])

    def test_partial_gui_workaround_is_rejected(self):
        settings = {'filament_settings_id': ['TINMORRY PETG-GF Profile @BBL P2S', 'Bambu PLA Basic @BBL P2S 0.6 nozzle']}
        normalize_material_mapping(settings)
        validate_material_mapping(settings)
        for key in ('filament_map_2', 'filament_nozzle_map', 'filament_volume_map'):
            broken = deepcopy(settings)
            broken[key] = ['1']
            with self.assertRaises(ValueError): validate_material_mapping(broken)

    def test_object_override_cannot_hide_behind_ten_percent_global(self):
        settings = {'sparse_infill_density': '10%', 'sparse_infill_pattern': 'gyroid'}
        validate_object_infill(settings, [{}], 'regular_wing')
        with self.assertRaises(ValueError):
            validate_object_infill(settings, [{'sparse_infill_density': '30%'}], 'regular_wing')

    def test_reversed_materials_are_rejected(self):
        with self.assertRaises(ValueError):
            normalize_material_mapping({'filament_settings_id': ['Bambu PLA Basic', 'TINMORRY PETG-GF']})


if __name__ == '__main__': unittest.main()
