import unittest
from pathlib import Path
from copy import deepcopy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from audit_changeover import audit_changeover_text
from lx521_baffle.print_policy import normalize_material_mapping, validate_material_mapping


def commands(volume=560, speed=12):
    length=volume/2.4053
    feed=speed/2.4053*60
    return '\n'.join(f'''M620 S{tool}A
M620.10 A0 F{feed} L{length} H0.6 T270 P260 S1
M620.10 A1 F{feed} L{length} H0.6 T240 P220 S1
T{tool}
; VFLUSH_START
;VG1 E{length} F299
; VFLUSH_END
M621 S{tool}A''' for tool in [1,0])


class ChangeoverTests(unittest.TestCase):
    def test_detects_stale_gcode_despite_correct_metadata(self):
        self.assertEqual(audit_changeover_text(commands())['change_count'],2)
        for text in [commands(volume=280), commands(speed=40), commands().replace('M620.10 A0','; M620.10 A0')]:
            with self.assertRaises(AssertionError): audit_changeover_text(text)

    def test_policy_preserves_model_flow_and_printing_temperatures(self):
        for original,wanted in [(['7','0'],['7','12']),(['7','8','0','0'],['7','8','12','12'])]:
            s=dict(filament_settings_id=['TINMORRY PETG-GF','Bambu PLA Basic'],
                   filament_flush_volumetric_speed=original,nozzle_temperature=['260','260','220','220'],
                   filament_max_volumetric_speed=['12','12','21','40'],flush_multiplier=['2'])
            before=deepcopy(s);normalize_material_mapping(s);validate_material_mapping(s)
            self.assertEqual(s['filament_flush_volumetric_speed'],wanted)
            self.assertEqual(s['flush_volumes_matrix'],['0','560','560','0'])
            self.assertEqual(s['flush_multiplier'],['1'])
            for key in ['nozzle_temperature','filament_max_volumetric_speed']:self.assertEqual(s[key],before[key])
            bad=deepcopy(s);bad['filament_flush_volumetric_speed'][-1]='40'
            with self.assertRaises(ValueError):validate_material_mapping(bad)


if __name__=='__main__':unittest.main()
