"""Regression checks for H2C nozzle reach, native defaults and CLI overrides."""
from copy import deepcopy
from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from lx521_baffle.h2c.printing import project_settings,job_process
from audit_h2c_print import deposition,pause_audit
from build_h2c_release import prepared_is_current
from lx521_baffle.io import sha256_file


class H2CPrintingTests(unittest.TestCase):
    def gcode(self,text):
        directory=tempfile.TemporaryDirectory();self.addCleanup(directory.cleanup)
        path=Path(directory.name)/'test.gcode';path.write_text(text);return path

    def test_material_commands_with_hotend_suffix(self):
        path=self.gcode('''M190 S70
M104 S260
G90
M83
T0 H0
G1 X40 Y40 Z0.2
; Z_HEIGHT: 0.2
; FEATURE: Outer wall
G1 X42 E0.1
T1001
G1 X44 E0.1
T1 H0
; FEATURE: Support interface
G1 Y42 E0.1
''')
        result=deposition(path)
        self.assertEqual(result['counts'],{'model':2,'interface':1})

    def test_pla_bead_must_stay_in_right_nozzle_reach(self):
        path=self.gcode('''M190 S70
G90
M83
T0 H0
G1 X40 Y40 Z0.2
; FEATURE: Outer wall
G1 X42 E0.1
T1 H0
; FEATURE: Support interface
G1 X25 Y50
G1 Y52 E0.1
''')
        with self.assertRaises(ValueError):deposition(path)

    def test_arc_extremum_checked_between_reachable_endpoints(self):
        path=self.gcode('''M190 S70
G90
M83
T0 H0
G1 X40 Y40 Z0.2
; FEATURE: Outer wall
G1 X42 E0.1
T1 H0
; FEATURE: Support interface
G1 X26 Y48
G2 X26 Y52 I0 J2 E0.1
''')
        with self.assertRaises(ValueError):deposition(path)

    def test_pause_restore_accepts_slicer_modal_feed_optimization(self):
        path=self.gcode('''M190 S70
M104 S260
G90
M83
; CHANGE_LAYER
; Z_HEIGHT: 0.2
; LAYER_HEIGHT: 0.2
; FEATURE: Custom
; LX521_H2C_MAGNET_INSERTION
; Insert one
G90
M400
G1 Z300 F1200
M400
M400 U1
G1 Z0.20
M400
; LX521_H2C_MAGNET_RESUME
G1 X40 Y40 Z0.2
; FEATURE: Outer wall
G1 X41 E0.1
''')
        self.assertEqual(len(pause_audit(path,[.2])),1)
        bad=self.gcode(path.read_text().replace('G1 Z0.20','G1 Z0.36'))
        with self.assertRaises((ValueError,StopIteration)):pause_audit(bad,[.2])

    def test_default_make_target_is_h2c_and_p2s_is_retired(self):
        current=subprocess.check_output(['make','-n','all'],cwd=ROOT,text=True)
        self.assertIn('scripts/h2c_pipeline.py all',current)
        self.assertNotIn('remote_cad.py',current)
        retired=subprocess.run(['make','-n','PRINTER=P2S','all'],cwd=ROOT,text=True,capture_output=True)
        self.assertNotEqual(retired.returncode,0)
        self.assertIn('PRINTER=P2S is deprecated; H2C is the only supported printer',retired.stderr)

    def test_preparation_cache_rejects_changed_project_or_geometry(self):
        project=self.gcode('prepared project bytes')
        source=self.gcode('source mesh bytes')
        inputs={'source_version':1}
        job=dict(project=str(project),project_sha256=sha256_file(project),prepare_inputs=inputs,
            preparation=dict(sources=[dict(path=str(source),sha256=sha256_file(source))]))
        self.assertTrue(prepared_is_current(job,inputs))
        self.assertFalse(prepared_is_current(job,{'source_version':2}))
        project.write_text('changed project')
        self.assertFalse(prepared_is_current(job,inputs))
        project.write_text('prepared project bytes')
        source.write_text('changed mesh')
        self.assertFalse(prepared_is_current(job,inputs))


if __name__=='__main__':unittest.main()
