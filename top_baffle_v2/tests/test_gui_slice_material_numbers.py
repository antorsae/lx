"""Small Bambu interface paths use E.03, which must not evade material QA."""
import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from audit_gui_slice import material_extrusions


def program(interface_tool):
    return f'''M83
T0 ; model filament
; FEATURE: Outer wall
G1 X1 Y1 E.04
; FEATURE: Support
G1 X2 Y1 E3e-2
T{interface_tool}
; FEATURE: Support interface
G1 X+.2 Y.3 E.0367
G1 X.4 Y.3 E-.02
'''


def test_fractional_interface_extrusion_is_counted(tmp_path):
    path=tmp_path/'slice.gcode';path.write_text(program(1))
    assert material_extrusions(path,True)==dict(model=1,support=1,interface=1)


def test_wrong_material_in_small_interface_is_rejected(tmp_path):
    path=tmp_path/'slice.gcode';path.write_text(program(0))
    with pytest.raises(ValueError,match='interface extrudes with T0'):
        material_extrusions(path,True)
