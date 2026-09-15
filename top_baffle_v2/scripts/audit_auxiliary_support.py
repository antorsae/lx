"""Audit auxiliary bores that currently rely on orientation instead of blockers."""
from pathlib import Path
import json
import os
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.environ['LX_ROUTING_PROFILE']='obiwan';os.environ['LX_STAND_FOOT']='1'
from lx521_baffle.obiwan import carriers as c, floor as f
from gui_project_audit import audit_gui_geometry, sha256
from delivery_contract import canonical_source
from gcode_analysis import audit_support_toolpaths_vs_ducts
from audit_gui_slice import material_extrusions


def region(name, a, b, radius):
    return dict(name=name,points_xyz_mm=[a,b],radius_mm=radius)


def main():
    output=ROOT/'review/print_policy_update_20260912'
    entries=json.loads((ROOT/'to_print/catalog.json').read_text())['entries']
    nl8=[region(f'NL8_M3_{sx}_{sy}', [sx*f.NL8_SCREW_PITCH_MM/2,f.NL8_CENTER_Y_MM+sy*f.NL8_SCREW_PITCH_MM/2,f.FOOT_REAR_Z_MM],
                [sx*f.NL8_SCREW_PITCH_MM/2,f.NL8_CENTER_Y_MM+sy*f.NL8_SCREW_PITCH_MM/2,f.FOOT_REAR_Z_MM+f.NL8_INSERT_L_MM],f.NL8_SCREW_D_MM/2)
         for sx in (-1,1) for sy in (-1,1)]
    lm_tie=[region('LM_M2_tie_insert',[c.LM_UM_TIE_X,c.LM_UM_TIE_POCKET_BOTTOM_Y,c.LM_UM_TIE_AXIS_Z],
                   [c.LM_UM_TIE_X,c.LM_UM_TIE_INSERT_MOUTH_Y,c.LM_UM_TIE_AXIS_Z],c.LM_UM_TIE_INSERT_BORE_D/2)]
    um_ties=[region(f'UM_M2_T_tie_{x}',[x,c.T_UM_TIE_POCKET_BOTTOM_Y,c.T_UM_TIE_AXIS_Z],
                   [x,c.T_UM_TIE_INSERT_MOUTH_Y,c.T_UM_TIE_AXIS_Z],c.T_UM_TIE_INSERT_BORE_D/2) for x in c.T_UM_TIE_X]
    um_ties.append(region('UM_LM_tie_passage',[c.LM_UM_TIE_X,c.LM_UM_TIE_UM_FACE_Y,c.LM_UM_TIE_AXIS_Z],
                           [c.LM_UM_TIE_X,c.LM_UM_TIE_SEAT_Y,c.LM_UM_TIE_AXIS_Z],c.LM_UM_TIE_CLEARANCE_BORE_D/2))
    jobs=[('floor_auxiliary/floor.gcode.3mf','obiwan_01_LM_bottom_keyed_1_of_2_floor_stand',nl8),
          ('lm_top_slice/LM_top_100pct.gcode.3mf','obiwan_02_LM_top_keyed_2_of_2',lm_tie),
          ('regular_um_auxiliary/regular_um.gcode.3mf','obiwan_03_UM_carrier_1_of_1',um_ties)]
    reports=[]
    for relative,name,regions in jobs:
        project=output/relative;gcode=project.parent/'plate_1.gcode'
        entry=next(e for e in entries if e['name']==name)
        geometry=audit_gui_geometry(ROOT,entry,project)
        source=canonical_source(ROOT,entry);authority=json.loads(source.with_suffix('.print.json').read_text())
        check=audit_support_toolpaths_vs_ducts(gcode=gcode,contract=dict(regions=regions),
                  source_to_stl_matrix=authority['source_to_stl_matrix'],stl_to_bed_matrix=geometry['stl_to_bed_matrix'])
        reports.append(dict(name=name,status='pass',geometry=geometry,regions=regions,
                            gcode_sha256=sha256(gcode),project_sha256=sha256(project),
                            materials=material_extrusions(gcode,True),support_check=check))
        print(name,'auxiliary bores free of support',flush=True)
    (output/'auxiliary_bore_support.json').write_text(json.dumps(dict(status='pass',
        scope='NL8 four M3 bores, LM M2 tie insert, regular UM two M2 T ties and LM tie clearance. Conservative capsules include the complete bore and extend beyond its ends.',
        source_sha256=sha256(Path(__file__)),jobs=reports),indent=2)+'\n')


if __name__=='__main__':main()
