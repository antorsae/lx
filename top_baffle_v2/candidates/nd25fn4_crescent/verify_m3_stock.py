"""Confirm the user-supplied M3 insert stock against shared and V4 dimensions."""
import hashlib
import json
import ast
from pathlib import Path
import v4_model as model
from validate_hardware import validate_hardware
from lx521_baffle.print_policy import policy, policy_sha256
from lx521_baffle.obiwan import floor

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    stock=policy()['hardware_stock']['M3_insert']
    assert (stock['thread'],stock['outer_diameter_mm'],stock['length_mm'])==('M3',5.,4.)
    interface=model.interface
    # Proud and Obi-Wan deliberately cannot be imported in one routing
    # process. These two proud bore dimensions are literal source constants.
    b2_path=ROOT/'src/lx521_baffle/proud/b2_split.py'
    wanted={'SEAM_B_M3_INSERT_BORE_D_MM','SEAM_B_M3_INSERT_DEPTH_MM'}
    b2={}
    for node in ast.parse(b2_path.read_text()).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in wanted:
                    b2[target.id]=ast.literal_eval(node.value)
    assert set(b2)==wanted
    bores={
        'UM_driver':(interface.UM_PILOT_D_MM,interface.UM_PILOT_DEPTH_MM),
        'LM_UM_joint':(interface.JOINT_INSERT_BORE_D,interface.JOINT_INSERT_DEPTH_MM),
        'UM_T_joint':(interface.TWEETER_JOINT_INSERT_BORE_D,interface.TWEETER_JOINT_INSERT_DEPTH_MM),
        'NL8':(floor.NL8_SCREW_D_MM,floor.NL8_INSERT_L_MM),
        'B2_seam':(b2['SEAM_B_M3_INSERT_BORE_D_MM'],b2['SEAM_B_M3_INSERT_DEPTH_MM']),
        'V4_retainer':(2*model.retained.M.insert_pilot_radius,model.retained.M.insert_length),
    }
    for name,dimensions in bores.items():
        assert dimensions==(4.6,stock['length_mm']), (name,dimensions)
    # Re-run the current exported-mesh bore, screw-head and access gauges.
    hardware=validate_hardware()
    m=model.retained.M
    tip=m.retainer_back_x+m.screw_length
    engagement=min(tip-m.service_end_x,stock['length_mm'])
    bottom_clearance=m.service_end_x+m.insert_length-tip
    insert_ligament=m.screw_circle_radius-stock['outer_diameter_mm']/2-m.flange_cavity_radius
    assert abs(engagement-3.7)<1e-8 and abs(bottom_clearance-.3)<1e-8
    assert insert_ligament>=1.39
    report=dict(status='pass',stock=stock,policy_sha256=policy_sha256(),
        image_sha256=sha(ROOT/stock['reference_image']),source_sha256=sha(__file__),
        dimensional_sources_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [ROOT/'src/lx521_baffle/base.py',Path(interface.__file__),Path(floor.__file__),b2_path,HERE/'v4_model.py']},
        body_sha256=sha(HERE/'STL'/model.BODY_FILE),retainer_sha256=sha(HERE/'STL/03_Tweeter_Retainer_PRINT_TWO.stl'),
        hardware_mesh_validation_sha256=sha(HERE/'hardware_validation.json'),
        shared_bores_mm=bores,nominal_diametral_interference_mm=round(stock['outer_diameter_mm']-2*m.insert_pilot_radius,6),
        screw_length_under_head_mm=m.screw_length,screw_bearing_plane_x_mm=m.retainer_back_x,
        insert_entry_x_mm=m.service_end_x,screw_tip_x_mm=tip,insert_floor_x_mm=m.service_end_x+m.insert_length,
        screw_engagement_mm=round(engagement,6),screw_tip_to_floor_mm=round(bottom_clearance,6),
        metal_insert_to_driver_cavity_ligament_mm=round(insert_ligament,6),
        geometry_changed=False,physical_heat_set_fit='not tested; uses the existing project pilot convention')
    path=ROOT/'review/print_policy_update_20260912/M3_stock_confirmation.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print('PASS: Hanglife M3 x 5 x 4 stock; shared D4.6 x 4 bores; M3x8 engagement 3.7 mm, floor clearance 0.3 mm')


if __name__=='__main__':main()
