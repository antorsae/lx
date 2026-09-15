"""Measure M3 retention on exported meshes, including removal and head access."""
import json
from pathlib import Path
import numpy as np
import trimesh
import v4_model as model
from mesh_ops import solid

HERE = Path(__file__).resolve().parent


def axial_cylinder(radius, lo, hi, y, z):
    matrix = trimesh.transformations.rotation_matrix(np.pi/2, [0, 1, 0])
    matrix[:3, 3] = [(lo+hi)/2, y, z]
    return solid(trimesh.creation.cylinder(radius=radius, height=hi-lo, sections=192, transform=matrix))


def validate_hardware():
    from validate import restored_print_parts
    _, parts = restored_print_parts()
    body = solid(parts['housing'])
    ring = trimesh.load_mesh(HERE/'assembly/retainer_local_mm.stl', process=True)
    rs = solid(ring)
    m = model.retained.M
    # Independently compare against the production UM convention.
    assert 2*m.insert_pilot_radius == model.interface.UM_PILOT_D_MM == 4.6
    assert m.insert_length == model.interface.UM_PILOT_DEPTH_MM == 4.0
    assert m.screw_clearance_radius == 1.7
    inner_wall = m.screw_circle_radius-m.insert_pilot_radius-m.flange_cavity_radius
    ring_edge = m.retainer_outer_radius-m.screw_circle_radius-m.screw_clearance_radius
    assert inner_wall >= 1.59 and ring_edge >= 1.09
    assert m.retainer_outer_radius < m.cap_radius+m.cap_clearance-.2
    rows = []
    for which in (0, 1):
        sign = 1 if which == 0 else -1
        for index, angle in enumerate(model.retained.SCREW_ANGLES):
            y, z = m.screw_circle_radius*np.array([np.cos(angle), np.sin(angle)])
            pilot = axial_cylinder(2.3-.035, 4.51, 8.49, y, z)
            head = axial_cylinder(2.75, -.2-2.6, .19, y, z)
            # Local pod -> package frame, same independently measured datum
            # as the shipped cap/driver. Negative X pod is the opposite end.
            def to_pod(obj):
                if which:
                    return obj.rotate((0, 180, 0)).translate((0, 0, 31))
                return obj.translate((0, 0, -31))
            collision = abs((body ^ to_pod(pilot)).volume())
            head_collision = abs((body ^ to_pod(head)).volume())
            assert collision < .01, (which, index, 'pilot obstruction', collision)
            assert head_collision < .01, (which, index, 'head obstruction', head_collision)
            clearance = abs((rs ^ axial_cylinder(1.665, .21, 3.19, y, z)).volume())
            assert clearance < .003, (index, 'retainer clearance', clearance)
            # Full annulus surrounding each pilot must remain material,
            # except the open chamber behind the insert's seating plane.
            outer = to_pod(axial_cylinder(2.3+.8, 4.65, 8.35, y, z))
            inner = to_pod(axial_cylinder(2.3+.06, 4.64, 8.36, y, z))
            annulus = outer-inner
            missing = abs((annulus-body).volume())
            assert missing < .025, (which, index, 'missing insert surround', missing)
            rows.append(dict(pod=which+1, screw=index+1, pilot_collision_mm3=collision,
                             head_collision_mm3=head_collision, missing_surround_mm3=missing))
    report = dict(status='pass', thread='M3', count=6, screw='M3 x 8 mm socket head',
                  insert='M3, 4 mm long, 4.6 mm pilot', screw_circle_diameter_mm=2*m.screw_circle_radius,
                  retainer_edge_ligament_mm=ring_edge, insert_to_driver_ligament_mm=inner_wall,
                  screw_engagement_mm=3.7, insert_bottom_clearance_mm=.3,
                  cap_entry_radial_clearance_mm=m.cap_radius+m.cap_clearance-m.retainer_outer_radius,
                  sites=rows, physical_fit='not yet tested')
    (HERE/'hardware_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    return report


if __name__ == '__main__':
    print(json.dumps(validate_hardware(), indent=2))
