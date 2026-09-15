"""Stock/Slim two-part core with contained seam-B registration.

Build from the original unsplit baffle, never by joining clearance-separated
print meshes. Two rear-biased pins belong to the vase, so the LM ends at the
unchanged Y315.95 joint. The hidden M3x20 clamp retains its original axis.
"""
from __future__ import annotations

from build123d import Box, Compound, Cylinder, Pos, Rot

from ..base import STAND_FOOT, THICKNESS_MM, baffle_solid
from ..cables import TS_ROUTE_CAPTIVE, UM_V1L_HANDOFF_KEY, cable_cutters
from ..proud import b2_split as split
from ..proud.b import TWEETER_DROP_MM, apply_magnet_base_cavities
from ..proud.b2 import OUTLINE_B2

JOINT_Y = split.SEAM_B_Y
JOINT_GAP = 0.05
PIN_X = (-24.0, 24.0)
PIN_Z = 9.5
PIN_D = 3.0
PIN_PROJECTION = 4.0
PIN_ROOT_OVERLAP = 1.0
SOCKET_D = 3.3
SOCKET_EXTRA_DEPTH = 0.25


def pin_tool(x: float, *, socket: bool):
    diameter = SOCKET_D if socket else PIN_D
    end = JOINT_Y + JOINT_GAP + PIN_ROOT_OVERLAP
    start = JOINT_Y - PIN_PROJECTION - (SOCKET_EXTRA_DEPTH if socket else 0)
    return Pos(x, (start + end) / 2, PIN_Z) * Rot(X=90) * Cylinder(
        diameter / 2, end - start)


def support_tools(family: str, owner: str):
    """Exact duct tools plus explicit insert/socket blockers, never printed."""
    from .. import base
    slim = family == "slim"
    tools = list(cable_cutters(
        um_handoff_key=UM_V1L_HANDOFF_KEY if slim else "proud",
        route_names={"ts"} if owner == "upper" else None,
        ts_y_range=(310, 1.e6) if owner == "upper" else (-1.e6, 317),
        ts_route_key=TS_ROUTE_CAPTIVE))
    if owner == 'upper_bmr':
        from ..proud import vase_tebm35c10_4 as bmr
        tools=[bmr._main_t_cable_duct(section_extra_mm=.3),bmr._upper_t_cable_duct(radial_extra_mm=.3)]
        for axis_y,clock,z0,z1 in (
            (bmr.LOWER_T_AXIS_Y_MM,bmr.LOWER_T_MOUNT_CLOCK_DEG,THICKNESS_MM-bmr.M2_INSERT_DEPTH_MM,THICKNESS_MM),
            (bmr.UPPER_T_AXIS_Y_MM,bmr.UPPER_T_MOUNT_CLOCK_DEG,bmr.REAR_T_MOUNT_Z_MM,bmr.REAR_T_MOUNT_Z_MM+bmr.M2_INSERT_DEPTH_MM)):
            for x,y in bmr._pilot_centers(axis_y,bmr.TEBM_MOUNT_PCD_MM,tuple(clock+90*i for i in range(4))):
                tools.append(bmr._vertical_cylinder(x,y,bmr.M2_INSERT_BORE_D_MM+.6,z0-.3,z1+.3))
    if owner == "lm":
        centres = base._pilot_centers(base.L22_CUTOUT[:2], base.L22_PILOT_PCD_MM, base.L22_PILOT_ANGLES_DEG)
        diameter, depth = base.M5_INSERT_ENTRY_D_MM, base.L22_PILOT_DEPTH_MM
        tools += [pin_tool(x, socket=True) for x in PIN_X]
        tools.append(split.seam_b_m3_mid_right_cutter())
        if not STAND_FOOT:
            tools += [Pos(x, y, base.BRIDGE_INSERT_DEPTH_MM / 2) * Cylinder(
                base.M5_INSERT_ENTRY_D_MM / 2 + .3, base.BRIDGE_INSERT_DEPTH_MM + .6)
                for x, y in base.BRIDGE_HOLE_XY]
    else:
        centres = base._pilot_centers(base.UM_CUTOUT[:2], base.UM_PILOT_PCD_MM, base.UM_PILOT_ANGLES_DEG)
        diameter, depth = base.UM_PILOT_D_MM, base.UM_PILOT_DEPTH_MM
        tools.append(split.seam_b_m3_vase_insert_cutter())
    tools += [Pos(x, y, THICKNESS_MM - depth / 2) * Cylinder(diameter / 2 + .3, depth + .6)
              for x, y in centres]
    return Compound(children=tools)


def gen_part(family: str, owner: str):
    """Return the LM or vase in installed coordinates for the active state."""
    if family not in {"stock", "slim"} or owner not in {"lm", "upper"}:
        raise ValueError((family, owner))
    from run_memory_guarded import require_guarded_build
    require_guarded_build("H2C proud core must run under the CAD memory guard")
    slim = family == "slim"
    cuts = []
    floor_law = None
    if slim:
        from ..proud.v1 import field_cutters, apply_v1_base_magnets, REAR_MM
        from ..proud.v1l_split import v1l_field_cutters, floor_ramp_wall_thickness_law
        cuts = list(v1l_field_cutters()) if owner == "lm" else list(field_cutters())
        floor_law = floor_ramp_wall_thickness_law if STAND_FOOT else None
    else:
        REAR_MM = 0.0

    body = baffle_solid(OUTLINE_B2, TWEETER_DROP_MM, crescent_rear_mm=REAR_MM)
    # Crop before expensive duct operations: no seam-A or seam-C masks exist.
    if owner == "lm":
        mask = Pos(0, (JOINT_Y - 10) / 2, -50) * Box(500, JOINT_Y + 10, 500)
    else:
        y0 = JOINT_Y + JOINT_GAP
        mask = Pos(0, (600 + y0) / 2, -50) * Box(500, 600 - y0, 500)
    body = body & mask
    for cutter in cuts:
        body -= cutter
    # No split at y=120: use the complete, continuous lower route system.
    ducts = cable_cutters(
        um_handoff_key=UM_V1L_HANDOFF_KEY if slim else "proud",
        route_names={"ts"} if owner == "upper" else None,
        ts_y_range=(310, 1.e6) if owner == "upper" else (-1.e6, 317),
        ts_route_key=TS_ROUTE_CAPTIVE)
    for duct in ducts:
        body -= duct
    if owner == "lm":
        if STAND_FOOT:
            body = split._option_b_floor_bottom(
                body, ducts, shape_cuts=cuts, wall_thickness_law=floor_law)
        body -= split.seam_b_m3_mid_right_cutter()
        for x in PIN_X:
            cavity = pin_tool(x, socket=True)
            # The hole may leave the top face only. A deficient intersection
            # here catches intrusion into ducts, the driver opening or rear.
            test = cavity & mask
            missing = (test - body).volume
            if missing > 0.01:
                raise RuntimeError(f"H2C {family} pin socket at X{x} intersects an existing void: {missing}")
            body -= cavity
    else:
        body = apply_v1_base_magnets(body) if slim else apply_magnet_base_cavities(body)
        body -= split.seam_b_m3_vase_insert_cutter()
        for x in PIN_X:
            pin = pin_tool(x, socket=False)
            root = pin & mask
            if (root - body).volume > 0.01:
                raise RuntimeError(f"H2C {family} pin root at X{x} lacks solid backing")
            body = body.fuse(pin)
    body = body.clean()
    if not body.is_valid or len(body.solids()) != 1 or body.volume <= 0:
        raise RuntimeError(f"H2C {family}/{owner} is not one valid solid")
    body.label = f"h2c_{family}_{owner}_{'floor_stand' if STAND_FOOT and owner == 'lm' else 'shared' if owner == 'upper' else 'no_floor_stand'}"
    return body
