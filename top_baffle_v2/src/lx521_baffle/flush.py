"""Shared flush-seat, pilot and LM-pad dimensions for skeletal Obi-Wan.

R6F builds the two carrier rings directly in
``lx521_baffle.obiwan.carriers``; it no longer thins a complete B2/V1L
field. Its LM-owned UM passage and LM/UM-owned T passage segments, including
their covered Z bumps, are owned by
``lx521_baffle.obiwan.route``.  The field-cutter
helpers at the bottom remain only for opening pre-R6 artifacts/coupons.

Drivers (LX521.4 production, SEAS "-SL" customs -- NOT the LX521
prototype W22EX001 / 10F/8424G00 some older comments name):
  LM: SEAS U22REX/P-SL (H1659-08), flange O220.6+-0.4 (datasheet),
      thickness measured 6.0 (datasheet drawing says 5.5+-0.2).
  UM: SEAS MU10RB-SL (H1658-04), flange O98+-0.4 (datasheet),
      thickness measured 4.0 (datasheet drawing says 5.4+-0.2).
Depths are OWNER-MEASURED values; confirm with the coupon seat block
before printing full pieces (a too-deep seat sinks the flange and
re-introduces a diffraction step at the rim).

SL's construction notes say the mids "must not be recessed" -- the
LX521.4 was voiced with proud flanges. Obi-Wan is a deliberate
experiment arm against that baseline (DSP re-EQ per configuration),
not a build error.

The core rear is z=6.8 and each surviving printed R6F passage ends in a plain,
service-accessible flush mouth at its owner boundary. Insert bypasses within
those printed spans are continuously covered. The UM cable then floats behind
UM and T floats behind the tweeter crescent; no Obi-Wan printed grommet or split
clip is modeled. Re-measure both physical flange edges before printing;
owner-measured values still override the nominal datasheet thicknesses below.

The W22-ring insert bores would break through the 11.5 plate once the
seat drops the front face to z=12.3 (bore floor 5.5 < rear plane 6.8),
so Obi-Wan keeps six pad buttons of material on the rear (nominal lower
face z=5.3, joining the carrier at z=6.8): the legacy V1L field cutters
are pre-punched with pad-shaped
reliefs rather than unioning bosses on afterwards (monolithic, no
glue-line-like union seams). Pilot bores are re-cut deeper here; the
base module's stock front-blind bores become 0.8 stubs inside the
recess void and are harmless.
"""

from __future__ import annotations

import math

from build123d import Cylinder, Pos

from .base import (
    L22_CUTOUT,
    L22_PILOT_ANGLES_DEG,
    L22_PILOT_D_MM,
    L22_PILOT_DEPTH_MM,
    L22_PILOT_PCD_MM,
    M5_INSERT_ENTRY_D_MM,
    THICKNESS_MM,
    UM_CUTOUT,
    UM_PILOT_ANGLES_DEG,
    UM_PILOT_D_MM,
    UM_PILOT_DEPTH_MM,
    UM_PILOT_PCD_MM,
    m5_insert_bore_cutter,
)

# Flange envelopes (SEAS datasheets) + drop-in clearance.
LM_FLANGE_D_MM = 220.6   # U22REX/P-SL o220.6+-0.4
LM_FLANGE_T_MM = 6.0     # owner-measured (datasheet 5.5+-0.2)
UM_FLANGE_D_MM = 98.0    # MU10RB-SL o98+-0.4
UM_FLANGE_T_MM = 4.0     # owner-measured (datasheet 5.4+-0.2)
RECESS_CLR_MM = 0.6      # diametral; tune on the coupon seat block

LM_SEAT_Z = THICKNESS_MM - LM_FLANGE_T_MM   # 12.3
UM_SEAT_Z = THICKNESS_MM - UM_FLANGE_T_MM   # 14.3
LM_RECESS_R = (LM_FLANGE_D_MM + RECESS_CLR_MM) / 2.0   # 110.6
UM_RECESS_R = (UM_FLANGE_D_MM + RECESS_CLR_MM) / 2.0   # 49.3

# Rear pad buttons under every W22-ring insert, sized around the
# owner's ACTUAL M5 x 5.8 inserts (not the generic 6.8 bore rule):
# the 5.5 wall under the seat can NEVER swallow a 5.8 insert, so some
# added rear material is irreducible. Minimum honest pad: bore =
# 5.8 + 0.4 settle = 6.2 below the seat (floor z=6.1) + 0.8 floor ->
# pad face z=5.3: straight O9.6 buttons, 1.5 proud, minimum rim 1.55 around
# the O6.5 x 2.0 entry relief (the deeper bore stays O6.4; the plate itself
# grips the insert's top 4.3 -- only
# the bottom 1.5 rides the pad), 0.6 x 45 deg rim chamfer. ALL pads
# concentric with their bores: the top one clears seam C (x=-5.6) by
# 0.8 -- the earlier flared/offset O13 version breached the seam
# plane and got sliced at the piece edge. NOTE: the shallower bore
# means an M5 x 14 + washer can bottom out before clamping -- use
# M5 x 12 at the U22 on Obi-Wan (see PRINTING.md).
LM_INSERT_L_MM = 5.8
LM_BORE_DEPTH_MM = LM_INSERT_L_MM + 0.4    # 6.2
PAD_FLOOR_MM = 0.8
PAD_D_MM = 9.6
PAD_FACE_Z = 18.3 - LM_FLANGE_T_MM - LM_BORE_DEPTH_MM - PAD_FLOOR_MM  # 5.3
PAD_CHAMFER_MM = 0.6
# R6F's thinned UM seat membrane needs local insert bosses. D8 retains a
# 1.7-mm radial wall around the D4.6 heat-set bore; a short spoke carries it
# to the structural outer lip. Two extrusion widths remain behind the blind
# 4.0-mm receiver; starting a boss at the bore bottom would leave it open.
UM_PAD_D_MM = 8.0
UM_PAD_FLOOR_MM = 0.8

_LM_C = (L22_CUTOUT[0], L22_CUTOUT[1])
_UM_C = (UM_CUTOUT[0], UM_CUTOUT[1])

# All released families now share the same W22 clock.  Keep the Obi-Wan name
# as an explicit compatibility contract for its carrier and route modules.
OBIWAN_LM_PILOT_ANGLES_DEG = L22_PILOT_ANGLES_DEG


def _pilot_xy(center, pcd, angles):
    r = pcd / 2.0
    return [(center[0] + r * math.cos(math.radians(a)),
             center[1] + r * math.sin(math.radians(a))) for a in angles]


LM_PILOT_XY = _pilot_xy(
    _LM_C, L22_PILOT_PCD_MM, OBIWAN_LM_PILOT_ANGLES_DEG)
UM_PILOT_XY = _pilot_xy(_UM_C, UM_PILOT_PCD_MM, UM_PILOT_ANGLES_DEG)


PAD_XY = list(LM_PILOT_XY)  # concentric with the bores


def recess_cutters():
    """Full front discs (the cutouts already remove the middles, so a
    disc == the flange annulus with no coincident inner rim faces)."""
    return [
        Pos(_LM_C[0], _LM_C[1], (LM_SEAT_Z + THICKNESS_MM + 0.2) / 2.0)
        * Cylinder(LM_RECESS_R, THICKNESS_MM + 0.2 - LM_SEAT_Z),
        Pos(_UM_C[0], _UM_C[1], (UM_SEAT_Z + THICKNESS_MM + 0.2) / 2.0)
        * Cylinder(UM_RECESS_R, THICKNESS_MM + 0.2 - UM_SEAT_Z),
    ]


def deep_pilot_cutters():
    """Insert bores re-cut from the recess floors (the stock
    front-blind bores end 0.8 under the seats). The U22 bores are
    6.2 deep -- the owner's 5.8 inserts + 0.4 settle -- so the rear
    pads stay minimal."""
    out = []
    for px, py in LM_PILOT_XY:
        out.append(m5_insert_bore_cutter(
            (px, py),
            opening_z=LM_SEAT_Z,
            total_depth=LM_BORE_DEPTH_MM,
            opening_side="+z",
            overshoot=0.10,
        ))
    for px, py in UM_PILOT_XY:
        z0 = UM_SEAT_Z - UM_PILOT_DEPTH_MM    # 10.3
        out.append(Pos(px, py, (z0 + UM_SEAT_Z + 0.1) / 2.0)
                   * Cylinder(UM_PILOT_D_MM / 2.0,
                              UM_PILOT_DEPTH_MM + 0.1))
    return out


def pad_relief_cylinders():
    """Punched OUT of the V1L field cutters so the plate keeps
    straight O9.6 pad buttons (z 5.3..6.8) of original material under
    each insert. The 0.6 x 45 rim chamfer comes from the frustum LIP
    of the relief: radius PAD/2 at the face, opening to PAD/2 + 0.6
    by z = face + 0.6, then straight -- adds NOTHING to the plan
    footprint (the chamfer cuts inward from the button edge)."""
    from build123d import Cone

    out = []
    r = PAD_D_MM / 2.0
    for px, py in PAD_XY:
        body = Pos(px, py, (PAD_FACE_Z + PAD_CHAMFER_MM + 7.4) / 2.0) \
            * Cylinder(r, 7.4 - PAD_FACE_Z - PAD_CHAMFER_MM)
        lip = Pos(px, py, PAD_FACE_Z + PAD_CHAMFER_MM / 2.0) * Cone(
            r - PAD_CHAMFER_MM, r, PAD_CHAMFER_MM)
        out.append(body + lip)
    return out


def obiwan_field_cutters():
    """V1L bottom/mids field cutters with the pad reliefs punched out
    (ramp loft included -- the bottom pad straddles the ramp end)."""
    from .proud import top_baffle_nd25fw4_v1l as v1l
    pads = pad_relief_cylinders()
    out = []
    for cutter in v1l.field_cutters():
        for pad in pads:
            cutter -= pad
        out.append(cutter)
    return out


def ceiling_at(x, y):
    """Front-side duct ceiling in Obi-Wan: the recess seats replace the
    front plane inside the flange rings."""
    if math.dist((x, y), _LM_C) <= LM_RECESS_R:
        return LM_SEAT_Z
    if math.dist((x, y), _UM_C) <= UM_RECESS_R:
        return UM_SEAT_Z
    return THICKNESS_MM


# -- self-checks (import-time; cheap arithmetic only) -----------------
def _static_asserts():
    from .proud import top_baffle_nd25fw4_v1l as v1l
    # pads restore a full insert stack: floor wall >= 0.75
    assert LM_SEAT_Z - LM_BORE_DEPTH_MM - PAD_FACE_Z >= 0.75
    # the bore actually swallows the owner's insert + settle room
    assert LM_BORE_DEPTH_MM >= LM_INSERT_L_MM + 0.3
    # UM bores stay blind in the 11.5 plate
    assert UM_SEAT_Z - UM_PILOT_DEPTH_MM - v1l.REAR_MM >= 3.0
    # every pad rims its bore (concentric)
    assert (PAD_D_MM - M5_INSERT_ENTRY_D_MM) / 2.0 >= 1.5
    # EVERY pad's full plan extent clears seam C (x=-5.6) -- the
    # flared/offset version breached the seam plane and got sliced
    # at the piece edge (found on a print render, 2026-07-10)
    for px, _py in PAD_XY:
        assert abs(px - (-5.6)) >= PAD_D_MM / 2.0 + 0.5, px


_static_asserts()
