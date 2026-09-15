"""CONCEPT - 10x4x2 N50 block magnet near the keyed seam.

Replicates the keyed-bottom ring band around theta=-18.6 deg exactly
(band radii, driver recess, buried UM duct torus, seam plane, registration
pin) and cuts the rectangular captive slot with the coupon-style tool
stack rotated for a block:

  * cradle: 10.2 x 2.1 x 4.2 well (0.2/0.1/0.2 clearance on 10x2x4)
  * face skin 0.52 behind the r113.79 cavity datum (pocket outer r113.27)
  * inner skin 0.52 to r110.65 (0.05 over the R110.6 recess wall)
  * gable: 45 deg, ridge tangential, closing the 2.1 radial span (h 1.05)
  * land: slot + 0.6 side margins + gable + 0.52 inner skin

The mirrored wing receiver sits across the released 1.24-mm face
separation.  Every claim is asserted against booleans; the script fails
loudly if any land, skin, duct, recess, seam or pin margin is violated.
Also exports a small flat fit-test coupon printable before committing the
real parts (pause at print z 4.8, drop the block flat, roof closes).
"""
import math
from build123d import (
    Align, Box, Cylinder, Plane, Pos, Rot, Torus, export_stl, extrude,
    Face, Polyline, make_face,
)

# ---- shared geometry (source coordinates of the LM carrier) ---------------
CX, CY = 0.0, 200.981
THICK = 18.3
SEAM_Y = 172.481
R_OD = 113.94          # visible ring OD
R_DATUM = 113.79       # cavity face datum (LM_CORE_R + 0.79)
R_RECESS = 110.6       # driver recess wall
RECESS_Z = 12.3        # recess floor
DUCT_R_C = 108.85      # buried UM arc centreline radius
DUCT_Z_C = 6.7         # measured duct centre height
DUCT_R = 4.18          # duct void radius (measured envelope)
PIN_X, PIN_Z = 108.92, 14.30
PIN_R = 0.8
PIN_ROOT_Y = SEAM_Y - 0.50

SITE_DEG = -18.6
MAG_L, MAG_H, MAG_T = 10.0, 4.0, 2.0          # largo x ancho x alto
SLOT_L, SLOT_H, SLOT_T = 10.2, 4.2, 2.1       # clearances 0.2 / 0.2 / 0.1
FACE_SKIN = 0.52
INNER_SKIN = 0.52
SIDE_MARGIN = 0.6
GABLE_H = SLOT_T / 2.0                        # 45 deg over the radial span
SLOT_Z_LO, SLOT_Z_HI = 13.5, 17.7             # slot floor sits on the 0.6 front land
INTERFACE_GAP = 0.05
RECV_STANDOFF = 0.57                          # 0.05 spacing + 0.52 receiver skin
PAIR_SEP = 1.24                               # released carrier/wing face spacing

EPS = 0.05

th = math.radians(SITE_DEG)
N = (math.cos(th), math.sin(th))              # radial outward
site_plane = Plane(
    origin=(CX + R_DATUM * N[0], CY + R_DATUM * N[1],
            (SLOT_Z_LO + SLOT_Z_HI) / 2.0),
    x_dir=(N[0], N[1], 0.0),                  # +X radial outward
    z_dir=(0.0, 0.0, 1.0),
)

def slot_tools(plane, sign):
    """Cradle + chimney + gable for one owner.

    sign=-1: carrier (material inward of the datum);
    sign=+1: wing receiver (material outward, datum offset by the gap).
    """
    if sign < 0:
        x_lo, x_hi = -(FACE_SKIN + SLOT_T), -FACE_SKIN
    else:
        x_lo, x_hi = RECV_STANDOFF, RECV_STANDOFF + SLOT_T
    cradle = plane * Pos((x_lo + x_hi) / 2.0, 0, 0) * Box(SLOT_T, SLOT_L, SLOT_H)
    chimney = plane * Pos((x_lo + x_hi) / 2.0, 0, -SLOT_H / 2.0 - EPS / 2.0) * Box(
        SLOT_T, SLOT_L, EPS)
    # 45-deg gable: ridge along local Y (tangential), closing the radial span.
    tri_face = make_face(Polyline(
        (x_lo, 0.0, -SLOT_H / 2.0 - EPS),
        (x_hi, 0.0, -SLOT_H / 2.0 - EPS),
        ((x_lo + x_hi) / 2.0, 0.0, -SLOT_H / 2.0 - EPS - GABLE_H),
        (x_lo, 0.0, -SLOT_H / 2.0 - EPS),
    ).edges())
    gable = plane * extrude(tri_face, amount=SLOT_L / 2.0, dir=(0, 1, 0),
                            both=True)
    # magnet seats against the interface-side wall and the front land
    # (source +z = print bottom): its z low edge sits on the slot floor.
    magnet = plane * Pos(
        (x_hi - MAG_T / 2.0) if sign < 0 else (x_lo + MAG_T / 2.0),
        0.0,
        (SLOT_H / 2.0) - (MAG_H / 2.0),
    ) * Box(MAG_T, MAG_L, MAG_H)
    land_x_lo = x_lo - (INNER_SKIN if sign < 0 else 0.0)
    land_x_hi = 0.0 if sign < 0 else x_hi + INNER_SKIN
    if sign > 0:
        land_x_lo = INTERFACE_GAP
    land_z_lo = -SLOT_H / 2.0 - EPS - GABLE_H - INNER_SKIN
    land_z_hi = SLOT_H / 2.0 + SIDE_MARGIN
    land = plane * Pos(
        (land_x_lo + land_x_hi) / 2.0,
        0.0,
        (land_z_lo + land_z_hi) / 2.0,
    ) * Box(land_x_hi - land_x_lo, SLOT_L + 2 * SIDE_MARGIN,
            land_z_hi - land_z_lo)
    return (cradle, chimney, gable), magnet, land

# ---- carrier coupon --------------------------------------------------------
sector = Pos(CX, CY, 0) * (
    Cylinder(R_OD, THICK, align=(Align.CENTER, Align.CENTER, Align.MIN))
    - Cylinder(106.0, THICK, align=(Align.CENTER, Align.CENTER, Align.MIN))
)
# driver recess
sector -= Pos(CX, CY, RECESS_Z) * Cylinder(
    R_RECESS, THICK, align=(Align.CENTER, Align.CENTER, Align.MIN))
# buried UM duct
sector -= Pos(CX, CY, DUCT_Z_C) * Torus(DUCT_R_C, DUCT_R)
# seam plane and angular trim (pie slice -34 deg .. -5 deg about the centre)
sector -= Pos(-200, SEAM_Y, -10) * Box(
    400, 100, 40, align=(Align.MIN, Align.MIN, Align.MIN))
def pie(deg_lo, deg_hi):
    pts = [(CX, CY, 0.0)]
    steps = 24
    for i in range(steps + 1):
        a = math.radians(deg_lo + (deg_hi - deg_lo) * i / steps)
        pts.append((CX + 200 * math.cos(a), CY + 200 * math.sin(a), 0.0))
    pts.append((CX, CY, 0.0))
    face = make_face(Polyline(*pts).edges())
    return Pos(0, 0, -1) * extrude(face, amount=THICK + 2)
wedge = pie(-34.0, -5.0)
sector &= wedge
# registration pin (context)
pin = Pos(PIN_X, PIN_ROOT_Y, PIN_Z) * Rot(X=-90) * Cylinder(
    PIN_R, 2.90, align=(Align.CENTER, Align.CENTER, Align.MIN))
sector += pin

(cradle, chimney, gable), magnet, land = slot_tools(site_plane, -1)

# ---- assertions: everything must hold BEFORE the cut ----------------------
def vol(shape):
    try:
        solids = list(shape.solids())
    except Exception:
        return 0.0
    return sum(s.volume for s in solids)

missing = vol(land - sector)
assert missing < 1e-6, f"captive land not fully solid: missing {missing:.4f} mm3"

duct_solid = Pos(CX, CY, DUCT_Z_C) * Torus(DUCT_R_C, DUCT_R)
recess_solid = Pos(CX, CY, RECESS_Z) * Cylinder(
    R_RECESS, THICK, align=(Align.CENTER, Align.CENTER, Align.MIN))
od_shell = Pos(CX, CY, -1) * (
    Cylinder(R_OD + 5, THICK + 2, align=(Align.CENTER, Align.CENTER, Align.MIN))
    - Cylinder(R_DATUM - 0.02, THICK + 2,
               align=(Align.CENTER, Align.CENTER, Align.MIN)))
seam_guard = Pos(-200, SEAM_Y - 2.0, -10) * Box(
    400, 100, 40, align=(Align.MIN, Align.MIN, Align.MIN))
pin_guard = Pos(PIN_X, PIN_ROOT_Y - 1.5, PIN_Z) * Rot(X=-90) * Cylinder(
    PIN_R + 1.5, 6.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
for label, cutter in (("cradle", cradle), ("chimney", chimney),
                      ("gable", gable)):
    for gname, guard in (("duct", duct_solid), ("recess", recess_solid),
                         ("OD skin", od_shell), ("seam wall", seam_guard),
                         ("pin", pin_guard)):
        overlap = vol(cutter & guard)
        assert overlap < 1e-6, (
            f"{label} breaches {gname}: {overlap:.4f} mm3")

before = vol(sector)
part = sector - cradle - chimney - gable
part = part.clean()
solids = list(part.solids())
assert len(solids) == 1 and part.is_valid, (
    f"carrier coupon invalid after cut: {len(solids)} solids")
removed = before - vol(part)
assert vol(magnet & part) < 1e-6, "seated magnet interferes with the part"
assert vol(magnet - (cradle + chimney)) < 1e-6, (
    "magnet not fully inside the cavity")

# ---- wing coupon -----------------------------------------------------------
wingslab = Pos(CX, CY, 6.8) * (
    Cylinder(R_OD + 6.0, THICK - 6.8,
             align=(Align.CENTER, Align.CENTER, Align.MIN))
    - Cylinder(R_OD + INTERFACE_GAP, THICK - 6.8,
               align=(Align.CENTER, Align.CENTER, Align.MIN)))
wing_plane = Plane(
    origin=(CX + R_OD * N[0], CY + R_OD * N[1],
            (SLOT_Z_LO + SLOT_Z_HI) / 2.0),
    x_dir=(N[0], N[1], 0.0), z_dir=(0.0, 0.0, 1.0))
wsector = wingslab - Pos(-200, SEAM_Y + 5.0, -10) * Box(
    400, 100, 40, align=(Align.MIN, Align.MIN, Align.MIN))
wsector &= wedge
(wc, wch, wg), wmagnet, wland = slot_tools(wing_plane, +1)
wmissing = vol(wland - wsector)
assert wmissing < 1e-6, f"wing land not solid: missing {wmissing:.4f} mm3"
wpart = (wsector - wc - wch - wg).clean()
assert vol(wmagnet & wpart) < 1e-6, "wing magnet interferes"

# pair spacing check: carrier magnet face r vs wing magnet face r
carrier_face_r = R_DATUM - FACE_SKIN
wing_face_r = R_OD + INTERFACE_GAP + RECV_STANDOFF - INTERFACE_GAP
sep = (R_OD + RECV_STANDOFF) - (R_DATUM - FACE_SKIN)
assert abs(sep - PAIR_SEP) < 1e-9, f"pair separation {sep} != {PAIR_SEP}"

# ---- printable fit-test coupon --------------------------------------------
# Print orientation directly: bed at z=0, pause at z=4.8.
FIT_W, FIT_D, FIT_H = 30.0, 12.0, 8.0
fit = Box(FIT_W, FIT_D, FIT_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
fit_slot = Pos(0, 0, 0.6 + SLOT_H / 2.0) * Box(SLOT_L, SLOT_T, SLOT_H)
fit_chim = Pos(0, 0, 0.6 + SLOT_H + EPS / 2.0) * Box(SLOT_L, SLOT_T, EPS)
ftri = make_face(Polyline(
    (0.0, -SLOT_T / 2.0, 0.6 + SLOT_H + EPS),
    (0.0, SLOT_T / 2.0, 0.6 + SLOT_H + EPS),
    (0.0, 0.0, 0.6 + SLOT_H + EPS + GABLE_H),
    (0.0, -SLOT_T / 2.0, 0.6 + SLOT_H + EPS),
).edges())
fgable = extrude(ftri, amount=SLOT_L / 2.0, dir=(1, 0, 0), both=True)
fit = (fit - fit_slot - fit_chim - fgable).clean()
assert len(list(fit.solids())) == 1 and fit.is_valid

# ---- exports ---------------------------------------------------------------
import os
HERE = os.path.dirname(os.path.abspath(__file__))
export_stl(part, os.path.join(HERE, "lm_block_magnet_carrier_coupon.stl"))
export_stl(wpart, os.path.join(HERE, "lm_block_magnet_wing_coupon.stl"))
export_stl(magnet, os.path.join(HERE, "lm_block_magnet_seated.stl"))
export_stl(wmagnet, os.path.join(HERE, "lm_block_magnet_wing_seated.stl"))
export_stl(fit, os.path.join(HERE, "lm_block_magnet_fit_coupon.stl"))

print("ALL ASSERTIONS PASSED")
print(f"  slot removed volume: {removed:.1f} mm3")
print(f"  carrier magnet faces r {R_DATUM - FACE_SKIN - MAG_T:.2f}..{R_DATUM - FACE_SKIN:.2f}")
print(f"  wing magnet faces    r {R_OD + RECV_STANDOFF:.2f}..{R_OD + RECV_STANDOFF + MAG_T:.2f}")
print(f"  paired face separation: {sep:.2f} mm (released contract)")
print(f"  land floor z {SLOT_Z_LO - EPS - GABLE_H - INNER_SKIN:.2f} vs duct roof 10.88")
print(f"  slot z {SLOT_Z_LO}..{SLOT_Z_HI}; pause print z {THICK - SLOT_Z_LO:.1f}")
print("exported: carrier/wing coupons, seated magnets, fit coupon")
