"""CONCEPT v2 - NL8 boss as a thin panel on a self-supporting lattice.

v1 proved the S-ramp removes the ceiling, but a solid ramp is heavy and
closes the view straight down.  Here the same ramp envelope is kept as the
*support surface* and then hollowed into a diagonal mesh: the windows are
diamonds cut through Y, so every window roof is a 45 deg gable rather than a
flat ceiling, and the frame between them carries the panel.

Print direction: z_print = -z_source, so surfaces facing +Z in source face
the bed.  A diamond apex is exactly the limit case and the lattice legs are
steeper still.
"""
import math
from build123d import (
    Box, Cylinder, Plane, Pos, Polyline, RectangleRounded, Rot, Spline,
    export_stl, extrude, fillet, make_face,
)

FOOT_W, FOOT_H = 64.0, 18.3
FOOT_REAR_Z = -150.0
PANEL_T = 2.5                       # thinned receptor, per brief
PANEL_INNER_Z = FOOT_REAR_Z + PANEL_T
PANEL_H = 44.0
NL8_CY, NL8_CUT_D, NL8_SCREW_D, NL8_PITCH = 22.0, 31.0, 3.2, 29.2
SECTION_FRONT_Z = -56.0
RAMP_START_Z = -98.0
CORNER_R, PAD_R = 12.0, 2.5
CELL, BAR = 15.0, 3.5               # finer mesh, thinner struts

foot = Pos(0, FOOT_H / 2, (SECTION_FRONT_Z + FOOT_REAR_Z) / 2) * Box(
    FOOT_W, FOOT_H, SECTION_FRONT_Z - FOOT_REAR_Z)

panel = extrude(
    Plane.XY.offset(FOOT_REAR_Z) * Pos(0, PANEL_H / 2)
    * RectangleRounded(FOOT_W, PANEL_H, CORNER_R), PANEL_T)

ramp_curve = Spline(
    [(FOOT_H, RAMP_START_Z),
     (FOOT_H + (PANEL_H - FOOT_H) * 0.28, RAMP_START_Z - 20.0),
     (FOOT_H + (PANEL_H - FOOT_H) * 0.80, PANEL_INNER_Z + 12.0),
     (PANEL_H, PANEL_INNER_Z)],
    tangents=((0, -1), (0, -1)))
pts = [(round(v.X, 4), round(v.Y, 4)) for v in
       (ramp_curve @ (i / 40.0) for i in range(41))]
pts += [(0.0, PANEL_INNER_Z), (0.0, RAMP_START_Z)]
clean = [pts[0]]
for q in pts[1:]:
    if abs(q[0] - clean[-1][0]) > 1e-4 or abs(q[1] - clean[-1][1]) > 1e-4:
        clean.append(q)
clean.append(clean[0])
ramp = extrude(Plane.YZ * make_face(Polyline(clean)), FOOT_W / 2, both=True)

# --- hollow the ramp into a mesh -------------------------------------
# Diamonds cut through Y: seen from above the boss is open, and each window
# roof is a pair of 45 deg planes instead of a flat lid.
holes = None
half = CELL / 2 - BAR / 2
z0, z1 = FOOT_REAR_Z + PANEL_T + 6.0, RAMP_START_Z - 4.0
x_n = int(FOOT_W / CELL) + 2
z_n = int(abs(z1 - z0) / CELL) + 2
# mesh only the frame above the foot; the foot itself stays solid
MESH_Y0, MESH_Y1 = FOOT_H + 2.0, PANEL_H - 2.0
for iz in range(z_n):
    for ix in range(x_n + 1):
        cz = z0 + iz * CELL + (CELL / 2 if ix % 2 else 0.0)
        cx = -FOOT_W / 2 + (ix - 0.5) * CELL
        if cz > z1 or abs(cx) > FOOT_W / 2 + CELL:
            continue
        d = Pos(cx, (MESH_Y0 + MESH_Y1) / 2, cz) * Rot(0, 45, 0) * Box(
            half * 1.414, MESH_Y1 - MESH_Y0, half * 1.414)
        holes = d if holes is None else holes + d
lattice = ramp - holes if holes is not None else ramp

solid = foot + panel + lattice

pad = extrude(Plane.XY.offset(FOOT_REAR_Z) * Pos(0, NL8_CY)
              * RectangleRounded(NL8_PITCH + 9.0, NL8_PITCH + 9.0, PAD_R), PANEL_T)
solid = solid + pad
solid -= Pos(0, NL8_CY, FOOT_REAR_Z + PANEL_T / 2) * Cylinder(NL8_CUT_D / 2, 60.0)
for sx in (-1, 1):
    for sy in (-1, 1):
        solid -= Pos(sx * NL8_PITCH / 2, NL8_CY + sy * NL8_PITCH / 2,
                     FOOT_REAR_Z + PANEL_T / 2) * Cylinder(NL8_SCREW_D / 2, 60.0)

try:
    solid = fillet([e for e in solid.edges() if e.length > 8.0], 1.5)
except Exception:
    pass

export_stl(solid, "/Users/antor/.claude/jobs/4808081d/tmp/concept/nl8_concept.stl")
print(f"volume {solid.volume/1000:.1f} cm3")
