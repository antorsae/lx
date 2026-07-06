"""Variant V1L: the LM section (piece_bottom + both mids) thinned to
t=12.3 and mounted FRONT-FLUSH: material z 6.0..18.3 above the foot
strip, so the whole plate shares the front plane. Enabled by the
round-4 "front-datum" routing (LM z=12.15 -- the 12.3 binder -- UM
z=12.55 outside the W22 ring, shared T duct z=11.5 on the left; the
strip feeders at z=3.7 live in the KEEP).

Keeps at full 18.3: the bottom strip y<70 (fused foot, bridge/support
hardware, cable feeders + z-step), faded out by y=85. The rear step at
seam B to a V1 vase (rear 6.8) is 0.8 -- both on the hidden side.
W22 mounting unchanged: M5 x 6 x O7 heat-sets, floor z=11.3 keeps a
5.3 wall over the new rear. Combine V1L bottom+mids with the V1 vase
for the complete ~12 mm front-flush baffle."""

from __future__ import annotations

from build123d import Box, Plane, Polyline, Pos, Wire, loft, make_face

from top_baffle_nd25fw4 import THICKNESS_MM

T_FIELD_MM = 12.3
REAR_MM = THICKNESS_MM - T_FIELD_MM   # 6.0
Y_KEEP, Y_FADE = 70.0, 85.0
Y_END = 315.95                        # seam B


def _rect(y, z_top):
    pts = [(-160.0, -0.7), (160.0, -0.7), (160.0, z_top),
           (-160.0, z_top), (-160.0, -0.7)]
    pl = Plane(origin=(0, y, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0))
    return pl * make_face(Wire(Polyline(*pts).edges()))


def field_cutters():
    fade = loft([_rect(Y_KEEP, 0.05), _rect(Y_FADE, REAR_MM)], ruled=True)
    body = Pos(0, (Y_FADE + Y_END) / 2, (REAR_MM - 0.7) / 2) * Box(
        320.0, Y_END - Y_FADE, REAR_MM + 0.7)
    return [fade, body]


def gen_step():
    from top_baffle_nd25fw4_v1l_split import pieces_v1l
    from build123d import Compound
    children = []
    for label, solid in pieces_v1l().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_v1l_split"
    return assembly
