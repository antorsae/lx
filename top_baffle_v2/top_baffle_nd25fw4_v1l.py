"""Variant V1L: the LM section (piece_bottom + both mids) thinned to
t=11.5 and mounted FRONT-FLUSH: material z 6.8..18.3 above the foot
strip, so the whole plate shares the front plane. Enabled by the
round-4 "front-datum" routing (LM O8.2 z=12.55 -- the 11.5 binder --
UM z=12.55 outside the W22 ring, shared T duct z=11.5 on the left;
the strip feeders at z=3.7/9.5 live in the KEEP).

Keeps at full 18.3: the bottom strip (fused foot, bridge/support
hardware incl. the washer seats behind the top pass-throughs at
(+-20, 70) + 5 mm margin, cable feeders + z-step). The thickness
transition is a SMOOTHSTEP ramp from full at y=78 to the thin field
at y=96 -- ending 10 mm short of the D190 cutout edge (y=105.98). The rear plane MATCHES the V1 vase (both 6.8): NO step at seam B.
W22 mounting: M5 x 5.8 x O6.3 heat-sets, floor z=11.5 keeps a
4.7 wall over the new rear. Combine V1L bottom+mids with the V1 vase
for the complete ~12 mm front-flush baffle."""

from __future__ import annotations

from build123d import Box, Plane, Polyline, Pos, Wire, loft, make_face

from top_baffle_nd25fw4 import THICKNESS_MM

T_FIELD_MM = 11.5
REAR_MM = THICKNESS_MM - T_FIELD_MM   # 6.8
RAMP_Y0, RAMP_Y1 = 78.0, 96.0
Y_END = 315.95                        # seam B


def _rect(y, z_top):
    pts = [(-160.0, -0.7), (160.0, -0.7), (160.0, z_top),
           (-160.0, z_top), (-160.0, -0.7)]
    pl = Plane(origin=(0, y, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0))
    return pl * make_face(Wire(Polyline(*pts).edges()))


def _smoothstep(u):
    u = max(0.0, min(1.0, u))
    return 3 * u * u - 2 * u * u * u


def field_cutters():
    """Sigmoid thickness ramp (smoothstep sections lofted ruled) from
    full 18.3 at y=RAMP_Y0 to the thin field at y=RAMP_Y1."""
    n = 9
    secs = []
    for i in range(n + 1):
        y = RAMP_Y0 + (RAMP_Y1 - RAMP_Y0) * i / n
        z_top = max(0.05, REAR_MM * _smoothstep(i / n))
        secs.append(_rect(y, z_top))
    fade = loft(secs, ruled=True)
    body = Pos(0, (RAMP_Y1 + Y_END) / 2, (REAR_MM - 0.7) / 2) * Box(
        320.0, Y_END - RAMP_Y1, REAR_MM + 0.7)
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
