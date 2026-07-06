"""Variant V0: minimalist UM vase, REAR-side knife bevel (same side
and philosophy as the C7 LM taper: front plane fully intact). The T
flank lanes run at z=3.7 only 4 mm inside the vase walls -- and they
cannot be raised (the 45-deg 10F pilot pins them: between the D82 rim
and the pilot keep-out there is no corridor at any higher z) -- so the
rear bevel is a REGULAR ~4 mm knife band along the flare/chamfer
outline: 18.3 -> ~0.5 over W=4, fading out at the seam-B land
(flush joint to the mids) and blending into the crescent's own rear
taper above y~400. No other keeps needed: the flange seat and M3
pilots are front-side features.

Magnet interface: one D5.4 x 1.0 pin pocket per side, bored vertically
into the REAR face on the lower corner triangle at (+-46, 324) --
plan-clear of the T elbow/step (>=12) and of the bevel band; scarf
attachments add a receiver there plus outline-kink registration.
Checked by test_v0_duct_corridor (rear z-containment)."""

from __future__ import annotations

import math

from build123d import Cylinder, Plane, Polyline, Pos, Wire, loft, make_face, mirror

from top_baffle_nd25fw4 import THICKNESS_MM, baffle_solid
from top_baffle_nd25fw4_b import TWEETER_DROP_MM
from top_baffle_nd25fw4_b2 import OUTLINE_B2

T_EDGE_MM = 0.5
W_SLIDE_MM = 2.8  # capped by the shared
# O6.0 T duct (z=11.5) hugging the left walls at ~1.6
_FLARE = ((38.113, 315.947), (60.654, 391.709))
_CHAMF = ((60.654, 391.709), (10.081, 418.176))
V0_MAGNET_SITES = [(46.0, 324.0)]   # right side; mirrored


def _smoothstep(u):
    u = max(0.0, min(1.0, u))
    return 3 * u * u - 2 * u * u * u


def _d_seg(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy)
                    / (vx * vx + vy * vy)))
    return math.dist(p, (a[0] + t * vx, a[1] + t * vy))


def rear_cut_at(x, y):
    """REAR-side removal depth of the V0 bevel (front plane intact)."""
    if not (315.9 < y < 419.0):
        return 0.0
    d = min(_d_seg((abs(x), y), *_FLARE), _d_seg((abs(x), y), *_CHAMF))
    target = T_EDGE_MM + (THICKNESS_MM - T_EDGE_MM) * _smoothstep(d / W_SLIDE_MM)
    keep = 1.0 - _smoothstep((y - 318.0) / 6.0)      # seam-B land
    keep = max(keep, _smoothstep((y - 400.0) / 8.0))  # crescent blend
    return (THICKNESS_MM - target) * (1.0 - keep)


def _wall_x(y):
    if y <= 391.709:
        return 38.113 + 0.29752 * (y - 315.947)
    return 60.654 - 1.9108 * (y - 391.709)


def _ysection(y, n=14):
    """Horizontal section of the rear bevel cut, right side."""
    xw = _wall_x(y)
    x0, x1 = xw - W_SLIDE_MM - 4.0, xw + 4.0
    pts = [(x0, -0.6)]
    for i in range(n + 1):
        x = x0 + (x1 - x0) * i / n
        pts.append((x, rear_cut_at(min(x, xw - 0.02), y)))
    pts.append((x1, -0.6))
    pts.append((x0, -0.6))
    pl = Plane(origin=(0, y, 0), x_dir=(1, 0, 0), z_dir=(0, -1, 0))
    return pl * make_face(Wire(Polyline(*pts).edges()))


def slide_cutters():
    ys = (316.5, 319.0, 322.0, 326.0, 332.0, 340.0, 350.0, 360.0,
          370.0, 380.0, 388.0, 392.0, 396.0, 400.0, 404.0, 408.0)
    right = loft([_ysection(y) for y in ys], ruled=True)
    return [right, mirror(right, about=Plane.YZ)]


def magnet_pocket_cutters():
    """Vertical D5.4 x 1.0 pin pockets in the REAR face (z 0..1)."""
    return [Pos(sx * x, y, 1.0) * Cylinder(2.7, 4.0)
            for sx in (1.0, -1.0) for x, y in V0_MAGNET_SITES]


def v0_solid():
    part = baffle_solid(OUTLINE_B2, TWEETER_DROP_MM)
    for c in slide_cutters() + magnet_pocket_cutters():
        part -= c
    return part


def gen_step():
    part = v0_solid()
    part.label = "lx521_4_top_baffle_nd25fw4_variant_v0"
    return part
