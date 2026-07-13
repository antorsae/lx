"""Variant V0: minimalist UM vase, REAR-side knife bevel (same side
and philosophy as the C7 LM taper: front plane fully intact). The
shared O6.0 T duct (z=11.5) hugs the LEFT vase walls 5.1 inside, so
the rear bevel is a REGULAR ~2.8 mm knife band along the flare/chamfer
outline: 18.3 -> ~0.5 over W=2.8, fading out at the seam-B land
(flush joint to the mids) and blending into the crescent's own rear
taper above y~400. No other keeps needed: the flange seat and M3
pilots are front-side features.

Magnet interface: one D5.2 x 2.2 FLUSH pocket per side, bored vertically
into the REAR face on the lower corner triangle at (+-46, 324) --
plan-clear of every duct (>=1.5 past radii) and of the bevel band; scarf
attachments add a receiver there plus outline-kink registration.
Checked by test_v0_duct_corridor (rear z-containment)."""

from __future__ import annotations

import math

from build123d import Cylinder, Plane, Polyline, Pos, Wire, loft, make_face, mirror

from top_baffle_nd25fw4 import THICKNESS_MM, baffle_solid
from top_baffle_nd25fw4_b import (
    MAG_FLUSH_DEPTH_MM,
    MAG_POCKET_D_MM,
    TWEETER_DROP_MM,
)
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


def _ts_corridor_pts():
    """Plan knots of the shared T duct through the crest transition /
    notch entry (mirrored to +x: the band law is |x|-symmetric, so the
    clamp relieves BOTH sides -- the right one purely cosmetically).
    The proud R6P TS oval (zc=10.45 at full drop) runs closer to the
    bevel than the old round duct did; without this clamp the band
    cuts to z~14 across the tube's outboard edge near y=398."""
    from top_baffle_nd25fw4_cables import _ts_route
    return [(abs(x), y) for x, y in _ts_route() if 378.0 <= y <= 424.0]


_TS_CLAMP_PTS = None


def _duct_clamp_cut(x, y):
    """Max REAR removal over the T-duct corridor: cut <= zc - drop - 1.8
    (LAW skin 1.8 nets >=1.4 on the ruled-loft solid, the same
    law-vs-solid allowance as C7's 2.0 -> 1.6). Fades 2.2 past the
    tube edge."""
    global _TS_CLAMP_PTS
    if _TS_CLAMP_PTS is None:
        _TS_CLAMP_PTS = _ts_corridor_pts()
    from top_baffle_nd25fw4_cables import ts_section
    p = (abs(x), y)
    d = min(_d_seg(p, a, b)
            for a, b in zip(_TS_CLAMP_PTS, _TS_CLAMP_PTS[1:]))
    w2, h2, zc = ts_section(y)
    if d >= w2 + 3.0:
        return THICKNESS_MM
    drop = h2 * math.sqrt(max(1.0 - min(d / w2, 1.0) ** 2, 0.0))
    base = zc - drop - 1.8
    if d > w2 + 1.5:  # 1.5 shelf past the edge, then smooth fade-out
        f = _smoothstep((d - w2 - 1.5) / 1.5)
        return base + f * (THICKNESS_MM - base)
    return base


def rear_cut_at(x, y):
    """REAR-side removal depth of the V0 bevel (front plane intact),
    clamped along the T-duct corridor (crest transition/notch entry)."""
    if not (315.9 < y < 419.0):
        return 0.0
    d = min(_d_seg((abs(x), y), *_FLARE), _d_seg((abs(x), y), *_CHAMF))
    target = T_EDGE_MM + (THICKNESS_MM - T_EDGE_MM) * _smoothstep(d / W_SLIDE_MM)
    keep = 1.0 - _smoothstep((y - 318.0) / 6.0)      # seam-B land
    keep = max(keep, _smoothstep((y - 400.0) / 8.0))  # crescent blend
    cut = (THICKNESS_MM - target) * (1.0 - keep)
    if 378.0 < y < 419.0:
        cut = min(cut, _duct_clamp_cut(x, y))
    return cut


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
          370.0, 380.0, 386.0, 390.0, 392.0, 394.0, 396.0, 398.0,
          400.0, 402.0, 404.0, 406.0, 408.0)
    right = loft([_ysection(y) for y in ys], ruled=True)
    return [right, mirror(right, about=Plane.YZ)]


def magnet_pocket_cutters():
    """Vertical D5.2 x 2.2 pockets in the rear face for D5x2 magnets.

    The 0.2 mm axial allowance is an adhesive bed; hold each magnet flush
    with the rear face while it cures instead of bottoming it.
    """
    overshoot = 0.5
    length = MAG_FLUSH_DEPTH_MM + overshoot
    center_z = (MAG_FLUSH_DEPTH_MM - overshoot) / 2.0
    return [Pos(sx * x, y, center_z) * Cylinder(
                MAG_POCKET_D_MM / 2.0, length)
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
