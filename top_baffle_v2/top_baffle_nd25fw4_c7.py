"""Variant C7: the B2 baffle with the LM section rear-tapered to a
knife edge -- full 18.3 mm around the W22, thinning REAR-SIDE (front
face stays a full plane, crescent-taper style) over the last 19 mm
inside the flank/chamfer edges down to a ~0.5 mm feather at the
outline. The ducts sit at FIXED z measured from the rear face, so the
governing constraint is z-INTERVAL containment, not total thickness:
the rear cut over a duct must stay above z_duct - r - skin (the
O8.5/8.6 mains at z=9.15 tolerate a cut of 3.25; the T ducts at z=3.7
tolerate ~0.2, i.e. none).

Shares EVERYTHING with B2: outline, drivers, magnet interface, seams,
and the (rerouted) common cable ducts -- C7 pieces are drop-in
replacements for the three LM-section pieces (bottom + both mids);
piece_top_b2 and all attachments are the same physical parts.

Kept full thickness: the core inside the taper band, the bottom strip
(y<~64, stand-foot / bridge interface -- the taper fades in over
y 52..70), and a recovery fade toward seam B (the cut scales to zero
over y 270..~304) so the joint to the vase piece is flush and the
seam-B dovetails keep their full section. All duct mains now run in
the FRONT half (round-4 routing: LM/UM z=12.55, TS z=11.5), so the
rear taper clears every duct with no ribs -- verified by
test_c7_duct_corridor (plain z-containment).

Clearances are asserted by test_clearances.py (make check).
"""

from __future__ import annotations

import math

from build123d import Plane, Polyline, Wire, loft, make_face, mirror

from top_baffle_nd25fw4 import THICKNESS_MM, baffle_solid
from top_baffle_nd25fw4_b import TWEETER_DROP_MM
from top_baffle_nd25fw4_b2 import OUTLINE_B2

T_EDGE_MM = 0.5          # knife feather (protects the front skin)
W_TAPER_MM = 19.0        # taper band width: keeps the front-half mains
                         # fully covered -- the ducts sit at FIXED z, so the
                         # rear-side cut must stay above z_duct-r-skin
Y_REC0, Y_REC1 = 270.0, 308.0   # seam-B recovery fade of the cut
Y_BOT0, Y_BOT1 = 52.0, 70.0     # bottom fade-in (foot/bridge strip)
# tapered edges (right side; mirrored): lower flank + LM chamfer
_FLANK_A, _FLANK_B = (76.2, 0.0), (152.401, 256.120)
_CHAMF_A, _CHAMF_B = (152.401, 256.120), (38.113, 315.947)
_N_FLANK = (0.95846, -0.28517)   # outward normals
_N_CHAMF = (0.46370, 0.88603)


def _smoothstep(u: float) -> float:
    u = max(0.0, min(1.0, u))
    return 3 * u * u - 2 * u * u * u


def t_of_d(d: float) -> float:
    """Taper law: local thickness vs distance inside the edge."""
    return T_EDGE_MM + (THICKNESS_MM - T_EDGE_MM) * _smoothstep(d / W_TAPER_MM)


def _d_seg(p, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    t = max(0.0, min(1.0, ((p[0] - a[0]) * vx + (p[1] - a[1]) * vy)
                    / (vx * vx + vy * vy)))
    return math.dist(p, (a[0] + t * vx, a[1] + t * vy))


def rec_fade(y: float) -> float:
    """Cut multiplier: 1 in the LM body, 0 at the seam-B land and below
    the bottom strip."""
    top = 1.0 - _smoothstep((y - Y_REC0) / (Y_REC1 - Y_REC0))
    bot = _smoothstep((y - Y_BOT0) / (Y_BOT1 - Y_BOT0))
    return top * bot


def thickness_at(x: float, y: float) -> float:
    """C7 local plate thickness by the taper law (rib NOT included)."""
    p = (abs(x), y)
    d = min(_d_seg(p, _FLANK_A, _FLANK_B), _d_seg(p, _CHAMF_A, _CHAMF_B))
    cut = (THICKNESS_MM - t_of_d(d)) * rec_fade(y)
    return THICKNESS_MM - cut


def _flank_pt(y):
    return (76.2 + 0.29752 * y, y)


def _chamf_pt(y):
    f = (y - 256.12) / 59.827
    return (152.401 - 114.288 * f, y)


def _profile_pts(scale: float, n: int = 12):
    """Closed section polyline in (inboard u, global z): the material
    removed from the rear (z=0) side, depth (18.3 - t(u)) * scale."""
    pts = [(-6.0, -0.6), (-6.0, (THICKNESS_MM - T_EDGE_MM) * scale)]
    for i in range(n + 1):
        u = W_TAPER_MM * i / n
        pts.append((u, (THICKNESS_MM - t_of_d(u)) * scale))
    pts.append((W_TAPER_MM + 2.0, -0.6))
    pts.append((-6.0, -0.6))
    return pts


def _station(pt2d, inboard2d, tangent2d, scale, sign):
    x, y = sign * pt2d[0], pt2d[1]
    nx, ny = sign * inboard2d[0], inboard2d[1]
    tx, ty = sign * tangent2d[0], tangent2d[1]
    if tx * ny - ty * nx < 0:   # keep the plane's y-axis at +Z so the
        tx, ty = -tx, -ty       # profile's depth maps to global z up
    pl = Plane(origin=(x, y, 0.0), x_dir=(nx, ny, 0.0), z_dir=(tx, ty, 0.0))
    return pl * make_face(Wire(Polyline(*_profile_pts(scale)).edges()))


def taper_cutters():
    """One loft per side: flank (with bottom fade-in), corner blend,
    chamfer (with the seam-B recovery fade)."""
    n_fl = (-_N_FLANK[0], -_N_FLANK[1])   # inboard = -outward
    t_fl = (0.28517, 0.95846)
    n_ch = (-_N_CHAMF[0], -_N_CHAMF[1])
    t_ch = (-0.88603, 0.46370)
    a_fl = math.atan2(n_fl[1], n_fl[0])
    a_ch = math.atan2(n_ch[1], n_ch[0])
    while a_ch - a_fl > math.pi:
        a_ch -= 2 * math.pi
    while a_ch - a_fl < -math.pi:
        a_ch += 2 * math.pi
    def corner_sec(f, sign):
        a = a_fl + f * (a_ch - a_fl)
        n = (math.cos(a), math.sin(a))
        return _station(_FLANK_B, n, (n[1], -n[0]), rec_fade(256.12), sign)

    # build the RIGHT side (three RULED lofts -- smooth lofting through
    # the stationary corner fan produces wild solids -- fused into one
    # cutter), then mirror the solid for the left: lofting mirrored
    # section wires directly twists (opposite winding).
    sign = 1.0
    flank_secs = [_station(_flank_pt(y), n_fl, t_fl, rec_fade(y), sign)
                  for y in (52.0, 58.0, 64.0, 72.0, 120.0, 180.0,
                            235.0, 252.0)]
    corner_secs = [corner_sec(f, sign) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
    chamf_secs = [_station(_chamf_pt(y), n_ch, t_ch,
                           max(rec_fade(y), 0.02), sign)
                  for y in (260.0, 268.0, 276.0, 284.0, 292.0,
                            299.0, 304.0)]
    right = (loft(flank_secs + corner_secs[:1], ruled=True)
             + loft(corner_secs, ruled=True)
             + loft(corner_secs[-1:] + chamf_secs, ruled=True))
    return [right, mirror(right, about=Plane.YZ)]


def c7_solid():
    part = baffle_solid(OUTLINE_B2, TWEETER_DROP_MM)
    for cutter in taper_cutters():
        part -= cutter
    return part


def gen_step():
    part = c7_solid()
    part.label = "lx521_4_top_baffle_nd25fw4_variant_c7"
    return part
