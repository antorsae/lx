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
and the proud-family R6P cable ducts -- C7 pieces are drop-in
replacements for the three LM-section pieces (bottom + both mids);
piece_top_b2 and all attachments are the same physical parts.

Kept full thickness: the core inside the taper band, the bottom strip
(y<~64, stand-foot / bridge interface -- the taper fades in over
y 52..70), and a recovery fade toward seam B (the cut scales to zero
over y 270..~304) so the joint to the vase piece is flush and the
seam-B integral joint envelopes retain their qualified protected land. The planar duct spans run in
the FRONT half (LM/UM z=12.55, TS z=11.5); the UM route then uses its
intentional R14 rear opening below seam B. The
rear taper clears every duct with no ribs -- verified by
test_c7_duct_corridor (plain z-containment).

Clearances are asserted by test_clearances.py (make check).
"""

from __future__ import annotations

import math

from build123d import Plane, Polyline, Wire, loft, make_face, mirror

from ..geom import (
    point_segment_distance as _d_seg,
    smoothstep01 as _smoothstep,
)
from ..base import THICKNESS_MM, baffle_solid
from ..cables import (
    DUCT_Z,
    UM_HANDOFF_D_MM,
    _um_plan_spline,
)
from .top_baffle_nd25fw4_b import TWEETER_DROP_MM
from .top_baffle_nd25fw4_b2 import OUTLINE_B2

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


def t_of_d(d: float) -> float:
    """Taper law: local thickness vs distance inside the edge."""
    return T_EDGE_MM + (THICKNESS_MM - T_EDGE_MM) * _smoothstep(d / W_TAPER_MM)


def rec_fade(y: float) -> float:
    """Cut multiplier: 1 in the LM body, 0 at the seam-B land and below
    the bottom strip."""
    top = 1.0 - _smoothstep((y - Y_REC0) / (Y_REC1 - Y_REC0))
    bot = _smoothstep((y - Y_BOT0) / (Y_BOT1 - Y_BOT0))
    return top * bot


# Duct corridors the taper must not breach LATERALLY: the slope
# crosses each bore sideways, so containment must hold across the
# duct's full plan width, not just under the centerline (the c7
# mid_right exposed-slot bug). (arc_r about LM center, duct z, bore r,
# side: +1 right / -1 left.)  TS remains the released circular span.  UM now
# leaves its old R119.5 arc to pass the shared 0/60-clock M5 receiver, so its
# C7 clamp follows the qualified wire instead of pretending the tail is still
# circular.
_DUCT_ARCS = ((116.5, 11.5, 3.0, -1.0),)
_LM_C = (0.0, 200.981)


def _sample_um_corridor(spacing_mm: float = 0.35):
    path = _um_plan_spline()
    count = max(64, int(math.ceil(float(path.length) / spacing_mm)))
    return tuple(
        (float((path @ (index / count)).X),
         float((path @ (index / count)).Y))
        for index in range(count + 1)
    )


_UM_CORRIDOR_POINTS = _sample_um_corridor()


def _um_corridor_distance(x: float, y: float) -> float:
    point = (x, y)
    return min(
        _d_seg(point, start, end)
        for start, end in zip(_UM_CORRIDOR_POINTS,
                              _UM_CORRIDOR_POINTS[1:])
    )


def _duct_min_t(x: float, y: float) -> float:
    """Minimum thickness so the rear surface clears every duct's
    circular envelope (+1.6 skin) at this plan point."""
    need = 0.0
    for arc_r, z_d, r, side in _DUCT_ARCS:
        if x * side < 0 or not 45.0 < y < 316.0:
            continue
        o = abs(math.dist((x, y), _LM_C) - arc_r)
        if o >= r + 2.0:
            continue
        # 2.0 skin in the LAW nets >=1.6 on the SOLID (ruled-loft
        # interpolation between stations undercuts the moving
        # envelope by ~0.35 through the pinch)
        drop = math.sqrt(max(r * r - o * o, 0.0)) if o < r else 0.0
        need = max(need, THICKNESS_MM - (z_d - drop - 2.0))
    if x >= 0.0 and 45.0 < y < 316.0:
        r = UM_HANDOFF_D_MM / 2.0
        o = _um_corridor_distance(x, y)
        if o < r + 2.0:
            drop = math.sqrt(max(r * r - o * o, 0.0)) if o < r else 0.0
            need = max(
                need,
                THICKNESS_MM - (DUCT_Z["um"] - drop - 2.0),
            )
    return need


def thickness_at(x: float, y: float) -> float:
    """C7 local plate thickness: taper law clamped by the duct
    corridors (see _duct_min_t)."""
    p = (abs(x), y)
    d = min(_d_seg(p, _FLANK_A, _FLANK_B), _d_seg(p, _CHAMF_A, _CHAMF_B))
    cut = (THICKNESS_MM - t_of_d(d)) * rec_fade(y)
    return max(THICKNESS_MM - cut, _duct_min_t(x, y))


def _flank_pt(y):
    return (76.2 + 0.29752 * y, y)


def _chamf_pt(y):
    f = (y - 256.12) / 59.827
    return (152.401 - 114.288 * f, y)


def _profile_pts(pt, inboard, scale: float, n: int = 24):
    """Closed section polyline in (inboard u, global z): the material
    removed from the rear (z=0) side. Depth sampled from the CLAMPED
    law at the true plan point (duct-corridor aware)."""
    pts = [(-6.0, -0.6), (-6.0, (THICKNESS_MM - T_EDGE_MM) * scale)]
    for i in range(n + 1):
        u = W_TAPER_MM * i / n
        px, py = pt[0] + inboard[0] * u, pt[1] + inboard[1] * u
        base = (THICKNESS_MM - t_of_d(u)) * scale
        cut = min(base, THICKNESS_MM - _duct_min_t(px, py))
        pts.append((u, max(cut, -0.3)))
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
    return pl * make_face(Wire(Polyline(
        *_profile_pts((x, y), (nx, ny), scale)).edges()))


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
                  for y in (52.0, 58.0, 64.0, 72.0, 100.0, 125.0,
                            140.0, 152.0, 164.0, 176.0, 190.0, 210.0,
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
