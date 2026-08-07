"""Variant V0: minimalist UM vase, REAR-side knife bevel (same side
and philosophy as the C7 LM taper: front plane fully intact). The
shared O6.0 T duct (z=11.5) hugs the LEFT vase walls 5.1 inside, so
the rear bevel is a REGULAR ~2.8 mm knife band along the flare/chamfer
outline: 18.3 -> ~0.5 over W=2.8, fading out at the seam-B land
(flush joint to the mids) and blending into the crescent's own rear
taper above y~400. No other keeps needed: the flange seat and M3
pilots are front-side features.

Magnet interface: one fully buried D5.2 x 2.10 cavity per side.  The legacy
rear-axis XY sites (+-46, 324) were invalid: their centres sat 5.263 mm
outside the B2 flare, leaving even the complete cavity detached.  A first
correction at (+-37.697, 326.470) connected the lands, but its left station
failed the T-route rule and its right station required a circular restore of
the rear knife bevel.  That restore made the pocket position visible from
the exterior and is forbidden.  V0 has no released mate, so both final sites
move to the symmetric inboard locations (+-6.690, 321.290), below the D82
cutout and between the two seam-B integral-joint stations.  The complete
R3.20 x 5.60 captive
lands already exist in the post-bevel host at these locations; applying the
magnet cavities only subtracts internal material and leaves the knife bevel,
front face, and rear face unchanged.  V0 prints front-face-down:
the 0.45-mm rear skin occupies z=0..0.45, a 45-degree conical closure
occupies z=0.45..3.05, and the cylindrical cavity occupies z=3.05..5.15.
This shifts the magnet centre from the former flush-pocket z=1.00 datum to
z=4.10 (+3.10 mm inward) while keeping the +Z axis.  The final sites are
plan-clear of every duct.  No released V0 mate exists;
marked-pole direction is provisionally outward/rear (-Z).
Checked by test_v0_duct_corridor (rear z-containment) and
test_v0_captive_geometry (final BREP land/cavity/skin containment)."""

from __future__ import annotations

import math

from build123d import (
    Plane,
    Polyline,
    Wire,
    loft,
    make_face,
    mirror,
)

from ..geom import (
    point_segment_distance as _d_seg,
    smoothstep01 as _smoothstep,
)
from ..base import THICKNESS_MM, baffle_solid
from .top_baffle_nd25fw4_b import TWEETER_DROP_MM
from .top_baffle_nd25fw4_b2 import OUTLINE_B2
from ..magnets import (
    CAVITY_DIAMETER_MM,
    DEFAULT_SPEC,
    apply_axial_cavity,
)

PRINT_ORIENTATION = "front-face-down"

T_EDGE_MM = 0.5
W_SLIDE_MM = 2.8  # capped by the shared
# O6.0 T duct (z=11.5) hugging the left walls at ~1.6
_FLARE = ((38.113, 315.947), (60.654, 391.709))
_CHAMF = ((60.654, 391.709), (10.081, 418.176))

# The former centres did not intersect the flare at all: their perpendicular
# distance outside the exact B2 line was 5.263036 mm, so even the R2.60
# cavity remained detached by 2.663036 mm and the qualified R3.20 land by
# 2.063036 mm.  There is no released V0 mate whose XY must be preserved.
# The first correction translated both along the exact inward normal just far
# enough to put the complete land inside the existing outline with 0.20 mm of
# Boolean/print connection.  That was geometrically connected but not route-
# safe on the mirrored left: its center was only 2.605 mm from the T route.
# With no released mate, move both stations to the symmetric connected strip
# below the D82 cutout and between the seam-B integral-joint stations.  At
# (+-6.69, 321.29)
# the complete axial land exists in the post-bevel host without a local keep.
# The pair retains at least 1.088 mm to the D82/seam rules and 18.58 mm of
# sampled T-route clearance; pilot and other route constraints are looser.
# This deliberately trades compatibility with the never-released scarf mate
# for an immutable, cue-free exterior.
V0_LEGACY_MAGNET_SITES = {
    "right": (46.0, 324.0),
    "left": (-46.0, 324.0),
}
V0_CAPTIVE_LAND_RADIUS_MM = (
    CAVITY_DIAMETER_MM / 2.0 + DEFAULT_SPEC.side_wall_margin_mm)
V0_CAPTIVE_LAND_OUTLINE_MARGIN_MM = 0.20


def _flare_outward_normal():
    (x0, y0), (x1, y1) = _FLARE
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    return dy / length, -dx / length


def _flare_signed_distance(point):
    """Positive outside the right flare, negative inside the baffle."""
    nx, ny = _flare_outward_normal()
    return ((point[0] - _FLARE[0][0]) * nx
            + (point[1] - _FLARE[0][1]) * ny)


V0_LEGACY_SITE_OUTSIDE_MM = _flare_signed_distance(
    V0_LEGACY_MAGNET_SITES["right"])
V0_LEGACY_CAVITY_DETACHMENT_MM = (
    V0_LEGACY_SITE_OUTSIDE_MM - CAVITY_DIAMETER_MM / 2.0)
V0_LEGACY_LAND_DETACHMENT_MM = (
    V0_LEGACY_SITE_OUTSIDE_MM - V0_CAPTIVE_LAND_RADIUS_MM)
V0_LEGACY_TO_FIRST_SHIFT_MM = (
    V0_LEGACY_SITE_OUTSIDE_MM
    + V0_CAPTIVE_LAND_RADIUS_MM
    + V0_CAPTIVE_LAND_OUTLINE_MARGIN_MM)
_V0_FLARE_NX, _V0_FLARE_NY = _flare_outward_normal()
_V0_FIRST_RIGHT = (
    round(V0_LEGACY_MAGNET_SITES["right"][0]
          - V0_LEGACY_TO_FIRST_SHIFT_MM * _V0_FLARE_NX, 6),
    round(V0_LEGACY_MAGNET_SITES["right"][1]
          - V0_LEGACY_TO_FIRST_SHIFT_MM * _V0_FLARE_NY, 6),
)
V0_FIRST_CORRECTION_MAGNET_SITES = {
    "right": _V0_FIRST_RIGHT,
    "left": (-_V0_FIRST_RIGHT[0], _V0_FIRST_RIGHT[1]),
}
V0_MAGNET_SITES = {
    "right": (6.690, 321.290),
    "left": (-6.690, 321.290),
}
V0_MAGNET_LAND_OUTLINE_CLEARANCE_MM = {
    side: (-_flare_signed_distance((abs(x), y))
           - V0_CAPTIVE_LAND_RADIUS_MM)
    for side, (x, y) in V0_MAGNET_SITES.items()
}
V0_TS_REQUIRED_CENTER_CLEARANCE_MM = (
    V0_CAPTIVE_LAND_RADIUS_MM + 3.30 + 1.50)
V0_KEEPOUT_QUALIFICATION_ALLOWANCE_MM = 0.20
V0_OLD_FLUSH_MAGNET_CENTER_Z_MM = 1.00
V0_CAPTIVE_MAGNET_CENTER_Z_MM = 4.10
V0_MAGNET_INWARD_SHIFT_MM = (
    V0_CAPTIVE_MAGNET_CENTER_Z_MM - V0_OLD_FLUSH_MAGNET_CENTER_Z_MM)


def _ts_corridor_pts():
    """Plan knots of the shared T duct through the crest transition /
    notch entry (mirrored to +x: the band law is |x|-symmetric, so the
    clamp relieves BOTH sides -- the right one purely cosmetically).
    The proud R6P TS oval (zc=10.45 at full drop) runs closer to the
    bevel than the old round duct did; without this clamp the band
    cuts to z~14 across the tube's outboard edge near y=398."""
    from ..cables import _ts_route
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
    from ..cables import ts_section
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


def apply_v0_magnets(part):
    """Apply the two front-down, rear-axis captive V0 cavities.

    V0 has no released mating print, so ``pair_axis=-Z`` records the
    provisional marked-pole-outward/rear convention without claiming a
    validated attraction pair.
    """
    result = part
    records = []
    for side, (x, y) in V0_MAGNET_SITES.items():
        result, tools = apply_axial_cavity(
            result,
            name=f"v0_rear_axis_{side}",
            face=(x, y, 0.0),
            inward=(0.0, 0.0, 1.0),
            pair_axis=(0.0, 0.0, -1.0),
            print_up=(0.0, 0.0, -1.0),
            bed_datum=(0.0, 0.0, THICKNESS_MM),
        )
        records.append(tools)
    return result, tuple(records)


def v0_solid():
    part = baffle_solid(OUTLINE_B2, TWEETER_DROP_MM)
    for c in slide_cutters():
        part -= c
    return apply_v0_magnets(part)[0]


def gen_step():
    part = v0_solid()
    part.label = "lx521_4_top_baffle_nd25fw4_variant_v0"
    return part
