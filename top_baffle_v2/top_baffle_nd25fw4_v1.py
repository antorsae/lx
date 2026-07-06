"""Variant V1: the UM vase thinned to t=11.5 and mounted FRONT-FLUSH:
material occupies z 6.8..18.3, so the 10F, the tweeter pair, and the
LM section all share ONE front plane -- the thinning is taken from the
REAR, enabled by the round-4 front-datum routing (UM O7.8 planar at
z=12.55, shared T O6.0 at z=11.5; UM exit opening z=12 through the
10F basket window zone) and the short M3 x 3 x O5 pilot inserts
(global). Any rear step at seam B (vs full-depth B2/C7 mids; none vs
V1L) is on the hidden sculpted side; keys auto-trim on both sides.

The crescent taper re-derives on the 6.8..18.3 slab (same 4.0 clamp
seat position at z 14.3..18.3 as stock, 0.4 tips); standoffs cut to
11.5. Magnets per side: one vertical rear-face pin pocket at
(+-46, 324) plus two wall pin pockets (zc=12.5 lower flare site,
zc=14.4 upper crescent-arc site). Binding constraint for 11.5: the
O7.8 UM vase main (z=12.55 needs 8.65..16.45 + skins)."""

from __future__ import annotations

import math

from build123d import Box, Cylinder, Pos

from top_baffle_nd25fw4 import THICKNESS_MM, baffle_solid
from top_baffle_nd25fw4_b import TWEETER_DROP_MM
from top_baffle_nd25fw4_b2 import OUTLINE_B2

T_FIELD_MM = 11.5
REAR_MM = THICKNESS_MM - T_FIELD_MM   # 6.8: new rear plane of the vase
Y_STEP = 315.95
V1_MAGNET_SITES = [(46.0, 324.0)]


def field_cutters():
    """One rear-side slab cut over the whole top piece: material keeps
    z REAR_MM..18.3 -> front-flush mounting."""
    return [Pos(0, (Y_STEP + 462.0) / 2, (REAR_MM - 0.7) / 2) * Box(
        160.0, 462.0 - Y_STEP, REAR_MM + 0.7)]


def magnet_pocket_cutters():
    """Vertical D5.4 x 1.0 pin pockets in the new rear face (z=6.8),
    plus BOTH wall pin pockets at the B2 site plan positions: the lower
    flare site at zc=12.5 (left: 2.3 wall to the ts funnel behind a
    1.0 pocket) and the upper crescent-arc site at zc=14.4 -- inside
    the as-tapered wall (1.6 floor, ~1.2 front: thin but internal, and
    pin magnets carry no load)."""
    from top_baffle_nd25fw4_b import (MAG_PIN_BASE_DEPTH_MM,
                                      MAG_POCKET_D_MM, MAGNET_SITES,
                                      _magnet_pocket)
    cutters = [Pos(sx * x, y, REAR_MM + 1.0) * Cylinder(2.7, 4.0)
               for sx in (1.0, -1.0) for x, y in V1_MAGNET_SITES]
    for site, zc in ((MAGNET_SITES[0], 12.5), (MAGNET_SITES[1], 14.4)):
        x, y, nx, ny, _pin, _zc = site
        for sx in (1.0, -1.0):
            cutters.append(_magnet_pocket(sx * x, y, sx * nx, ny, zc,
                                          MAG_POCKET_D_MM,
                                          MAG_PIN_BASE_DEPTH_MM, True))
    return cutters


def magnet_boss_adds():
    """No bosses: the top site sits at zc=14.4, inside the as-tapered
    wall (floor 1.6 over the local rear, 1.2 front) -- thin but
    internal walls beat any visible pad on the sculpted rear."""
    return []


def all_cutters():
    return field_cutters() + magnet_pocket_cutters()


def v1_solid():
    part = baffle_solid(OUTLINE_B2, TWEETER_DROP_MM,
                        crescent_rear_mm=REAR_MM)
    for a in magnet_boss_adds():
        part += a
    for c in all_cutters():
        part -= c
    return part


def gen_step():
    part = v1_solid()
    part.label = "lx521_4_top_baffle_nd25fw4_variant_v1"
    return part
