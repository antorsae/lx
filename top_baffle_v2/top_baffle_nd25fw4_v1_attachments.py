"""V1-family attachment pieces (shoulders/wings for the 11.5 vase).

Same exact-boolean construction as the B2 family, but both bases are
built with the V1 treatment (crescent on the 6.8..18.3 slab, rear
field cut to z=6.8 above seam B), so every attachment comes out 11.5
deep with the V1-scaled taper carried through -- flush front, flat
rear at z=6.8.

Anchoring: BOTH B2 site plan positions -- lower flare site at
zc=12.5, upper crescent-arc site at zc=14.4, bored INSIDE the
as-tapered wall (thin ~1.2 front walls: internal and no-load). No
lips, no hooks, no bosses. Wing bottoms are trimmed parallel to the
chamfer-extension edge (+2.2), dropping the collapsed B1-B2 wedge.
"""

from __future__ import annotations

import math

from build123d import Box, Compound, Pos

from top_baffle_nd25fw4 import baffle_solid
from top_baffle_nd25fw4_a_comp import OUTLINE_A_COMP
from top_baffle_nd25fw4_attachments import MIN_SOLID_MM3, _box, _two_sides
from top_baffle_nd25fw4_b import (
    A_COMP_CREST_Y,
    MAG_PIN_RECEIVER_DEPTH_MM,
    MAG_RECEIVER_D_MM,
    MAGNET_SITES,
    TWEETER_DROP_MM,
    _magnet_pocket,
)
from top_baffle_nd25fw4_b1 import OUTLINE_B1
from top_baffle_nd25fw4_b2 import OUTLINE_B2
from top_baffle_nd25fw4_v1 import REAR_MM, Y_STEP

V1_SITE_ZC = 12.5  # one pocket height everywhere on V1


def _v1_base(outline):
    part = baffle_solid(outline, TWEETER_DROP_MM, crescent_rear_mm=REAR_MM)
    part -= Pos(0.0, (Y_STEP + 500.0) / 2.0, (REAR_MM - 1.0) / 2.0) * Box(
        400.0, 500.0 - Y_STEP, REAR_MM + 1.0)
    # below seam B the tabs mate the V1L mids (rear z=6.8)
    # overlap 1 past the seam so the two rear cuts share one face
    part -= Pos(0.0, (300.0 + Y_STEP + 1.0) / 2.0, 2.9) * Box(
        400.0, Y_STEP + 1.0 - 300.0, 7.8)
    return part




def v1_attachments() -> dict:
    b2 = _v1_base(OUTLINE_B2)
    keep = _box(303.0, 500.0)
    def _receivers(solid, sites):
        for site, zc in zip(sites, (12.5, 14.4)):
            x, y, nx, ny, _pin, _zc = site
            for sx in (1.0, -1.0):
                solid -= _magnet_pocket(sx * x, y, sx * nx, ny, zc,
                                        MAG_RECEIVER_D_MM,
                                        MAG_PIN_RECEIVER_DEPTH_MM, False)
        return solid

    out: dict = {}
    a_comp = _v1_base(OUTLINE_A_COMP) & keep
    diff = _receivers(a_comp - b2, MAGNET_SITES[:2])
    _two_sides(diff & _box(303.0, A_COMP_CREST_Y),
               "attach_v1a_shoulder_bottom", out)
    _two_sides(diff & _box(A_COMP_CREST_Y, 500.0),
               "attach_v1a_shoulder_top", out)

    b1 = _v1_base(OUTLINE_B1) & keep
    # NO bottom trim: the raw (b1 - b2) wing already has its inner edge
    # exactly on the B2 flank -- flush with the V1L mid -- and tapers
    # naturally to a point where B1 meets B2 (~y=304), with no outline
    # step. A parallel-offset trim here shaved the inner edge by
    # 2.2/sin(27.6 deg) = 4.75 mm in X and opened a gap to the mid; a
    # horizontal cut left a 4.5 mm inward notch. The natural taper has
    # neither. (`keep` = _box(303, 500) already drops the sub-y=303 dust.)
    diff = _receivers(b1 - b2, MAGNET_SITES[:2])
    _two_sides(diff, "attach_v1b1_wing", out)
    return out


def gen_step():
    children = []
    for label, solid in v1_attachments().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_v1_attachments"
    return assembly
