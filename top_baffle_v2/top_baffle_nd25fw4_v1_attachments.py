"""V1-family attachment pieces (shoulders/wings for the 11.5 vase).

Same exact-boolean construction as the B2 family, but both bases are
built with the V1 treatment (crescent on the 6.8..18.3 slab, rear
field cut to z=6.8 above seam B), so every attachment comes out 11.5
deep with the V1-scaled taper carried through -- flush front, flat
rear at z=6.8.

Anchoring: BOTH B2 site plan positions use the common front-biased
zc=15.10 plane and closed pause-and-bury cavities behind 0.45-mm qualified
skins. Receiver faces retain a solid 0.05-mm spacing standoff;
their marked/N vectors follow the base-to-receiver pair axis, so their
interface-facing poles are opposite the base's and attract.  The curved
upper host needs no local relief or fused land; there are no lips, hooks,
attachment-side proud bosses, or pocket-shaped exterior cues. Wing bottoms are trimmed parallel
to the chamfer-extension edge (+2.2), dropping the collapsed B1-B2 wedge.
"""

from __future__ import annotations

from build123d import Box, Compound, Pos

from top_baffle_nd25fw4 import baffle_solid
from top_baffle_nd25fw4_a_comp import OUTLINE_A_COMP
from top_baffle_nd25fw4_attachments import MIN_SOLID_MM3, _box, _two_sides
from top_baffle_nd25fw4_b import (
    A_COMP_CREST_Y,
    TWEETER_DROP_MM,
    apply_magnet_attachment_cavities,
)
from top_baffle_nd25fw4_b1 import OUTLINE_B1
from top_baffle_nd25fw4_b2 import OUTLINE_B2
from top_baffle_nd25fw4_v1 import REAR_MM, V1_MAGNET_ZC, Y_STEP

PRINT_ORIENTATION = "front-face-down"


def _v1_base(outline):
    part = baffle_solid(outline, TWEETER_DROP_MM, crescent_rear_mm=REAR_MM)
    part -= Pos(0.0, (Y_STEP + 500.0) / 2.0, (REAR_MM - 1.0) / 2.0) * Box(
        400.0, 500.0 - Y_STEP, REAR_MM + 1.0)
    # below seam B the tabs mate the V1L mids (rear z=6.8)
    # overlap 1 past the seam so the two rear cuts share one face
    part -= Pos(0.0, (300.0 + Y_STEP + 1.0) / 2.0, 2.9) * Box(
        400.0, Y_STEP + 1.0 - 300.0, 7.8)
    return part




def v1_attachments(*, magnet_cavities: bool = True) -> dict:
    """Return the six slim attachment prints.

    ``magnet_cavities=False`` exposes the exact pre-cavity individual hosts
    for the no-growth/no-visible-pocket regression.  Release exports retain
    the default.
    """
    b2 = _v1_base(OUTLINE_B2)
    keep = _box(303.0, 500.0)
    def _receivers(solid):
        if not magnet_cavities:
            return solid
        return apply_magnet_attachment_cavities(
            solid, site_zc=V1_MAGNET_ZC, lower_rear_caps=False,
            name_prefix="v1")

    out: dict = {}
    a_comp = _v1_base(OUTLINE_A_COMP) & keep
    diff = _receivers(a_comp - b2)
    _two_sides(diff & _box(303.0, A_COMP_CREST_Y),
               "attach_v1a_shoulder_bottom", out)
    _two_sides(diff & _box(A_COMP_CREST_Y, 500.0),
               "attach_v1a_shoulder_top", out)

    b1 = _v1_base(OUTLINE_B1) & keep
    # Blunt the fragile feather where B1 meets B2 (~y=306.5) with a cut
    # PERPENDICULAR to the flank through a point ON the B2 line -- so the
    # inner edge stays exactly on B2 (flush with the V1L mid, no gap) and
    # the bottom becomes a short blunt end normal to the taper (no thin
    # line). A horizontal cut would notch, a flank-parallel offset would
    # shave the inner edge (2.2/sin27.6 = 4.75 mm gap) -- this does
    # neither. Cut point on the B2 flank at y=CUT_Y.
    from build123d import Plane
    CUT_Y = 309.5
    fdir = (0.88592, -0.46375)         # down-flank (LM chamfer dir)
    bx = 38.113 + (315.947 - CUT_Y) / 59.827 * 114.288  # B2 x at CUT_Y
    wing = b1 - b2
    for sgn in (1.0, -1.0):
        # half-space plane through the B2 point, normal = down-flank
        # (mirrored on x); the box fills the removed (down-flank) side
        n = (sgn * fdir[0], fdir[1], 0.0)
        cut = (Plane(origin=(sgn * bx, CUT_Y, 9.15), z_dir=n)
               * Pos(0, 0, 30.0) * Box(140.0, 140.0, 60.0))
        wing -= cut
    diff = _receivers(wing)
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
