"""Variant V1: the UM vase thinned to t=11.5 and mounted FRONT-FLUSH:
material occupies z 6.8..18.3, so the MU10, tweeter pair, and the
LM section all share ONE front plane -- the thinning is taken from the
REAR. Proud R6P's D8.2 UM route and R14 rear outlet finish below seam B;
only the shared T O6.0/oval traverses this vase at z~11.5. Short M3 x 3
x O5 pilot inserts remain front-blind. Any rear step at seam B (vs
full-depth B2/C7 mids; none vs
V1L) is on the hidden sculpted side; keys auto-trim on both sides.

The crescent taper re-derives on the 6.8..18.3 slab (same 4.0 clamp
seat position at z 14.3..18.3 as stock, 0.4 tips); standoffs cut to
11.5. The binding cable constraint is the shared TS oval under the
MU10 seat; the UM main is no longer in the vase. Magnets use the two
pause-and-bury wall stations described below."""

from __future__ import annotations

from build123d import Box, Pos

from top_baffle_nd25fw4 import THICKNESS_MM, baffle_solid
from top_baffle_nd25fw4_b import (
    TWEETER_DROP_MM,
    apply_magnet_base_cavities,
)
from top_baffle_nd25fw4_b2 import OUTLINE_B2

PRINT_ORIENTATION = "front-face-down"

T_FIELD_MM = 11.5
REAR_MM = THICKNESS_MM - T_FIELD_MM   # 6.8: new rear plane of the vase
Y_STEP = 315.95


def field_cutters():
    """One rear-side slab cut over the whole top piece: material keeps
    z REAR_MM..18.3 -> front-flush mounting."""
    return [Pos(0, (Y_STEP + 462.0) / 2, (REAR_MM - 0.7) / 2) * Box(
        160.0, 462.0 - Y_STEP, REAR_MM + 0.7)]


V1_MAGNET_ZC = (12.5, 14.4)


def apply_v1_base_magnets(part):
    """Bury four V1/V1L base magnets at the released site heights.

    Both 45-degree roofs fit inside the z=6.8..18.3 vase without the local
    rear cap needed by the standard zc=5.0 lower station.  The upper curved
    site still receives the common <=0.134666-mm tangent-plane land boss,
    paired to the attachment relief; the lower straight site is locally
    trimmed to its released datum.  Marked poles point OUT along the
    mirrored base-to-attachment normals.
    """
    return apply_magnet_base_cavities(
        part, site_zc=V1_MAGNET_ZC, lower_rear_caps=False,
        name_prefix="v1")


def v1_solid():
    part = baffle_solid(OUTLINE_B2, TWEETER_DROP_MM,
                        crescent_rear_mm=REAR_MM)
    for c in field_cutters():
        part -= c
    return apply_v1_base_magnets(part)


def gen_step():
    part = v1_solid()
    part.label = "lx521_4_top_baffle_nd25fw4_variant_v1"
    return part
