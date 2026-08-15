"""Variant V1: the UM vase thinned to t=11.5 and mounted FRONT-FLUSH:
material occupies z 6.8..18.3, so the MU10, tweeter pair, and the
LM section all share ONE front plane -- the thinning is taken from the
REAR. Proud R6P's D8.2 UM route and R14 rear outlet finish below seam B;
only the shared T O6.0/oval traverses this vase at z~11.5. Short M3 x 3
x O5 pilot inserts remain front-blind. Any rear step at seam B (vs
full-depth B2 mids; none vs
V1L) is on the hidden sculpted side; keys auto-trim on both sides.

The crescent taper re-derives on the 6.8..18.3 slab (same 4.0 clamp
seat position at z 14.3..18.3 as stock, 0.4 tips); standoffs cut to
11.5. The binding cable constraint is the shared TS oval under the
MU10 seat; the UM main is no longer in the vase. Magnets use the two
pause-and-bury wall stations described below."""

from __future__ import annotations

from build123d import Box, Pos

from ..base import THICKNESS_MM, baffle_solid
from .b import (
    STANDARD_MAGNET_Z_MM,
    TWEETER_DROP_MM,
    apply_magnet_base_cavities,
)
from .b2 import OUTLINE_B2

PRINT_ORIENTATION = "front-face-down"

T_FIELD_MM = 11.5
REAR_MM = THICKNESS_MM - T_FIELD_MM   # 6.8: new rear plane of the vase
Y_STEP = 315.95


def field_cutters():
    """One rear-side slab cut over the whole top piece: material keeps
    z REAR_MM..18.3 -> front-flush mounting."""
    return [Pos(0, (Y_STEP + 462.0) / 2, (REAR_MM - 0.7) / 2) * Box(
        160.0, 462.0 - Y_STEP, REAR_MM + 0.7)]


V1_MAGNET_Z_MM = STANDARD_MAGNET_Z_MM
V1_MAGNET_ZC = (V1_MAGNET_Z_MM, V1_MAGNET_Z_MM)


def apply_v1_base_magnets(part):
    """Bury four V1/V1L base magnets at the released site heights.

    Both 45-degree roofs share the front-biased z=15.10 plane and fit inside
    the z=6.8..18.3 vase.  The magnet-free host owns a broad, symmetric rear
    taper thickness band, so neither station receives a local cap, boss, or
    backfill.  Marked poles point OUT along the mirrored
    base-to-attachment normals.
    """
    return apply_magnet_base_cavities(
        part, site_zc=V1_MAGNET_ZC, lower_rear_caps=False,
        name_prefix="v1")


def v1_magnet_free_solid():
    """Return the released V1 body before its four captive base cavities.

    Obi-Wan reuses only the rear-tapered acoustic crescent and has its own M3
    T--UM half-laps.  Keeping this explicit source authority prevents those
    downstream crops from inheriting sealed, functionless V1 magnet voids.
    """
    part = baffle_solid(OUTLINE_B2, TWEETER_DROP_MM,
                        crescent_rear_mm=REAR_MM)
    for c in field_cutters():
        part -= c
    return part


def v1_solid():
    return apply_v1_base_magnets(v1_magnet_free_solid())


def gen_step():
    part = v1_solid()
    part.label = "lx521_4_top_baffle_variant_v1"
    return part
