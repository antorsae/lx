"""Variant B2: mini-LM upper-mid with a constant wall around the 10F.

Flare and chamfer keep the LM tilts but are both tangent to the r=50.83
circle about the UM center: 9.83 mm wall at the D82 cutout and 2.1 mm to
the D97.5 flange at both tangential points. The chamfer runs from the
flare corner (+/-60.65, 391.71; max width 121.3 mm) up to the crescent's
D102.11 arc, which extends past the old prong base to (+/-10.08, 418.18).
See :mod:`lx521_baffle.proud.b` for shared geometry."""

from __future__ import annotations

from .b import (
    B2_EXTRA_DROP_STARTS,
    B2_LEFT_SEGS,
    B2_RIGHT_SEGS,
    variant_outline,
    variant_solid,
)

OUTLINE_B2 = variant_outline(
    right_segs=B2_RIGHT_SEGS,
    left_segs=B2_LEFT_SEGS,
    extra_drop_starts=B2_EXTRA_DROP_STARTS,
)


def gen_step():
    return variant_solid(OUTLINE_B2, "lx521_4_top_baffle_variant_b2")
