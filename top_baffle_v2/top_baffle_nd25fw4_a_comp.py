"""Variant A-comp: straight-sided tower, buildable from the B2 print pieces.

Per the compromise sketch: vertical flanks at +/-60.654 mm (tangent to B2's
flare crest) run from the extended top edge (y=453.457) down to the LM
chamfer-extension line at (+/-60.654, ~304.15), which continues -- collinear
with the mids' outline -- out to the LM section. The tweeter section sits at
the B2 drop (14.857 mm) so piece_top_b2 carries the crescent unchanged.

A-comp minus B2 is exactly four shoulder attachments (top/bottom per side),
split at the crest tangent point (see top_baffle_nd25fw4_attachments.py).
"""

from __future__ import annotations

from top_baffle_nd25fw4_b import (
    A_COMP_EXTRA_DROP_STARTS,
    A_COMP_LEFT_SEGS,
    A_COMP_RIGHT_SEGS,
    variant_outline,
    variant_solid,
)

OUTLINE_A_COMP = variant_outline(
    right_segs=A_COMP_RIGHT_SEGS,
    left_segs=A_COMP_LEFT_SEGS,
    extra_drop_starts=A_COMP_EXTRA_DROP_STARTS,
)


def gen_step():
    return variant_solid(OUTLINE_A_COMP, "lx521_4_top_baffle_nd25fw4_variant_a_comp")
