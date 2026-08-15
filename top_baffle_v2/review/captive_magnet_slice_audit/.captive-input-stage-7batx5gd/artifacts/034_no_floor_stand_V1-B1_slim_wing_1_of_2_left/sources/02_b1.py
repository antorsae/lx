"""Variant B1: mini-LM upper-mid; the flank is one straight line from
the crescent horn corner (36.813, 432.866) through the max-width point
(83.807, 399.591) to the V-waist at (+/-56.12, 306.5) -- extended to the
horn so the top magnet site lands in the B1 wing too.
See :mod:`lx521_baffle.proud.b` for shared geometry."""

from __future__ import annotations

from .b import (
    B1_EXTRA_DROP_STARTS,
    B1_LEFT_SEGS,
    B1_RIGHT_SEGS,
    variant_outline,
    variant_solid,
)

OUTLINE_B1 = variant_outline(
    right_segs=B1_RIGHT_SEGS,
    left_segs=B1_LEFT_SEGS,
    extra_drop_starts=B1_EXTRA_DROP_STARTS,
)


def gen_step():
    return variant_solid(OUTLINE_B1, "lx521_4_top_baffle_variant_b1")
