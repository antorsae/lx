"""Print-split of the variant-C7 baffle (LM knife-edge taper): the same
four-piece split, seams, dovetails, ducts, and vase piece as B2 -- only
the three LM-section pieces differ (rear taper + T-duct ribs). Any C7
piece is a drop-in replacement for its B2 counterpart.  The captive
magnet-bearing top piece prints front-face-down."""

from __future__ import annotations

from ..assembly import ordered_labeled_compound
from .top_baffle_nd25fw4_b2_split import pieces
from .top_baffle_nd25fw4_c7 import taper_cutters

PRINT_ORIENTATION = "front-face-down"


def pieces_c7() -> dict:
    return pieces(shape_cuts=taper_cutters())


def gen_step():
    return ordered_labeled_compound(
        pieces_c7(), label="lx521_4_top_baffle_nd25fw4_c7_split")
