"""V0 print split: same seams/ducts as B2 -- only piece_top differs
(the minimalist front-slide vase). Mix with B2 or C7 bottom/mids.  The
magnet-bearing top piece prints front-face-down."""

from __future__ import annotations

from build123d import Compound

from top_baffle_nd25fw4_b2_split import pieces
from top_baffle_nd25fw4_v0 import apply_v0_magnets, slide_cutters

PRINT_ORIENTATION = "front-face-down"


def pieces_v0() -> dict:
    # V0 owns only its two rear-axis captive stations.  Suppress the generic
    # B2 wall-normal sites: those four inherited pockets never had a V0 mate
    # and were an accidental consequence of the shared split builder.
    result = pieces(shape_cuts=list(slide_cutters()), magnet_cavities=False)
    result["piece_top_b2"] = apply_v0_magnets(result["piece_top_b2"])[0]
    return result


def gen_step():
    children = []
    for label, solid in pieces_v0().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_v0_split"
    return assembly
