"""V0 print split: same seams/ducts as B2 -- only piece_top differs
(the minimalist front-slide vase). Mix with B2 or C7 bottom/mids."""

from __future__ import annotations

from build123d import Compound

from top_baffle_nd25fw4_b2_split import pieces
from top_baffle_nd25fw4_v0 import magnet_pocket_cutters, slide_cutters


def pieces_v0() -> dict:
    return pieces(shape_cuts=list(slide_cutters()) + magnet_pocket_cutters())


def gen_step():
    children = []
    for label, solid in pieces_v0().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_v0_split"
    return assembly
