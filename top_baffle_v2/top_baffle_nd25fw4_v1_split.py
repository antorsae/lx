"""V1 print split: piece_top thinned to 11.5 (see _v1.py); seams and
ducts identical to B2 -- mixes with B2 or C7 bottom/mids."""

from __future__ import annotations

from build123d import Compound

from top_baffle_nd25fw4_b2_split import pieces
from top_baffle_nd25fw4_v1 import REAR_MM, all_cutters, magnet_boss_adds


def pieces_v1() -> dict:
    return pieces(shape_cuts=all_cutters(),
                  shape_adds=magnet_boss_adds(), magnet_pockets=False,
                  crescent_rear_mm=REAR_MM)


def gen_step():
    children = []
    for label, solid in pieces_v1().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_v1_split"
    return assembly
