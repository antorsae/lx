"""V1 print split: piece_top thinned to 11.5 (see _v1.py); seams and
duct sections remain identical to B2.  Only the TS centerline gets a smooth
0.20-mm local captive-land detour, so the set still mixes with B2/C7
bottom/mids.  The magnet-bearing top piece prints front-face-down."""

from __future__ import annotations

from build123d import Compound

from top_baffle_nd25fw4_b2_split import pieces
from top_baffle_nd25fw4_cables import TS_ROUTE_CAPTIVE
from top_baffle_nd25fw4_v1 import (
    REAR_MM,
    apply_v1_base_magnets,
    field_cutters,
)

PRINT_ORIENTATION = "front-face-down"


def pieces_v1() -> dict:
    result = pieces(shape_cuts=field_cutters(), magnet_cavities=False,
                    crescent_rear_mm=REAR_MM,
                    ts_route_key=TS_ROUTE_CAPTIVE)
    result["piece_top_b2"] = apply_v1_base_magnets(result["piece_top_b2"])
    return result


def gen_step():
    children = []
    for label, solid in pieces_v1().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_v1_split"
    return assembly
