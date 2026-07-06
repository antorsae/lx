"""V1L print split: bottom + mids thinned front-flush (see _v1l.py);
combine with the V1 vase for the full ~12 mm front-flush baffle."""

from __future__ import annotations

from build123d import Compound

from top_baffle_nd25fw4_b2_split import pieces
from top_baffle_nd25fw4_v1 import all_cutters as v1_vase_cutters
from top_baffle_nd25fw4_v1 import REAR_MM as V1_VASE_REAR_MM
from top_baffle_nd25fw4_v1l import field_cutters


def pieces_v1l() -> dict:
    """The complete thin baffle: V1L field on the LM section plus the
    V1 vase cuts on the top piece (one build, all four pieces)."""
    return pieces(shape_cuts=list(field_cutters()) + list(v1_vase_cutters()),
                  magnet_pockets=False,
                  crescent_rear_mm=V1_VASE_REAR_MM)


def gen_step():
    children = []
    for label, solid in pieces_v1l().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_v1l_split"
    return assembly
