"""V1LF print split: V1L (thin bottom + mids + V1 vase) with FLUSH
driver mounting -- front flange recesses for the U22REX/P-SL and
MU10RB-SL, deepened insert bores, and rear pad buttons under the six
W22-ring inserts (uncut plate material via pad-punched field cutters;
see the flush module). Routing round-5 keeps every duct clear of the
recess rings, so the four pieces split cleanly on the same seams."""

from __future__ import annotations

from build123d import Compound

from top_baffle_nd25fw4_b2_split import DOVETAILS_B_V1LF, pieces
from top_baffle_nd25fw4_flush import (
    deep_pilot_cutters,
    recess_cutters,
    v1lf_field_cutters,
)
from top_baffle_nd25fw4_v1 import REAR_MM as V1_VASE_REAR_MM
from top_baffle_nd25fw4_v1 import all_cutters as v1_vase_cutters


def pieces_v1lf() -> dict:
    """All four flush pieces in one build: pad-relieved V1L field cuts,
    the V1 vase cuts, both flange recesses, and full-depth pilot
    re-bores -- applied to the parent solid BEFORE the seam split, so
    tabs and pockets inherit the recess shaves consistently."""
    cuts = (list(v1lf_field_cutters()) + list(v1_vase_cutters())
            + recess_cutters() + deep_pilot_cutters())
    # seam-B keys flipped (tabs DOWN from the vase, mid pockets below
    # the seam) so neither flange-recess seat lands on a dovetail
    # joint -- see DOVETAILS_B_V1LF in the split module
    return pieces(shape_cuts=cuts, magnet_pockets=False,
                  crescent_rear_mm=V1_VASE_REAR_MM,
                  seam_b_dovetails=DOVETAILS_B_V1LF,
                  seam_b_tabs_up=False)


def gen_step():
    children = []
    for label, solid in pieces_v1lf().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_v1lf_split"
    return assembly
