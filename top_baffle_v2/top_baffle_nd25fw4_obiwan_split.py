"""R6F Obi-Wan core-only STEP entry point.

The historical four-piece, full-outline Obi-Wan was intentionally removed.
The mandatory core is intrinsically two-carrier: one LM carrier and one UM
carrier, registered by two rounded M3 half-lap ears.  The LM remains
available as this monolithic carrier; a mutually-exclusive top/bottom
hidden-keyed print option is exported by
``top_baffle_nd25fw4_obiwan_lm_split``. Every non-driver function is an add-on
from ``top_baffle_nd25fw4_obiwan_attachments``.

In no-floor state the immutable stock-bridge XY interface sits in one
front-flush solid web fused into the LM carrier; it is not a third print
part. Floor state has no such web.
"""

from __future__ import annotations

from build123d import Compound

from top_baffle_nd25fw4 import STAND_FOOT
from top_baffle_nd25fw4_cables import ROUTING_PROFILE
from top_baffle_nd25fw4_obiwan import core_parts

if ROUTING_PROFILE != "obiwan":
    raise RuntimeError(
        "Obi-Wan requires LX_ROUTING_PROFILE=obiwan (R6F); proud-family "
        "routing is physically incompatible with the flush collars"
    )


def pieces_obiwan():
    """Compatibility name used by the STL exporter: the two core parts."""
    return core_parts()


def gen_step():
    children = []
    for label, solid in pieces_obiwan().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    state = "floor" if STAND_FOOT else "no_floor_fused_solid_web"
    assembly.label = f"lx521_obiwan_r6f_core_2piece_{state}"
    return assembly
