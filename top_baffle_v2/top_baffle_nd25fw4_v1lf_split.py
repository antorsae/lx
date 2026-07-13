"""R6F V1LF core-only STEP entry point.

The historical four-piece, full-outline V1LF was intentionally removed.
The print split is intrinsic: one LM carrier and one UM carrier,
registered by two rounded M3 half-lap ears.  Every non-driver function is an add-on
from ``top_baffle_nd25fw4_v1lf_attachments``.

In no-floor state the immutable stock-bridge XY interface sits in one
front-flush solid web fused into the LM carrier; it is not a third print
part. Floor state has no such web.
"""

from __future__ import annotations

from build123d import Compound

from top_baffle_nd25fw4 import STAND_FOOT
from top_baffle_nd25fw4_cables import ROUTING_PROFILE
from top_baffle_nd25fw4_v1lf import core_parts

if ROUTING_PROFILE != "v1lf":
    raise RuntimeError(
        "V1LF requires LX_ROUTING_PROFILE=v1lf (R6F); proud-family "
        "routing is physically incompatible with the flush collars"
    )


def pieces_v1lf():
    """Compatibility name used by the STL exporter: the two core parts."""
    return core_parts()


def gen_step():
    children = []
    for label, solid in pieces_v1lf().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    state = "floor" if STAND_FOOT else "no_floor_fused_solid_web"
    assembly.label = f"lx521_v1lf_r6f_core_2piece_{state}"
    return assembly
