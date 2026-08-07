"""R6F Obi-Wan core-only STEP entry point.

The historical four-piece, full-outline Obi-Wan was intentionally removed.
The mandatory core is intrinsically two-carrier: one LM carrier and one UM
carrier, registered by two rounded M3 half-lap ears.  The LM remains
available as this monolithic carrier; a mutually-exclusive top/bottom
hidden-keyed print option is exported by
``obiwan_lm_split``. Every non-driver function is an add-on
from ``obiwan_attachments``.

In no-floor state the immutable stock-bridge XY interface sits in one
front-flush solid web fused into the LM carrier; it is not a third print
part. Floor state has no such web.
"""

from __future__ import annotations

from ..assembly import ordered_labeled_compound
from ..base import STAND_FOOT
from ..cables import ROUTING_PROFILE
from .carriers import core_parts

if ROUTING_PROFILE != "obiwan":
    raise RuntimeError(
        "Obi-Wan requires LX_ROUTING_PROFILE=obiwan (R6F); proud-family "
        "routing is physically incompatible with the flush collars"
    )


def pieces_obiwan():
    """Compatibility name used by the STL exporter: the two core parts."""
    return core_parts()


def gen_step():
    state = "floor" if STAND_FOOT else "no_floor_fused_solid_web"
    return ordered_labeled_compound(
        pieces_obiwan(), label=f"lx521_obiwan_r6f_core_2piece_{state}")
