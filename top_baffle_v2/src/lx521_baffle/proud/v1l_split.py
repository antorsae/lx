"""V1L print split: bottom + mids thinned front-flush (see _v1l.py);
combine with the V1 vase for the full ~12 mm front-flush baffle.  Its TS
centreline shares V1's smooth 0.60-mm local captive-land detour while the
duct section remains unchanged.  The captive-magnet top piece prints
front-face-down."""

from __future__ import annotations

from ..assembly import ordered_labeled_compound
from ..base import STAND_FOOT
from .b2_split import SEAM_A_Y, pieces
from ..cables import (
    TS_ROUTE_CAPTIVE,
    UM_V1L_HANDOFF_KEY,
)
from ..floor_bend import centerline_controls as floor_bend_centerline_controls
from ..geom import smootherstep01
from .v1 import apply_v1_base_magnets
from .v1 import field_cutters as v1_vase_field_cutters
from .v1 import REAR_MM as V1_VASE_REAR_MM
from .v1l import field_cutters

PRINT_ORIENTATION = "front-face-down"

# Floor state: piece_bottom's rear-thickness transition.
#
# The no-floor bottom is a flat plate, so its ramp can sit anywhere below
# the D190 aperture and the short y=78..96 smoothstep is unremarkable.  The
# floor bottom is different: everything below the Option-B vertical tangent
# is the stand, so an 18-mm ramp ending 4 mm above it reads as a sudden
# sigmoid swelling right where the plate meets its stand.  Run the same
# transition over the entire span the stand leaves free instead -- from
# 2 mm below the seam-A dovetail root down to the tangent -- on the quintic
# whose flat endpoints leave the thin field and rejoin the arc's vertical
# rear face without a knee at either end.
#
# The 2 mm of retained slim field matters: the seam-A dovetails and their
# root stay at the thin section, so the mating faces the shared mid pieces
# see are the same in both stand states.
FLOOR_RAMP_SLIM_MARGIN_MM = 2.0
FLOOR_RAMP_SLIM_Y_MM = SEAM_A_Y - FLOOR_RAMP_SLIM_MARGIN_MM
FLOOR_RAMP_FULL_DEPTH_Y_MM = float(floor_bend_centerline_controls()[-1][1])
# ~2 mm between ruled stations keeps the lofted facet error under 0.01 mm,
# an order below the 0.06 mm the short no-floor ramp already accepts.
FLOOR_RAMP_SECTIONS = 22


def v1l_field_cutters():
    """V1L rear-thickness cutters for the active stand state."""
    if not STAND_FOOT:
        return field_cutters()
    return field_cutters(
        y_full=FLOOR_RAMP_FULL_DEPTH_Y_MM,
        y_slim=FLOOR_RAMP_SLIM_Y_MM,
        ease=smootherstep01,
        sections=FLOOR_RAMP_SECTIONS,
        min_cut_mm=0.0,
    )


def pieces_v1l(only: str | None = None,
                include_cables: bool = True) -> dict:
    """The complete thin baffle: V1L field on the LM section plus the
    V1 vase cuts on the top piece.  ``only`` forwards the generic split
    builder's low-memory single-piece mode.  ``include_cables=False``
    exists for split/depth regression probes only; manufacturing exports
    always retain the default complete piece-local route subset."""
    # In single-piece mode, omit shapes and high-face-count cable cutters
    # that are provably disjoint from the requested split region.  This
    # is the serial export path used to preserve the macOS free-memory
    # floor; full-assembly callers still receive the complete build.
    if only == "piece_top_b2":
        shape_cuts = list(v1_vase_field_cutters())
    elif only in {"piece_mid_left", "piece_mid_right"}:
        # Both mids own a seam-B male dovetail that projects 6 mm into
        # the vase (y > 315.95).  The V1L field slab stops on the seam,
        # so the low-memory one-piece exporter must also apply the V1
        # vase rear slab to those projecting teeth.  Omitting it left
        # both keys at the stock 18.3-mm depth while every adjacent V1L
        # face is front-flush 11.5 mm (rear plane z=6.8).
        shape_cuts = (list(v1l_field_cutters())
                      + list(v1_vase_field_cutters()))
    elif only is not None:
        shape_cuts = list(v1l_field_cutters())
    else:
        shape_cuts = list(v1l_field_cutters()) + list(v1_vase_field_cutters())
    route_subset = {
        "piece_bottom": None,
        "piece_mid_left": {"ts"},
        "piece_mid_right": {"um"},
        "piece_top_b2": {"ts"},
    }.get(only)
    if not include_cables:
        route_subset = set()
    ts_y_range = {
        # Include the full dovetail depth plus a ruled-loft station on
        # either side.  The TS helper retains its original global grid,
        # so these partial cutters are identical to the full cutter over
        # the solid region they subtract.
        "piece_bottom": (-1.0e6, 130.0),
        "piece_mid_left": (110.0, 327.0),
        "piece_mid_right": None,
        "piece_top_b2": (310.0, 1.0e6),
    }.get(only)
    result = pieces(shape_cuts=shape_cuts,
                    magnet_cavities=False,
                    crescent_rear_mm=V1_VASE_REAR_MM,
                    um_handoff_key=UM_V1L_HANDOFF_KEY,
                    only=only,
                    cable_routes=route_subset,
                    cable_y_range=ts_y_range,
                    ts_route_key=TS_ROUTE_CAPTIVE)
    if "piece_top_b2" in result:
        result["piece_top_b2"] = apply_v1_base_magnets(
            result["piece_top_b2"])
    return result


def gen_step():
    return ordered_labeled_compound(
        pieces_v1l(), label="lx521_4_top_baffle_v1l_split")
