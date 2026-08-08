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
from ..floor_bend import (
    BEND_CENTERLINE_LENGTH_MM,
    centerline_arc_length as floor_bend_arc_length,
    centerline_controls as floor_bend_centerline_controls,
)
from ..geom import smootherstep01
from .v1l import REAR_MM as V1L_REAR_MM
from .v1l import T_FIELD_MM as V1L_T_FIELD_MM
from .v1 import apply_v1_base_magnets
from .v1 import field_cutters as v1_vase_field_cutters
from .v1 import REAR_MM as V1_VASE_REAR_MM
from .v1l import field_cutters

PRINT_ORIENTATION = "front-face-down"

# Floor state: piece_bottom's rear-thickness transition.
#
# The no-floor bottom is a flat plate, so its ramp can sit anywhere below
# the D190 aperture and the short y=78..96 smoothstep is unremarkable.  The
# floor bottom is different: the plate does not end at the stand, it TURNS
# into it.  Landing the ramp on the Option-B vertical tangent therefore put
# full depth at the station where the arc has only started to turn, and the
# section still read as a swelling right at the join.
#
# The transition instead runs on ONE quintic in PATH LENGTH along the whole
# combined profile: s=0 at the slim field 2 mm below the seam-A dovetail
# root, down the flat plate to the vertical tangent, and on along the bend
# centreline as it sweeps, reaching full 18.3 mm exactly at the HORIZONTAL
# tangent where the arc has finished turning and the foot begins.  Because
# it is a single smootherstep over one parameter there is no knee anywhere,
# and in particular the value and slope are continuous at the vertical
# tangent by construction rather than by matching two stitched ramps.
#
# The 2 mm of retained slim field matters: the seam-A dovetails and their
# root stay at the thin section, so the mating faces the shared mid pieces
# see are the same in both stand states.
FLOOR_RAMP_SLIM_MARGIN_MM = 2.0
FLOOR_RAMP_SLIM_Y_MM = SEAM_A_Y - FLOOR_RAMP_SLIM_MARGIN_MM
FLOOR_RAMP_VERTICAL_TANGENT_Y_MM = float(
    floor_bend_centerline_controls()[-1][1])
FLOOR_RAMP_FLAT_LENGTH_MM = (
    FLOOR_RAMP_SLIM_Y_MM - FLOOR_RAMP_VERTICAL_TANGENT_Y_MM)
FLOOR_RAMP_BEND_LENGTH_MM = BEND_CENTERLINE_LENGTH_MM
FLOOR_RAMP_TOTAL_LENGTH_MM = (
    FLOOR_RAMP_FLAT_LENGTH_MM + FLOOR_RAMP_BEND_LENGTH_MM)
# ~2 mm between ruled stations keeps the lofted facet error under 0.01 mm,
# an order below the 0.06 mm the short no-floor ramp already accepts.
FLOOR_RAMP_SECTIONS = 22


def floor_ramp_thickness_mm(path_length_mm: float) -> float:
    """Wall thickness at path length ``s`` measured from the slim field."""
    return V1L_T_FIELD_MM + V1L_REAR_MM * smootherstep01(
        float(path_length_mm) / FLOOR_RAMP_TOTAL_LENGTH_MM)


def floor_ramp_rear_cut_mm(path_length_mm: float) -> float:
    """Depth removed from the rear/concave face at path length ``s``."""
    return V1L_REAR_MM - (
        floor_ramp_thickness_mm(path_length_mm) - V1L_T_FIELD_MM)


def floor_ramp_plate_ease(fraction: float) -> float:
    """Rear-cut fraction for :func:`~lx521_baffle.proud.v1l.field_cutters`.

    ``fraction`` is 0 at the cutter's deep end (the vertical tangent) and 1
    at the slim end, so the plate station is ``s = flat * (1 - fraction)``.
    """
    return floor_ramp_rear_cut_mm(
        FLOOR_RAMP_FLAT_LENGTH_MM * (1.0 - float(fraction))) / V1L_REAR_MM


def floor_ramp_wall_thickness_law(parameter: float) -> float:
    """Bend-wall thickness at Option-B cubic parameter ``u``.

    ``u=1`` is the vertical tangent, so the path length there is the flat
    plate run and grows with the arc travelled toward ``u=0``.
    """
    travelled = floor_bend_arc_length(parameter)
    return floor_ramp_thickness_mm(
        FLOOR_RAMP_FLAT_LENGTH_MM
        + (FLOOR_RAMP_BEND_LENGTH_MM - travelled))


def v1l_field_cutters():
    """V1L rear-thickness cutters for the active stand state."""
    if not STAND_FOOT:
        return field_cutters()
    return field_cutters(
        y_full=FLOOR_RAMP_VERTICAL_TANGENT_Y_MM,
        y_slim=FLOOR_RAMP_SLIM_Y_MM,
        ease=floor_ramp_plate_ease,
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
                    floor_wall_thickness_law=(
                        floor_ramp_wall_thickness_law if STAND_FOOT else None),
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
