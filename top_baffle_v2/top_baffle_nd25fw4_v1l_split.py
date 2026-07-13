"""V1L print split: bottom + mids thinned front-flush (see _v1l.py);
combine with the V1 vase for the full ~12 mm front-flush baffle."""

from __future__ import annotations

from build123d import Compound

from top_baffle_nd25fw4_b2_split import pieces
from top_baffle_nd25fw4_cables import UM_V1L_HANDOFF_KEY
from top_baffle_nd25fw4_v1 import all_cutters as v1_vase_cutters
from top_baffle_nd25fw4_v1 import field_cutters as v1_vase_field_cutters
from top_baffle_nd25fw4_v1 import REAR_MM as V1_VASE_REAR_MM
from top_baffle_nd25fw4_v1l import field_cutters


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
        shape_cuts = list(v1_vase_cutters())
    elif only in {"piece_mid_left", "piece_mid_right"}:
        # Both mids own a seam-B male dovetail that projects 6 mm into
        # the vase (y > 315.95).  The V1L field slab stops on the seam,
        # so the low-memory one-piece exporter must also apply the V1
        # vase rear slab to those projecting teeth.  Omitting it left
        # both keys at the stock 18.3-mm depth while every adjacent V1L
        # face is front-flush 11.5 mm (rear plane z=6.8).
        shape_cuts = (list(field_cutters())
                      + list(v1_vase_field_cutters()))
    elif only is not None:
        shape_cuts = list(field_cutters())
    else:
        shape_cuts = list(field_cutters()) + list(v1_vase_cutters())
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
    return pieces(shape_cuts=shape_cuts,
                  magnet_pockets=False,
                  crescent_rear_mm=V1_VASE_REAR_MM,
                  um_handoff_key=UM_V1L_HANDOFF_KEY,
                  only=only,
                  cable_routes=route_subset,
                  cable_y_range=ts_y_range)


def gen_step():
    children = []
    for label, solid in pieces_v1l().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_v1l_split"
    return assembly
