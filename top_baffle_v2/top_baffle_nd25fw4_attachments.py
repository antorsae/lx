"""Attachment pieces that turn the B2 print set into variants A-comp or B1.

Exact booleans against the B2 solid:

  attach_a_shoulder_top_left/right = (A-comp minus B2), above the crest
      the region between the vertical flank and B2's chamfer/arc/corner,
      apex reaching the notch corner (+/-10.08, 418.18); includes the top
      edge extension out to +/-60.654.

  attach_a_shoulder_bottom_left/right = (A-comp minus B2), below the crest
      the wedge between the vertical flank and B2's flare, running down to
      the LM chamfer-extension line (~y=304.15). Its lower ~12 mm bonds to
      the mids, so it also splints seam B.

  attach_b1_wing_left/right = B1 minus B2
      long wings (~127 mm) restoring B1's full mini-LM flare; they bond
      along B2's flare/chamfer/arc flank and reach ~9 mm below seam B.

All six are flat 18.3 mm extrusions (print front face up, no supports).
Glue with zero designed clearance -- these are edge-bonded, not inserted.
"""

from __future__ import annotations

from build123d import Box, Compound, Pos

from top_baffle_nd25fw4 import baffle_solid
from top_baffle_nd25fw4_a_comp import OUTLINE_A_COMP
from top_baffle_nd25fw4_b import (
    A_COMP_CREST_Y,
    TWEETER_DROP_MM,
    magnet_attachment_cutters,
)
from top_baffle_nd25fw4_b1 import OUTLINE_B1
from top_baffle_nd25fw4_b2 import OUTLINE_B2

MIN_SOLID_MM3 = 500.0  # ignore boolean dust


def _box(y0: float, y1: float):
    return Pos(0.0, (y0 + y1) / 2.0, 9.15) * Box(400.0, y1 - y0, 40.0)


def _two_sides(diff, prefix: str, out: dict) -> None:
    solids = [s for s in diff.solids() if s.volume > MIN_SOLID_MM3]
    if len(solids) != 2:
        raise RuntimeError(
            f"{prefix}: expected 2 solids, got {len(solids)} "
            f"(volumes {[round(s.volume) for s in diff.solids()]})"
        )
    for s in solids:
        side = "left" if s.bounding_box().center().X < 0 else "right"
        out[f"{prefix}_{side}"] = s


def attachments() -> dict:
    b2 = baffle_solid(OUTLINE_B2, TWEETER_DROP_MM)
    # All attachments live above y=303; trimming there discards any hairline
    # boolean slivers along the collinear chamfer-extension edges.
    keep = _box(303.0, 500.0)
    out = {}

    pockets = magnet_attachment_cutters()

    def _pocketed(diff):
        for cutter in pockets:  # D10x3 magnet pockets on the mating walls
            diff -= cutter
        return diff

    a_diff = _pocketed((baffle_solid(OUTLINE_A_COMP, TWEETER_DROP_MM) - b2) & keep)
    # top and bottom shoulders touch only at the crest tangent point; split
    # there. The bottom shoulder's 16.6 deg feather toward the crest is
    # blunted at y=390 (~0.45 mm wide, ~1 perimeter -- printable flat);
    # only a 1.7 x 0.45 hairline at the exact tangent point remains open.
    _two_sides(a_diff & _box(A_COMP_CREST_Y, 500.0), "attach_a_shoulder_top", out)
    _two_sides(a_diff & _box(303.0, 390.0), "attach_a_shoulder_bottom", out)

    # The wing feathers to zero at B1's waist kink (y=306.5) where its two
    # boundary lines become coincident; blunt it at y=307.8 (~2.8 mm wide).
    b1_diff = _pocketed((baffle_solid(OUTLINE_B1, TWEETER_DROP_MM) - b2) & _box(307.8, 500.0))
    _two_sides(b1_diff, "attach_b1_wing", out)
    return out


def gen_step():
    children = []
    for label, solid in attachments().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_attachments"
    return assembly
