"""Shared builder for the B-family variants of the modified LX521.4 top
baffle: mini-LM upper-mid section + tweeter pair lowered toward the 10F.

Common to all B variants
------------------------
* UM center aligned to the stock LX521.4 baffle (y=366.081, i.e. 5.857
  below the V2 drawing); the tweeter section moves with it, so the total
  drop from the drawing is 9.0 + 5.857 = 14.857 mm. The 9.0 mm component
  is the front-face clearance: the LOWER tweeter faces forward (stock
  arrangement), its D104 faceplate sharing the front plane with the 10F's
  D97.5 flange; axis spacing 102.84 mm vs the 100.75 mm contact limit
  leaves a 2.1 mm edge gap. Scallop-to-flange 14.1 mm; scallop-to-D82 web
  21.9 mm (both preserved by the joint UM+T shift).
* Clean V-waist: the LM chamfer edge continues straight past its drawing
  endpoint (57.151, 305.981) until it meets the mini-vase flare, so there
  are no steps or stubs at the waist. Both waist edges are parallel to the
  LM section's edges. Below y=306 the outline is exactly the V2 drawing.

Variants (see top_baffle_nd25fw4_b1/_b2.py)
-------------------------------------------
* B1: flare through the old full-width seam point (57.151, 310) ->
  waist kink at (+/-56.12, 306.5), max width 167.6 mm at y=399.59.
* B2: constant wall tangent to r=50.83 about the UM center -> max width
  121.3 mm at y=391.71, waist (+/-38.1, 315.95) on the LM chamfer line.
"""

from __future__ import annotations

import math

from build123d import Cylinder, Pos, Rot

from top_baffle_nd25fw4 import OUTLINE, THICKNESS_MM, baffle_solid

# 9.0 (front-face T/UM clearance) + 5.857 (stock UM alignment)
TWEETER_DROP_MM = 14.857

# B1 flank: the wing now extends UP to the crescent horn corner
# (36.813, 432.866) -- one straight line from the horn to the max-width
# point (no prong notch), so the top magnet site's receiver lands in the
# B1 wing as well. Waist kinks as before (flare vs extended LM chamfer).
B1_RIGHT_SEGS = [
    ("L", (36.813, 432.866), (83.807, 399.591)),
    ("L", (83.807, 399.591), (56.116, 306.523)),
    ("L", (56.116, 306.523), (152.401, 256.120)),
]
B1_LEFT_SEGS = [
    ("L", (-152.401, 256.155), (-56.125, 306.553)),
    ("L", (-56.125, 306.553), (-83.807, 399.591)),
    ("L", (-83.807, 399.591), (-36.811, 432.879)),
]
B1_EXTRA_DROP_STARTS = {(36.813, 447.723), (-36.811, 439.046)}

# B2: constant wall around the 10F. Flare and chamfer are both tangent to
# the r=50.83 circle about the UM center (9.83 mm wall at the D82 cutout,
# 2.1 mm to the D97.5 flange at both tangential points, same tilts as the
# LM edges). The chamfer ends on the crescent's D102.11 arc, extended past
# the old prong base down to (+/-10.08, 418.18); max width 121.3 mm at
# y=391.71, waist (+/-38.1, 315.95) on the LM chamfer line.
B2_RIGHT_SEGS = [
    ("A", (36.813, 432.866), (24.570, 423.478), (10.081, 418.176)),
    ("L", (10.081, 418.176), (60.654, 391.709)),
    ("L", (60.654, 391.709), (38.113, 315.947)),
    ("L", (38.113, 315.947), (152.401, 256.120)),
]
B2_LEFT_SEGS = [
    ("L", (-152.401, 256.155), (-38.122, 315.977)),
    ("L", (-38.122, 315.977), (-60.654, 391.709)),
    ("L", (-60.654, 391.709), (-10.081, 418.176)),
    ("A", (-10.081, 418.176), (-24.570, 423.478), (-36.811, 432.879)),
]
# The prong verticals are absorbed by the extended crescent arc.
B2_EXTRA_DROP_STARTS = {(36.813, 447.723), (-36.811, 439.046)}

# A-comp: variant A rebuilt from the B2 pieces plus four shoulder pieces.
# Straight vertical flanks at +/-60.654 (tangent to B2's flare crest) run
# from an extended top edge (y=453.457) down to the LM chamfer-extension
# line, which they meet at (+/-60.654, ~304.15) -- collinear with the mids'
# outline, so everything below is untouched. Per side, A-comp minus B2 is
# a TOP shoulder (vertical vs B2's chamfer/arc/corner, apex at the notch
# corner (+/-10.08, 418.18)) and a BOTTOM shoulder (vertical vs B2's flare,
# feathering at the crest), split exactly at the crest y=391.709.
A_COMP_RIGHT_SEGS = [
    ("L", (36.483, 453.457), (60.654, 453.457)),    # top edge extension
    ("L", (60.654, 453.457), (60.654, 304.147)),    # vertical flank
    ("L", (60.654, 304.147), (152.401, 256.120)),   # LM chamfer extension
]
A_COMP_LEFT_SEGS = [
    ("L", (-152.401, 256.155), (-60.654, 304.182)),
    ("L", (-60.654, 304.182), (-60.654, 453.457)),
    ("L", (-60.654, 453.457), (-36.468, 453.457)),
]
# Prong verticals, corner beziers, and top-edge stubs are interior now.
A_COMP_EXTRA_DROP_STARTS = {
    (36.483, 468.314),   # right top edge
    (49.177, 468.314),   # right corner bezier
    (36.813, 447.723),   # right prong vertical
    (-36.811, 439.046),  # left prong vertical
    (-36.811, 447.736),  # left corner bezier
    (-49.161, 468.314),  # left top edge
}
A_COMP_CREST_Y = 391.709  # split plane between top and bottom shoulders

# --- Magnet attachment system (neodymium N52 discs D5 x 2) ---------------
# superimanes.com ref D-05-02-N52 (0.68 kg holding per pair). 12 needed:
# 4 in the base (dot/marked pole OUT), 1 per A shoulder + 2 per B1 wing
# (marked pole IN). ------------------------------------------------------
# TWO sites per flank side (minimized), all on piece_top_b2's edge walls:
#   flare-wall site at the wall's BOTTOM end -- the one spot on that wall
#     farther from the UM driver (58.7 mm vs ~51 anywhere mid-wall); holds
#     the A bottom shoulder and the B1 wing's lower end. Pin pocket only
#     1.0 deep (the 2.0 magnet sits 1.0 proud); the T duct parallels the
#     wall ~4 mm inside (the wall's one unavoidable wire proximity --
#     measured clearances in test_clearances.py).
#   crescent-arc site, as far down-arc as the RECEIVER allows: its bore
#     lives in the narrowing wedge between the arc and the chamfer face
#     that the A top shoulder / B1 wing mate against B2, and the wall at
#     its bottom corner is the binding constraint (1.3 mm); holds the A
#     top shoulder and the wing's top end. (Any point of this wall is
#     ~51 mm from the tweeter center -- the wall is an arc about it --
#     so no driver-distance freedom exists.)
# All magnets are FLUSH: a 2.0 mm disc in a 2.0 mm pocket on BOTH faces,
# so the two magnets meet face-to-face when the attachment mates the
# base (no proud pin, no buried receiver). The outline kinks/corners do
# the shear registration (a flush disc gives no shear key).
# Pockets are bored at mid-thickness with the axis normal to the wall.
MAGNET_D_MM = 5.0
MAGNET_T_MM = 2.0
MAG_POCKET_D_MM = 5.4            # glued magnet, snug
MAG_RECEIVER_D_MM = 5.4          # flush magnet, snug (no proud pin)
MAG_FLUSH_DEPTH_MM = 2.0         # magnet 2.0 -> FLUSH
MAG_PIN_BASE_DEPTH_MM = 2.0      # magnet 2.0 -> FLUSH
MAG_PIN_RECEIVER_DEPTH_MM = 2.0  # magnet 2.0 -> FLUSH (was a deep receiver)

# (x, y, nx, ny, pin, zc) on the right flank; the left flank is
# mirrored. zc is the bore height (mid-thickness unless the crescent
# rear taper forces it forward). Bottom site at the waist-kink end of
# the flare wall (59.2 from the UM center). Top site on the crescent
# arc at theta=-69.5 deg, the balance point of the apex wedge: the
# receiver's bottom corner keeps 1.3 mm to the shoulder/wing's chamfer
# mating face, while the rear taper still leaves ~12.2 mm of wall (bore
# raised to zc=10.7: ~1.7 mm behind the pocket floor). T ducts pass
# 2.2 mm below the pocket floor; 21.7 from the clamp hole, 57.2 from
# the UM center. All checked by test_clearances.py (make check).
MAGNET_SITES = [
    (40.0, 322.4, 0.95853, -0.28518, True, 5.0),  # zc low: the raised
    # vase-side TS duct (z=11.5) passes above this wall site
    (17.880, 420.371, 0.35021, -0.93667, True, 10.7),
]


def _magnet_pocket(x, y, nx, ny, zc, dia, depth, into_base: bool):
    """Cylindrical pocket at height zc, axis along the wall normal.
    ``into_base`` bores against the normal (into the B2 piece); otherwise
    along it (into the attachment)."""
    ang = math.degrees(math.atan2(ny, nx))
    # span exactly [-depth, +1] along the outward normal (1 mm overshoot
    # OUTSIDE the wall only; the bore is exactly `depth` deep)
    length = depth + 1.0
    shift = (1.0 - depth) / 2.0 if into_base else (depth - 1.0) / 2.0
    return (
        Pos(x, y, zc)
        * Rot(Z=ang)
        * Rot(Y=90)
        * Pos(0, 0, shift)
        * Cylinder(dia / 2.0, length)
    )


def _mirrored_sites():
    for x, y, nx, ny, pin, zc in MAGNET_SITES:
        yield (x, y, nx, ny, pin, zc)
        yield (-x, y, -nx, ny, pin, zc)


def magnet_base_cutters():
    """Pockets for piece_top_b2 (glued base magnets; pin sites shallower)."""
    return [
        _magnet_pocket(
            x, y, nx, ny, zc, MAG_POCKET_D_MM,
            MAG_PIN_BASE_DEPTH_MM if pin else MAG_FLUSH_DEPTH_MM,
            into_base=True,
        )
        for x, y, nx, ny, pin, zc in _mirrored_sites()
    ]


def magnet_attachment_cutters():
    """Pockets for the attachments (deeper, wider receivers at pin sites)."""
    return [
        _magnet_pocket(
            x, y, nx, ny, zc,
            MAG_RECEIVER_D_MM if pin else MAG_POCKET_D_MM,
            MAG_PIN_RECEIVER_DEPTH_MM if pin else MAG_FLUSH_DEPTH_MM,
            into_base=False,
        )
        for x, y, nx, ny, pin, zc in _mirrored_sites()
    ]


def _dropped(pt):
    return (pt[0], pt[1] - TWEETER_DROP_MM)


def variant_outline(*, right_segs, left_segs, extra_drop_starts=()):
    """Splice mini-LM flanks into the exact variant-A outline and translate
    the retained tweeter section down by the tweeter drop.

    ``right_segs``/``left_segs`` are explicit outline segments in post-drop
    coordinates. Segments are matched against OUTLINE by their exact start
    coordinates; every anchor must be consumed, so an edit to OUTLINE that
    silently misses an anchor raises instead of producing a self-
    intersecting outline.
    """
    pending = {
        (-57.149, 371.938),   # neck top transition arc
        (-57.110, 374.027),   # neck upper straight
        (-57.048, 409.062),   # flare
        (-60.918, 439.046),   # shelf
        (-57.151, 306.016),   # neck lower arc (trailing)
        (60.921, 439.046),    # flare
        (57.046, 409.062),    # neck upper straight
        (57.111, 374.071),    # neck transition arc
        (57.151, 371.938),    # neck straight
        (57.151, 305.981),    # chamfer (absorbed into the waist line)
    } | set(extra_drop_starts)
    spliced_right = spliced_left = False
    outline = []
    for seg in OUTLINE:
        start, end = seg[1], seg[-1]
        if start in pending:
            pending.discard(start)
            continue
        if start == (36.813, 439.046) and end == (60.921, 439.046):
            outline.extend(right_segs)
            spliced_right = True
            continue
        if start == (-152.401, 256.155) and end == (-57.151, 306.016):
            outline.extend(left_segs)
            spliced_left = True
            continue
        if min(pt[1] for pt in seg[1:]) >= 439.0:  # tweeter section: shift down
            outline.append((seg[0], *[_dropped(pt) for pt in seg[1:]]))
            continue
        outline.append(seg)
    if pending or not (spliced_right and spliced_left):
        raise RuntimeError(
            "variant_outline: outline anchors out of sync -- unmatched drop "
            f"starts {sorted(pending)}, spliced right/left = "
            f"{spliced_right}/{spliced_left}"
        )
    return outline


def variant_solid(outline, label):
    part = baffle_solid(outline, tweeter_drop_mm=TWEETER_DROP_MM)
    part.label = label
    return part
