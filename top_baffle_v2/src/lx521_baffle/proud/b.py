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

Variants live in the sibling B1 and B2 modules.
-------------------------------------------
* B1: flare through the old full-width seam point (57.151, 310) ->
  waist kink at (+/-56.12, 306.5), max width 167.6 mm at y=399.59.
* B2: constant wall tangent to r=50.83 about the UM center -> max width
  121.3 mm at y=391.71, waist (+/-38.1, 315.95) on the LM chamfer line.
"""

from __future__ import annotations

from ..base import OUTLINE, THICKNESS_MM, baffle_solid
from ..magnets import (
    CAVITY_DEPTH_MM,
    CAVITY_DIAMETER_MM,
    FACE_SKIN_MM,
    INNER_SKIN_MM,
    INTERFACE_GAP_MM,
    MAGNET_DEPTH_MM,
    MAGNET_DIAMETER_MM,
    apply_wall_cavity,
)

PRINT_ORIENTATION = "front-face-down"

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

# --- Captive magnet attachment system (neodymium N52 discs D5 x 2) --------
# superimanes.com ref D-05-02-N52 (0.68 kg holding per pair). 12 needed:
# 4 in the base (dot/marked pole faces the receiver), 1 per A shoulder +
# 2 per B1 wing (the receiver's marked pole faces away from the base).
# Both marked/N vectors follow the same base-to-receiver pair axis, so the
# two interface-facing poles are opposite and attract. -------------------
# TWO sites per flank side (minimized), all on piece_top_b2's edge walls:
#   flare-wall site at the wall's BOTTOM end -- the one spot on that wall
#     farther from the UM driver (58.7 mm vs ~51 anywhere mid-wall); holds
#     the A bottom shoulder and the B1 wing's lower end. The T duct
#     parallels the wall ~4 mm inside (the wall's one unavoidable wire
#     proximity -- measured clearances in test_clearances.py).
#   crescent-arc site, as far down-arc as the RECEIVER allows: its bore
#     lives in the narrowing wedge between the arc and the chamfer face
#     that the A top shoulder / B1 wing mate against B2, and the wall at
#     its bottom corner is the binding constraint (1.3 mm); holds the A
#     top shoulder and the wing's top end. (Any point of this wall is
#     ~51 mm from the tweeter center -- the wall is an arc about it --
#     so no driver-distance freedom exists.)
# Both magnets are printed captive behind 0.45 mm plastic skins.  The
# receiver keeps a solid 0.05 mm spacing standoff ahead of its qualified
# 0.45 mm skin, so the nominal magnet-face spacing remains
# 0.45 + 0.05 + 0.45 = 0.95 mm without a visible local air notch. There is
# no glue or post-print access.
# The 45-degree roofs close only after the insertion pause; the outline
# kinks/corners continue to provide shear registration and magnets receive
# no structural-load credit.  Standard/V1/V1L parts print front-face-down.
MAGNET_D_MM = MAGNET_DIAMETER_MM
MAGNET_T_MM = MAGNET_DEPTH_MM
MAG_CAVITY_D_MM = CAVITY_DIAMETER_MM  # D5: 0.1 mm radial clearance
MAG_CAVITY_DEPTH_MM = CAVITY_DEPTH_MM  # D5x2: 0.10 mm axial clearance
MAG_FACE_SKIN_MM = FACE_SKIN_MM
MAG_INNER_SKIN_MM = INNER_SKIN_MM
MAG_INTERFACE_GAP_MM = INTERFACE_GAP_MM
MAG_LAND_DEPTH_MM = MAG_FACE_SKIN_MM + MAG_CAVITY_DEPTH_MM + MAG_INNER_SKIN_MM
MAGNET_QUALIFIED_LAND_WIDTH_MM = MAG_CAVITY_D_MM + 1.20
# One front-biased source plane governs every stock and slim transverse
# station.  Front-face-down export maps it to a 3.20-mm print-space axis;
# the D5.20 cradle begins 0.60 mm behind the acoustic front and the complete
# gable/inner-skin land remains at source Z >= 9.45 mm.  No local rear cap or
# station-shaped taper backfill is permitted.
STANDARD_MAGNET_Z_MM = 15.10
# Exact plan deviations at the released, rounded interface datums.  These are
# analytic bounds against the actual B2 line/ThreePointArc, not allowances:
# see test_standard_interface_plane_facts.  Cutter reach adds a small robust
# overtravel, but only existing host material is removed.
LOWER_INTERFACE_DATUM_DEVIATION_MAX_MM = 0.031572
UPPER_INTERFACE_CURVE_DEVIATION_MAX_MM = 0.134666
UPPER_INTERFACE_CURVE_DEVIATION_AREA_MM2 = 0.430824

# The curved upper tangent loses at most 0.134666 mm across the qualified
# 6.40-mm land.  Recessing the base cavity face by 0.14 mm keeps the complete
# 0.45-mm interface skin inside the unchanged host instead of growing a local
# boss.  The receiver keeps the 0.05-mm spacing allowance as solid material
# behind the shared flush interface, so no local pocket-width relief appears
# on an exterior edge.
BASE_CAVITY_FACE_INSET_MM = (0.0, 0.14)

# (x, y, nx, ny, pin, zc) on the right flank; the left flank is
# mirrored.  Both use the one front-biased source-Z plane.  The bottom site
# lies at the waist-kink end of the flare wall (59.2 mm from the UM center).
# The top site lies on the crescent arc at theta=-69.5 deg, where the broad,
# smooth taper shelf contains the complete captive land without a local boss
# or rear cue.  The protected T route clears both envelopes.  All of these
# facts are checked against the immutable production hosts by
# test_clearances.py (make check).
MAGNET_SITES = [
    (40.0, 322.4, 0.95853, -0.28518, True, STANDARD_MAGNET_Z_MM),
    (17.880, 420.371, 0.35021, -0.93667, True, STANDARD_MAGNET_Z_MM),
]


def _apply_magnets(part, owner: str, site_zc=None,
                   lower_rear_caps: bool = False,
                   name_prefix: str = "stock"):
    """Apply all four stock wall-normal captive stations.

    ``site_zc`` may override the common source-Z axis for controlled probes;
    production stock and slim callers use 15.10 mm at every station.
    ``lower_rear_caps`` remains as a rejected compatibility argument so a
    stale caller cannot silently restore the visible rear box. ``outward``
    always points from base to receiver.
    Both marked/N vectors follow that same pair axis: the base's marked pole
    faces the receiver and the receiver's marked pole faces away from the
    base, including at mirrored sites.
    """
    if lower_rear_caps:
        raise ValueError(
            "local rear magnet caps are forbidden; move the common plane "
            "inside the immutable host")
    result = part
    records = []
    z_by_index = tuple(site_zc) if site_zc is not None else None
    for site_index, site in enumerate(MAGNET_SITES):
        x, y, nx, ny, _pin, released_zc = site
        zc = z_by_index[site_index] if z_by_index is not None else released_zc
        site_key = "lower" if site_index == 0 else "upper"
        for side, sx in (("right", 1.0), ("left", -1.0)):
            px, pnx = sx * x, sx * nx
            normal_length = (pnx * pnx + ny * ny) ** 0.5
            onx, ony = pnx / normal_length, ny / normal_length
            base_inset = (
                BASE_CAVITY_FACE_INSET_MM[site_index]
                if owner == "base" else 0.0)
            tool_face = (
                px - base_inset * onx,
                y - base_inset * ony,
                zc,
            )
            tool_kwargs = {
                "name": f"{name_prefix}_{site_key}_{side}_{owner}",
                "face": tool_face,
                "outward": (onx, ony, 0.0),
                "owner": owner,
                "print_up": (0.0, 0.0, -1.0),
                "bed_datum": (0.0, 0.0, THICKNESS_MM),
            }
            result, tools = apply_wall_cavity(
                result,
                **tool_kwargs,
            )
            records.append(tools)
    return result, tuple(records)


def apply_magnet_base_cavities(part, *, site_zc=None,
                               lower_rear_caps: bool = False,
                               name_prefix: str = "stock"):
    """Bury four base magnets; marked pole points OUT toward the mate."""
    return _apply_magnets(
        part, "base", site_zc=site_zc,
        lower_rear_caps=lower_rear_caps,
        name_prefix=name_prefix,
    )[0]


def apply_magnet_attachment_cavities(part, *, site_zc=None,
                                     lower_rear_caps: bool = False,
                                     name_prefix: str = "stock"):
    """Bury four receivers; marked pole faces away from the mating base."""
    return _apply_magnets(
        part, "receiver", site_zc=site_zc,
        lower_rear_caps=lower_rear_caps,
        name_prefix=name_prefix,
    )[0]


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
