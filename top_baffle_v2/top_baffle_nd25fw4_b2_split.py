"""Print-split of the variant-B2 top baffle (see top_baffle_nd25fw4_b2.py).

The one-piece baffle (304.8 x 453.5 x 18.3 mm) exceeds a 256 mm print bed,
so it is split into four flat pieces joined by through-thickness dovetail
keys:

  seam A  y = 120   full-width horizontal cut (crosses the D190 cutout, so
                    two ~58 mm land segments are actually joined)
  seam B  y = 315.95  horizontal cut exactly through B2's waist kinks, so
                    the seam hides in the outline crease and BOTH pieces get
                    obtuse corners there (top foot ~107 deg vs the flare,
                    mids ~152 deg vs the chamfer) -- no brittle knife-tips
  seam C  x = -5.6  vertical cut between seams A and B (mostly inside the
                    D190 cutout; ~20 mm of real joint above the cutout).
                    Offset left of center so its dovetail pocket clears
                    the 90-deg W22 insert bore at (0, 305.7) by 1.55 mm.

Pieces (all fit a 256 x 256 bed front-face-down):
  piece_bottom     ~250.6 x 125 mm   (male dovetails up into the mids)
  piece_mid_left   ~146.7 x 202 mm   (male dovetail up into the top)
  piece_mid_right  ~162.0 x 202 mm   (male dovetails up + into mid_left)
  piece_top_b2     ~121.3 x 138 mm   (the whole B2 mini-vase + crescent)

Female sides are opened up by CLEARANCE_MM (0.05, a snug fit --
tune X-Y hole compensation on the coupon first) via a mitred offset,
so every mating surface has a uniform assembly gap.
"""

from __future__ import annotations

from build123d import (
    Box,
    Compound,
    Cylinder,
    Plane,
    Polyline,
    Pos,
    Wire,
    extrude,
    make_face,
)
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from top_baffle_nd25fw4 import STAND_FOOT, THICKNESS_MM, baffle_solid
from top_baffle_nd25fw4_b import (
    TWEETER_DROP_MM,
    apply_magnet_base_cavities,
)
from top_baffle_nd25fw4_b2 import OUTLINE_B2
from top_baffle_nd25fw4_cables import ROUTING_PROFILE, cable_cutters
from top_baffle_nd25fw4_cables import TS_ROUTE_CAPTIVE

PRINT_ORIENTATION = "front-face-down"

if ROUTING_PROFILE != "proud":
    raise RuntimeError(
        "B2/C7/V0/V1/V1L pieces require LX_ROUTING_PROFILE=proud; "
        "Obi-Wan has a separate skeletal builder"
    )

CLEARANCE_MM = 0.05

# Fused stand foot on piece_bottom (STAND_FOOT flag lives in the base
# module -- it also removes the bridge pass-throughs and reroutes the
# cable ducts down through the foot). The foot is the baffle's own
# bottom strip (y 0..18.3, side faces continuing the flank slopes:
# +/-76.2 at the floor widening to +/-81.64) extruded 150 mm rearward.
# The plate and foot share the same floor plane -- no step below the
# bottom edge; the inner corner is a plain 90 deg joint (no rib).
FOOT_THICK = 18.3         # vertical height of the foot slab
FOOT_DEPTH_REAR = 150.0   # behind the rear face (z=0)
FLANK_SLOPE = 0.29752     # dx/dy of the lower flanks
# The foot tapers in plan CONTINUOUSLY -- one straight line per side
# from the strip corners (+/-81.6 at the plate) to 38 wide at the panel
# inner face (z=-146), whose last 4 mm carry a minimal panel wall
# (38 x 44 x 4) for a Neutrik NL8MPXX-BAG: D31 cutout at (0, 20.5) +
# 4 x D3.2 on the 29.2 x 29.2 pattern. The center is channeled to a
# 4.0 floor between side rails so the D30.5 body fits; the four duct
# runs (packed at x -13.9/-5.45/+5.4/+13.9) exit through the channel's
# step face at z=-99, leaving ~40 mm of open channel to the connector
# tabs.
TONGUE_HALF_W = 19.0
PANEL_T = 4.0
PANEL_H = 44.0
CHANNEL_HALF_W = 17.0   # rails 2.0 thick; >=1.75 around the D30.5 body
CHANNEL_FLOOR = 4.0
CHANNEL_STEP_Z = -99.0
NL8_CENTER_Y = 20.5
NL8_CUTOUT_D = 31.0
NL8_SCREW_D = 3.2
NL8_SCREW_PITCH = 29.2

SEAM_A_Y = 120.0
SEAM_B_Y = 315.95  # exactly at B2's waist kinks -> obtuse seam corners

# Dovetail keys: (center along seam, neck width, head width, depth).
# Seam A sits at y=120 so its keys at +-89 live in the 16 mm-wide
# FULL-DEPTH window between the T arc (r=110, crossing at x~72.6) and
# the C7 taper boundary (x~92) -- keys clear of every duct AND fully
# out of the taper in all variants. Seam-B teeth occupy the clear
# lands between the proud R6P return corridor and the waist kink.
# TWO per mid (was one each) so neither half pivots on a single
# tab. Spread across the flank band, straddling the duct
# crossings (TS at x=-80, UM at x=+88).
# inner teeth SHALLOW (depth 3) and at +-66 so the wall to the LM
# cutout (which bulges to x=57 at the seam-band top) stays ~7 mm, not
# ~1.5; outer teeth (+-103, depth 5) are the main anchor.  The positive
# seam-B tooth is centred at x=29 rather than x=30 so its grown female
# pocket stays clear of the complete lower-right captive-magnet sealing
# land.  This is a broad registration-key placement change, not a local
# magnet-shaped patch, and leaves the acoustic/front/rear envelope unchanged.
DOVETAILS_A = [(-103.0, 6.0, 7.0, 5.0), (-66.0, 7.0, 9.0, 3.0),
               (66.0, 7.0, 9.0, 3.0), (103.0, 6.0, 7.0, 5.0)]
DOVETAILS_B = [(-19.0, 10.0, 14.0, 6.0), (29.0, 10.0, 14.0, 6.0)]
# ONE tooth in the ~20 mm mid-mid neck (per user preference), centered
# and beefed (neck 7, head 8.5). mid_right carries the tab.
DOVETAILS_C = [(305.0, 7.0, 8.5, 4.0)]
SEAM_C_X = -5.6  # clears the 90-deg W22 insert bore (edge at x=-3.9)

XMAX = 200.0  # anything beyond the baffle outline


def _trapezoid_up(cx: float, y: float, neck: float, head: float, depth: float) -> Polygon:
    return Polygon(
        [
            (cx - neck / 2.0, y),
            (cx + neck / 2.0, y),
            (cx + head / 2.0, y + depth),
            (cx - head / 2.0, y + depth),
        ]
    )


def _trapezoid_down(cx: float, y: float, neck: float, head: float, depth: float) -> Polygon:
    return Polygon(
        [
            (cx - neck / 2.0, y),
            (cx + neck / 2.0, y),
            (cx + head / 2.0, y - depth),
            (cx - head / 2.0, y - depth),
        ]
    )


def _below_region(seam_y: float, dovetails, tabs_up: bool = True) -> Polygon:
    """tabs_up: the below piece carries up-tabs (stock). Otherwise the
    below piece carries POCKETS (down-trapezoid notches) and the piece
    above inherits down-tabs via the same complement/grow machinery --
    growing a notched polygon shrinks the notches, which lands the
    0.05 clearance on the male side instead of the female (same fit)."""
    below = box(-XMAX, -20.0, XMAX, seam_y)
    if tabs_up:
        return unary_union(
            [below] + [_trapezoid_up(cx, seam_y, n, h, d)
                       for cx, n, h, d in dovetails])
    for cx, n, h, d in dovetails:
        below = below.difference(_trapezoid_down(cx, seam_y, n, h, d))
    return below


def _above_region(seam_y: float, dovetails) -> Polygon:
    """Exact complement of _below_region within the part's extent (used to
    CUT male-side pieces -- OCC's common op is flaky on the ducted solid,
    subtraction is robust)."""
    above = box(-XMAX, seam_y, XMAX, 600.0)
    for cx, n, h, d in dovetails:
        above = above.difference(_trapezoid_up(cx, seam_y, n, h, d))
    return above


def _right_region() -> Polygon:
    s = SEAM_C_X
    parts = [box(s, 50.0, XMAX, 400.0)]
    for cy, neck, head, depth in DOVETAILS_C:
        parts.append(Polygon([
            (s, cy - neck / 2.0),
            (s, cy + neck / 2.0),
            (s - depth, cy + head / 2.0),
            (s - depth, cy - head / 2.0),
        ]))
    return unary_union(parts)


def _prism(poly: Polygon):
    pts = list(poly.exterior.coords)
    face = make_face(Wire(Polyline(*pts).edges()))
    return extrude(Plane.XY.offset(-1.0) * face, amount=THICKNESS_MM + 2.0)


def _grown(poly: Polygon) -> Polygon:
    return poly.buffer(CLEARANCE_MM, join_style=2, mitre_limit=10.0)


def pieces(outline=OUTLINE_B2, tweeter_drop_mm: float = TWEETER_DROP_MM,
           shape_cuts=(), shape_adds=(), magnet_cavities=True,
           crescent_front_mm=None, crescent_rear_mm=0.0,
           seam_b_dovetails=None, seam_b_tabs_up: bool = True,
           um_handoff_key: str = "proud",
           only: str | None = None,
           cable_routes=None,
           cable_y_range=None,
           ts_route_key: str = TS_ROUTE_CAPTIVE) -> dict:
    """Split the (optionally re-shaped) baffle into the four print
    pieces. ``shape_cuts``/``shape_adds`` are applied before the ducts
    are cut -- used by variant C7 (LM knife-edge taper + T-duct ribs);
    the ducts then re-cut through any added material. Obi-Wan is a separate
    two-carrier R6F core and never passes through this four-piece builder.
    ``um_handoff_key`` is explicit so V1L can select its rear-plane axis
    handoff while every other proud-family caller keeps the default.
    ``only`` constructs one requested split solid without retaining the
    other three OCC trees; guarded validation/export uses this to reduce the
    local process-tree RSS peak. ``cable_routes``
    may further omit cutters spatially disjoint from that one piece; the
    default still builds the complete cable set. ``cable_y_range`` trims
    the high-face-count TS loft on its original section grid for a known
    split band. ``ts_route_key`` defaults to the shared stock/slim captive-
    land keepout; explicit diagnostic callers may still request the legacy
    unnudged centerline."""
    piece_order = (
        "piece_bottom", "piece_mid_left", "piece_mid_right", "piece_top_b2")
    if only is not None and only not in piece_order:
        raise ValueError(f"unknown split piece {only!r}; choose {piece_order}")
    requested = set(piece_order if only is None else (only,))

    baffle = baffle_solid(outline, tweeter_drop_mm,
                          crescent_front_mm or THICKNESS_MM,
                          crescent_rear_mm)
    for add in shape_adds:
        baffle += add
    for cutter in shape_cuts:
        baffle -= cutter
    ducts = cable_cutters(
        um_handoff_key=um_handoff_key,
        route_names=cable_routes,
        ts_y_range=cable_y_range,
        ts_route_key=ts_route_key)  # internal cable ducts (LM/UM/T)
    for duct in ducts:
        baffle -= duct

    below_a = _below_region(SEAM_A_Y, DOVETAILS_A)
    result = {}

    if "piece_bottom" in requested:
        bottom = baffle - _prism(_above_region(SEAM_A_Y, DOVETAILS_A))
        if STAND_FOOT:
            w0 = 76.2
            w1 = 76.2 + FLANK_SLOPE * FOOT_THICK
            strip = make_face(Wire(Polyline(
                (-w0, 0.0), (w0, 0.0), (w1, FOOT_THICK),
                (-w1, FOOT_THICK), (-w0, 0.0)
            ).edges()))
            strip_prism = extrude(
                Plane.XY.offset(-FOOT_DEPTH_REAR - 1) * strip,
                amount=FOOT_DEPTH_REAR + 1)
            h = TONGUE_HALF_W
            zp = -FOOT_DEPTH_REAR + PANEL_T  # panel inner face
            plan = make_face(Wire(Polyline(
                (-82.0, 0.5), (82.0, 0.5), (h, zp),
                (h, -FOOT_DEPTH_REAR), (-h, -FOOT_DEPTH_REAR), (-h, zp),
                (-82.0, 0.5)
            ).edges()))
            plan_prism = extrude(
                Plane((0, FOOT_THICK + 1.0, 0),
                      x_dir=(1, 0, 0), z_dir=(0, -1, 0)) * plan,
                amount=FOOT_THICK + 2.0)
            foot = strip_prism & plan_prism
            panel = Pos(0, PANEL_H / 2,
                        -FOOT_DEPTH_REAR + PANEL_T / 2) * Box(
                2 * TONGUE_HALF_W, PANEL_H, PANEL_T)
            bottom = bottom + foot + panel
            # connector channel between the side rails (step face z=-99)
            bottom -= Pos(
                0, (CHANNEL_FLOOR + PANEL_H + 6) / 2,
                (CHANNEL_STEP_Z + (-FOOT_DEPTH_REAR + PANEL_T)) / 2
            ) * Box(
                2 * CHANNEL_HALF_W, PANEL_H + 6 - CHANNEL_FLOOR,
                -(-FOOT_DEPTH_REAR + PANEL_T) + CHANNEL_STEP_Z)
            for cx, cy, d in [(0.0, NL8_CENTER_Y, NL8_CUTOUT_D)] + [
                (sx * NL8_SCREW_PITCH / 2,
                 NL8_CENTER_Y + sy * NL8_SCREW_PITCH / 2, NL8_SCREW_D)
                for sx in (1, -1) for sy in (1, -1)
            ]:
                bottom -= Pos(
                    cx, cy, -FOOT_DEPTH_REAR + PANEL_T / 2
                ) * Cylinder(d / 2, PANEL_T + 2)
            for duct in ducts:  # re-cut: foot tunnels cross the union
                bottom -= duct
        result["piece_bottom"] = bottom

    upper_requested = requested - {"piece_bottom"}
    if upper_requested:
        below_b = _below_region(
            SEAM_B_Y, seam_b_dovetails or DOVETAILS_B,
            tabs_up=seam_b_tabs_up)
        rest = baffle - _prism(_grown(below_a))
        if "piece_top_b2" in upper_requested:
            top = rest - _prism(_grown(below_b))
            if magnet_cavities:
                top = apply_magnet_base_cavities(top)
            result["piece_top_b2"] = top
        mids_requested = upper_requested & {
            "piece_mid_left", "piece_mid_right"}
        if mids_requested:
            mids = rest & _prism(below_b)
            right_c = _right_region()
            if "piece_mid_right" in mids_requested:
                result["piece_mid_right"] = mids & _prism(right_c)
            if "piece_mid_left" in mids_requested:
                result["piece_mid_left"] = mids - _prism(_grown(right_c))

    return {name: result[name] for name in piece_order if name in result}


def gen_step():
    children = []
    for label, solid in pieces().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_b2_split"
    return assembly
