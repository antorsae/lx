"""Candidate Obi-Wan tweeter options for ONE open-back (dipole) flat tweeter, two layouts x two orientations.

A dipole tweeter radiates from both faces, so one driver replaces the back-to-back pairs of the other families.
Two drivers are supported (LX_DIPOLE_T_DRIVER):

  pt2522_4   GRS PT2522-4 planar (vendor/GRS/PT2522-4). Thin (grille-only) face to the listener; the flange seats in
             a 1.0 mm recess with pockets under its six assembly-screw heads; the motor housing, terminal block and
             the two L-shaped solder lugs sit in one open window through the part. The leads are soldered in there.
  amt17d2_1  Mundorf hifiAMT17D2.1 (vendor/Mundorf/AMT17D2.1). The slotted front plate goes to the listener and
             seats in a 1.5 mm recess shaped to it and its four ears; the body sits in one open window through the
             part. Its flying leads leave one short edge: a blind channel from the rear takes them out of the window.

and two orientations (LX_DIPOLE_T_ORIENTATION): 'portrait' (driver y = installed y; PT2522 terminals down, AMT leads
down, near the T side) or 'landscape' (the driver turned 90 deg, terminals/leads towards +x, the side the UM's T
lumen arrives on; the opening is mirrored so the part stays symmetric). Driver envelopes come from the vendor
drawings; the photo estimates in them must be measured on a physical driver before release.

Variant ``crescent`` (above the UM)
  A minimal crescent on the released UM half-lap mate (x=+/-24, y=421.5) with the released T--UM joint. The driver
  sits on the front face; the frame is the driver's own outline (grown only where the window and pilots need a
  wall), so from the front only the tweeter shows. Portrait: a waisted neck to the bosses, as low as the released
  wing envelope allows, so the regular Obiwan wings still fit. Landscape: down on the UM (0.75 mm off its released
  clearance) with a saddle hugging the UM ring's profile; the regular wings do not fit. No duct: the T cable leaves
  the UM's released mouth behind the part and reaches the driver through the open rear window.

Variant ``um_t`` (one piece: LM -> tweeter -> MU10)
  The released UM carrier is split just above its LM half-laps (y=322). The lower piece keeps the complete LM
  interface; the ring (MU10 seat and pilots) moves up just far enough to clear the tweeter; a sculpted body with a
  razor side edge joins them. The UM's T lumen branches into the tweeter's window; above the split the T lumen is
  filled and the crescent half-lap horns are removed. The MU10 lead runs up the body's left side, outboard of the
  tweeter's features, to the raised MU10 seat. The regular Obiwan wings do not fit this variant.

Fixing: four self-tapping screws into printed pilots (no room for heat-set inserts at the driver's holes).

Nothing here is release-authorized.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

from build123d import Box, Part, Plane, Pos, Solid, Vector, import_brep
from shapely import affinity
from shapely.geometry import LineString, Point, Polygon, box as sbox
from shapely.geometry.polygon import orient as _orient
from shapely.ops import unary_union

from ..base import THICKNESS_MM, UM_CUTOUT
from .attachments import _cylinder_at, _fuse_required
from .carriers import (CORE_REAR_Z, _apply_complete_um_tweeter_joint, _enforce_junction_plan_ownership, _plan_prism,
                       _require_guarded_build)
from . import bmr_pod

ROOT = Path(__file__).resolve().parents[3]
RELEASE_AUTHORIZED = False
PHYSICAL_MEASURE_REQUIRED = True

# --- common mounting parameters ---
RECESS_CLEAR_MM = 0.25            # per side around the plate/flange
WINDOW_CLEAR_MM = 0.5             # per side around the body/housing
OPENING_PILOT_WALL_MM = 1.2       # opening to the mounting pilots
FRAME_WALL_MM = 1.6               # least frame wall beside the window (crescent)
T_DUCT_D = bmr_pod.CABLE_DUCT_D_MM          # 6.0, the UM's own T lumen


def _rounded_rect(cx, cy, w, h, r):
    return sbox(cx - w / 2 + r, cy - h / 2 + r, cx + w / 2 - r, cy + h / 2 - r).buffer(r, resolution=32)


class _PT2522:
    """GRS PT2522-4 planar, driver frame (terminals at -y)."""
    key, name, label = 'pt2522_4', 'planar_pt2522', 'GRS PT2522-4 planar, open back (dipole)'
    vendor = ROOT / 'vendor/GRS/PT2522-4'
    TERMINAL_CLEAR_MM = 0.8           # around the insulator block and lugs
    SCREW_HEAD_CLEAR_MM = 0.5         # radial, and below each thick-face screw head
    RECESS_CORNER_R = 2.25
    OPENING_CORNER_R = 3.5
    pilot_d, pilot_depth, screw = 2.6, 8.0, 'M3 self-tapping'
    t_duct_d = 6.0                    # the UM's own T lumen: the D5.2 T cable runs to the solder lugs
    lead_exit = None                  # solder lugs in the window: no lead channel

    def __init__(self):
        d = json.loads((self.vendor / 'PT2522-4.json').read_text()); dr, ph = d['drawing'], d['photo_estimates']
        self.plate_w, self.plate_h, self.plate_t = dr['plate_w'], dr['plate_h'], ph['plate_t']
        self.hole_pitch = tuple(dr['hole_pitch'])
        self.housing = (dr['grille_w'], dr['grille_h'], ph['housing_r'], ph['grille_center_y'])
        self.terminal_block, self.terminal_y = ph['terminal_block'], ph['terminal_center_y']
        self.lug = (ph['lug_x'], dr['lug_w'], ph['lug_h'])
        self.screws = [tuple(s) for s in ph['screws']]
        self.screw_head = (ph['screw_head_d'], ph['screw_head_h'])
        self.depth_behind_front = ph['plate_t'] + ph['thick_bulge']          # 7.0: flange front to housing rear
        self.corner_r = 2.0                                                 # flange corner radius (vendor model)

    def build(self):
        sys.path.insert(0, str(self.vendor)); import pt2522_4; return pt2522_4.build()

    def plate(self):
        return _rounded_rect(0, 0, self.plate_w, self.plate_h, self.corner_r)

    def recess(self):
        return _rounded_rect(0, 0, self.plate_w + 2 * RECESS_CLEAR_MM, self.plate_h + 2 * RECESS_CLEAR_MM, self.RECESS_CORNER_R)

    def pilots(self):
        return [(sx * self.hole_pitch[0] / 2, sy * self.hole_pitch[1] / 2) for sx in (-1, 1) for sy in (-1, 1)]

    def opening(self):
        """The housing window flowing into a lower zone for the terminal block, both lugs and the two bottom
        flange-screw heads: one convex outline (no lobes, notches or steps), clear of the bottom pilots."""
        hw, hh, hr, hcy = self.housing
        window = _rounded_rect(0, hcy, hw + 2 * WINDOW_CLEAR_MM, hh + 2 * WINDOW_CLEAR_MM, hr + WINDOW_CLEAR_MM)
        tw, th = self.terminal_block[0] / 2, self.terminal_block[1] / 2
        lug_x, lug_w, lug_h = self.lug
        lower_half_w = self.hole_pitch[0] / 2 - self.pilot_d / 2 - OPENING_PILOT_WALL_MM
        lower_bottom = self.terminal_y - max(th, lug_h / 2) - self.TERMINAL_CLEAR_MM
        lower = _rounded_rect(0, (hcy + lower_bottom) / 2, 2 * lower_half_w, hcy - lower_bottom, self.OPENING_CORNER_R)
        if max(tw, lug_x + lug_w / 2) + self.TERMINAL_CLEAR_MM > lower_half_w:
            raise ValueError('terminal block or lugs outside the opening')
        heads = [Point(x, y).buffer(self.screw_head[0] / 2 + self.SCREW_HEAD_CLEAR_MM, resolution=32) for x, y in self._bottom_screws()]
        return unary_union([window, lower, *heads]).convex_hull

    def _bottom_screws(self):
        return [(x, y) for x, y in self.screws if y < -self.housing[1] / 2]

    def pockets(self):
        """Blind pockets below the recess floor: (x, y, radius, depth below the floor)."""
        r = self.screw_head[0] / 2 + self.SCREW_HEAD_CLEAR_MM
        return [(x, y, r, self.screw_head[1] + self.SCREW_HEAD_CLEAR_MM) for x, y in self.screws if (x, y) not in self._bottom_screws()]

    def lead_point(self):
        """Where a T lead branch ends in the window (driver frame, portrait routing)."""
        return (0.0, self.terminal_y)


class _AMT17D2:
    """Mundorf hifiAMT17D2.1, driver frame (ears at +/-y, flying leads leave the -y edge)."""
    key, name, label = 'amt17d2_1', 'amt17d2', 'Mundorf hifiAMT17D2.1 AMT, open back (dipole)'
    vendor = ROOT / 'vendor/Mundorf/AMT17D2.1'
    pilot_d, pilot_depth, screw = 3.4, 5.5, 'M4 x 6 self-tapping'  # D4.5 ear holes; 5.5 deep leaves room for a duct below
    t_duct_d = 4.0                    # the AMT's own two flying leads (owner decision): D4 in every part
    LEAD_CHANNEL_W, LEAD_CHANNEL_DEPTH = 5.0, 4.0                  # closed channel: the leads meet the T duct here

    def __init__(self):
        d = json.loads((self.vendor / 'AMT17D2.1.json').read_text()); dr, ph = d['drawing'], d['photo_estimates']
        self.plate_w, self.plate_h, self.plate_t = dr['plate_w'], dr['plate_h'], ph['plate_t']
        self.overall_h, self.hole_pitch = dr['overall_h'], tuple(dr['hole_pitch'])
        self.ear_w, self.corner_r = ph['ear_w'], ph['plate_corner_r']
        self.body = (ph['body_w'], ph['body_h'])
        self.depth_behind_front = dr['depth']                       # 10.0: plate front to body rear
        self.lead_exit = (ph['lead_x'], -ph['body_h'] / 2)           # on the -y edge

    def build(self):
        sys.path.insert(0, str(self.vendor)); import amt17d2_1; return amt17d2_1.build()

    def plate(self):
        hx, hy = self.hole_pitch[0] / 2, self.hole_pitch[1] / 2
        ear_r = self.overall_h / 2 - hy
        ears = [unary_union([sbox(sx * hx - self.ear_w / 2, min(sy * self.plate_h / 2, sy * hy), sx * hx + self.ear_w / 2,
                                  max(sy * self.plate_h / 2, sy * hy)), Point(sx * hx, sy * hy).buffer(ear_r, resolution=32)])
                for sx in (-1, 1) for sy in (-1, 1)]
        return unary_union([_rounded_rect(0, 0, self.plate_w, self.plate_h, self.corner_r), *ears])

    def recess(self):
        return self.plate().buffer(RECESS_CLEAR_MM, resolution=32)

    def pilots(self):
        return [(sx * self.hole_pitch[0] / 2, sy * self.hole_pitch[1] / 2) for sx in (-1, 1) for sy in (-1, 1)]

    def opening(self):
        bw, bh = self.body
        return _rounded_rect(0, 0, bw + 2 * WINDOW_CLEAR_MM, bh + 2 * WINDOW_CLEAR_MM, self.corner_r + WINDOW_CLEAR_MM)

    def pockets(self):
        return []

    def lead_channel(self):
        """Blind channel (plan, driver frame) off the window edge where the flying leads leave the body."""
        x, y = self.lead_exit
        w, d = self.LEAD_CHANNEL_W, self.LEAD_CHANNEL_DEPTH
        return _rounded_rect(x, y - WINDOW_CLEAR_MM - d / 2 + 1.0, w, d + 2.0, w / 2 - 0.01)

    def lead_point(self):
        x, y = self.lead_exit
        return (x, y - WINDOW_CLEAR_MM - self.LEAD_CHANNEL_DEPTH + 1.5)


DRIVERS = {'pt2522_4': _PT2522, 'amt17d2_1': _AMT17D2}
DRIVER_KEY = os.environ.get('LX_DIPOLE_T_DRIVER', 'pt2522_4')
if DRIVER_KEY not in DRIVERS: raise ValueError(f'LX_DIPOLE_T_DRIVER={DRIVER_KEY!r}; one of {sorted(DRIVERS)}')
DRIVER = DRIVERS[DRIVER_KEY]()
# Integrated variant (LX_DIPOLE_T_INTEGRATED=1, see lm_um_t.py): the keyed LM top half, the T below and the MU10 in
# one piece. Each cable keeps to its own side from the LM (T on -x, UM on +x), so rotated drivers turn the other way
# (terminals / leads towards -x), and the driver is fixed with the project's standard M3 heat-set inserts
# (print_policy.json hardware_stock.M3_insert, Hanglife HLTI-M3-001 M3 x 5 x 4, in the shared D4.6 x 4.0 bore).
INTEGRATED = os.environ.get('LX_DIPOLE_T_INTEGRATED') == '1'
# Pair variant (LX_DIPOLE_T_PAIR=1, implies INTEGRATED; see lm_um_t.py): for a layout whose one piece is longer than the
# H2C's nozzle reach (PT2522 upright), the same ducts and frame in TWO prints that meet at the released LM--UM joint --
# a keyed LM top variant carrying the T duct (-x) and a closed UM duct (+x) up to the joint, and the T + UM piece.
PAIR = os.environ.get('LX_DIPOLE_T_PAIR') == '1'
if PAIR and not INTEGRATED: raise RuntimeError('LX_DIPOLE_T_PAIR=1 needs LX_DIPOLE_T_INTEGRATED=1')
M3_INSERT_BORE_D_MM, M3_INSERT_BORE_DEPTH_MM = 4.6, 4.0
# Every released dipole part (T above too, since 2026-10-10) takes the standard inserts; LX_DIPOLE_T_INSERTS=0 restores
# the drivers' own self-tapping pilots for the older study parts.
INSERTS = INTEGRATED or os.environ.get('LX_DIPOLE_T_INSERTS', '1') == '1'
if INSERTS:
    DRIVER.pilot_d, DRIVER.pilot_depth, DRIVER.screw = M3_INSERT_BORE_D_MM, M3_INSERT_BORE_DEPTH_MM, 'M3 into M3 heat-set inserts'
PLATE_T = DRIVER.plate_t
PILOT_D_MM, PILOT_DEPTH_MM = DRIVER.pilot_d, DRIVER.pilot_depth

# --- orientation ---
ORIENTATION = os.environ.get('LX_DIPOLE_T_ORIENTATION', 'portrait')
if ORIENTATION not in ('portrait', 'landscape'): raise ValueError(f'LX_DIPOLE_T_ORIENTATION={ORIENTATION!r}')
LANDSCAPE = ORIENTATION == 'landscape'
ROT_SIGN = -1.0 if INTEGRATED else 1.0            # +90: terminals / leads towards +x; integrated: -90, towards -x
ROT_DEG = 90.0 * ROT_SIGN if LANDSCAPE else 0.0


def _rot(x, y):
    """Driver frame -> installed offsets from the plate centre."""
    if not LANDSCAPE: return (x, y)
    return (-y, x) if ROT_SIGN > 0 else (y, -x)


def _turn(g):
    """A driver-frame outline turned into the installed orientation (still centred on 0, 0)."""
    return affinity.rotate(g, ROT_DEG, origin=(0, 0))


def _mirror_x(g):
    return affinity.scale(g, -1.0, 1.0, origin=(0, 0))


def installed_opening():
    """The through opening, installed orientation, centred on 0, 0. Landscape mirrors it across the centre line so
    the part stays symmetric (turned, a PT2522's terminal zone lies on one side)."""
    o = _turn(DRIVER.opening())
    if LANDSCAPE: o = unary_union([o, _mirror_x(o)]).convex_hull
    return o


def installed_features():
    """(recess, opening, pockets, pilots, lead channel) outlines, installed orientation, centred on 0, 0."""
    recess = _turn(DRIVER.recess())
    pockets = [Point(*_rot(x, y)).buffer(r, resolution=32) for x, y, r, _ in DRIVER.pockets()]
    pilots = [Point(*_rot(x, y)) for x, y in DRIVER.pilots()]
    channel = _turn(DRIVER.lead_channel()) if DRIVER.lead_exit else None
    return recess, installed_opening(), pockets, pilots, channel


_R, _O, _P, _PIL, _CH = installed_features()
EXT_W = _turn(DRIVER.plate()).bounds[2] - _turn(DRIVER.plate()).bounds[0]    # driver outline extents, installed
EXT_H = _turn(DRIVER.plate()).bounds[3] - _turn(DRIVER.plate()).bounds[1]
OPENING_TOP = _O.bounds[3]                                                     # installed, above the plate centre
_KEEP = unary_union([_O, *_P, *[q.buffer(PILOT_D_MM / 2 + OPENING_PILOT_WALL_MM, resolution=32) for q in _PIL]])
FEATURE_HALF_W = -_KEEP.bounds[0]                                              # outboard edge of the driver's features
if min(_O.distance(q) for q in _PIL) - PILOT_D_MM / 2 < OPENING_PILOT_WALL_MM - 1e-6:
    raise ValueError('opening too close to the mounting pilots')


def planar_cutters(cx, cy, z_rear, z_front=THICKNESS_MM, pilot_depth=None, part_rear=None):
    """Recess, the through opening, blind pockets, the lead channel and four pilots, driver centred at (cx, cy).
    Returns (cutters, lead_point): the cutter solids and where a lead duct should end, behind the driver.
    part_rear: the part's rear face there (default z_rear); the lead channel stays closed behind it."""
    pilot_depth = PILOT_DEPTH_MM if pilot_depth is None else pilot_depth
    part_rear = z_rear if part_rear is None else part_rear
    recess_floor = z_front - PLATE_T
    place = lambda g: affinity.translate(g, cx, cy)
    cutters = [_plan_prism(place(_R), recess_floor, z_front + 1), _plan_prism(place(_O), z_rear - 1, z_front + 1)]
    for (x, y, r, depth), g in zip(DRIVER.pockets(), _P):
        cutters.append(_plan_prism(place(g), recess_floor - depth, recess_floor + 0.5))
    if _CH is not None:              # closed front and rear: the leads turn out of the window into a hidden duct
        lo = max(z_front - DRIVER.depth_behind_front, part_rear + FRAME_SKIN_MM)
        cutters.append(_plan_prism(place(_CH), lo, recess_floor - FRAME_SKIN_MM))
    for q in _PIL:
        cutters.append(_cylinder_at(cx + q.x, cy + q.y, PILOT_D_MM / 2, recess_floor - pilot_depth, recess_floor + 0.5))
    lx, ly = _rot(*DRIVER.lead_point())
    return cutters, (cx + lx, cy + ly, (z_rear + recess_floor) / 2)


def planar_features(part, cx, cy, z_rear, z_front=THICKNESS_MM):
    cutters, slot = planar_cutters(cx, cy, z_rear, z_front)
    for c in cutters: part -= c
    return part, slot


FRAME_SKIN_MM = 0.8


# ======================= variant 1: minimal crescent above the UM =======================
RING_FUSE_R = 50.5                 # one-piece UM + T above: the frame fuses onto the UM ring outside this radius
# The driver sits ON the crescent's front face (no recess): the frame is the driver's own outline, grown only where
# the window or a pilot needs a wall, so from the front only the tweeter shows.
# No cable duct: the T lead leaves the UM's released mouth behind the core rear plane and reaches the driver
# directly through the open rear window, a few mm above.
# Portrait: the lowest centre with the frame >= 2 mm clear of the released wing envelope (regular wings fit).
# Landscape: as close to the UM as it goes, 0.75 mm off the released UM clearance; the regular wings do not fit.
CRESCENT_FITS_WINGS = not LANDSCAPE
CRESCENT_WING_GAP_MM, CRESCENT_UM_GAP_MM = 2.0, 0.75
JOINT_BOSS_WALL_MM = 1.2           # the driver's cut-outs stay this far off the T--UM joint bosses (crescent half)
UM_RING_OUTER_R = 52.64            # released UM ring outer radius (the R51.90 clearance only applies at the mate)
CRESCENT_SADDLE_GAP_MM = 0.2       # the released mate gap, kept along the ring's side too
CRESCENT_SADDLE_BLEND_R = 15.0      # softens the kink where the saddle flank leaves the frame side
CRESCENT_MIN_FEATURE_MM = 0.6       # the saddle's tangent tip is trimmed where it gets thinner than 2x this
CRESCENT_FRAME_R = DRIVER.corner_r
CRESCENT_WAIST_FILLET_R = 9.0      # concave blend between the neck and the frame
_FRAME_BOUNDS = unary_union([_turn(DRIVER.plate()), _O.buffer(FRAME_WALL_MM, join_style=2),
                             *[q.buffer(PILOT_D_MM / 2 + OPENING_PILOT_WALL_MM) for q in _PIL]]).bounds
FRAME_W, FRAME_H = 2 * max(-_FRAME_BOUNDS[0], _FRAME_BOUNDS[2]), 2 * max(-_FRAME_BOUNDS[1], _FRAME_BOUNDS[3])


def _frame_at(cy):
    return _rounded_rect(0.0, cy, FRAME_W, FRAME_H, CRESCENT_FRAME_R)


def _driver_cutouts():
    """Everything the driver mount cuts through or into the frame (installed, centred on 0, 0)."""
    return unary_union([_O, *_P, *([_CH] if _CH is not None else []), *[q.buffer(PILOT_D_MM / 2, resolution=32) for q in _PIL]])


def _joint_gap(cy):
    """Clearance from the driver's cut-outs to the T--UM joint bosses (the crescent's half-lap receivers)."""
    return affinity.translate(_driver_cutouts(), 0.0, cy).distance(unary_union(bmr_pod._boss_discs()))


def _lowest_centre():
    """Lowest frame centre (0.5 mm steps) meeting the clearance rule for this orientation, with the joint bosses
    intact."""
    um, wing = bmr_pod._um_clearance_disc(), bmr_pod._wing_keepout_plan()
    cy = math.floor(UM_CUTOUT[1] + 52.0)
    while True:
        f = _frame_at(cy)
        ok = (f.distance(wing) >= CRESCENT_WING_GAP_MM and f.distance(um) >= CRESCENT_UM_GAP_MM) if CRESCENT_FITS_WINGS \
            else f.distance(um) >= CRESCENT_UM_GAP_MM
        if ok and _joint_gap(cy) >= JOINT_BOSS_WALL_MM: return cy
        cy += 0.5


CRESCENT_CENTER = (0.0, _lowest_centre())


def crescent_land_plan():
    return _frame_at(CRESCENT_CENTER[1])


def crescent_plan(fused=False):
    """fused=True: the plan for a one-piece UM + T above (overlaps the UM ring instead of keeping the mate gap)."""
    land = crescent_land_plan()
    bosses = bmr_pod._boss_discs()
    bx = max(abs(d.centroid.x) for d in bosses)
    frame_bottom = CRESCENT_CENTER[1] - FRAME_H / 2
    neck = unary_union([*bosses, sbox(-bx, bosses[0].centroid.y, bx, frame_bottom + 1.0)]).convex_hull
    r = CRESCENT_WAIST_FILLET_R
    plan = unary_union([land, neck]).buffer(r, resolution=32).buffer(-r, resolution=32)
    um = Point(*UM_CUTOUT[:2]).buffer(RING_FUSE_R, resolution=128) if fused else bmr_pod._um_clearance_disc()
    if not CRESCENT_FITS_WINGS:
        plan = unary_union([plan, _crescent_saddle(frame_bottom, fused)])
        plan = plan.buffer(CRESCENT_SADDLE_BLEND_R, resolution=32).buffer(-CRESCENT_SADDLE_BLEND_R, resolution=32)
    plan = unary_union([plan, land]).difference(um)
    if CRESCENT_FITS_WINGS: plan = plan.difference(bmr_pod._wing_keepout_plan())
    else:
        f = CRESCENT_MIN_FEATURE_MM
        plan = unary_union([plan.buffer(-f, resolution=32).buffer(f, resolution=32), land.difference(um)])
    if plan.geom_type != 'Polygon': plan = max(plan.geoms, key=lambda g: g.area)
    return plan


def _crescent_saddle(frame_bottom, fused=False, cx=None):
    """Landscape: the frame sits ON the UM's profile. From each bottom frame corner a straight flank runs down
    tangent to the UM ring's outer circle (plus the 0.2 mm mate gap); the region between the flanks, the frame and
    that circle is filled, so the part hugs the UM's profile."""
    c = um_centre = (UM_CUTOUT[0], UM_CUTOUT[1])
    r = UM_RING_OUTER_R + CRESCENT_SADDLE_GAP_MM
    pts = []
    for sx in (-1, 1):
        qx, qy = (CRESCENT_CENTER[0] if cx is None else cx) + sx * FRAME_W / 2, frame_bottom
        dx, dy = qx - c[0], qy - c[1]; d = math.hypot(dx, dy)
        a = math.atan2(dy, dx) - sx * math.acos(r / d)                    # outer tangent point
        pts.append(((qx, qy + CRESCENT_FRAME_R), (c[0] + r * math.cos(a), c[1] + r * math.sin(a))))
    (ql, tl), (qr, tr) = pts
    from shapely.geometry import Polygon
    ring = Point(*c).buffer(RING_FUSE_R if fused else r / math.cos(math.pi / 256), resolution=64)   # circumscribed: the gap survives faceting
    return Polygon([ql, tl, um_centre, tr, qr]).difference(ring)


def crescent():
    _require_guarded_build(); bmr_pod.check_released_mate()
    plan = crescent_plan(); land = crescent_land_plan()
    if land.difference(plan).area > 1.0:
        raise RuntimeError('the released wing envelope or UM clearance cuts into the tweeter frame')
    if land.distance(bmr_pod._um_clearance_disc()) < 0.5:
        raise RuntimeError('tweeter frame closer than 0.5 mm to the released UM clearance')
    if _joint_gap(CRESCENT_CENTER[1]) < JOINT_BOSS_WALL_MM:
        raise RuntimeError('the driver mount cuts into the T--UM joint bosses')
    part = _plan_prism(plan, *bmr_pod.SKIRT_Z)
    part = _enforce_junction_plan_ownership(part, 't_um', 'tweeter')
    part = _apply_complete_um_tweeter_joint(part, 'tweeter')
    # Surface mount: the flange's rear face seats on the front face (recess floor = front face).
    part, _ = planar_features(part, *CRESCENT_CENTER, z_rear=bmr_pod.SKIRT_Z[0], z_front=THICKNESS_MM + PLATE_T)
    return _one_solid(part, 'dipole tweeter crescent')


def capsule_cut(mesh, tubes):
    """Subtract capsule-chain tubes (hull of consecutive spheres) from a watertight mesh with manifold3d."""
    import manifold3d as m3
    import numpy as np
    import trimesh
    part = m3.Manifold(m3.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32), tri_verts=np.asarray(mesh.faces, np.uint32)))
    for points, radius in tubes:
        balls = [m3.Manifold.sphere(radius, 48).translate(list(p)) for p in points]
        for a_, b_ in zip(balls, balls[1:]):
            part = part - m3.Manifold.batch_hull([a_, b_])
    if part.status() != m3.Error.NoError or part.is_empty(): raise RuntimeError(f'tube cut failed: {part.status()}')
    out = part.to_mesh()
    return trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)


# ======================= variant 2: one-piece UM + T =======================
from ..flush import UM_RECESS_R, UM_SEAT_Z

SPLIT_Y = 322.0
# The driver's recess starts this far above the split (per driver and orientation). Owner-approved raises
# (2026-10-03): the hidden UM cable guide leaves the LM's UM duct exit at x~27 and must get round the window's
# bottom corner (and, AMT rotated, the T duct passing outboard of the driver's mounting pilot) at its bend rule:
# PT2522 upright 2 -> 18, PT2522 rotated 2 -> 26, AMT rotated 10 -> 14; the AMT upright keeps 2.
PLATE_MARGIN_BELOW = {('pt2522_4', 'portrait'): 18.0, ('pt2522_4', 'landscape'): float(os.environ.get('LX_PT_ROT_MARGIN', '26.0')),
                      ('amt17d2_1', 'portrait'): 2.0, ('amt17d2_1', 'landscape'): 14.0}[(DRIVER_KEY, ORIENTATION)]
BELOW_RAISE_MM = float(os.environ.get('LX_DIPOLE_T_BELOW_RAISE_MM', '0'))   # study knob: extra height above the split
if INTEGRATED:                              # no split, no external cable climb: the original heights
    PLATE_MARGIN_BELOW = {('pt2522_4', 'portrait'): 2.0, ('pt2522_4', 'landscape'): 2.0,
                          ('amt17d2_1', 'portrait'): 2.0, ('amt17d2_1', 'landscape'): 10.0}[(DRIVER_KEY, ORIENTATION)]
PLATE_MARGIN_BELOW += BELOW_RAISE_MM
PLATE_CENTER_Y = SPLIT_Y + PLATE_MARGIN_BELOW + RECESS_CLEAR_MM + EXT_H / 2


def _integrated_lowest_centre():
    """Integrated part: the driver as close to the LM as it goes -- the lowest plate centre (0.25 mm steps) whose recess,
    window, lead channel and pockets keep RECESS_WEB_MM to the woofer's flange-recess circle, and whose insert bores
    keep their 1.2 mm wall clear of the woofer's M5 insert bosses."""
    from ..base import L22_CUTOUT, L22_PILOT_PCD_MM, L22_PILOT_ANGLES_DEG
    from ..flush import LM_RECESS_R
    circle = Point(*L22_CUTOUT[:2]).buffer(LM_RECESS_R, resolution=256)
    bosses = [Point(L22_CUTOUT[0] + L22_PILOT_PCD_MM / 2 * math.cos(math.radians(a)),
                    L22_CUTOUT[1] + L22_PILOT_PCD_MM / 2 * math.sin(math.radians(a))) for a in L22_PILOT_ANGLES_DEG]
    web = [_R, _O, *_P] + ([_CH] if _CH is not None else [])
    walls = [q.buffer(PILOT_D_MM / 2 + OPENING_PILOT_WALL_MM, resolution=32) for q in _PIL]
    cy, best = PLATE_CENTER_Y + 40.0, None
    while cy > L22_CUTOUT[1]:
        at = lambda g: affinity.translate(g, 0.0, cy)
        if not (all(at(g).distance(circle) >= RECESS_WEB_MM for g in web)
                and all(at(g).distance(b) >= INTEGRATED_BOSS_CLEAR_MM for g in walls for b in bosses)):
            break
        best, cy = cy, cy - 0.25
    return best


RECESS_WEB_MM = 2.0                         # front web between the driver's features and a neighbouring recess
INTEGRATED_BOSS_CLEAR_MM = 5.2              # lm_um_t.WOOFER_BOSS_R: the woofer's M5 insert boss
# Pair: the T + UM piece meets the released keyed LM top at the LM--UM joint: its bar top (y 315.68 across the
# joint face) and the two ear bosses (D9.8 at x=+/-32, EAR_Y). The driver's features keep the fit gap plus
# RECESS_WEB_MM to them, the insert bores their 1.2 mm wall.
PAIR_LM_BAR_TOP_Y = 315.68


def _pair_lowest_centre():
    lm = unary_union([sbox(-37.0, 0.0, 37.0, PAIR_LM_BAR_TOP_Y),
                      *[Point(sx * 32.0, 315.770102).buffer(9.8 / 2, resolution=64) for sx in (-1, 1)]])
    web = [_R, _O, *_P] + ([_CH] if _CH is not None else [])
    walls = [q.buffer(PILOT_D_MM / 2 + OPENING_PILOT_WALL_MM, resolution=32) for q in _PIL]
    cy = 300.0
    while not (all(affinity.translate(g, 0.0, cy).distance(lm) >= 0.2 + RECESS_WEB_MM for g in web)
               and all(affinity.translate(g, 0.0, cy).distance(lm) >= 0.2 for g in walls)):
        cy += 0.25
    return cy


if INTEGRATED:
    PLATE_CENTER_Y = ((_pair_lowest_centre() if PAIR else _integrated_lowest_centre())
                      + float(os.environ.get('LX_DIPOLE_T_INT_RAISE_MM', '0')))   # study knob
    PLATE_MARGIN_BELOW = PLATE_CENTER_Y - EXT_H / 2 - RECESS_CLEAR_MM - SPLIT_Y   # may be below the old split
PLATE_RECESS_TOP_Y = PLATE_CENTER_Y + EXT_H / 2 + RECESS_CLEAR_MM


def _mu10_shift():
    """Lowest MU10 rise with its flange recess >= RECESS_WEB_MM clear of the driver's recess and window, and clear of
    the blind pockets and the pilots' walls (bisection). For a plain rectangular recess this is recess top + web + R."""
    place = lambda g: affinity.translate(g, 0.0, PLATE_CENTER_Y)
    hard = place(unary_union([_R, _O]))
    soft = place(unary_union([*_P, *[q.buffer(PILOT_D_MM / 2 + OPENING_PILOT_WALL_MM, resolution=32) for q in _PIL]]))
    def ok(s):
        d = Point(UM_CUTOUT[0], UM_CUTOUT[1] + s).buffer(UM_RECESS_R, resolution=256)
        return d.distance(hard) >= RECESS_WEB_MM and not d.intersects(soft)
    lo, hi = 0.0, 200.0
    while hi - lo > 1e-6:
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return round(hi, 3)


MU10_SHIFT = _mu10_shift()                  # PT2522 portrait 99.0
MU10_CENTER = (UM_CUTOUT[0], UM_CUTOUT[1] + MU10_SHIFT)
# Beside the driver the body is as thick as the driver: front on its front face, rear on its body's rear.
FRAME_REAR_Z = THICKNESS_MM - DRIVER.depth_behind_front             # PT2522 11.3, AMT17D2.1 8.3
FRAME_PILOT_DEPTH_MM = THICKNESS_MM - PLATE_T - FRAME_REAR_Z - FRAME_SKIN_MM
if INTEGRATED: FRAME_PILOT_DEPTH_MM = min(FRAME_PILOT_DEPTH_MM, M3_INSERT_BORE_DEPTH_MM)
# Outline and edge (symmetric). The side edge sits BODY_HALF_W out; beside the planar the front flares over
# RAZOR_FLARE_MM from flat to a RAZOR_EDGE_MM edge (a smootherstep S-curve: tangent to the front where it starts,
# tangent to the rear at the edge). The flat front runs SIDE_WALL_MM past the driver's outermost feature.
# The T duct belongs to the driver (DRIVER.t_duct_d: PT2522 D6 for its T cable, AMT D4 for its leads), in every part;
# where the part is too thin for it, it runs in a rear cover of the released lumen cover's section (T_COVER_WALL_MM).
# The UM (MU10) cable, free behind the released UM, gets a rear guide beside the window (UM_GUIDE_*).
SIDE_WALL_MM = 3.8
RAZOR_FLARE_MM = 8.0
if INTEGRATED: SIDE_WALL_MM, RAZOR_FLARE_MM = 2.0, 3.0      # narrower: a short edge taper, 2 mm past the features
# The UM (MU10) cable leaves the LM's UM duct on +x (released route) and runs hidden inside the T-below parts in a
# guide on that same side: it starts on the released cable at the LM exit, climbs into the part inside a short rear
# cover continuing the LM's duct cover, rises hidden beside the window (behind / outboard of the T duct) and ends in
# the raised MU10's open back near its terminals (see um_guide). The frame's depth is not increased: where it is too shallow for a round
# D8.2 duct the guide is a flat stadium of the same section area. The lane width below sizes the (symmetric) body.
from .route import DUCT_D as UM_GUIDE_D                          # 8.2, the project's UM cable duct
UM_GUIDE_WALL_MM = 0.8
# It must fit the shallowest stretch it crosses: the flush frame beside the PT2522 (7.0 mm) and, for both drivers, the
# ring band under the MU10 seat on the way into the MU10's back (UM_SEAT_Z - CORE_REAR_Z = 7.5 mm), with
# UM_GUIDE_SLACK_MM to spare. So one flat 11.6 x 5.0 stadium (D8.2 area) for the UM cable in every T-below part.
UM_GUIDE_SLACK_MM = 0.4
_frame_depth = min(drv().depth_behind_front for drv in DRIVERS.values())   # the same cable gets the same guide: the
_inner_h = min(_frame_depth, UM_SEAT_Z - CORE_REAR_Z) - 2 * UM_GUIDE_WALL_MM - UM_GUIDE_SLACK_MM   # shallowest frame
UM_GUIDE_H = min(UM_GUIDE_D, round(_inner_h, 3))                   # 5.0 for both drivers
UM_GUIDE_W = UM_GUIDE_H + (math.pi * UM_GUIDE_D ** 2 / 4 - math.pi * UM_GUIDE_H ** 2 / 4) / UM_GUIDE_H   # same area
UM_GUIDE_LANE_Z = FRAME_REAR_Z + UM_GUIDE_WALL_MM + UM_GUIDE_H / 2
# Integrated AMT: its 10 mm frame walls a round D8.2 beside the window (flat only into the MU10's seat band), so its
# lane is sized for the round duct; the PT2522's 7 mm frame keeps the flat one.
_round_lane = INTEGRATED and THICKNESS_MM - FRAME_REAR_Z >= UM_GUIDE_D + 2 * UM_GUIDE_WALL_MM
UM_LANE_W, UM_LANE_H = (UM_GUIDE_D, UM_GUIDE_D) if _round_lane else (UM_GUIDE_W, UM_GUIDE_H)


def _guide_lane_x():
    """Lane centre: walls clear of the window, and of every pilot / pocket whose depth overlaps the guide's."""
    half = UM_LANE_W / 2 + UM_GUIDE_WALL_MM
    x = _O.bounds[2] + half
    zc = FRAME_REAR_Z + UM_GUIDE_WALL_MM + UM_LANE_H / 2
    z0, z1 = zc - UM_LANE_H / 2 - UM_GUIDE_WALL_MM, zc + UM_LANE_H / 2 + UM_GUIDE_WALL_MM
    floor = THICKNESS_MM - PLATE_T
    feats = [(q.x, PILOT_D_MM / 2 + OPENING_PILOT_WALL_MM - UM_GUIDE_WALL_MM, floor - FRAME_PILOT_DEPTH_MM, floor)
             for q in _PIL]                                          # pilots keep their 1.2 mm wall
    feats += [(g.centroid.x, (g.bounds[2] - g.bounds[0]) / 2, floor - dz, floor) for g, (_, _, _, dz) in zip(_P, DRIVER.pockets())]
    for fx, fr, fz0, fz1 in feats:
        if fx > 0 and fz0 < z1 and fz1 > z0: x = max(x, fx + fr + half)
    return x


UM_GUIDE_LANE_X = _guide_lane_x()
UM_GUIDE_R_OUT = UM_LANE_W / 2 + UM_GUIDE_WALL_MM                # half-width of the guide (lane) with its walls
BODY_HALF_W = max(FEATURE_HALF_W + SIDE_WALL_MM, UM_GUIDE_LANE_X + UM_GUIDE_R_OUT) + RAZOR_FLARE_MM
# Integrated AMT rotated: its upper insert bore sits beside the UM duct's lane, which would otherwise end exactly at the
# flat front's edge -- 1 mm more frame per side gives the duct its slack.
if INTEGRATED: BODY_HALF_W += {('amt17d2_1', 'landscape'): 1.0}.get((DRIVER_KEY, ORIENTATION), 0.0)
RAZOR_EDGE_MM = 0.3
# Tangent blends: the outline from the LM-joint piece to the full width, and from the full width into the ring.
# The body leaves the LM-joint ear boss (D9.8 at x=+/-32, y=315.77) tangentially at its equator, where the circle is
# vertical, and widens along a smootherstep to the full width: no corner between ear and flare. Below the split it
# is only the front layer (the UM half-lap's own layer), clear of the LM tongue behind.
EAR_Y = 315.770102
EAR_OUTER_X = 32.0 + 9.8 / 2
BOTTOM_BLEND_Y = (EAR_Y, SPLIT_Y + 15.0)
BOTTOM_HALF_W = EAR_OUTER_X
FRONT_LAYER_REAR_Z = 12.6                   # below the split: in front of the LM tongue (UM half-lap layer 12.4)
RAZOR_FADE_BOTTOM_Y = (318.0, 332.0)        # the razor flare starts from the ear's full-thickness front
if INTEGRATED:   # one piece with the LM top: no joint ears to taper to, no LM tongue behind -- the frame's sides run
    # straight down into the LM ring (lm_um_t trims the body to outside the woofer's flange-recess circle), full depth
    BOTTOM_BLEND_Y, BOTTOM_HALF_W, FRONT_LAYER_REAR_Z = (288.0, 289.0), BODY_HALF_W, CORE_REAR_Z
LM_JOINT_TOP_HALF_W = 31.8                  # the LM-joint piece's full-height front at the split
LM_JOINT_REAR_Z = CORE_REAR_Z               # the LM-joint piece's rear base at the split (6.8)
TOP_BLEND_Y = (PLATE_RECESS_TOP_Y + 1.0, MU10_CENTER[1] - 8.0)       # outline and rear blend into the ring
RING_OUTER_R = 52.64
RING_REAR_Z = CORE_REAR_Z                   # 6.8
# The rear stays flush with the planar behind its housing, then reaches the ring's rear plane at the ring's bottom.
REAR_TOP_BLEND_Y = (PLATE_CENTER_Y + OPENING_TOP, MU10_CENTER[1] - RING_OUTER_R)
if REAR_TOP_BLEND_Y[1] - REAR_TOP_BLEND_Y[0] < 5.0:          # portrait has 7.96 mm, landscape 3.3 mm
    # The window ends just below the ring, so the frame thickens behind the window's top edge instead
    # (still a tangent blend, over 10 mm).
    REAR_TOP_BLEND_Y = (REAR_TOP_BLEND_Y[1] - 10.0, REAR_TOP_BLEND_Y[1])
LM_FIT_GAP_MM = 0.2                         # the body keeps this clear of the LM's half-lap tongues
SPLIT_REAR_BLEND_MM = 12.0                  # from the LM-joint piece's own rear profile (its duct covers) to the frame
# Below a window raised well above the split the rear holds the joint piece's profile (no deeper than the part already
# is) and only blends to the flush frame over the last SPLIT_REAR_BLEND_MM under the window: room for the T duct and
# the UM guide's climb there (the flush 7 mm frame cannot wall a D6 duct); the blend ends SPLIT_REAR_HOLD_PAST_MM up
# beside the window's lowest corners, so the T duct still has that depth where it enters the window.
SPLIT_REAR_HOLD_PAST_MM = 10.0
SPLIT_REAR_HOLD_MM = (PLATE_MARGIN_BELOW - SPLIT_REAR_BLEND_MM + SPLIT_REAR_HOLD_PAST_MM
                      if PLATE_MARGIN_BELOW > SPLIT_REAR_BLEND_MM else 0.0)
LIP_FILL_HALF_W, LIP_FILL_Z0 = 27.5, 15.0   # the released UM's front lip relief, filled flush
RING_CUTOUT_R = UM_CUTOUT[2] / 2
RING_BODY_CLEAR_R = 50.5                    # the sculpted body stays outside this; the ring band carries the rest
TOP_CUT_Y = 395.0                           # above this (released UM coordinates) the crescent horns go
RING_FILL_INNER_R = 49.35
T_BRANCH_D = DRIVER.t_duct_d
# A T duct narrower than the D6 T lumen it leaves (the AMT's D4) starts at the lumen's own D6 and narrows over
# T_TAPER_MM along a smootherstep, so a cable fished up from the LM meets no shoulder (no inner corner) at the branch.
T_TAPER_MM = 15.0


def t_taper_d(arc_mm):
    """T duct diameter at `arc_mm` from its branch off the D6 T lumen."""
    import numpy as np
    from .route import TS_DUCT_D
    return T_BRANCH_D + max(TS_DUCT_D - T_BRANCH_D, 0.0) * (1.0 - _smootherstep(np.asarray(arc_mm, float) / T_TAPER_MM))
# Where the round T duct cannot stay hidden in the flush frame, a flat (stadium) duct of the same section area.
T_FLAT = {}.get((DRIVER_KEY, ORIENTATION))        # flat T duct depth where a round one cannot stay hidden (none needed)
T_FLAT_W = (T_FLAT + (math.pi * T_BRANCH_D ** 2 / 4 - math.pi * T_FLAT ** 2 / 4) / T_FLAT) if T_FLAT else None
UM_T_LUMEN_AT_SPLIT = (20.8, 9.8)


def _smootherstep(t):
    import numpy as np
    t = np.clip(t, 0.0, 1.0); return t * t * t * (t * (t * 6 - 15) + 10)


def body_outline_half_width(y):
    """Symmetric half-width of the sculpted body at height y (C2 blends at both ends)."""
    import numpy as np
    y = np.asarray(y, float)
    w = BODY_HALF_W + (BOTTOM_HALF_W - BODY_HALF_W) * (1 - _smootherstep((y - BOTTOM_BLEND_Y[0]) / (BOTTOM_BLEND_Y[1] - BOTTOM_BLEND_Y[0])))
    ring = np.sqrt(np.clip(RING_OUTER_R ** 2 - (y - MU10_CENTER[1]) ** 2, 0, None))
    top = _smootherstep((y - TOP_BLEND_Y[0]) / (TOP_BLEND_Y[1] - TOP_BLEND_Y[0]))
    return w + (np.maximum(ring, 0) - w) * top


def body_rear_z(y):
    import numpy as np
    y = np.asarray(y, float)
    z = LM_JOINT_REAR_Z + (FRAME_REAR_Z - LM_JOINT_REAR_Z) * _smootherstep((y - BOTTOM_BLEND_Y[0]) / (BOTTOM_BLEND_Y[1] - BOTTOM_BLEND_Y[0]))
    y0, y1 = REAR_TOP_BLEND_Y
    return z + (RING_REAR_Z - z) * _smootherstep((y - y0) / (y1 - y0))


def body_front_z(x, y):
    """Front surface: flat, then the razor flare beside the planar; square (ring-like) edge towards the ring."""
    import numpy as np
    x, y = np.asarray(x, float), np.asarray(y, float)
    rear = body_rear_z(y); full = THICKNESS_MM - rear
    s = body_outline_half_width(y) - np.abs(x)
    razor = (1 - _smootherstep((y - TOP_BLEND_Y[0]) / (TOP_BLEND_Y[1] - TOP_BLEND_Y[0]))) * \
        _smootherstep((y - RAZOR_FADE_BOTTOM_Y[0]) / (RAZOR_FADE_BOTTOM_Y[1] - RAZOR_FADE_BOTTOM_Y[0]))   # razor only beside the planar
    # The front stays flat over the LM-joint piece's full-height front, so the flare narrows near the split.
    flare = np.clip(body_outline_half_width(y) - LM_JOINT_TOP_HALF_W, 1.0, RAZOR_FLARE_MM)
    g = _smootherstep(s / flare)
    g = razor * g + (1 - razor)
    return rear + np.maximum(full * g, RAZOR_EDGE_MM)


def um_t_body(split_rear=None):
    """The sculpted body joining the LM-joint piece, the planar and the ring (manifold3d Manifold).

    split_rear: (xs, zs), the LM-joint piece's rear surface along the split (NaN where it has no material); the
    body's rear starts there and blends tangentially to the frame rear, so the joint's duct covers fade out."""
    import manifold3d as m3
    import numpy as np
    y0, y1 = BOTTOM_BLEND_Y[0], TOP_BLEND_Y[1]
    # Subdivided in real millimetres (0.6 mm), then warped: x scales to the outline, z spans rear to front.
    unit = m3.Manifold.cube([2 * BODY_HALF_W, y1 - y0, 1.0]).translate([-BODY_HALF_W, y0, 0.0]).refine_to_length(0.6)

    def warp(v):
        v = np.array(v, float); u, y, h = v[:, 0] / BODY_HALF_W, v[:, 1], v[:, 2]
        x = u * body_outline_half_width(y)
        rear = body_rear_z(y); front = body_front_z(x, y)
        w = _smootherstep((y - SPLIT_Y - SPLIT_REAR_HOLD_MM) / SPLIT_REAR_BLEND_MM)   # 0 at the split (and hold) -> 1 at the frame
        # Outboard of the joint piece: the front layer below the split (the LM's half-lap tongue is behind it); above
        # the split (no LM there) the rear eases in over 4 mm to the inboard rear: no cliff at the joint piece's edge.
        held = LM_JOINT_REAR_Z + (FRAME_REAR_Z - LM_JOINT_REAR_Z) * w
        outboard = np.where(y < SPLIT_Y, FRONT_LAYER_REAR_Z,
                            FRONT_LAYER_REAR_Z + (held - FRONT_LAYER_REAR_Z) * _smootherstep((y - SPLIT_Y) / 4.0))
        if split_rear is not None:
            zs = np.interp(x, split_rear[0], split_rear[1], left=np.nan, right=np.nan)
            # Below the split follow the joint piece's own rear exactly (incl. its half-lap layers, clear of the LM
            # tongue); above it blend from the smoothed profile to the frame rear. No jump at the split.
            zs_raw = np.interp(x, split_rear[0], split_rear[2], left=np.nan, right=np.nan)
            inboard = np.where(y < SPLIT_Y, zs_raw, zs + (FRAME_REAR_Z - zs) * w)
            near = np.where(np.isnan(zs), outboard, inboard)
        else:
            near = outboard
        # Near the bottom the joint profile governs; higher up the regular rear (frame, then into the ring) takes over.
        use_near = y < SPLIT_Y + SPLIT_REAR_HOLD_MM + SPLIT_REAR_BLEND_MM
        rear = np.where(use_near, near, rear)
        rear = np.minimum(rear, front - RAZOR_EDGE_MM)
        return np.c_[x, y, rear + h * (front - rear)]
    body = unit.warp_batch(warp)
    ring_hole = m3.Manifold.cylinder(100, RING_BODY_CLEAR_R, RING_BODY_CLEAR_R, 256).translate([MU10_CENTER[0], MU10_CENTER[1], -40])
    return body - ring_hole


def _pilot_rear_z(x, y):
    """Conservative rear face under a pilot: just above the split, outboard of the LM-joint piece, the front layer
    (FRONT_LAYER_REAR_Z) still blends to the flush frame rear."""
    if abs(x) > LM_JOINT_TOP_HALF_W and y < SPLIT_Y + SPLIT_REAR_HOLD_MM + SPLIT_REAR_BLEND_MM:
        w = float(_smootherstep((y - SPLIT_Y - SPLIT_REAR_HOLD_MM) / SPLIT_REAR_BLEND_MM))
        held = LM_JOINT_REAR_Z + (FRAME_REAR_Z - LM_JOINT_REAR_Z) * w
        return FRONT_LAYER_REAR_Z + (held - FRONT_LAYER_REAR_Z) * float(_smootherstep((y - SPLIT_Y) / 4.0))
    return FRAME_REAR_Z


def um_t_cutters():
    cutters = planar_cutters(0.0, PLATE_CENTER_Y, z_rear=0.0, pilot_depth=FRAME_PILOT_DEPTH_MM, part_rear=FRAME_REAR_Z)[0]
    # pilots only as deep as the material under them allows (0.8 mm floor), e.g. low on the rotated PT2522
    floor = THICKNESS_MM - PLATE_T
    for q in _PIL:
        x, y = q.x, PLATE_CENTER_Y + q.y
        depth = min(FRAME_PILOT_DEPTH_MM, floor - _pilot_rear_z(x, y) - FRAME_SKIN_MM)
        if depth < FRAME_PILOT_DEPTH_MM - 1e-6:
            cutters = [c for c in cutters if not _is_pilot_at(c, x, y)]
            cutters.append(_cylinder_at(x, y, PILOT_D_MM / 2, floor - depth, floor + 0.5))
    return cutters


def _is_pilot_at(c, x, y):
    bb = c.bounding_box()
    return abs((bb.min.X + bb.max.X) / 2 - x) < 0.01 and abs((bb.min.Y + bb.max.Y) / 2 - y) < 0.01 and \
        abs((bb.max.X - bb.min.X) - PILOT_D_MM) < 0.05


def um_t_tools():
    """Driver mount cutters and the two lead tubes (from the routes file)."""
    cutters = um_t_cutters()
    key = f'{DRIVER_KEY}/{ORIENTATION}'
    routes = json.loads(UM_T_ROUTES_FILE.read_text()) if UM_T_ROUTES_FILE.exists() else {}
    if key not in routes:
        raise RuntimeError(f'no searched lead routes for {key} in {UM_T_ROUTES_FILE.name}')
    tubes = []
    for lead, d_expected in (('t', T_BRANCH_D),):
        rt = routes[key][lead]
        import numpy as np
        pts = np.asarray(rt['points'], float)
        if abs(rt['d'] - d_expected) > 1e-6: raise RuntimeError(f'{key} {lead}: route was searched for D{rt["d"]}, not D{d_expected}')
        r_min = min_bend_radius(pts[rt.get('bend_from', 0):])
        if r_min < rt['min_bend_r_mm'] - 0.05: raise RuntimeError(f'{key} {lead}: bends at R{r_min:.1f} < recorded R{rt["min_bend_r_mm"]}')
        tubes.append((pts[::2].tolist() + [pts[-1].tolist()], d_expected / 2))
    return cutters, tubes


# Lead routes for the T-below parts: smooth B-spline centrelines (no corners) searched against each part's own solid
# for the largest minimum bend radius with every point of the duct skin in material (see the file's provenance).
UM_T_ROUTES_FILE = Path(os.environ.get('LX_DIPOLE_T_ROUTES', Path(__file__).with_name(
    'dipole_tweeter_pair_routes.json' if PAIR else 'dipole_tweeter_lm_routes.json' if INTEGRATED
    else 'dipole_tweeter_routes.json')))   # env: route studies
UM_T_FILL_FROM_Y = 309.0            # the released T lumen is filled from here on; the T branch leaves it before
# The LM--UM M2 tie (carriers.LM_UM_TIE_*): the lower piece keeps its clearance bore and counterbore; with the MU10
# raised, its opening no longer reaches the screw head, so a sleeve and the D4.4 counterbore continue up into the
# driver window (screw driven through the window, driver out). The stub that rose with the ring is filled.
from .carriers import LM_UM_TIE_X, LM_UM_TIE_AXIS_Z, LM_UM_TIE_CBORE_D
TIE = dict(x=LM_UM_TIE_X, z=LM_UM_TIE_AXIS_Z, y0=319.5, y1=330.0, r=2.4, cbore_r=LM_UM_TIE_CBORE_D / 2, sleeve_wall=1.2)


def tie_window_y():
    """Where the tie axis meets the driver window's lower edge (installed y)."""
    ln = LineString([(TIE['x'], -100.0), (TIE['x'], 100.0)]).intersection(_O)
    if ln.is_empty: raise RuntimeError('the M2 tie axis misses the driver window')
    return PLATE_CENTER_Y + ln.bounds[1]


T_COVER_WALL_MM = 0.8               # the released T lumen cover: R3.8 about a D6 lumen
UM_GUIDE_Z = 1.85                   # the free UM cable's depth where it passes the split (released route)
# The project's UM cable bend rule (LM exit R14). AMT upright: R10 accepted by the owner as an exception (2026-10-02),
# the guide has to turn out of the released cable onto its lane beside the AMT's window within a short climb (its
# route: R11.7 including the join to the released cable). The AMT rotated's T duct (D4) bends at R10 beside the
# driver's mounting pilot it has to pass outboard of (owner exception 2026-10-03; see the routes file).
UM_GUIDE_MIN_BEND_R = {('pt2522_4', 'portrait'): 14.0, ('pt2522_4', 'landscape'): 14.0,
                       ('amt17d2_1', 'portrait'): 10.0, ('amt17d2_1', 'landscape'): 14.0}[(DRIVER_KEY, ORIENTATION)]


def stadium_tube(points, w, h):
    """A tube of stadium section (w wide in the part's plane, h deep; scalars or one value per point) along a
    polyline: per segment the hull of four spheres of radius h/2, offset +/-(w-h)/2 sideways."""
    import manifold3d as m3
    import numpy as np
    P = np.asarray(points, float)
    w = np.broadcast_to(np.asarray(w, float), len(P)); h = np.broadcast_to(np.asarray(h, float), len(P))
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
    side = np.cross(T, [0.0, 0.0, 1.0]); n = np.linalg.norm(side, axis=1, keepdims=True)
    side = np.where(n > 1e-6, side / np.maximum(n, 1e-12), [1.0, 0.0, 0.0])
    off = (w - h) / 2
    balls = [(m3.Manifold.sphere(hh / 2, 24).translate(list(p + o * sd)), m3.Manifold.sphere(hh / 2, 24).translate(list(p - o * sd)))
             for p, sd, o, hh in zip(P, side, off, h)]
    segs = [m3.Manifold.batch_hull([a[0], a[1], b[0], b[1]]) for a, b in zip(balls, balls[1:])]
    return m3.Manifold.batch_boolean(segs, m3.OpType.Add)


UM_GUIDE_RING_Z = (CORE_REAR_Z + UM_SEAT_Z) / 2                     # centred in the ring band under the MU10 seat


def um_guide():
    """Centreline of the hidden UM cable guide (routes file, key 'um') and the length of its rear cover.

    It starts ON the released UM cable where it leaves the LM's UM duct (same point, same tangent). The cable sits
    well behind the part there (z~1.6), so over its first `cover` mm the guide climbs inside a rear cover that continues
    the LM's own duct cover (stadium, UM_GUIDE_WALL_MM wall, clear of the LM by its fit gap) until it is buried in the
    frame; from there it runs hidden beside the window on the +x side (behind / outboard of the T duct) and ends
    radially in the raised MU10's open back near its terminals. Searched against the part's own solid; every bend
    >= UM_GUIDE_MIN_BEND_R."""
    import numpy as np
    from .route import route_cable_points
    key = f'{DRIVER_KEY}/{ORIENTATION}'
    routes = json.loads(UM_T_ROUTES_FILE.read_text()) if UM_T_ROUTES_FILE.exists() else {}
    if 'um' not in routes.get(key, {}):
        raise RuntimeError(f'no searched UM cable guide for {key} in {UM_T_ROUTES_FILE.name}')
    rt = routes[key]['um']
    pts = np.asarray(rt['points'], float)
    if abs(rt['w'] - UM_GUIDE_W) > 1e-6 or abs(rt['h'] - UM_GUIDE_H) > 1e-6:
        raise RuntimeError(f'{key} um: guide searched for {rt["w"]}x{rt["h"]}, not {UM_GUIDE_W:.3f}x{UM_GUIDE_H:.3f}')
    rel = np.asarray(route_cable_points(0.5))
    if np.linalg.norm(rel - pts[0], axis=1).min() > 0.01:
        raise RuntimeError(f'{key} um: the guide does not start on the released UM cable')
    r_min = min_bend_radius(pts)
    if r_min < max(rt['min_bend_r_mm'] - 0.05, UM_GUIDE_MIN_BEND_R):
        raise RuntimeError(f'{key} um: bends at R{r_min:.1f} (recorded R{rt["min_bend_r_mm"]}, rule R{UM_GUIDE_MIN_BEND_R})')
    if rt.get('section_knots'): w, h = section_from_knots(pts, rt['section_knots'])
    else: w, h = um_guide_section(pts, rt.get('round_to_mm'), rt.get('flat_from_mm'))
    return pts, float(rt['cover_arc_mm']), w, h


def section_from_knots(pts, knots):
    """Per-point duct section from [[arc_mm, w, h], ...] knots: smootherstep between neighbouring knots."""
    import numpy as np
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.asarray(pts, float), axis=0), axis=1))]
    K = np.asarray(knots, float); w = np.full(len(arc), K[0, 1]); h = np.full(len(arc), K[0, 2])
    for (a0, w0, h0), (a1, w1, h1) in zip(K, K[1:]):
        t = _smootherstep((arc - a0) / max(a1 - a0, 1e-6)); sel = arc >= a0
        w = np.where(sel, w0 + (w1 - w0) * t, w); h = np.where(sel, h0 + (h1 - h0) * t, h)
    return w, h


def um_guide_section(pts, round_to=None, flat_from=None):
    """Per-point guide section (w, h): round UM_GUIDE_D inside its rear cover up to `round_to` mm of arc (depth is
    free there and the round section is narrower beside the LM), easing (smootherstep) to the flat UM_GUIDE_W x
    UM_GUIDE_H by `flat_from`, flat from there on. Without them the guide is flat throughout."""
    import numpy as np
    arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(np.asarray(pts, float), axis=0), axis=1))]
    t = np.ones(len(arc)) if round_to is None else _smootherstep((arc - round_to) / max(flat_from - round_to, 1e-6))
    return UM_GUIDE_D + (UM_GUIDE_W - UM_GUIDE_D) * t, UM_GUIDE_D + (UM_GUIDE_H - UM_GUIDE_D) * t


def um_t_mesh(um_installed, tessellate, lm_installed, solid_only=False, stage=None):
    """Mesh pipeline (exact BREP booleans on the full UM exceed 40 GB).

    Lower piece: the released UM below SPLIT_Y (complete LM interface and both lead entries). Ring: the released
    UM above SPLIT_Y without the crescent mount (horns, receiver notches, T lumen and its rear cover) and without
    the wing-magnet flats and pockets, completed to a full circle, raised by MU10_SHIFT. The sculpted body joins
    them around the planar. `tessellate(shape) -> trimesh`. Route-search stages: 'pre_t' (= solid_only: no T duct,
    no UM guide; the guide is searched against it), 'pre_t_guide' (UM guide and its cover, no T duct; the T duct is
    searched against it), 'pre_guide' (T duct, no UM guide); default 'final'. 'lm_base' (integrated variant,
    lm_um_t.py): the lower piece, body and ring only -- no LM clearance (the LM top is fused on), no M2 tie, no ducts,
    no driver cutters."""
    import manifold3d as m3
    import numpy as np
    import trimesh
    from .route import TS_DUCT_D, ts_cable_points

    def M(t): return m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    def box(x0, y0, z0, x1, y1, z1): return m3.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    def disc(r, z0=-50.0, z1=80.0, c=UM_CUTOUT):
        return m3.Manifold.cylinder(z1 - z0, r, r, 256).translate([c[0], c[1], z0])

    um = M(um_installed)
    path = [tuple(p) for p in np.asarray(ts_cable_points(1.0)) if p[1] >= UM_T_FILL_FROM_Y - 1.0]
    balls = [m3.Manifold.sphere(TS_DUCT_D / 2 * 1.08, 32).translate(list(p)) for p in path]
    fill = m3.Manifold.batch_boolean([m3.Manifold.batch_hull([a, b]) for a, b in zip(balls, balls[1:])], m3.OpType.Add)
    # Below the split: inside the UM's own hull and clear of the LM (with its fit gap), so the lumen fill never adds
    # material where the released UM has none or where the LM goes.
    lm_fit = (M(lm_installed) ^ box(-200, SPLIT_Y - 20, -50, 200, SPLIT_Y + 20, 80)).minkowski_sum(
        m3.Manifold.cube([2 * LM_FIT_GAP_MM] * 3, center=True))
    um = um + ((fill ^ box(-200, UM_T_FILL_FROM_Y, -50, 200, SPLIT_Y, 80) ^ um.hull()) - lm_fit) + (fill ^ box(-200, SPLIT_Y, -50, 200, 600, 80) ^ disc(51.9))
    b = TIE                          # the tie stub above the split rises with the ring: fill it there only
    plug = m3.Manifold.cylinder(b['y1'] - SPLIT_Y, b['r'], b['r'], 48).rotate([-90, 0, 0]).translate([b['x'], SPLIT_Y, b['z']])
    # trimmed on the UM's own faceted opening wall (its mesh vertices there), so the plug leaves no trace on the wall
    v = np.asarray(um_installed.vertices)
    rr = np.hypot(v[:, 0] - UM_CUTOUT[0], v[:, 1] - UM_CUTOUT[1])
    w = v[(np.abs(rr - UM_CUTOUT[2] / 2) < 0.05) & (np.abs(v[:, 0] - b['x']) < 9.0) & (v[:, 1] < UM_CUTOUT[1])]
    ang = np.arctan2(w[:, 1] - UM_CUTOUT[1], w[:, 0] - UM_CUTOUT[0])
    wall = np.unique(np.round(np.c_[ang, w[:, 0], w[:, 1]], 4), axis=0)
    from shapely.geometry import Polygon as _Pg
    from shapely.geometry.polygon import orient as _or
    fan = _or(_Pg([(UM_CUTOUT[0], UM_CUTOUT[1]), *[(x, y) for _, x, y in wall]]), sign=1.0)
    opening = m3.CrossSection([np.asarray(fan.exterior.coords)[:-1]]).extrude(60).translate([0, 0, -20])
    um = um + (plug - opening)
    # Ring: the released UM above the split, flat rear, no horns; outer band made a clean circle at full depth; the
    # lower arc rebuilt as a clean annulus (below the split it belonged to the LM-joint bar).
    ring = um ^ box(-1000, SPLIT_Y, CORE_REAR_Z, 1000, SPLIT_Y + 1000, 1000)
    ring = ring - (box(-200, TOP_CUT_Y, -50, 200, 600, 80) - disc(RING_OUTER_R))
    ring = ring + (disc(RING_OUTER_R, CORE_REAR_Z, THICKNESS_MM) - disc(RING_FILL_INNER_R))
    arc = (disc(RING_OUTER_R, CORE_REAR_Z, THICKNESS_MM) - disc(RING_CUTOUT_R)) ^ box(-200, 0, -50, 200, SPLIT_Y + 0.5, 80)
    arc = arc - disc(UM_RECESS_R, UM_SEAT_Z, THICKNESS_MM + 1)
    ring = (ring + arc).translate([0, MU10_SHIFT, 0])
    lower = um ^ box(-1000, SPLIT_Y - 1000, -1000, 1000, SPLIT_Y, 1000)
    cutters = um_t_cutters()
    stage = stage or ('pre_t' if solid_only else 'final')
    tubes = None if stage in ('pre_t', 'pre_t_guide', 'lm_base', 'pair_base') else um_t_tools()[1]
    def chain(points, r):
        balls = [m3.Manifold.sphere(r, 32).translate(list(map(float, q))) for q in points]
        return m3.Manifold.batch_boolean([m3.Manifold.batch_hull([a, b]) for a, b in zip(balls, balls[1:])], m3.OpType.Add)
    # Keep the body clear of the LM's half-lap tongues (they reach up into the UM zone around the split).
    lm = M(lm_installed) ^ box(-200, SPLIT_Y - 20, -50, 200, SPLIT_Y + 20, 80)
    lm = lm.minkowski_sum(m3.Manifold.cube([2 * LM_FIT_GAP_MM] * 3, center=True))
    if stage == 'lm_base': lm = m3.Manifold()          # fused with the LM top: no clearance to keep
    # 'pair_base' (pair variant, lm_um_t.py): as 'lm_base', but the piece stays a separate print on the released
    # LM--UM joint -- the LM's fit gap is kept, and the M2 tie keeps its sleeve and counterbore up into the window.
    # The LM-joint piece's rear along the split (its base and duct covers), for a tangent rear blend.
    lw = lower.to_mesh(); lmesh = trimesh.Trimesh(np.asarray(lw.vert_properties)[:, :3], np.asarray(lw.tri_verts))
    xs = np.arange(-BODY_HALF_W, BODY_HALF_W + 0.01, 0.25)
    loc, idx, _ = lmesh.ray.intersects_location(np.c_[xs, np.full_like(xs, SPLIT_Y - 0.05), np.full_like(xs, -20.0)],
                                                np.tile([0, 0, 1.0], (len(xs), 1)), multiple_hits=False)
    zs = np.full(len(xs), np.nan); zs[idx] = loc[:, 2]
    zs_raw = zs.copy()
    # The half-lap layers (front-only, z~12.4: the LM tongue fills their rear below the split) are not rear surfaces
    # to follow above the split; take the base there, then smooth so the blend carries no ridges.
    from scipy.ndimage import gaussian_filter1d
    zs = np.where(zs > LM_JOINT_REAR_Z + 1.0, LM_JOINT_REAR_Z, zs)
    inside = ~np.isnan(zs)
    zs[inside] = gaussian_filter1d(zs[inside], sigma=6)
    # Fill the released UM's front relief along its lower lip (for the regular wings' tongues and the closure web;
    # neither mounts on this part): flush to the front, inside the part's own outline, clear of the LM.
    # Bounded by the joint piece's own outline, so the fill always sits on its material.
    from shapely.geometry.polygon import orient
    sec = lmesh.section([0, 0, 1], [0, 0, 8.0])        # the full-depth body of the joint piece (outer outlines only)
    polys = [orient(pg, sign=1.0) for pg in sec.to_2D(to_2D=np.eye(4))[0].polygons_full] if sec is not None else []
    loops = [np.asarray(pg.exterior.coords)[:-1] for pg in polys]
    outline = m3.CrossSection(loops).extrude(THICKNESS_MM - LIP_FILL_Z0).translate([0, 0, LIP_FILL_Z0])
    lower = lower + ((box(-LIP_FILL_HALF_W, SPLIT_Y - 12.0, LIP_FILL_Z0, LIP_FILL_HALF_W, SPLIT_Y + 0.5, THICKNESS_MM) ^ outline) - lm)
    part = lower + (um_t_body((xs, zs, zs_raw)) - lm) + ring
    if stage == 'pair_base':
        yw = tie_window_y()
        cyl = lambda r, y0, y1: m3.Manifold.cylinder(y1 - y0, r, r, 48).rotate([-90, 0, 0]).translate([TIE['x'], y0, TIE['z']])
        part = part + ((cyl(TIE['cbore_r'] + TIE['sleeve_wall'], TIE['y0'], yw) ^ box(-200, 0, CORE_REAR_Z, 200, 600, 80)) - lm)
    if stage in ('lm_base', 'pair_base'):
        out = part.to_mesh()
        return trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)
    guide = stage in ('final', 'pre_t_guide')
    if guide:
        gpts, cover, gw, gh = um_guide()
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(gpts, axis=0), axis=1))]
        # rear cover over the guide's climb (continuing the LM's duct cover), clear of the LM; added before the driver
        # cutters so the window and pilots always cut it
        k = np.flatnonzero(arc <= cover + 2.0)[::2]
        if len(k) > 1:
            cov = [stadium_tube(gpts[k], gw[k] + 2 * UM_GUIDE_WALL_MM, gh[k] + 2 * UM_GUIDE_WALL_MM)]
            # where the cover runs behind the part's rear it is also grown 1 mm towards the front, so it fuses with
            # the rear (no air sliver between them); inside the frame it is not (it would break through the front)
            kb = k[gpts[k, 2] + gh[k] / 2 + UM_GUIDE_WALL_MM < FRAME_REAR_Z + 0.5]
            if len(kb) > 1:
                cov += [stadium_tube(gpts[kb] + [0.0, 0.0, dz], gw[kb] + 2 * UM_GUIDE_WALL_MM, gh[kb] + 2 * UM_GUIDE_WALL_MM) for dz in (0.5, 1.0)]
            part = part + (m3.Manifold.batch_boolean(cov, m3.OpType.Add) - lm)
    for c in cutters: part = part - M(tessellate(c))
    if guide:
        # the cut starts 3 mm back along the released cable, so the guide stays open towards the LM's exit
        from .route import route_cable_points
        rel = np.asarray(route_cable_points(0.5)); j = int(np.argmin(np.linalg.norm(rel - gpts[0], axis=1)))
        pre = rel[max(j - 6, 0):j]; k = np.r_[np.arange(0, len(gpts) - 1, 2), len(gpts) - 1]
        part = part - stadium_tube(np.r_[pre, gpts[k]], np.r_[np.full(len(pre), gw[0]), gw[k]], np.r_[np.full(len(pre), gh[0]), gh[k]])
    if tubes is not None and T_FLAT:                 # flat T duct: cut here, not as a round capsule chain
        part = part - stadium_tube(tubes[0][0], T_FLAT_W, T_FLAT)
        tubes = tubes[1:]
    # M2 tie access: sleeve (clear of the LM) and the counterbore up into the window
    yw = tie_window_y()
    cyl = lambda r, y0, y1: m3.Manifold.cylinder(y1 - y0, r, r, 48).rotate([-90, 0, 0]).translate([TIE['x'], y0, TIE['z']])
    sleeve = (cyl(TIE['cbore_r'] + TIE['sleeve_wall'], TIE['y0'], yw) ^ box(-200, 0, CORE_REAR_Z, 200, 600, 80)) - lm
    part = part + sleeve - cyl(TIE['cbore_r'], TIE['y0'], yw + 1.0)
    if part.status() != m3.Error.NoError or part.is_empty(): raise RuntimeError(f'UM + T boolean failed: {part.status()}')
    out = part.to_mesh()
    if stage in ('pre_t', 'pre_t_guide'):
        return trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)
    mesh = trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)
    if tubes: mesh = capsule_cut(mesh, tubes)
    bodies = mesh.split(only_watertight=False)
    main = max(bodies, key=lambda b: abs(b.volume))
    stray = [abs(b.volume) for b in bodies if b is not main and b.volume > 0]
    if any(v > 1.0 for v in stray): raise RuntimeError(f'UM + T: disconnected material {stray}')
    return main                              # no sealed cavities


def um_t_plate_center():
    return (0.0, PLATE_CENTER_Y)


# ======================= variant 3: one-piece UM + T above (AMT: integrated T routing) =======================
# The whole released UM with the tweeter frame fused on top (the crescent's plan, overlapping the ring instead of
# keeping the mate gap; the half-lap joint bosses become solid; upright, the neck is filled down onto the ring). The
# T leads branch off the UM's own T lumen high on the ring's right side, where the lumen already runs inside the body,
# and follow ONE smooth spline (tangent to the lumen at the branch, straight into the driver's lead channel at the
# end, every bend >= ABOVE_DUCT_MIN_BEND_R in plan and depth) to the driver. The lumen's old tail past the branch is
# filled and its rear cover removed: nothing shows front or rear.
ABOVE_DUCT_D = DRIVER.t_duct_d
ABOVE_DUCT_MIN_BEND_R = 15.0       # the route rule (UM terminal plan bend R15; LM exit R14)
# Upright: the duct stays in the T lumen past the MU10 pilot (it runs under it, as the lumen does), branches at
# (34.0, 397.5) and rises in one sweep (min R25.6) into the neck, inside a collar continuing the lumen's rear cover
# while it climbs out of the rear plane. Rotated: branches high (z 10.45, already inside the body) and runs up the
# saddle and the lead-side lane. Both found by searching branch point and handle lengths for the largest minimum
# radius with every skin point of the duct in solid material.
ABOVE_FILL_FROM_XY = (42.48, 384.62)    # the T lumen is filled from here on (still inside the body here); the duct
                                        # re-opens it up to its branch and leaves it there (see the routes file)
ABOVE_COLLAR_WALL_MM = bmr_pod.ENTRY_COLLAR_WALL_MM               # 1.2, the UM lumen covers' own wall
TS_COVER_OUTER_R = 3.8             # the released T lumen cover's outer radius (measured on the audited UM)
TS_COVER_CLEAR_MM = 6.0            # plan half-width cleared behind the rear plane along the old lumen tail (cover R4.2+)


INTERFACE_SECTOR_DEG = (40.0, 140.0)   # crescent-interface reliefs (half-lap receivers, mate slit) and the wing-magnet
                                        # pockets (51 / 129 deg): no wings on the fused parts, and the saddle would seal them
INTERFACE_MIN_R = 49.0                  # ... outboard of the MU10 flange recess (its pilots sit at R44.8)


def _interface_voids(um_installed):
    """Boxes covering the UM's crescent-interface reliefs (pockets that only exist for a separate crescent): every
    void in the ring band's top sector, outboard of R49, apart from the T lumen. The MU10 pilots (R44.8) and the wing
    magnet pockets (51 / 129 deg) are outside this test and stay."""
    import numpy as np
    from scipy import ndimage
    from scipy.spatial import cKDTree
    lum, _ = _lumen()
    xs = np.arange(-46.0, 46.01, 0.5); ys = np.arange(385.0, 420.01, 0.5); zs = np.arange(CORE_REAR_Z + 0.2, THICKNESS_MM, 0.5)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij'); P = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    R = np.hypot(P[:, 0] - UM_CUTOUT[0], P[:, 1] - UM_CUTOUT[1])
    cand = (R > 41.5) & (R < 52.3) & ~((R < UM_RECESS_R) & (P[:, 2] > UM_SEAT_Z)) & (cKDTree(lum).query(P)[0] > 3.3)
    void = np.zeros(len(P), bool); void[cand] = ~um_installed.contains(P[cand])
    lab, n = ndimage.label(void.reshape(X.shape))
    boxes = []
    for k in range(1, n + 1):
        idx = np.argwhere(lab == k)
        if len(idx) < 6: continue
        q = np.c_[xs[idx[:, 0]], ys[idx[:, 1]], zs[idx[:, 2]]]; c = q.mean(0)
        ang = math.degrees(math.atan2(c[1] - UM_CUTOUT[1], c[0] - UM_CUTOUT[0]))
        if INTERFACE_SECTOR_DEG[0] <= ang <= INTERFACE_SECTOR_DEG[1] and math.hypot(c[0], c[1] - UM_CUTOUT[1]) > INTERFACE_MIN_R:
            boxes.append((*(q.min(0) - 0.75), *(q.max(0) + 0.75)))
    return boxes


def _lumen(xy=ABOVE_FILL_FROM_XY):
    import numpy as np
    from .route import ts_cable_points
    p = np.asarray(ts_cable_points(0.5))
    i = int(np.argmin(np.hypot(p[:, 0] - xy[0], p[:, 1] - xy[1])))
    return p, i


def min_bend_radius(pts):
    import numpy as np
    a, b, c = pts[:-2], pts[1:-1], pts[2:]
    ab, bc, ca = np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    with np.errstate(divide='ignore'):
        r = np.where(area2 > 1e-12, ab * bc * ca / (2 * area2), np.inf)
    return float(r.min())


FUSED_PLATE_GAP_MM = 0.5            # the driver's plate stays this far outside the MU10 flange recess (both on the front)


def _above_centre():
    """One-piece UM + T above: the lowest driver centre (0.25 mm steps) with the plate clear of the MU10 flange recess,
    a RECESS_WEB_MM web from the recess to the window and lead channel, and the pockets and pilot walls clear of it."""
    rec = Point(*UM_CUTOUT[:2]).buffer(UM_RECESS_R, resolution=256)
    plate = _turn(DRIVER.plate())
    hard = unary_union([_O] + ([_CH] if _CH is not None else []))
    soft = unary_union([*_P, *[q.buffer(PILOT_D_MM / 2 + OPENING_PILOT_WALL_MM, resolution=32) for q in _PIL]])
    cy = math.floor(UM_CUTOUT[1] + 40.0)
    while True:
        at = lambda g: affinity.translate(g, 0.0, cy)
        if (at(plate).distance(rec) >= FUSED_PLATE_GAP_MM and at(hard).distance(rec) >= RECESS_WEB_MM
                and not at(soft).intersects(rec)):
            return cy
        cy += 0.25


ABOVE_RAISE_MM = float(os.environ.get('LX_DIPOLE_T_ABOVE_RAISE_MM', '0'))   # study knob: extra height over the lowest
ABOVE_CENTER = (0.0, _above_centre() + ABOVE_RAISE_MM)


def above_plan():
    """Plan of the frame fused onto the UM: the driver frame, a saddle down onto the UM ring's profile, overlapping
    the ring outside RING_FUSE_R (the UM itself is the material inside)."""
    cx, cy = ABOVE_CENTER
    land = _rounded_rect(cx, cy, FRAME_W, FRAME_H, CRESCENT_FRAME_R)
    frame_bottom = cy - FRAME_H / 2
    r = CRESCENT_SADDLE_BLEND_R
    plan = unary_union([land, _crescent_saddle(frame_bottom, True, cx)]).buffer(r, resolution=32).buffer(-r, resolution=32)
    plan = unary_union([plan, land]).difference(Point(*UM_CUTOUT[:2]).buffer(RING_FUSE_R, resolution=128))
    f = CRESCENT_MIN_FEATURE_MM
    plan = plan.buffer(-f, resolution=32).buffer(f, resolution=32)
    if plan.geom_type != 'Polygon': plan = max(plan.geoms, key=lambda g: g.area)
    return plan


def um_t_above_cutters():
    cx, cy = ABOVE_CENTER
    return planar_cutters(cx, cy, z_rear=CORE_REAR_Z, z_front=THICKNESS_MM + PLATE_T, part_rear=CORE_REAR_Z)[0]


def um_t_above_tools():
    """Driver cutters and the T duct (sampled centreline, radius) for the one-piece UM + T above."""
    import numpy as np
    cx, cy = ABOVE_CENTER
    z_front = THICKNESS_MM + PLATE_T                      # the plate seats ON the front face
    cutters, _ = planar_cutters(cx, cy, z_rear=CORE_REAR_Z, z_front=z_front, part_rear=CORE_REAR_Z)
    key = f'above/{DRIVER_KEY}/{ORIENTATION}'
    routes = json.loads(UM_T_ROUTES_FILE.read_text()) if UM_T_ROUTES_FILE.exists() else {}
    if key not in routes:
        raise RuntimeError(f'no searched T duct for {key} in {UM_T_ROUTES_FILE.name}')
    rt = routes[key]['t']
    pts = np.asarray(rt['points'], float)
    if abs(rt['d'] - ABOVE_DUCT_D) > 1e-6: raise RuntimeError(f'{key}: duct searched for D{rt["d"]}, not D{ABOVE_DUCT_D}')
    rmin = min_bend_radius(pts[rt.get('bend_from', 0):])
    if rmin < rt['min_bend_r_mm'] - 0.05: raise RuntimeError(f'{key}: bends at R{rmin:.1f} < recorded R{rt["min_bend_r_mm"]}')
    return cutters, (pts, ABOVE_DUCT_D / 2)


def um_t_above_mesh(um_installed, tessellate, solid_only=False):
    """The released UM + the fused tweeter frame, driver cut-outs and the hidden T duct (mesh pipeline)."""
    import manifold3d as m3
    import numpy as np
    import trimesh
    from shapely.geometry.polygon import orient
    from .route import TS_DUCT_D

    def M(t): return m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
    def chain(points, r):
        balls = [m3.Manifold.sphere(r, 32).translate(list(map(float, q))) for q in points]
        return m3.Manifold.batch_boolean([m3.Manifold.batch_hull([a, b]) for a, b in zip(balls, balls[1:])], m3.OpType.Add)
    def box(x0, y0, z0, x1, y1, z1): return m3.Manifold.cube([x1 - x0, y1 - y0, z1 - z0]).translate([x0, y0, z0])
    plan = above_plan()
    plan = orient(plan, sign=1.0)
    loops = [np.asarray(plan.exterior.coords)[:-1]] + [np.asarray(q.coords)[:-1] for q in plan.interiors]
    frame = m3.CrossSection(loops).extrude(THICKNESS_MM - CORE_REAR_Z).translate([0, 0, CORE_REAR_Z])
    um = M(um_installed)
    # Fill the crescent-interface reliefs (inside the UM outline, never into the MU10 flange recess or opening).
    keep = (m3.Manifold.cylinder(40, UM_RING_OUTER_R, UM_RING_OUTER_R, 256).translate([UM_CUTOUT[0], UM_CUTOUT[1], -10])
            - m3.Manifold.cylinder(40, UM_CUTOUT[2] / 2, UM_CUTOUT[2] / 2, 256).translate([UM_CUTOUT[0], UM_CUTOUT[1], -10])
            - m3.Manifold.cylinder(10, UM_RECESS_R, UM_RECESS_R, 256).translate([UM_CUTOUT[0], UM_CUTOUT[1], UM_SEAT_Z]))
    for x0, y0, z0, x1, y1, z1 in _interface_voids(um_installed):
        um = um + (box(x0, y0, max(z0, CORE_REAR_Z), x1, y1, min(z1, THICKNESS_MM)) ^ keep)
    lum, i = _lumen()
    tail = lum[i:]                                        # filled from ABOVE_FILL_FROM_XY; the duct re-opens up to its branch
    # Old lumen tail past the branch: fill it inside the UM's outline, and drop its rear cover behind the core plane.
    um = um + (chain(tail[::2], 3.0 * 1.08) ^ m3.Manifold.cylinder(100, 51.9, 51.9, 256).translate([UM_CUTOUT[0], UM_CUTOUT[1], -40]))
    # Everything behind the core rear plane along the old tail (its cover) goes, in one clean cut; the new collar
    # continues the cover from the branch.
    from shapely.geometry import LineString as _LS
    strip = orient(_LS(tail[:, :2]).buffer(TS_COVER_CLEAR_MM, resolution=32), sign=1.0)
    strip_cs = m3.CrossSection([np.asarray(strip.exterior.coords)[:-1]])
    um = um - strip_cs.extrude(CORE_REAR_Z + 10.0).translate([0, 0, -10.0])
    if solid_only:
        out = (um + frame).to_mesh()
        cutters = um_t_above_cutters()
        part = um + frame
        for c in cutters: part = part - M(tessellate(c))
        out = part.to_mesh()
        return trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True)
    cutters, (pts, r) = um_t_above_tools()
    part = um + frame
    # Collar (the lumen cover's continuation) wherever the duct is still too shallow for the body to wall it.
    # Same section as the released lumen cover (measured: R3.8 about the lumen axis), and carried on until it is buried
    # in the body, so the cover simply continues and has no visible end.
    rc = TS_COVER_OUTER_R
    shallow = np.flatnonzero(pts[:, 2] - rc < CORE_REAR_Z + 0.05)
    low = pts[: shallow.max() + 8] if len(shallow) else pts[:0]
    if len(low) > 1:
        col = chain(low, rc) ^ box(-100, 300, bmr_pod.ENTRY_COLLAR_Z[0], 100, 500, CORE_REAR_Z + 0.5)
        col = col - m3.Manifold.cylinder(40, UM_CUTOUT[2] / 2, UM_CUTOUT[2] / 2, 256).translate([UM_CUTOUT[0], UM_CUTOUT[1], -10])
        # nor into the driver window / lead channel: their cutters start at the core plane, the collar runs behind it
        cx, cy = ABOVE_CENTER
        for g in [_O] + ([_CH] if _CH is not None else []):
            g = orient(affinity.translate(g, cx, cy), sign=1.0)
            col = col - m3.CrossSection([np.asarray(g.exterior.coords)[:-1]]).extrude(60).translate([0, 0, -30])
        part = part + col
    for c in cutters: part = part - M(tessellate(c))
    if ABOVE_DUCT_D < TS_DUCT_D:      # open the duct's start out to the lumen's D6 (t_taper_d)
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        k = np.flatnonzero(arc <= T_TAPER_MM + 1.0)
        dk = t_taper_d(arc[k])
        part = part - stadium_tube(pts[k], dk, dk)
    if part.status() != m3.Error.NoError or part.is_empty(): raise RuntimeError(f'UM + T above boolean failed: {part.status()}')
    out = part.to_mesh()
    mesh = capsule_cut(trimesh.Trimesh(np.asarray(out.vert_properties)[:, :3], np.asarray(out.tri_verts), process=True),
                       [(pts[::2].tolist() + [pts[-1].tolist()], r)])
    bodies = mesh.split(only_watertight=False)
    main = max(bodies, key=lambda b: abs(b.volume))
    stray = [abs(b.volume) for b in bodies if b is not main and b.volume > 0]
    if any(v > 1.0 for v in stray): raise RuntimeError(f'UM + T above: disconnected material {stray}')
    return main


def _one_solid(part, label):
    part = part.clean(); solids = list(part.solids())
    if not part.is_valid or not solids:
        raise RuntimeError(f'{label}: invalid result')
    main = max(solids, key=lambda s: s.volume)
    stray = [s.volume for s in solids if s is not main]
    if any(v > 1.0 for v in stray):
        raise RuntimeError(f'{label}: disconnected material {stray}')
    return Part([main])


def facts():
    land = crescent_land_plan(); wing = bmr_pod._wing_keepout_plan(); um = bmr_pod._um_clearance_disc()
    return dict(
        driver=DRIVER.label + ', ' + ('landscape, terminals/leads to +x' if LANDSCAPE else 'portrait, terminals/leads down'),
        driver_key=DRIVER_KEY, orientation=ORIENTATION,
        release_authorized=RELEASE_AUTHORIZED, physical_measure_required=PHYSICAL_MEASURE_REQUIRED,
        fixing=f'4 x {DRIVER.screw} into D{PILOT_D_MM} pilots on {DRIVER.hole_pitch[0]} x {DRIVER.hole_pitch[1]}',
        crescent=dict(plate_center_xy_mm=list(CRESCENT_CENTER), mu10_to_planar_mm=round(CRESCENT_CENTER[1] - UM_CUTOUT[1], 1),
                      land_wing_overlap_mm2=round(land.intersection(wing).area, 2),
                      land_gap_to_um_clearance_mm=round(land.distance(um), 1), frame_mm=[round(FRAME_W, 2), round(FRAME_H, 2)], surface_mount=True, regular_wings_fit=CRESCENT_FITS_WINGS,
                      t_cable='free, UM mouth to the open rear window (no duct)'),
        um_t=dict(split_y_mm=SPLIT_Y, mu10_shift_mm=MU10_SHIFT, mu10_center_xy_mm=[MU10_CENTER[0], round(MU10_CENTER[1], 3)],
                  plate_center_xy_mm=[0.0, PLATE_CENTER_Y], mu10_to_planar_mm=round(MU10_CENTER[1] - PLATE_CENTER_Y, 1),
                  recess_web_mm=RECESS_WEB_MM, body_half_width_mm=round(BODY_HALF_W, 2), razor_flare_mm=RAZOR_FLARE_MM,
                  razor_edge_mm=RAZOR_EDGE_MM, frame_rear_z_mm=FRAME_REAR_Z, pilot_depth_mm=round(FRAME_PILOT_DEPTH_MM, 2),
                  rear_blends=dict(lm_joint=[BOTTOM_BLEND_Y, LM_JOINT_REAR_Z], ring=[list(TOP_BLEND_Y), RING_REAR_Z]),
                  m2_tie=dict(x_mm=TIE['x'], access='counterbore D4.4 up into the driver window'), um_cable='free behind the part (as released)', t_branch_d_mm=T_BRANCH_D,
                  symmetric=True, captive_magnets=0, regular_wings_fit=False))
