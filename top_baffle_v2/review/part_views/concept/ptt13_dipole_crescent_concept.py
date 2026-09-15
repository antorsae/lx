"""CONCEPT v4 - PTT1.3T04-HAG-01 (WG104) dipole crescent, option (b).

Layout as v3 (flush front unit, z-mirrored rear unit at 92.5-mm centres,
solid body, buried waist terminal chamber, cable lumen inside the rear
mounting stem -- zero surface features near either radiating face), with
every exterior transition reworked into the waveguide's own shape
language:

  * The WG104 dish measures as a soft S-flare (trough at r23 easing up
    ~4.8 mm to the rim).  The body's outer wrap uses the same family:
    a rolled front edge (r2.5), a short cylindrical land, then ONE
    tangent-smooth smoothstep flare from R55.3 down to R44 over 31 mm,
    finishing in a rolled rear corner -- no stepped bands anywhere
    outside (the functional stepped bores remain inside, hidden).
  * The waist web is rounded-plan (r14).
  * The stem is a loft (wide rounded section at the body easing to a
    narrower one at the foot) and its top edge is carved along the rear
    waveguide's rim circle with a 1-mm reveal, so it never touches the
    mouth; a 0.5 relief keeps it off the front unit's exposed back plate.

Internal cavity bands (from the vendor STEP, face plane y=13.5, depth
42.5): rim 52.0 (0..4.5) / skirt 46.0 (..12.5) / flange 45.1 (..22.5) /
tab band 44.0 (..30) / back can 38.3 (..42.5); terminal block ~+-20 deg,
r to 44.9, 19.5..29.5 behind the face.  The flare stays >=0.77 clear of
every cavity band.
"""
import math
import os
from build123d import (
    Align, Axis, Box, Cylinder, Polyline, Pos, RectangleRounded, Rot,
    export_stl, extrude, loft, make_face, revolve,
)

DEPTH = 42.5
SPACING = 92.5
CLEAR = 0.35

BANDS = (
    (0.0, 4.5, 52.0),
    (4.5, 12.5, 46.0),
    (12.5, 22.5, 45.1),
    (22.5, 30.0, 44.0),
    (30.0, DEPTH, 38.3),
)

def cavity_profile(extra, over=0.0):
    pts = [(0.0, 0.0)]
    for z0, z1, r in BANDS:
        zz1 = z1 + (over if z1 == DEPTH else 0.0)
        pts.append((r + extra, -z0))
        pts.append((r + extra, -zz1))
    pts.append((0.0, -(DEPTH + over)))
    pts.append((0.0, 0.0))
    return make_face(Polyline(*[(x, 0.0, z) for x, z in pts]).edges())

def smoothstep(t):
    return 3 * t * t - 2 * t * t * t

def wrap_profile():
    pts = [(0.0, 0.0), (52.8, 0.0)]
    # rolled front edge r2.5
    for i in range(1, 7):
        a = math.pi / 2 * i / 6
        pts.append((52.8 + 2.5 * math.sin(a), -2.5 + 2.5 * math.cos(a)))
    pts.append((55.3, -7.0))
    # waveguide-family smoothstep flare 55.3 -> 44.0 over z -7..-38
    for i in range(1, 17):
        t = i / 16
        pts.append((55.3 - 11.3 * smoothstep(t), -7.0 - 31.0 * t))
    # rolled rear corner into the back face
    for i in range(1, 6):
        a = math.pi / 2 * i / 5
        pts.append((44.0 - 3.0 * (1 - math.cos(a)), -38.0 - 4.5 * math.sin(a)))
    pts.append((0.0, -DEPTH))
    pts.append((0.0, 0.0))
    return make_face(Polyline(*[(x, 0.0, z) for x, z in pts]).edges())

wrap = revolve(wrap_profile(), Axis.Z, 360)
cav = revolve(Pos(0, 0, 0.2) * cavity_profile(CLEAR, over=0.4), Axis.Z, 360)

lower_wrap = wrap
lower_cav = cav
upper_wrap = Pos(0, SPACING, -DEPTH) * Rot(X=180) * wrap
upper_cav = Pos(0, SPACING, -DEPTH) * Rot(X=180) * cav

web = Pos(0, SPACING / 2.0, -1.0) * extrude(
    RectangleRounded(44.0, 34.0, 14.0), amount=-(DEPTH - 2.0))

# Buried terminal chamber (z -6.5..-36 keeps 1.75 skin under each rim
# recess floor); widened to x +-21 so the riser mouth opens into it.
chamber = Pos(0, (37.5 + 54.5) / 2.0, (-6.5 - 36.0) / 2.0) * Box(
    42.0, 54.5 - 37.5, 36.0 - 6.5)

# Lofted rear mounting stem, top edge carved along the rear waveguide's
# rim circle (R52.35 + 1 reveal), 0.5 relief over the front back plate.
stem_top = Pos(0, -6.0, -DEPTH) * RectangleRounded(50.0, 96.0, 16.0)
stem_bot = Pos(0, -8.0, -50.0) * RectangleRounded(42.0, 88.0, 13.0)
stem = loft([stem_top, stem_bot])
stem -= Pos(0, SPACING, -53.0) * Cylinder(
    53.35, 12.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
stem -= Pos(0, 0, -43.0) * Cylinder(
    39.5, 0.5, align=(Align.CENTER, Align.CENTER, Align.MIN))

# Concealed cable lumen in the x=+20 plane: the corridor between the
# front cavity (needs 37.5, has 42.5 at y 38.8) and the rear rim recess
# (kept 0.8 clear).  Horizontal run in the stem, elbow, riser to chamber.
LUM_X, LUM_Y, LUM_Z = 20.0, 38.8, -46.25
lumen = (
    Pos(LUM_X, -50.0, LUM_Z) * Rot(X=-90) * Cylinder(
        3.25, 92.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    + Pos(LUM_X, LUM_Y, -47.0) * Cylinder(
        3.25, 17.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
)

body = (lower_wrap + upper_wrap + web + stem
        - lower_cav - upper_cav - chamber - lumen).clean()
solids = list(body.solids())
assert len(solids) == 1 and body.is_valid, f"body: {len(solids)} solids"
vol = solids[0].volume
assert vol > 50000, vol

def blocked(probe):
    return sum(s.volume for s in (body & probe).solids())

# through-bores open at both faces
for cy in (0.0, SPACING):
    for zc in (-0.05, -DEPTH + 0.05):
        assert blocked(Pos(0, cy, zc) * Cylinder(30, 0.06)) < 1e-6, (cy, zc)
# chamber void reaches both tab positions
assert blocked(Pos(0, 44.0, -24.0) * Box(3, 3, 3)) < 1e-6, "front tabs"
assert blocked(Pos(0, 48.5, -18.0) * Box(3, 3, 3)) < 1e-6, "rear tabs"
# lumen continuous: foot mouth, mid-stem, riser, chamber entry
for p in ((LUM_X, -49.5, LUM_Z), (LUM_X, 0.0, LUM_Z),
          (LUM_X, LUM_Y, -44.0), (LUM_X, LUM_Y, -34.0)):
    assert blocked(Pos(*p) * Box(2.0, 2.0, 2.0)) < 1e-6, p
# riser corridor wall toward the front driver's can stays solid
assert blocked(Pos(19.0, 35.5, -40.0) * Box(1.0, 1.0, 1.0)) > 1e-5, (
    "riser-to-can wall")
# skins intact: under each rim recess, and the waist faces
assert blocked(Pos(0, 46.0, -5.6) * Box(2, 2, 1.2)) > 1e-4, "front recess skin"
assert blocked(Pos(0, 46.0, -36.85) * Box(2, 2, 1.2)) > 1e-4, "rear recess skin"
assert blocked(Pos(0, 53.8, -1.5) * Box(2, 2, 2)) > 1e-3, "front web face"
assert blocked(Pos(0, 39.4, -41.0) * Box(1.2, 1.2, 2)) > 1e-4, "rear web face"
# stem never touches the rear waveguide mouth (1-mm reveal ring is void)
for xx in (0.0, 12.0, 20.0):
    yy = SPACING - math.sqrt(52.9 ** 2 - xx ** 2)
    assert blocked(Pos(xx, yy, -43.2) * Box(0.6, 0.6, 0.8)) < 1e-6, (
        f"stem reveal at x={xx}")
# flanks clean
bb = body.bounding_box()
assert abs(bb.max.X) <= 55.31 and abs(bb.min.X) <= 55.31, (bb.min.X, bb.max.X)
assert abs(bb.min.Z - (-50.0)) < 1e-6 and abs(bb.max.Z) < 1e-6, (
    bb.min.Z, bb.max.Z)

HERE = os.path.dirname(os.path.abspath(__file__))
export_stl(body, os.path.join(HERE, "ptt13_dipole_crescent_concept.stl"))
print(f"body volume {vol/1000:.0f} cm3, z {bb.min.Z:.1f}..{bb.max.Z:.1f}, "
      f"y {bb.min.Y:.1f}..{bb.max.Y:.1f}, x {bb.min.X:.1f}..{bb.max.X:.1f}")
print("exported ptt13_dipole_crescent_concept.stl")
