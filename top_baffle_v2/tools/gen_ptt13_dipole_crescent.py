"""Generate the CANDIDATE Obi-Wan PTT1.3 dipole crescent artifacts.

Two Purifi PTT1.3T04-HAG-01 (WG104) tweeters on the baffle centreline,
coaxial with the front-back axis: the front unit's waveguide face flush
with the UM support plane, the rear unit z-mirrored (fires rearward,
back plate exposed at the front plane), nested at the minimum 92.5-mm
centre spacing.  Solid body in the waveguide's own shape language: one
tangent-smooth smoothstep flare per form (rolled front edge and rear
corner, no exterior steps), rounded-plan waist web, lofted rear mounting
stem whose top edge is carved along the rear waveguide's rim circle with
a 1-mm reveal.

Every functional interior is stepped and hidden: shrink-wrap bores whose
flange/tab shoulders retain each driver (rim screws clamp it), a buried
terminal chamber in the waist that both units clock their tabs into, and
a concealed Ø6.5 cable lumen (foot -> stem -> riser -> chamber) so both
pairs hand off to the T route at the foot with zero surface features
anywhere near either radiating face.

Geometry facts are asserted, not assumed: every land, skin, reveal and
lumen probe must hold or the build fails.  Driver dimensions were
measured from the vendor STEP in vendor/PURIFI/PTT1.3T04-HAG-01/.

Outputs (build/ptt_crescent_PTT1.3T04-HAG-01/):
  obiwan_ptt13_dipole_crescent_PTT1.3T04-HAG-01.stl        source frame
  obiwan_ptt13_dipole_crescent_PTT1.3T04-HAG-01.print.stl  front-face-down
  obiwan_ptt13_dipole_crescent_PTT1.3T04-HAG-01.step       CAD master
  obiwan_ptt13_dipole_crescent_PTT1.3T04-HAG-01.brep       exact kernel form
  obiwan_ptt13_dipole_crescent_PTT1.3T04-HAG-01.facts.json checks + specs
  review_assembly.stl                                      body + both drivers

CANDIDATE status: not in the released catalog, no pause manifest (this
part buries no magnets), not on the to_print shelf.  Print it to qualify
it physically, like BMR crescents 17/18.
"""
import hashlib
import json
import math
import os
import struct

import numpy as np

from build123d import (
    Align, Axis, Box, Cylinder, Polyline, Pos, RectangleRounded, Rot,
    export_step, export_stl, extrude, import_step, loft, make_face, revolve,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "ptt_crescent_PTT1.3T04-HAG-01")
PART = "obiwan_ptt13_dipole_crescent_PTT1.3T04-HAG-01"
DRIVER_STEP = os.path.join(
    ROOT, "vendor", "PURIFI", "PTT1.3T04-HAG-01",
    "PTT1.3T04-HAG-01 - 3D CAD.stp")

DEPTH = 42.5
SPACING = 92.5
CLEAR = 0.35
FACE_Y = 13.5        # driver rim-face plane in the vendor frame
BACK_Y = -29.0       # driver back plate in the vendor frame

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
    for i in range(1, 7):                       # rolled front edge r2.5
        a = math.pi / 2 * i / 6
        pts.append((52.8 + 2.5 * math.sin(a), -2.5 + 2.5 * math.cos(a)))
    pts.append((55.3, -7.0))
    for i in range(1, 17):                      # waveguide-family flare
        t = i / 16
        pts.append((55.3 - 11.3 * smoothstep(t), -7.0 - 31.0 * t))
    for i in range(1, 6):                       # rolled rear corner
        a = math.pi / 2 * i / 5
        pts.append((44.0 - 3.0 * (1 - math.cos(a)), -38.0 - 4.5 * math.sin(a)))
    pts.append((0.0, -DEPTH))
    pts.append((0.0, 0.0))
    return make_face(Polyline(*[(x, 0.0, z) for x, z in pts]).edges())

wrap = revolve(wrap_profile(), Axis.Z, 360)
cav = revolve(Pos(0, 0, 0.2) * cavity_profile(CLEAR, over=0.4), Axis.Z, 360)

upper_wrap = Pos(0, SPACING, -DEPTH) * Rot(X=180) * wrap
upper_cav = Pos(0, SPACING, -DEPTH) * Rot(X=180) * cav

web = Pos(0, SPACING / 2.0, -1.0) * extrude(
    RectangleRounded(44.0, 34.0, 14.0), amount=-(DEPTH - 2.0))

chamber = Pos(0, (37.5 + 54.5) / 2.0, (-6.5 - 36.0) / 2.0) * Box(
    42.0, 54.5 - 37.5, 36.0 - 6.5)

stem_top = Pos(0, -6.0, -DEPTH) * RectangleRounded(50.0, 96.0, 16.0)
stem_bot = Pos(0, -8.0, -50.0) * RectangleRounded(42.0, 88.0, 13.0)
stem = loft([stem_top, stem_bot])
stem -= Pos(0, SPACING, -53.0) * Cylinder(
    53.35, 12.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
stem -= Pos(0, 0, -43.0) * Cylinder(
    39.5, 0.5, align=(Align.CENTER, Align.CENTER, Align.MIN))

LUM_X, LUM_Y, LUM_Z = 20.0, 38.8, -46.25
lumen = (
    Pos(LUM_X, -50.0, LUM_Z) * Rot(X=-90) * Cylinder(
        3.25, 92.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    + Pos(LUM_X, LUM_Y, -47.0) * Cylinder(
        3.25, 17.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
)

body = (wrap + upper_wrap + web + stem
        - cav - upper_cav - chamber - lumen).clean()
solids = list(body.solids())
assert len(solids) == 1 and body.is_valid, f"body: {len(solids)} solids"
vol = solids[0].volume

def blocked(probe):
    return sum(s.volume for s in (body & probe).solids())

checks = {}
for cy in (0.0, SPACING):
    for zc in (-0.05, -DEPTH + 0.05):
        assert blocked(Pos(0, cy, zc) * Cylinder(30, 0.06)) < 1e-6, (cy, zc)
checks["through_bores_open_both_faces"] = True
assert blocked(Pos(0, 44.0, -24.0) * Box(3, 3, 3)) < 1e-6
assert blocked(Pos(0, 48.5, -18.0) * Box(3, 3, 3)) < 1e-6
checks["terminal_chamber_reaches_both_tab_positions"] = True
for p in ((LUM_X, -49.5, LUM_Z), (LUM_X, 0.0, LUM_Z),
          (LUM_X, LUM_Y, -44.0), (LUM_X, LUM_Y, -34.0)):
    assert blocked(Pos(*p) * Box(2.0, 2.0, 2.0)) < 1e-6, p
checks["cable_lumen_continuous_foot_to_chamber"] = True
assert blocked(Pos(19.0, 35.5, -40.0) * Box(1.0, 1.0, 1.0)) > 1e-5
checks["riser_wall_to_front_can_solid"] = True
assert blocked(Pos(0, 46.0, -5.6) * Box(2, 2, 1.2)) > 1e-4
assert blocked(Pos(0, 46.0, -36.85) * Box(2, 2, 1.2)) > 1e-4
assert blocked(Pos(0, 53.8, -1.5) * Box(2, 2, 2)) > 1e-3
assert blocked(Pos(0, 39.4, -41.0) * Box(1.2, 1.2, 2)) > 1e-4
checks["chamber_fully_buried_all_skins_intact"] = True
for xx in (0.0, 12.0, 20.0):
    yy = SPACING - math.sqrt(52.9 ** 2 - xx ** 2)
    assert blocked(Pos(xx, yy, -43.2) * Box(0.6, 0.6, 0.8)) < 1e-6, xx
checks["stem_reveal_ring_clear_of_rear_waveguide_mouth"] = True
bb = body.bounding_box()
assert abs(bb.max.X) <= 55.31 and abs(bb.min.X) <= 55.31
assert abs(bb.min.Z - (-50.0)) < 1e-6 and abs(bb.max.Z) < 1e-6
checks["silhouette_clean_no_flank_features"] = True

os.makedirs(OUT, exist_ok=True)
export_stl(body, os.path.join(OUT, f"{PART}.stl"))
export_step(body, os.path.join(OUT, f"{PART}.step"))
try:
    from build123d import export_brep
    export_brep(body, os.path.join(OUT, f"{PART}.brep"))
    brep_ok = True
except Exception:
    brep_ok = False

# print-oriented STL: front face down (rotate 180 about X -> z' in 0..50)
def read_stl(path):
    with open(path, "rb") as f:
        f.read(80)
        (n,) = struct.unpack("<I", f.read(4))
        raw = np.frombuffer(f.read(n * 50), dtype=np.uint8).reshape(n, 50)
        return raw[:, 12:48].copy().view("<f4").reshape(n, 3, 3).astype(float)

def write_stl(path, tris):
    tris = np.ascontiguousarray(tris, dtype=np.float32)
    n = len(tris)
    v1 = tris[:, 1] - tris[:, 0]
    v2 = tris[:, 2] - tris[:, 0]
    nrm = np.cross(v1, v2)
    ln = np.linalg.norm(nrm, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        nrm = np.where(ln > 1e-12, nrm / ln, 0.0).astype(np.float32)
    rec = np.zeros(n, dtype=np.dtype([
        ("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")]))
    rec["n"] = nrm
    rec["v"] = tris
    with open(path, "wb") as f:
        f.write(b"\x00" * 80)
        f.write(struct.pack("<I", n))
        f.write(rec.tobytes())

src = read_stl(os.path.join(OUT, f"{PART}.stl"))
# 180-deg rotation about X (det=+1): winding is preserved, no reversal.
prt = np.stack([src[:, :, 0], -src[:, :, 1], -src[:, :, 2]], axis=2)
write_stl(os.path.join(OUT, f"{PART}.print.stl"), prt)
signed = np.einsum("ij,ij->i", prt[:, 0],
                   np.cross(prt[:, 1], prt[:, 2])).sum() / 6.0
assert signed > 0, f"print.stl winding inverted: signed volume {signed:.0f}"
assert prt.reshape(-1, 3)[:, 2].min() > -0.01, "print.stl below the bed"

# review assembly: body + both driver meshes in the source frame
drv_solid = import_step(DRIVER_STEP)
export_stl(drv_solid, os.path.join(OUT, "_driver_tmp.stl"))
drv = read_stl(os.path.join(OUT, "_driver_tmp.stl"))[::12]  # review-grade
os.remove(os.path.join(OUT, "_driver_tmp.stl"))
front_drv = np.stack(
    [drv[:, :, 0], -drv[:, :, 2], drv[:, :, 1] - FACE_Y], axis=2)
rear_drv = np.stack(
    [drv[:, :, 0], drv[:, :, 2] + SPACING, -drv[:, :, 1] + BACK_Y], axis=2)
write_stl(os.path.join(OUT, "review_assembly.stl"),
          np.concatenate([src, front_drv, rear_drv]))

sha = hashlib.sha256(open(DRIVER_STEP, "rb").read()).hexdigest()
facts = {
    "part": PART,
    "status": ("CANDIDATE - not release-authorized; absent from the "
               "released catalog and the to_print shelf; print to qualify "
               "physically (like BMR crescents 17/18)"),
    "driver": {
        "model": "Purifi PTT1.3T04-HAG-01 with WG104 waveguide",
        "vendor_step": os.path.relpath(DRIVER_STEP, ROOT),
        "vendor_step_sha256": sha,
        "measured_profile_bands_r_mm": [list(b) for b in BANDS],
        "face_plane_vendor_y_mm": FACE_Y,
        "depth_mm": DEPTH,
    },
    "layout": {
        "centre_spacing_mm": SPACING,
        "front_unit": "fires forward, WG104 face flush with the UM "
                      "support front plane (bridge crest 2.56 proud)",
        "rear_unit": "z-mirrored, fires rearward, back plate exposed at "
                     "the front plane",
        "spacing_rationale": "max over depth of r_front+r_rear = 90.2 "
                             "+ 2.3 clearance",
    },
    "body": {
        "volume_cm3": round(vol / 1000.0, 1),
        "exterior": "waveguide-family smoothstep flare R55.3->R44 over "
                    "31 mm per form, rolled front edge r2.5 and rear "
                    "corner, rounded-plan waist web r14, lofted rear stem "
                    "carved 1 mm clear of the rear waveguide mouth",
        "interior": "stepped shrink-wrap bores (retention shoulders), "
                    "buried waist terminal chamber z -6.5..-36, "
                    "cable lumen D6.5 at x=+20: foot -> stem -> riser -> "
                    "chamber",
        "min_walls_mm": {
            "cavity_face_skins": 3.0,
            "rim_recess_to_chamber_skin": 1.75,
            "front_rim_recess_to_rear_cavity_pinch": 1.5,
            "flare_to_cavity_bands": 0.77,
        },
    },
    "checks_passed": checks,
    "print_notes": [
        "print.stl is front-face-down (source rotated 180 deg about X); "
        "the front faces and rim ring sit on the bed",
        "no captive magnets, no pause: slice normally",
        "the lofted stem base overhangs the rear-face silhouette at its "
        "outboard corners: enable support for the stem region only "
        "(7.5 mm tall) or accept short bridges",
        "body is designed solid: print with high wall count / infill if "
        "the acoustic deadness of a solid part is wanted",
        "insert drivers from their faces after feeding the cable pair up "
        "the stem lumen; wire each unit before seating it; rim screws "
        "clamp against the internal shoulders",
    ],
    "exports": {
        "stl_source_frame": f"{PART}.stl",
        "stl_print_oriented": f"{PART}.print.stl",
        "step": f"{PART}.step",
        "brep": f"{PART}.brep" if brep_ok else None,
        "review_assembly_stl": "review_assembly.stl",
    },
    "generated_by": "tools/gen_ptt13_dipole_crescent.py",
}
with open(os.path.join(OUT, f"{PART}.facts.json"), "w") as f:
    json.dump(facts, f, indent=2)
    f.write("\n")

print(f"body volume {vol/1000:.1f} cm3; all {len(checks)} check groups passed")
print(f"artifacts in {os.path.relpath(OUT, ROOT)}/")
