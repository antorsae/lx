"""Generate the CANDIDATE Obi-Wan PTT1.3 dipole crescent artifacts.

Two Purifi PTT1.3T04-HAG-01 (WG104) tweeters on the baffle centreline,
coaxial with the front-back axis: the front unit's waveguide face flush
with the UM support plane, the rear unit z-mirrored (fires rearward,
back plate exposed at the front plane), nested at the minimum 92.5-mm
centre spacing.  Solid body in the waveguide's own shape language: one
tangent-smooth smoothstep flare per form (rolled front edge and rear
corner, no exterior steps) and a rounded-plan waist web.

Fastening and cabling (all datasheet/STEP-verified):

  * Each driver bolts with SIX M3 screws on the vendor's D98 pattern
    (D3.5 rim holes verified on the STEP at 30 deg + k*60 from the
    terminal block): the body carries 6 heat-set insert bores D4.6 x 6
    behind each rim recess floor, clocked so the terminal block faces
    the buried waist chamber.
  * Both units clock their tabs into that chamber; connections are fully
    internal (1.75-mm skins under each rim recess).
  * A concealed D6.5 cable lumen runs from the chamber down a riser at
    x=+20 (the corridor clearing both drivers), S-bends on the hidden
    rear side to the centreline, and exits through the FOOT face into a
    D8 x 2.5 socket -- the seamless T-duct handoff to the UM support.
    The foot pad also carries two blind M3 heat-set receivers (rear-
    opening, the same idiom as the released crescent joints) so the UM
    support clamps the crescent with rear-driven screws.
  * The old slab stem is gone: the only rear-side material is a slim
    17-mm spine following the lumen path plus the compact foot pad, both
    carved 1 mm clear of the rear waveguide mouth and relieved 0.5 off
    the front unit's exposed back plate.

Geometry facts are asserted, not assumed: every land, skin, reveal,
insert bore and lumen probe must hold or the build fails.  Driver
dimensions were measured from the vendor STEP; the mounting pattern is
cross-checked against the data sheet (rev 1.00) in
vendor/PURIFI/PTT1.3T04-HAG-01/.

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
    Align, Axis, Box, Circle, Cylinder, Plane, Polyline, Pos,
    RectangleRounded, Rot, export_step, export_stl, extrude, import_step,
    make_face, revolve,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "build", "ptt_crescent_PTT1.3T04-HAG-01")
PART = "obiwan_ptt13_dipole_crescent_PTT1.3T04-HAG-01"
VENDOR_DIR = os.path.join(ROOT, "vendor", "PURIFI", "PTT1.3T04-HAG-01")
DRIVER_STEP = os.path.join(VENDOR_DIR, "PTT1.3T04-HAG-01 - 3D CAD.stp")

DEPTH = 42.5
SPACING = 92.5
CLEAR = 0.35
FACE_Y = 13.5          # driver rim-face plane in the vendor frame
BACK_Y = -29.0         # driver back plate in the vendor frame

BANDS = (
    (0.0, 4.5, 52.0),
    (4.5, 12.5, 46.0),
    (12.5, 22.5, 45.1),
    (22.5, 30.0, 44.0),
    (30.0, DEPTH, 38.3),
)

# vendor mounting pattern (datasheet Table 4, verified on the STEP)
MOUNT_BC_R = 49.0
MOUNT_AZ_FROM_TERMINAL_DEG = 30.0
INSERT_BORE_D = 4.6    # repo-standard M3 heat-set receiver
INSERT_BORE_DEPTH = 6.0
TERMINAL_AZ_DEG = 90.0  # both units clock their tabs toward the waist

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

# Buried terminal chamber (z -6.5..-36 keeps 1.75 skin under each rim
# recess floor); x +-21 so the riser mouth opens into it.
chamber = Pos(0, (37.5 + 54.5) / 2.0, (-6.5 - 36.0) / 2.0) * Box(
    42.0, 54.5 - 37.5, 36.0 - 6.5)

# ---- concealed cable lumen -------------------------------------------------
# Riser at x=+20 (clears the front cavity by 2.5 and the rear rim recess
# by 0.8), then at z -46.25 an S-bend on the hidden rear side brings the
# run to the centreline and out through the foot face.
LUM_X, LUM_Y, LUM_Z = 20.0, 38.9, -46.25
S_R = 38.8
S_TH = 42.1  # degrees; lateral shift 20 over ~52 of run
ARC1_C = (LUM_X - S_R, 12.0)          # az 0 -> -S_TH
ARC2_C = (S_R, -40.0)                 # az 180-S_TH -> 180

def lumen_path_points():
    """The S-path at z=LUM_Z, chamber riser base to past the foot face."""
    pts = [(LUM_X, LUM_Y)]
    for t in np.linspace(0.0, 1.0, 4)[1:]:
        pts.append((LUM_X, LUM_Y - (LUM_Y - 12.0) * t))
    for t in np.linspace(0.0, 1.0, 9)[1:]:
        a = math.radians(-S_TH * t)
        pts.append((ARC1_C[0] + S_R * math.cos(a),
                    ARC1_C[1] + S_R * math.sin(a)))
    for t in np.linspace(0.0, 1.0, 9)[1:]:
        a = math.radians(180.0 - S_TH + S_TH * t)
        pts.append((ARC2_C[0] + S_R * math.cos(a),
                    ARC2_C[1] + S_R * math.sin(a)))
    return pts

def _segment_cyl(p, q, r, over=0.8):
    d = (q[0] - p[0], q[1] - p[1], q[2] - p[2])
    ln = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    u = (d[0] / ln, d[1] / ln, d[2] / ln)
    start = (p[0] - u[0] * over, p[1] - u[1] * over, p[2] - u[2] * over)
    return Plane(origin=start, z_dir=u) * Cylinder(
        r, ln + 2 * over, align=(Align.CENTER, Align.CENTER, Align.MIN))

# The lumen is cut as a chain of short overlapping cylinders: exact
# torus segments defeat the kernel at their tangent junctions, and a
# 3.5-deg faceted bore (sagitta 0.02) is identical for a cable.
lumen_pieces = [
    Pos(LUM_X, LUM_Y, -47.0) * Cylinder(
        3.25, 17.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
]
_path = [(px, py, LUM_Z) for px, py in lumen_path_points()]
for _p, _q in zip(_path[:-1], _path[1:]):
    lumen_pieces.append(_segment_cyl(_p, _q, 3.25))
# Tail + T-duct handoff socket as ONE stepped coaxial cutter (the D6.5
# tail run to the foot face with the D8 nozzle counterbore): pre-unioned
# so the kernel never cuts coaxially through an existing void.
tail_socket = (
    Pos(0, -55.0, LUM_Z) * Rot(X=-90) * Cylinder(
        3.25, 15.8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    + Pos(0, -55.0, LUM_Z) * Rot(X=-90) * Cylinder(
        4.0, 5.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
)

# ---- minimal rear spine + ducted foot --------------------------------------
def chain_plan():
    faces = []
    def add(px, py):
        faces.append(Pos(px, py) * Circle(8.5))
    for t in np.linspace(0.0, 1.0, 14):
        add(LUM_X, LUM_Y - (LUM_Y - 12.0) * t)
    for t in np.linspace(0.0, 1.0, 12):
        a = math.radians(-S_TH * t)
        add(ARC1_C[0] + S_R * math.cos(a), ARC1_C[1] + S_R * math.sin(a))
    for t in np.linspace(0.0, 1.0, 12):
        a = math.radians(180.0 - S_TH + S_TH * t)
        add(ARC2_C[0] + S_R * math.cos(a), ARC2_C[1] + S_R * math.sin(a))
    for t in np.linspace(0.0, 1.0, 6):
        add(0.0, -40.0 - 8.0 * t)
    plan = faces[0]
    for f in faces[1:]:
        plan += f
    pad = Pos(0, -44.0) * RectangleRounded(42.0, 16.0, 7.0)
    pad &= Circle(53.0)               # keep the corners inside the flare
    return plan + pad

spine = Pos(0, 0, -15.0) * extrude(chain_plan(), amount=-35.0)
# 1-mm reveal to the rear waveguide mouth, behind the rear face only
spine -= Pos(0, SPACING, -55.0) * Cylinder(
    53.35, 12.6, align=(Align.CENTER, Align.CENTER, Align.MIN))
# 0.5 relief off the front unit's exposed back plate
spine -= Pos(0, 0, -43.0) * Cylinder(
    39.5, 0.5, align=(Align.CENTER, Align.CENTER, Align.MIN))

# blind M3 receivers in the foot (rear-opening, crescent-joint idiom)
foot_inserts = [
    Pos(sx * 14.0, -44.0, -50.0 - 0.1) * Cylinder(
        INSERT_BORE_D / 2.0, INSERT_BORE_DEPTH + 0.1,
        align=(Align.CENTER, Align.CENTER, Align.MIN))
    for sx in (-1.0, 1.0)
]

# ---- driver mounting inserts: 6 x M3 on the vendor D98 pattern -------------
def driver_insert_cutters():
    cutters = []
    for k in range(6):
        az = math.radians(
            TERMINAL_AZ_DEG + MOUNT_AZ_FROM_TERMINAL_DEG + 60.0 * k)
        px, py = MOUNT_BC_R * math.cos(az), MOUNT_BC_R * math.sin(az)
        c = Pos(px, py, -4.55 - INSERT_BORE_DEPTH - 0.2) * Cylinder(
            INSERT_BORE_D / 2.0, INSERT_BORE_DEPTH + 0.4,
            align=(Align.CENTER, Align.CENTER, Align.MIN))
        cutters.append(c)
    return cutters

front_bores = driver_insert_cutters()
rear_bores = [Pos(0, SPACING, -DEPTH) * Rot(X=180) * c for c in front_bores]

body = (wrap + upper_wrap + web + spine
        - cav - upper_cav - chamber)
for c in (*lumen_pieces, tail_socket, *front_bores, *rear_bores,
          *foot_inserts):
    body -= c
body = body.clean()
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
_pp = lumen_path_points()
lum_probes = (
    (LUM_X, LUM_Y, -34.0),                    # riser
    (_pp[2][0], _pp[2][1], LUM_Z),            # straight run
    (_pp[7][0], _pp[7][1], LUM_Z),            # arc 1 mid
    (_pp[15][0], _pp[15][1], LUM_Z),          # arc 2 mid
    (0.0, -48.0, LUM_Z),                      # tail
    (0.0, -52.4, LUM_Z),                      # through the foot face / socket
)
for p in lum_probes:
    assert blocked(Pos(*p) * Box(1.6, 1.6, 1.6)) < 1e-6, p
checks["cable_lumen_continuous_chamber_to_foot_socket"] = True
assert blocked(Pos(19.0, 35.5, -40.0) * Box(1.0, 1.0, 1.0)) > 1e-5
checks["riser_wall_to_front_can_solid"] = True
for az_deg in (120.0, 240.0):
    az = math.radians(az_deg)
    px, py = MOUNT_BC_R * math.cos(az), MOUNT_BC_R * math.sin(az)
    assert blocked(Pos(px, py, -8.0) * Box(1.5, 1.5, 1.5)) < 1e-6, az_deg
    assert blocked(
        Pos(px, SPACING - py, -DEPTH + 8.0) * Box(1.5, 1.5, 1.5)) < 1e-6
checks["driver_insert_bores_open_D98_pattern_both_units"] = True
assert blocked(Pos(21.6, 42.4, -8.0) * Box(0.5, 0.5, 2.0)) > 1e-6
checks["insert_bore_to_chamber_wall_solid"] = True
for sx in (-1.0, 1.0):
    assert blocked(Pos(sx * 14.0, -44.0, -46.5) * Box(1.5, 1.5, 1.5)) < 1e-6
checks["foot_M3_receivers_open"] = True
assert blocked(Pos(0, 46.0, -5.6) * Box(2, 2, 1.2)) > 1e-4
assert blocked(Pos(0, 46.0, -36.85) * Box(2, 2, 1.2)) > 1e-4
assert blocked(Pos(0, 53.8, -1.5) * Box(2, 2, 2)) > 1e-3
assert blocked(Pos(0, 39.4, -41.0) * Box(1.2, 1.2, 2)) > 1e-4
checks["chamber_fully_buried_all_skins_intact"] = True
for xx in (0.0, 12.0, 20.0):
    yy = SPACING - math.sqrt(52.9 ** 2 - xx ** 2)
    assert blocked(Pos(xx, yy, -43.2) * Box(0.6, 0.6, 0.8)) < 1e-6, xx
checks["spine_reveal_ring_clear_of_rear_waveguide_mouth"] = True
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

def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

facts = {
    "part": PART,
    "status": ("CANDIDATE - not release-authorized; absent from the "
               "released catalog and the to_print shelf; print to qualify "
               "physically (like BMR crescents 17/18)"),
    "driver": {
        "model": "Purifi PTT1.3T04-HAG-01 with WG104 waveguide",
        "vendor_step": os.path.relpath(DRIVER_STEP, ROOT),
        "vendor_step_sha256": sha256(DRIVER_STEP),
        "vendor_datasheet_sha256": sha256(os.path.join(
            VENDOR_DIR, "PTT1.3T04-HAG-01 - Data Sheet.pdf")),
        "measured_profile_bands_r_mm": [list(b) for b in BANDS],
        "face_plane_vendor_y_mm": FACE_Y,
        "depth_mm": DEPTH,
        "gasket_note": "vendor's 1.5 gasket omitted: the rim seats "
                       "directly on the printed recess floor so the "
                       "front face stays flush",
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
    "mounting": {
        "per_driver": "6 x M3 heat-set inserts, bores D4.6 x 6.0, on the "
                      "vendor D98 pattern (D3.5 rim holes), clocked "
                      "terminal+30+k*60 with the terminal at the waist",
        "crescent_to_um_support": "2 x blind M3 heat-set receivers in the "
                                  "foot pad at (+-14, -44), rear-opening "
                                  "(rear-driven screws, crescent-joint "
                                  "idiom)",
    },
    "cabling": {
        "lumen_d_mm": 6.5,
        "path": "waist chamber -> riser at x=+20 (corridor clearing both "
                "drivers) -> z -46.25 S-bend (R38.8, tangent-continuous) "
                "to the centreline on the hidden rear side -> foot face",
        "handoff": "D8 x 2.5 socket in the foot face at (0, -52, -46.25) "
                   "seats the UM support's T-duct nozzle: continuous "
                   "concealed duct, zero exposed cable",
    },
    "body": {
        "volume_cm3": round(vol / 1000.0, 1),
        "exterior": "waveguide-family smoothstep flare R55.3->R44 over "
                    "31 mm per form, rolled front edge r2.5 and rear "
                    "corner, rounded-plan waist web r14; rear side "
                    "carries only a 17-mm lumen spine and the compact "
                    "foot pad (carved 1 mm clear of the rear waveguide "
                    "mouth, relieved 0.5 off the front back plate)",
        "interior": "stepped shrink-wrap bores (retention shoulders "
                    "back up the 6 rim screws), buried waist terminal "
                    "chamber z -6.5..-36",
        "min_walls_mm": {
            "cavity_face_skins": 3.0,
            "rim_recess_to_chamber_skin": 1.75,
            "front_rim_recess_to_rear_cavity_pinch": 1.5,
            "flare_to_cavity_bands": 0.77,
            "lumen_wall_at_reveal_carve": 0.6,
            "insert_bore_to_chamber": 1.2,
        },
    },
    "checks_passed": checks,
    "print_notes": [
        "print.stl is front-face-down (source rotated 180 deg about X); "
        "the front faces and rim ring sit on the bed",
        "no captive magnets, no pause: slice normally",
        "the foot pad and spine sit at print z 42.5-50 partly over the "
        "flare: enable support for that small rear region only",
        "all 14 insert bores are blind and print support-free",
        "body is designed solid: print with high wall count / infill if "
        "the acoustic deadness of a solid part is wanted",
        "assembly: feed the cable pair up the foot socket and lumen into "
        "the chamber; wire each unit, seat it, and fit 6 x M3 into the "
        "heat-set inserts behind the rim; the UM support clamps the foot "
        "with 2 rear-driven M3 screws while its duct nozzle seats in the "
        "D8 socket",
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
