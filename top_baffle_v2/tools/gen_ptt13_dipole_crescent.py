"""Generate the CANDIDATE Obi-Wan PTT1.3 dipole crescent artifacts.

Two Purifi PTT1.3T04-HAG-01 (WG104) tweeters on the baffle centreline,
coaxial with the front-back axis: the front unit's waveguide face flush
with the UM support plane, the rear unit z-mirrored (fires rearward,
back plate exposed at the front plane), nested at the minimum 92.5-mm
centre spacing.  Solid body in the waveguide's own shape language: one
tangent-smooth smoothstep flare per form (rolled front edge and rear
corner, no exterior steps) and a rounded-plan waist web.

OBI-WAN ONLY: this crescent pairs with the Obi-Wan UM carrier's
released tweeter-joint contract and replaces that profile's crescent
slot options; it does not apply to stock or slim.

Fastening and cabling (all datasheet/STEP-verified):

  * Each driver bolts with SIX M3 screws on the vendor's D98 pattern
    (D3.5 rim holes verified on the STEP at 30 deg + k*60 from the
    terminal block): the body carries 6 heat-set insert bores D4.6 x 6
    behind each rim recess floor, clocked so the terminal block faces
    the buried waist chamber.  Both units clock their tabs into that
    chamber; connections are fully internal.
  * TWIN redundant D4.2 cable ducts, fully buried inside the body,
    follow the front unit's circle at r=50 at the baffle's own duct
    depth: each starts at a waist-chamber wall (x=+-21), sweeps its own
    side, and converges into the stepped D8 T-duct socket in the lower
    edge -- the seamless handoff to the UM support's duct nozzle.
  * The crescent attaches exactly per the released Obi-Wan tweeter
    joint: bosses on the +-24 spacing (TWEETER_JOINT_X) in the addon's
    front z-band, rear-opening M3 heat-set receivers D4.6 x 4.2 at the
    released bore band, ear notches clearing the UM core's functional
    ears; the core's rear-driven M3 screws clamp it.  The FIRST tweeter
    (front-firing) is the one nearest this joint.
  * NOTHING remains behind the rear face: both waveguide mouths and
    back plates sit in a completely clean surface.

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
    RectangleRounded, RegularPolygon, Rot, export_step, export_stl, extrude,
    import_step, make_face, revolve,
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
INSERT_BORE_D = 4.6    # the released M3 heat-set receiver bore, exactly
INSERT_BORE_DEPTH = 4.2  # as the tweeter-joint receivers: 4.0 insert + 0.2
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

# The solids are axisymmetric, so a 30-deg clock moves the revolve seam
# meridian exactly between the insert-bore azimuths: bores cut through a
# seam edge are silently dropped by the kernel.
wrap = Rot(Z=30) * revolve(wrap_profile(), Axis.Z, 360)
cav = Rot(Z=30) * revolve(
    Pos(0, 0, 0.2) * cavity_profile(CLEAR, over=0.4), Axis.Z, 360)

upper_wrap = Pos(0, SPACING, -DEPTH) * Rot(X=180) * wrap
upper_cav = Pos(0, SPACING, -DEPTH) * Rot(X=180) * cav

web = Pos(0, SPACING / 2.0, -1.0) * extrude(
    RectangleRounded(44.0, 34.0, 14.0), amount=-(DEPTH - 2.0))

# Buried terminal chamber (z -6.5..-36 keeps 1.75 skin under each rim
# recess floor); x +-21 so the riser mouth opens into it.
chamber = Pos(0, (37.5 + 54.5) / 2.0, (-6.5 - 36.0) / 2.0) * Box(
    42.0, 54.5 - 37.5, 36.0 - 6.5)

# ---- twin concealed cable ducts on the back-circle outline -----------------
# Two redundant D4.2 ducts, fully buried inside the body, following the
# front unit's circle at r=50 (between the driver cavity, the M3 insert
# bores above, and the flare skin outside), at the baffle's own duct
# depth (z centre -13.6 = baffle 4.7).  Each starts at a waist-chamber
# wall (x=+-21), sweeps its own side of the circle, and converges at the
# bottom into the stepped T-duct socket in the crescent's lower edge --
# NOTHING remains behind the rear face.
DUCT_R = 50.0
DUCT_Z = -13.6
DUCT_D = 4.2
DUCT_AZ_TOP = math.degrees(math.acos(21.0 / DUCT_R))   # chamber wall

def _segment_cyl(p, q, r, over=0.8):
    d = (q[0] - p[0], q[1] - p[1], q[2] - p[2])
    ln = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2)
    u = (d[0] / ln, d[1] / ln, d[2] / ln)
    start = (p[0] - u[0] * over, p[1] - u[1] * over, p[2] - u[2] * over)
    return Plane(origin=start, z_dir=u) * Cylinder(
        r, ln + 2 * over, align=(Align.CENTER, Align.CENTER, Align.MIN))

def duct_arc_points(side):
    pts = []
    for i in range(21):
        a = math.radians(DUCT_AZ_TOP + (-90.0 - DUCT_AZ_TOP) * i / 20)
        pts.append((side * DUCT_R * math.cos(a), DUCT_R * math.sin(a),
                    DUCT_Z))
    return pts

duct_pieces = []
for side in (1.0, -1.0):
    pp = duct_arc_points(side)
    for p, q in zip(pp[:-1], pp[1:]):
        duct_pieces.append(_segment_cyl(p, q, DUCT_D / 2.0))

# Stepped T-duct handoff socket in the crescent's lower edge: D8 nozzle
# seat, then a D4.5 wire neck meeting the two arc ends at (0, -50).
tail_socket = (
    Pos(0, -56.0, DUCT_Z) * Rot(X=-90) * Cylinder(
        4.0, 6.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    + Pos(0, -56.0, DUCT_Z) * Rot(X=-90) * Cylinder(
        2.25, 9.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
)

# ---- UM tweeter-joint interface (released contract, local frame) -----------
# Baffle -> local: z_local = z_baffle - 18.3.  The addon owns the front
# band (TWEETER_ADDON_JOINT_Z 12.4..18.3 -> local -5.9..0) and carries
# rear-opening M3 insert receivers at TWEETER_JOINT_INSERT_BORE_Z
# 12.2..16.4 -> local -6.1..-1.9, on the released +-24 spacing
# (TWEETER_JOINT_X), boss D9.8; the UM core's rear-driven M3 screws pass
# through its D3.4 holes into these inserts.  The first tweeter (the
# front-firing unit) is the one nearest this joint.
JOINT_X = 24.0
JOINT_Y = -50.0
joint_bosses = []
joint_bores = []
ear_notches = []
for sx in (-1.0, 1.0):
    joint_bosses.append(
        Pos(sx * JOINT_X, JOINT_Y, -5.9) * Cylinder(
            4.9, 5.9, align=(Align.CENTER, Align.CENTER, Align.MIN))
        + Pos(sx * JOINT_X, JOINT_Y + 3.0, -5.9) * extrude(
            RectangleRounded(9.8, 8.0, 2.4), amount=5.9))
    joint_bores.append(
        Pos(sx * JOINT_X, JOINT_Y, -6.1) * Cylinder(
            2.3, 4.2, align=(Align.CENTER, Align.CENTER, Align.MIN)))
    # notch the core band so the UM core's functional ear (D9.8 + clear)
    # laps under the boss
    ear_notches.append(
        Pos(sx * JOINT_X, JOINT_Y - 2.0, (-11.7 - 6.0) / 2.0) * Box(
            10.4, 14.0, 11.7 - 6.0))

# ---- driver mounting inserts: 6 x M3 on the vendor D98 pattern -------------
# The SAME receiver recipe as every other M3 insert in the release:
# a plain D4.6 blind bore, 4.0 insert depth + 0.2 relief, opening 0.2
# through the seat face (the rim recess floor here) -- identical to the
# tweeter-joint receiver bores.
front_bores = []
rear_bores = []
for k in range(6):
    az = math.radians(
        TERMINAL_AZ_DEG + MOUNT_AZ_FROM_TERMINAL_DEG + 60.0 * k)
    px, py = MOUNT_BC_R * math.cos(az), MOUNT_BC_R * math.sin(az)
    # front unit: opening at its recess floor z -4.75 (+0.2 overshoot)
    front_bores.append(
        Pos(px, py, -4.55 - INSERT_BORE_DEPTH) * Cylinder(
            INSERT_BORE_D / 2.0, INSERT_BORE_DEPTH + 0.2,
            align=(Align.CENTER, Align.CENTER, Align.MIN)))
    # rear unit: mirrored, opening at its recess floor z -37.75
    rear_bores.append(
        Pos(px, SPACING - py, -DEPTH + 4.35) * Cylinder(
            INSERT_BORE_D / 2.0, INSERT_BORE_DEPTH + 0.2,
            align=(Align.CENTER, Align.CENTER, Align.MIN)))

body = wrap + upper_wrap + web
for b in joint_bosses:
    body += b
body = body - cav - upper_cav - chamber
for group in (duct_pieces, [tail_socket], front_bores, rear_bores,
              joint_bores, ear_notches):
    body = body.clean()
    for c in group:
        cut = body - c
        if not cut.is_valid:
            cut = (body.clean() - c).clean()
        body = cut
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
for side in (1.0, -1.0):
    for az_deg in (45.0, 0.0, -45.0, -88.0):
        a = math.radians(az_deg)
        p = (side * DUCT_R * math.cos(a), DUCT_R * math.sin(a), DUCT_Z)
        assert blocked(Pos(*p) * Box(1.6, 1.6, 1.6)) < 1e-6, (side, az_deg)
assert blocked(Pos(0, -54.5, DUCT_Z) * Box(1.6, 1.6, 1.6)) < 1e-6
checks["twin_ducts_continuous_chamber_to_socket_both_sides"] = True
# duct burial: flare cover outside, wall to the cavity inside, and the
# separation up to the driver insert bores
assert blocked(Pos(52.9 * math.cos(math.radians(-30)),
                   52.9 * math.sin(math.radians(-30)), DUCT_Z)
               * Box(0.8, 0.8, 2.0)) > 1e-5
assert blocked(Pos(46.9, 0.0, DUCT_Z) * Box(0.8, 0.8, 2.0)) > 1e-5
assert blocked(Pos(49.0, 0.0, -11.22) * Box(1.0, 1.0, 0.3)) > 1e-6
checks["ducts_fully_buried_with_solid_walls"] = True
for k in range(6):
    az = math.radians(
        TERMINAL_AZ_DEG + MOUNT_AZ_FROM_TERMINAL_DEG + 60.0 * k)
    px, py = MOUNT_BC_R * math.cos(az), MOUNT_BC_R * math.sin(az)
    assert blocked(Pos(px, py, -7.0) * Box(1.5, 1.5, 1.2)) < 1e-6, (
        "front bore", k)
    assert blocked(
        Pos(px, SPACING - py, -DEPTH + 7.0) * Box(1.5, 1.5, 1.2)) < 1e-6, (
        "rear bore", k)
checks["driver_insert_bores_open_D98_pattern_all_12"] = True
assert blocked(Pos(21.6, 42.4, -8.0) * Box(0.5, 0.5, 2.0)) > 1e-6
checks["insert_bore_to_chamber_wall_solid"] = True
for sx in (-1.0, 1.0):
    assert blocked(Pos(sx * JOINT_X, JOINT_Y, -4.0) * Box(1.5, 1.5, 1.5)) < 1e-6
    assert blocked(Pos(sx * JOINT_X, JOINT_Y, -0.9) * Box(1.4, 1.4, 1.0)) > 1e-4
    assert blocked(Pos(sx * JOINT_X, JOINT_Y - 2.0, -9.0) * Box(2, 2, 2)) < 1e-6
checks["um_joint_receivers_on_released_24mm_spacing"] = True
assert blocked(Pos(0, 46.0, -5.6) * Box(2, 2, 1.2)) > 1e-4
assert blocked(Pos(0, 46.0, -36.85) * Box(2, 2, 1.2)) > 1e-4
assert blocked(Pos(0, 53.8, -1.5) * Box(2, 2, 2)) > 1e-3
assert blocked(Pos(0, 39.4, -41.0) * Box(1.2, 1.2, 2)) > 1e-4
checks["chamber_fully_buried_all_skins_intact"] = True
assert blocked(Pos(0, 20.0, -44.0) * Box(3, 3, 2.0)) < 1e-6
assert blocked(Pos(30.0, -30.0, -44.0) * Box(3, 3, 2.0)) < 1e-6
checks["rear_face_completely_clean_nothing_behind_it"] = True
bb = body.bounding_box()
assert abs(bb.max.X) <= 55.31 and abs(bb.min.X) <= 55.31
assert abs(bb.min.Z - (-DEPTH)) < 1e-6 and abs(bb.max.Z) < 1e-6
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

# Mesh-based verification of every fastener bore and the duct mouth: the
# kernel's own intersection can false-pass exactly where a cut failed,
# so the exported tessellation is the authority.
def _mesh_solid_fn(tris):
    ma, mb, mc = tris[:, 0], tris[:, 1], tris[:, 2]
    dn = ((mb[:, 1] - mc[:, 1]) * (ma[:, 0] - mc[:, 0])
          + (mc[:, 0] - mb[:, 0]) * (ma[:, 1] - mc[:, 1]))
    okm = np.abs(dn) > 1e-12
    dd = np.where(okm, dn, 1.0)
    def solid(x, y, z):
        w1 = ((mb[:, 1] - mc[:, 1]) * (x - mc[:, 0])
              + (mc[:, 0] - mb[:, 0]) * (y - mc[:, 1])) / dd
        w2 = ((mc[:, 1] - ma[:, 1]) * (x - mc[:, 0])
              + (ma[:, 0] - mc[:, 0]) * (y - mc[:, 1])) / dd
        w3 = 1 - w1 - w2
        h = okm & (w1 >= -1e-9) & (w2 >= -1e-9) & (w3 >= -1e-9)
        if not h.any():
            return False
        zz = (w1[h] * ma[h][:, 2] + w2[h] * mb[h][:, 2]
              + w3[h] * mc[h][:, 2])
        return (zz < z).sum() % 2 == 1
    return solid

_solid = _mesh_solid_fn(src)
for k in range(6):
    az = math.radians(
        TERMINAL_AZ_DEG + MOUNT_AZ_FROM_TERMINAL_DEG + 60.0 * k)
    px, py = MOUNT_BC_R * math.cos(az), MOUNT_BC_R * math.sin(az)
    assert not _solid(px, py, -7.0), f"front bore {k} closed in the mesh"
    assert not _solid(px, SPACING - py, -DEPTH + 7.0), (
        f"rear bore {k} closed in the mesh")
for sx in (1.0, -1.0):
    assert not _solid(sx * JOINT_X, JOINT_Y, -4.0), "joint receiver closed"
assert not _solid(0.0, -54.5, DUCT_Z), "duct socket closed in the mesh"
checks["mesh_verified_all_fastener_bores_and_socket"] = True

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
    "scope": "Obi-Wan profile only (pairs with the obiwan UM carrier's "
             "tweeter joint); not applicable to stock or slim",
    "mounting": {
        "per_driver": "6 x M3 heat-set inserts, bores D4.6 x 4.2 (the "
                      "released receiver recipe), on the "
                      "vendor D98 pattern (D3.5 rim holes), clocked "
                      "terminal+30+k*60 with the terminal at the waist",
        "crescent_to_um_support": "released Obi-Wan tweeter-joint "
                                  "contract: D9.8 bosses at the +-24 "
                                  "spacing (TWEETER_JOINT_X) in the "
                                  "addon front z-band (local -5.9..0), "
                                  "rear-opening M3 heat-set receivers "
                                  "D4.6 x 4.2 at local z -6.1..-1.9 "
                                  "(TWEETER_JOINT_INSERT_BORE_Z mapped), "
                                  "ear notches for the core's functional "
                                  "ears; first tweeter = front unit, "
                                  "nearest the joint; final y-"
                                  "registration to TWEETER_JOINT_Y at "
                                  "release integration",
    },
    "cabling": {
        "duct_d_mm": 4.2,
        "path": "TWIN redundant ducts fully buried at r=50 around the "
                "front unit, z centre -13.6 (baffle duct depth): waist "
                "chamber wall (x=+-21) -> each side's circle arc -> "
                "converge at the bottom",
        "handoff": "stepped socket in the lower edge at (0, y -56..-49.5, "
                   "z -13.6): D8 nozzle seat + D4.5 wire neck meeting "
                   "both arcs: continuous concealed duct, zero exposed "
                   "cable, nothing behind the rear face",
    },
    "body": {
        "volume_cm3": round(vol / 1000.0, 1),
        "exterior": "waveguide-family smoothstep flare R55.3->R44 over "
                    "31 mm per form, rolled front edge r2.5 and rear "
                    "corner, rounded-plan waist web r14, two joint boss "
                    "lobes at the bottom edge; the rear face is "
                    "completely clean -- no spine, no stem, no ducts "
                    "visible anywhere",
        "interior": "stepped shrink-wrap bores (retention shoulders "
                    "back up the 6 rim screws), buried waist terminal "
                    "chamber z -6.5..-36",
        "min_walls_mm": {
            "cavity_face_skins": 3.0,
            "rim_recess_to_chamber_skin": 1.75,
            "front_rim_recess_to_rear_cavity_pinch": 1.5,
            "flare_to_cavity_bands": 0.77,
            "duct_to_cavity_wall": 1.55,
            "insert_pocket_to_skirt_clearance": 0.35,
            "duct_flare_cover": 1.7,
            "duct_to_driver_insert_bore": 2.75,
            "insert_bore_to_chamber": 1.2,
        },
    },
    "checks_passed": checks,
    "print_notes": [
        "print.stl is front-face-down (source rotated 180 deg about X); "
        "the front faces and rim ring sit on the bed",
        "no captive magnets, no pause: slice normally",
        "prints fully support-free: nothing behind the rear face, all "
        "16 insert/joint bores are blind, and the buried D4.2/D8 ducts "
        "are self-bridging",
        "body is designed solid: print with high wall count / infill if "
        "the acoustic deadness of a solid part is wanted",
        "install the 12 heat-set inserts BEFORE the drivers: the inner "
        "arc of each pocket wall is 0.35 to the skirt clearance gap, so "
        "any melt bulge lands in that gap and is cleared before the "
        "driver goes in",
        "assembly: feed the cable pairs up the socket and twin ducts "
        "into the chamber; wire each unit, seat it, and fit 6 x M3 into "
        "the heat-set inserts behind the rim; the UM core's rear-driven "
        "M3 screws clamp the joint bosses while its duct nozzle seats "
        "in the D8 socket",
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
