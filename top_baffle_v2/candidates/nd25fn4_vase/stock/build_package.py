"""Stock ND25FN-4 vase for the H2C shelf: seam-B joint of the H2C Stock LM, print authorities and support blocker.

The vase body (vase25) was modelled against the P2S seam B (two female dovetail pockets). The H2C Stock LM ends flat at
y 315.95 and takes two D3 pins that the H2C upper carries. This script fills the two dovetail pockets, pins included,
with the released H2C Stock upper's own material inside the pocket boxes, then proves:
  * zero overlap with both H2C Stock LM states, and the pins enter their D3.3 sockets;
  * in the pocket boxes the result equals the released H2C upper (no other seam feature changes);
  * the vase's ramped cable vestibule contains the LM's D6 duct mouth; M3 clamp insert bore is unchanged.
Support blocker: released H2C upper M3 insert + 4 MU10 pilot blockers, plus the vase's own T/t lanes and vestibule.
Hands: LX_HAND=std (tweeters right of the MU10, seen from the front) or mirror (tweeters left; built in the x-mirrored
frame with the lanes re-cut to the LM duct, its H2C pocket fill done inside conform.py before the vestibule cut).
usage: build_package.py OUT_DIR VASE_DIR H2C_STL_DIR H2C_BLOCKER_DIR   (VASE_DIR: body, caps and c7.json in the design frame)
"""
import os, sys, json, hashlib
from pathlib import Path
import numpy as np, trimesh, manifold3d as m3
from shapely.geometry import Polygon, box as sbox

out, vd, h2c, h2cb = (Path(a) for a in sys.argv[1:5])
C = json.load(open(vd / "c7.json"))
MIR = os.environ.get("LX_HAND", "std") == "mirror"
PFX = "nd25fn4_vase_stock_tweeters_" + ("left" if MIR else "right")
BODY = "um_vase_nd25fn4_c25m_body.stl" if MIR else "um_vase_nd25fn4_c25_body.stl"
SEAM_Y, CLEAR = 315.95, 0.25
POCKETS = [(11.8, 26.2), (-36.2, -21.8)] if MIR else [(-26.2, -11.8), (21.8, 36.2)]   # DOVETAILS_B at x -19 / 29 (mirrored: 19 / -29), head 14 + 0.05 female growth
to_m = lambda q: m3.Manifold(m3.Mesh(vert_properties=np.array(q.vertices, np.float32), tri_verts=np.array(q.faces, np.uint32)))
def to_t(M): mm = M.to_mesh(); return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def src(n):
    m = trimesh.load(h2c / f"{n}.stl"); a = json.load(open(h2c / f"{n}.print.json")); M = np.array(a["source_to_stl_matrix"])
    m.apply_transform(np.linalg.inv(M)); return m, M
def bx(x0, x1, y0, y1, z0, z1): return to_m(trimesh.creation.box(bounds=[[x0, y0, z0], [x1, y1, z1]]))
def tube(points, r):
    P = np.asarray(points, float); B = to_m(trimesh.creation.icosphere(subdivisions=2, radius=r).apply_translation(P[0]))
    for a, b in zip(P[:-1], P[1:]):
        if np.linalg.norm(b - a) > 1e-6: B = B + to_m(trimesh.creation.cylinder(radius=r, segment=[a, b], sections=24))
        B = B + to_m(trimesh.creation.icosphere(subdivisions=2, radius=r).apply_translation(b))
    return B
(out / "STL").mkdir(parents=True, exist_ok=True); (out / "support_blockers").mkdir(exist_ok=True)

# ---------------------------------------------------------------- seam B: dovetail pockets -> H2C pin joint
V = to_m(trimesh.load(vd / BODY))
U_t, _ = src("h2c_stock_upper"); U = to_m(U_t)
if MIR: body = V                                               # already filled inside conform.py, before the lanes and vestibule were cut
else:
    fill = None
    for x0, x1 in POCKETS:
        k = bx(x0, x1, SEAM_Y - 4.5, SEAM_Y + 6.2, 0.0, 18.3); f = U ^ k
        fill = f if fill is None else fill + f
    body = V + fill
bt = to_t(body)
assert bt.is_watertight and bt.is_volume and len(bt.split(only_watertight=False)) == 1, "stock vase must be one closed solid"
for x0, x1 in POCKETS[:1] if MIR else POCKETS:                 # (mirrored: the second box also holds the vestibule over the LM duct)
    k = bx(x0, x1, SEAM_Y - 4.5, SEAM_Y + 6.2, 0.0, 18.3)
    d = ((body ^ k) - (U ^ k)).volume() + ((U ^ k) - (body ^ k)).volume()
    print(f"pocket x {x0}..{x1}: difference to the released H2C upper {d:.3f} mm3"); assert d < 1.0
pins = U ^ bx(-40, 40, SEAM_Y - 5.0, SEAM_Y, 0.0, 18.3); missing = (pins - body).volume()
print(f"D3 pins below the seam: released {pins.volume():.1f} mm3, missing from the vase {missing:.3f} mm3"); assert pins.volume() > 50 and missing < 0.05
for n in ("h2c_stock_lm_no_floor_stand", "h2c_stock_lm_upper_floor_stand"):
    L, _ = src(n); ov = (body ^ to_m(L)).volume(); print(f"{n}: overlap {ov:.3f} mm3"); assert ov < 0.01
sec = bt.section(plane_origin=[0, SEAM_Y + 0.1, 0], plane_normal=[0, 1, 0])
P2, _ = sec.to_2D(to_2D=np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, SEAM_Y + 0.1], [0, 0, 0, 1]]))
holes = [Polygon(h) for p in P2.polygons_full for h in p.interiors]
lm = C["ports"]["lm_duct"]; mouth = sbox(lm["x"] - 3.0, lm["z"] - 3.0, lm["x"] + 3.0, lm["z"] + 3.0)
assert any(h.buffer(0.05).contains(mouth) for h in holes), "cable vestibule must contain the LM D6 duct mouth"
assert any(abs(h.centroid.x) < 0.05 and abs(h.centroid.y - 12.55) < 0.05 for h in holes), "M3 clamp insert bore missing"
print("seam holes:", [(round(h.centroid.x, 2), round(h.centroid.y, 2), round(h.area, 1)) for h in holes])

# ---------------------------------------------------------------- duct contract + blocker
slot = C["ports"]["slot"]
regions = [dict(kind="polyline_tube", name=f"vase_{'front_T' if k == 'T' else 'rear_t'}_lane_lumen", points_xyz_mm=np.round(np.asarray(p, float), 4).tolist(), radius_mm=C["LANE_R"])
           for k, p in C["lanes"].items()]
def to_mouth(r, key):
    """end a lane's audit region at its mouth in the tweeter-cavity wall. The polyline runs on into the cavity, and
    support there is ordinary cavity support (removable through the tweeter opening), not support in the lumen."""
    P = np.asarray(r["points_xyz_mm"]); c = np.asarray(C[key], float)
    d = P[-1, :2] - c; d /= np.linalg.norm(d)
    hits, _, _ = bt.ray.intersects_location([[c[0], c[1], P[-1, 2]]], [[d[0], d[1], 0.0]])
    wall = min(np.linalg.norm(h[:2] - c) for h in hits)
    rad = np.linalg.norm(P[:, :2] - c, axis=1); k = int(np.argmax(rad < wall))    # first polyline point inside the cavity
    a, b = P[k - 1], P[k]; t = (rad[k - 1] - wall) / (rad[k - 1] - rad[k])
    r["points_xyz_mm"] = np.round(np.vstack([P[:k], a + (b - a) * t]), 4).tolist(); r["mouth_wall_radius_mm"] = round(float(wall), 3)
blocker_regions = [dict(r) for r in regions]                    # the blocker keeps the full polylines
for r, key in zip(regions, C["lanes"].keys()): to_mouth(r, key)
print("lane mouths:", [(r["name"], r["mouth_wall_radius_mm"]) for r in regions])
regions.append(dict(kind="polyline_tube", name="vase_cable_vestibule", radius_mm=3.0,
                    points_xyz_mm=[[lm["x"], SEAM_Y, lm["z"]], [lm["x"], SEAM_Y + slot["len"], (slot["z_top"] + slot["z_floor_end"]) / 2]]))
blk = None
for r in blocker_regions:
    T_ = tube(r["points_xyz_mm"], r["radius_mm"] + CLEAR); blk = T_ if blk is None else blk + T_
vest = trimesh.convex.convex_hull(np.array([[x, y, z] for x in (slot["x_out"] - CLEAR, slot["x_in"] + CLEAR) for y, zs in
       ((SEAM_Y - 0.5, (slot["z_floor_seam"] - CLEAR, slot["z_top"] + CLEAR)), (SEAM_Y + slot["len"], (slot["z_floor_end"] - CLEAR, slot["z_top"] + CLEAR))) for z in zs]))
blk = blk + to_m(vest)
rb = trimesh.load(h2cb / "h2c_stock_upper.stl"); rb.apply_transform(np.linalg.inv(np.array(json.load(open(h2c / "h2c_stock_upper.print.json"))["source_to_stl_matrix"])))
kept = [c for c in rb.split(only_watertight=False) if 50.0 < c.volume < 120.0 and c.bounds[0, 1] > 315.0]
assert len(kept) == 5, len(kept)                              # M3 clamp insert + 4 MU10 pilots
for c in kept: blk = blk + to_m(c)
seat_keep = 18.3 - 0.8 - 0.5                                  # MU10 seat plane minus shelf dilation and first ceiling layers
assert max(np.max(np.asarray(r["points_xyz_mm"])[:, 2]) + r["radius_mm"] + CLEAR for r in regions) < seat_keep
def reloads_clean(M):
    """the shelf reloads the blocker STL with vertex merging, then rebuilds each component as a manifold"""
    import io
    t = to_t(M); buf = io.BytesIO(); t.export(buf, file_type="stl"); buf.seek(0)
    r = trimesh.load_mesh(buf, file_type="stl", process=True); r.update_faces(r.nondegenerate_faces(height=1e-12)); r.remove_unreferenced_vertices()
    for c in r.split(only_watertight=False):
        q = m3.Manifold(m3.Mesh(vert_properties=np.array(c.vertices, np.float32), tri_verts=np.array(c.faces, np.uint32)))
        if q.status() != m3.Error.NoError or c.volume < 1.0: return False
    return True
for tol in (0.0, 0.005, 0.01, 0.02):                 # collapse near-coincident vertices left by the tube unions
    cand = blk if tol == 0 else blk.simplify(tol)
    if reloads_clean(cand): blk = cand; break
else: raise SystemExit("support blocker does not survive an STL round trip")
print("blocker simplify tolerance", tol)
blk_t = to_t(blk)
assert blk_t.is_watertight and blk_t.is_volume, "support blocker must be a closed volume"

def publish(mesh, name, what, installed_to_source=None):
    p = out / "STL" / f"{name}.stl"; mesh.export(p)
    extra = {} if installed_to_source is None else dict(installed_to_source_matrix=installed_to_source.tolist(),
        source_frame_note="cap print frame: installed pose rotated 180 deg about X, so the shelf's front-down pose puts the visible skin on top")
    json.dump(dict(schema_version=1, part=name, stl=p.name, stl_sha256=sha(p), description=what,
                   source_frame="Stock installed design frame: X lateral, Y up, Z forward (towards the listener), mm",
                   source_to_stl_matrix=np.eye(4).tolist(), qualification="digital checks passed; not yet printed", **extra),
              open(p.with_suffix(".print.json"), "w"), indent=2)
    return p
up = publish(bt, f"{PFX}_upper", "Stock upper module with the MU10 and front (T) / rear (t) ND25FN-4 waveguides, tweeters "
             + ("left" if MIR else "right") + " of the MU10 seen from the front; H2C seam-B pin joint")
bp = out / "support_blockers" / f"{PFX}_upper.support_blocker.stl"; blk_t.export(bp)
json.dump(dict(schema_version=1, kind="bambu_support_blocker", part=up.stem, purpose="forbid_support_inside_functional_ducts",
               main_stl=f"../STL/{up.name}", main_stl_sha256=sha(up), support_blocker=bp.name, support_blocker_sha256=sha(bp),
               source_to_stl_matrix=np.eye(4).tolist(), modifier_clearance_mm=CLEAR,
               duct_collision_contract=dict(schema_version=1, coordinate_space="authoritative_source_mm", modifier_clearance_mm=CLEAR, regions=regions)),
          open(bp.with_suffix(".json"), "w"), indent=1)
publish(trimesh.load(vd / "cap_front_tweeter_REAR_FACE.stl"), f"{PFX}_cap_front_tweeter", "twist-lock cap closing the front tweeter (T) from the rear face")
FLIP = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])            # rear-tweeter cap: skin on the front face -> print skin-up
publish(trimesh.load(vd / "cap_rear_tweeter_FRONT_FACE.stl").apply_transform(FLIP), f"{PFX}_cap_rear_tweeter", "twist-lock cap closing the rear tweeter (t) from the front face", FLIP)
print("blocker: %.0f mm3, %d components" % (blk_t.volume, len(blk_t.split(only_watertight=False))))
for p in sorted((out / "STL").glob("*.stl")):
    m = trimesh.load(p); print(f"{p.name}: watertight {m.is_watertight}, {m.volume / 1000:.1f} cm3, bbox {np.round(m.extents, 1).tolist()}")
