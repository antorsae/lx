"""Approved Obiwan ND25FN-4 vase parts for the H2C shelf: design-frame STLs, print authorities and the top's support blocker.

The geometry comes from the Osado concept pipeline (see PROVENANCE.md); this script only packages the approved meshes
and derives the support blocker + duct contract of the top:
  * the released keyed LM top's blocker and duct contract below y 278, where the vase top's rebuilt band starts,
    plus all four of its L22 M5 pilot blockers (0/60/120/180 deg; the band rebuilds the 60/120 bores at the same place);
  * the vase top's own lumens: T duct (D6), T->t link (D6), MU10 lead (D8.2) including its rear dive;
  * 4 x MU10 M3 pilot blockers, sized like the released UM pilot blockers;
  * duct blockers stay 0.8 + 0.5 mm clear of the MU10 and L22 seat ceilings (seat-clear rule, obiwan_support_blocker.py).
Hands: LX_HAND=std (tweeters right of the MU10, seen from the front) or mirror (tweeters left: built in the x-mirrored
frame and mirrored back, so the released keyed LM region stays exactly as released). LX_WING_PREFIX names the wing inputs.
usage: build_package.py OUT_DIR TOP CAP_FRONT CAP_REAR BUILD_JSON RELEASED_TOP_BLOCKER_JSON WING_DIR
"""
import os, sys, json, math, hashlib
from pathlib import Path
import numpy as np, trimesh, manifold3d as m3

out, top_p, capF_p, capR_p, build_p, rel_json_p, wing_dir = (Path(a) for a in sys.argv[1:8])
C = json.load(open(build_p))
MIR = os.environ.get("LX_HAND", "std") == "mirror"
PFX = "nd25fn4_vase_obiwan_tweeters_" + ("left" if MIR else "right"); WPFX = os.environ.get("LX_WING_PREFIX", "obiwan_v11_top")
CLEAR = 0.25                                  # lumen -> blocker clearance, as the released blockers
Y2 = C["Y2"]                                  # 278: the vase top rebuilds everything above this
LM, SEAT_LM, R_LMrec = np.array([0.0, 200.981]), 12.3, 110.6
MU, SEAT_UM, R_UMrec = np.array(C["MU"]), 14.3, 49.3
UM_PILOT_R, UM_PILOT_ANG, UM_PILOT_D, UM_PILOT_DEPTH = 89.5 / 2, (58.0, 148.0, 238.0, 328.0), 4.6, 4.0
EXTRA, LAYERS = 0.8, 0.5                      # shelf dilation of blockers; first seat-ceiling layers that must keep their support

to_m = lambda q: m3.Manifold(m3.Mesh(vert_properties=np.array(q.vertices, np.float32), tri_verts=np.array(q.faces, np.uint32)))
def to_t(M): mm = M.to_mesh(); return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cyl(c, r, z0, z1, n=64): return to_m(trimesh.creation.cylinder(radius=r, segment=[[c[0], c[1], z0], [c[0], c[1], z1]], sections=n))
def tube(points, r, h=0.2):
    """closed tube around a polyline: marching cubes of its distance field (one clean surface, no sphere/cylinder unions)"""
    from scipy.spatial import cKDTree
    from skimage.measure import marching_cubes
    P = np.asarray(points, float); dense = [P[0]]
    for a, b in zip(P[:-1], P[1:]):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / 0.05))); dense += list(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
    dense = np.array(dense); lo = dense.min(0) - r - 3 * h; hi = dense.max(0) + r + 3 * h
    axes = [np.arange(lo[k], hi[k] + h, h) for k in range(3)]
    G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    d = cKDTree(dense).query(G, workers=-1)[0].reshape(len(axes[0]), len(axes[1]), len(axes[2]))
    v, f, _, _ = marching_cubes(d, level=r + h / 2, spacing=(h, h, h))       # + h/2: the facetted surface stays outside r
    t = trimesh.Trimesh(v + lo, f[:, ::-1], process=True)
    if t.volume < 0: t.invert()
    return to_m(t)
def clip_below(points, y):
    """polyline part with y <= y, cut exactly at y"""
    P, keep = np.asarray(points, float), []
    for a, b in zip(P[:-1], P[1:]):
        if a[1] <= y: keep.append(a)
        if (a[1] - y) * (b[1] - y) < 0: keep.append(a + (b - a) * (y - a[1]) / (b[1] - a[1])); keep.append(None)
    if P[-1][1] <= y: keep.append(P[-1])
    runs, cur = [], []
    for q in keep + [None]:
        if q is None:
            if len(cur) >= 2: runs.append(np.array(cur))
            cur = []
        else: cur.append(q)
    return runs
(out / "STL").mkdir(parents=True, exist_ok=True); (out / "support_blockers").mkdir(exist_ok=True)

# ---------------------------------------------------------------- duct contract
V = [("vase_t_duct_lumen", C["tubeT"], C["TS_R"]), ("vase_T_to_t_link_lumen", C["tubet"], 3.0), ("vase_mu10_lead_lumen", C["tubeUM"], 4.1)]
regions = [dict(kind="polyline_tube", name=n, points_xyz_mm=np.round(np.asarray(p, float), 4).tolist(), radius_mm=r) for n, p, r in V]
def outside_chamber(points, c, r):
    """cut a lumen's audit region where its end enters a tweeter wiring chamber (R_CAPB), pulled back by the lumen radius:
    support inside the chamber is ordinary cavity support (removable through the cap bore), not support in the duct"""
    P = np.asarray(points, float); c = np.asarray(c, float); lim = C["R_CAPB"] + r
    ends = [0, len(P) - 1]; keep = np.ones(len(P), bool)
    for e_, step in ((0, 1), (len(P) - 1, -1)):
        if np.linalg.norm(P[e_, :2] - c) >= lim: continue
        i = e_
        while 0 <= i < len(P) and np.linalg.norm(P[i, :2] - c) < lim: keep[i] = False; i += step
        a_, b_ = P[i], P[i - step]; da, db = np.linalg.norm(a_[:2] - c), np.linalg.norm(b_[:2] - c); t_ = (da - lim) / (da - db)
        P[i - step] = a_ + (b_ - a_) * t_; keep[i - step] = True
    return P[keep]
for reg, cen in ((regions[0], C["T"]), (regions[1], C["T"]), (regions[1], C["t"])):
    reg["points_xyz_mm"] = np.round(outside_chamber(reg["points_xyz_mm"], cen, reg["radius_mm"]), 4).tolist()
blocker_V = V                                                   # the blocker keeps the full lumens
rel = json.load(open(rel_json_p)); rc = rel["duct_collision_contract"]
for r in rc["regions"]:
    for i, run in enumerate(clip_below(r["points_xyz_mm"], Y2)):
        regions.append(dict(kind=r["kind"], name=f"released_{r['name']}" + (f"_{i}" if i else ""), points_xyz_mm=np.round(run, 4).tolist(), radius_mm=r["radius_mm"]))

# ---------------------------------------------------------------- blocker solid
Mrel = np.array(rel["source_to_stl_matrix"])
rb = trimesh.load(rel_json_p.parent / rel["support_blocker"]); rb.apply_transform(np.linalg.inv(Mrel))
rb_comps = rb.split(only_watertight=False)
big = [c for c in rb_comps if c.volume > 1000.0]
# all four released L22 M5 insert-pilot blockers (0/60/120/180 deg; 268.8 mm3 each). The two 216 mm3 pieces are the
# released UM pilots at y 315, replaced by the vase's MU10 pilots below.
l22 = [c for c in rb_comps if abs(c.volume - 268.8) < 5.0 and np.hypot(c.centroid[0], c.centroid[1] - 200.981) > 100.0]
assert len(big) == 2 and len(l22) == 4, (len(big), len(l22))
slab = to_m(trimesh.creation.box(bounds=[[-400, -100, -100], [400, Y2, 100]]))
keep = [cyl(cxy, rr + EXTRA + 0.05, seat - EXTRA - LAYERS, 200.0, 256) for cxy, rr, seat in ((MU, R_UMrec, SEAT_UM), (LM, R_LMrec, SEAT_LM))]
def seat_clear(M):
    for k in keep: M = M - k
    return M
def reloads_clean(t):
    """the shelf reloads the blocker STL with vertex merging, then rebuilds each component as a manifold"""
    import io
    buf = io.BytesIO(); t.export(buf, file_type="stl"); buf.seek(0)
    r = trimesh.load_mesh(buf, file_type="stl", process=True); r.update_faces(r.nondegenerate_faces(height=1e-12)); r.remove_unreferenced_vertices()
    for c in r.split(only_watertight=False):
        q = m3.Manifold(m3.Mesh(vert_properties=np.array(c.vertices, np.float32), tri_verts=np.array(c.faces, np.uint32)))
        if q.status() != m3.Error.NoError or c.volume < 1.0: return False
    return True
def clean(M, label):
    """one closed part that survives the STL round trip; tiny tolerances only collapse near-coincident vertices"""
    for tol in (0.0, 0.005, 0.01, 0.02, 0.05):
        t = to_t(M if tol == 0 else M.simplify(tol))
        t = trimesh.util.concatenate([c for c in t.split(only_watertight=False) if c.volume > 1.0])
        if reloads_clean(t): return t
    raise SystemExit(f"support blocker part {label} does not survive an STL round trip")
# The shelf unions the parts itself (after its 0.8 mm dilation), so the parts are exported as separate overlapping shells:
# a single pre-unioned blocker left near-coincident vertices that an STL round trip turns non-manifold.
parts = [clean(seat_clear(to_m(c) ^ slab), f"released duct {i}") for i, c in enumerate(big)]
parts += [clean(seat_clear(tube(p, r + CLEAR)), n) for n, p, r in blocker_V]
parts += [clean(to_m(c), "released L22 pilot") for c in l22]
for ang in UM_PILOT_ANG:                      # (both hands: the real MU10 pilots stay at 58/148/238/328 deg) released UM pilot blockers: D5.2, from 0.3 below the bore floor to 2.4 above the seat
    c = MU + UM_PILOT_R * np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
    parts.append(clean(cyl(c, UM_PILOT_D / 2 + 0.3, SEAT_UM - UM_PILOT_DEPTH - 0.3, SEAT_UM + 2.4), f"MU10 pilot {ang:.0f}"))
blk_t = trimesh.util.concatenate(parts)
assert reloads_clean(blk_t), "support blocker parts share vertices after the STL round trip"
assert all(p.is_watertight and p.is_volume for p in parts), "each support blocker part must be a closed volume"

# ---------------------------------------------------------------- publish
def publish(mesh, name, what, installed_to_source=None):
    p = out / "STL" / f"{name}.stl"; mesh.export(p)
    extra = {} if installed_to_source is None else dict(installed_to_source_matrix=installed_to_source.tolist(),
        source_frame_note="cap print frame: installed pose rotated 180 deg about X, so the shelf's front-down pose puts the visible skin on top")
    json.dump(dict(schema_version=1, part=name, stl=p.name, stl_sha256=sha(p), description=what,
                   source_frame="Obi-Wan installed design frame: X lateral, Y up, Z forward (towards the listener), mm",
                   source_to_stl_matrix=np.eye(4).tolist(), qualification="digital checks passed; not yet printed", **extra),
              open(p.with_suffix(".print.json"), "w"), indent=2)
    return p
top_path = publish(trimesh.load(top_p), f"{PFX}_top", "keyed LM top fused with the MU10 seat and the front (T) / rear (t) ND25FN-4 waveguides, tweeters "
                   + ("left" if MIR else "right") + " of the MU10 seen from the front")
bp = out / "support_blockers" / f"{PFX}_top.support_blocker.stl"; blk_t.export(bp)
json.dump(dict(schema_version=1, kind="bambu_support_blocker", part=top_path.stem, purpose="forbid_support_inside_functional_ducts",
               main_stl=f"../STL/{top_path.name}", main_stl_sha256=sha(top_path), support_blocker=bp.name, support_blocker_sha256=sha(bp),
               source_to_stl_matrix=np.eye(4).tolist(), modifier_clearance_mm=CLEAR,
               duct_collision_contract=dict(schema_version=1, coordinate_space="authoritative_source_mm", modifier_clearance_mm=CLEAR,
                                            split_half=rc["split_half"], split_seam_y_mm=rc["split_seam_y_mm"], regions=regions)),
          open(bp.with_suffix(".json"), "w"), indent=1)
publish(trimesh.load(capF_p), f"{PFX}_cap_front_tweeter", "twist-lock cap closing the front tweeter (T) from the rear face")
FLIP = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])            # rear-tweeter cap: skin on the front face -> print skin-up
publish(trimesh.load(capR_p).apply_transform(FLIP), f"{PFX}_cap_rear_tweeter", "twist-lock cap closing the rear tweeter (t) from the front face", FLIP)
for slug in ("flat", "graded"):
    for side in ("left", "right"):
        publish(trimesh.load(wing_dir / f"{WPFX}_wing_{slug}_{side}.stl"), f"{PFX}_wing_{slug}_{side}", f"{slug} {side} wing fitted to the vase top")

print("regions:", [(r["name"], len(r["points_xyz_mm"]), r["radius_mm"]) for r in regions])
print("blocker: %d parts, %.0f mm3 summed (overlaps counted twice), watertight %s" % (len(parts), sum(p.volume for p in parts), blk_t.is_watertight))
for p in sorted((out / "STL").glob("*.stl")):
    m = trimesh.load(p); print(f"{p.name}: watertight {m.is_watertight}, {m.volume / 1000:.1f} cm3, bbox {np.round(m.extents, 1).tolist()}")
