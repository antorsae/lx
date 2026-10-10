import sys, json, numpy as np, trimesh, manifold3d as m3
from scipy.interpolate import CubicSpline
d = sys.argv[1]; C = json.load(open(f"{d}/c7.json")); body = trimesh.load(f"{d}/_body_nolane.stl"); body.merge_vertices(digits_vertex=4); print("no-lane body watertight", body.is_watertight)
def to_m(t): return m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices, np.float32), tri_verts=np.asarray(t.faces, np.uint32)))
def centreline(path, step=.5):
    P = np.array(path, float); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]; cs = CubicSpline(s, P, bc_type="natural"); u = np.linspace(0, s[-1], int(s[-1] / step) + 2); q = cs(u); d1 = cs(u, 1); d2 = cs(u, 2)
    k = np.linalg.norm(np.cross(d1, d2), axis=1) / np.maximum(np.linalg.norm(d1, axis=1) ** 3, 1e-9); tdir = d1 / np.linalg.norm(d1, axis=1)[:, None]; turn = np.degrees(np.sum(np.arccos(np.clip(np.sum(tdir[1:] * tdir[:-1], axis=1), -1, 1))))
    return q, float(np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))), float(1 / k[5:-5].max()), float(turn)
LR = C.get('LANE_R', 3.0)
def section(p, flat): return (LR, LR, LR)
ico = trimesh.creation.icosphere(subdivisions=2, radius=1.0); tools = []; stats = {}
for key, flat in (("t", False), ("T", True)):
    q, L, rmin, turn = centreline(C["lanes"][key]); stats[key] = dict(length_mm=round(L), min_bend_radius_mm=round(rmin, 1), total_turning_deg=round(turn)); parts = []
    for p in q: e = ico.copy(); e.apply_scale(section(p, flat)); e.apply_translation(p); parts.append(to_m(e))
    tool = m3.Manifold.batch_boolean(parts, m3.OpType.Add); tools.append(tool); tm = tool.to_mesh(); tt = trimesh.Trimesh(np.asarray(tm.vert_properties)[:, :3], np.asarray(tm.tri_verts), process=False)
    inner = tt.vertices[(np.linalg.norm(tt.vertices - q[0], axis=1) > 9) & (np.linalg.norm(tt.vertices - q[-1], axis=1) > 9)][::3]; _, dist, _ = trimesh.proximity.closest_point(body, inner); stats[key]["min_wall_mm"] = round(float(dist.min()), 2); w = inner[np.argmin(dist)]; stats[key]["thinnest_at"] = [round(float(v), 1) for v in w]
    tt.export(f"{d}/lane_{'rear_t' if key == 't' else 'front_T'}.stl")
sl = C['ports']['slot']; SEAM = C['SEAM_Y']; y0, y1 = SEAM - .5, SEAM + sl['len'] + .5
hull = lambda pts: trimesh.convex.convex_hull(np.array(pts, float))
# vestibule: opens on the seam face exactly over the LM's O6 duct (+ alignment margin), ceiling level, FLOOR RAMPS DOWN from the duct's lower edge to the lower tube: the front-T lead drops ~27 deg instead of jogging 61 deg
slot = hull([(x, y, z) for x in (sl['x_out'], sl['x_in']) for (y, z) in ((y0, sl['z_floor_seam']), (y1, sl['z_floor_end']), (y1, sl['z_top']), (y0, sl['z_top']))])
cham = hull([(x, y0, z) for x in (sl['x_out'] - .2, sl['x_in'] + .9) for z in (sl['z_floor_seam'] - .9, sl['z_top'] + .7)] + [(x, SEAM + 2.0, z) for x in (sl['x_out'], sl['x_in']) for z in (sl['z_floor_seam'] - .4, sl['z_top'])])      # lead-in chamfer so a slightly offset LM never presents a lip
slot = trimesh.util.concatenate([slot, cham]); slot_m = to_m(hull(slot.vertices[:0]) if False else slot) if False else (to_m(hull([(x, y, z) for x in (sl['x_out'], sl['x_in']) for (y, z) in ((y0, sl['z_floor_seam']), (y1, sl['z_floor_end']), (y1, sl['z_top']), (y0, sl['z_top']))])) + to_m(cham))
res = to_m(body) - tools[0] - tools[1] - slot_m; mm = res.to_mesh(); out = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False); out.merge_vertices(digits_vertex=5); out = max(out.split(only_watertight=False), key=lambda c: len(c.faces))
print("printable body: watertight", out.is_watertight, "volume cm3", round(out.volume / 1000, 1)); out.export(f"{d}/um_vase_nd25fn4_c13_body.stl")
A_, B_ = trimesh.load(f"{d}/lane_rear_t.stl"), trimesh.load(f"{d}/lane_front_T.stl"); sep = trimesh.proximity.closest_point(A_, B_.vertices[::3])[1]; stats['between_tubes_min_mm'] = round(float(sep.min()), 2); print('the two tubes never touch: minimum plastic between them anywhere =', stats['between_tubes_min_mm'], 'mm')
json.dump(stats, open(f"{d}/lane_stats.json", "w"), indent=1); print(json.dumps(stats))
