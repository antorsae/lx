"""V7 wings: the released Obi-Wan wings (flat + graded, left + right) trimmed to clear the V7 outline, with their upper magnet
moved from 64/116 deg to 40/140 deg (same rotation as the LM side). Everything else (shoulder magnet, lower contact, outer
silhouette below the shoulder line, grading) is the released part unchanged."""
import os, sys, json, math, numpy as np, trimesh, manifold3d as m3
from shapely.geometry import Polygon, Point, box
from shapely.ops import unary_union
d, V = sys.argv[1], sys.argv[2]; REPO = os.environ.get("LX_REPO_INPUTS", "/Users/antor/gh/lx/top_baffle_v2/build/h2c/STL/")
C = json.load(open(f"{d}/{V}_build.json")); plan = Polygon(C["plan"]).buffer(0); LM = np.array([0, 200.981])
GAP, R_ROUND = 0.05, 10.0
def to_m(q): return m3.Manifold(m3.Mesh(vert_properties=np.asarray(q.vertices, np.float32), tri_verts=np.asarray(q.faces, np.uint32)))
def to_t(M): mm = M.to_mesh(); return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)
def prism(poly, z0, z1):
    out = None
    for g in ([poly] if poly.geom_type == "Polygon" else list(poly.geoms)):
        rings = [np.asarray(g.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in g.interiors]
        s = m3.Manifold.extrude(m3.CrossSection(rings, m3.FillRule.EvenOdd), z1 - z0).translate((0, 0, z0)); out = s if out is None else out + s
    return out
body = trimesh.load(f"{d}/{V}.stl"); bodyM = to_m(max(body.split(only_watertight=False), key=lambda p: abs(p.volume)))
report = {}
for kind in ("flat", "graded"):
    for side, sgn, a0, a1 in (("left", -1, 116.0, 171.0), ("right", 1, 64.0, 9.0)):   # V13: stations at 9/171 deg (clear of the released lumens)
        nm = f"h2c_obiwan_wing_{kind}_{side}"; P = json.load(open(f"{REPO}{nm}.print.json")); w = trimesh.load(f"{REPO}{nm}.stl")
        w.apply_transform(np.linalg.inv(np.array(P["source_to_stl_matrix"])))
        parts = w.split(only_watertight=False); main = max(parts, key=lambda p: abs(p.volume)); vo = [p for p in parts if p is not main and abs(p.volume) > 10]
        # cutter: the V7 outline (+gap) and everything above the released shoulder line, closed with R10 so the new corner is rounded
        x_out, y_out, x_in, y_in = 152.4, 256.0 - 1.5, 60.0, 303.5 - 1.5   # 1.5 mm below the released shoulder edge: no hairline strip where the two nearly coincide
        line_poly = Polygon([(sgn * 400, y_out + (400 - x_out) * (y_in - y_out) / (x_in - x_out) * -1 if False else y_out - (400 - x_out) * (y_in - y_out) / (x_out - x_in)),
                             (0, y_out + x_out * (y_in - y_out) / (x_out - x_in)), (0, 800), (sgn * 400, 800)])
        cut2d = unary_union([plan.buffer(GAP, 64), line_poly]).buffer(R_ROUND, 64).buffer(-R_ROUND, 64)
        nw = to_m(main) - prism(cut2d, -40, 60)
        out = to_t(nw); out.merge_vertices(); out = max(out.split(only_watertight=False), key=lambda p: abs(p.volume))
        sec = out.section(plane_origin=[0, 0, 12.0], plane_normal=[0, 0, 1]); P2 = unary_union([Polygon(e[:, :2]).buffer(0) for e in sec.discrete if len(e) > 3])
        st = LM + 116.0 * np.array([math.cos(math.radians(a1)), math.sin(math.radians(a1))])
        sliver = P2.difference(P2.buffer(-2.5, 32).buffer(2.5, 32)).intersection(box(-400, 262, 400, 400)).buffer(0.3).difference(Point(*st).buffer(4.0))   # keep the magnet station's land intact   # anything narrower than 5 mm at the new top (the hooked tip at the body corner)
        if not sliver.is_empty:
            out = to_t(to_m(out) - prism(sliver, -40, 60)); out.merge_vertices(); out = max(out.split(only_watertight=False), key=lambda p: abs(p.volume))
        keep_v = [p for p in vo if out.contains(p.vertices).all()]
        new_v = []
        for p in vo:
            c = p.vertices.mean(0); a = math.degrees(math.atan2(c[1] - LM[1], c[0] - LM[0]))
            if abs(a - a0) < 1.0:
                q = p.copy(); q.apply_transform(trimesh.transformations.rotation_matrix(math.radians(a1 - a0), [0, 0, 1], [LM[0], LM[1], 0])); new_v.append(q)
        ok_new = all(out.contains(q.vertices).all() for q in new_v)
        wall = min(abs(trimesh.proximity.signed_distance(out, q.vertices)).min() for q in new_v) if new_v else None
        clash = (bodyM ^ to_m(out)).volume()
        mind = trimesh.proximity.closest_point(body, out.vertices[np.random.default_rng(0).choice(len(out.vertices), min(4000, len(out.vertices)), replace=False)])[1].min()
        final = trimesh.util.concatenate([out] + keep_v + new_v); final.export(f"{d}/{V}_wing_{kind}_{side}.stl")
        report[f"{kind}_{side}"] = dict(vol_cm3=round(out.volume / 1000, 1), was_cm3=round(main.volume / 1000, 1), watertight=bool(out.is_watertight), magnets_kept=len(keep_v), magnet_new_inside=bool(ok_new),
                                        new_magnet_min_wall=None if wall is None else round(float(wall), 2), clash_with_body_mm3=round(clash, 2), min_gap_to_body=round(float(mind), 3), top_y=round(float(out.bounds[1, 1]), 1))
        print(nm, report[f"{kind}_{side}"], flush=True)
json.dump(report, open(f"{d}/{V}_wings_report.json", "w"), indent=1)
