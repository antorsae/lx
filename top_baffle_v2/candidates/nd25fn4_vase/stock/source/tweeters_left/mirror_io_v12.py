"""Mirror helpers for the V12 left/right pair (x -> -x in the Obi-Wan design frame).
usage: mirror_io_v12.py stl IN OUT            mirror a mesh (trimesh keeps outward winding; buried cavities stay inward)
       mirror_io_v12.py tweeter IN OUT        mirror the vendor ND25FN-4 model in its own local z (azimuths negate)
       mirror_io_v12.py json IN OUT           mirror every design-frame point / polyline / angle in a V12 build json"""
import sys, json, math, numpy as np, trimesh
mode, src, dst = sys.argv[1:4]
if mode in ("stl", "tweeter"):
    m = trimesh.load(src); m.apply_transform(np.diag([-1.0, 1, 1, 1]) if mode == "stl" else np.diag([1.0, 1, -1, 1])); m.export(dst)
else:
    C = json.load(open(src)); out = {}
    for k, v in C.items():
        a = np.asarray(v, dtype=object) if isinstance(v, list) else None
        if k in ("a_kT", "a_kt"): out[k] = math.pi - v                                   # angle of a direction: mirrored
        elif k in ("T_rim", "t_rim"): out[k] = dict(v, l=v["r"], r=v["l"])
        elif isinstance(v, list) and len(v) and isinstance(v[0], list): out[k] = [[-p[0], *p[1:]] for p in v]
        elif isinstance(v, list) and len(v) in (2, 3) and all(isinstance(x, (int, float)) for x in v): out[k] = [-v[0], *v[1:]]
        else: out[k] = v
    out["hand"] = "mirror"; json.dump(out, open(dst, "w"), indent=1)
print("mirrored", mode, src, "->", dst)
