"""Free space around each ND25FN-4 Faston tab in V11 (vendor model, installed pose), against the top and that tweeter's cap."""
import sys, json, math, numpy as np, trimesh
d = sys.argv[1]; P = json.load(open(f"{d}/tweeter_placement.json"))
body = trimesh.load(f"{d}/obiwan_v11_top.stl"); body = max(body.split(only_watertight=False), key=lambda p: abs(p.volume))
caps = {"T": f"{d}/obiwan_v11_top_cap_frontT.stl", "t": f"{d}/obiwan_v11_top_cap_reart.stl"}
# tab geometry in the vendor-aligned model frame (x = depth behind the dome apex, measured on the vendor STL)
TABS = {"A (2.8 class, 91 deg)": dict(az=91.3, x0=22.0, x1=31.4, r_in=19.95, r_out=20.95, width=3.0),
        "B (4.8 class, 175 deg)": dict(az=175.4, x0=22.0, x1=29.9, r_in=19.6, r_out=20.6, width=5.4)}
for key in ("T", "t"):
    T4 = np.array(P[key]); cap = trimesh.load(caps[key]); obst = trimesh.util.concatenate([body, cap])
    print(f"== tweeter {key} ({'front, cap on the rear face' if key == 'T' else 'rear, cap on the front face'})")
    for name, g in TABS.items():
        a = math.radians(g["az"]); u = np.array([0, math.cos(a), math.sin(a)]); w = np.array([0, -math.sin(a), math.cos(a)]); ex = np.array([1., 0, 0])
        rm = (g["r_in"] + g["r_out"]) / 2
        tip = ex * g["x1"] + u * rm; mid = ex * (g["x0"] + g["x1"]) / 2 + u * g["r_out"]
        R = T4[:3, :3]; t = T4[:3, 3]
        def W(p): return R @ p + t
        def D(v): return R @ v
        probes = {"outward from the tab face (radial)": (mid + u * 0.05, u),
                  "rearward past the tab tip (axial)": (tip + ex * 0.05, ex),
                  "sideways +": (ex * g["x1"] - ex * 2 + u * rm + w * (g["width"] / 2 + 0.05), w),
                  "sideways -": (ex * g["x1"] - ex * 2 + u * rm - w * (g["width"] / 2 + 0.05), -w)}
        tipw = W(tip)
        print(f"  tab {name}: tip at ({tipw[0]:.1f}, {tipw[1]:.1f}, z {tipw[2]:.1f}); tab runs {g['x1']-g['x0']:.1f} mm, {g['x1']-6.05:.1f} mm behind the flange face")
        for lab, (p, dv) in probes.items():
            loc, ir, _ = obst.ray.intersects_location([W(p)], [D(dv)], multiple_hits=False)
            dist = np.linalg.norm(loc[0] - W(p)) if len(loc) else float("inf")
            print(f"     {lab:36s} {dist:6.1f} mm")
