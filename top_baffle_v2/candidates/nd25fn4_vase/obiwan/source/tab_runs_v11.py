"""Cable run from each tweeter's entry-duct mouth to the solder point (tab hole) of each tab."""
import sys, json, math, numpy as np
d = sys.argv[1]; C = json.load(open(f"{d}/obiwan_v11_top_build.json")); P = json.load(open(f"{d}/tweeter_placement.json"))
R_CAPB, ZB_T, TH = C["R_CAPB"], C["ZB_T"], C["TH"]
T, t = np.array(C["T"]), np.array(C["t"])
dKT = np.array([math.cos(C["a_kT"]), math.sin(C["a_kT"])])
u_tt = (t - T) / np.linalg.norm(t - T)
mouths = {"T": np.r_[T + (R_CAPB - 2.0) * dKT, -15.5], "t": np.r_[t - (R_CAPB - 1.5) * u_tt, TH - 5.0]}
TABS = {"A (narrow)": (89.7, 22.0, 31.43, 19.83), "B (wide)": (177.6, 22.0, 29.86, 19.34)}
def world(T4, p): T4 = np.array(T4); return T4[:3, :3] @ np.asarray(p, float) + T4[:3, 3]
for k in ("T", "t"):
    c = np.array(C[k]); m = mouths[k]; am = math.degrees(math.atan2(m[1] - c[1], m[0] - c[0]))
    print(f"tweeter {k}: entry mouth at ({m[0]:.1f}, {m[1]:.1f}, z {m[2]:.1f}), {am:.1f} deg around the tweeter axis")
    runs = {}
    for name, (az, x0, x1, rt) in TABS.items():
        xh = x0 + 0.55 * (x1 - x0)                                   # solder at the tab hole, ~mid-tab
        p = world(P[k], [xh, (rt + 0.5) * math.cos(math.radians(az)), (rt + 0.5) * math.sin(math.radians(az))])
        ap = math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))
        runs[name] = np.linalg.norm(p - m)
        print(f"   tab {name}: solder point ({p[0]:.1f}, {p[1]:.1f}, z {p[2]:.1f}) at {ap:.1f} deg; straight run from the mouth {runs[name]:.1f} mm; angular offset from the mouth {((ap - am + 180) % 360) - 180:+.1f} deg")
