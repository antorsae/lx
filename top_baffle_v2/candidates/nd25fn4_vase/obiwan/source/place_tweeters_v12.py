"""ND25FN-4 placement for a V12 top: flange front on the land, tab midpoint clocked to each cable mouth (as assemble_v11_drivers).
With LX_HAND=mirror the caller works in the x-mirrored frame and supplies the vendor model mirrored in its local z
(nd25fn4_vendor_rimcentred.stl in DIR): the tab azimuths then read negated, so the clocking target is -133.65 deg.
usage: place_tweeters_v12.py DIR VARIANT   (reads VARIANT_build.json; writes tweeter_placement.json)"""
import os, sys, json, math, numpy as np, trimesh
d, V = sys.argv[1], sys.argv[2]; C = json.load(open(f"{d}/{V}_build.json")); MIR = os.environ.get("LX_HAND", "std") == "mirror"
drv = trimesh.load(f"{d}/nd25fn4_vendor_rimcentred.stl"); v = drv.vertices; r = np.hypot(v[:, 1], v[:, 2])
XF = float(np.percentile(v[(r > 15.9) & (r < 21.5) & (v[:, 0] < 12), 0], 0.5))
TERM_AZ = (-89.7, -177.6) if MIR else (89.7, 177.6); blk = math.radians(-133.65 if MIR else 133.65)
def place(c, a_key, land_z, front):
    ex = np.array([0, 0, -1.0 if front else 1.0]); best = None
    for phi in np.radians(np.arange(0, 360, 0.5)):
        ey = np.array([math.cos(phi), math.sin(phi), 0]); ez = np.cross(ex, ey); M = np.c_[ex, ey, ez]
        b = M @ np.array([0, math.cos(blk), math.sin(blk)]); err = abs((math.atan2(b[1], b[0]) - a_key + math.pi) % (2 * math.pi) - math.pi)
        if best is None or err < best[0]: best = (err, M)
    T4 = np.eye(4); T4[:3, :3] = best[1]; T4[:3, 3] = np.array([c[0], c[1], land_z]) - best[1] @ np.array([XF, 0, 0]); return T4
P = dict(T=place(C["T"], C["a_kT"], C["LAND_T"], True).tolist(), t=place(C["t"], C["a_kt"], C["LAND_t"], False).tolist(), XF=XF, TERM_AZ=TERM_AZ, hand="mirror" if MIR else "std")
json.dump(P, open(f"{d}/tweeter_placement.json", "w")); print("placement: XF %.2f, tab midpoint clocked to the mouths, hand %s" % (XF, P["hand"]))
