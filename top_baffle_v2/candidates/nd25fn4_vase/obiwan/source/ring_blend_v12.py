"""V12 (owner): slow, tangent rear transition from the vase down the LM ring to the end of the top piece.

V11 lowered the vase rear right up to the y 278 cut, where the released ring's flat z 6.8 rear begins: a crease/step
along the inner half of the ring on both sides. Here every radial line of the ring (R95..R113.5 about the LM centre)
gets a new rear profile along its arc: a C2 quintic Hermite from 20 mm above the cut (matching the vase's height
and slope there) to y 180 (matching the released ring's height and slope, 7.5 mm clear of the keyed seam at 172.48).
Only material BEHIND the existing rear surface is added (a heightfield solid from the new rear up to 0.4 mm inside the
body), so no duct, bore, seat, magnet or seam can be touched. The fade to "no addition" is C1 (soft ramp, eps 0.5 mm).
usage: ring_blend_v12.py IN_STL OUT_STL [report.json]
"""
import sys, json, math, numpy as np, trimesh, manifold3d as m3
from scipy.ndimage import gaussian_filter1d, maximum_filter

src, dst = sys.argv[1], sys.argv[2]; rep_p = sys.argv[3] if len(sys.argv) > 3 else None
LM = np.array([0.0, 200.981]); Y2, Y_END, LU = 278.0, 180.0, 14.0
R_IN, R_OUT, OVER, EPS, H, BIAS = 95.02, 113.5, 0.4, 0.5, 0.25, 0.15
to_m = lambda q: m3.Manifold(m3.Mesh(vert_properties=np.array(q.vertices, np.float32), tri_verts=np.array(q.faces, np.uint32)))
def to_t(M): mm = M.to_mesh(); return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts), process=False)

mesh = trimesh.load(src); parts = mesh.split(only_watertight=False)
main = max(parts, key=lambda p: abs(p.volume)); voids = [p for p in parts if p is not main and p.volume < 0]
def rear(xy):
    O = np.c_[xy, np.full(len(xy), -80.0)]; D = np.tile([0, 0, 1.0], (len(xy), 1))
    loc, idx, _ = main.ray.intersects_location(O, D, multiple_hits=False)
    z = np.full(len(xy), np.nan); z[idx] = loc[:, 2]; return z

rs = np.arange(R_IN, R_OUT + 1e-9, H); report = {}
adds = []
for sgn, side in ((1, "right"), (-1, "left")):
    phc = np.arcsin((Y2 - LM[1]) / rs); phe = np.arcsin((Y_END - LM[1]) / rs)
    Lr = rs * (phc - phe) + LU                                              # arc length of each radial line's blend, 20 mm above the cut to y 180
    ts = np.linspace(0.0, 1.0, int(math.ceil(Lr.max() / H)) + 1)
    Rg, Tg = np.meshgrid(rs, ts, indexing="ij"); Sg = -LU + Tg * Lr[:, None]
    phi = phc[:, None] - Sg / Rg
    def xy_of(R_, phi_): return np.c_[(sgn * R_ * np.cos(phi_)).ravel(), (LM[1] + R_ * np.sin(phi_)).ravel()]
    Hz = rear(xy_of(np.clip(Rg, 95.3, 113.3), phi)).reshape(Rg.shape)       # sample just inside the bore / outer edges
    for jj in range(Hz.shape[1]):                                          # the few misses right at an edge: radial neighbours
        col = Hz[:, jj]; bad = np.isnan(col)
        if bad.any() and (~bad).any(): Hz[bad, jj] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), col[~bad])
    assert not np.isnan(Hz).any(), f"{side}: rear misses inside the ring band"
    Hs = gaussian_filter1d(gaussian_filter1d(Hz, 2.0 / H, axis=0, mode="nearest"), 8, axis=1, mode="nearest")
    dt = ts[1] - ts[0]; k = 4
    h0, h1 = Hs[:, 0], Hs[:, -1]
    d0 = (Hs[:, k] - Hs[:, 0]) / (k * dt); d1 = (Hs[:, -1] - Hs[:, -1 - k]) / (k * dt)   # slopes per unit t (= per blend length)
    t = Tg
    H0 = 1 - 10 * t**3 + 15 * t**4 - 6 * t**5; H1 = t - 6 * t**3 + 8 * t**4 - 3 * t**5; H5 = 10 * t**3 - 15 * t**4 + 6 * t**5; H4 = -4 * t**3 + 7 * t**4 - 3 * t**5
    G = h0[:, None] * H0 + d0[:, None] * H1 + h1[:, None] * H5 + d1[:, None] * H4      # C2 quintic: heights and slopes matched, zero curvature at both ends
    d = Hz - G
    gain = np.maximum((d + np.sqrt(d * d + EPS * EPS)) / 2 - EPS / 2, 0.0)          # C1 soft ramp: 0 where the target is not below the rear
    taper = np.minimum(1.0, np.minimum(t, 1 - t) / 0.03)                              # exactly 0 at both ends of every line
    gain = gain * taper
    report[side] = dict(max_gain_mm=round(float(gain.max()), 2), area_gain_gt_0p1_mm2=round(float((gain > 0.1).sum() * H * float(np.mean(Lr)) / len(ts)), 0),
                        profile_samples_r_h0_h1=[(round(float(r), 2), round(float(a), 2), round(float(b), 2)) for r, a, b in list(zip(rs, h0, h1))[:: max(1, len(rs) // 8)]])
    XY = xy_of(Rg, phi).reshape(len(rs), len(ts), 2); nr, ns = XY.shape[:2]; idx = np.arange(nr * ns).reshape(nr, ns)
    S5 = lambda u: (lambda v: v * v * v * (v * (v * 6 - 15) + 10))(np.clip(u, 0, 1))
    bot = Hz + BIAS - gain - BIAS * S5(gain / 0.3)                                  # where nothing is added the bottom stays BIAS inside the body: no coincident faces
    V = np.vstack([np.c_[XY.reshape(-1, 2), bot.ravel()], np.c_[XY.reshape(-1, 2), (Hz + OVER).ravel()]]); n = nr * ns   # top only OVER inside the rear skin (>= 0.75 mm): never reaches a lumen or pilot bore
    a, b, c, e = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    Fb = np.r_[np.c_[a, e, c], np.c_[a, c, b]]; Ft = Fb[:, ::-1] + n
    ring = np.r_[idx[:, 0], idx[-1, 1:], idx[::-1, -1][1:], idx[0, ::-1][1:-1]]
    Fs = np.vstack([[[ring[q], ring[(q + 1) % len(ring)] + n, ring[q] + n], [ring[q], ring[(q + 1) % len(ring)], ring[(q + 1) % len(ring)] + n]] for q in range(len(ring))])
    add = trimesh.Trimesh(V, np.vstack([Fb, Ft, Fs]), process=True)
    trimesh.repair.fix_winding(add); trimesh.repair.fix_inversion(add)
    assert add.is_watertight and add.is_winding_consistent and add.volume > 0, f"{side}: add-on not a closed solid"
    adds.append(add)
B = to_m(main)
for a_ in adds:
    A = to_m(a_); assert A.status() == m3.Error.NoError, A.status(); B = B + A
import io
buf = io.BytesIO(); to_t(B).export(buf, file_type="stl"); buf.seek(0)                  # judge the result as every later step reads it: STL, merged vertices
out = trimesh.load_mesh(buf, file_type="stl", process=True)
parts2 = out.split(only_watertight=False); new_main = max(parts2, key=lambda p: abs(p.volume))
extra = [p for p in parts2 if p is not new_main]
print("ring blend: extra shells after the union: %d, |volume| sum %.3f mm3 (dropped)" % (len(extra), sum(abs(p.volume) for p in extra)), flush=True)
# where the add-on's top crosses a vertical step (L22 membrane / collar wall, R ~103) it can undercut the step and close tiny
# wedge pockets; dropping those inward shells fills them (magnet cavities are 65 mm3 and carried separately from the input)
for p in extra:
    if abs(p.volume) > 0.05: print("   shell %.2f mm3 at %s (filled)" % (p.volume, np.round(p.bounds, 1).tolist()), flush=True)
assert all(abs(p.volume) < 20.0 for p in extra), "the union trapped a pocket larger than a step wedge"
added = new_main.volume - main.volume
lo_y = min(float(a_.bounds[0, 1]) for a_ in adds)
print(f"ring blend: added {added:.0f} mm3, lowest y {lo_y:.1f}, max gain R {report['right']['max_gain_mm']} / L {report['left']['max_gain_mm']} mm, watertight {new_main.is_watertight}", flush=True)
trimesh.util.concatenate([new_main] + voids).export(dst)
report.update(added_mm3=round(added, 1), lowest_add_y_mm=round(lo_y, 2), watertight=bool(new_main.is_watertight))
if rep_p: json.dump(report, open(rep_p, "w"), indent=1)
