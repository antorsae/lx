"""Key ring behind each tweeter land: the inner outline follows the tweeter (max radius per degree over everything that
passes through the ring zone on insertion) + CLR, so the four rim lugs get notches and the tweeter is centred and clocked.
usage: key_ring_v11.py DIR  (reads obiwan_v11_top_unkeyed.stl, tweeter_placement.json, nd25fn4_vendor_rimcentred.stl; writes obiwan_v11_top.stl)"""
import sys, json, math, numpy as np, trimesh, manifold3d as m3
d = sys.argv[1]; P = json.load(open(f"{d}/tweeter_placement.json")); XF = P["XF"]
CLR, X_END, R_WALL = 0.25, 12.0, 23.7                 # radial clearance; ring runs from the land (x=XF) to x=X_END; 0.5 mm into the R23.2 wall
to_m = lambda q: m3.Manifold(m3.Mesh(vert_properties=np.asarray(q.vertices, np.float32), tri_verts=np.asarray(q.faces, np.uint32)))
tw = trimesh.load(f"{d}/nd25fn4_vendor_rimcentred.stl"); v = tw.vertices
k = v[:, 0] <= X_END + 0.3; r = np.hypot(v[k, 1], v[k, 2]); az = np.degrees(np.arctan2(v[k, 2], v[k, 1]))
bins = np.arange(-180, 181, 1.0); prof = np.zeros(360)
for i in range(360):
    s = (az >= bins[i] - 1.0) & (az < bins[i + 1] + 1.0)          # +-1 deg window so notches are not undersized between vertices
    prof[i] = r[s].max() if s.any() else 0
prof = np.maximum(prof, 17.0) + CLR
print(f"inner outline: r {prof.min():.2f}..{prof.max():.2f}; lug notch bins (r > 20): {(prof > 20.0).sum()} deg")
ang = np.radians(bins[:-1] + 0.5)
inner = np.c_[prof * np.cos(ang), prof * np.sin(ang)]
outer = np.c_[R_WALL * np.cos(ang), R_WALL * np.sin(ang)]
cs = m3.CrossSection([outer.tolist(), inner[::-1].tolist()])
body = trimesh.load(f"{d}/obiwan_v11_top_unkeyed.stl"); _parts = body.split(only_watertight=False); body = max(_parts, key=lambda q: abs(q.volume)); B = to_m(body)
voids = [q for q in _parts if q.volume < 0]                                                   # buried LM wing magnet cavities (inward shells): carried through unchanged
for key in ("T", "t"):
    T4 = np.array(P[key])
    ring = m3.Manifold.extrude(cs, X_END - (XF - 0.3))                 # local: extrude along +z_local, then map local z -> model x
    ring_t = trimesh.Trimesh(*(lambda mm: (np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts)))(ring.to_mesh()), process=False)
    Vl = ring_t.vertices.copy(); model = np.c_[Vl[:, 2] + (XF - 0.3), Vl[:, 0], Vl[:, 1]]        # (x, y, z) in the tweeter model frame
    ring_t.vertices = (T4[:3, :3] @ model.T).T + T4[:3, 3]
    if np.linalg.det(T4[:3, :3]) < 0: ring_t.invert()
    if ring_t.volume < 0: ring_t.invert()
    R = to_m(ring_t); add = (R - B).volume(); B = B + R
    print(f"tweeter {key}: key ring {ring_t.volume:.0f} mm3, {add:.0f} mm3 new material")
# Anti-rotation posts behind the two lugs that carry no terminal: only the correct clocking puts the terminal insulator
# blocks (x 12..22, behind the lugs at the tab azimuths) where there is no post, so the 4-fold lug key becomes one-way.
import os
TAB_AZ = (89.7, 177.6)
lug_az = []
if not os.environ.get('LX_POSTS'): lug_az = None                                                        # lug centres from the outline (runs of r > 20)
if lug_az is None: print('anti-rotation posts: off (the insulator blocks do not reach beyond the lugs, so no one-way post is possible)')
else:
  hi = prof > 20.0; i0 = np.argmax(~hi); rolled = np.roll(hi, -i0); ang_r = np.roll(bins[:-1] + 0.5, -i0); j = 0
  while j < 360:
      if rolled[j]:
          k2 = j
          while k2 + 1 < 360 and rolled[k2 + 1]: k2 += 1
          lug_az.append(float(np.degrees(np.arctan2(np.sin(np.radians(ang_r[j:k2 + 1])).mean(), np.cos(np.radians(ang_r[j:k2 + 1])).mean())))); j = k2 + 1
      else: j += 1
  free = [a for a in lug_az if min(abs((a - t + 180) % 360 - 180) for t in TAB_AZ) > 30]
  blk = (v[:, 0] > 12.5) & (v[:, 0] < 21.5); rb = np.hypot(v[blk, 1], v[blk, 2]); ab = np.degrees(np.arctan2(v[blk, 2], v[blk, 1]))
  blk_r = min(rb[np.abs((ab - t + 180) % 360 - 180) < 8].max() for t in TAB_AZ)
  lug_r = prof.max() - CLR
  R_POST = lug_r + 0.30
  other = rb[np.all([np.abs((ab - t + 180) % 360 - 180) > 25 for t in TAB_AZ], axis=0)].max()
  print(f"lug centres {np.round(lug_az, 1)}; posts at {np.round(free, 1)}; lug r {lug_r:.2f}, block r {blk_r:.2f}, other body r at x 12.5-21.5 {other:.2f}; post inner r {R_POST:.2f}")
  assert blk_r > R_POST + 0.3 and other < R_POST - 0.3, "posts would not discriminate / would touch the correctly clocked body"
  for key in ("T", "t"):
      T4 = np.array(P[key])
      for a in free:
          aa = np.radians(np.linspace(a - 12, a + 12, 25)); poly = [[R_WALL * math.cos(q), R_WALL * math.sin(q)] for q in aa] + [[R_POST * math.cos(q), R_POST * math.sin(q)] for q in aa[::-1]]
          post = m3.Manifold.extrude(m3.CrossSection([poly]), 21.5 - 12.5)
          pt = trimesh.Trimesh(np.asarray(post.to_mesh().vert_properties)[:, :3], np.asarray(post.to_mesh().tri_verts), process=False)
          Vl = pt.vertices.copy(); model = np.c_[Vl[:, 2] + 12.5, Vl[:, 0], Vl[:, 1]]; pt.vertices = (T4[:3, :3] @ model.T).T + T4[:3, 3]
          if pt.volume < 0: pt.invert()
          B = B + to_m(pt)
      print(f"tweeter {key}: 2 anti-rotation posts added")
mm = B.to_mesh(); out = trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3], np.asarray(mm.tri_verts))
print("keyed top watertight", out.is_watertight, "volume %.1f cm3" % (out.volume / 1000), "| magnet cavities kept", len(voids), "inside", all(out.contains(q.vertices).all() for q in voids))
trimesh.util.concatenate([out] + voids).export(f"{d}/obiwan_v11_top.stl")
