import os, sys
os.environ.update(LX_DIPOLE_T_INTEGRATED='1', LX_DIPOLE_T_PAIR='1', LX_ROUTING_PROFILE='obiwan')
sys.argv = [sys.argv[0]]; sys.path[:0] = ['scripts', 'src', 'tools', 'vendor/GRS/PT2522-4']
import numpy as np, trimesh, manifold3d as m3
import build_dipole_tweeter as b
import lx521_baffle.obiwan.lm_um_t as L
lm, piece, cutters, fits, ducts = L._pair_parts(b.released_installed('h2c_obiwan_lm_keyed_top'), b.released_um_installed(), b.tessellate, 'final')
tubes = {w: L._pair_duct_tube(w) for w in ducts}
def tm(m):
    o = m.to_mesh(); return trimesh.Trimesh(np.asarray(o.vert_properties)[:, :3], np.asarray(o.tri_verts), process=True)
m = lm
for c in cutters: m = m - c
print('after cutters cavities', [(round(-x.volume, 2), np.round(x.bounds, 1).tolist()) for x in tm(m).split(only_watertight=False) if x.volume < -0.5])
for w, t in tubes.items():
    m = m - t
    print('after', w, 'tube cavities', [(round(-x.volume, 2), np.round(x.bounds, 1).tolist()) for x in tm(m).split(only_watertight=False) if x.volume < -0.5])
print('before cutters (lm alone) cavities', [(round(-x.volume, 2), np.round(x.bounds, 1).tolist()) for x in tm(lm).split(only_watertight=False) if x.volume < -0.5])
