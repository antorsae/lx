"""T-above taper check: the D6->D4 start of the AMT T duct keeps its skin in material (0.3 mm into the wall)."""
import os, sys
sys.path[:0] = ['scripts', 'src', 'tools', 'vendor/GRS/PT2522-4']
import numpy as np, trimesh
import lx521_baffle.obiwan.dipole_tweeter as d
sfx = '_landscape' if d.LANDSCAPE else ''
m = trimesh.load(f'build/{d.DRIVER.name}/obiwan_{d.DRIVER.name}_um_t_above{sfx}.stl')
_, (pts, r) = d.um_t_above_tools(); pts = np.asarray(pts)
arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]; k = np.flatnonzero(arc <= d.T_TAPER_MM + 1)
P = pts[k]; R = d.t_taper_d(arc[k]) / 2
T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1, keepdims=True)
sd = np.cross(T, [0, 0, 1.]); sd /= np.linalg.norm(sd, axis=1, keepdims=True); up = np.cross(sd, T)
S = np.vstack([P + (R + 0.3)[:, None] * (np.cos(a) * sd + np.sin(a) * up) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)])
inner = np.vstack([P + (R - 0.2)[:, None] * (np.cos(a) * sd + np.sin(a) * up) for a in np.linspace(0, 2 * np.pi, 24, endpoint=False)])
print(f'{d.DRIVER_KEY} {d.ORIENTATION} T-above taper: d {2*R[0]:.1f}->{2*R[-1]:.1f} over {arc[k][-1]:.1f} mm; skin exposed {int((~m.contains(S)).sum())} of {len(S)}; '
      f'void open (not solid inside) {int((~m.contains(inner)).sum())} of {len(inner)}; watertight {m.is_watertight}')
