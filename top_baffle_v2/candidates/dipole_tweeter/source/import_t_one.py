"""Write the best T-below T route of one part (seeds RAISE_TAG _x1.._x3) into the routes file, keeping its 'um'."""
import json, os, sys, glob, numpy as np
sys.path[:0] = ['src', 'scripts', 'tools']
from lx521_baffle.obiwan.route import ts_cable_points
import lx521_baffle.obiwan.dipole_tweeter as d
SC = os.environ['SC']; key = f'{d.DRIVER_KEY}_{d.ORIENTATION}'
L = np.asarray(ts_cable_points(0.5)); i0 = int(np.argmax((L[:, 1] >= d.UM_T_FILL_FROM_Y) & (np.abs(L[:, 0]) < 30)))
runs = [json.load(open(f))['t'] for f in sorted(glob.glob(SC + f'routes3_{key}' + os.environ.get('SEARCH_GLOB', '_x*') + '.json'))]
print(key, [(r['bad'], round(r['R'], 1)) for r in runs])
ok = [r for r in runs if r['bad'] == 0]
if not ok: sys.exit('no valid T route')
r = max(ok, key=lambda r: r['R']); P = np.asarray(r['points']); k = 0
while k < len(P) and np.linalg.norm(P[k] - L[i0 + k]) < 1e-3: k += 1
bf = max(k - 4, 0)
f = str(d.UM_T_ROUTES_FILE); out = json.load(open(f))
out[f'{d.DRIVER_KEY}/{d.ORIENTATION}']['t'] = dict(points=P.round(4).tolist(), d=r['d'], min_bend_r_mm=round(d.min_bend_radius(P[bf:]), 1), bend_from=bf,
    length_mm=round(float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))), 1), total_turn_deg=round(r['turn'], 1), bad=r['bad'],
    x_max_mm=float(os.environ.get('T_XMAX', 'nan')))
json.dump(out, open(f, 'w'), indent=1)
print('picked R', out[f'{d.DRIVER_KEY}/{d.ORIENTATION}']['t']['min_bend_r_mm'], 'x max', round(float(P[:, 0].max()), 1))
