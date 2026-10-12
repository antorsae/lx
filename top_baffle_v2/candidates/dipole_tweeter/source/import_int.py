"""Best seed of an integrated-duct search (int_<key>_<DUCT><TAG>_s*.json) -> routes[key][DUCT] in LX_DIPOLE_T_ROUTES."""
import os, sys, glob, json, math
os.environ['LX_DIPOLE_T_INTEGRATED'] = '1'; sys.path[:0] = ['src', 'scripts', 'tools', 'vendor/GRS/PT2522-4']
import numpy as np
import lx521_baffle.obiwan.lm_um_t as L
d = L.d
SC = os.environ['SC']; DUCT = os.environ['DUCT']; TAG = os.environ.get('RAISE_TAG', '')
key = f'{d.DRIVER_KEY}_{d.ORIENTATION}'
runs = [json.load(open(f)) for f in sorted(glob.glob(SC + f'int_{key}_{DUCT}{TAG}_s*.json'))]
print(key, DUCT, [(r['bad'], round(r['R'], 1), round(r.get('cover_arc', 0), 1)) for r in runs])
ok = [r for r in runs if r['bad'] == 0]
if not ok: sys.exit(f'{key} {DUCT}: no valid route')
r = max(ok, key=lambda r: r['R'])
REL = L.released_route(DUCT); J = int(np.argmin(np.linalg.norm(REL - np.asarray(r['points'][0]), axis=1)))
f = L.ROUTES_FILE
out = json.loads(f.read_text()) if f.exists() else {}
out.setdefault('_provenance', 'Integrated LM + T + UM ducts (lm_um_t.py): searched clamped cubic B-splines (int_opt).')
out.setdefault(f'{d.DRIVER_KEY}/{d.ORIENTATION}', {})[DUCT] = dict(points=r['points'], section_knots=r['section_knots'], d=r['d'], w=r['w'], h=r['h'],
    min_bend_r_mm=math.floor(min(r['R'], d.min_bend_radius(L.resample(np.vstack([REL[J - 8:J], np.asarray(r['points'])])))) * 10) / 10, cover_arc_mm=round(r['cover_arc'], 2), start_y=round(r['start_y'], 2),
    length_mm=round(r['len'], 1), turn_deg=round(r['turn'], 1))
f.write_text(json.dumps(out, indent=1))
print(key, DUCT, 'picked R', round(r['R'], 2), 'cover', round(r['cover_arc'], 1), 'start y', round(r['start_y'], 1))
