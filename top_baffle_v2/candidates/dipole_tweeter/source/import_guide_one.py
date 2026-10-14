"""Write the best UM guide of one part/study (guide_<key><RAISE_TAG>_s*.json) into the routes file in use, keeping 't'."""
import json, os, sys, glob, math, numpy as np
sys.path[:0] = ['src', 'scripts', 'tools']
import lx521_baffle.obiwan.dipole_tweeter as d
SC = os.environ['SC']; key = f'{d.DRIVER_KEY}_{d.ORIENTATION}' + os.environ.get('RAISE_TAG', '')
runs = [json.load(open(f)) for f in sorted(glob.glob(SC + f'guide_{key}_s*.json'))]
print(key, [(r['bad'], round(r['R'], 1), round(r.get('cover_arc', 0), 1)) for r in runs])
ok = [r for r in runs if r['bad'] == 0 and r['R'] >= d.UM_GUIDE_MIN_BEND_R]
if not ok: sys.exit(f'{key}: no valid UM guide')
r = max(ok, key=lambda r: r['R']); P = np.asarray(r['points'])
f = str(d.UM_T_ROUTES_FILE); out = json.load(open(f))
out.setdefault(f'{d.DRIVER_KEY}/{d.ORIENTATION}', {})['um'] = dict(points=r['points'], w=r['w'], h=r['h'],
    min_bend_r_mm=math.floor(d.min_bend_radius(P) * 10) / 10, start_y=round(r['start_y'], 2), length_mm=round(r['len'], 1),
    turn_deg=round(r['turn'], 1), cover_arc_mm=round(r['cover_arc'], 2), lm_exit_gap_mm=round(r['gap'], 2),
    **({'round_to_mm': round(r['round_to'], 3), 'flat_from_mm': round(r['flat_from'], 3)} if 'round_to' in r else {}))
json.dump(out, open(f, 'w'), indent=1)
print(key, 'picked R', round(r['R'], 2), 'cover', round(r['cover_arc'], 1), 'gap', round(r['gap'], 1))
