"""Pick the best seed per part (no violations, largest min bend) and write it as routes[key]['um']."""
import json, os, glob, math, numpy as np
SC = os.environ['SC']; F = 'src/lx521_baffle/obiwan/dipole_tweeter_routes.json'
routes = json.load(open(F))
def minR(P):
    a, b, c = P[:-2], P[1:-1], P[2:]
    area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    return float((np.linalg.norm(b - a, axis=1) * np.linalg.norm(c - b, axis=1) * np.linalg.norm(a - c, axis=1) / np.maximum(2 * area2, 1e-12)).min())
for drv in ('pt2522_4', 'amt17d2_1'):
    for ori in ('portrait', 'landscape'):
        runs = [json.load(open(f)) for f in sorted(glob.glob(SC + f'guide_{drv}_{ori}_s*.json'))]
        ok = [r for r in runs if r['bad'] == 0 and r['R'] >= {('pt2522_4', 'portrait'): 14.0, ('pt2522_4', 'landscape'): 14.0, ('amt17d2_1', 'portrait'): 10.0, ('amt17d2_1', 'landscape'): 8.5}[(drv, ori)]]
        print(drv, ori, [(round(r['R'], 1), r['bad']) for r in runs])
        if not ok: print('   NO VALID GUIDE'); continue
        r = max(ok, key=lambda r: r['R']); P = np.asarray(r['points'])
        routes[f'{drv}/{ori}']['um'] = dict(points=r['points'], w=r['w'], h=r['h'], min_bend_r_mm=math.floor(minR(P) * 10) / 10,
                                            start_y=round(r['start_y'], 2), length_mm=round(r['len'], 1), turn_deg=round(r['turn'], 1),
                                            cover_arc_mm=round(r['cover_arc'], 2), lm_exit_gap_mm=round(r['gap'], 2),
    **({'round_to_mm': round(r['round_to'], 3), 'flat_from_mm': round(r['flat_from'], 3)} if 'round_to' in r else {}))
        print('   picked R', round(r['R'], 2), 'start y', round(r['start_y'], 1), 'len', round(r['len'], 1), 'turn', round(r['turn'], 1))
prov = routes['_provenance']
add = (' T-below key "um": the hidden UM (MU10) cable guide, starting ON the released UM cable past the LM duct exit'
       ' (same point and tangent), round D8.2 inside a short rear cover (cover_arc_mm) continuing the LM duct cover, easing'
       ' to a flat 11.6 x 5.0 stadium (D8.2 area) by flat_from_mm, otherwise walled 0.8 mm in material, clear of the LM,'
       ' ending radially in the raised MU10 open back; every bend >= R14 (AMT owner exceptions 2026-10-02: upright R10,'
       ' rotated R8.8).')
if 'key "um"' not in prov: routes['_provenance'] = prov + add
json.dump(routes, open(F, 'w'), indent=1)
