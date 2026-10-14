"""Merge the final T-below routes (T duct + UM guide per part) into the repo routes file."""
import json
F = 'src/lx521_baffle/obiwan/dipole_tweeter_routes.json'
out = json.load(open(F))
src = {'pt2522_4/portrait': 'dtw/routes_p16.json', 'pt2522_4/landscape': 'dtw/routes_x26.json',
       'amt17d2_1/portrait': 'dtw/routes_au.json', 'amt17d2_1/landscape': 'dtw/routes_ar4.json'}
for key, f in src.items():
    r = json.load(open(f))[key]
    out[key] = {'t': r['t'], 'um': r['um']}
    print(key, 'T R', r['t']['min_bend_r_mm'], 'guide R', r['um']['min_bend_r_mm'], 'cover', r['um']['cover_arc_mm'])
prov = out['_provenance']
add = (' T-below (2026-10-03): key "um" is the hidden UM (MU10) cable guide. It starts ON the released UM cable at'
       ' the LM UM duct exit (same point and tangent), climbs inside a short rear cover (cover_arc_mm) continuing the'
       ' LM duct cover (round D8.2 there when round_to_mm/flat_from_mm are given), then runs as a flat 11.6 x 5.0'
       ' stadium (D8.2 area) hidden beside the window and ends radially in the raised MU10 open back. It was searched'
       ' first against the part with the released T-lumen stretch kept free; the T duct was searched after it. Owner'
       ' raises: PT2522 upright +16, rotated +24, AMT rotated +4 mm. Owner exceptions: AMT upright guide R10 (route'
       ' R11.7 incl. the join), AMT rotated T duct R10 (it passes outboard of the driver mounting pilot).')
if 'T-below (2026-10-03)' not in prov: out['_provenance'] = prov + add
json.dump(out, open(F, 'w'), indent=1)
