"""Write T-below route files (RAISE_TAG _h) from the search output into the repo routes file."""
import json,os,sys,numpy as np
sys.path[:0]=['src','scripts','tools']
from lx521_baffle.obiwan.route import ts_cable_points
import lx521_baffle.obiwan.dipole_tweeter as d
SC=os.environ['SC'];tag=os.environ.get('RAISE_TAG','')
L=np.asarray(ts_cable_points(0.5));i0=int(np.argmax((L[:,1]>=d.UM_T_FILL_FROM_Y)&(np.abs(L[:,0])<30)))
f='src/lx521_baffle/obiwan/dipole_tweeter_routes.json';out=json.load(open(f))
for drv in ('pt2522_4','amt17d2_1'):
    for o in ('portrait','landscape'):
        r=json.load(open(SC+f'routes3_{drv}_{o}{tag}.json'))['t'];P=np.asarray(r['points']);k=0
        while k<len(P) and np.linalg.norm(P[k]-L[i0+k])<1e-3: k+=1
        bf=max(k-4,0)
        out[f'{drv}/{o}']={'t':dict(points=P.round(4).tolist(),d=r['d'],min_bend_r_mm=round(d.min_bend_radius(P[bf:]),1),bend_from=bf,length_mm=round(float(np.sum(np.linalg.norm(np.diff(P,axis=0),axis=1))),1),total_turn_deg=round(r['turn'],1),bad=r['bad'])}
        print(drv,o,'bad',r['bad'],'R',out[f'{drv}/{o}']['t']['min_bend_r_mm'])
open(f,'w').write(json.dumps(out)+'\n')
