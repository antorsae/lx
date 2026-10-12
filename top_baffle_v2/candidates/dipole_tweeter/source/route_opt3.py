"""Optimise smooth (B-spline) lead routes for a UM + T below part against its pre-lead solid."""
import sys,os,json,time;sys.path[:0]=['scripts','src','tools']
import numpy as np,trimesh,manifold3d as m3
from scipy.interpolate import BSpline
from scipy.optimize import differential_evolution
from shapely import contains_xy
import lx521_baffle.obiwan.dipole_tweeter as d
SC=os.environ.get('SC','/private/tmp/claude-501/-Users-antor-gh-lx-top-baffle-v2/2d2e67bb-b5f6-4669-8c80-db03dec4a417/scratchpad/')
key=f'{d.DRIVER_KEY}_{d.ORIENTATION}'+os.environ.get('RAISE_TAG','')
sol=trimesh.load(SC+f'solid_{key}.stl')
man=m3.Manifold(m3.Mesh(vert_properties=np.asarray(sol.vertices,np.float32),tri_verts=np.asarray(sol.faces,np.uint32)))
PITCH=0.3
lo=np.array([-60,305,0.0]);hi=np.array([60,470,18.6])
xs=np.arange(lo[0],hi[0],PITCH);ys=np.arange(lo[1],hi[1],PITCH);zs=np.arange(lo[2],hi[2],PITCH)
X,Y=np.meshgrid(xs,ys,indexing='ij')
occ=np.zeros((len(xs),len(ys),len(zs)),bool)
from shapely.geometry import Polygon
from shapely.ops import unary_union
for k,z in enumerate(zs):
    cs=man.slice(float(z))
    polys=cs.to_polygons()
    if not polys: continue
    # even-odd fill
    acc=np.zeros(X.shape,bool)
    for loop in polys:
        pg=Polygon(np.asarray(loop))
        if pg.area<1e-6: continue
        acc^=contains_xy(pg,X,Y)
    occ[:,:,k]=acc
# pilots: 0.4 mm bigger than drilled, so the 0.8 mm skin rule leaves the 1.2 mm pilot wall
_fl=d.THICKNESS_MM-d.PLATE_T
for _q in d._PIL:
    _px,_py=_q.x,_q.y+d.PLATE_CENTER_Y
    _pad=float(os.environ.get('TPILOT_PAD',0.4));_m=(np.hypot(X-_px,Y-_py)<d.PILOT_D_MM/2+_pad)
    _kz=(zs>=_fl-d.FRAME_PILOT_DEPTH_MM-_pad)&(zs<=_fl+0.5)
    occ[:,:,_kz]&=~_m[:,:,None]
if os.environ.get('ERODE'):
    from scipy.ndimage import binary_erosion
    occ=binary_erosion(occ,iterations=int(os.environ['ERODE']))
print('occupancy built',occ.sum(),flush=True)
def solid(P):
    idx=np.floor((P-lo)/PITCH+0.5).astype(int)
    ok=(idx>=0).all(1)&(idx[:,0]<len(xs))&(idx[:,1]<len(ys))&(idx[:,2]<len(zs))
    out=np.zeros(len(P),bool);i=idx[ok];out[ok]=occ[i[:,0],i[:,1],i[:,2]];return out
def bspline(ctrl,n=260):
    ctrl=np.asarray(ctrl,float);k=3;m=len(ctrl)
    t=np.r_[[0]*k,np.linspace(0,1,m-k+1),[1]*k]
    return BSpline(t,ctrl,k)(np.linspace(0,1,n))
def minR(P):
    a,b,c=P[:-2],P[1:-1],P[2:]
    area2=np.linalg.norm(np.cross(b-a,c-a),axis=1)
    R=np.linalg.norm(b-a,axis=1)*np.linalg.norm(c-b,axis=1)*np.linalg.norm(a-c,axis=1)/np.maximum(2*area2,1e-12)
    return float(R.min())
def skin(P,r,s=0.8,n=12):
    if getattr(d,'T_FLAT',None) and abs(r-d.T_BRANCH_D/2)<1e-9:      # flat (stadium) T duct
        T=np.gradient(P,axis=0);T/=np.maximum(np.linalg.norm(T,axis=1,keepdims=True),1e-12)
        sd=np.cross(T,[0,0,1.]);sd/=np.maximum(np.linalg.norm(sd,axis=1,keepdims=True),1e-12);up=np.cross(sd,T)
        h=d.T_FLAT/2+s;off=(d.T_FLAT_W-d.T_FLAT)/2;pts=[]
        for k in np.linspace(-off,off,5): pts+=[P+k*sd+h*up,P+k*sd-h*up]
        for sg in (-1,1):
            for t in np.linspace(-np.pi/2,np.pi/2,7): pts.append(P+sg*(off+h*np.cos(t))*sd+h*np.sin(t)*up)
        return np.vstack(pts)
    T=np.gradient(P,axis=0);T/=np.linalg.norm(T,axis=1,keepdims=True)
    U=np.cross(T,[0,0,1.]);bad=np.linalg.norm(U,axis=1)<1e-6;U[bad]=np.cross(T[bad],[1,0,0.]);U/=np.linalg.norm(U,axis=1,keepdims=True);V=np.cross(T,U)
    ang=np.linspace(0,2*np.pi,n,endpoint=False)
    S=P[:,None,:]+(r+s)*(np.cos(ang)[None,:,None]*U[:,None,:]+np.sin(ang)[None,:,None]*V[:,None,:])
    return S.reshape(-1,3)
EOF_MARK=None
from lx521_baffle.obiwan.route import ts_cable_points
L=np.asarray(ts_cable_points(0.5))
i_fill=int(np.argmax((L[:,1]>=d.UM_T_FILL_FROM_Y)&(np.abs(L[:,0])<30)))
j=i_fill
while L[j,1]<=d.SPLIT_Y+1.0: j+=1
cand=np.arange(i_fill+6,j)
print('T lumen fill from',np.round(L[i_fill],2),'branch candidates',len(cand),flush=True)
c=d.PLATE_CENTER_Y
rT=d.T_BRANCH_D/2
zT=d.THICKNESS_MM-d.PLATE_T-d.FRAME_SKIN_MM-rT
lx,ly=d._rot(*d.DRIVER.lead_point());lead=np.array([lx,c+ly])
ends={('pt2522_4','portrait'):(lead[0],lead[1]-2.0),('pt2522_4','landscape'):(31.8,c-26.0),
      ('amt17d2_1','portrait'):tuple(lead),('amt17d2_1','landscape'):tuple(lead)}
TEND=np.array([*ends[(d.DRIVER_KEY,d.ORIENTATION)],zT])
up=np.array([0,1.,0])
FREE_END=d.DRIVER_KEY=='pt2522_4'
CH_END=bool(os.environ.get('CH_END')) and not FREE_END
TURN_MAX=float(os.environ.get('TURN_MAX','150'))
from shapely import affinity as _af
WIN=_af.translate(d._O,0,c)
CHAN=_af.translate(d._CH,0,c) if d._CH is not None else None
def t_ctrl(v):
    bi=int(np.clip(round(v[0]),0,len(cand)-1));b=int(cand[bi])
    t0=L[b+4]-L[b-4];t0/=np.linalg.norm(t0)
    a,bb=v[1],v[2];C1=v[3:6];C2=v[6:9]
    if FREE_END:
        E=np.array([v[9],v[10],zT]);th=np.radians(v[11]);te=np.array([np.cos(th),np.sin(th),0])
    elif CH_END:
        E=np.array([TEND[0]+v[9],TEND[1],v[10]]);th=np.radians(v[11]);te=np.array([np.cos(th),np.sin(th),0])
    else:
        E=TEND;te=up
    return b,[L[b],L[b]+a*t0,C1,C2,E-bb*te,E]
def t_curve(v):
    b,ctrl=t_ctrl(v)
    return b,np.vstack([L[i_fill:b],bspline(ctrl)])
def t_score(v,ret=False):
    b,P=t_curve(v);S=skin(P[(b-i_fill):],rT)
    from scipy.spatial import cKDTree
    keep=cKDTree(L[i_fill-4:b+6]).query(S)[0]>3.0+0.8+0.3          # inside the released lumen near the branch
    E=t_ctrl(v)[1][-1]
    keep&=np.linalg.norm(S-E,axis=1)>2*rT+1.5                    # the end opens into the window / channel
    # inside the window (or lead channel) only over the last 5 mm, where the duct ends
    from shapely import contains_xy as _cx
    Pq=P[(b-i_fill):];Lc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(Pq,axis=0),axis=1))];tl=Pq[Lc>Lc[-1]-5.0]
    from scipy.spatial import cKDTree as _K
    near_end=_K(tl).query(S)[0]<rT+0.8+0.5
    OPEN=WIN if CHAN is None else WIN.union(CHAN)
    keep&=~(_cx(OPEN.buffer(0.3),S[:,0],S[:,1])&near_end)
    if FREE_END and not _cx(WIN.buffer(-rT-0.5),E[0],E[1]): return 5000 if not ret else dict(bad=9999,R=0,len=0)
    S_all=S;S=S[keep];okk=solid(S)
    if os.environ.get('T_XMAX'): okk&=S[:,0]<=float(os.environ['T_XMAX'])
    # pilots keep their wall even where the duct opens into the window (the end exemption must not reach them)
    # skin points sit 0.8 outside the duct: 1.2 wall + 0.8 skin + TPILOT_PAD-0.4 margin from the pilot bore
    _pd=1.2+0.8+float(os.environ.get('TPILOT_PAD',0.4))-0.4;_fl2=d.THICKNESS_MM-d.PLATE_T;_pilot_bad=0
    for _q in d._PIL:
        _near=(np.hypot(S_all[:,0]-_q.x,S_all[:,1]-(_q.y+d.PLATE_CENTER_Y))<d.PILOT_D_MM/2+_pd)&(S_all[:,2]>=_fl2-d.FRAME_PILOT_DEPTH_MM-_pd)
        _pilot_bad+=int(_near.sum())
    # rear cover (the released lumen cover's section) wherever the part is too thin: allowed behind the frame rear,
    # outside the window / channel and the MU10 opening; its length is penalised
    from scipy.spatial import cKDTree as _K2
    Pd=P[(b-i_fill):]
    cov=(not os.environ.get('NO_COVER'))&(S[:,2]<d.FRAME_REAR_Z+0.5)&(_K2(Pd).query(S)[0]<rT+0.8+0.3)&~_cx(OPEN.buffer(0.5),S[:,0],S[:,1])&(np.hypot(S[:,0]-d.MU10_CENTER[0],S[:,1]-d.MU10_CENTER[1])>41.5)
    collar=int((~okk&cov).sum())
    bad=int((~(okk|cov)).sum())+_pilot_bad
    R=minR(P[max(b-i_fill-4,0):]);Lg=np.sum(np.linalg.norm(np.diff(P,axis=0),axis=1))+0.05*collar
    Q=P[max(b-i_fill,0):];Tg=np.diff(Q,axis=0);Tg=Tg[np.linalg.norm(Tg,axis=1)>1e-9];Tg/=np.linalg.norm(Tg,axis=1,keepdims=True)
    turn=float(np.degrees(np.arccos(np.clip(np.sum(Tg[1:]*Tg[:-1],axis=1),-1,1)).sum()))
    if turn>TURN_MAX: bad+=int(turn-TURN_MAX)+1
    if ret: return dict(bad=bad,R=R,len=Lg,turn=turn)
    return (1000+bad) if bad else -min(R,60)+float(os.environ.get('LEN_W',0.02))*Lg
lim=[(0,len(cand)-1),(2,30),(2,30),(-55,55),(300,c+20),(7,17),(-55,55),(300,c+40),(7,17)]
if FREE_END:
    wb=WIN.bounds;lim+=[(wb[0],wb[2]),(wb[1],wb[1]+20),(0,180)]
elif CH_END:
    lim+=[(-1.5,1.5),(float(os.environ.get('CH_ZMIN',10.0)),15.0),(20,160)]
if os.environ.get('T_CONFINE'):
    lim[3:9]=[(-5,40),(300,c-20),(7,17),(-5,40),(300,c-15),(7,17)]
import os
t0=time.time()
prev=SC+f'routes_{key}.json'
prev3=SC+f'routes3_{key}.json'
if os.path.exists(prev3): prev=prev3
class _R: pass
if os.environ.get('SKIP_T') and os.path.exists(prev):
    rt=_R();rt.x=np.array(json.load(open(prev))['t']['x'])
else:
  t0=time.time()
  rt=differential_evolution(t_score,lim,popsize=int(os.environ.get('POP',32)),maxiter=int(os.environ.get('IT',400)),tol=1e-7,seed=int(os.environ.get('SEED',1)),polish=False,updating='deferred',workers=1)
print('T',d.DRIVER_KEY,d.ORIENTATION,t_score(rt.x,True),np.round(rt.x,3).tolist(),round(time.time()-t0),'s',flush=True)
if os.environ.get('TDIAG'):
    b_,P_=t_curve(rt.x);Sd=skin(P_[(b_-i_fill):],rT);ok_=solid(Sd)
    from scipy.spatial import cKDTree as _KK
    near_l=_KK(L[i_fill-4:b_+6]).query(Sd)[0]<=3.0+0.8+0.3
    Pq=P_[(b_-i_fill):];Lc_=np.r_[0,np.cumsum(np.linalg.norm(np.diff(Pq,axis=0),axis=1))]
    own_=np.repeat(np.arange(len(Pq)),len(Sd)//len(Pq))
    badd=~ok_&~near_l
    print('T end',np.round(Pq[-1],1),'branch',np.round(Pq[0],1))
    for i_ in np.unique(own_[badd])[::2]: print('  arc %.1f centre %s bad %d e.g. %s'%(Lc_[i_],np.round(Pq[i_],1),int((badd&(own_==i_)).sum()),np.round(Sd[badd&(own_==i_)][0],1)))
    sys.exit()
bT,PT=t_curve(rt.x)
_,PTs=t_curve(rt.x)
json.dump(dict(t=dict(x=rt.x.tolist(),points=np.round(PTs,4).tolist(),d=2*rT,**{k:float(v) for k,v in t_score(rt.x,True).items()})),open(SC+f'routes3_{key}.json','w'),indent=1)
