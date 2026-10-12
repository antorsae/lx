"""Search the T duct of a one-piece UM + T above part: from the T lumen (re-opened from ABOVE_FILL_FROM) into the
PT2522 window (free end) or the AMT lead channel (side entry), largest min bend radius, no loops, sealed skin
(collar allowed behind the rear plane, like the lumen's own cover), pilots kept 1.2 mm away."""
import sys,os,json,time;sys.path[:0]=['scripts','src','tools']
import numpy as np,trimesh,manifold3d as m3
from scipy.interpolate import BSpline
from scipy.optimize import differential_evolution
from scipy.spatial import cKDTree
from shapely import contains_xy, affinity
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
import lx521_baffle.obiwan.dipole_tweeter as d
SC=os.environ.get('SC','/private/tmp/claude-501/-Users-antor-gh-lx-top-baffle-v2/2d2e67bb-b5f6-4669-8c80-db03dec4a417/scratchpad/')
key=f'{d.DRIVER_KEY}_{d.ORIENTATION}'+os.environ.get('RAISE_TAG','')
sol=trimesh.load(SC+f'above_solid_{key}.stl')
man=m3.Manifold(m3.Mesh(vert_properties=np.asarray(sol.vertices,np.float32),tri_verts=np.asarray(sol.faces,np.uint32)))
PITCH=0.3;lo=np.array([-60,330,-2.0]);hi=np.array([60,520,18.6])
xs=np.arange(lo[0],hi[0],PITCH);ys=np.arange(lo[1],hi[1],PITCH);zs=np.arange(lo[2],hi[2],PITCH)
X,Y=np.meshgrid(xs,ys,indexing='ij');occ=np.zeros((len(xs),len(ys),len(zs)),bool)
for k,z in enumerate(zs):
    acc=np.zeros(X.shape,bool)
    for loop in man.slice(float(z)).to_polygons():
        pg=Polygon(np.asarray(loop))
        if pg.area>1e-6: acc^=contains_xy(pg,X,Y)
    occ[:,:,k]=acc
cx,cy=d.ABOVE_CENTER;floor=d.THICKNESS_MM
for q in d._PIL:
    m_=np.hypot(X-(cx+q.x),Y-(cy+q.y))<d.PILOT_D_MM/2+0.4;kz=(zs>=floor-d.PILOT_DEPTH_MM-0.4)&(zs<=floor+0.5);occ[:,:,kz]&=~m_[:,:,None]
if os.environ.get('ERODE'):
    from scipy.ndimage import binary_erosion
    occ=binary_erosion(occ,iterations=int(os.environ['ERODE']))
print('occupancy',occ.sum(),flush=True)
def solid(P):
    idx=np.floor((P-lo)/PITCH+0.5).astype(int);ok=(idx>=0).all(1)&(idx[:,0]<len(xs))&(idx[:,1]<len(ys))&(idx[:,2]<len(zs))
    out=np.zeros(len(P),bool);i=idx[ok];out[ok]=occ[i[:,0],i[:,1],i[:,2]];return out
def bspline(ctrl,n=300):
    ctrl=np.asarray(ctrl,float);k=3;m=len(ctrl);t=np.r_[[0]*k,np.linspace(0,1,m-k+1),[1]*k];return BSpline(t,ctrl,k)(np.linspace(0,1,n))
def minR(P):
    a,b,c=P[:-2],P[1:-1],P[2:];area2=np.linalg.norm(np.cross(b-a,c-a),axis=1)
    return float((np.linalg.norm(b-a,axis=1)*np.linalg.norm(c-b,axis=1)*np.linalg.norm(a-c,axis=1)/np.maximum(2*area2,1e-12)).min())
def skin(P,r,s=0.8,n=12):
    T=np.gradient(P,axis=0);T/=np.maximum(np.linalg.norm(T,axis=1,keepdims=True),1e-12)
    U=np.cross(T,[0,0,1.]);U/=np.maximum(np.linalg.norm(U,axis=1,keepdims=True),1e-12);V=np.cross(T,U)
    a=np.linspace(0,2*np.pi,n,endpoint=False)
    return (P[:,None,:]+(r+s)*(np.cos(a)[None,:,None]*U[:,None,:]+np.sin(a)[None,:,None]*V[:,None,:])).reshape(-1,3)
L,i_fill=d._lumen()
j=i_fill
while L[j,1]<417.5 and j<len(L)-10: j+=1
cand=np.arange(i_fill+4,j)
r=d.ABOVE_DUCT_D/2;RC=d.TS_COVER_OUTER_R
WIN=affinity.translate(unary_union([d._O]+([d._CH] if d._CH is not None else [])),cx,cy)
WINO=affinity.translate(d._O,cx,cy)
PT=d.DRIVER_KEY=='pt2522_4'
z_front=d.THICKNESS_MM+d.PLATE_T
lx,ly=d._rot(*d.DRIVER.lead_point());lead=np.array([cx+lx,cy+ly])
ch_lo=max(z_front-d.DRIVER.depth_behind_front,d.CORE_REAR_Z+0.8);ch_hi=floor-0.8
up=np.array([0,1.,0])
def curve(v):
    bi=int(np.clip(round(v[0]),0,len(cand)-1));b=int(cand[bi])
    t0=L[min(b+4,len(L)-1)]-L[b-4];t0/=np.linalg.norm(t0)
    if PT: E=np.array([v[7],v[8],v[9]])
    else: E=np.array([lead[0]+v[7],lead[1]+v[8],v[9]])
    th=np.radians(v[10]);te=np.array([np.cos(th),np.sin(th),0])
    P0=L[b];ch=E-P0;C1=P0+0.35*ch+np.array([v[3],v[4],0]);C1[2]=v[5];C2=P0+0.65*ch+np.array([v[11],v[12],0]);C2[2]=v[6]
    ctrl=[P0,P0+v[1]*t0,C1,C2,E-v[2]*te,E]
    return b,np.vstack([L[i_fill:b],bspline(ctrl)])
def score(v,ret=False):
    b,P=curve(v);E=P[-1]
    if PT and not contains_xy(WINO.buffer(-r-0.5),E[0],E[1]): return 9000 if not ret else dict(bad=9999)
    S=skin(P[(b-i_fill):],r)
    keep=cKDTree(L[i_fill:b+6]).query(S)[0]>3.0+0.8+0.3
    keep&=np.linalg.norm(S-E,axis=1)>2*r+1.5
    Pq=P[(b-i_fill):];Lc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(Pq,axis=0),axis=1))];tl=Pq[Lc>Lc[-1]-5.0]
    near_end=cKDTree(tl).query(S)[0]<r+0.8+0.5
    keep&=~(contains_xy(WIN.buffer(0.3),S[:,0],S[:,1])&near_end)        # in the window / channel only where it ends
    S=S[keep]
    ok=solid(S)
    # collar behind the rear plane, outside the window and the MU10 opening
    shallow=P[P[:,2]-RC<d.CORE_REAR_Z+0.05]
    if len(shallow):
        inc=(S[:,2]<d.CORE_REAR_Z+0.3)&(cKDTree(shallow).query(S)[0]<RC+0.3)&~contains_xy(WINO.buffer(0.5),S[:,0],S[:,1])&(np.hypot(S[:,0],S[:,1]-d.UM_CUTOUT[1])>41.5)
        ok|=inc
    bad=int((~ok).sum())
    Q=P[(b-i_fill):];Tg=np.diff(Q,axis=0);Tg=Tg[np.linalg.norm(Tg,axis=1)>1e-9];Tg/=np.linalg.norm(Tg,axis=1,keepdims=True)
    turn=float(np.degrees(np.arccos(np.clip(np.sum(Tg[1:]*Tg[:-1],axis=1),-1,1)).sum()))
    if turn>150: bad+=int(turn-150)+1
    R=minR(P[max(b-i_fill-4,0):]);Lg=float(np.sum(np.linalg.norm(np.diff(P,axis=0),axis=1)))
    collar=float(np.sum(np.linalg.norm(np.diff(shallow,axis=0),axis=1))) if len(shallow)>1 else 0.0
    if ret: return dict(bad=bad,R=R,len=Lg,turn=turn,collar_mm=collar)
    return (1000+bad) if bad else -min(R,60)+0.03*Lg+0.05*collar
wb=WINO.bounds
lim=[(0,len(cand)-1),(1,25),(1,25),(-15,15),(-12,12),(6,16),(6,16)]
lim+=([(wb[0],wb[2]),(wb[1],wb[1]+25),(8,15)] if PT else [(-1.5,1.5),(-1.0,1.0),(ch_lo,ch_hi)])
lim+=[(0,180),(-15,15),(-12,12)]
t0=time.time()
res=differential_evolution(score,lim,popsize=int(os.environ.get('POP',36)),maxiter=int(os.environ.get('IT',450)),tol=1e-7,seed=3,polish=False,updating='deferred',workers=1)
info=score(res.x,True);b,P=curve(res.x)
print('ABOVE',key,info,round(time.time()-t0),'s',flush=True)
json.dump(dict(x=res.x.tolist(),points=np.round(P,4).tolist(),d=2*r,branch_xy=np.round(L[b][:2],3).tolist(),**{k:float(v) for k,v in info.items()}),open(SC+f'above_route_{key}.json','w'))
