"""Strict final check of one built part (VARIANT = um_t | um_t_above)."""
import sys,os;sys.path[:0]=['scripts','src','tools']
import numpy as np,trimesh
from shapely import contains_xy, affinity
from shapely.geometry import LineString
from scipy.spatial import cKDTree
import lx521_baffle.obiwan.dipole_tweeter as d
v=os.environ['VARIANT'];sfx='_landscape' if d.LANDSCAPE else ''
m=trimesh.load(f'build/{d.DRIVER.name}/obiwan_{d.DRIVER.name}_{v}{sfx}.stl')
tag=f'{d.DRIVER_KEY} {d.ORIENTATION} {v}'
if v=='um_t': _,tubes=d.um_t_tools();cx,cy=0.0,d.PLATE_CENTER_Y;mu=d.MU10_CENTER
else: _,t=d.um_t_above_tools();tubes=[t];cx,cy=d.ABOVE_CENTER;mu=(d.UM_CUTOUT[0],d.UM_CUTOUT[1])
win=affinity.translate(d._O,cx,cy);chan=affinity.translate(d._CH,cx,cy) if d._CH is not None else None
def ring(P,r,s):
    P=np.asarray(P,float);P=P[np.r_[True,np.linalg.norm(np.diff(P,axis=0),axis=1)>1e-6]]
    T=np.gradient(P,axis=0);T/=np.linalg.norm(T,axis=1,keepdims=True);U=np.cross(T,[0,0,1.]);U/=np.linalg.norm(U,axis=1,keepdims=True);V=np.cross(T,U)
    a=np.linspace(0,2*np.pi,24,endpoint=False)
    return P,(P[:,None,:]+(r+s)*(np.cos(a)[None,:,None]*U[:,None,:]+np.sin(a)[None,:,None]*V[:,None,:])).reshape(-1,3)
voids=[(np.asarray(p,float),r) for p,r in tubes]
extra=[]
if v=='um_t':
    gp,_,gw,gh=d.um_guide();voids.append((gp,float(gh.min())/2))
for P,r in [(np.asarray(p,float),r) for p,r in tubes]:
    if v=='um_t' and getattr(d,'T_FLAT',None):                     # flat T duct: its stadium outline
        P=P[np.r_[True,np.linalg.norm(np.diff(P,axis=0),axis=1)>1e-6]]
        T=np.gradient(P,axis=0);T/=np.linalg.norm(T,axis=1,keepdims=True);sd=np.cross(T,[0,0,1.]);sd/=np.linalg.norm(sd,axis=1,keepdims=True);up=np.cross(sd,T)
        h=d.T_FLAT/2+0.05;off=(d.T_FLAT_W-d.T_FLAT)/2;pp=[]
        for k in np.linspace(-off,off,7): pp+=[P+k*sd+h*up,P+k*sd-h*up]
        for sg in (-1,1):
            for t in np.linspace(-np.pi/2,np.pi/2,9): pp.append(P+sg*(off+h*np.cos(t))*sd+h*np.sin(t)*up)
        S=np.vstack(pp);r=d.T_FLAT_W/2
    else:
        P,S=ring(P,r,0.05)
    void=np.zeros(len(S),bool)
    for Q,rq in voids: void|=cKDTree(Q).query(S)[0]<rq-0.02
    Lc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))];near_end=cKDTree(P[Lc>Lc[-1]-5.0]).query(S)[0]<r+0.55
    in_open=contains_xy(win,S[:,0],S[:,1])|(contains_xy(chan,S[:,0],S[:,1]) if chan is not None else False)
    ok=void|(in_open&near_end)|(np.hypot(S[:,0]-mu[0],S[:,1]-mu[1])<d.UM_CUTOUT[2]/2)|(S[:,1]<318.0)
    out=(~ok)&~m.contains(S)
    lbl=(f'{d.T_FLAT_W:.1f}x{d.T_FLAT:.1f}' if (v=='um_t' and getattr(d,'T_FLAT',None)) else f'D{2*r}')
    print(f'{tag} T duct {lbl}: min bend R{d.min_bend_radius(P[20:]):.1f}, exposed {int(out.sum())} of {len(S)}',np.round(S[out][:3],1).tolist(),flush=True)
if v=='um_t':
    # the guide's real (stadium) section: wall samples on its outline, offset 0.05 outward
    P=np.asarray(gp,float);T=np.gradient(P,axis=0);T/=np.linalg.norm(T,axis=1,keepdims=True)
    sd=np.cross(T,[0,0,1.]);sd/=np.linalg.norm(sd,axis=1,keepdims=True);up=np.cross(sd,T)
    GOFF=float(os.environ.get('GUIDE_OFF',0.3))                    # samples this far into the wall (0.8 nominal)
    h=(gh/2+GOFF)[:,None];off=((gw-gh)/2)[:,None]
    a=np.linspace(0,2*np.pi,24,endpoint=False);pts=[]
    for k in np.linspace(-1,1,7):                                # flat faces
        pts+= [P+k*off*sd+h*up,P+k*off*sd-h*up]
    for sgn in (-1,1):                                            # round ends
        for t in np.linspace(-np.pi/2,np.pi/2,9): pts.append(P+sgn*(off+h*np.cos(t))*sd+h*np.sin(t)*up)
    S=np.vstack(pts)
    own=np.tile(np.arange(len(P)),len(pts))
    Lc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(P,axis=0),axis=1))]
    in_mu=np.hypot(S[:,0]-mu[0],S[:,1]-mu[1])<d.UM_CUTOUT[2]/2
    outside=~m.contains(S)
    rear_ok=(Lc[own]<3.0)                  # only the guide's open start (its cut runs 3 mm back along the cable) is open
    out=outside&~in_mu&~rear_ok
    # thinnest wall along the guide (outside the open start and the MU10 opening): material depth out from the void
    from trimesh.proximity import ProximityQuery
    wall_s=S[~in_mu&~rear_ok][::7]; inside=m.contains(wall_s)
    thin=float(ProximityQuery(m).signed_distance(wall_s[inside]).min()) if inside.any() else 0.0
    print(f'{tag} UM guide wall: samples {GOFF} mm into the wall; thinnest remaining wall beyond them {thin:.2f} mm')
    import build_dipole_tweeter as bdt
    lmm=bdt.released_installed('h2c_obiwan_core_lm_carrier_no_floor_stand')
    in_lm=int(lmm.contains(S[S[:,1]<326.0]).sum())
    import manifold3d as m3
    MM=lambda t: m3.Manifold(m3.Mesh(vert_properties=np.asarray(t.vertices,np.float32),tri_verts=np.asarray(t.faces,np.uint32)))
    print(f'{tag} UM guide vs LM: wall samples inside the LM {in_lm}; part/LM clash {(MM(m)^MM(lmm)).volume():.2f} mm3; rear cover over the first {d.um_guide()[1]:.1f} mm of the guide; section D{gw[0]:.1f} at the start, {gw[-1]:.1f}x{gh[-1]:.1f} in the frame')
    from lx521_baffle.obiwan.route import route_cable_points
    rel=np.asarray(route_cable_points(0.5));j=int(np.argmin(np.linalg.norm(rel-P[0],axis=1)))
    tr=(rel[j+2]-rel[j-2]);tr/=np.linalg.norm(tr);tg=(P[3]-P[0]);tg/=np.linalg.norm(tg)
    print(f'{tag} UM guide {d.UM_GUIDE_W:.1f}x{d.UM_GUIDE_H:.1f}: starts on released cable (off {np.linalg.norm(rel[j]-P[0]):.3f} mm, tangent {np.degrees(np.arccos(np.clip(tr@tg,-1,1))):.1f} deg) at y {P[0,1]:.1f}, min bend R{d.min_bend_radius(np.vstack([rel[j-8:j],P])):.1f}, exposed {int(out.sum())} of {len(S)}',np.round(S[out][:3],1).tolist())
    # tie: counterbore axis open from y0 to the window, wall around it closed
    yw=d.tie_window_y();ax=np.c_[np.full(20,d.TIE['x']),np.linspace(d.TIE['y0']+0.3,yw-0.3,20),np.full(20,d.TIE['z'])]
    _,Sw=ring(ax,d.TIE['cbore_r'],0.6)
    print(f'{tag} M2 tie: counterbore open to window {int((~m.contains(ax)).sum())}/20, sleeve wall missing {int((~m.contains(Sw[Sw[:,1]<yw-0.5])).sum())}')
for q in d._PIL:
    px,py=cx+q.x,cy+q.y;best=99
    floor=(d.THICKNESS_MM-d.PLATE_T) if v=='um_t' else d.THICKNESS_MM
    depth=min(d.FRAME_PILOT_DEPTH_MM,floor-d._pilot_rear_z(px,py)-d.FRAME_SKIN_MM) if v=='um_t' else d.PILOT_DEPTH_MM
    bot=np.array([[px,py,floor-depth-0.05-k*0.05] for k in range(40)]);flo=int(m.contains(bot).sum())*0.05
    for z in np.linspace(floor-depth+0.3,floor-0.5,6):
        for ang in np.linspace(0,2*np.pi,72,endpoint=False):
            rr=np.arange(d.PILOT_D_MM/2+0.05,7,0.05);Q=np.c_[px+rr*np.cos(ang),py+rr*np.sin(ang),np.full(len(rr),z)]
            o=~m.contains(Q)
            if o.any(): best=min(best,rr[o.argmax()]-d.PILOT_D_MM/2)
    print(f'{tag} pilot ({px:.1f},{py:.1f}) depth {depth:.2f} wall {best:.2f} floor {flo:.2f}')
print(f'{tag} front: max z {m.bounds[1][2]:.2f} (front plane {d.THICKNESS_MM}); vertices proud of it {int((m.vertices[:,2]>d.THICKNESS_MM+0.01).sum())}')
print(f'{tag} bodies {len([b for b in m.split(only_watertight=False) if abs(b.volume)>1])} watertight {m.is_watertight} vol {m.volume/1e3:.1f} cm3')
