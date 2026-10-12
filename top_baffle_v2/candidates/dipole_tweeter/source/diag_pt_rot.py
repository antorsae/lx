import os,sys,json
src=open("dtw/route_opt3.py").read()
src=src[:src.index("lim=[(0,len(cand)-1)")]
exec(src)
print("window bounds",np.round(WIN.bounds,1),"plate centre",round(c,1),"zT",round(zT,2),"T",d.T_FLAT_W,d.T_FLAT)
# simple candidates: branch at several lumen points, straight-ish into the lower right of the window
res=[]
for bi in (10,20,30,38):
    b=int(cand[bi]);t0=L[b+4]-L[b-4];t0/=np.linalg.norm(t0)
    for ex,ey in ((22,WIN.bounds[1]+4),(28,WIN.bounds[1]+4),(16,WIN.bounds[1]+4)):
        for ez in (11.5,12.5,13.5):
            E=np.array([ex,ey,ez])
            ctrl=[L[b],L[b]+6*t0,L[b]+0.5*(E-L[b])+np.array([0,0,0]),E-6*np.array([0,1.,0]),E]
            P=np.vstack([L[i_fill:b],bspline(ctrl)])
            S=skin(P[(b-i_fill):],rT)
            from scipy.spatial import cKDTree as K
            keep=K(L[i_fill-4:b+6]).query(S)[0]>3.0+0.8+0.3
            keep&=np.linalg.norm(S-E,axis=1)>2*rT+1.5
            S=S[keep];bad=S[~solid(S)]
            res.append((len(bad),np.round(L[b],1).tolist(),E.tolist(),np.round(bad[::max(1,len(bad)//6)],1).tolist()))
            if False:
                print("branch",np.round(L[b],1),"end",E,"bad",len(bad),"R",round(minR(P[(b-i_fill):]),1), np.round(bad[::max(1,len(bad)//4)],1).tolist())
res.sort(key=lambda t:t[0])
for r_ in res[:5]: print(r_)
