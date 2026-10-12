import os,sys,json
src=open("dtw/above_opt.py").read()
src=src[:src.index("wb=WINO.bounds")]
exec(src)
r_=json.load(open(SC+f"above_route_{key}.json"))
x=np.array(r_["x"]);b,P=curve(x);E=P[-1]
S=skin(P[(b-i_fill):],r)
keep=cKDTree(L[i_fill:b+6]).query(S)[0]>3.0+0.8+0.3
keep&=np.linalg.norm(S-E,axis=1)>2*r+1.5
S2=S[keep];ok=solid(S2)
shallow=P[P[:,2]-RC<d.CORE_REAR_Z+0.05]
bad=S2[~ok]
print(key,"bad",len(bad))
for q in bad[::max(1,len(bad)//10)]:
    print(np.round(q,1).tolist(),"R from UM",round(float(np.hypot(q[0],q[1]-d.UM_CUTOUT[1])),1),"in window",bool(contains_xy(WINO,q[0],q[1])))
print("end",np.round(E,1).tolist(),"window bottom",round(WINO.bounds[1],1),"branch",np.round(L[b],1).tolist())
