import re
p = 'dtw/route_opt3.py'; s = open(p).read()
if 'TDIAG' not in s:
    a = "print('T',d.DRIVER_KEY,d.ORIENTATION,t_score(rt.x,True)"
    i = s.index(a); j = s.index('\n', i)
    s = s[:j + 1] + """if os.environ.get('TDIAG'):
    b_,P_=t_curve(rt.x);Sd=skin(P_[(b_-i_fill):],rT);ok_=solid(Sd)
    from scipy.spatial import cKDTree as _KK
    near_l=_KK(L[i_fill-4:b_+6]).query(Sd)[0]<=3.0+0.8+0.3
    Pq=P_[(b_-i_fill):];Lc_=np.r_[0,np.cumsum(np.linalg.norm(np.diff(Pq,axis=0),axis=1))]
    own_=np.tile(np.arange(len(Pq)),len(Sd)//len(Pq))
    badd=~ok_&~near_l
    print('T end',np.round(Pq[-1],1),'branch',np.round(Pq[0],1))
    for i_ in np.unique(own_[badd])[::2]: print('  arc %.1f centre %s bad %d e.g. %s'%(Lc_[i_],np.round(Pq[i_],1),int((badd&(own_==i_)).sum()),np.round(Sd[badd&(own_==i_)][0],1)))
    sys.exit()
""" + s[j + 1:]
    open(p, 'w').write(s)
print('patched')
