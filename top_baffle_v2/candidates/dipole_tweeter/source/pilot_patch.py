p = 'dtw/route_opt3.py'; s = open(p).read()
a = "    if os.environ.get('T_XMAX'): okk&=S[:,0]<=float(os.environ['T_XMAX'])\n"
b = a + """    # pilots keep their wall even where the duct opens into the window (the end exemption must not reach them)
    _pd=float(os.environ.get('TPILOT_PAD',0.4))+0.8;_fl2=d.THICKNESS_MM-d.PLATE_T
    for _q in d._PIL:
        _near=(np.hypot(S[:,0]-_q.x,S[:,1]-(_q.y+d.PLATE_CENTER_Y))<d.PILOT_D_MM/2+_pd)&(S[:,2]>=_fl2-d.FRAME_PILOT_DEPTH_MM-_pd)
        okk&=~_near
"""
if '_pd=float' not in s:
    assert s.count(a) == 1; s = s.replace(a, b)
    # these checks must apply to every skin point, so move the exemption filter after them: count exempted pilots too
    s = s.replace("    S=S[keep];okk=solid(S)\n", "    S_all=S;S=S[keep];okk=solid(S)\n")
    open(p, 'w').write(s)
print('patched' if '_pd=float' in s else 'not patched')
