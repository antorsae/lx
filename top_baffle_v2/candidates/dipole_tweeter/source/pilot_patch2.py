p = 'dtw/route_opt3.py'; s = open(p).read()
a = """    _pd=float(os.environ.get('TPILOT_PAD',0.4))+0.8;_fl2=d.THICKNESS_MM-d.PLATE_T
    for _q in d._PIL:
        _near=(np.hypot(S[:,0]-_q.x,S[:,1]-(_q.y+d.PLATE_CENTER_Y))<d.PILOT_D_MM/2+_pd)&(S[:,2]>=_fl2-d.FRAME_PILOT_DEPTH_MM-_pd)
        okk&=~_near
"""
b = """    # skin points sit 0.8 outside the duct: 1.2 wall + 0.8 skin + TPILOT_PAD-0.4 margin from the pilot bore
    _pd=1.2+0.8+float(os.environ.get('TPILOT_PAD',0.4))-0.4;_fl2=d.THICKNESS_MM-d.PLATE_T;_pilot_bad=0
    for _q in d._PIL:
        _near=(np.hypot(S_all[:,0]-_q.x,S_all[:,1]-(_q.y+d.PLATE_CENTER_Y))<d.PILOT_D_MM/2+_pd)&(S_all[:,2]>=_fl2-d.FRAME_PILOT_DEPTH_MM-_pd)
        _pilot_bad+=int(_near.sum())
"""
assert s.count(a) == 1; s = s.replace(a, b)
a2 = "    bad=int((~(okk|cov)).sum())\n"
assert s.count(a2) == 1; s = s.replace(a2, "    bad=int((~(okk|cov)).sum())+_pilot_bad\n")
open(p, 'w').write(s); print('ok')
