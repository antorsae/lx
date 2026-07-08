import os, sys, math
sys.path.insert(0, ".")

def run(mode):
    os.environ["LX_STAND_FOOT"] = mode
    for m in list(sys.modules):
        if "top_baffle" in m: del sys.modules[m]
    from top_baffle_nd25fw4_cables import CABLE_D, DUCT_Z, route_points
    from top_baffle_nd25fw4_v0_split import pieces_v0
    from top_baffle_nd25fw4_v1_split import pieces_v1
    from top_baffle_nd25fw4_v1l_split import pieces_v1l
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_State

    def classifier(sol):
        c = BRepClass3d_SolidClassifier(sol.wrapped)
        def inside(x,y,z):
            c.Perform(gp_Pnt(x,y,z),1e-6); return c.State()==TopAbs_State.TopAbs_IN
        return inside

    def densify(pts, step=2.0):
        out=[]
        for a,b in zip(pts,pts[1:]):
            d=math.dist(a[:2],b[:2]); n=max(1,int(d/step))
            for k in range(n):
                out.append(tuple(a[i]+(b[i]-a[i])*k/n for i in range(3)))
        out.append(pts[-1]); return out

    def wall_below(inside,x,y,z):   # material depth below z (floor wall): scan down
        s=0.0
        while s<8.0:
            if not inside(x,y,z-s): return s
            s+=0.1
        return 8.0
    def wall_above(inside,x,y,z):
        s=0.0
        while s<8.0:
            if not inside(x,y,z+s): return s
            s+=0.1
        return 8.0

    variants = {"V0":pieces_v0(), "V1":pieces_v1(), "V1L":pieces_v1l()}
    SEAM_A, SEAM_B, SEAM_C = 120.0, 315.95, -5.6
    results={}
    for vn, pieces in variants.items():
        cls = {k:classifier(v) for k,v in pieces.items()}
        worst=(9,None); breaches=[]
        for name in ("lm","um","ts","t1f","t2f"):
            r = CABLE_D.get(name,3.8)/2.0
            pts = densify([p for p in route_points(name)])
            for i in range(1,len(pts)):
                x,y,z = pts[i]
                px,py,_ = pts[i-1]
                dx,dy = x-px,y-py; L=math.hypot(dx,dy)
                if L<1e-6: continue
                nx,ny = -dy/L, dx/L
                # pick piece by band, with 7mm seam margins; skip entry plumbing y<64
                if y < 64: continue
                if abs(y-SEAM_A)<7 or abs(y-SEAM_B)<7: continue
                if y < SEAM_A: continue  # bottom entry region, by-design
                elif y < SEAM_B:
                    pc = "piece_mid_right" if x > SEAM_C else "piece_mid_left"
                else:
                    pc = "piece_top_b2"
                if pc not in cls: continue
                ins = cls[pc]
                for o in (-r,-0.7*r,0.0,0.7*r,r):
                    ox,oy = x+o*nx, y+o*ny
                    drop = math.sqrt(max(r*r-o*o,0.0))
                    wf = wall_below(ins, ox,oy, z-drop)
                    wr = wall_above(ins, ox,oy, z+drop)
                    w = min(wf,wr)
                    if w < worst[0]: worst=(w,(name,round(x,1),round(y,1),round(o,1),pc,'floor' if wf<wr else 'roof'))
                    if w < 0.9:
                        breaches.append((name,round(ox,1),round(oy,1),round(z,1),round(o,1),round(w,2),pc))
        results[vn]=(worst,breaches)
    return results

for mode in ("0","1"):
    print(f"\n===== LX_STAND_FOOT={mode} =====")
    res = run(mode)
    for vn,(worst,breaches) in res.items():
        print(f"\n{vn}: worst realized wall = {worst[0]:.2f} mm at {worst[1]}")
        if breaches:
            print(f"  {len(breaches)} sub-0.9mm points:")
            for b in breaches[:12]: print("   ",b)
        else:
            print("  no sub-0.9mm walls")
