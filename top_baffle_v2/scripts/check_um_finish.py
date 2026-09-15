"""Measure exported UM meshes: solid seat, closed handoff and usable M2 tie."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--um-stl", type=Path, required=True)
    parser.add_argument("--lm-state", choices=("no_floor_stand", "floor_stand"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gallery-reference", type=Path,
                        help="hash-bound V4 replacement cable paths, in installed mm")
    args = parser.parse_args()
    os.environ["LX_ROUTING_PROFILE"] = "obiwan"
    os.environ["LX_STAND_FOOT"] = "1" if args.lm_state == "floor_stand" else "0"
    import numpy as np
    import trimesh
    import manifold3d as md
    from scipy.spatial import cKDTree
    from lx521_baffle.obiwan import carriers as c, route

    def load(path):
        authority = json.loads(path.with_suffix('.print.json').read_text())
        assert authority['stl_sha256'] == sha(path)
        mesh = trimesh.load_mesh(path, process=True)
        assert mesh.is_watertight and mesh.is_winding_consistent
        assert sum(s.volume > 0 for s in mesh.split(only_watertight=False)) == 1
        mesh.apply_transform(np.linalg.inv(authority['source_to_stl_matrix']))
        return mesh

    def solid(mesh):
        result = md.Manifold(md.Mesh64(np.asarray(mesh.vertices), np.asarray(mesh.faces, dtype=np.uint64)))
        assert result.status() == md.Error.NoError
        return result

    um = load(args.um_stl)
    lm_path = ROOT / 'build' / args.lm_state / 'stl/obiwan_core_1_of_2_lm_carrier.stl'
    lm = load(lm_path)
    overlap = abs((solid(um) ^ solid(lm)).volume())
    assert overlap < .002, ('LM overlap', overlap)

    # Witness the entire former annular cavity, excluding declared passages.
    # These are independently spaced points in the old empty Z band, not
    # queries of the new positive/fill constructor.
    cy = c.UM_CUTOUT[1]
    witnesses = np.array([[r*np.cos(a), cy+r*np.sin(a), z]
                         for r in (42.0, 43.5, 45., 46.5, 48.)
                         for a in np.deg2rad(np.arange(0., 360., 5.))
                         for z in (7.05, 8.5, 10., 11.5, 13.1)])
    cable = route.ts_cable_points(.08)
    old_cable = cable[cable[:,1] < 326.] if args.gallery_reference else cable
    keep = cKDTree(old_cable).query(witnesses)[0] > 3.25
    if args.gallery_reference:
        gallery = json.loads(args.gallery_reference.read_text())
        for name, digest in gallery['source_sha256'].items():
            assert sha(ROOT/name) == digest, ('stale gallery reference', name)
        for path in gallery['paths'].values():
            distance, nearest = cKDTree(path['centers_mm']).query(witnesses)
            keep &= distance > np.asarray(path['radii_mm'])[nearest] + .25
    for x, y in c.UM_PILOT_XY:
        keep &= np.linalg.norm(witnesses[:,:2]-[x,y],axis=1) > c.UM_PILOT_D_MM/2+.12
    keep &= ~((abs(witnesses[:,0]-c.LM_UM_TIE_X)<2.85)
              & (witnesses[:,1]<335) & (abs(witnesses[:,2]-c.LM_UM_TIE_AXIS_Z)<2.85))
    for x in c.T_UM_TIE_X:
        keep &= ~((abs(witnesses[:,0]-x)<1.85)
                  & (witnesses[:,1]>400) & (abs(witnesses[:,2]-c.T_UM_TIE_AXIS_Z)<1.85))
    witnesses = witnesses[keep]
    inside = um.contains(witnesses)
    assert inside.all(), ('unfilled annulus', witnesses[~inside].tolist())

    # A driver-body gauge and a flange gauge must still pass unchanged.
    driver_gauges = []
    for radius, lo, hi in ((40.94, 6.5, 14.24), (49.24, 14.32, 18.28)):
        gauge = md.Manifold.cylinder(hi-lo, radius, radius, 256).translate((0,cy,lo))
        intrude = abs((solid(um)^gauge).volume())
        assert intrude < .002, ('driver clearance', intrude)
        driver_gauges.append(intrude)

    # Head travels from the open D82 bore; four rays land on the ORIGINAL
    # bearing plane. A D3.9 gauge is clear up to that seat.
    bearing_y = []
    for a in np.linspace(0, 2*np.pi, 4, endpoint=False):
        origin = np.array([c.LM_UM_TIE_X+1.8*np.cos(a), 340., c.LM_UM_TIE_AXIS_Z+1.8*np.sin(a)])
        points,_,_ = um.ray.intersects_location([origin], [[0,-1,0]], multiple_hits=True)
        ys = np.sort(points[:,1])[::-1]
        assert len(ys) and abs(ys[0]-c.LM_UM_TIE_SEAT_Y) < .015, ('bearing seat', ys)
        bearing_y.append(float(ys[0]))
    screw = np.array([[c.LM_UM_TIE_X+r*np.cos(a),y,c.LM_UM_TIE_AXIS_Z+r*np.sin(a)]
                      for y in np.linspace(c.LM_UM_TIE_SEAT_Y+.08,340,70)
                      for a in np.linspace(0,2*np.pi,24,endpoint=False) for r in (0.,1.95)])
    assert not um.contains(screw).any(), 'M2 head access blocked'

    # Sample the physical T cable AND its roof around the actual handoff.
    # Both independently generated mating parts must cover every shell
    # witness, apart from their <=0.10-mm mesh/assembly seam neighborhood.
    centers = cable[(cable[:,1]>312.5)&(cable[:,1]<323.5)][::3]
    tangent = np.gradient(centers,axis=0)
    tangent /= np.linalg.norm(tangent,axis=1)[:,None]
    u = np.cross(tangent, [0.,0.,1.]);u /= np.linalg.norm(u,axis=1)[:,None]
    v = np.cross(tangent,u)
    directions = [u*np.cos(a)+v*np.sin(a) for a in np.linspace(0,2*np.pi,48,endpoint=False)]
    shell = np.concatenate([centers+r*d for r in (3.2,3.55) for d in directions])
    covered = um.contains(shell) | lm.contains(shell)
    missing = shell[~covered]
    distances = []
    if len(missing):
        du = trimesh.proximity.closest_point(um,missing)[1]
        dl = trimesh.proximity.closest_point(lm,missing)[1]
        distances = np.maximum(du,dl)
        assert np.max(distances)<.10, ('exposed handoff roof', missing[distances>=.10].tolist())
    gauge = np.concatenate([centers+2.8*d for d in directions])
    assert not (um.contains(gauge) | lm.contains(gauge)).any(), 'D5.6 T cable obstructed'
    report = {'status':'passed','source_sha256':sha(__file__),
              'um_stl':str(args.um_stl),'um_sha256':sha(args.um_stl),
              'LM_state':args.lm_state,'LM_sha256':sha(lm_path),
              'LM_overlap_mm3':overlap,'solid_annulus_witnesses':len(witnesses),
              'driver_gauge_overlap_mm3':driver_gauges,
              'M2_bearing_plane_Y_mm':bearing_y,'M2_head_gauge_diameter_mm':3.9,
              'M2_head_gauge_witnesses':len(screw),'T_gauge_diameter_mm':5.6,
              'T_shell_witnesses':len(shell),'seam_witnesses':len(missing),
              'maximum_seam_distance_to_both_parts_mm':float(np.max(distances)) if len(distances) else 0.,
              'gallery_reference_sha256':sha(args.gallery_reference) if args.gallery_reference else None,
              'limits':'Digital fit and sampled wall checks; no physical or acoustic qualification.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
