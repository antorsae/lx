"""Discover insertion pauses against the actual sliced angled magnet slots."""
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
from shapely.geometry import MultiPoint, LineString

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import v4_model as model
from gcode_analysis import parse_gcode


def magnet_geometry(stl, authority, offset):
    matrix=np.asarray(authority['source_to_stl_matrix']).copy()
    matrix[:3,3]+=np.asarray(offset)
    printed=trimesh.load_mesh(stl,process=True)
    installed=printed.copy();installed.apply_transform(np.linalg.inv(np.asarray(authority['source_to_stl_matrix'])))
    specs=[]
    for cavity in installed.split(only_watertight=False):
        if cavity.volume>=0:continue
        if cavity.center_mass[1]>340:
            site=min(model.magnet_sites(),key=lambda s:np.linalg.norm(s['contact']-cavity.center_mass))
            n,t,u,contact=[site[k] for k in ['normal','tangent','up','contact']]
            p=cavity.vertices-contact
            owner='body' if stl.name.startswith('01_') else 'wing'
            axial=p@n
            a=max(axial)-1.5 if owner=='body' else min(axial)+1.5
            center=contact+a*n+((p@t).min()+(p@t).max())/2*t+((p@u).max()-3.1)*u
            diam,depth=6.,3.
            name=f"{owner}_{site['angle_deg']:g}deg"
        else:
            # The lower wing pocket retains the regular LM D5x2 interface.
            catalog=json.loads((ROOT/'review/captive_magnet_release_catalog.json').read_text())
            source=authority.get('source_stl','')
            rows=[a for a in catalog['artifacts'] if a.get('stl','').endswith(Path(source).name)]
            if not rows:
                rows=[a for a in catalog['artifacts'] if 'wing' in a['part']]
            sites=[s for row in rows for s in row['sites']]
            s=min(sites,key=lambda s:np.linalg.norm(np.array(s['cavity_center_xyz_mm'])-cavity.center_mass))
            center=np.array(s['seated_magnet_center_xyz_mm']);n=np.array(s['installed_marked_pole_axis_xyz'])
            u=np.array([0.,0.,1.]);t=np.cross(u,n);diam,depth=5.,2.;name='preserved_LM'
        basis=np.eye(4);basis[:3,:3]=np.column_stack([t,u,n]);basis[:3,3]=center
        magnet=trimesh.creation.cylinder(radius=diam/2,height=depth,sections=128,transform=basis)
        magnet.apply_transform(matrix)
        # Bring the disc down along the slot's actual inclined loading axis.
        approach=matrix[:3,:3]@(-u);approach/=np.linalg.norm(approach)
        assert approach[2]>.5
        sweep=trimesh.convex.convex_hull(np.r_[magnet.vertices,magnet.vertices+approach*45])
        specs.append(dict(name=name,diameter_mm=diam,depth_mm=depth,
                          center_bed_mm=trimesh.transform_points([center],matrix)[0].tolist(),
                          pole_axis_bed=(matrix[:3,:3]@n).tolist(),
                          approach_axis_bed=approach.tolist(),
                          seated_top_z_mm=float(magnet.bounds[1,2]),
                          seated_bottom_z_mm=float(magnet.bounds[0,2]),
                          _magnet=magnet,_sweep=sweep))
    return specs


def section_polygon(mesh,z):
    lines=trimesh.intersections.mesh_plane(mesh,[0,0,1],[0,0,z])
    if len(lines)==0:return None
    return MultiPoint(lines.reshape(-1,3)[:,:2]).convex_hull


def discover(stl,authority,offset,gcode):
    specs=magnet_geometry(stl,authority,offset)
    roi=[]
    for spec in specs:
        lo,hi=spec['_sweep'].bounds
        roi.append((lo[0]-1,lo[1]-1,hi[0]+1,hi[1]+1))
    parsed=parse_gcode(gcode,retain_regions=roi)
    result=[]
    for spec in specs:
        first=None;collisions=[];last_open=None;layer_records=[]
        for layer in parsed.layers:
            if layer.z<spec['seated_bottom_z_mm']-.3:continue
            if layer.z>spec['seated_top_z_mm']+10:break
            # Evaluate the full bead height, not only its centre plane.
            height=layer.layer_height or .16
            polygons=[section_polygon(spec['_sweep'],layer.z-height/2)]
            polygons=[p for p in polygons if p is not None]
            if not polygons:continue
            hits=[]
            for seg in layer.segments:
                if seg.feature.lower() in {'custom','undefined','prime tower'}:continue
                line=LineString([(seg.x0,seg.y0),(seg.x1,seg.y1)])
                width=seg.line_width or .62
                penetration=max(width/2-line.distance(poly) for poly in polygons)
                # The bead rectangle overestimates rounded bead corners; the
                # 0.1-mm radial seating allowance is also below one layer.
                # Preserve every >0.06-mm intrusion as an obstruction witness.
                if penetration>.06:
                    hits.append(dict(line=seg.line_number,feature=seg.feature,intrusion_mm=float(penetration)))
            layer_records.append(dict(z=layer.z,intrusion_mm=max([p['intrusion_mm'] for p in hits],default=0),count=len(hits)))
            if hits and first is None:
                first=layer;collisions=hits
            if first is None:last_open=layer
        record={k:v for k,v in spec.items() if not k.startswith('_')}
        record.update(first_obstructing_layer_z_mm=first.z if first else None,
                      last_open_layer_z_mm=last_open.z if last_open else None,
                      collision_witnesses=collisions[:5],layer_records=layer_records)
        record['pause_before_z_mm']=first.z if first else None
        if spec['name']=='preserved_LM':
            # Keep the already-qualified start-of-roof pause, before the
            # following layer first physically obstructs a D5 disc.
            assert first and first.z>=5.96
            record['pause_before_z_mm']=5.96
            record['pause_selection']='retained LM start-of-roof layer; loading path checked through prior layers'
        record['magnet_below_resume_plane']=bool(first and spec['seated_top_z_mm']<record['pause_before_z_mm']-.04)
        record['opening_clear_through_prior_layers']=last_open is not None
        result.append(record)
    return result


def main():
    key=sys.argv[1] if len(sys.argv)>1 else 'body'
    work=ROOT/'review/nd25fn4_print';prep=json.loads((work/'preparation.json').read_text())[key]
    path=ROOT/prep['path'];stl=ROOT/prep['sources'][0]['path']
    authority=json.loads(stl.with_suffix('.print.json').read_text())
    rows=discover(stl,authority,prep['offset'],path.parent/'plate_1.gcode')
    (path.parent/'magnet_discovery.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
