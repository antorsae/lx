"""Matching upper wings for the compact, rounded UM/V4 carrier.

Reshape the inner edge with a 0.35 mm geometric gap, pair two UM shoulder
pockets per wing, and retain the LM pocket and original split joints.
"""
import json
import hashlib
import inspect
from pathlib import Path

import numpy as np
import trimesh

import v4_model as model
from rebuild import sha, verify_package
from validate import solid, restored_print_parts

HERE = Path(__file__).resolve().parent
CLEARANCE = model.WING_GAP
# The flush UM foot reaches below the former raised apron's clipping height.
# End the clearance below that foot while preserving the entire LM magnet
# (its highest point is Y306.930) and the lower split interface unchanged.
PROTECTED_BELOW_Y = 308.0
CUTTER_BOTTOM_Y = 308.4
CUTTER_TOP_Y = 460.0
CUTTER_REAR_X = 1.5


def clearance_field(x,y,z):
    g = model.retained
    # Use the filled exterior: a wing may not protrude into an acoustic mouth,
    # service cavity or wire outlet simply because that feature is empty.
    envelope = model.envelope_field(x,y,z)
    # Follow the actual inclined wall through depth. The wider front lip
    # reveals the UM naturally; a cylindrical projection would leave an
    # unnecessary wedge-shaped air gap and separate the magnet pairs.
    return np.maximum.reduce(np.broadcast_arrays(
        envelope, z+model.INSTALLED_Y_OFFSET-CUTTER_TOP_Y,
        CUTTER_BOTTOM_Y-(z+model.INSTALLED_Y_OFFSET), CUTTER_REAR_X-x,x-g.P.overall_depth/2))


def offset_envelope(mesh):
    """Euclidean mesh offset; the retained implicit field is not a true SDF.

    Using a raw field-value allowance pinches the clearance at the steep rear
    shoulder. VTK evaluates signed closest-surface distance in compiled code.
    """
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    points = vtk.vtkPoints(); points.SetData(numpy_to_vtk(np.asarray(mesh.vertices),deep=True))
    cells = vtk.vtkCellArray()
    cells.SetCells(len(mesh.faces),numpy_to_vtkIdTypeArray(
        np.c_[np.full(len(mesh.faces),3),mesh.faces].ravel().astype(np.int64),deep=True))
    surface=vtk.vtkPolyData();surface.SetPoints(points);surface.SetPolys(cells)
    distance=vtk.vtkImplicitPolyDataDistance();distance.SetInput(surface)
    lo=mesh.bounds[0]-.8; hi=mesh.bounds[1]+.8
    samples=np.ceil((hi-lo)/.35).astype(int)+1
    sample=vtk.vtkSampleFunction();sample.SetImplicitFunction(distance)
    sample.SetModelBounds(*np.c_[lo,hi].ravel());sample.SetSampleDimensions(*samples)
    sample.ComputeNormalsOff();sample.SetOutputScalarTypeToFloat()
    # Persist the costly exact distance grid independently of contouring,
    # so topology/precision fixes never require resampling the same mesh.
    key=hashlib.sha256(np.asarray(mesh.vertices,dtype=np.float32).tobytes()
        +np.asarray(mesh.faces,dtype=np.uint32).tobytes()+np.asarray([*lo,*hi,*samples],dtype=float).tobytes()).hexdigest()
    grid_path=HERE/'assembly'/f'wing_distance_{key}.vti'
    if grid_path.exists():
        reader=vtk.vtkXMLImageDataReader();reader.SetFileName(str(grid_path));reader.Update()
        grid=reader.GetOutput()
    else:
        sample.Update();grid=sample.GetOutput()
        writer=vtk.vtkXMLImageDataWriter();writer.SetFileName(str(grid_path));writer.SetInputData(grid);writer.Write()
    # Contour the corrected signed distance with topologically consistent
    # marching cubes; do not cap or patch numerical openings afterward.
    from skimage.measure import marching_cubes
    volume=vtk_to_numpy(grid.GetPointData().GetScalars()).reshape(tuple(samples[::-1])).copy()
    # VTK's nearest-facet pseudonormal can reverse the sign beside nearly
    # collinear front-cap triangles, even outside the bounding box. Keep
    # its exact mesh distance magnitude; take inside/outside from the
    # closed analytic field that generated this same clearance mesh.
    gy=np.linspace(lo[1],hi[1],samples[1]);gz=np.linspace(lo[2],hi[2],samples[2])
    yy,zz=np.meshgrid(gy,gz,indexing='xy')
    for i,xx in enumerate(np.linspace(lo[0],hi[0],samples[0])):
        inside=clearance_field(xx,yy,zz)<0
        volume[:,:,i]=np.where(inside,-abs(volume[:,:,i]),abs(volume[:,:,i]))
    vertices,faces,_,_=marching_cubes(volume,CLEARANCE,spacing=tuple((hi-lo)[::-1]/(samples[::-1]-1)),
                                     allow_degenerate=False,method='lewiner')
    result=trimesh.Trimesh(vertices[:,::-1].astype(float)+lo,faces,process=True)
    # Remove only true export degeneracies, then orient this solid cutter.
    result=preserve_void_winding(result)
    result.fix_normals(multibody=True)
    assert result.is_watertight and result.volume>mesh.volume
    return result


def to_trimesh(manifold):
    m = manifold.to_mesh64()
    return trimesh.Trimesh(np.asarray(m.vert_properties)[:,:3], np.asarray(m.tri_verts), process=True)


def preserve_void_winding(mesh):
    """Float32 STL cleanup without turning inward cavity shells outward.

    Manifold supplies consistent oriented boundaries. Generic per-component
    normal repair would fill the wing's buried magnet cavities.
    """
    from mesh_ops import preserve_void_winding as clean
    return clean(mesh)


def organic_wing_backing(sign):
    """A rolled outer band keeps both buried UM pockets in one strong wing."""
    def field(x,y,z):
        radius=64.+2*np.sqrt(np.maximum(0,1-((z-10.15)/8.15)**2))
        radial=np.hypot(x,y-model.interface.UM_CUTOUT[1])-radius
        axial=np.maximum(2.-z,z-18.3)
        end=np.maximum(326-y,y-425)
        # Rounded end transitions are clear of the LM split and all magnets.
        joined=-model.retained.smooth_union(-radial,-end,3.)
        return np.maximum.reduce(np.broadcast_arrays(joined,axial,-sign*x,
            53.-np.hypot(x,y-model.interface.UM_CUTOUT[1])))
    bounds=([[-72,1],[325,426],[1.5,18.8]] if sign<0 else [[-1,72],[325,426],[1.5,18.8]])
    return solid(model.extract_surface(field,bounds,.30,[]))


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-clearance',action='store_true')
    args=parser.parse_args()
    verify_package()
    output = HERE/"STL/wings"
    output.mkdir(exist_ok=True)
    print("Building shared V4 wing clearance surface",flush=True)
    g = model.retained
    base = model.extract_surface(clearance_field,
                     [[.8,19.5],[-74.,74.],[307-model.INSTALLED_Y_OFFSET,463-model.INSTALLED_Y_OFFSET]], .30,
                     np.r_[model.special_x(),CUTTER_REAR_X])
    assert base.is_watertight,'Open wing-clearance source surface'
    # Key the expensive Euclidean offset by the actual exterior geometry.
    # Internal routing/export changes do not invalidate an identical surface.
    fingerprint=hashlib.sha256(np.asarray(base.vertices,dtype=np.float32).tobytes()
                              +np.asarray(base.faces,dtype=np.uint32).tobytes()).hexdigest()
    key = hashlib.sha256((fingerprint+inspect.getsource(offset_envelope)+repr(CLEARANCE)).encode()).hexdigest()
    cache = HERE/"assembly"/f"wing_clearance_{key}.stl"
    if cache.exists():
        cutter_mesh = trimesh.load_mesh(cache,process=True)
    else:
        print("Offsetting actual exterior for an even wing seam",flush=True)
        cutter_mesh = offset_envelope(base)
        cutter_mesh.export(cache)
        cutter_mesh = trimesh.load_mesh(cache,process=True)
    cutter = solid(model.installed(cutter_mesh))
    assert cutter.bounding_box()[1] > PROTECTED_BELOW_Y
    print("Clearance surface ready",flush=True)
    if args.prepare_clearance:
        return
    # Keep the delivered set intact if a fit or enclosed-pocket check fails.
    # Promote the four wings only after every staged STL has passed.
    staging=HERE/'assembly/wing_build_staging'
    staging.mkdir(exist_ok=True)
    _,parts = restored_print_parts()
    housing = solid(model.installed(parts["housing"]))
    report = {"source_sha256": sha(__file__), "model_source_sha256": sha(HERE/"v4_model.py"),
              "housing_stl_sha256": sha(HERE/"STL"/model.BODY_FILE),
              "euclidean_offset_mm": CLEARANCE,
              "protected_below_installed_y_mm": PROTECTED_BELOW_Y,
              "parts": {}}
    for variant in ["flat","graded"]:
        for side in ["left","right"]:
            name = f"obiwan_wing_{variant}_{side}_split2_2_of_2_lm_um_upper"
            source = model.ROOT/f"build/wings/{variant}/stl/{name}.stl"
            authority_path = source.with_suffix(".print.json")
            authority = json.loads(authority_path.read_text())
            assert sha(source) == authority["stl_sha256"]
            original = trimesh.load_mesh(source,process=True)
            original.apply_transform(np.linalg.inv(np.array(authority["source_to_stl_matrix"])))
            original_solid = solid(original).set_tolerance(.00005)
            # Fill the old upper magnet. Extend only the inner shoulder so
            # the wing also follows portions of the newly rounded UM that
            # recede from the previous cylindrical edge.
            relocated = model.magnet_pockets(original,'wing')
            assert len(relocated)==2
            filled = original_solid + solid(model.positive_void(relocated[0][0]))
            sign = 1 if side=='right' else -1
            from validate import box_solid
            roi=box_solid(([0,330,6.8] if sign>0 else [-75,330,6.8]),
                          ([75,449.1,18.3] if sign>0 else [0,449.1,18.3]))
            grown = filled + (filled.translate((-sign*4,0,0)) ^ roi)
            grown+=organic_wing_backing(sign)
            shaped = grown - cutter
            for _,pocket,_ in relocated:
                shaped-=solid(pocket)
            printable = to_trimesh(shaped.set_tolerance(.0001))
            printable.apply_transform(np.array(authority["source_to_stl_matrix"]))
            printable = preserve_void_winding(printable)
            # Re-normalise the print origin if a removed edge owned a minimum.
            shift = -printable.bounds[0]
            printable.apply_translation(shift)
            matrix = np.array(authority["source_to_stl_matrix"])
            matrix[:3,3] += shift
            filename = f"V4_{variant}_{side}_UPPER.stl"
            target = staging/filename
            preserve_void_winding(printable).export(target)
            reloaded = trimesh.load_mesh(target,process=True)
            assert reloaded.is_watertight and reloaded.is_winding_consistent and reloaded.volume > 0
            components = reloaded.split(only_watertight=False)
            assert sum(p.volume > 0 for p in components) == 1, "detached wing material"
            assert np.all(reloaded.area_faces>0) and np.all(reloaded.unique_faces())
            installed = reloaded.copy()
            installed.apply_transform(np.linalg.inv(matrix))
            final = solid(installed)
            intersection = abs((final ^ housing).volume())
            gap = final.min_gap(housing,1.)
            original_um,_ = model.restored_core("no_floor_stand")
            baseline_gap = original_solid.min_gap(solid(original_um),.5)
            assert intersection < 1e-6 and gap+.0002 >= baseline_gap, (filename,intersection,gap,baseline_gap)
            import manifold3d
            top_crop = solid(trimesh.creation.box([300,200,100],
                transform=trimesh.transformations.translation_matrix([0,527,0])))
            top_gap = final.min_gap(housing ^ top_crop,1.)
            assert top_gap > .15, (filename,"upper shoulder clearance",top_gap)
            # Direct intersection avoids zero-volume coplanar remnants in
            # B - (B - C), whose bounding box is a poor locality test.
            removed = original_solid ^ cutter
            removed_bounds = removed.bounding_box()
            assert removed_bounds[1] > PROTECTED_BELOW_Y, (filename,"protected mount region altered",removed_bounds)
            assert removed.volume() > 0
            # Preserve the original buried cavities as independent closed void
            # shells, and check that the cut did not expose any of those shells.
            original_voids = [p for p in original.split(only_watertight=False) if p.volume < 0]
            final_voids = [p for p in installed.split(only_watertight=False) if p.volume < 0]
            assert len(final_voids)==3
            # The original LM pocket remains at its original coordinates;
            # the other cavity must coincide with the relocated UM template.
            expected_pockets=[model.positive_void(v) for v in original_voids if v.center_mass[1]<350]
            expected_pockets += [row[1] for row in relocated]
            for expected in expected_pockets:
                actual=min(final_voids,key=lambda v:np.linalg.norm(v.center_mass-expected.center_mass))
                assert np.linalg.norm(actual.center_mass-expected.center_mass)<.0002
                assert abs(actual.volume+expected.volume)<.01
            magnets=[]
            for site,(_,pocket,transform) in zip(model.magnet_sites(sign),relocated):
                n=site['normal']
                from validate import ray_distances
                face=model.magnet_burial(site['angle_deg'],'wing')
                origin=site['contact']+(face+model.MAGNET_CAVITY_DEPTH/2)*n
                hits=ray_distances(installed,origin,-n)
                assert len(hits)>=2
                skin=float(hits[1]-hits[0])
                assert skin>model.MAGNET_SKIN-.04,(filename,'buried wing skin',skin)
                magnets.append({'angle_deg':site['angle_deg'],'center_mm':pocket.center_mass.tolist(),
                    'normal':n.tolist(),'contact_point_mm':site['contact'].tolist(),
                    'pocket_face_offset_mm':face,'cavity_diameter_mm':model.MAGNET_CAVITY_DIAMETER,
                    'cavity_depth_mm':model.MAGNET_CAVITY_DEPTH,
                    'transform':transform.tolist(),'face_skin_mm':skin})
            row = {"relocated_UM_magnets":magnets,
                   "rolled_UM_backing_band":True,
                   "side":side,
                   "preserved_LM_magnet":True,
                   "source_stl": str(source.relative_to(model.ROOT)),
                   "source_stl_sha256": sha(source), "source_print_authority_sha256": sha(authority_path),
                   "stl_sha256": sha(target), "triangles": len(reloaded.faces),
                   "watertight": True, "positive_material_components": 1,
                   "preserved_closed_voids": len(final_voids), "size_mm": reloaded.extents.tolist(),
                   "removed_material_mm3": removed.volume(), "removed_bounds_installed_mm": list(removed_bounds),
                   "housing_overlap_mm3": intersection, "minimum_housing_gap_mm": gap,
                   "original_UM_wing_gap_mm": baseline_gap,
                   "minimum_upper_shoulder_gap_mm": top_gap,
                   "source_to_stl_matrix": matrix.tolist()}
            report["parts"][filename] = row
            sidecar = {"schema_version": 1, "part": target.stem, "stl": target.name,
                       "stl_sha256": sha(target), "source_to_stl_matrix": matrix.tolist(),
                       "print_orientation": "front_face_down", "variant": variant, "side": side,
                       "qualification": "unsliced V4 matching upper wing candidate",
                       "source_stl": row["source_stl"]}
            target.with_suffix(".print.json").write_text(json.dumps(sidecar,indent=2)+"\n")
            print(filename, "removed",round(removed.volume(),3),"mm3; gap",round(gap,4),"mm",flush=True)
    for filename in report['parts']:
        for path in (staging/filename,(staging/filename).with_suffix('.print.json')):
            path.replace(output/path.name)
    pending_report=HERE/'assembly/wing_validation.pending.json'
    pending_report.write_text(json.dumps(report,indent=2)+'\n')
    pending_report.replace(HERE/'wing_validation.json')


if __name__ == "__main__":
    main()
