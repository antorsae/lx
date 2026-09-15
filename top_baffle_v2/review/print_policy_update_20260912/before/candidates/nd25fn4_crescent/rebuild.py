"""Rebuild the shared retained V4 fused UM body and matching service pieces.

Run with Python containing the project's CAD dependencies plus numpy, scipy,
trimesh and scikit-image. Source package files are verified and never changed.
No acoustic report, predictions, slicer project or printer job is generated.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import importlib.metadata
import json
import stat
import time
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / "design_inputs/MU10_ND25FN_V4_Retained"
ZIP_SHA256 = "a09b718ecb998404e03d69802c4a6f4e8b58f568356947072bc697fa8db6a363"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def outer_source_sha():
    """Conservative cache key excluding only final fusion/print operations."""
    tree=ast.parse((HERE/'v4_model.py').read_text())
    tree.body=[node for node in tree.body if not (
        isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef))
        and node.name in {'protected_regions','fused_body','print_pose'})]
    import v4_model as model
    value=ast.dump(tree,include_attributes=False)+sha(model.interface.__file__)+sha(model.route.__file__)
    return hashlib.sha256(value.encode()).hexdigest()


def verify_package():
    archive = ROOT / "MU10_ND25FN_V4_Retained_Package.zip"
    if sha(archive) != ZIP_SHA256:
        raise ValueError("Retained package ZIP differs from the reviewed design input")
    destination = PACKAGE.parent.resolve()
    if not PACKAGE.exists():
        with zipfile.ZipFile(archive) as package:
            for entry in package.infolist():
                target = (destination/entry.filename).resolve()
                if not target.is_relative_to(destination) or stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError(f"Unsafe archive entry: {entry.filename}")
            package.extractall(destination)
    manifest = json.loads((PACKAGE/"MANIFEST_SHA256.json").read_text())
    for entry in manifest["files"]:
        path = PACKAGE / entry["path"]
        if path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            raise ValueError(f"Retained input changed: {entry['path']}")
    return len(manifest["files"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", type=float, default=.30)
    parser.add_argument('--reuse-outer',action='store_true',
                        help='Reuse a hash-verified outer surface when only final fusion changed')
    args = parser.parse_args()
    if not .15 <= args.step <= .35:
        raise ValueError("Use 0.15–0.35 mm sampling for the delivered geometry")
    entries = verify_package()
    import numpy as np
    import trimesh
    import v4_model as model
    g = model.retained
    from mesh_ops import preserve_void_winding
    output = HERE / "STL"
    assembly = HERE / "assembly"
    output.mkdir(exist_ok=True)
    assembly.mkdir(exist_ok=True)
    result = {"design_input_zip_sha256": ZIP_SHA256,
              "verified_manifest_entries": entries,
              "model_source_sha256": sha(HERE/"v4_model.py"),
              "builder_source_sha256": sha(__file__),
              "mesh_ops_source_sha256": sha(HERE/"mesh_ops.py"),
              "route_source_sha256": sha(model.route.__file__),
              "UM_source": {},
              "compatible_LM_configurations": list(model.LM_STATES),
              "upper_gallery_routing_profile": "fixed_floor_profile_independent_of_LM_choice",
              "obiwan_interface_source_sha256": sha(model.interface.__file__),
              "python_dependencies": {name: importlib.metadata.version(name) for name in
                                      ["numpy","scipy","trimesh","scikit-image","build123d"]},
              "mesh_step_mm": args.step, "units": "mm", "files": {},
              "package_to_installed_matrix": model.PACKAGE_TO_INSTALLED.tolist()}

    def write(name, mesh, transform=None):
        path = output/name
        # The only intentional sealed cavities are the four relocated UM
        # magnets. Fill sealed remnants of the obsolete route/tie pockets,
        # including remnants sealed by final precision simplification.
        from mesh_ops import solid, to_trimesh
        body = None
        old_um,_=model.restored_core(model.UM_SOURCE_STATE)
        magnet_volumes=[p.volume for _,p,_ in model.magnet_pockets(old_um,'body')]
        for shell in mesh.split(only_watertight=False):
            if shell.volume<0 and min(abs(abs(shell.volume)-v) for v in magnet_volumes)>.01:
                if body is None:
                    body = solid(mesh)
                shell.faces = shell.faces[:, [0, 2, 1]]
                body += solid(shell)
        if body is not None:
            mesh = to_trimesh(body.set_tolerance(.0001))
        preserve_void_winding(mesh).export(path)
        m = trimesh.load_mesh(path, process=True)
        components = m.split(only_watertight=False)
        facts = {"sha256": sha(path), "triangles": len(m.faces),
                 "watertight": bool(m.is_watertight),
                 "winding_consistent": bool(m.is_winding_consistent),
                 "components": len(components), "volume_mm3": float(m.volume),
                 "bounds_mm": m.bounds.tolist()}
        if transform is not None:
            facts["assembly_to_print_matrix"] = transform.tolist()
        assert facts["watertight"] and facts["winding_consistent"]
        assert sum(c.volume > 0 for c in components) == 1 and m.volume > 0, facts
        assert sum(c.volume < 0 for c in components) == (model.UM_MAGNET_COUNT if name.startswith('01_') else 0)
        result["files"][name] = facts
        print(name, json.dumps(facts), flush=True)

    start = time.monotonic()
    print("Rebuilding compact V4 with organic UM waist, relocated magnets and routed cable", flush=True)
    outer_path=assembly/'organic_outer_package_mm.stl'
    cache_path=assembly/'organic_outer_cache.json'
    if args.reuse_outer:
        cache=json.loads(cache_path.read_text())
        assert cache['outer_source_sha256']==outer_source_sha(),'Outer source changed; regenerate it'
        assert cache['mesh_step_mm']==args.step and cache['stl_sha256']==sha(outer_path)
        upper=trimesh.load_mesh(outer_path,process=True)
        print('Reusing verified outer surface; rebuilding native-interface fusion',flush=True)
    else:
        upper = model.extract_surface(model.housing_field,
            [[-19.5,19.5+model.UM_RIM_RISE],[-72,72],[295-model.INSTALLED_Y_OFFSET,76]],
            args.step,model.special_x())
        upper.export(outer_path)
        cache={'outer_source_sha256':outer_source_sha(),'mesh_step_mm':args.step,
               'stl_sha256':sha(outer_path),'original_model_source_sha256':sha(HERE/'v4_model.py')}
        cache_path.write_text(json.dumps(cache,indent=2)+'\n')
    result['outer_surface_sha256']=sha(outer_path)
    result['outer_surface_cache_sha256']=sha(cache_path)
    result['outer_surface_reused']=args.reuse_outer
    housing, provenance = model.fused_body(upper)
    result["UM_source"] = provenance
    printable, transform = model.print_pose(housing)
    filename = model.BODY_FILE
    write(filename,printable,transform)
    authority = {"schema_version":1,"part":"nd25fn4_v4_fused_um_shared",
        "stl":filename,"stl_sha256":result["files"][filename]["sha256"],
        "source_frame":"Obi-Wan installed, X lateral / Y up / Z forward, mm",
        "source_to_stl_matrix":(transform @ np.linalg.inv(model.PACKAGE_TO_INSTALLED)).tolist(),
        "print_orientation":"front_face_down_45_degree_diagonal",
        "compatible_LM_configurations":list(model.LM_STATES),
        "replaces":"UM carrier plus separate V4 housing, with either LM configuration",
        "qualification":"unsliced mechanical candidate"}
    (output/filename).with_suffix(".print.json").write_text(json.dumps(authority,indent=2)+"\n")
    print(f"Shared fused body complete in {time.monotonic()-start:.1f}s", flush=True)

    print("Rebuilding matching V4 cap and retaining ring", flush=True)
    cap = g.extract(lambda x,y,z: g.cap_field(x,y,z,0),
                    [[-19.5,-5.0],[-29,29],[-61,-1]], min(args.step,.25), g.exact_special_x())
    cap.export(assembly/"cap_lower_package_mm.stl")
    cp = cap.copy()
    vertices = cp.vertices.copy()
    cp.vertices = np.c_[vertices[:,1], -(vertices[:,2]+31), -6.05-vertices[:,0]]
    cp.fix_normals()
    cp.apply_translation([0,0,-cp.bounds[0,2]])
    write("02_Closed_Cap_PRINT_TWO.stl", cp)

    ring = g.extract(g.retainer_local,
                     [[-.8,5.2],[-27.4,27.4],[-27.4,27.4]], min(args.step,.20),
                     [g.M.retainer_back_x, g.M.retainer_front_x, g.M.retainer_land_top_x])
    ring.export(assembly/"retainer_local_mm.stl")
    rp = ring.copy()
    vertices = rp.vertices.copy()
    rp.vertices = np.c_[vertices[:,1],vertices[:,2],vertices[:,0]-g.M.retainer_back_x]
    rp.fix_normals()
    write("03_Tweeter_Retainer_PRINT_TWO.stl", rp)
    # Ensure no builder mutated the retained package as a side effect.
    verify_package()
    (HERE/"build_manifest.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"STLs rebuilt in {time.monotonic()-start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
