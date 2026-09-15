"""Proof required when rebinding the existing catalog after the UM finish repair."""
import json
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from lx521_baffle.magnets import wall_cavity_tools
from lx521_baffle.obiwan.carriers import side_magnet_sites, SIDE_INTERFACE_GAP
from lx521_baffle.base import THICKNESS_MM
from lx521_baffle.io import sha256_file


def local_binding_proof(path, candidate):
    old = json.loads(path.read_text())
    before = {a['id']: a for a in old['artifacts']}
    changes = []
    for a in candidate['artifacts']:
        prior = before[a['id']]
        if prior['stl_sha256'] == a['stl_sha256']:
            continue
        if a['part'] != 'obiwan_core_2_of_2_um_carrier':
            raise ValueError(f'Local refresh has no geometry proof for {a["id"]}')
        mesh = trimesh.load_mesh(path.parent/a['stl'], process=True)
        assert mesh.is_watertight and mesh.is_winding_consistent
        mesh.apply_transform(np.linalg.inv(np.asarray(a['source_to_stl_matrix'])))
        voids = [s for s in mesh.split(only_watertight=False) if s.volume < 0]
        assert len(voids) == 2
        rows = []
        for site in side_magnet_sites('um'):
            tools = wall_cavity_tools(name=site['name'], face=site['face'],
                outward=(*site['normal'], 0.), owner='base', axis_z=site['z_mm'],
                print_up=(0.,0.,-1.), front_z=THICKNESS_MM, interface_gap_mm=SIDE_INTERFACE_GAP)
            shape = tools.cutters[0].fuse(*tools.cutters[1:])
            vertices, faces = shape.tessellate(.01, .05)
            reference = trimesh.Trimesh(np.asarray([tuple(v) for v in vertices]), np.asarray(faces), process=True)
            actual = min(voids, key=lambda s: np.linalg.norm(s.center_mass-reference.center_mass))
            # Surface distance, independent of facet order/resolution. The
            # tolerance is smaller than the qualified retaining bead width.
            sample = actual.vertices
            _, distance, _ = trimesh.proximity.closest_point(reference, sample)
            vol_error = abs(abs(actual.volume)-abs(reference.volume))
            assert distance.max() < .04 and vol_error < .08, (site['name'], distance.max(), vol_error)
            rows.append(dict(site=site['name'], max_surface_error_mm=float(distance.max()),
                             cavity_volume_error_mm3=float(vol_error)))
        changes.append(dict(id=a['id'], previous_stl_sha256=prior['stl_sha256'],
                            stl_sha256=a['stl_sha256'], unchanged_D5_cavities=rows))
    return dict(kind='existing_local_artifact_revalidation',
        note='Current existing STL/sidecar and source bindings. No remote rebuild or new physical qualification claimed. Unchanged STLs retain their bytes; changed UM cavities checked against shared analytic geometry.',
        previous_catalog_sha256=sha256_file(path), previous_source_revision=old['source_revision'],
        validator_sha256=sha256_file(Path(__file__)), changed_geometry=changes,
        bindings=[dict(id=a['id'], stl=a['stl_sha256'], authority=a['print_sidecar_sha256'],
                       source=a['source_file_sha256']) for a in candidate['artifacts']])
