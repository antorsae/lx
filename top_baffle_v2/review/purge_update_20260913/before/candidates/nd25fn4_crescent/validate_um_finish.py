"""Refresh the declared V4 gallery and measure the lower UM with both LMs."""
import json
import subprocess
import sys

import numpy as np
import trimesh

import v4_model as model
from rebuild import sha


def main():
    routes, _, _, _ = model.wiring()
    reference = {
        'source_sha256': {
            str(model.HERE.relative_to(model.ROOT)/name): sha(model.HERE/name)
            for name in ('v4_model.py', 'build_manifest.json')},
        'paths': {
            name: {'centers_mm': trimesh.transform_points(points, model.PACKAGE_TO_INSTALLED).tolist(),
                   'radii_mm': np.broadcast_to(radii, len(points)).tolist()}
            for name, (points, radii) in routes.items()}}
    path = model.HERE/'assembly/wire_gallery_reference.json'
    path.write_text(json.dumps(reference)+'\n')
    for state, suffix in [('no_floor_stand', 'no_floor'), ('floor_stand', 'floor')]:
        subprocess.run([
            sys.executable, str(model.ROOT/'scripts/check_um_finish.py'),
            '--um-stl', 'candidates/nd25fn4_crescent/STL/'+model.BODY_FILE,
            '--lm-state', state,
            '--gallery-reference', 'candidates/nd25fn4_crescent/assembly/wire_gallery_reference.json',
            '--output', f'candidates/nd25fn4_crescent/um_finish_{suffix}_validation.json'],
            cwd=model.ROOT, check=True)


if __name__ == '__main__':
    main()
