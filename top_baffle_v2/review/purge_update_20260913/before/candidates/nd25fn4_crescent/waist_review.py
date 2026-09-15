"""Identical-camera renders of actual STL surfaces at the corrected waist."""
import json
from pathlib import Path
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont

import v4_model as model
from rebuild import sha
from review import render_object

HERE = Path(__file__).resolve().parent
ARCHIVE = model.ROOT / 'review/nd25fn4_print/pre_waist_and_insert_fix'


def installed_body(folder):
    facts = json.loads((folder / 'build_manifest.json').read_text())['files'][model.BODY_FILE]
    path = folder / 'STL' / model.BODY_FILE
    if not path.exists():
        path = folder / model.BODY_FILE
    assert sha(path) == facts['sha256']
    mesh = trimesh.load_mesh(path, process=True)
    mesh.apply_transform(np.linalg.inv(facts['assembly_to_print_matrix']))
    return model.installed(mesh), path


def main():
    import render_mesh as renderer
    original = renderer.actor
    def exact_actor(*args, **kwargs):
        actor = original(*args, **kwargs)
        actor.GetProperty().SetInterpolationToFlat()
        return actor
    renderer.actor = exact_actor
    output = HERE / 'views'
    source_paths = []
    panels = []
    for label, folder in [('before', ARCHIVE), ('after', HERE)]:
        mesh, path = installed_body(folder)
        source_paths.append(path)
        objects = [render_object(mesh, [66, 141, 189, 255])]
        for view, camera in [('rear', (-155, 130, 45)), ('front', (190, 125, 30))]:
            target = output / f'waist_{view}_{label}.png'
            renderer.render(objects, target, camera=camera, focus=(2, 45, 3),
                            scale=26, size=(1200, 1000))
            panels.append(target)
    canvas = Image.new('RGB', (1840, 1610), '#f5f7f9')
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', 28)
    for column, label in enumerate(['before', 'after']):
        for row, view in enumerate(['rear', 'front']):
            x, y = 20 + 920*column, 20 + 800*row
            draw.text((x, y), f'{label.capitalize()} — {view} oblique', font=font, fill='#213441')
            panel = Image.open(output / f'waist_{view}_{label}.png').convert('RGB')
            canvas.paste(panel.resize((900, 750), Image.Resampling.LANCZOS), (x, y+42))
    target = output / 'waist_comparison.png'
    canvas.save(target)
    report = dict(source_sha256=sha(__file__),
        source_build_manifest_sha256=sha(HERE/'build_manifest.json'),
        inputs={str(p.relative_to(model.ROOT)): sha(p) for p in source_paths},
        images={p.name: sha(p) for p in [*panels, target]},
        note='Actual full-resolution STL surfaces, identical orthographic cameras and flat surface lighting; no geometry retouching.')
    (output/'waist_review_manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(target, flush=True)


if __name__ == '__main__':
    main()
