"""Render true installed-axis views from the checked, full-resolution print meshes.

Installed axes are +X right, +Y up, +Z forward. This only writes review
images and their provenance; it does not rebuild or alter printable geometry.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import vtk
from vtk.util.numpy_support import vtk_to_numpy

import v4_model as model
from rebuild import sha
from review import mu10_reference
from validate import restored_print_parts
import render_mesh


HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "views"
PIXELS_PER_MM = 5.0
BACKGROUND = (245, 247, 249)
INK = (31, 48, 60)
MUTED = (87, 106, 119)
BLUE = (66, 141, 189)
ORANGE = (238, 148, 43)
VIEWS = {
    "top": {
        "title": "TOP", "caption": "Looking down · front at bottom",
        "direction": [0, 1, 0], "up": [0, 0, -1], "size": [900, 360],
    },
    "front": {
        "title": "FRONT", "caption": "Looking at the forward-facing tweeter",
        "direction": [0, 0, 1], "up": [0, 1, 0], "size": [900, 1460],
    },
    "side": {
        "title": "RIGHT SIDE", "caption": "Front at left · rear at right",
        "direction": [1, 0, 0], "up": [0, 1, 0], "size": [640, 1460],
    },
}


def font(size, bold=False):
    filename = "Arial Bold.ttf" if bold else "Arial.ttf"
    for folder in (Path("/System/Library/Fonts/Supplemental"),
                   Path("/Library/Fonts")):
        if (folder / filename).is_file():
            return ImageFont.truetype(str(folder / filename), size)
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def scene_parts():
    _, restored = restored_print_parts()
    parts = []
    for name, mesh in restored.items():
        color = BLUE if name == "housing" else (
            (52, 108, 153) if name.startswith("cap") else (100, 173, 204))
        parts.append((name, model.installed(mesh), color))
    for index in (0, 1):
        parts.append((f"Dayton_ND25FN4_{index}_reference",
                      model.installed(model.retained.driver_mesh(index)), ORANGE))
    parts.append(("SEAS_MU10_reference_terminals_omitted", mu10_reference(), (213, 137, 49)))
    return parts


def render_views(parts):
    renderer = vtk.vtkRenderer()
    renderer.SetBackground(*(channel / 255 for channel in BACKGROUND))
    for _, mesh, color in parts:
        actor = render_mesh.actor(mesh, tuple(channel / 255 for channel in color))
        # Match the checked review renderer: tiny boolean slivers must not
        # interpolate their normals across otherwise planar carrier faces.
        actor.GetProperty().SetInterpolationToFlat()
        renderer.AddActor(actor)
    bounds = np.array([mesh.bounds for _, mesh, _ in parts])
    lower, upper = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    focus = (lower + upper) / 2
    corners = np.array(np.meshgrid(*zip(lower, upper))).T.reshape(-1, 3)
    window = vtk.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    window.SetMultiSamples(8)
    window.AddRenderer(renderer)
    pictures, cameras = {}, {}
    try:
        for name, view in VIEWS.items():
            direction = np.array(view["direction"], dtype=float)
            up = np.array(view["up"], dtype=float)
            right = np.cross(-direction, up)
            width, height = view["size"]
            half_frame = np.array([width, height]) / (2 * PIXELS_PER_MM)
            projection = (corners - focus) @ np.stack([right, up], axis=1)
            assert np.all(np.abs(projection) < half_frame - 5), "Image would crop the assembly"
            camera = renderer.GetActiveCamera()
            camera.SetPosition(*(focus + 1000 * direction))
            camera.SetFocalPoint(*focus)
            camera.SetViewUp(*up)
            camera.ParallelProjectionOn()
            camera.SetParallelScale(height / (2 * PIXELS_PER_MM))
            renderer.ResetCameraClippingRange()
            window.SetSize(width, height)
            window.Render()
            capture = vtk.vtkWindowToImageFilter()
            capture.SetInput(window)
            capture.SetInputBufferTypeToRGB()
            capture.ReadFrontBufferOff()
            capture.Update()
            pixels = vtk_to_numpy(capture.GetOutput().GetPointData().GetScalars())
            pictures[name] = Image.fromarray(np.flipud(pixels.reshape(height, width, 3)).copy())
            cameras[name] = {
                "projection": "orthographic", "position_mm": list(camera.GetPosition()),
                "focus_mm": list(camera.GetFocalPoint()), "up": list(camera.GetViewUp()),
                "parallel_scale_mm": camera.GetParallelScale(), "render_size_px": view["size"],
                "camera_basis_orthogonal": bool(np.dot(direction, up) == 0),
                "minimum_frame_margin_mm": float(np.min(half_frame - np.abs(projection))),
            }
            print(f"Rendered {name}", flush=True)
    finally:
        window.Finalize()
    return pictures, cameras


def panel(name, picture):
    view = VIEWS[name]
    result = Image.new("RGB", (picture.width, picture.height + 105), BACKGROUND)
    draw = ImageDraw.Draw(result)
    draw.text((30, 15), view["title"], font=font(32, True), fill=INK)
    draw.text((30, 58), view["caption"], font=font(23), fill=MUTED)
    result.paste(picture, (0, 105))
    return result


def main():
    OUTPUT.mkdir(exist_ok=True)
    parts = scene_parts()
    pictures, cameras = render_views(parts)
    panels = {name: panel(name, image) for name, image in pictures.items()}
    outputs = {}
    for name, image in panels.items():
        standalone = Image.new("RGB", (image.width, image.height + 100), BACKGROUND)
        standalone.paste(image, (0, 0))
        draw = ImageDraw.Draw(standalone)
        y = image.height + 12
        draw.rectangle((30, y + 3, 49, y + 22), fill=BLUE)
        draw.text((60, y), "Printed body + service parts", font=font(22), fill=MUTED)
        draw.rectangle((30, y + 40, 49, y + 59), fill=ORANGE)
        draw.text((60, y + 37), "Dayton ND25FN-4 + SEAS MU10", font=font(22), fill=MUTED)
        path = OUTPUT / f"orthographic_{name}.png"
        standalone.save(path)
        outputs[name] = {"path": path.name, "sha256": sha(path), "size_px": list(standalone.size)}

    sheet = Image.new("RGB", (1660, 2250), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((45, 32), "ND25FN-4 CRESCENT + UM", font=font(47, True), fill=INK)
    draw.text((47, 96), "Retained V4 · fused carrier · installed driver references", font=font(27), fill=MUTED)
    sheet.paste(panels["top"], (30, 165))
    sheet.paste(panels["front"], (30, 655))
    sheet.paste(panels["side"], (990, 655))
    draw = ImageDraw.Draw(sheet)
    x, y = 1020, 205
    draw.rectangle((x, y + 5, x + 27, y + 32), fill=BLUE)
    draw.text((x + 43, y), "PRINTED PIECE", font=font(26, True), fill=INK)
    draw.text((x + 43, y + 42), "Fused UM + V4 body", font=font(24), fill=MUTED)
    draw.text((x + 43, y + 77), "Caps and retainers installed", font=font(24), fill=MUTED)
    y += 147
    draw.rectangle((x, y + 5, x + 27, y + 32), fill=ORANGE)
    draw.text((x + 43, y), "ACTUAL DRIVER MESHES", font=font(26, True), fill=INK)
    draw.text((x + 43, y + 42), "2 × Dayton ND25FN-4", font=font(24), fill=MUTED)
    draw.text((x + 43, y + 77), "1 × SEAS MU10", font=font(24), fill=MUTED)
    draw.text((x, 513), "Orthographic · all views at the same scale", font=font(23), fill=MUTED)
    draw.text((x, 551), "LM and wings omitted for clarity", font=font(23), fill=MUTED)
    draw.text((47, 2215), "Shared STL for either LM configuration · MU10 reference omits electrical terminals", font=font(21), fill=MUTED)
    path = OUTPUT / "orthographic_top_front_side.png"
    sheet.save(path)
    outputs["sheet"] = {"path": path.name, "sha256": sha(path), "size_px": list(sheet.size)}

    sources = [HERE / "build_manifest.json", Path(__file__), HERE / "review.py",
               HERE / "validate.py", HERE / "v4_model.py", Path(render_mesh.__file__),
               model.PACKAGE / "reference/ND25FN_aligned_reference.stl",
               model.PACKAGE / "source/mechanical_v4.py",
               model.ROOT / "vendor/SEAS/MU10RB-SL/H1658-04_MU10RB-SL_driver.stl"]
    build = json.loads((HERE / "build_manifest.json").read_text())
    for filename in (model.BODY_FILE, "02_Closed_Cap_PRINT_TWO.stl", "03_Tweeter_Retainer_PRINT_TWO.stl"):
        sources.append(HERE / "STL" / filename)
        assert sha(sources[-1]) == build["files"][filename]["sha256"]
    manifest = {
        "source_sha256": {str(path.relative_to(model.ROOT)): sha(path) for path in sources},
        "configuration": "shared", "compatible_LM_configurations": list(model.LM_STATES),
        "installed_axes": {"right": "+X", "up": "+Y", "front": "+Z"},
        "pixels_per_mm": PIXELS_PER_MM, "cameras": cameras,
        "parts": [{"name": name, "triangles": len(mesh.faces), "color_rgb": color}
                  for name, mesh, color in parts],
        "outputs": outputs, "geometry_changed": False,
        "rendering": "VTK, full-resolution exported STL geometry, parallel projection, flat surface normals",
        "notes": ["LM and wings omitted to expose the fused assembly.",
                  "Service caps and retainers are separate parts, shown installed.",
                  "MU10 vendor reference is open and omits electrical terminals.",
                  "Opaque exterior views; internal wiring and hidden driver bodies are not exposed."],
    }
    (OUTPUT / "orthographic_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
