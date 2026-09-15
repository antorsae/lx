#!/usr/bin/env python3
"""Render one complete parts/magnet PNG from current STL/STEP artifacts.

No production geometry or slicer project is changed. STL orientations are
inverted through their hash-checked sidecars. STEP-only candidates are
tessellated in memory for illustration, never exported as print meshes.
"""
from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import trimesh

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images/generated/catalog"
V4 = ROOT / "candidates/nd25fn4_crescent"
sys.path.insert(0, str(ROOT / "scripts"))
from delivery_contract import canonical_source

BG = "#edf1f2"
PAPER = "#ffffff"
INK = "#19313f"
MUTED = "#536974"
LINE = "#d7e1e5"
COLORS = {"stock": "#608ca9", "slim": "#83aba7", "obiwan": "#548ab4",
          "v4": "#3a779f", "candidate": "#9382a4", "test": "#92a1ac"}
MAGNETS = {
    "D5": {"size_mm": [5, 2], "grade": "N52", "color": "#007a60",
           "supplier": "Superimanes D-05-02-N52", "label": "Ø5×2 mm · N52",
           "url": "https://www.superimanes.com/imanes-de-neodimio/discos/iman-neodimio-disco-5x2-mm-n52"},
    "D6": {"size_mm": [6, 3], "grade": "N45", "color": "#ca481d",
           "supplier": "Superimanes D-06-03", "label": "Ø6×3 mm · N45",
           "url": "https://www.superimanes.com/imanes-de-neodimio/discos/iman-neodimio-disco-6x3-mm"},
}
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
RENDER_REVISION = 1
SOURCES: dict[str, str] = {}
PARTS: dict[str, dict] = {}
CARDS: dict[str, dict] = {}
SHELF = json.loads((ROOT / "to_print/catalog.json").read_text())["entries"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    path = Path(path)
    SOURCES[str(path.relative_to(ROOT))] = sha(path)
    return json.loads(path.read_text())


def normalize_site(site, authority):
    size = float(site.get("magnet_diameter_mm", 5))
    return {"name": site["name"], "type": "D6" if size == 6 else "D5",
            "center_mm": site.get("seated_magnet_center_xyz_mm", site.get("cavity_center_xyz_mm")),
            "axis": site.get("installed_marked_pole_axis_xyz", site.get("marked_pole_axis_xyz")),
            "authority": str(authority.relative_to(ROOT))}


def load_station_catalogs():
    index = {}
    catalogs = [ROOT / "review/captive_magnet_release_catalog.json",
                *sorted((ROOT / "build/bmr_crescent_TEBM35C10-4").glob("*.catalog.json")),
                *sorted((ROOT / "build/vase_TEBM35C10-4").glob("*/*.catalog.json"))]
    for path in catalogs:
        for artifact in read_json(path)["artifacts"]:
            source = (path.parent / artifact["stl"]).resolve()
            if sha(source) != artifact["stl_sha256"]:
                raise ValueError(f"Stale magnet catalog: {source}")
            index[source] = [normalize_site(s, path) for s in artifact["sites"]]
    return index


STATIONS = load_station_catalogs()


def add_part(key, path, title, *, family, state="shared", status="Current STL",
             frame="installed", sites=None, note="", label=None):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    assert path.is_file(), path
    rel = str(path.relative_to(ROOT))
    digest = sha(path)
    SOURCES[rel] = digest
    sidecar = path.with_suffix(".print.json")
    matrix = np.eye(4)
    if frame == "installed" and sidecar.is_file():
        authority = read_json(sidecar)
        assert authority["stl_sha256"] == digest, (path, "sidecar hash")
        matrix = np.linalg.inv(np.asarray(authority["source_to_stl_matrix"]))
    PARTS[key] = dict(id=key, path=rel, sha256=digest, title=title,
                      label=label or title, family=family, state=state,
                      status=status, frame=frame, mesh_to_view_matrix=matrix.tolist(),
                      sites=sites if sites is not None else STATIONS.get(path.resolve(), []),
                      note=note)
    return key


def card(key, title, ids, note, *, columns=None, subtitle="", view="front"):
    CARDS[key] = dict(id=key, title=title, parts=list(ids), note=note,
                      columns=columns or min(len(ids), 2), subtitle=subtitle, view=view)
    return key


def inventory():
    sections = []
    for family, title in [("stock", "STOCK  /  full-depth 18.3 mm"),
                          ("slim", "SLIM  /  11.5 mm acoustic field")]:
        entries = [e for e in SHELF if e["family"] == family]
        ids = []
        short = ["No floor stand", "Floor stand", "Middle left", "Middle right", "UM + integral tweeter",
                 "Lower left", "Upper left", "Lower right", "Upper right", "Left", "Right"]
        for e, label in zip(entries, short):
            ids.append(add_part(e["name"], canonical_source(ROOT, e), e["description"], family=family,
                                state=e.get("state", "shared"), label=label))
        groups = [card(f"{family}_lower", "01 · LM bottom", ids[:2], "Choose ONE stand state; 02–04 stay shared.", subtitle="2 stand alternatives"),
                  card(f"{family}_mids", "02 + 03 · LM middle", ids[2:4], "Print both left and right pieces."),
                  card(f"{family}_vase", "04 · UM / tweeter vase", ids[4:5], "4 D5 buried in the vase: upper/lower, left/right.", subtitle="Dayton ND25FW-4 pair"),
                  card(f"{family}_shoulders", "05–08 · A shoulders", [ids[6],ids[8],ids[5],ids[7]], "1 D5 per piece; four pieces = 4 D5.", subtitle="Alternative to B1 wings"),
                  card(f"{family}_wings", "09 + 10 · B1 wings", ids[9:11], "2 D5 per wing; left + right = 4 D5.", subtitle="Alternative to A shoulders")]
        grommets = []
        for half in "ab":
            grommets.append(add_part(f"{family}_grommet_{half}", f"build/no_floor_stand/stl/{family}_um_grommet_half_{half}.stl",
                f"{family.title()} UM grommet half {half.upper()}", family=family, label=f"Half {half.upper()}"))
        groups.append(card(f"{family}_grommets", "UM cable grommet", grommets,
                           "TPU pair; use the matching Stock/Slim outlet."))
        sections.append(dict(title=title, description="Bare, A shoulders or B1 wings. Select one perimeter set. All magnetic stations are D5.",
                             cards=groups, columns=6, height=590, color=COLORS[family]))

    # Keep shelf keys; resolve the service lid independently of slot spelling.
    obi = [e for e in SHELF if e["family"] == "obiwan" and not e.get("composite_plate")
           and (e["name"].startswith(("obiwan_01_", "obiwan_02_", "obiwan_03_", "obiwan_04_", "obiwan_NL8_")))]
    oi = []
    for e, label in zip(obi, ["No floor stand", "Floor stand", "Shared LM top", "Regular UM", "ND25FW-4 crescent", "Floor service lid"]):
        oi.append(add_part(e["name"], canonical_source(ROOT, e), e["description"], family="obiwan", label=label,
                           state=e.get("state", "shared")))
    assert len(oi) == 6
    monoliths = []
    for state in ["no_floor_stand", "floor_stand"]:
        monoliths.append(add_part(f"obi_monolith_{state}", f"build/{state}/stl/obiwan_core_1_of_2_lm_carrier.stl",
            f"LM monolith / {state}", family="obiwan", state=state, status="Large-format STL; outside P2S shelf",
            label="Floor" if state == "floor_stand" else "No floor"))
    sections.append(dict(title="OBI-WAN  /  regular core and crescent", description="The same LM top accepts the regular UM or fused Dayton ND25FN-4. Floor / no-floor changes the lower LM only.",
                         cards=[card("obi_lower", "01 · Keyed LM bottom", oi[:2], "2 D5 per version, one at each lower shoulder."),
                                card("obi_top", "02 · Keyed LM top", oi[2:3], "2 D5: upper left + upper right LM rim."),
                                card("obi_um", "03 · Regular UM", oi[3:4], "2 D5: upper left + upper right UM rim."),
                                card("obi_t", "04 · Regular crescent", oi[4:5], "No magnets. Screws attach it to the UM.", subtitle="Dayton ND25FW-4 pair"),
                                card("obi_lid", "NL8 service lid", oi[5:6], "Floor-stand service cover; no magnets."),
                                card("obi_monolith", "Alternative · LM monolith", monoliths, "4 D5 each. Replaces BOTH keyed LM halves.", subtitle="Large printer only · not P2S")],
                         columns=6, height=620, color=COLORS["obiwan"]))
    wings = []
    for style in ["flat", "graded"]:
        for role, number, label in [("lm_lower", 1, "LOWER"), ("lm_um_upper", 2, "regular UPPER")]:
            pair = []
            for side in ["left", "right"]:
                stem = f"obiwan_wing_{style}_{side}_split2_{number}_of_2_{role}"
                pair.append(add_part(stem, f"build/wings/{style}/stl/{stem}.stl", f"{style} {role} {side}", family="obiwan", label=side.title()))
            note = ("1 D5 per side at lower LM. Shared with Dayton ND25FN-4." if number == 1 else
                    "2 D5 per side: upper LM + regular UM. Use only with regular UM.")
            wings.append(card(f"wings_{style}_{role}", f"{style.title()} · {label} wings", pair, note, subtitle="Left + right · same in both stand states"))
    sections.append(dict(title="OBI-WAN WINGS  /  flat or graded", description="Two pieces per side: LOWER + UPPER. Choose one style. The LOWER pieces also fit the Dayton ND25FN-4 assembly.",
                         cards=wings, columns=4, height=590, color=COLORS["obiwan"]))

    # V4 seats are recovered from the actual closed cavities using the same
    # geometry function as the qualified pause workflow, then unplaced.
    sys.path.insert(0, str(V4))
    import print_magnets
    def v4_part(key, path, title, label=None):
        authority = read_json(path.with_suffix(".print.json"))
        specs = print_magnets.magnet_geometry(path, authority, [0., 0., 0.])
        inverse = np.linalg.inv(np.asarray(authority["source_to_stl_matrix"]))
        sites = []
        for spec in specs:
            sites.append(dict(name=spec["name"], type="D6" if spec["diameter_mm"] == 6 else "D5",
                center_mm=trimesh.transform_points([spec["center_bed_mm"]], inverse)[0].tolist(),
                axis=(inverse[:3, :3] @ spec["pole_axis_bed"]).tolist(),
                authority="candidates/nd25fn4_crescent/print_magnets.py:magnet_geometry"))
        return add_part(key, path, title, label=label, family="v4", sites=sites, status="Sliced; physical qualification pending")
    body = v4_part("v4_body", V4 / "STL/01_UM_Crescent_V4.stl", "Fused UM + ND25FN-4 waveguide", "Shared fused body")
    cap = add_part("v4_cap", V4 / "STL/02_Closed_Cap_PRINT_TWO.stl", "ND25FN-4 closed cap", family="v4", frame="raw", label="Print TWO", status="Sliced; physical qualification pending")
    retainer = add_part("v4_retainer", V4 / "STL/03_Tweeter_Retainer_PRINT_TWO.stl", "ND25FN-4 tweeter retainer", family="v4", frame="raw", label="Print TWO", status="Sliced; physical qualification pending")
    v4cards = [card("v4_body", "01 · Fused UM + ND25FN-4", [body], "4 D6: two buried at each UM flank. Same body for both stands.", subtitle="Dayton ND25FN-4 pair"),
               card("v4_caps", "02 · Closed caps", [cap], "2 caps per body. No magnets; O-ring fit.", view="rear"),
               card("v4_retainers", "03 · Tweeter retainers", [retainer], "2 retainers. M3 screws + Ø5×4 mm M3 inserts; no magnets.")]
    for style in ["flat", "graded"]:
        pair = [v4_part(f"v4_{style}_{side}", V4 / f"STL/wings/V4_{style}_{side}_UPPER.stl",
                        f"ND25FN-4 {style} upper {side}", side.title()) for side in ["left", "right"]]
        v4cards.append(card(f"v4_{style}_wings", f"{style.title()} · ND25FN-4 UPPER wings", pair,
                            "Each: 2 D6 at UM + 1 D5 at LM. Use the matching regular LOWER pair."))
    sections.append(dict(title="OBI-WAN + DAYTON ND25FN-4  /  fused UM and waveguide", description="Replaces regular UM + separate crescent. Uses its own UPPER wings. 8 D6 + 8 D5 for a complete winged speaker.",
                         cards=v4cards, columns=5, height=650, color=COLORS["v4"]))

    alternatives = []
    slim_alternatives = []
    for family in ["stock", "slim"]:
        p = ROOT / f"build/vase_TEBM35C10-4/{family}/vase_TEBM35C10-4.stl"
        k = add_part(f"bmr_vase_{family}", p, f"BMR vase / {family}", family="candidate", status="Qualification candidate")
        alternatives.append(card(k, f"{family.title()} · BMR vase", [k], "4 D5 on the two tweeter lands; matching perimeter not supplied.", subtitle="TEBM35C10-4 opposed pair"))
    for style, stem in [("Coaxial", "obiwan_bmr_crescent_TEBM35C10-4"), ("Opposed", "obiwan_bmr_crescent_opposed_TEBM35C10-4")]:
        k = add_part(f"bmr_{style.lower()}", f"build/bmr_crescent_TEBM35C10-4/{stem}.stl", f"Obi-Wan BMR {style.lower()}", family="candidate", status="Qualification candidate")
        alternatives.append(card(k, f"Obi-Wan · BMR {style.lower()}", [k],
                                f"{len(PARTS[k]['sites'])} D5; future side attachments, no released magnet mate.", subtitle="TEBM35C10-4 pair"))
    ptt = add_part("purifi", "build/ptt_crescent_PTT1.3T04-HAG-01/obiwan_ptt13_dipole_crescent_PTT1.3T04-HAG-01.stl",
                   "Purifi dipole crescent", family="candidate", status="Qualification candidate", frame="raw")
    alternatives.append(card("purifi", "Obi-Wan · Purifi crescent", [ptt], "No magnets. UM joint integration / physical fit remain to qualify.", subtitle="PTT1.3T04-HAG-01 + WG104 pair"))
    sections.append(dict(title="ALTERNATE TWEETERS  /  full-land BMR and Purifi", description="Prototype alternatives: select one compatible carrier per assembly. BMR magnet stations have no delivered matching perimeter.",
                         cards=alternatives, columns=5, height=600, color=COLORS["candidate"]))
    for family in ["stock", "slim"]:
        p = ROOT / f"build/bmr_slim_TEBM35C10-4/proud/{family}/vase_TEBM35C10-4.step"
        f = read_json(p.with_suffix(".facts.json"))
        facts = f["design"]["t_captive_magnets"]
        # The facts give the actual land faces and axes; the frozen CAD-only
        # topology uses a .45 skin. Seat against that skin: 1 mm half-disc.
        sites = [dict(name=f"land_{i}_{side}", type="D5", center_mm=[x-math.copysign(1.45,x), y, facts["axis_z_mm"]],
                      axis=[math.copysign(1,x),0,0], authority=str(p.with_suffix(".facts.json").relative_to(ROOT)))
                 for i,y in enumerate(facts["axis_y_mm"]) for side,x in enumerate(facts["interface_face_x_mm"])]
        k = add_part(f"bmr_slim_vase_{family}", p, f"BMR-slim vase / {family}", family="candidate", sites=sites, status="STEP only; no print project", frame="raw")
        slim_alternatives.append(card(k, f"{family.title()} · BMR-slim vase", [k], "4 D5. Slim driver lands; matching perimeter not supplied.", subtitle="STEP-only candidate"))
    for style, stem in [("coaxial", "obiwan_bmr_slim_crescent_TEBM35C10-4"), ("opposed", "obiwan_bmr_slim_crescent_opposed_TEBM35C10-4")]:
        p = ROOT / f"build/bmr_slim_TEBM35C10-4/{stem}.step"
        f = read_json(p.with_suffix(".facts.json"))
        sites = [normalize_site(s, p.with_suffix(".facts.json")) for s in f["design"]["magnets"]["stations"]]
        k = add_part(f"bmr_slim_{style}", p, f"Obi-Wan BMR-slim {style}", family="candidate", sites=sites, status="STEP only; no print project", frame="raw")
        slim_alternatives.append(card(k, f"Obi-Wan · BMR-slim {style}", [k], f"{len(sites)} D5. No released magnet mate.", subtitle="STEP-only candidate"))
    sections.append(dict(title="BMR-SLIM  /  additional CAD-only variants", description="Ø56 driver-following core with local magnet lobes. These four alternatives have STEP geometry, but no delivered STL / 3MF.",
                         cards=slim_alternatives, columns=4, height=540, color=COLORS["candidate"]))

    tests = []
    coupon_groups = [([1,2], "01 + 02 · Fit plate / key", "Fit, inserts and keys; plate has 1 D5."),
                     ([3,4], "03 + 04 · Cable entries", "Fish entry / proud UM outlet. No magnets."),
                     ([5,6], "05 + 06 · Cable turns", "Tweeter dive / foot route. No magnets."),
                     ([7,8], "07 + 08 · Seat / oval duct", "Recess seat / proud tweeter oval. No magnets."),
                     ([9,12], "09 + 12 · UM fit / bypass", "Faston clocking / closed-bore bump. No magnets.")]
    for numbers, title, note in coupon_groups:
        pair = []
        for n in numbers:
            p = next((ROOT / "build/no_floor_stand/stl").glob(f"lx521_coupon_{n}_*.stl"))
            pair.append(add_part(f"coupon_{n}", p, p.stem.removeprefix("lx521_").replace("_", " "), family="test",
                                 status="Diagnostic; not an assembly part", label=f"Coupon {n:02d}"))
        tests.append(card(f"coupon_{numbers[0]}", title, pair, note))

    def buried_void_sites(path, diameter):
        """Locate coupon cavity markers from enclosed negative-volume shells."""
        mesh = trimesh.load_mesh(path, process=True)
        return [dict(name=f"cavity_{i+1}", type=f"D{diameter}", center_mm=s.center_mass.tolist(), axis=None,
                     authority=f"{Path(path).relative_to(ROOT)}: enclosed cavity centroid")
                for i,s in enumerate(mesh.split(only_watertight=False)) if s.volume < -1]
    pair = []
    for side in ["carrier", "ae_wing"]:
        p = ROOT / f"coupons/obiwan_ae_embed/stl/lx521_coupon_obiwan_ae_embed_{side}.stl"
        pair.append(add_part(f"d5_coupon_{side}", p, f"D5 process coupon {side}", family="test", frame="raw",
            sites=buried_void_sites(p, 5), label="Carrier" if side == "carrier" else "Wing", status="Diagnostic; frozen coupon"))
    tests.append(card("d5_coupon", "D5 embedded-magnet test", pair, "2 D5 per strip. Frozen process coupon; not production pull-force proof."))
    pair = []
    for stem, label in [("01_D6_body_coupon", "Body station"), ("D6_wing_coupon", "Wing station")]:
        p = V4 / f"print/qualification/{stem}.stl"
        pair.append(add_part(stem, p, f"D6 test {label}", family="test", frame="raw", sites=buried_void_sites(p, 6), label=label,
                             status="Sliced qualification pair; physical result pending"))
    tests.append(card("d6_coupon", "D6 body / wing test", pair, "1 D6 per piece. Actual ND25FN-4 pocket and skin; physical print result pending."))
    pair = [add_part(f"registration_{s}", f"qualification/{s}_pair.stl", s.replace("_", " "), family="test", frame="raw", label=label,
                     status="Unsliced qualification fixture") for s,label in [("male_pin","Pins"),("female_socket","Sockets")]]
    tests.append(card("registration", "LM registration test", pair, "Braced pin / socket pair. No magnets; not a carrier alternative."))
    pair = [add_part(s, f"qualification/{s}.stl", s.replace("_", " "), family="test", frame="raw", label=label,
                     status="Unsliced qualification fixture") for s,label in [("native_route_cover","Route cover"),("supported_roof_and_wall_steps","Roof / wall steps")]]
    tests.append(card("process", "Route / support witnesses", pair, "Support removal, cover and wall tests. No magnets."))
    pair = [add_part(f"polar_{s}", f"build/floor_stand/stl/lx521_polar_base_{s}.stl", f"Polar index {label.lower()}", family="test", frame="raw", label=label,
                     status="Measurement accessory") for s,label in [("1of2_base","Base"),("2of2_rotor","Rotor")]]
    tests.append(card("polar", "Polar measurement base", pair, "Measurement / indexing accessory. No magnets."))
    sections.append(dict(title="TEST PIECES & MEASUREMENT ACCESSORIES", description="Separate fixtures, not extra parts in a speaker. Stock/Slim grommets above replace the old coupon 10/11 names.",
                         cards=tests, columns=5, height=560, color=COLORS["test"]))
    return sections


@lru_cache(None)
def font(size, bold=False):
    return ImageFont.truetype(str(FONT_DIR / ("Arial Bold.ttf" if bold else "Arial.ttf")), size)


def rgb(color):
    return tuple(int(color[i:i+2],16)/255 for i in [1,3,5])


def triangles(part):
    path = ROOT / part["path"]
    if path.suffix == ".step":
        from build123d import import_step
        shape = import_step(path)
        vertices, faces = shape.tessellate(.09, .25)
        points = np.asarray([tuple(v) for v in vertices])
        tri = points[np.asarray(faces)]
    else:
        raw = path.read_bytes()
        n = int.from_bytes(raw[80:84],"little")
        if len(raw) == 84 + n*50:
            dtype = np.dtype([("n","<f4",3),("p","<f4",(3,3)),("a","<u2")])
            tri = np.frombuffer(raw,dtype=dtype,count=n,offset=84)["p"].astype(float)
        else:
            tri = trimesh.load_mesh(path, process=False).triangles
    matrix = np.asarray(part["mesh_to_view_matrix"])
    return tri @ matrix[:3,:3].T + matrix[:3,3]


def render(part, width, height, view="front"):
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
    spec = dict(part=part, size=[width,height], view=view, revision=RENDER_REVISION)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    cache = OUT / "tiles" / f"{part['id']}_{width}x{height}.png"
    check = cache.with_suffix(".json")
    if cache.exists() and check.exists() and json.loads(check.read_text()).get("input_sha256") == digest:
        return Image.open(cache).convert("RGB")
    print(f"render {part['id']}", flush=True)
    renderer = vtk.vtkRenderer(); renderer.SetBackground(*rgb(PAPER))
    clouds=[]
    for component in part.get("components",[part]):
        tri = triangles(component)
        points = tri.reshape(-1,3);clouds.append(points)
        cloud = vtk.vtkPoints()
        cloud.SetData(numpy_to_vtk(points, deep=True))
        cells = np.column_stack([np.full(len(tri),3,dtype=np.int64), np.arange(len(points),dtype=np.int64).reshape(-1,3)])
        polygons = vtk.vtkCellArray()
        polygons.ImportLegacyFormat(numpy_to_vtkIdTypeArray(cells.ravel(), deep=True))
        poly = vtk.vtkPolyData(); poly.SetPoints(cloud); poly.SetPolys(polygons)
        clean = vtk.vtkCleanPolyData(); clean.SetInputData(poly)
        normals = vtk.vtkPolyDataNormals(); normals.SetInputConnection(clean.GetOutputPort())
        normals.SetFeatureAngle(45); normals.ConsistencyOn()
        mapper = vtk.vtkPolyDataMapper(); mapper.SetInputConnection(normals.GetOutputPort())
        actor = vtk.vtkActor(); actor.SetMapper(mapper)
        actor.GetProperty().SetColor(*rgb(component.get("assembly_color",COLORS[component["family"]])))
        actor.GetProperty().SetAmbient(.30); actor.GetProperty().SetDiffuse(.7)
        actor.GetProperty().SetSpecular(.15); actor.GetProperty().SetSpecularPower(28)
        renderer.AddActor(actor)
    points=np.concatenate(clouds)
    direction = np.asarray([.24,.15,1.]) if view=="front" else np.asarray([-.24,.15,-1.])
    # Orient raw print fixtures with print Z vertical; their broad faces still
    # read in the same shallow oblique projection as the acoustic parts.
    direction /= np.linalg.norm(direction)
    right = np.cross([0.,1.,0.], direction); right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    center = (points.min(axis=0)+points.max(axis=0))/2
    px,py = (points-center)@right, (points-center)@up
    center += right*(px.min()+px.max())/2 + up*(py.min()+py.max())/2
    aspect = width/height
    span = max(float(np.ptp(py)), float(np.ptp(px))/aspect)
    camera = renderer.GetActiveCamera(); camera.ParallelProjectionOn()
    camera.SetFocalPoint(*center); camera.SetPosition(*(center+direction*1500)); camera.SetViewUp(*up)
    camera.SetParallelScale(span*.57)
    renderer.ResetCameraClippingRange()
    win = vtk.vtkRenderWindow(); win.SetOffScreenRendering(1); win.SetMultiSamples(4)
    win.SetSize(width,height); win.AddRenderer(renderer); win.Render()
    cap = vtk.vtkWindowToImageFilter(); cap.SetInput(win); cap.SetInputBufferTypeToRGB(); cap.ReadFrontBufferOff(); cap.Update()
    pixels = vtk_to_numpy(cap.GetOutput().GetPointData().GetScalars()).reshape(height,width,3)[::-1].copy()
    positions = []
    for site in part["sites"]:
        renderer.SetWorldPoint(*site["center_mm"],1); renderer.WorldToDisplay()
        x,y,_ = renderer.GetDisplayPoint()
        assert 5 < x < width-5 and 5 < height-y < height-5, (part["id"],site["name"],x,height-y)
        positions.append((x,height-y,site["type"]))
    win.Finalize()
    result = Image.fromarray(pixels); draw = ImageDraw.Draw(result)
    radius = 9 if part.get("paired_markers") else max(10, round(min(width,height)*.034))
    for x,y,kind in positions:
        color = MAGNETS[kind]["color"]
        draw.ellipse((x-radius-3,y-radius-3,x+radius+3,y+radius+3),fill="white")
        draw.ellipse((x-radius,y-radius,x+radius,y+radius),fill="white" if part.get("paired_markers") else color,outline=color,width=3)
        draw.text((x,y),kind[-1],font=font(round(radius*1.45),True),fill=color if part.get("paired_markers") else "white",anchor="mm")
    cache.parent.mkdir(parents=True,exist_ok=True); result.save(cache)
    check.write_text(json.dumps(dict(input_sha256=digest,source=part["path"],source_sha256=part["sha256"],
                                    marker_count=len(positions),marker_pixels=positions),indent=2)+"\n")
    return result


def wrapped(draw, text, xy, width, size, *, color=INK, bold=False, leading=1.25):
    x,y = xy; words = text.split(); lines=[]; line=""
    f=font(size,bold)
    for word in words:
        candidate=(line+" "+word).strip()
        if draw.textlength(candidate,font=f)>width and line:
            lines.append(line);line=word
        else:line=candidate
    if line:lines.append(line)
    for line in lines:
        draw.text((x,y),line,font=f,fill=color)
        y += round(size*leading)
    return y


def counts(ids):
    return Counter(s["type"] for pid in ids for s in PARTS[pid]["sites"])


def card_image(c, width, height):
    canvas = Image.new("RGB",(width,height),PAPER); d=ImageDraw.Draw(canvas)
    pad=24
    title_bottom=wrapped(d,c["title"],(pad,18),width-2*pad,29,bold=True)
    if c["subtitle"]:
        wrapped(d,c["subtitle"],(pad,title_bottom+5),width-2*pad,22,color=MUTED)
    image_top=112
    image_bottom=height-148
    cols=c["columns"]; rows=math.ceil(len(c["parts"])/cols)
    pw=(width-2*pad)//cols; ph=(image_bottom-image_top)//rows
    for index,pid in enumerate(c["parts"]):
        part=PARTS[pid]; x=pad+(index%cols)*pw; y=image_top+(index//cols)*ph
        im=render(part,pw-10,ph-36,c["view"])
        canvas.paste(im,(x,y))
        label=part["label"]
        size=21
        while d.textlength(label,font=font(size))>pw-12 and size>15: size-=1
        d.text((x+(pw-10)/2,y+ph-28),label,font=font(size),fill=MUTED,anchor="mt")
    d.line((pad,height-140,width-pad,height-140),fill=LINE,width=2)
    tally=counts(c["parts"])
    x=pad; y=height-127
    if tally:
        for kind,n in sorted(tally.items()):
            text=f"{n} × {kind}"
            d.ellipse((x,y+2,x+23,y+25),fill=MAGNETS[kind]["color"])
            d.text((x+32,y),text,font=font(24,True),fill=MAGNETS[kind]["color"])
            x+=d.textlength(text,font=font(24,True))+67
        if len(c["parts"])>1:
            suffix="shown" if any(PARTS[p]["state"] in ["floor_stand","no_floor_stand"] for p in c["parts"]) else "in shown set"
            d.text((x,y+2),suffix,font=font(21),fill=MUTED)
    else:
        d.text((x,y),"No magnets",font=font(24,True),fill=MUTED)
    bottom=wrapped(d,c["note"],(pad,height-88),width-2*pad,23,color=MUTED,leading=1.18)
    assert bottom<=height-6,(c["id"],bottom,height)
    return canvas


def assembly_maps():
    scenes=[]
    for family, perimeter in [("stock","shoulders"),("slim","wings")]:
        base=[e["name"] for e in SHELF if e["family"]==family and e["selection"]=="core" and e["state"]!="floor_stand"]
        ids=base+CARDS[f"{family}_{perimeter}"]["parts"]
        scenes.append((family, f"{family.title()} + {'A shoulders' if perimeter=='shoulders' else 'B1 wings'}",ids,
                       "4 paired D5 joints · 8 magnets total"))
    lm=CARDS["obi_lower"]["parts"][:1]+CARDS["obi_top"]["parts"]
    lower=CARDS["wings_graded_lm_lower"]["parts"]
    scenes.append(("obiwan", "Regular Obi-Wan + graded wings",lm+CARDS["obi_um"]["parts"]+CARDS["obi_t"]["parts"]+
                   lower+CARDS["wings_graded_lm_um_upper"]["parts"], "6 paired D5 joints · 12 magnets total"))
    scenes.append(("v4", "Dayton ND25FN-4 + matching graded wings",lm+["v4_body"]+lower+CARDS["v4_graded_wings"]["parts"],
                   "4 paired D5 + 4 paired D6 · 16 magnets total"))
    result=[]
    for family,title,ids,caption in scenes:
        components=[dict(PARTS[k],assembly_color="#b7c3ca" if "wing" in PARTS[k]["path"].lower() or "shoulder" in PARTS[k]["path"] else COLORS[family]) for k in ids]
        remaining=[s for p in components for s in p["sites"]]; paired=[]
        while remaining:
            a=remaining.pop(0)
            candidates=[(float(np.linalg.norm(np.asarray(a["center_mm"])-b["center_mm"])),i) for i,b in enumerate(remaining) if a["type"]==b["type"]]
            assert candidates,(family,a)
            distance,i=min(candidates);assert distance<8,(family,distance)
            b=remaining.pop(i)
            paired.append(dict(a,center_mm=((np.asarray(a["center_mm"])+b["center_mm"])/2).tolist(),name=a["name"]+" <> "+b["name"]))
        expected=4 if family in ["stock","slim"] else 6 if family=="obiwan" else 8
        assert len(paired)==expected,(family,len(paired))
        result.append(dict(id=f"assembly_{family}",title=title,caption=caption,components=components,
                           sites=paired,paired_markers=True,path="assembled from catalog components",
                           sha256=hashlib.sha256("".join(p["sha256"] for p in components).encode()).hexdigest(),family=family))
    return result


def poster(sections, assemblies):
    width=4200; margin=75; gap=20
    header=505; footer=480; overview=730
    height=header+overview+sum(130+math.ceil(len(s["cards"])/s["columns"])*(s["height"]+gap)+24 for s in sections)+footer
    canvas=Image.new("RGB",(width,height),BG); d=ImageDraw.Draw(canvas)
    d.rectangle((0,0,width,header-25),fill=INK)
    d.text((margin,40),"LX521.4  /  ALL PARTS & MAGNETS",font=font(78,True),fill="white")
    d.text((margin,142),"STOCK  ·  SLIM  ·  OBI-WAN  ·  DAYTON ND25FN-4  ·  TWEETER ALTERNATIVES  ·  TEST PIECES",font=font(31),fill="#c3d6df")
    d.text((margin,201),"Earlier P2S parts · labels updated 15 September 2026 · 78 individual part variants · magnify to inspect",font=font(31),fill="white")
    for x,kind,desc in [(margin,"D5","Regular Stock / Slim / Obi-Wan; BMR side stations"), (width//2+50,"D6","Dayton ND25FN-4: fused UM + matching upper-wing stations")]:
        d.ellipse((x,281,x+65,346),fill=MAGNETS[kind]["color"])
        d.text((x+32,313),kind[-1],font=font(42,True),fill="white",anchor="mm")
        d.text((x+86,280),f"{kind}  {MAGNETS[kind]['label']}  /  {MAGNETS[kind]['supplier']}",font=font(35,True),fill="white")
        d.text((x+86,330),desc,font=font(28),fill="#c3d6df")
    d.text((margin,408),"Dots reveal buried locations through the plastic; enlarged for clarity, not surface holes. Filled dot = one magnet.",font=font(31),fill="white")
    y=header
    d.text((margin,y),"ASSEMBLED LOCATION MAPS",font=font(43,True),fill=INK)
    d.text((margin,y+59),"Examples with no floor stand. Grey = optional perimeter. Outlined dot = one mating PAIR (two buried magnets).",font=font(29),fill=MUTED)
    cw=(width-2*margin-3*gap)//4
    for i,assembly in enumerate(assemblies):
        x=margin+i*(cw+gap)
        d.rectangle((x,y+114,x+cw,y+overview-27),fill="white")
        wrapped(d,assembly["title"],(x+24,y+134),cw-48,29,bold=True)
        im=render(assembly,cw-30,453)
        canvas.paste(im,(x+15,y+192))
        d.text((x+24,y+overview-65),assembly["caption"],font=font(26,True),fill=INK)
    y+=overview
    for section in sections:
        d.rectangle((margin,y+4,margin+12,y+88),fill=section["color"])
        d.text((margin+30,y),section["title"],font=font(47,True),fill=INK)
        wrapped(d,section["description"],(margin+30,y+66),width-2*margin-30,29,color=MUTED)
        y+=130
        cols=section["columns"]; cw=(width-2*margin-(cols-1)*gap)//cols
        for i,key in enumerate(section["cards"]):
            im=card_image(CARDS[key],cw,section["height"])
            canvas.paste(im,(margin+(i%cols)*(cw+gap),y+(i//cols)*(section["height"]+gap)))
        y+=math.ceil(len(section["cards"])/cols)*(section["height"]+gap)+24
    d.rectangle((margin,y+10,width-margin,y+13),fill=LINE)
    y+=46
    d.text((margin,y),"ASSEMBLY COUNTS  /  one speaker with one complete wing or shoulder set",font=font(34,True),fill=INK)
    y+=55
    d.text((margin,y),"Stock or Slim: 8 D5     ·     Regular Obi-Wan: 12 D5     ·     Fused Dayton ND25FN-4: 8 D5 + 8 D6",font=font(34,True),fill=INK)
    y+=61
    y=wrapped(d,"These totals exclude alternatives and coupons. Each magnet pair has one magnet in the body and one in the attachment. Magnets align and retain optional wings; screws carry the structural joints. Insert at the verified print pause, check attraction, then bury under the printed roof.",(margin,y),width-2*margin,29,color=MUTED)+18
    y=wrapped(d,"Stock + Slim mixed-thickness alternatives: Stock lower/mids + Slim vase, or Slim lower/mids + Stock vase. Use perimeter parts matching the vase; the hidden rear face has a 6.8 mm step. Obi-Wan does not mix with Stock/Slim. The ND25FN-4 fused body needs its matching upper wings; its lower wings and LM remain common.",(margin,y),width-2*margin,27,color=MUTED)+15
    y=wrapped(d,"Scope: earlier P2S product meshes, STEP-only BMR-slim alternatives, monolithic LM options and diagnostic fixtures. The 4 Obi-Wan combo plates only regroup the pictured parts. Stand-duplicate copies, review assemblies, support blockers and retired C7 / V0 / three-piece wings are not extra products. Views fit each part to its frame; size is not comparable across cards.",(margin,y),width-2*margin,25,color=MUTED)+14
    y=wrapped(d,"Sources: to_print/catalog.json · review/captive_magnet_release_catalog.json · candidate facts / manifests · current STL sidecars. Dot positions come from declared seats or enclosed coupon cavities. CAD-only is not print-ready; physical fit, retention and acoustic performance remain to qualify. Generator and file map: scripts/generate_parts_magnet_poster.py / images/generated/catalog/parts_catalog.json.",(margin,y),width-2*margin,24,color=MUTED)
    assert y<height,(y,height)
    path=OUT/"ALL_ITEMS_MAGNET_CATALOG.png"
    canvas.save(path,optimize=True,dpi=(200,200))
    preview=canvas.copy(); preview.thumbnail((1400,3000));preview.save(OUT/"ALL_ITEMS_MAGNET_CATALOG_preview.png",optimize=True)
    return path,canvas.size


def validate(sections):
    by_path={str((ROOT/p["path"]).resolve()) for p in PARTS.values()}
    covered=[]; combos=[]
    for entry in SHELF:
        if entry.get("composite_plate"):
            combos.append(entry["name"]);continue
        source=canonical_source(ROOT,entry)
        assert str(source.resolve()) in by_path,(entry["name"],source)
        covered.append(entry["name"])
    shown=[p for c in CARDS.values() for p in c["parts"]]
    assert set(shown)==set(PARTS) and len(shown)==len(set(shown)),"Missing or duplicated part card"
    assert len(PARTS)==78,len(PARTS)
    assert len(PARTS["v4_body"]["sites"])==4
    for style in ["flat","graded"]:
        for side in ["left","right"]:
            assert counts([f"v4_{style}_{side}"])==Counter(D6=2,D5=1)
    assert counts(["d5_coupon_carrier","d5_coupon_ae_wing"])==Counter(D5=4)
    assert counts(["01_D6_body_coupon","D6_wing_coupon"])==Counter(D6=2)
    return dict(status="passed",individual_part_variants=len(PARTS),shelf_choices=len(SHELF),
                shelf_part_choices_covered=covered,composite_plate_choices_accounted_for=combos,
                scope="Earlier P2S product and diagnostic identities; identical stand copies and plate layouts are not extra part variants.",
                required_meshes=len(PARTS),magnet_position_authorities="Bound per part/site; source bytes hashed.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory-only",action="store_true")
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    SOURCES["to_print/catalog.json"]=sha(ROOT/"to_print/catalog.json")
    sections=inventory()
    proof=validate(sections)
    assemblies=assembly_maps()
    sources=dict(SOURCES)
    sources["scripts/generate_parts_magnet_poster.py"]=sha(__file__)
    for path in [V4/"print_magnets.py",V4/"v4_model.py"]:
        sources[str(path.relative_to(ROOT))]=sha(path)
    pairing_maps=[dict(id=a["id"],title=a["title"],parts=[p["id"] for p in a["components"]],pairs=a["sites"],
                       pair_counts=dict(Counter(s["type"] for s in a["sites"]))) for a in assemblies]
    data=dict(schema_version=1,date="2026-09-15",magnet_types=MAGNETS,parts=list(PARTS.values()),cards=list(CARDS.values()),
              sections=sections,assembly_maps=pairing_maps,source_sha256=sources,coverage=proof)
    (OUT/"parts_catalog.json").write_text(json.dumps(data,indent=2)+"\n")
    if args.inventory_only:
        print(json.dumps(proof,indent=2));return
    path,size=poster(sections,assemblies)
    proof.update(png=str(path.relative_to(ROOT)),png_sha256=sha(path),pixels=size,
                 marker_count=sum(len(p["sites"]) for p in PARTS.values()),
                 assembly_pair_counts={a["id"]:a["pair_counts"] for a in pairing_maps},visual_review="pending")
    (OUT/"validation.json").write_text(json.dumps(proof,indent=2)+"\n")
    print(json.dumps(proof,indent=2))


if __name__=="__main__":
    main()
