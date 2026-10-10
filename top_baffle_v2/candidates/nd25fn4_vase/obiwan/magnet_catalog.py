"""Captive-magnet catalog for the Obiwan ND25FN-4 vase top and its wings.

The vase top moves the two LM-upper wing magnets from 64/116 deg to 9/171 deg on the LM ring, and the wings move
their matching magnet by the same rigid rotation about the LM centre. Every other magnet keeps its released site. The
sites here are therefore the released catalog records, rotated where they moved, and each one is checked against
the actual buried cavity in the packaged STL.
usage: magnet_catalog.py  (run from the repository root)
"""
import json, math
from pathlib import Path
import numpy as np, trimesh

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
LM = np.array([0.0, 200.981])
MOVE = {"lm_upper_left": 55.0, "lm_upper_right": -55.0}             # 116 -> 171 and 64 -> 9 deg (V13: clear of the released lumens)
released = json.loads((ROOT / "review/captive_magnet_release_catalog.json").read_text())["artifacts"]

def site(part, name, state="shared"):
    rows = [s for a in released if a["part"] == part and a["state"] == state for s in a["sites"] if s["name"] == name]
    assert len(rows) == 1, (part, name, len(rows)); return dict(rows[0])

def rotated(s):
    a = math.radians(MOVE.get(s["name"], 0.0)); c, n = math.cos(a), math.sin(a); R = np.array([[c, -n], [n, c]])
    out = {k: v for k, v in s.items() if k != "print_space"}
    for k, v in s.items():
        if k == "bed_datum_source_xyz" or not isinstance(v, list) or len(v) not in (2, 3): continue
        if k.endswith("_xyz_mm") or k.endswith("_xy_mm"):
            out[k] = [*(LM + R @ (np.array(v[:2]) - LM)).tolist(), *v[2:]]
        elif k.endswith("_xyz"):
            out[k] = [*(R @ np.array(v[:2])).tolist(), v[2]]
    if s["name"] in MOVE: out["relocated_from_released_deg"] = MOVE[s["name"]]
    return out

parts = {}
for hand in ("right", "left"):                                   # both hands share the LM magnet sites: the ring and the 9/171 deg stations are symmetric
    P_ = f"nd25fn4_vase_obiwan_tweeters_{hand}"
    parts[f"{P_}_top"] = [site("obiwan_optional_lm_keyed_2_of_2_top", n, "floor_stand") for n in ("lm_upper_left", "lm_upper_right")]
    for kind in ("flat", "graded"):
        for side in ("left", "right"):
            lower = site(f"obiwan_wing_{kind}_{side}_split2_1_of_2_lm_lower", f"lm_lower_{side}")
            upper = site(f"obiwan_wing_{kind}_{side}_split2_2_of_2_lm_um_upper", f"lm_upper_{side}")
            parts[f"{P_}_wing_{kind}_{side}"] = [lower, upper]

artifacts = []
for part, sites in parts.items():
    sites = [rotated(s) for s in sites]
    stl = HERE / "STL" / f"{part}.stl"; auth = json.loads(stl.with_suffix(".print.json").read_text())
    m = trimesh.load_mesh(stl, process=True); m.apply_transform(np.linalg.inv(np.asarray(auth["source_to_stl_matrix"])))
    cav = [c.center_mass for c in m.split(only_watertight=False) if c.volume < 0]
    assert len(cav) == len(sites), (part, len(cav), len(sites))
    for s in sites:
        c = min(cav, key=lambda c: np.linalg.norm(np.asarray(s["cavity_center_xyz_mm"]) - c)); d = c - np.asarray(s["cavity_center_xyz_mm"]); err = float(np.linalg.norm(d[:2]))
        print(f"{part} {s['name']}: cavity centroid - catalog centre: xy {err:.3f} mm, z {d[2]:.3f} mm (gable roof)"); assert err < 0.1 and abs(d[2]) < 1.5
    artifacts.append(dict(id=f"shared:nd25fn4_vase:{part}", part=part, state="shared", stl=f"STL/{part}.stl", stl_sha256=auth["stl_sha256"], sites=sites))
(HERE / "magnets.catalog.json").write_text(json.dumps(dict(schema_version=1, catalog_kind="captive_magnet_sites",
    generated_by="candidates/nd25fn4_vase/obiwan/magnet_catalog.py", derived_from="review/captive_magnet_release_catalog.json",
    artifacts=artifacts), indent=1) + "\n")
