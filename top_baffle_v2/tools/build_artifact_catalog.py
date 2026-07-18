#!/usr/bin/env python3
"""Build the product-oriented artifact catalog without duplicating CAD bytes.

The authoritative generators still write the state-oriented build trees used by
the validation pipeline.  This tool creates stable relative symlinks under
``artifacts/`` and deterministic SHA-256 manifests for the three products a
human actually chooses: standard, slim, and Obi-Wan.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = ROOT / "artifacts"


@dataclass(frozen=True)
class Link:
    path: str
    source: str
    role: str


def _latest(pattern: str) -> str:
    matches = sorted(ROOT.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"no generated image matches {pattern!r}")
    return matches[-1].relative_to(ROOT).as_posix()


def _print_pair(destination_root: str, source_root: str, stem: str) -> list[Link]:
    return [
        Link(f"{destination_root}/{stem}.stl", f"{source_root}/{stem}.stl", "print_mesh"),
        Link(
            f"{destination_root}/{stem}.print.json",
            f"{source_root}/{stem}.print.json",
            "print_orientation_authority",
        ),
    ]


def _standard_links() -> list[Link]:
    state = "no_floor_stand"
    stl_source = f"{state}/stl"
    links = [
        Link("cad/base.step", f"{state}/top_baffle_nd25fw4_b2.step", "canonical_design"),
        Link(
            "cad/base_print_assembly.step",
            f"{state}/top_baffle_nd25fw4_b2_split.step",
            "print_assembly",
        ),
        Link(
            "cad/shoulders_assembly.step",
            f"{state}/top_baffle_nd25fw4_a_comp_assembled.step",
            "optional_shoulders_assembly",
        ),
        Link(
            "cad/wings_assembly.step",
            f"{state}/top_baffle_nd25fw4_b1_assembled.step",
            "optional_wings_assembly",
        ),
        Link("cad/attachments.step", "top_baffle_nd25fw4_attachments.step", "attachment_set"),
        Link("images/plan.png", f"{state}/baffle_variants_drivers.png", "generated_plan"),
        Link("images/routing.png", f"{state}/baffle_cable_routing_proud.png", "generated_routing"),
    ]
    for view in ("base_iso", "base_iso_opposite", "base_top", "base_front"):
        links.append(
            Link(
                f"images/{view.removeprefix('base_')}.png",
                _latest(f"images/generated/standard/{view}_*.png"),
                "cad_snapshot",
            )
        )
    stems = [
        "lx521_top_base_1of4_bottom",
        "lx521_top_base_2of4_mid_left",
        "lx521_top_base_3of4_mid_right",
        "lx521_top_base_4of4_vase_b2",
        "lx521_top_addonA_1of4_shoulder_top_left",
        "lx521_top_addonA_2of4_shoulder_top_right",
        "lx521_top_addonA_3of4_shoulder_bottom_left",
        "lx521_top_addonA_4of4_shoulder_bottom_right",
        "lx521_top_addonB1_1of2_wing_left",
        "lx521_top_addonB1_2of2_wing_right",
        "lx521_top_proud_addon_um_grommet_half_a",
        "lx521_top_proud_addon_um_grommet_half_b",
    ]
    for stem in stems:
        links.extend(_print_pair("stl", stl_source, stem))
    return links


def _slim_links() -> list[Link]:
    state = "no_floor_stand"
    stl_source = f"{state}/stl"
    links = [
        Link(
            "cad/base_print_assembly.step",
            f"{state}/top_baffle_nd25fw4_v1l_split.step",
            "print_assembly",
        ),
        Link("cad/top.step", f"{state}/top_baffle_nd25fw4_v1.step", "thin_top_design"),
        Link(
            "cad/attachments.step",
            f"{state}/top_baffle_nd25fw4_v1_attachments.step",
            "thin_attachment_set",
        ),
    ]
    for view in ("base_iso", "base_iso_opposite", "base_top", "base_front"):
        links.append(
            Link(
                f"images/{view.removeprefix('base_')}.png",
                _latest(f"images/generated/slim/{view}_*.png"),
                "cad_snapshot",
            )
        )
    stems = [
        "lx521_top_v1l_1of4_bottom",
        "lx521_top_v1l_2of4_mid_left",
        "lx521_top_v1l_3of4_mid_right",
        "lx521_top_v1l_4of4_vase_b2",
        "lx521_top_v1addonA_shoulder_top_left",
        "lx521_top_v1addonA_shoulder_top_right",
        "lx521_top_v1addonA_shoulder_bottom_left",
        "lx521_top_v1addonA_shoulder_bottom_right",
        "lx521_top_v1addonB1_wing_left",
        "lx521_top_v1addonB1_wing_right",
        "lx521_top_v1l_addon_um_grommet_half_a",
        "lx521_top_v1l_addon_um_grommet_half_b",
    ]
    for stem in stems:
        links.extend(_print_pair("stl", stl_source, stem))
    return links


def _obiwan_state_links(state: str, destination: str) -> list[Link]:
    source_stl = f"{state}/stl"
    links = [
        Link(f"states/{destination}/cad/core.step", f"{state}/top_baffle_nd25fw4_obiwan_split.step", "core_assembly"),
        Link(
            f"states/{destination}/cad/lm_keyed_split.step",
            f"{state}/top_baffle_nd25fw4_obiwan_lm_split.step",
            "optional_lm_print_split",
        ),
        Link(
            f"states/{destination}/cad/attachments.step",
            f"{state}/top_baffle_nd25fw4_obiwan_attachments.step",
            "optional_tweeter_attachment",
        ),
        Link(
            f"states/{destination}/cad/review_assembly.step",
            f"{state}/top_baffle_nd25fw4_obiwan_assembled.step",
            "non_manufacturing_review_assembly",
        ),
        Link(
            f"states/{destination}/manifest.json",
            f"{state}/obiwan_release_manifest.json",
            "candidate_manifest",
        ),
        Link(
            f"states/{destination}/images/routing.png",
            f"{state}/baffle_cable_routing_obiwan.png",
            "generated_routing",
        ),
    ]
    image_state = "no_floor" if state == "no_floor_stand" else "floor"
    for view in ("core_iso", "core_iso_opposite", "core_top", "core_front"):
        links.append(
            Link(
                f"states/{destination}/images/{view.removeprefix('core_')}.png",
                _latest(f"images/generated/obiwan/{image_state}/{view}_*.png"),
                "cad_snapshot",
            )
        )
    stems = [
        "lx521_top_obiwan_core_1of2_lm_carrier",
        "lx521_top_obiwan_core_2of2_um_carrier",
        "lx521_top_obiwan_optional_lm_keyed_1of2_bottom",
        "lx521_top_obiwan_optional_lm_keyed_2of2_top",
        "lx521_top_obiwan_addon_tweeter_crescent",
    ]
    for stem in stems:
        links.extend(_print_pair(f"states/{destination}/stl", source_stl, stem))
    if state == "floor_stand":
        links.extend(
            [
                Link(
                    "states/floor/qualification/integrated_floor_strength.json",
                    "floor_stand/obiwan_integrated_floor_strength.json",
                    "analytical_screen",
                ),
                Link(
                    "states/floor/qualification/integrated_floor_strength.md",
                    "floor_stand/obiwan_integrated_floor_strength.md",
                    "analytical_screen_report",
                ),
            ]
        )
    return links


def _obiwan_wing_links(slug: str) -> list[Link]:
    source = f"wings/{slug}"
    destination = f"wings/{slug}"
    links = [
        Link(
            f"{destination}/cad/monolithic_pair.step",
            f"{source}/top_baffle_nd25fw4_obiwan_wing_{slug}.step",
            "canonical_wing_pair",
        ),
        Link(
            f"{destination}/cad/print_assembly.step",
            f"{source}/top_baffle_nd25fw4_obiwan_wing_{slug}_assembled.step",
            "six_piece_print_assembly",
        ),
        Link(
            f"{destination}/facts.json",
            f"{source}/obiwan_wing_{slug}_facts.json",
            "geometry_facts",
        ),
        Link(
            f"{destination}/manifest.json",
            f"{source}/obiwan_wing_{slug}_print_manifest.json",
            "wing_manifest",
        ),
    ]
    for image_name in ("front", "rear", "side_section", "split_exploded", "magnet_roots"):
        links.append(
            Link(
                f"{destination}/images/{image_name}.png",
                f"{source}/review/obiwan_wing_{slug}_{image_name}.png",
                "generated_wing_review",
            )
        )
    for side in ("left", "right"):
        for order, role in ((1, "lm_lower"), (2, "lm_upper"), (3, "um")):
            stem = f"lx521_top_obiwan_wing_{slug}_{side}_{order}of3_{role}"
            links.extend(_print_pair(f"{destination}/stl", f"{source}/stl", stem))
    return links


def _obiwan_links() -> list[Link]:
    links = [
        Link("images/wing_design_map.png", "obiwan_wing_design_map.png", "generated_wing_design_map"),
        Link("qualification.md", "obiwan_physical_qualification.md", "physical_qualification_template"),
    ]
    links.extend(_obiwan_state_links("no_floor_stand", "no_floor"))
    links.extend(_obiwan_state_links("floor_stand", "floor"))
    links.extend(_obiwan_wing_links("ac"))
    links.extend(_obiwan_wing_links("ae"))
    return links


PRODUCTS = {
    "standard": {
        "title": "Standard R6P baffle",
        "status": "canonical_cad",
        "release_authorized": None,
        "intent": "Full-depth B2 base with mutually exclusive A-comp shoulders or B1 wings.",
        "dimensions": {
            "nominal_depth_mm": 18.3,
            "width_mm": 304.802,
            "height_mm": 453.457,
        },
        "links": _standard_links,
    },
    "slim": {
        "title": "Slim R6P baffle",
        "status": "experimental_cad",
        "release_authorized": False,
        "intent": "V1L lower/mids plus V1 top with an 11.5 mm front-flush acoustic field and matching thin attachments.",
        "dimensions": {
            "acoustic_field_depth_mm": 11.5,
            "rear_offset_mm": 6.8,
            "structural_bottom_strip_depth_mm": 18.3,
        },
        "links": _slim_links,
    },
    "obiwan": {
        "title": "Obi-Wan R6F collar baffle",
        "status": "candidate_not_release_authorized",
        "release_authorized": False,
        "intent": "Two mandatory flush carriers, optional tweeter crescent, and Ac/Ae magnetic acoustic wings in floor and stock-bridge states.",
        "dimensions": {
            "carrier_field_depth_mm": 11.5,
            "front_plane_z_mm": 18.3,
            "lm_um_axis_spacing_mm": 165.1,
        },
        "links": _obiwan_links,
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload(product_id: str, metadata: dict, links: Iterable[Link]) -> dict:
    records = []
    for link in sorted(links, key=lambda item: item.path):
        source = ROOT / link.source
        if not source.is_file():
            raise FileNotFoundError(f"missing catalog source: {link.source}")
        records.append(
            {
                "path": link.path,
                "role": link.role,
                "source": link.source,
                "bytes": source.stat().st_size,
                "sha256": _sha256(source),
            }
        )
    return {
        "schema_version": 1,
        "product": product_id,
        "title": metadata["title"],
        "status": metadata["status"],
        "release_authorized": metadata["release_authorized"],
        "intent": metadata["intent"],
        "dimensions_mm": metadata["dimensions"],
        "files": records,
    }


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _write_or_check(path: Path, content: bytes, check: bool) -> None:
    if check:
        if not path.is_file() or path.read_bytes() != content:
            raise RuntimeError(f"stale or missing generated catalog file: {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _sync_link(product_id: str, link: Link, check: bool) -> Path:
    source = ROOT / link.source
    destination = ARTIFACT_ROOT / product_id / link.path
    relative_target = os.path.relpath(source, destination.parent)
    if check:
        if not destination.is_symlink():
            raise RuntimeError(f"missing artifact link: {destination.relative_to(ROOT)}")
        if os.readlink(destination) != relative_target:
            raise RuntimeError(f"stale artifact link: {destination.relative_to(ROOT)}")
        if destination.resolve() != source.resolve():
            raise RuntimeError(f"artifact link resolves incorrectly: {destination.relative_to(ROOT)}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        raise RuntimeError(f"refusing to replace non-symlink artifact: {destination}")
    destination.symlink_to(relative_target)
    return destination


def build(check: bool) -> None:
    expected_links: set[Path] = set()
    manifest_summaries = []
    for product_id, metadata in PRODUCTS.items():
        links = metadata["links"]()
        payload = _payload(product_id, metadata, links)
        for link in links:
            expected_links.add(_sync_link(product_id, link, check))
        manifest_path = ARTIFACT_ROOT / product_id / "manifest.json"
        manifest_bytes = _json_bytes(payload)
        _write_or_check(manifest_path, manifest_bytes, check)
        manifest_summaries.append(
            {
                "product": product_id,
                "manifest": f"{product_id}/manifest.json",
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "file_count": len(payload["files"]),
                "status": payload["status"],
                "release_authorized": payload["release_authorized"],
            }
        )

    catalog_payload = {"schema_version": 1, "products": manifest_summaries}
    _write_or_check(ARTIFACT_ROOT / "catalog.json", _json_bytes(catalog_payload), check)

    if not check and ARTIFACT_ROOT.exists():
        for path in ARTIFACT_ROOT.rglob("*"):
            if path.is_symlink() and path not in expected_links:
                path.unlink()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify links and manifests without writing")
    args = parser.parse_args()
    build(check=args.check)
    print("artifact catalog is current" if args.check else "artifact catalog rebuilt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
