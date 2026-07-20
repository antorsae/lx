"""Contracts for the Obi-Wan full-depth LM--UM and T--UM closure webs.

The cheap source test runs without importing OCC.  Analytic/BREP checks are
guarded because released Obi-Wan carrier construction is remote-only.  Run the
complete file through the authenticated osado CAD test path.
"""

from __future__ import annotations

import ast
from functools import lru_cache
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CORE_SOURCE = ROOT / "top_baffle_nd25fw4_obiwan.py"
ADDON_SOURCE = ROOT / "top_baffle_nd25fw4_obiwan_attachments.py"

# Frozen 0.12-mm-inset physical closure silhouettes.  These coordinates were
# captured from the approved released ring/crescent/ear geometry, then made
# deliberately conservative by the inset and simplification.  They are test
# data, not generated from ``junction_closure_polygons`` at test time: an
# implementation that shrinks its own target therefore cannot shrink the
# required front material with it.  The exact 0.05-mm fit seams and functional
# receiver/route clearances are subtracted explicitly during each Z audit.
FROZEN_REQUIRED_FRONT_WKT = {
    "lm_um": (
        "POLYGON ((-27.810339 311.159780, -32.456234 311.566523, "
        "-34.254250 312.202471, -35.286103 312.950552, -35.980000 "
        "313.945133, -35.980000 317.595112, -35.285754 318.594508, "
        "-33.845477 319.548535, -31.946290 320.088096, -29.139922 "
        "320.256917, -26.065629 319.969761, -25.879414 321.186986, "
        "-20.415990 318.452232, -14.198583 316.244144, -7.518376 "
        "314.809312, -0.834300 314.267720, 6.681174 314.693512, "
        "13.363315 316.013707, 19.964401 318.261930, 25.879414 "
        "321.186986, 26.077174 319.967437, 30.304450 320.250772, "
        "33.408424 319.720429, 34.975794 318.874479, 35.980000 "
        "317.595112, 35.980000 313.945133, 34.976258 312.673321, "
        "33.845870 312.006770, 31.945520 311.465232, 27.899305 "
        "311.271695, 28.095431 310.554282, 22.067895 311.927572, "
        "15.015487 313.099996, 8.342625 313.792944, 1.667994 "
        "314.088701, -5.839971 313.950150, -13.347050 313.310829, "
        "-20.706196 312.189756, -28.095431 310.554282, -27.810339 "
        "311.159780))"
    ),
    "t_um": (
        "MULTIPOLYGON (((6.606206 417.477990, 14.449415 419.145765, "
        "22.570824 423.238596, 24.962560 424.159273, 26.842485 "
        "424.283508, 27.980000 423.323733, 27.980000 419.674432, "
        "27.097290 418.444764, 25.717381 417.410564, 23.488585 "
        "416.432018, 20.925309 415.809411, 18.222820 415.533264, "
        "18.222193 415.316308, 18.622606 415.320634, 18.523383 "
        "414.476550, 12.695680 416.321741, 6.606206 417.477990)), "
        "((-18.523383 414.476550, -18.622606 415.320634, -18.222193 "
        "415.316308, -18.222820 415.533264, -20.925309 415.809411, "
        "-23.488585 416.432018, -25.717381 417.410564, -27.097290 "
        "418.444764, -27.980000 419.674432, -27.980000 423.323733, "
        "-27.623596 423.852122, -26.520828 424.337550, -24.516282 "
        "424.027913, -14.449415 419.145765, -6.606206 417.477990, "
        "-12.695680 416.321741, -18.523383 414.476550)))"
    ),
}


def _guarded_build() -> bool:
    try:
        import run_memory_guarded as memory_guard
        return bool(memory_guard.is_guarded_process())
    except Exception:
        return False


FULL_TEST_ENV = "LX_OBIWAN_CLOSURE_FULL_TEST"
PLAN_ONLY_ENV = "LX_OBIWAN_CLOSURE_PLAN_ONLY"
BASE_ONLY_ENV = "LX_OBIWAN_CLOSURE_BASE_ONLY"
DENSE_CASE_ENV = "LX_OBIWAN_CLOSURE_DENSE_CASE"
DENSE_SHARD_ENV = "LX_OBIWAN_CLOSURE_DENSE_SHARD"
DENSE_SHARD_COUNT = 4
DENSE_CASES = {
    "no_floor_lm_um": ("no_floor", "lm_um"),
    "floor_lm_um": ("floor", "lm_um"),
    "no_floor_t_um": ("no_floor", "t_um"),
    "floor_t_um": ("floor", "t_um"),
}


def _require_guarded_test() -> None:
    assert _guarded_build(), (
        "Obi-Wan analytic/BREP acceptance requires guarded remote CAD")


def _function_call_names(source: str, function_name: str) -> list[str]:
    tree = ast.parse(source)
    function = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == function_name
    )
    names = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.append(node.func.attr)
    return names


def test_source_owns_full_depth_webs_before_functional_recuts() -> None:
    """Every printable owner explicitly receives its complementary web."""
    core = CORE_SOURCE.read_text(encoding="utf-8")
    addon = ADDON_SOURCE.read_text(encoding="utf-8")
    v1 = (ROOT / "top_baffle_nd25fw4_v1.py").read_text(encoding="utf-8")
    ast.parse(core)
    ast.parse(addon)
    ast.parse(v1)

    assert 'PRINT_ORIENTATION = "front-face-down"' in core
    assert 'PRINT_ORIENTATION = "front-face-down"' in addon
    assert "def v1_magnet_free_solid():" in v1
    assert ("return apply_v1_base_magnets(v1_magnet_free_solid())"
            in v1)
    assert "raw = v1_magnet_free_solid()" in addon
    assert "raw = v1_solid()" not in addon
    magnet_land_helper = core[
        core.index("def _verify_side_magnet_lands"):
        core.index("def joint_load_facts")]
    assert "required_land - part" in magnet_land_helper
    assert "_ensure_shell_contained" not in magnet_land_helper
    assert "_add_side_magnet_ears" not in core
    assert "JUNCTION_WEB_Z = (CORE_REAR_Z, THICKNESS_MM)" in core
    assert ("TWEETER_CORE_BORE_TOP_Z = "
            "TWEETER_CORE_JOINT_Z[1] + 0.35") in core
    assert core.count("TWEETER_CORE_BORE_TOP_Z)") == 2
    assert addon.count("TWEETER_CORE_BORE_TOP_Z)") == 0
    t_helper = core[core.index("def _apply_complete_um_tweeter_joint"):
                    core.index("def _ensure_shell_contained")]
    assert t_helper.index("TWEETER_JOINT_HOLE_D / 2.0") < (
        t_helper.index("TWEETER_JOINT_INSERT_BORE_D / 2.0"))
    assert "TWEETER_CORE_JOINT_Z[1] + 0.3)" not in core
    assert "JUNCTION_WEB_LENS_FUSION_MM = 0.45" in core
    assert "JUNCTION_WEB_MIN_LENS_AREA_MM2 = 0.05" in core
    assert "LM_T_CLOSURE_HANDOFF_RELIEF_MM = 0.0" in core
    assert "LM_UM_REAR_BACKFILL_Z = (" in core
    assert "UM_T_REAR_BACKFILL_Z = (" in core
    assert "_printable_lens_components" in core
    assert '_junction_closure_web("lm_um", "lm")' in core
    assert '_junction_closure_web("lm_um", "um")' in core
    assert '_junction_closure_web("t_um", "um")' in core
    assert '_junction_closure_web("t_um", "tweeter")' in addon
    assert '_enforce_junction_plan_ownership(part, "lm_um", "lm")' in core
    assert '_enforce_junction_plan_ownership(part, "lm_um", "um")' in core
    assert '_enforce_junction_plan_ownership(part, "t_um", "um")' in core
    assert ('_enforce_junction_plan_ownership(part, "t_um", "tweeter")'
            in addon)
    assert "JOINT_CLEARANCE_BORE_D = 3.4" in core
    assert "JOINT_INSERT_BORE_D = 4.6" in core
    assert "JOINT_INSERT_DEPTH_MM = 4.0" in core
    assert "JOINT_FUNCTIONAL_BOSS_D = 9.8" in core
    assert "TWEETER_JOINT_INSERT_DEPTH_MM = 4.0" in core
    assert "TWEETER_JOINT_FUNCTIONAL_BOSS_D = 9.8" in core
    assert 'part = _apply_complete_lm_um_joint(part, "lm")' in core
    assert 'part = _apply_complete_lm_um_joint(part, "um")' in core
    assert 'part = _apply_complete_um_tweeter_joint(part, "um")' in core
    assert ('part = _apply_complete_um_tweeter_joint(part, "tweeter")'
            in addon)
    assert "\nJOINT_HOLE_D =" not in core
    # Ear clearance is established by the existing complementary Z-half
    # receiver recuts.  A plan keepout subtracted from both owners recreates
    # the visible moat and must never be reintroduced.
    lm_plan_source = core[core.index("def lm_um_closure_polygons"):
                          core.index("def _t_crescent_boundary_y")]
    t_plan_source = core[core.index("def t_um_closure_polygons"):
                         core.index("def junction_closure_polygons")]
    assert ".difference(ear_keepout)" not in lm_plan_source
    assert ".difference(keepout)" not in t_plan_source
    assert '"audit_domain": audit_domain' in lm_plan_source
    assert '"audit_domain": audit_domain' in t_plan_source
    assert '"terminal_drain": terminal_drain' in lm_plan_source
    assert '"terminal_drain": terminal_drain' in t_plan_source

    # The closure must enter the massive blank before route cutters hollow
    # it; otherwise a late front-only patch could conceal a rear cavity.
    lm_body = core[core.index("def lm_carrier_outer_blank"):
                   core.index("def apply_lm_route_cutter")]
    assert lm_body.index('_junction_closure_web("lm_um", "lm")') < (
        lm_body.index('route_outer_covers("lm")'))
    assert lm_body.index("_lm_um_rear_recess_backfill()") < (
        lm_body.index('route_outer_covers("lm")'))
    um_body = core[core.index("def um_carrier"):
                   core.index("def core_parts")]
    for needle in ('_junction_closure_web("lm_um", "um")',
                   '_junction_closure_web("t_um", "um")'):
        assert um_body.index(needle) < um_body.index(
            'route_outer_covers("um")')
    assert um_body.index("_um_t_rear_recess_backfill()") < (
        um_body.index('route_outer_covers("um")'))
    lm_finalize_start = core.index("def finalize_lm_carrier")
    lm_finalize = core[lm_finalize_start:
                       core.index("\ndef lm_carrier():", lm_finalize_start)]
    assert lm_finalize.index(
        '_enforce_junction_plan_ownership(part, "lm_um", "lm")') < (
        lm_finalize.index("_lm_t_closure_handoff_cutters()"))

    # Static call inventory guards against silently deleting either CAD
    # authority while refactoring the independently printable owners.
    assert "_junction_closure_web" in _function_call_names(
        core, "lm_carrier_outer_blank")
    assert "_junction_closure_web" in _function_call_names(
        core, "um_carrier")
    assert "_junction_closure_web" in _function_call_names(
        addon, "tweeter_crescent")
    assert "_apply_complete_um_tweeter_joint" in _function_call_names(
        core, "um_carrier")
    assert "_apply_complete_um_tweeter_joint" in _function_call_names(
        addon, "tweeter_crescent")


def test_make_jobserver_owns_the_complete_dense_matrix() -> None:
    """Keep concurrency/dependency authority in Make, not this test file."""
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    source = Path(__file__).read_text(encoding="utf-8")
    for case in DENSE_CASES:
        assert case in makefile
    assert "JUNCTION_CLOSURE_DENSE_STAMPS" in makefile
    assert "LX_OBIWAN_CLOSURE_BASE_ONLY=1" in makefile
    assert "LX_OBIWAN_CLOSURE_DENSE_CASE=$(1)" in makefile
    assert "LX_OBIWAN_CLOSURE_DENSE_SHARD=$(2)/4" in makefile
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "multiprocessing" not in imported
    assert "concurrent.futures" not in imported


def test_dense_shards_are_an_exact_disjoint_partition() -> None:
    samples = tuple(float(index) for index in range(97))
    shards = tuple(
        _dense_shard_samples(samples, f"{index}/{DENSE_SHARD_COUNT}")
        for index in range(DENSE_SHARD_COUNT))
    assert all(
        set(first).isdisjoint(second)
        for index, first in enumerate(shards)
        for second in shards[index + 1:])
    assert tuple(sorted(sum(shards, ()))) == samples


def _pieces(geometry):
    if geometry.geom_type == "Polygon":
        return (geometry,)
    return tuple(piece for piece in geometry.geoms if not piece.is_empty)


def _assert_fit_seam_is_declared(target, owner_union, fit_seam,
                                 seam_gap: float) -> None:
    """The uncovered assembly seam must equal its explicit authority."""
    uncovered = target.difference(owner_union).buffer(0)
    assert uncovered.area > 0.0
    assert uncovered.area < target.length * (seam_gap + 0.03)
    assert uncovered.symmetric_difference(fit_seam).area <= 1.0e-8


def test_analytic_closure_plans_have_only_open_assembly_seams() -> None:
    _require_guarded_test()
    import top_baffle_nd25fw4_obiwan as cad
    from shapely.affinity import scale
    from shapely.geometry import LineString, Point, Polygon
    from shapely.ops import unary_union

    assert cad.PRINT_ORIENTATION == "front-face-down"
    assert cad.JUNCTION_WEB_Z == (cad.CORE_REAR_Z, cad.THICKNESS_MM)
    assert math.isclose(cad.TWEETER_CORE_BORE_TOP_Z, 12.55, abs_tol=1.0e-9)
    assert math.isclose(
        cad.LM_T_CLOSURE_HANDOFF_RELIEF_MM, 0.0, abs_tol=1.0e-12)
    assert 12.50 < cad.TWEETER_CORE_BORE_TOP_Z
    assert (cad.TWEETER_ADDON_JOINT_Z[0]
            < cad.TWEETER_CORE_BORE_TOP_Z
            < cad.TWEETER_ADDON_JOINT_Z[1])
    assert (cad.TWEETER_ADDON_JOINT_Z[0] - 0.2
            < cad.TWEETER_CORE_BORE_TOP_Z)
    bambu_world_layers = []
    layer_z = cad.THICKNESS_MM - 0.20
    while layer_z >= cad.CORE_REAR_Z - 1.0e-9:
        bambu_world_layers.append(layer_z)
        layer_z -= 0.16
    assert min(abs(layer - cad.TWEETER_CORE_BORE_TOP_Z)
               for layer in bambu_world_layers) > 0.04
    assert math.isclose(
        cad.JUNCTION_WEB_SEAM_GAP, cad.SIDE_INTERFACE_GAP,
        abs_tol=1.0e-12)

    # The route-pinched rear flange-recess sliver is closed by two mirrored,
    # solid crescents below the immutable flange seat.  Their outer 0.25 mm
    # fuses into the R110.6 lip; the inner edge stays far outside D190.
    rear_plan = cad._lm_um_rear_recess_backfill_plan()
    rear_pieces = _pieces(rear_plan)
    assert len(rear_pieces) == 2
    mirrored = unary_union([
        Polygon([(-x, y) for x, y in piece.exterior.coords])
        for piece in rear_pieces
    ]).buffer(0)
    assert rear_plan.symmetric_difference(mirrored).area <= 1.0e-7
    lm_center = Point(*cad.L22_CUTOUT[:2])
    inner_limit = cad.LM_RECESS_R - 1.06
    outer_limit = cad.LM_RECESS_R + 0.26
    lip = Point(*cad.L22_CUTOUT[:2]).buffer(
        cad.LM_CORE_R, resolution=256).difference(
            Point(*cad.L22_CUTOUT[:2]).buffer(
                cad.LM_RECESS_R, resolution=256))
    driver_opening = Point(*cad.L22_CUTOUT[:2]).buffer(
        cad.L22_CUTOUT[2] / 2.0, resolution=256)
    for piece in rear_pieces:
        radii = [Point(x, y).distance(lm_center)
                 for x, y in piece.exterior.coords]
        assert min(radii) >= inner_limit - 0.01
        assert max(radii) <= outer_limit + 0.01
        assert piece.intersection(lip).area > 0.25
        assert piece.intersection(driver_opening).is_empty

    # The second route-pinched pocket lies on the lower-right UM recess lip,
    # safely outside the R3 T lumen.  Close it below the membrane and mirror
    # the backing land so the carrier remains structurally symmetric.
    um_rear_plan = cad._um_t_rear_recess_backfill_plan()
    um_rear_pieces = _pieces(um_rear_plan)
    assert len(um_rear_pieces) == 2
    um_mirrored = unary_union([
        Polygon([(-x, y) for x, y in piece.exterior.coords])
        for piece in um_rear_pieces
    ]).buffer(0)
    assert um_rear_plan.symmetric_difference(um_mirrored).area <= 1.0e-7
    um_center = Point(*cad.UM_CUTOUT[:2])
    um_lip = Point(*cad.UM_CUTOUT[:2]).buffer(
        cad.UM_CORE_R, resolution=256).difference(
            Point(*cad.UM_CUTOUT[:2]).buffer(
                cad.UM_RECESS_R, resolution=256))
    um_driver_opening = Point(*cad.UM_CUTOUT[:2]).buffer(
        cad.UM_CUTOUT[2] / 2.0, resolution=256)
    for piece in um_rear_pieces:
        radii = [Point(x, y).distance(um_center)
                 for x, y in piece.exterior.coords]
        assert min(radii) >= cad.UM_RECESS_R - 2.51
        assert max(radii) <= cad.UM_RECESS_R + 0.91
        assert piece.intersection(um_lip).area > 0.20
        assert piece.intersection(um_driver_opening).is_empty

    plans = cad.junction_closure_polygons()
    owner_map = {
        "lm_um": ("lm", "um"),
        "t_um": ("um", "tweeter"),
    }
    for junction, owners in owner_map.items():
        record = plans[junction]
        target = record["target"]
        assert target.is_valid and target.area > 1.0
        target_holes = [
            Polygon(ring) for piece in _pieces(target)
            for ring in piece.interiors
            if Polygon(ring).area > 1.0e-8
        ]
        assert not target_holes, (
            f"{junction}: closure target contains holes "
            f"{[(hole.area, hole.bounds) for hole in target_holes]}")
        first, second = (record[name] for name in owners)
        assert first.is_valid and second.is_valid
        assert first.area > 1.0 and second.area > 1.0
        owner_overlap = first.intersection(second)
        assert owner_overlap.area <= 1.0e-7, (
            f"{junction}: analytic owners overlap "
            f"area={owner_overlap.area:.9f} bounds={owner_overlap.bounds} "
            f"components={[(piece.area, piece.bounds) for piece in _pieces(owner_overlap)]}")
        owner_union = unary_union((first, second))
        assert owner_union.buffer(
            cad.JUNCTION_WEB_SEAM_GAP / 2.0 + 0.015,
            join_style=1).covers(target)
        _assert_fit_seam_is_declared(
            target, owner_union, record["fit_seam"],
            cad.JUNCTION_WEB_SEAM_GAP)

        drain = record["terminal_drain"]
        assert len(_pieces(drain)) == 2
        for component in _pieces(drain):
            assert component.intersection(record["fit_seam"]).area > 0.0
            assert component.boundary.intersection(
                record["audit_domain"].boundary).length > 0.01
        if junction == "lm_um":
            # This is only the full-depth closure-web partition.  The final
            # functional construction restores one complete Z-owned boss per
            # print, so this base-plan seam is never the authority for either
            # the LM clearance annulus or the UM insert annulus.
            lm_disk = Point(*cad.L22_CUTOUT[:2]).buffer(
                cad.LM_CORE_R, resolution=128).difference(
                    Point(*cad.L22_CUTOUT[:2]).buffer(
                        cad.LM_RECESS_R, resolution=128))
            um_disk = Point(*cad.UM_CUTOUT[:2]).buffer(
                cad.UM_CORE_R, resolution=128).difference(
                    Point(*cad.UM_CUTOUT[:2]).buffer(
                        cad.UM_RECESS_R, resolution=128))
            lm_blocked = unary_union((
                target.difference(record["lm"]), drain)).buffer(0)
            um_blocked = unary_union((
                target.difference(record["um"]), drain)).buffer(0)

            # Freeze the closure-clipped *base-web* footprint independently
            # of the complete functional ears. This still catches accidental
            # web shrink, but must never be reused as the printable bore or
            # insert boss oracle.
            frozen_ear_plans = {
                "lm": {
                    "raw_area_mm2": 81.494594604,
                    "owned_area_mm2": 49.438471980,
                    "clipped_area_mm2": 32.056122624,
                    "blocked_area_mm2": 30.999091793,
                    "unsupported_areas_mm2": (0.487431975, 0.569598855),
                },
                "um": {
                    "raw_area_mm2": 92.174133308,
                    "owned_area_mm2": 60.139574842,
                    "clipped_area_mm2": 32.034558466,
                    "blocked_area_mm2": 30.926525381,
                    "unsupported_areas_mm2": (0.487431975, 0.620601109),
                },
            }
            owned_by_owner_and_x = {"lm": {}, "um": {}}
            for owner_name, center, radius, blocked in (
                    ("lm", cad.L22_CUTOUT[:2], cad.LM_CORE_R,
                     lm_blocked),
                    ("um", cad.UM_CUTOUT[:2], cad.UM_CORE_R,
                     um_blocked)):
                frozen = frozen_ear_plans[owner_name]
                support = unary_union((
                    Point(*center).buffer(radius, resolution=128),
                    record[owner_name],
                )).buffer(0)
                for x in cad.JOINT_EAR_X:
                    raw = cad.joint_ear_polygon(owner_name, x)
                    owned = cad._owned_joint_ear_plan(owner_name, x)
                    remainder_components = _pieces(
                        raw.difference(blocked).buffer(0))
                    supported = [
                        piece for piece in remainder_components
                        if piece.intersection(support).area > 1.0e-8
                    ]
                    unsupported = [
                        piece for piece in remainder_components
                        if piece.intersection(support).area <= 1.0e-8
                    ]
                    assert len(supported) == 1
                    assert len(unsupported) == 2
                    assert owned.symmetric_difference(
                        supported[0]).area <= 1.0e-8

                    clipped = raw.difference(owned).buffer(0)
                    declared_clip = unary_union((
                        raw.intersection(blocked), *unsupported,
                    )).buffer(0)
                    assert clipped.symmetric_difference(
                        declared_clip).area <= 1.0e-8
                    assert math.isclose(
                        raw.area, frozen["raw_area_mm2"],
                        abs_tol=2.0e-6)
                    assert math.isclose(
                        owned.area, frozen["owned_area_mm2"],
                        abs_tol=2.0e-6)
                    assert math.isclose(
                        clipped.area, frozen["clipped_area_mm2"],
                        abs_tol=2.0e-6)
                    assert math.isclose(
                        raw.intersection(blocked).area,
                        frozen["blocked_area_mm2"], abs_tol=2.0e-6)
                    assert all(
                        math.isclose(actual, expected, abs_tol=2.0e-6)
                        for actual, expected in zip(
                            sorted(piece.area for piece in unsupported),
                            frozen["unsupported_areas_mm2"]))
                    owned_by_owner_and_x[owner_name][x] = owned

                left = owned_by_owner_and_x[owner_name][
                    min(cad.JOINT_EAR_X)]
                mirrored_right = scale(
                    owned_by_owner_and_x[owner_name][
                        max(cad.JOINT_EAR_X)],
                    xfact=-1.0, yfact=1.0, origin=(0.0, 0.0))
                assert left.symmetric_difference(
                    mirrored_right).area <= 1.0e-8

            lm_ears = unary_union([
                cad._complete_joint_ear_plan("lm", x)
                for x in cad.JOINT_EAR_X
            ])
            um_ears = unary_union([
                cad._complete_joint_ear_plan("um", x)
                for x in cad.JOINT_EAR_X
            ])
            lm_receivers = unary_union([
                cad._complete_joint_ear_plan(
                    "um", x, cad.JOINT_RECEIVER_RADIAL_CLEAR)
                for x in cad.JOINT_EAR_X
            ])
            um_receivers = unary_union([
                cad._complete_joint_ear_plan(
                    "lm", x, cad.JOINT_RECEIVER_RADIAL_CLEAR)
                for x in cad.JOINT_EAR_X
            ])
            planned_sections = {
                "lm": unary_union((lm_disk, record["lm"], lm_ears))
                      .difference(lm_receivers).buffer(0),
                "um": unary_union((um_disk, record["um"], um_ears))
                      .difference(um_receivers).buffer(0),
            }
            for owner_name, section in planned_sections.items():
                components = _pieces(section)
                assert len(components) == 1, (
                    f"lm_um/{owner_name}: planned receiver leaves detached "
                    f"components {[(piece.area, piece.bounds) for piece in components]}")
        else:
            # The add-on receivers are blind, so the T seam has an explicit
            # 0.05-mm continuation through each boss perimeter.  It must
            # overlap the fit seam and reach the independent audit boundary.
            for component in _pieces(drain):
                assert component.bounds[3] - component.bounds[1] <= 0.051
            assert drain.intersection(record["route_keepout"]).is_empty

            # Freeze the raw-to-owned T closure-base clipping independently
            # of ``_owned_tweeter_joint_plan``.  These areas describe only
            # the legacy full-depth web contribution; standalone BREP/STL
            # gates deliberately use complete D9.8 functional ears instead.
            # Keeping the diagnostic record prevents a later closure-plan
            # change from silently inventing detached web slivers.
            frozen_ear_plans = {
                "um": {
                    "raw_area_mm2": 96.922625703,
                    "owned_area_mm2": 52.632729701,
                    "clipped_area_mm2": 44.289896003,
                    "blocked_area_mm2": 30.212059867,
                    "unsupported_areas_mm2": (0.487431975, 13.590404160),
                },
                "tweeter": {
                    "raw_area_mm2": 96.922625703,
                    "owned_area_mm2": 43.820883354,
                    "clipped_area_mm2": 53.101742349,
                    "blocked_area_mm2": 31.347357623,
                    "unsupported_areas_mm2": (0.487431975, 21.266952751),
                },
            }
            owned_by_owner_and_x = {"um": {}, "tweeter": {}}
            for owner_name, center, radius in (
                    ("um", cad.UM_CUTOUT[:2], cad.UM_CORE_R),
                    ("tweeter", cad.T_CRESCENT_ARC_CENTER,
                     cad.T_CRESCENT_ARC_R)):
                frozen = frozen_ear_plans[owner_name]
                blocked = unary_union((
                    target.difference(record[owner_name]), drain,
                )).buffer(0)
                support = unary_union((
                    Point(*center).buffer(radius, resolution=128),
                    record[owner_name],
                )).buffer(0)
                for x in cad.TWEETER_JOINT_X:
                    raw = cad.tweeter_joint_polygon(x)
                    owned = cad._owned_tweeter_joint_plan(owner_name, x)
                    remainder_components = _pieces(
                        raw.difference(blocked).buffer(0))
                    supported = [
                        piece for piece in remainder_components
                        if piece.intersection(support).area > 1.0e-8
                    ]
                    unsupported = [
                        piece for piece in remainder_components
                        if piece.intersection(support).area <= 1.0e-8
                    ]
                    assert len(supported) == 1
                    assert len(unsupported) == 2
                    assert owned.symmetric_difference(
                        supported[0]).area <= 1.0e-8

                    clipped = raw.difference(owned).buffer(0)
                    declared_clip = unary_union((
                        raw.intersection(blocked), *unsupported,
                    )).buffer(0)
                    assert clipped.symmetric_difference(
                        declared_clip).area <= 1.0e-8
                    assert math.isclose(
                        raw.area, frozen["raw_area_mm2"],
                        abs_tol=2.0e-6)
                    assert math.isclose(
                        owned.area, frozen["owned_area_mm2"],
                        abs_tol=2.0e-6)
                    assert math.isclose(
                        clipped.area, frozen["clipped_area_mm2"],
                        abs_tol=2.0e-6)
                    assert math.isclose(
                        raw.intersection(blocked).area,
                        frozen["blocked_area_mm2"], abs_tol=2.0e-6)
                    assert all(
                        math.isclose(actual, expected, abs_tol=2.0e-6)
                        for actual, expected in zip(
                            sorted(piece.area for piece in unsupported),
                            frozen["unsupported_areas_mm2"]))
                    owned_by_owner_and_x[owner_name][x] = owned

                left = owned_by_owner_and_x[owner_name][
                    min(cad.TWEETER_JOINT_X)]
                mirrored_right = scale(
                    owned_by_owner_and_x[owner_name][
                        max(cad.TWEETER_JOINT_X)],
                    xfact=-1.0, yfact=1.0, origin=(0.0, 0.0))
                assert left.symmetric_difference(
                    mirrored_right).area <= 1.0e-8

        # Independent endpoint requirement: every construction chord must be
        # hidden inside an existing rounded ear.  This rejects the former
        # x=+/-32 and x=+/-24 vertical caps even if the CAD generator and its
        # expected-plan oracle make the same mistake.
        if junction == "lm_um":
            ears = unary_union([
                cad.joint_ear_polygon(owner, x)
                for owner in ("lm", "um") for x in cad.JOINT_EAR_X
            ])
        else:
            ears = unary_union([
                cad.tweeter_joint_polygon(x)
                for x in cad.TWEETER_JOINT_X
            ])
        exposed_chord = record["terminal_chords"].difference(
            ears.buffer(0.01, join_style=1))
        assert exposed_chord.is_empty or exposed_chord.length <= 1.0e-4, (
            f"{junction}: exposed constant-X closure chord "
            f"length={exposed_chord.length:.6f} mm")
        assert (record["closure_lenses"].is_valid
                and 0.5 < record["closure_lenses"].area < 15.0), (
            f"{junction}: independent ring/ear lens fill missing or implausible "
            f"area={record['closure_lenses'].area:.6f} mm2")
        lenses = _pieces(record["closure_lenses"])
        assert all(
            lens.area >= cad.JUNCTION_WEB_MIN_LENS_AREA_MM2
            for lens in lenses), (
            f"{junction}: retained sub-resolution lens components "
            f"{[(lens.area, lens.bounds) for lens in lenses]}")
        if junction == "t_um":
            assert len(lenses) == 2, (
                "T closure must retain only its two physical side lenses, "
                f"got {[(lens.area, lens.bounds) for lens in lenses]}")

        # A valid OCC touch is insufficient: the approved anti-void lenses
        # must remain connected after erosion by half one 0.42-mm Classic
        # wall path.  This rejects point/line contacts and sub-slicer necks.
        erosion = 0.21
        for lens in lenses:
            probe = lens.representative_point()
            assigned = [
                owner for owner in owners
                if record[owner].buffer(0.002).covers(probe)
            ]
            assert len(assigned) == 1, (
                f"{junction}: lens ownership is not unique: {assigned}")
            eroded = record[assigned[0]].buffer(
                -erosion, join_style=1).buffer(0)
            lens_core = lens.buffer(-0.03, join_style=1)
            if lens_core.is_empty:
                lens_core = lens
            connected = [
                component for component in _pieces(eroded)
                if component.intersection(lens_core).area > 1.0e-4
            ]
            assert connected and any(
                component.difference(
                    lens.buffer(0.05, join_style=1)).area > 0.01
                for component in connected), (
                f"{junction}/{assigned[0]}: lens lacks a continuous "
                "0.42-mm slicable fusion land")

    # Frozen physical witness stations prevent the generator from shrinking
    # its own audit domain around a defect.  These spans are defined only by
    # the released outer circles/crescent, not by the Bezier owner paths.
    lm_record = plans["lm_um"]
    assert math.isclose(lm_record["target"].bounds[0], -36.1, abs_tol=0.015)
    assert math.isclose(lm_record["target"].bounds[2], 36.1, abs_tol=0.015)
    for x in (-20.0, -10.0, 0.0, 10.0, 20.0):
        lower = cad._circle_branch_y(
            cad.L22_CUTOUT[:2], cad.LM_CORE_R, x, upper=True)
        upper = cad._circle_branch_y(
            cad.UM_CUTOUT[:2], cad.UM_CORE_R, x, upper=False)
        witness = LineString(((x, lower + 0.01), (x, upper - 0.01)))
        assert lm_record["target"].buffer(0.012).covers(witness)

    t_record = plans["t_um"]
    assert math.isclose(t_record["target"].bounds[0], -28.1, abs_tol=0.015)
    assert math.isclose(t_record["target"].bounds[2], 28.1, abs_tol=0.015)
    for x in (-14.0, -10.0, -6.1, 6.1, 10.0, 14.0):
        lower = cad._circle_branch_y(
            cad.UM_CUTOUT[:2], cad.UM_CORE_R, x, upper=True)
        upper = cad._t_crescent_boundary_y(x)
        witness = LineString(((x, lower + 0.01), (x, upper - 0.01)))
        assert t_record["target"].buffer(0.012).covers(witness)

    assert t_record["target"].intersection(
        t_record["route_keepout"]).area <= 1e-8
    assert t_record["um"].intersection(
        t_record["route_keepout"]).area <= 1e-8
    assert t_record["tweeter"].intersection(
        t_record["route_keepout"]).area <= 1e-8
    target_pieces = _pieces(t_record["target"])
    assert len(target_pieces) >= 2
    assert any(piece.bounds[2] <= -cad.T_UM_CABLE_MOUTH_HALF_WIDTH + 1e-8
               for piece in target_pieces)
    assert any(piece.bounds[0] >= cad.T_UM_CABLE_MOUTH_HALF_WIDTH - 1e-8
               for piece in target_pieces)
    assert t_record["audit_domain"].difference(
        t_record["target"]).symmetric_difference(
            t_record["terminal_drain"].difference(
                t_record["target"])).area <= 1.0e-8
    assert lm_record["audit_domain"].difference(
        lm_record["target"]).symmetric_difference(
            lm_record["terminal_drain"].difference(
                lm_record["target"])).area <= 1.0e-8


def test_brep_web_prisms_are_solid_through_the_complete_obiwan_depth() -> None:
    _require_guarded_test()
    import top_baffle_nd25fw4_obiwan as cad

    plans = cad.junction_closure_polygons()
    for junction, owner in (
            ("lm_um", "lm"),
            ("lm_um", "um"),
            ("t_um", "um"),
            ("t_um", "tweeter")):
        web = cad._junction_closure_web(junction, owner)
        bounds = web.bounding_box()
        assert math.isclose(bounds.min.Z, cad.CORE_REAR_Z, abs_tol=1e-7)
        assert math.isclose(bounds.max.Z, cad.THICKNESS_MM, abs_tol=1e-7)
        want = (plans[junction][owner].area
                * (cad.THICKNESS_MM - cad.CORE_REAR_Z))
        assert math.isclose(web.volume, want, rel_tol=2e-5, abs_tol=0.05)
        solids = list(web.solids())
        assert solids and all(
            solid.is_valid and solid.volume > 0.01 for solid in solids), (
                f"{junction}/{owner}: invalid full-depth prism members "
                f"{[(solid.is_valid, solid.volume) for solid in solids]}")

    rear_fill = cad._lm_um_rear_recess_backfill()
    rear_bounds = rear_fill.bounding_box()
    assert math.isclose(
        rear_bounds.min.Z, cad.CORE_REAR_Z, abs_tol=1.0e-7)
    assert math.isclose(
        rear_bounds.max.Z,
        cad.LM_SEAT_Z - cad.SEAT_MEMBRANE_T, abs_tol=1.0e-7)
    rear_solids = tuple(rear_fill.solids())
    assert len(rear_solids) == 2
    assert all(solid.is_valid and solid.volume > 1.0
               for solid in rear_solids)
    expected_rear_volume = (
        cad._lm_um_rear_recess_backfill_plan().area
        * (cad.LM_UM_REAR_BACKFILL_Z[1]
           - cad.LM_UM_REAR_BACKFILL_Z[0]))
    assert math.isclose(
        rear_fill.volume, expected_rear_volume,
        rel_tol=2.0e-5, abs_tol=0.02)

    um_rear_fill = cad._um_t_rear_recess_backfill()
    um_rear_bounds = um_rear_fill.bounding_box()
    assert math.isclose(
        um_rear_bounds.min.Z, cad.CORE_REAR_Z, abs_tol=1.0e-7)
    assert math.isclose(
        um_rear_bounds.max.Z,
        cad.UM_SEAT_Z - cad.SEAT_MEMBRANE_T, abs_tol=1.0e-7)
    um_rear_solids = tuple(um_rear_fill.solids())
    assert len(um_rear_solids) == 2
    assert all(solid.is_valid and solid.volume > 1.0
               for solid in um_rear_solids)
    expected_um_rear_volume = (
        cad._um_t_rear_recess_backfill_plan().area
        * (cad.UM_T_REAR_BACKFILL_Z[1]
           - cad.UM_T_REAR_BACKFILL_Z[0]))
    assert math.isclose(
        um_rear_fill.volume, expected_um_rear_volume,
        rel_tol=2.0e-5, abs_tol=0.02)


def _missing_volume(required, owner) -> float:
    missing = (required - owner).clean()
    return sum(solid.volume for solid in missing.solids())


def _intersection_volume(first, second) -> float:
    intersection = (first & second).clean()
    return sum(solid.volume for solid in intersection.solids())


@lru_cache(maxsize=2)
def obiwan_actual_parts(stand_foot: bool = False):
    """Import exact hash-validated release-stage owners for one state."""
    _require_guarded_test()
    from build123d import import_brep
    from export_obiwan_staged import load_stage_manifest, staged_part_paths

    state = "floor_stand" if stand_foot else "no_floor_stand"
    manifest = ROOT / state / ".obiwan_stage/manifest.json"
    payload = load_stage_manifest(manifest, stand_foot=stand_foot)
    paths = staged_part_paths(manifest, payload)
    return {
        "lm": import_brep(str(paths["core_lm_carrier"])),
        "um": import_brep(str(paths["core_um_carrier"])),
        "tweeter": import_brep(str(paths["addon_tweeter_crescent"])),
    }


def test_lm_um_individual_breps_own_complete_fastener_features() -> None:
    """Each separate print owns a complete usable bore/insert boss."""
    _require_guarded_test()
    import top_baffle_nd25fw4_obiwan as cad

    assert math.isclose(
        cad.UM_JOINT_Z[0] - cad.LM_JOINT_Z[1], 0.20,
        abs_tol=1.0e-9)
    assert math.isclose(
        cad.JOINT_INSERT_BORE_Z[1] - cad.UM_JOINT_Z[0],
        cad.JOINT_INSERT_DEPTH_MM, abs_tol=1.0e-9)
    assert math.isclose(
        cad.JOINT_INSERT_FRONT_FLOOR_MM, 1.90, abs_tol=1.0e-9)
    assert math.isclose(
        cad.JOINT_CLEARANCE_BORE_TOP_Z - cad.JOINT_INSERT_BORE_Z[0],
        0.35, abs_tol=1.0e-9)

    for stand_foot, state in ((False, "no_floor"), (True, "floor")):
        actual = obiwan_actual_parts(stand_foot)
        lm = actual["lm"]
        um = actual["um"]
        for x in cad.JOINT_EAR_X:
            lm_ear = cad._complete_joint_ear("lm", x)
            um_ear = cad._complete_joint_ear("um", x)
            assert _intersection_volume(um, lm_ear) <= 0.01, (
                f"{state}/x={x:g}: UM crosses into the standalone LM ear")
            assert _intersection_volume(lm, um_ear) <= 0.01, (
                f"{state}/x={x:g}: LM crosses into the standalone UM ear")

            # Complete 360-degree printable walls, inset from nominal faces
            # so Boolean/section tolerances cannot turn a tangent into a pass.
            lm_outer = cad._cylinder_at(
                x, cad.JOINT_EAR_Y,
                cad.JOINT_FUNCTIONAL_BOSS_D / 2.0 - 0.05,
                cad.LM_JOINT_Z[0] + 0.03,
                cad.LM_JOINT_Z[1] - 0.03)
            lm_inner = cad._cylinder_at(
                x, cad.JOINT_EAR_Y,
                cad.JOINT_CLEARANCE_BORE_D / 2.0 + 0.05,
                cad.LM_JOINT_Z[0] + 0.02,
                cad.LM_JOINT_Z[1] - 0.02)
            lm_annulus = (lm_outer - lm_inner).clean()
            assert _missing_volume(lm_annulus, lm) <= 0.02, (
                f"{state}/x={x:g}: LM clearance boss is not 360 degrees")

            um_outer = cad._cylinder_at(
                x, cad.JOINT_EAR_Y,
                cad.JOINT_FUNCTIONAL_BOSS_D / 2.0 - 0.05,
                cad.UM_JOINT_Z[0] + 0.03,
                cad.JOINT_INSERT_BORE_Z[1] - 0.03)
            um_inner = cad._cylinder_at(
                x, cad.JOINT_EAR_Y,
                cad.JOINT_INSERT_BORE_D / 2.0 + 0.05,
                cad.UM_JOINT_Z[0] + 0.02,
                cad.JOINT_INSERT_BORE_Z[1] - 0.02)
            um_annulus = (um_outer - um_inner).clean()
            assert _missing_volume(um_annulus, um) <= 0.02, (
                f"{state}/x={x:g}: UM insert boss is not 360 degrees")

            clearance_path = cad._cylinder_at(
                x, cad.JOINT_EAR_Y,
                cad.JOINT_CLEARANCE_BORE_D / 2.0 - 0.05,
                cad.LM_JOINT_Z[0], cad.JOINT_CLEARANCE_BORE_TOP_Z)
            assert _intersection_volume(lm, clearance_path) <= 0.01, (
                f"{state}/x={x:g}: LM clearance passage is obstructed")
            insert_receiver = cad._cylinder_at(
                x, cad.JOINT_EAR_Y,
                cad.JOINT_INSERT_BORE_D / 2.0 - 0.05,
                cad.JOINT_INSERT_BORE_Z[0],
                cad.JOINT_INSERT_BORE_Z[1] - 0.02)
            assert _intersection_volume(um, insert_receiver) <= 0.01, (
                f"{state}/x={x:g}: UM blind insert receiver is obstructed")

            # The individual UM print retains a complete floor in front of
            # the blind receiver; the functional D9.8 boss is also absent from
            # both prints throughout the intentional 0.20-mm assembly gap.
            front_floor = cad._cylinder_at(
                x, cad.JOINT_EAR_Y,
                cad.JOINT_FUNCTIONAL_BOSS_D / 2.0 - 0.10,
                cad.JOINT_INSERT_BORE_Z[1] + 0.03,
                cad.UM_JOINT_Z[1] - 0.03)
            assert _missing_volume(front_floor, um) <= 0.02, (
                f"{state}/x={x:g}: UM blind receiver lost its front floor")
            axial_gap = cad._cylinder_at(
                x, cad.JOINT_EAR_Y,
                cad.JOINT_FUNCTIONAL_BOSS_D / 2.0 - 0.05,
                cad.LM_JOINT_Z[1] + 0.01,
                cad.UM_JOINT_Z[0] - 0.01)
            assert _intersection_volume(lm, axial_gap) <= 0.01
            assert _intersection_volume(um, axial_gap) <= 0.01

            # Both negative cutters overlap across the split, providing one
            # unobstructed rear-driven screw/insert approach in assembly.
            common_path = cad._cylinder_at(
                x, cad.JOINT_EAR_Y,
                cad.JOINT_CLEARANCE_BORE_D / 2.0 - 0.05,
                cad.JOINT_INSERT_BORE_Z[0] + 0.01,
                cad.JOINT_CLEARANCE_BORE_TOP_Z - 0.01)
            assert _intersection_volume(lm, common_path) <= 0.01
            assert _intersection_volume(um, common_path) <= 0.01


def test_um_tweeter_individual_breps_own_complete_fastener_features() -> None:
    """UM and crescent each print a complete T ear and usable fastener void."""
    _require_guarded_test()
    import top_baffle_nd25fw4_obiwan as cad

    assert math.isclose(
        cad.TWEETER_ADDON_JOINT_Z[0] - cad.TWEETER_CORE_JOINT_Z[1],
        0.20, abs_tol=1.0e-9)
    assert math.isclose(
        cad.TWEETER_JOINT_INSERT_BORE_Z[1]
        - cad.TWEETER_ADDON_JOINT_Z[0],
        cad.TWEETER_JOINT_INSERT_DEPTH_MM, abs_tol=1.0e-9)
    assert math.isclose(
        cad.TWEETER_JOINT_INSERT_FRONT_FLOOR_MM,
        1.90, abs_tol=1.0e-9)
    assert math.isclose(
        cad.TWEETER_CORE_BORE_TOP_Z
        - cad.TWEETER_JOINT_INSERT_BORE_Z[0],
        0.35, abs_tol=1.0e-9)

    for stand_foot, state in ((False, "no_floor"), (True, "floor")):
        actual = obiwan_actual_parts(stand_foot)
        um = actual["um"]
        tweeter = actual["tweeter"]
        assert _intersection_volume(um, tweeter) <= 0.01
        for x in cad.TWEETER_JOINT_X:
            um_ear = cad._complete_tweeter_joint_ear("um", x)
            tweeter_ear = cad._complete_tweeter_joint_ear("tweeter", x)
            assert _intersection_volume(tweeter, um_ear) <= 0.01, (
                f"{state}/x={x:g}: crescent crosses into standalone UM ear")
            assert _intersection_volume(um, tweeter_ear) <= 0.01, (
                f"{state}/x={x:g}: UM crosses into standalone crescent ear")

            um_outer = cad._cylinder_at(
                x, cad.TWEETER_JOINT_Y,
                cad.TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0 - 0.05,
                cad.TWEETER_CORE_JOINT_Z[0] + 0.03,
                cad.TWEETER_CORE_JOINT_Z[1] - 0.03)
            um_inner = cad._cylinder_at(
                x, cad.TWEETER_JOINT_Y,
                cad.TWEETER_JOINT_HOLE_D / 2.0 + 0.05,
                cad.TWEETER_CORE_JOINT_Z[0] + 0.02,
                cad.TWEETER_CORE_JOINT_Z[1] - 0.02)
            assert _missing_volume(
                (um_outer - um_inner).clean(), um) <= 0.02, (
                f"{state}/x={x:g}: UM T clearance wall is not 360 degrees")

            tweeter_outer = cad._cylinder_at(
                x, cad.TWEETER_JOINT_Y,
                cad.TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0 - 0.05,
                cad.TWEETER_ADDON_JOINT_Z[0] + 0.03,
                cad.TWEETER_JOINT_INSERT_BORE_Z[1] - 0.03)
            tweeter_inner = cad._cylinder_at(
                x, cad.TWEETER_JOINT_Y,
                cad.TWEETER_JOINT_INSERT_BORE_D / 2.0 + 0.05,
                cad.TWEETER_ADDON_JOINT_Z[0] + 0.02,
                cad.TWEETER_JOINT_INSERT_BORE_Z[1] - 0.02)
            assert _missing_volume(
                (tweeter_outer - tweeter_inner).clean(), tweeter) <= 0.02, (
                f"{state}/x={x:g}: crescent receiver wall is not 360 degrees")

            clearance_path = cad._cylinder_at(
                x, cad.TWEETER_JOINT_Y,
                cad.TWEETER_JOINT_HOLE_D / 2.0 - 0.05,
                cad.TWEETER_CORE_JOINT_Z[0],
                cad.TWEETER_CORE_BORE_TOP_Z)
            assert _intersection_volume(um, clearance_path) <= 0.01
            receiver = cad._cylinder_at(
                x, cad.TWEETER_JOINT_Y,
                cad.TWEETER_JOINT_INSERT_BORE_D / 2.0 - 0.05,
                cad.TWEETER_JOINT_INSERT_BORE_Z[0],
                cad.TWEETER_JOINT_INSERT_BORE_Z[1] - 0.02)
            assert _intersection_volume(tweeter, receiver) <= 0.01

            front_floor = cad._cylinder_at(
                x, cad.TWEETER_JOINT_Y,
                cad.TWEETER_JOINT_INSERT_BORE_D / 2.0 - 0.05,
                cad.TWEETER_JOINT_INSERT_BORE_Z[1] + 0.03,
                cad.TWEETER_ADDON_JOINT_Z[1] - 0.03)
            assert _missing_volume(front_floor, tweeter) <= 0.02, (
                f"{state}/x={x:g}: crescent blind receiver lost its floor")

            axial_gap = cad._cylinder_at(
                x, cad.TWEETER_JOINT_Y,
                cad.TWEETER_JOINT_FUNCTIONAL_BOSS_D / 2.0 - 0.05,
                cad.TWEETER_CORE_JOINT_Z[1] + 0.01,
                cad.TWEETER_ADDON_JOINT_Z[0] - 0.01)
            assert _intersection_volume(um, axial_gap) <= 0.01
            assert _intersection_volume(tweeter, axial_gap) <= 0.01

            common_path = cad._cylinder_at(
                x, cad.TWEETER_JOINT_Y,
                cad.TWEETER_JOINT_HOLE_D / 2.0 - 0.05,
                cad.TWEETER_JOINT_INSERT_BORE_Z[0] + 0.01,
                cad.TWEETER_CORE_BORE_TOP_Z - 0.01)
            assert _intersection_volume(um, common_path) <= 0.01
            assert _intersection_volume(tweeter, common_path) <= 0.01


@lru_cache(maxsize=4)
def _cropped_junction_parts(stand_foot: bool, junction: str):
    """Crop large release BREPs once before the dense layer sweep."""
    import top_baffle_nd25fw4_obiwan as cad

    owners = (("lm", "um") if junction == "lm_um"
              else ("um", "tweeter"))
    # The 1-mm collar keeps true connectivity immediately outside the fixed
    # audit rectangle while reducing every subsequent OCC section from a
    # complete carrier to the small physical junction.
    clip_plan = _independent_junction_window(junction).buffer(
        1.0, join_style=2)
    clip = cad._plan_prism(
        clip_plan, cad.CORE_REAR_Z - 0.01, cad.THICKNESS_MM + 0.01)
    actual = obiwan_actual_parts(stand_foot)
    return {
        owner: (actual[owner] & clip).clean()
        for owner in owners
    }


def _front_plan_contract(cad, junction: str):
    """Exact front owner/void plans after complementary receiver recuts."""
    from shapely.ops import unary_union

    record = cad.junction_closure_polygons()[junction]
    audit = record["audit_domain"]
    if junction == "lm_um":
        # The UM insert receiver ends at z=16.4, leaving a fully solid 1.9-mm
        # front floor. At the front audit plane there is therefore no bore;
        # UM owns the complete boss and LM retains only its buffered mating
        # clearance around that boss.
        front_ears = unary_union([
            cad._complete_joint_ear_plan("um", x)
            for x in cad.JOINT_EAR_X
        ])
        receiver = unary_union([
            cad._complete_joint_ear_plan(
                "um", x, cad.JOINT_RECEIVER_RADIAL_CLEAR)
            for x in cad.JOINT_EAR_X
        ])
        owners = {
            "lm": record["lm"].difference(receiver).buffer(0),
            "um": unary_union((record["um"], front_ears)).buffer(0),
        }
        receiver_clearance = receiver.difference(front_ears).buffer(0)
        functional = receiver_clearance
        mouth = record["target"].difference(record["target"])
    elif junction == "t_um":
        front_ears = unary_union([
            cad._complete_tweeter_joint_ear_plan("tweeter", x)
            for x in cad.TWEETER_JOINT_X
        ])
        receiver = unary_union([
            cad._complete_tweeter_joint_ear_plan(
                "tweeter", x, cad.TWEETER_JOINT_CLEAR)
            for x in cad.TWEETER_JOINT_X
        ])
        owners = {
            "um": record["um"].difference(receiver).buffer(0),
            "tweeter": unary_union(
                (record["tweeter"], front_ears)).buffer(0),
        }
        receiver_clearance = receiver.difference(front_ears).buffer(0)
        functional = receiver_clearance
        # The exact route section, not this plan-only closure authority,
        # proves/permits the central cable mouth at every sampled Z.
        mouth = record["target"].difference(record["target"])
    else:
        raise ValueError(junction)

    material = unary_union(tuple(owners.values())).intersection(audit)
    expected_void = audit.difference(material).buffer(0)
    owner_union = unary_union((record["lm"], record["um"])) \
        if junction == "lm_um" else unary_union(
            (record["um"], record["tweeter"]))
    fit_seam = record["fit_seam"]
    # Receiver fit is allowed only as an open assembly seam.  Never bless a
    # bounded moat merely because it lies inside a buffered ear footprint.
    for component in _pieces(receiver_clearance.intersection(audit)):
        assert (component.distance(audit.boundary) <= 0.008
                or component.distance(fit_seam) <= 0.008), (
            f"{junction}: bounded receiver-clearance moat "
            f"area={component.area:.6f} mm2")
    terminal_drain = record["terminal_drain"]
    allowed_void = unary_union(
        (fit_seam, terminal_drain, functional, mouth)).buffer(0)
    undeclared = expected_void.difference(
        allowed_void.buffer(0.012, join_style=1)).buffer(0)
    assert undeclared.is_empty or undeclared.area <= 1.0e-5, (
        f"{junction}: analytic front contract created undeclared void "
        f"area={undeclared.area:.6f} bounds={undeclared.bounds} "
        f"components="
        f"{[(piece.area, piece.bounds) for piece in _pieces(undeclared)]}")
    return audit, owners, expected_void, allowed_void


def _full_depth_safe_plan(cad, junction: str, owner: str):
    """Web interior unaffected by any complementary half-lap receiver."""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    record = cad.junction_closure_polygons()[junction]
    if junction == "lm_um":
        opposing = "um" if owner == "lm" else "lm"
        receivers = unary_union([
            cad._complete_joint_ear_plan(
                opposing, x, cad.JOINT_RECEIVER_RADIAL_CLEAR)
            for x in cad.JOINT_EAR_X
        ])
        bore_d = (cad.JOINT_CLEARANCE_BORE_D if owner == "lm"
                  else cad.JOINT_INSERT_BORE_D)
        holes = unary_union([
            Point(x, cad.JOINT_EAR_Y).buffer(
                bore_d / 2.0, resolution=32)
            for x in cad.JOINT_EAR_X
        ])
        keepout = unary_union((receivers, holes))
    elif junction == "t_um":
        receivers = unary_union([
            cad._complete_tweeter_joint_ear_plan(
                "tweeter", x, cad.TWEETER_JOINT_CLEAR)
            for x in cad.TWEETER_JOINT_X
        ])
        keepout = receivers
    else:
        raise ValueError(junction)
    return record[owner].difference(keepout).buffer(-0.025, join_style=1)


def test_final_owner_breps_keep_safe_web_interiors_solid_full_depth() -> None:
    """Later functional recuts cannot hide a cavity behind the front face."""
    _require_guarded_test()
    import top_baffle_nd25fw4_obiwan as cad

    actual = obiwan_actual_parts()
    for junction, owner in (
            ("lm_um", "lm"),
            ("lm_um", "um"),
            ("t_um", "um"),
            ("t_um", "tweeter")):
        plan = _full_depth_safe_plan(cad, junction, owner)
        assert not plan.is_empty and plan.area > 0.5
        required = cad._plan_prism(plan, *cad.JUNCTION_WEB_Z)
        if owner in {"lm", "um"}:
            for cutter in cad.route_inner_cutters(owner):
                required -= cutter
        if owner == "lm":
            required -= cad.lm_free_lead_relief_cutter()
            for cutter in cad._lm_t_closure_handoff_cutters():
                required -= cutter
        required = required.clean()
        missing = _missing_volume(required, actual[owner])
        assert missing <= max(0.04, required.volume * 4.0e-4), (
            f"{junction}/{owner}: hidden full-depth closure loss "
            f"{missing:.6f} mm3")


def test_brep_front_plane_contains_every_expected_owner_region() -> None:
    """Exact owner BREPs contain their complete post-receiver front plans."""
    _require_guarded_test()
    import top_baffle_nd25fw4_obiwan as cad
    actual = obiwan_actual_parts()
    z0 = cad.THICKNESS_MM - 0.08
    z1 = cad.THICKNESS_MM - 0.02
    for junction in ("lm_um", "t_um"):
        _audit, owner_plans, _void, _allowed = _front_plan_contract(
            cad, junction)
        for owner, plan in owner_plans.items():
            audit_plan = plan.buffer(-0.025, join_style=1)
            assert not audit_plan.is_empty and audit_plan.area > 0.5
            required = cad._plan_prism(audit_plan, z0, z1)
            missing = _missing_volume(required, actual[owner])
            assert missing <= max(0.03, required.volume * 3.0e-4), (
                f"{junction}/{owner} front closure missing "
                f"{missing:.6f} mm3")


def _section_polygons(shape, z_mm: float):
    """Sample exact OCC section wires into Shapely residual components."""
    from build123d import Plane, Rectangle, Wire
    from shapely.geometry import Polygon

    section_face = Plane.XY.offset(z_mm) * Rectangle(1000.0, 1000.0)
    polygons = []
    for solid in shape.solids():
        _vertices, edges = solid._ocp_section(section_face)
        for wire in Wire.combine(edges, tol=1.0e-7):
            if not wire.is_closed or wire.length <= 0.01:
                continue
            sample_count = max(96, int(math.ceil(wire.length / 0.02)))
            points = []
            for index in range(sample_count):
                point = wire.position_at(index / sample_count)
                points.append((float(point.X), float(point.Y)))
            repaired = Polygon(points).buffer(0)
            for polygon in _pieces(repaired):
                if (polygon.geom_type == "Polygon"
                        and polygon.area > 1.0e-5):
                    polygons.append(polygon)
    return polygons


def _section_material_plan(shape, z_mm: float):
    """Reconstruct exact section material using loop-containment parity."""
    from shapely.geometry import GeometryCollection
    from shapely.ops import unary_union

    loops = sorted(
        _section_polygons(shape, z_mm), key=lambda polygon: -polygon.area)
    material = GeometryCollection()
    accepted = []
    for loop in loops:
        probe = loop.representative_point()
        depth = sum(
            1 for outer in accepted
            if outer.buffer(1.0e-6, join_style=1).covers(probe))
        if depth % 2 == 0:
            material = unary_union((material, loop))
        else:
            material = material.difference(loop)
        accepted.append(loop)
    return material.buffer(0)


def _independent_junction_window(junction: str):
    """Fixed physical inspection window, independent of generated webs.

    Each rectangle crosses both adjacent driver apertures at opposite edges.
    Consequently ordinary acoustic openings are connected to the rectangle
    boundary, while the former triangular ear/cusp islands are strictly
    interior.  A generator cannot make this audit smaller by shrinking its
    own ``target`` or ``audit_domain`` plan.
    """
    from shapely.geometry import box

    if junction == "lm_um":
        return box(-45.0, 306.0, 45.0, 330.0)
    if junction == "t_um":
        return box(-35.0, 408.0, 35.0, 429.0)
    raise ValueError(junction)


@lru_cache(maxsize=2)
def _frozen_required_front_domain(junction: str):
    from shapely import wkt

    required = wkt.loads(FROZEN_REQUIRED_FRONT_WKT[junction])
    assert required.is_valid and required.area > 100.0
    return required


def _dense_section_samples(cad):
    """Actual front-down 0.20/0.16-mm layers plus topology probes."""
    z_min = cad.CORE_REAR_Z + 0.03
    z_max = cad.THICKNESS_MM - 0.03
    samples = []
    # The physically qualified Bambu profile starts with a 0.20-mm first
    # layer and then advances in 0.16-mm layers.  Front-face-down reverses
    # that slicer-Z schedule into the CAD world frame, so anchor the sweep at
    # the front datum instead of starting an arbitrary grid at CORE_REAR_Z.
    z_value = cad.THICKNESS_MM - 0.20
    while z_value >= z_min - 1.0e-9:
        samples.append(z_value)
        z_value -= 0.16
    transitions = {
        cad.CORE_REAR_Z,
        cad.THICKNESS_MM,
        *cad.LM_JOINT_Z,
        *cad.UM_JOINT_Z,
        cad.JOINT_CLEARANCE_BORE_TOP_Z,
        *cad.JOINT_INSERT_BORE_Z,
        *cad.TWEETER_CORE_JOINT_Z,
        *cad.TWEETER_ADDON_JOINT_Z,
        cad.LM_JOINT_Z[1] + cad.JOINT_RECEIVER_RADIAL_CLEAR,
        cad.UM_JOINT_Z[0] - cad.JOINT_RECEIVER_RADIAL_CLEAR,
        cad.TWEETER_CORE_JOINT_Z[1] + cad.TWEETER_JOINT_CLEAR,
        cad.TWEETER_CORE_BORE_TOP_Z,
        cad.TWEETER_ADDON_JOINT_Z[0] - cad.TWEETER_JOINT_CLEAR,
        cad.TWEETER_ADDON_JOINT_Z[0] + 4.0,
    }
    for transition in transitions:
        for delta in (-0.03, 0.03):
            sample = transition + delta
            if z_min <= sample <= z_max:
                samples.append(sample)
    return tuple(sorted({round(sample, 5) for sample in samples}))


def _dense_shard_samples(samples, shard: str):
    samples = tuple(samples)
    if not shard:
        return samples
    index_text, separator, count_text = shard.partition("/")
    assert separator and index_text.isdigit() and count_text.isdigit(), (
        f"invalid {DENSE_SHARD_ENV}={shard!r}; expected INDEX/COUNT")
    shard_index = int(index_text)
    shard_count = int(count_text)
    assert shard_count == DENSE_SHARD_COUNT
    assert 0 <= shard_index < shard_count
    selected = tuple(
        sample for index, sample in enumerate(samples)
        if index % shard_count == shard_index)
    assert selected
    return selected


@lru_cache(maxsize=2)
def _route_cutter_shape(owner: str):
    """Exact final route void authority, built once per carrier owner."""
    from build123d import Compound
    import top_baffle_nd25fw4_obiwan as cad

    children = list(cad.route_inner_cutters(owner))
    if owner == "lm":
        children.append(cad.lm_free_lead_relief_cutter())
        children.extend(cad._lm_t_closure_handoff_cutters())
    return Compound(children=children)


@lru_cache(maxsize=4)
def _cropped_route_cutter_shape(owner: str, junction: str):
    """Keep only the exact route authority relevant to one fixed window."""
    import top_baffle_nd25fw4_obiwan as cad

    clip = cad._plan_prism(
        _independent_junction_window(junction).buffer(1.0, join_style=2),
        cad.CORE_REAR_Z - 0.01, cad.THICKNESS_MM + 0.01)
    return (_route_cutter_shape(owner) & clip).clean()


@lru_cache(maxsize=512)
def _route_void_plan(owner: str, junction: str, z_mm: float):
    from shapely.ops import unary_union

    polygons = _section_polygons(
        _cropped_route_cutter_shape(owner, junction), z_mm)
    return unary_union(polygons).buffer(0) if polygons else None


def _assembled_plan_oracle(cad, junction: str, z_mm: float):
    """Expected material from full webs plus complementary Z-half ears."""
    from shapely.geometry import Point
    from shapely.ops import unary_union

    record = cad.junction_closure_polygons()[junction]
    if junction == "lm_um":
        lm_ears = unary_union([
            cad._complete_joint_ear_plan("lm", x)
            for x in cad.JOINT_EAR_X
        ])
        um_ears = unary_union([
            cad._complete_joint_ear_plan("um", x)
            for x in cad.JOINT_EAR_X
        ])
        lm_receiver = unary_union([
            cad._complete_joint_ear_plan(
                "lm", x, cad.JOINT_RECEIVER_RADIAL_CLEAR)
            for x in cad.JOINT_EAR_X
        ])
        um_receiver = unary_union([
            cad._complete_joint_ear_plan(
                "um", x, cad.JOINT_RECEIVER_RADIAL_CLEAR)
            for x in cad.JOINT_EAR_X
        ])
        clearance_holes = unary_union([
            Point(x, cad.JOINT_EAR_Y).buffer(
                cad.JOINT_CLEARANCE_BORE_D / 2.0, resolution=32)
            for x in cad.JOINT_EAR_X
        ])
        insert_holes = unary_union([
            Point(x, cad.JOINT_EAR_Y).buffer(
                cad.JOINT_INSERT_BORE_D / 2.0, resolution=32)
            for x in cad.JOINT_EAR_X
        ])
        active_clearances = []
        active_bores = []

        lm_ear_active = cad.LM_JOINT_Z[0] < z_mm < cad.LM_JOINT_Z[1]
        um_ear_active = cad.UM_JOINT_Z[0] < z_mm < cad.UM_JOINT_Z[1]
        lm_plan = record["lm"]
        if lm_ear_active:
            lm_plan = unary_union((lm_plan, lm_ears))
        # LM is completely relieved for the opposing UM ear from its rear
        # split plane onward. Where the UM ear is absent (the 0.20-mm axial
        # gap and the front overshoot), the whole receiver is intentional
        # open assembly space; otherwise only its radial fit clearance is.
        if (cad.LM_JOINT_Z[1] < z_mm
                < cad.UM_JOINT_Z[1] + cad.JOINT_RECEIVER_RADIAL_CLEAR):
            lm_plan = lm_plan.difference(um_receiver)
            active_clearances.append(
                (um_receiver.difference(um_ears).buffer(0)
                 if um_ear_active else um_receiver))
        lm_bore_active = (
            cad.CORE_REAR_Z - cad.JOINT_BORE_REAR_OVERSHOOT
            < z_mm < cad.JOINT_CLEARANCE_BORE_TOP_Z)
        if lm_bore_active:
            lm_plan = lm_plan.difference(clearance_holes)
            active_bores.append(clearance_holes)
        lm_plan = lm_plan.buffer(0)

        um_plan = record["um"]
        if um_ear_active:
            um_plan = unary_union((um_plan, um_ears))
        if (cad.LM_JOINT_Z[0] - cad.JOINT_RECEIVER_RADIAL_CLEAR
                < z_mm
                < cad.UM_JOINT_Z[0]):
            um_plan = um_plan.difference(lm_receiver)
            active_clearances.append(
                (lm_receiver.difference(lm_ears).buffer(0)
                 if lm_ear_active else lm_receiver))
        insert_bore_active = (
            cad.JOINT_INSERT_BORE_Z[0]
            < z_mm < cad.JOINT_INSERT_BORE_Z[1])
        if insert_bore_active:
            um_plan = um_plan.difference(insert_holes)
            active_bores.append(insert_holes)
        um_plan = um_plan.buffer(0)
        owner_plans = {"lm": lm_plan, "um": um_plan}
        functional_bores = unary_union(
            (*active_bores, *active_clearances)).buffer(0)
    elif junction == "t_um":
        um_ears = unary_union([
            cad._complete_tweeter_joint_ear_plan("um", x)
            for x in cad.TWEETER_JOINT_X
        ])
        tweeter_ears = unary_union([
            cad._complete_tweeter_joint_ear_plan("tweeter", x)
            for x in cad.TWEETER_JOINT_X
        ])
        core_receiver = unary_union([
            cad._complete_tweeter_joint_ear_plan(
                "um", x, cad.TWEETER_JOINT_CLEAR)
            for x in cad.TWEETER_JOINT_X
        ])
        addon_receiver = unary_union([
            cad._complete_tweeter_joint_ear_plan(
                "tweeter", x, cad.TWEETER_JOINT_CLEAR)
            for x in cad.TWEETER_JOINT_X
        ])
        core_holes = unary_union([
            Point(x, cad.TWEETER_JOINT_Y).buffer(
                cad.TWEETER_JOINT_HOLE_D / 2.0, resolution=32)
            for x in cad.TWEETER_JOINT_X
        ])
        insert_holes = unary_union([
            Point(x, cad.TWEETER_JOINT_Y).buffer(
                cad.TWEETER_JOINT_INSERT_BORE_D / 2.0, resolution=32)
            for x in cad.TWEETER_JOINT_X
        ])
        active_clearances = []
        um_plan = record["um"]
        um_ear_active = (
            cad.TWEETER_CORE_JOINT_Z[0]
            < z_mm < cad.TWEETER_CORE_JOINT_Z[1])
        if um_ear_active:
            um_plan = unary_union((um_plan, um_ears))
        if (cad.TWEETER_CORE_JOINT_Z[1]
                < z_mm
                < cad.TWEETER_ADDON_JOINT_Z[1]
                + cad.TWEETER_JOINT_CLEAR):
            um_plan = um_plan.difference(addon_receiver)
            active_clearances.append(
                (addon_receiver.difference(tweeter_ears).buffer(0)
                 if (cad.TWEETER_ADDON_JOINT_Z[0]
                     < z_mm < cad.TWEETER_ADDON_JOINT_Z[1])
                 else addon_receiver))
        core_holes_active = (cad.TWEETER_CORE_JOINT_Z[0] - 0.2
                             <= z_mm
                             <= cad.TWEETER_CORE_BORE_TOP_Z)
        if core_holes_active:
            um_plan = um_plan.difference(core_holes)
        um_plan = um_plan.buffer(0)

        tweeter_plan = record["tweeter"]
        tweeter_ear_active = (
            cad.TWEETER_ADDON_JOINT_Z[0]
            < z_mm < cad.TWEETER_ADDON_JOINT_Z[1])
        if tweeter_ear_active:
            tweeter_plan = unary_union((tweeter_plan, tweeter_ears))
        if (cad.TWEETER_CORE_JOINT_Z[0] - cad.TWEETER_JOINT_CLEAR
                < z_mm < cad.TWEETER_ADDON_JOINT_Z[0]):
            tweeter_plan = tweeter_plan.difference(core_receiver)
            active_clearances.append(
                (core_receiver.difference(um_ears).buffer(0)
                 if um_ear_active else core_receiver))
        # The rear-driven M3 service path crosses both independently printed
        # owners.  The full-depth tweeter web must therefore carry the same
        # core-hole recut as the UM half before it widens into the blind
        # insert receiver.
        if core_holes_active:
            tweeter_plan = tweeter_plan.difference(core_holes)
        insert_holes_active = (
            cad.TWEETER_JOINT_INSERT_BORE_Z[0]
            < z_mm < cad.TWEETER_JOINT_INSERT_BORE_Z[1])
        if insert_holes_active:
            tweeter_plan = tweeter_plan.difference(insert_holes)
        tweeter_plan = tweeter_plan.buffer(0)
        owner_plans = {"um": um_plan, "tweeter": tweeter_plan}
        active_bores = []
        if core_holes_active:
            active_bores.append(core_holes)
        if insert_holes_active:
            active_bores.append(insert_holes)
        active_bores.extend(active_clearances)
        functional_bores = (unary_union(active_bores).buffer(0)
                            if active_bores else
                            record["target"].difference(record["target"]))
    else:
        raise ValueError(junction)

    route_plans = []
    for owner, plan in tuple(owner_plans.items()):
        if owner not in {"lm", "um"}:
            continue
        route_plan = _route_void_plan(owner, junction, z_mm)
        if route_plan is not None and not route_plan.is_empty:
            owner_plans[owner] = plan.difference(route_plan).buffer(0)
            route_plans.append(route_plan)
    route_union = (unary_union(route_plans).buffer(0)
                   if route_plans else record["target"].difference(
                       record["target"]))
    material = unary_union(tuple(owner_plans.values())).intersection(
        record["audit_domain"])
    expected_void = record["audit_domain"].difference(material).buffer(0)
    terminal_drain = record["terminal_drain"]
    return owner_plans, expected_void, unary_union(
        (functional_bores, route_union, terminal_drain)).buffer(0)


def test_assembled_sections_match_complementary_ownership_through_depth():
    """Every print layer has the frozen material and no bounded void island."""
    _require_guarded_test()
    from shapely.ops import unary_union
    import top_baffle_nd25fw4_obiwan as cad

    selected_case = os.environ.get(DENSE_CASE_ENV, "").strip()
    if selected_case:
        assert selected_case in DENSE_CASES, (
            f"unknown {DENSE_CASE_ENV}={selected_case!r}; expected one of "
            f"{', '.join(DENSE_CASES)}")
        selected_state, selected_junction = DENSE_CASES[selected_case]
    else:
        selected_state = selected_junction = None
    samples = _dense_shard_samples(
        _dense_section_samples(cad),
        os.environ.get(DENSE_SHARD_ENV, "").strip())
    pairs = {
        "lm_um": ("lm", "um"),
        "t_um": ("um", "tweeter"),
    }
    section_cache = {}
    for junction, owners in pairs.items():
        if selected_junction is not None and junction != selected_junction:
            continue
        record = cad.junction_closure_polygons()[junction]
        fit_seam = record["fit_seam"]
        independent_window = _independent_junction_window(junction)
        frozen_required = _frozen_required_front_domain(junction)
        assert independent_window.covers(frozen_required)
        assemblies = [
            ("no_floor", _cropped_junction_parts(False, junction)),
            ("floor", _cropped_junction_parts(True, junction)),
        ]
        for state, state_parts in assemblies:
            if selected_state is not None and state != selected_state:
                continue
            for z_mid in samples:
                plans, expected_void_plan, bounded_allowed = (
                    _assembled_plan_oracle(cad, junction, z_mid))
                section_plans = {}
                for owner in owners:
                    key = (state, junction, owner, z_mid)
                    if key not in section_cache:
                        section_cache[key] = _section_material_plan(
                            state_parts[owner], z_mid)
                    section_plans[owner] = section_cache[key]
                for owner in owners:
                    actual_owner = section_plans[owner].intersection(
                        record["audit_domain"]).buffer(0)
                    expected_owner = plans[owner].intersection(
                        record["audit_domain"]).buffer(0)
                    owner_extra_geometry = actual_owner.difference(
                        expected_owner.buffer(0.01, join_style=1)).buffer(0)
                    owner_missing_geometry = expected_owner.difference(
                        actual_owner.buffer(0.01, join_style=1)).buffer(0)
                    owner_extra = owner_extra_geometry.area
                    owner_missing = owner_missing_geometry.area
                    assert owner_extra <= 0.02 and owner_missing <= 0.02, (
                        f"{state}/{junction}/{owner} z={z_mid:.2f}: "
                        f"owner mismatch extra={owner_extra:.6f} "
                        f"missing={owner_missing:.6f} mm2 "
                        f"extra_bounds={owner_extra_geometry.bounds} "
                        f"missing_bounds={owner_missing_geometry.bounds}")
                collision = section_plans[owners[0]].intersection(
                    section_plans[owners[1]]).intersection(
                        independent_window).area
                assert collision <= 0.01, (
                    f"{state}/{junction} z={z_mid:.2f}: complementary "
                    f"owners overlap by {collision:.6f} mm2")
                assembled_plan = unary_union(
                    tuple(section_plans.values())).buffer(0)
                actual_material = assembled_plan.intersection(
                    record["audit_domain"]).buffer(0)
                expected_material = unary_union(
                    tuple(plans.values())).intersection(
                        record["audit_domain"]).buffer(0)
                extra = actual_material.difference(
                    expected_material.buffer(0.01, join_style=1)).area
                missing = expected_material.difference(
                    actual_material.buffer(0.01, join_style=1)).area
                assert extra <= 0.02 and missing <= 0.02, (
                    f"{state}/{junction} z={z_mid:.2f}: "
                    f"actual/declared section mismatch "
                    f"extra={extra:.6f} missing={missing:.6f} mm2")

                # Independent material oracle: this conservative silhouette
                # is frozen test data, not output from the current web
                # generator.  It catches an open cusp even when that notch is
                # connected to a driver aperture and therefore is not a
                # Polygon interior ring.
                declared_open = unary_union((
                    bounded_allowed,
                    fit_seam,
                    record["terminal_drain"],
                )).buffer(0)
                required_here = frozen_required.difference(
                    declared_open.buffer(0.012, join_style=1)).buffer(0)
                frozen_missing = required_here.difference(
                    assembled_plan.buffer(0.012, join_style=1)).area
                assert frozen_missing <= 0.02, (
                    f"{state}/{junction} z={z_mid:.2f}: frozen physical "
                    f"closure missing {frozen_missing:.6f} mm2")

                # Complement topology in the fixed physical window.  Driver
                # apertures/exterior gaps reach the fixed rectangle boundary;
                # every strictly interior component must be an explicitly
                # active receiver, fastener bore, route lumen, or fit seam.
                actual_void = independent_window.difference(
                    assembled_plan).buffer(0)
                for component in _pieces(actual_void):
                    if component.distance(
                            independent_window.boundary) <= 0.012:
                        continue
                    undeclared = component.difference(
                        declared_open.buffer(0.025, join_style=1)).buffer(0)
                    assert (undeclared.is_empty
                            or undeclared.area <= 0.01), (
                        f"{state}/{junction} z={z_mid:.2f}: bounded "
                        f"front-plane void island area={component.area:.6f} "
                        f"undeclared={undeclared.area:.6f} mm2 "
                        f"bounds={component.bounds}")
                for component in _pieces(expected_void_plan):
                    if bounded_allowed.buffer(0.02).covers(component):
                        continue
                    assert component.distance(
                        record["audit_domain"].boundary) <= 0.012, (
                        f"{state}/{junction} z={z_mid:.2f}: bounded "
                        f"undeclared void area={component.area:.6f} mm2")


def test_brep_assembled_front_sections_match_declared_voids_only() -> None:
    """Actual front sections equal the seam/receiver/hole/mouth contract."""
    _require_guarded_test()
    from build123d import Compound
    import top_baffle_nd25fw4_obiwan as cad

    actual = obiwan_actual_parts()
    pairs = {
        "lm_um": ("lm", "um"),
        "t_um": ("um", "tweeter"),
    }
    z0 = cad.THICKNESS_MM - 0.08
    z1 = cad.THICKNESS_MM - 0.02
    z_mid = 0.5 * (z0 + z1)

    for junction, owners in pairs.items():
        audit, _owner_plans, expected_void_plan, _allowed_void = (
            _front_plan_contract(cad, junction))
        collision_clip = cad._plan_prism(audit, *cad.JUNCTION_WEB_Z)
        first = (actual[owners[0]] & collision_clip).clean()
        second = (actual[owners[1]] & collision_clip).clean()
        overlap = (first & second).clean()
        overlap_volume = sum(solid.volume for solid in overlap.solids())
        assert overlap_volume <= 0.02, (
            f"{owners[0]}/{owners[1]}: complementary owners collide by "
            f"{overlap_volume:.6f} mm3")
        assert not expected_void_plan.is_empty
        required = cad._plan_prism(audit, z0, z1)
        # Intersect first so this is explicitly the union of the two actual
        # independently printable front-slab members, not an analytic proxy.
        members = []
        for owner in owners:
            section_member = (actual[owner] & required).clean()
            if section_member.volume > 1.0e-6:
                members.append(section_member)
        assert len(members) == 2
        assembled_section = Compound(children=members)
        residual = (required - assembled_section).clean()

        actual_missing = sum(solid.volume for solid in residual.solids())
        expected_void = cad._plan_prism(expected_void_plan, z0, z1)
        expected_missing = expected_void.volume
        assert math.isclose(
            actual_missing, expected_missing,
            rel_tol=0.03, abs_tol=0.015), (
            f"{junction}: assembled missing {actual_missing:.6f} mm3 vs "
            f"declared voids {expected_missing:.6f} mm3")

        # Coincident thin OCC prisms can report an empty intersection and
        # fragment their set difference into hundreds of slivers even when
        # their middle sections agree.  Volume equality above plus this
        # bidirectional reconstructed section comparison is the stable exact
        # contract: no extra void and no silently filled seam/service void.
        residual_section = _section_material_plan(residual, z_mid)
        expected_void_section = _section_material_plan(expected_void, z_mid)
        extra_void_area = residual_section.difference(
            expected_void_section).area
        filled_void_area = expected_void_section.difference(
            residual_section).area
        assert extra_void_area <= 0.015, (
            f"{junction}: unexpected assembled front-section void "
            f"{extra_void_area:.6f} mm2")
        assert filled_void_area <= 0.015, (
            f"{junction}: declared fit/service front-section void was "
            f"filled by {filled_void_area:.6f} mm2")

        residual_plans = _pieces(residual_section)
        expected_components = _pieces(expected_void_section)
        assert len(residual_plans) == len(expected_components), (
            f"{junction}: {len(residual_plans)} residual components vs "
            f"{len(expected_components)} declared components")
        for component in residual_plans:
            assert expected_void_section.buffer(
                0.02, join_style=1).covers(component), (
                f"{junction}: undeclared assembled residual "
                f"area={component.area:.6f} mm2")


def test_fixed_windows_have_no_3d_collision_and_fronts_are_coplanar() -> None:
    """Independent print owners never overlap or project above z=18.3."""
    _require_guarded_test()
    import top_baffle_nd25fw4_obiwan as cad

    pairs = {
        "lm_um": ("lm", "um"),
        "t_um": ("um", "tweeter"),
    }
    for stand_foot, state in ((False, "no_floor"), (True, "floor")):
        actual = obiwan_actual_parts(stand_foot)
        for owner in {name for pair in pairs.values() for name in pair}:
            bounds = actual[owner].bounding_box()
            assert math.isclose(
                bounds.max.Z, cad.THICKNESS_MM, abs_tol=1.0e-6), (
                f"{state}/{owner}: front datum is not exactly coplanar, "
                f"maxZ={bounds.max.Z:.9f}")
            assert bounds.max.Z <= cad.THICKNESS_MM + 1.0e-6, (
                f"{state}/{owner}: proud material above the front datum")
        for junction, owners in pairs.items():
            clip = cad._plan_prism(
                _independent_junction_window(junction),
                cad.CORE_REAR_Z, cad.THICKNESS_MM)
            first = (actual[owners[0]] & clip).clean()
            second = (actual[owners[1]] & clip).clean()
            overlap = (first & second).clean()
            volume = sum(solid.volume for solid in overlap.solids())
            assert volume <= 0.02, (
                f"{state}/{junction}: exact fixed-window 3-D owner "
                f"collision {volume:.6f} mm3")

FULL_CHECKS = (
    test_analytic_closure_plans_have_only_open_assembly_seams,
    test_brep_web_prisms_are_solid_through_the_complete_obiwan_depth,
    test_lm_um_individual_breps_own_complete_fastener_features,
    test_um_tweeter_individual_breps_own_complete_fastener_features,
    test_final_owner_breps_keep_safe_web_interiors_solid_full_depth,
    test_brep_front_plane_contains_every_expected_owner_region,
    test_brep_assembled_front_sections_match_declared_voids_only,
    test_fixed_windows_have_no_3d_collision_and_fronts_are_coplanar,
    test_assembled_sections_match_complementary_ownership_through_depth,
)


def main() -> int:
    test_source_owns_full_depth_webs_before_functional_recuts()
    test_make_jobserver_owns_the_complete_dense_matrix()
    test_dense_shards_are_an_exact_disjoint_partition()
    print("PASS source ownership/order/front-face-down contract", flush=True)
    if not _guarded_build():
        if os.environ.get(FULL_TEST_ENV, "").strip().lower() in {
                "1", "true", "yes", "on"}:
            raise RuntimeError(
                f"{FULL_TEST_ENV}=1 requires authenticated guarded CAD")
        print(
            "SKIP guarded analytic/BREP closure checks; run through the "
            "remote CAD guard or set LX_OBIWAN_CLOSURE_FULL_TEST=1 to require "
            "them",
            flush=True,
        )
        return 0
    dense_case = os.environ.get(DENSE_CASE_ENV, "").strip()
    if dense_case:
        assert dense_case in DENSE_CASES, (
            f"unknown {DENSE_CASE_ENV}={dense_case!r}; expected one of "
            f"{', '.join(DENSE_CASES)}")
        checks = (
            test_assembled_sections_match_complementary_ownership_through_depth,
        )
    elif os.environ.get(PLAN_ONLY_ENV, "").strip().lower() in {
            "1", "true", "yes", "on"}:
        checks = FULL_CHECKS[:2]
    elif os.environ.get(BASE_ONLY_ENV, "").strip().lower() in {
            "1", "true", "yes", "on"}:
        checks = FULL_CHECKS[:-1]
    else:
        checks = FULL_CHECKS
    for check in checks:
        check()
        print(f"PASS {check.__name__}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
