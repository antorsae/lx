"""Conservative analytical screen for the integral V1LF floor stand.

This is deliberately not presented as finite-element analysis.  It is a
closed-form beam/section screen tied to the production stand dimensions and
to the three buried floor-lane lumens.  Manufacturer coupon values are kept
separate from the project allowables: Bambu's specimens are controlled,
100%-infill coupons and do not qualify an arbitrary unannealed printed stand.

The report also contains a deliberately incomplete shoulder/ring diagnostic:
it puts the full design load through only the two uninterrupted printed outer
lip ligaments at the lower D190 tangent.  That is a conservative lower bound
on one parallel path, not an FEA result or a failure prediction for the
installed driver-flange-reinforced assembly.

The installed driver flange, not the concealed alignment key, must bridge the
optional LM split.  The key receives zero structural credit and the split
configuration remains subject to the physical proof/creep gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


SCHEMA_VERSION = 2
GRAVITY_M_S2 = 9.80665
DESIGN_MASS_KG = 4.0
DESIGN_CG_Y_MM = 230.0
DESIGN_REAR_CG_MM = 70.0

FLOOR_Y_MM = 0.0
LM_AXIS_Y_MM = 200.981
FOOT_WIDTH_MM = 64.0
FOOT_HEIGHT_MM = 18.3
FOOT_REAR_Z_MM = -150.0
FOOT_FRONT_Z_MM = 18.3
ROOT_FILLET_R_MM = 12.0
ROOT_SECTION_Y_MM = FOOT_HEIGHT_MM + ROOT_FILLET_R_MM
STEM_EFFECTIVE_LENGTH_MM = LM_AXIS_Y_MM - FOOT_HEIGHT_MM
SOLID_MODIFIER_MAX_Y_MM = LM_AXIS_Y_MM - 95.0
# Explicit local-geometry/model uncertainty applied to every nominal section
# stress.  It is not a substitute for a notch FEA: the R12 root and smooth
# longitudinal bores justify a modest factor, while the physical proof gate
# remains authoritative for printed-layer and lumen-edge effects.
ROOT_STRESS_CONCENTRATION_FACTOR = 1.25

# Printed shoulder-to-ring diagnostic.  These are the exact R6F LM carrier
# dimensions at the lower D190 tangent where the integral stem reaches the
# ring.  The diagnostic deliberately credits only the two uninterrupted
# outer-lip ligaments.  It does not credit the 0.85-mm seat membrane, the
# integrated shoulder below the tangent, route-cover material, insert
# bosses, the installed metal driver flange or magnets.  That makes the
# result a conservative *partial-load-path lower bound*, not a prediction of
# the strength of the complete installed assembly.
LM_RING_CUTOUT_R_MM = 95.0
LM_RING_LIP_INNER_R_MM = 110.6
LM_RING_LIP_OUTER_R_MM = 113.0
LM_RING_LIP_REAR_Z_MM = 6.8
LM_RING_FRONT_Z_MM = 18.3
SHOULDER_RING_TANGENT_Y_MM = LM_AXIS_Y_MM - LM_RING_CUTOUT_R_MM

# Exact floor-lane sections at the first straight stem section above the
# R12 root blend.  The D6 T lumen carries the same shared tweeter bundle as
# the already-released LM/UM carrier route; it is not split into two weaker
# floor bores.
ROOT_LUMENS = (
    {"name": "lm", "diameter_mm": 9.0,
     "center_x_mm": 0.0, "center_z_mm": 12.55},
    {"name": "um", "diameter_mm": 8.2,
     "center_x_mm": 11.0, "center_z_mm": 12.55},
    {"name": "t", "diameter_mm": 6.0,
     "center_x_mm": -11.0, "center_z_mm": 6.20},
)

# These conservative design inputs use weak-direction lower bounds and
# explicit unannealed/process/creep/environment derating.  They are project
# assumptions, not values certified by Bambu Lab.  Lite is additionally
# fail-closed because no product-specific official TDS was available.
MATERIALS = {
    "Bambu PLA Tough+": {
        "source_url": (
            "https://store.bblcdn.eu/s8/default/"
            "f0874452d01249dba4ab6fc68ca972e4/"
            "BambuPLA_Tough_TechnicalData_Sheet_%282%29.pdf"),
        "source_revision": "TDS V3",
        "density_g_cm3": 1.21,
        "z_tensile_strength_mpa": 20.9,
        "z_flexural_strength_mpa": 54.0,
        "z_flexural_modulus_mpa": 2066.0,
        "design_flexural_modulus_mpa": 1280.0,
        "sustained_allowable_mpa": 3.4,
        "transient_allowable_mpa": 6.6,
        "provisional": False,
    },
    "Bambu PLA Basic": {
        "source_url": (
            "https://store.bblcdn.eu/s8/default/"
            "073e722a4aa44f7cbfdc419d597475cc/"
            "Bambu_PLA_Basic_Technical_Data_Sheet.pdf"),
        "source_revision": "TDS V3",
        "density_g_cm3": 1.24,
        "z_tensile_strength_mpa": 31.0,
        "z_flexural_strength_mpa": 59.0,
        "z_flexural_modulus_mpa": 2370.0,
        "design_flexural_modulus_mpa": 1440.0,
        "sustained_allowable_mpa": 4.9,
        "transient_allowable_mpa": 9.3,
        "provisional": False,
    },
    "Bambu PLA Lite": {
        "source_url": (
            "https://store.bblcdn.com/s7/default/"
            "ecb663b46ebb4fb984786d33befb8d2f/PLA_Pure_TDS.pdf"),
        "source_revision": "PLA Pure comparison TDS V1",
        "density_g_cm3": 1.40,
        "z_tensile_strength_mpa": None,
        "z_flexural_strength_mpa": 25.1,
        "z_flexural_modulus_mpa": 2069.8,
        "design_flexural_modulus_mpa": 1080.0,
        "sustained_allowable_mpa": 3.0,
        "transient_allowable_mpa": 5.8,
        "provisional": True,
        "source_warning": (
            "no product-specific official Lite TDS found; comparison PDF "
            "has an apparent strength/modulus row-label transposition"),
    },
    "Bambu PLA Matte": {
        "source_url": (
            "https://store.bblcdn.eu/s8/default/"
            "82bab351a9494e318ab485f7c31a01b3/"
            "Bambu_PLA_Matte_Technical_Data_Sheet.pdf"),
        "source_revision": "TDS V3",
        "density_g_cm3": 1.31,
        "z_tensile_strength_mpa": 22.0,
        "z_flexural_strength_mpa": 29.0,
        "z_flexural_modulus_mpa": 1770.0,
        "design_flexural_modulus_mpa": 1010.0,
        "sustained_allowable_mpa": 3.1,
        "transient_allowable_mpa": 6.0,
        "provisional": False,
    },
    "Bambu PLA Silk+": {
        "source_url": (
            "https://store.bblcdn.eu/s8/default/"
            "d0de0f57694b406dbf3e9b2345b7dbb9/"
            "Bambu_PLA_Silk__Technical_Data_Sheet.pdf"),
        "source_revision": "TDS V1",
        "density_g_cm3": 1.27,
        "z_tensile_strength_mpa": 25.0,
        "z_flexural_strength_mpa": 30.0,
        "z_flexural_modulus_mpa": 2150.0,
        "design_flexural_modulus_mpa": 1290.0,
        "sustained_allowable_mpa": 3.6,
        "transient_allowable_mpa": 7.0,
        "provisional": False,
    },
}

MIN_SF_1G_SUSTAINED = 2.0
MIN_SF_3G_TRANSIENT = 1.5
MIN_SF_5G_TRANSIENT = 1.05
MAX_DIAGNOSTIC_DEFLECTION_1G_MM = 2.0


def _net_root_section() -> dict:
    """Exact rectangle-minus-circles section about the world X axis."""
    width = FOOT_WIDTH_MM
    depth = FOOT_HEIGHT_MM
    gross_area = width * depth
    gross_q = gross_area * depth / 2.0
    net_area = gross_area
    net_q = gross_q
    for lumen in ROOT_LUMENS:
        diameter = lumen["diameter_mm"]
        area = math.pi * diameter ** 2 / 4.0
        net_area -= area
        net_q -= area * lumen["center_z_mm"]
    centroid_z = net_q / net_area
    inertia_x = (
        width * depth ** 3 / 12.0
        + gross_area * (depth / 2.0 - centroid_z) ** 2)
    inertia_z = depth * width ** 3 / 12.0
    for lumen in ROOT_LUMENS:
        diameter = lumen["diameter_mm"]
        area = math.pi * diameter ** 2 / 4.0
        inertia_x -= (
            math.pi * diameter ** 4 / 64.0
            + area * (lumen["center_z_mm"] - centroid_z) ** 2)
        inertia_z -= (
            math.pi * diameter ** 4 / 64.0
            + area * lumen["center_x_mm"] ** 2)
    rear_distance = centroid_z
    front_distance = depth - centroid_z
    section_modulus = inertia_x / max(rear_distance, front_distance)
    return {
        "gross_area_mm2": gross_area,
        "net_area_mm2": net_area,
        "centroid_z_mm": centroid_z,
        "second_moment_x_mm4": inertia_x,
        "second_moment_z_mm4": inertia_z,
        "rear_section_modulus_mm3": inertia_x / rear_distance,
        "front_section_modulus_mm3": inertia_x / front_distance,
        "governing_section_modulus_mm3": section_modulus,
        "lateral_section_modulus_mm3": inertia_z / (width / 2.0),
        "net_area_ratio": net_area / gross_area,
        "lumens": ROOT_LUMENS,
    }


def _shoulder_ring_lip_section() -> dict:
    """Two-ligament lower-bound section at the lower D190 tangent.

    A horizontal cut immediately on the ring side of the integral shoulder
    intersects the R110.6..R113 outer lip twice.  At radial ordinate ``dy``
    each half-ligament has width::

        sqrt(R_outer**2 - dy**2) - sqrt(R_inner**2 - dy**2)

    The credited ligaments are rectangular in XZ and share the exact
    z=6.8..18.3 outer-lip depth.  Since the two rectangles have the same Z
    extent, their rear-eccentric section properties equal one rectangle with
    their combined width.  Omitting every other parallel load path can only
    reduce credited section capacity, but it also means a failed threshold
    cannot by itself establish failure of the installed, flange-reinforced
    assembly.
    """
    dy = LM_AXIS_Y_MM - SHOULDER_RING_TANGENT_Y_MM
    if not (0.0 <= dy < LM_RING_LIP_INNER_R_MM):
        raise RuntimeError("LM shoulder tangent misses the outer ring lip")
    inner_x = math.sqrt(LM_RING_LIP_INNER_R_MM ** 2 - dy ** 2)
    outer_x = math.sqrt(LM_RING_LIP_OUTER_R_MM ** 2 - dy ** 2)
    half_width = outer_x - inner_x
    combined_width = 2.0 * half_width
    depth = LM_RING_FRONT_Z_MM - LM_RING_LIP_REAR_Z_MM
    area = combined_width * depth
    centroid_z = (
        LM_RING_LIP_REAR_Z_MM + LM_RING_FRONT_Z_MM) / 2.0
    inertia_x = combined_width * depth ** 3 / 12.0
    section_modulus = inertia_x / (depth / 2.0)
    return {
        "plane_y_mm": SHOULDER_RING_TANGENT_Y_MM,
        "radial_ordinate_mm": dy,
        "lip_inner_radius_mm": LM_RING_LIP_INNER_R_MM,
        "lip_outer_radius_mm": LM_RING_LIP_OUTER_R_MM,
        "inner_intersection_x_mm": inner_x,
        "outer_intersection_x_mm": outer_x,
        "half_ligament_width_mm": half_width,
        "ligament_count": 2,
        "combined_width_mm": combined_width,
        "rear_z_mm": LM_RING_LIP_REAR_Z_MM,
        "front_z_mm": LM_RING_FRONT_Z_MM,
        "depth_mm": depth,
        "net_area_mm2": area,
        "centroid_z_mm": centroid_z,
        "second_moment_x_mm4": inertia_x,
        "governing_section_modulus_mm3": section_modulus,
    }


def _stress_for_g(section: dict, g_load: float) -> dict:
    force = DESIGN_MASS_KG * GRAVITY_M_S2 * g_load
    moment = force * DESIGN_REAR_CG_MM
    axial_nominal = force / section["net_area_mm2"]
    bending_nominal = moment / section["governing_section_modulus_mm3"]
    shear_nominal = 1.5 * force / section["net_area_mm2"]
    nominal = axial_nominal + bending_nominal
    return {
        "force_n": force,
        "rear_moment_nmm": moment,
        "axial_stress_nominal_mpa": axial_nominal,
        "bending_stress_nominal_mpa": bending_nominal,
        "combined_normal_stress_nominal_mpa": nominal,
        "stress_concentration_factor": ROOT_STRESS_CONCENTRATION_FACTOR,
        "combined_normal_stress_design_mpa": (
            nominal * ROOT_STRESS_CONCENTRATION_FACTOR),
        "max_shear_stress_design_mpa": (
            shear_nominal * ROOT_STRESS_CONCENTRATION_FACTOR),
    }


def _junction_stress_for_g(section: dict, g_load: float) -> dict:
    """Rear-eccentric normal/shear stress in the credited lip ligaments."""
    force = DESIGN_MASS_KG * GRAVITY_M_S2 * g_load
    moment = force * DESIGN_REAR_CG_MM
    axial_nominal = force / section["net_area_mm2"]
    bending_nominal = (
        moment / section["governing_section_modulus_mm3"])
    shear_nominal = 1.5 * force / section["net_area_mm2"]
    nominal = axial_nominal + bending_nominal
    return {
        "force_n": force,
        "rear_moment_nmm": moment,
        "axial_stress_nominal_mpa": axial_nominal,
        "bending_stress_nominal_mpa": bending_nominal,
        "combined_normal_stress_nominal_mpa": nominal,
        "stress_concentration_factor": ROOT_STRESS_CONCENTRATION_FACTOR,
        "combined_normal_stress_design_mpa": (
            nominal * ROOT_STRESS_CONCENTRATION_FACTOR),
        "max_shear_stress_design_mpa": (
            shear_nominal * ROOT_STRESS_CONCENTRATION_FACTOR),
    }


def _shoulder_ring_junction_facts() -> dict:
    """Conservative diagnostic independent of the root pass/fail result."""
    section = _shoulder_ring_lip_section()
    loads = {
        str(g): _junction_stress_for_g(section, float(g))
        for g in (1, 3, 5)
    }
    materials = {}
    for name, source in MATERIALS.items():
        sf_1g = (
            source["sustained_allowable_mpa"]
            / loads["1"]["combined_normal_stress_design_mpa"])
        sf_3g = (
            source["transient_allowable_mpa"]
            / loads["3"]["combined_normal_stress_design_mpa"])
        sf_5g = (
            source["transient_allowable_mpa"]
            / loads["5"]["combined_normal_stress_design_mpa"])
        materials[name] = {
            "sf_1g_sustained": sf_1g,
            "sf_3g_transient": sf_3g,
            "sf_5g_transient": sf_5g,
            "lower_bound_meets_root_thresholds": (
                sf_1g >= MIN_SF_1G_SUSTAINED
                and sf_3g >= MIN_SF_3G_TRANSIENT
                and sf_5g >= MIN_SF_5G_TRANSIENT),
            "provisional": source["provisional"],
        }
    return {
        "analysis_kind": (
            "conservative_unreinforced_outer_lip_lower_bound_diagnostic"),
        "section": section,
        "loads": loads,
        "materials": materials,
        "equations": {
            "half_ligament_width": (
                "sqrt(R_outer^2-dy^2)-sqrt(R_inner^2-dy^2)"),
            "area": "2*half_ligament_width*lip_depth",
            "second_moment_x": (
                "(2*half_ligament_width)*lip_depth^3/12"),
            "section_modulus": "I_x/(lip_depth/2)",
            "design_normal_stress": (
                "K_t*(F/A + F*rear_eccentricity/S_x)"),
        },
        "credited_load_path": (
            "two uninterrupted R110.6..R113 printed outer-lip ligaments"),
        "excluded_from_credit": (
            "0.85-mm seat membrane",
            "integral shoulder material below the tangent",
            "route covers and insert bosses",
            "installed LM driver flange and fasteners",
            "magnets",
        ),
        "interpretation": {
            "changes_root_analytical_screen_pass": False,
            "complete_assembly_failure_prediction": False,
            "installed_lm_flange_required": True,
            "physical_proof_and_creep_gate_required": True,
            "threshold_result": (
                "DIAGNOSTIC_LOWER_BOUND_BELOW_THRESHOLDS"
                if not all(
                    item["lower_bound_meets_root_thresholds"]
                    for item in materials.values())
                else "DIAGNOSTIC_LOWER_BOUND_MEETS_THRESHOLDS"),
        },
        "limitations": (
            "vertical load plus rear eccentricity only; anchored lateral "
            "load remains screened at the full root section",
            "the deliberately omitted installed metal flange is a parallel "
            "load path and must be present for service",
            "a below-threshold lower bound mandates physical proof but does "
            "not establish failure of the complete installed assembly",
        ),
    }


def _lateral_stress_for_g(section: dict, g_load: float) -> dict:
    """Anchored lateral acceleration; stability is screened separately."""
    force = DESIGN_MASS_KG * GRAVITY_M_S2 * g_load
    moment = force * (DESIGN_CG_Y_MM - ROOT_SECTION_Y_MM)
    nominal = moment / section["lateral_section_modulus_mm3"]
    return {
        "force_n": force,
        "root_moment_nmm": moment,
        "bending_stress_nominal_mpa": nominal,
        "stress_concentration_factor": ROOT_STRESS_CONCENTRATION_FACTOR,
        "bending_stress_design_mpa": (
            nominal * ROOT_STRESS_CONCENTRATION_FACTOR),
        "qualification_condition": "anchored_or_anti_tip_restrained",
    }


def integral_floor_strength_facts() -> dict:
    section = _net_root_section()
    loads = {str(g): _stress_for_g(section, float(g)) for g in (1, 3, 5)}
    shoulder_ring_junction = _shoulder_ring_junction_facts()
    lateral_loads = {
        str(g): _lateral_stress_for_g(section, float(g))
        for g in (1, 3, 5)
    }
    materials = {}
    for name, source in MATERIALS.items():
        stress_1g = loads["1"]["combined_normal_stress_design_mpa"]
        stress_3g = loads["3"]["combined_normal_stress_design_mpa"]
        stress_5g = loads["5"]["combined_normal_stress_design_mpa"]
        lateral_1g = lateral_loads["1"]["bending_stress_design_mpa"]
        lateral_3g = lateral_loads["3"]["bending_stress_design_mpa"]
        lateral_5g = lateral_loads["5"]["bending_stress_design_mpa"]
        modulus = source["design_flexural_modulus_mpa"]
        deflection = (
            loads["1"]["rear_moment_nmm"]
            * STEM_EFFECTIVE_LENGTH_MM ** 2
            / (2.0 * modulus * section["second_moment_x_mm4"]))
        sf_1g = source["sustained_allowable_mpa"] / stress_1g
        sf_3g = source["transient_allowable_mpa"] / stress_3g
        sf_5g = source["transient_allowable_mpa"] / stress_5g
        lateral_sf_1g = source["sustained_allowable_mpa"] / lateral_1g
        lateral_sf_3g = source["transient_allowable_mpa"] / lateral_3g
        lateral_sf_5g = source["transient_allowable_mpa"] / lateral_5g
        record = dict(source)
        record.update({
            "sf_1g_sustained": sf_1g,
            "sf_3g_transient": sf_3g,
            "sf_5g_transient": sf_5g,
            "anchored_lateral_sf_1g_sustained": lateral_sf_1g,
            "anchored_lateral_sf_3g_transient": lateral_sf_3g,
            "anchored_lateral_sf_5g_transient": lateral_sf_5g,
            "diagnostic_tip_deflection_1g_mm": deflection,
            "analytical_screen_pass": (
                sf_1g >= MIN_SF_1G_SUSTAINED
                and sf_3g >= MIN_SF_3G_TRANSIENT
                and sf_5g >= MIN_SF_5G_TRANSIENT
                and lateral_sf_1g >= MIN_SF_1G_SUSTAINED
                and lateral_sf_3g >= MIN_SF_3G_TRANSIENT
                and lateral_sf_5g >= MIN_SF_5G_TRANSIENT
                and deflection <= MAX_DIAGNOSTIC_DEFLECTION_1G_MM),
            "physical_qualification": "PENDING",
        })
        materials[name] = record

    rear_margin = abs(FOOT_REAR_Z_MM) - abs(DESIGN_REAR_CG_MM)
    front_margin = FOOT_FRONT_Z_MM + abs(DESIGN_REAR_CG_MM)
    return {
        "schema_version": SCHEMA_VERSION,
        "analysis_kind": "closed_form_net_section_screen_not_fea",
        "geometry": {
            "floor_y_mm": FLOOR_Y_MM,
            "lm_axis_y_mm": LM_AXIS_Y_MM,
            "lm_axis_to_floor_mm": LM_AXIS_Y_MM - FLOOR_Y_MM,
            "foot_width_mm": FOOT_WIDTH_MM,
            "foot_height_mm": FOOT_HEIGHT_MM,
            "foot_z_mm": (FOOT_REAR_Z_MM, FOOT_FRONT_Z_MM),
            "root_fillet_r_mm": ROOT_FILLET_R_MM,
            "root_section_y_mm": ROOT_SECTION_Y_MM,
            "stem_effective_length_mm": STEM_EFFECTIVE_LENGTH_MM,
            "root_stress_concentration_factor": (
                ROOT_STRESS_CONCENTRATION_FACTOR),
        },
        "design_load": {
            "mass_kg": DESIGN_MASS_KG,
            "cg_y_mm": DESIGN_CG_Y_MM,
            "rear_cg_mm": DESIGN_REAR_CG_MM,
            "load_cases_g": (1, 3, 5),
            "load_case_direction": "vertical proof load with rear eccentricity",
            "magnet_load_credit_n": 0.0,
            "concealed_split_key_load_credit_n": 0.0,
        },
        "net_root_section": section,
        "loads": loads,
        "anchored_lateral_loads": lateral_loads,
        "shoulder_ring_junction": shoulder_ring_junction,
        "acceptance_thresholds": {
            "min_sf_1g_sustained": MIN_SF_1G_SUSTAINED,
            "min_sf_3g_transient": MIN_SF_3G_TRANSIENT,
            "min_sf_5g_transient": MIN_SF_5G_TRANSIENT,
            "max_diagnostic_deflection_1g_mm": (
                MAX_DIAGNOSTIC_DEFLECTION_1G_MM),
        },
        "materials": materials,
        "stability": {
            "lateral_half_width_mm": FOOT_WIDTH_MM / 2.0,
            "rear_margin_mm": rear_margin,
            "front_margin_mm": front_margin,
            "lateral_tip_acceleration_g": (
                (FOOT_WIDTH_MM / 2.0) / DESIGN_CG_Y_MM),
            "rear_tip_acceleration_g": rear_margin / DESIGN_CG_Y_MM,
            "front_tip_acceleration_g": front_margin / DESIGN_CG_Y_MM,
            "warning": (
                "strength load cases do not qualify free-standing lateral "
                "stability; anchor or provide an anti-tip system"),
        },
        "split_configuration": {
            "stand_owner": "optional_lm_keyed_1of2_bottom",
            "concealed_key_structural_credit_n": 0.0,
            "installed_driver_flange_required_to_bridge_seam": True,
            "physical_proof_and_creep_test_required": True,
        },
        "manufacturing_contract": {
            "strength_model_requires_solid_root_section": True,
            "required_local_infill_percent": 100,
            "solid_modifier_min_y_mm": FLOOR_Y_MM,
            "solid_modifier_max_y_mm": SOLID_MODIFIER_MAX_Y_MM,
            "minimum_wall_loops": 6,
            "maximum_layer_height_mm": 0.20,
            "split_bottom_build_axis": "original_minus_Y_after_Xminus90",
            "material_basis": (
                "weak/Z coupon data and project derating; no credit for "
                "sparse infill"),
        },
        "limitations": (
            "dry qualified print; at least six walls and 100% local solid "
            "modifier through the complete root/load stem",
            "ambient service <=35 C and no sunlight/radiator hot soak",
            "connector pull, acoustic resonance, impact and long-term creep "
            "are outside this closed-form screen",
            "PLA Lite result is provisional pending a product-specific TDS",
        ),
        "physical_gate": {
            "status": "PENDING",
            "proof": (
                "2x assembled service load for 24 h at 35 C; no cracks or "
                "whitening; residual set <=0.5 mm or <=10% of loaded "
                "deflection"),
            "creep": (
                "1.5x service load for at least 168 h at the worst credible "
                "room temperature"),
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_reports(
        json_path: Path, markdown_path: Path,
        production_steps: tuple[Path, ...] = ()) -> None:
    facts = integral_floor_strength_facts()
    if production_steps:
        records = []
        for production_step in production_steps:
            if not production_step.is_file():
                raise FileNotFoundError(production_step)
            records.append({
                "path": str(production_step),
                "bytes": production_step.stat().st_size,
                "sha256": _sha256(production_step),
            })
        facts["production_geometry"] = {
            "derivation": (
                "canonical and optional split are derived from the same "
                "final staged LM BREP"),
            "artifacts": records,
        }
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(facts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    lines = [
        "# V1LF integral floor-stand analytical screen",
        "",
        "This is a conservative closed-form net-section screen, not FEA or "
        "physical qualification. All reported stresses include the explicit "
        f"{ROOT_STRESS_CONCENTRATION_FACTOR:.2f} geometry/model factor.",
        "",
        "| Material | Vertical 1g/3g/5g SF | Anchored lateral 1g/3g/5g SF | "
        "1g deflection (mm) | Result |",
        "|---|---:|---:|---:|---|",
    ]
    for name, record in facts["materials"].items():
        result = "PASS (analytical)" if record["analytical_screen_pass"] else "FAIL"
        if record["provisional"]:
            result += "; provisional data"
        lines.append(
            f"| {name} | {record['sf_1g_sustained']:.2f} / "
            f"{record['sf_3g_transient']:.2f} / "
            f"{record['sf_5g_transient']:.2f} | "
            f"{record['anchored_lateral_sf_1g_sustained']:.2f} / "
            f"{record['anchored_lateral_sf_3g_transient']:.2f} / "
            f"{record['anchored_lateral_sf_5g_transient']:.2f} | "
            f"{record['diagnostic_tip_deflection_1g_mm']:.2f} | {result} |")
    stability = facts["stability"]
    junction = facts["shoulder_ring_junction"]
    if production_steps:
        lines.extend((
            "",
            "## Bound production geometry",
            "",
        ))
        for record in facts["production_geometry"]["artifacts"]:
            lines.append(
                f"- `{record['path']}` — SHA-256 `{record['sha256']}`")
    lines.extend((
        "",
        "## Shoulder-to-LM-ring diagnostic",
        "",
        "This deliberately conservative lower bound credits only the two "
        "uninterrupted printed outer-lip ligaments at the lower D190 "
        "tangent. It gives no credit to the seat membrane, integrated "
        "shoulder below the tangent, route covers, insert bosses, magnets "
        "or installed metal LM flange. It therefore does not redefine the "
        "root analytical result and is not a complete-assembly failure "
        "prediction.",
        "",
        "| Material | 1g sustained SF | 3g transient SF | "
        "5g transient SF | Lower-bound threshold |",
        "|---|---:|---:|---:|---|",
    ))
    for name, record in junction["materials"].items():
        result = (
            "MEETS" if record["lower_bound_meets_root_thresholds"]
            else "BELOW")
        if record["provisional"]:
            result += "; provisional data"
        lines.append(
            f"| {name} | {record['sf_1g_sustained']:.2f} | "
            f"{record['sf_3g_transient']:.2f} | "
            f"{record['sf_5g_transient']:.2f} | {result} |")
    lines.extend((
        "",
        "The lip-only lower bound is below the project thresholds. The "
        "installed LM flange and fasteners are therefore required parallel "
        "load paths, and the documented assembled proof/creep gate remains "
        "mandatory.",
        "",
        "## Governing limitations",
        "",
        f"- Exact nominal root section: "
        f"{facts['net_root_section']['net_area_mm2']:.1f} mm²; "
        f"governing section modulus "
        f"{facts['net_root_section']['governing_section_modulus_mm3']:.1f} "
        "mm³ after subtracting D9, D8.2 and D6 lumens.",
        "- The section result is valid only with the required 100% local "
        "solid modifier through the complete stem/root; sparse infill gets "
        "no structural credit.",
        f"- Free-standing lateral tip threshold: "
        f"{stability['lateral_tip_acceleration_g']:.3f} g. This is a "
        "stability limit, not a PLA strength limit.",
        "- The optional hidden split key receives 0 N structural credit; "
        "the installed LM driver flange must bridge the seam.",
        "- Every material/process remains **PENDING** until the documented "
        "proof and creep tests pass.",
        "",
    ))
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    parser.add_argument(
        "--step", type=Path, action="append", default=[],
        help="production STEP to hash-bind; repeat for canonical and split")
    args = parser.parse_args()
    write_reports(args.json, args.markdown, tuple(args.step))


if __name__ == "__main__":
    main()
