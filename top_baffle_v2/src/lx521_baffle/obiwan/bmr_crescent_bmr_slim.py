"""Explicit coaxial Obi-Wan TEBM35C10-4 BMR-slim candidate.

The coaxial arrangement remains owned by ``bmr_crescent``.  This thin module
selects the shared BMR-slim land and supplies a distinct candidate artifact
identity without forking any pod, skirt, cable, driver, or magnet geometry.
"""

from __future__ import annotations

from ..assembly import ordered_labeled_compound
from ..tebm35c10_4_land import BMR_SLIM_LAND
from . import bmr_crescent as _family


PART_NAME = "obiwan_bmr_slim_crescent_TEBM35C10-4"
RELEASE_VARIANT = "Obiwan-TEBM35C10-4-BMR-crescent-bmr-slim"
VARIANT = "coaxial-bmr-slim"
LAND_TOPOLOGY = BMR_SLIM_LAND
PRINT_ORIENTATION = _family.PRINT_ORIENTATION
RELEASE_AUTHORIZED = False
PHYSICAL_MEASURE_REQUIRED = True
MAGNET_COUNT = _family.MAGNET_COUNT
ATTACHMENT_NAME = "addon_bmr_slim_crescent"


def build_model() -> _family.BmrCrescentModel:
    """Build the shared coaxial pod with the BMR-slim land."""
    model = _family.build_model(LAND_TOPOLOGY)
    model.solid.label = PART_NAME
    return model


def design_facts(magnet_tools: tuple = ()) -> dict[str, object]:
    """Return coaxial facts with this explicit candidate identity."""
    facts = _family.design_facts(magnet_tools, LAND_TOPOLOGY)
    facts.update({
        "part": PART_NAME,
        "release_variant": RELEASE_VARIANT,
        "variant": VARIANT,
    })
    return facts


def gen_step():
    """Return a labelled one-occurrence STEP-ready candidate compound."""
    model = build_model()
    return ordered_labeled_compound(
        {ATTACHMENT_NAME: model.solid},
        label="lx521_obiwan_r6f_bmr_slim_crescent_candidate",
    )


__all__ = [
    "ATTACHMENT_NAME",
    "LAND_TOPOLOGY",
    "MAGNET_COUNT",
    "PART_NAME",
    "PHYSICAL_MEASURE_REQUIRED",
    "PRINT_ORIENTATION",
    "RELEASE_AUTHORIZED",
    "RELEASE_VARIANT",
    "VARIANT",
    "build_model",
    "design_facts",
    "gen_step",
]
