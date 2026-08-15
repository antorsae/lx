"""Explicit opposed Obi-Wan TEBM35C10-4 BMR-slim candidate.

The opposed arrangement remains owned by ``bmr_crescent_opposed``.  This thin
module selects the shared BMR-slim land and gives that geometry a distinct
candidate identity without copying its pod, skirt, route, or magnet logic.
"""

from __future__ import annotations

from ..assembly import ordered_labeled_compound
from ..tebm35c10_4_land import BMR_SLIM_LAND
from . import bmr_crescent_opposed as _family


PART_NAME = "obiwan_bmr_slim_crescent_opposed_TEBM35C10-4"
RELEASE_VARIANT = "Obiwan-TEBM35C10-4-BMR-crescent-opposed-bmr-slim"
VARIANT = "opposed-bmr-slim"
LAND_TOPOLOGY = BMR_SLIM_LAND
PRINT_ORIENTATION = _family.PRINT_ORIENTATION
RELEASE_AUTHORIZED = False
PHYSICAL_MEASURE_REQUIRED = True
MAGNET_COUNT = _family.MAGNET_COUNT
ATTACHMENT_NAME = "addon_bmr_slim_crescent_opposed"


def build_model() -> _family.BmrCrescentOpposedModel:
    """Build the shared opposed pod with the BMR-slim lands."""
    model = _family.build_model(LAND_TOPOLOGY)
    model.solid.label = PART_NAME
    return model


def design_facts(magnet_tools: tuple = ()) -> dict[str, object]:
    """Return opposed facts with this explicit candidate identity."""
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
        label="lx521_obiwan_r6f_bmr_slim_crescent_opposed_candidate",
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
