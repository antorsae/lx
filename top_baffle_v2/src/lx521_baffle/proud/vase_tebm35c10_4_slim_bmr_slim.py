"""Explicit Slim-envelope TEBM35C10-4 BMR-slim vase candidate.

This module binds the existing Slim product envelope to the independent
``bmr-slim`` land topology.  It owns no CAD geometry of its own.
"""

from __future__ import annotations

from ..tebm35c10_4_land import BMR_SLIM_LAND
from . import vase_tebm35c10_4 as _family


PART_NAME = _family.PART_NAME
RELEASE_VARIANT = "Slim-TEBM35C10-4-BMR-slim"
VARIANT = "slim-bmr-slim"
PROFILE = _family.SLIM_PROFILE
LAND_TOPOLOGY = BMR_SLIM_LAND
PRINT_ORIENTATION = _family.PRINT_ORIENTATION
RELEASE_AUTHORIZED = False
PHYSICAL_MEASURE_REQUIRED = True
MAGNET_COUNT = _family.T_MAGNET_TOTAL


def build_model() -> _family.VaseTEBMModel:
    """Build the shared BMR-slim topology on the Slim vase envelope."""
    model = _family.build_model(PROFILE, LAND_TOPOLOGY)
    model.solid.label = PART_NAME
    return model


def design_facts() -> dict[str, object]:
    """Return family facts with this explicit artifact identity."""
    facts = _family.design_facts(PROFILE, LAND_TOPOLOGY)
    facts.update({
        "part": PART_NAME,
        "release_variant": RELEASE_VARIANT,
        "variant": VARIANT,
    })
    return facts


def gen_step():
    """Return the identity-labelled STEP-ready solid."""
    return build_model().solid


__all__ = [
    "LAND_TOPOLOGY",
    "MAGNET_COUNT",
    "PART_NAME",
    "PHYSICAL_MEASURE_REQUIRED",
    "PRINT_ORIENTATION",
    "PROFILE",
    "RELEASE_AUTHORIZED",
    "RELEASE_VARIANT",
    "VARIANT",
    "build_model",
    "design_facts",
    "gen_step",
]
