"""Optional R6F add-ons for the extreme two-carrier Obi-Wan core.

The floor-state stem, foot and connector are integral LM geometry.  This
module therefore exports only the optional tweeter crescent.  Six fully
buried captive side-magnet interfaces remain alignment/anti-rattle features
with zero load credit.
"""

from __future__ import annotations

from build123d import Box, Compound, Cylinder, Part, Pos

from top_baffle_nd25fw4 import UM_CUTOUT
from top_baffle_nd25fw4_cables import ROUTING_PROFILE
from top_baffle_nd25fw4_obiwan import (
    TWEETER_ADDON_JOINT_Z,
    TWEETER_CORE_BORE_TOP_Z,
    TWEETER_CORE_JOINT_Z,
    TWEETER_JOINT_CLEAR,
    TWEETER_JOINT_HOLE_D,
    TWEETER_JOINT_INSERT_BORE_D,
    TWEETER_JOINT_X,
    TWEETER_JOINT_Y,
    _junction_closure_web,
    _owned_tweeter_joint_plan,
    _plan_prism,
    _require_guarded_build,
    _subtract_plan_prisms,
    tweeter_joint_polygon,
)


if ROUTING_PROFILE != "obiwan":
    raise RuntimeError(
        "Obi-Wan add-ons require LX_ROUTING_PROFILE=obiwan (R6F)")

PRINT_ORIENTATION = "front-face-down"


def _cylinder_at(x, y, radius, z0, z1):
    return Pos(x, y, (z0 + z1) / 2.0) * Cylinder(radius, z1 - z0)


def _fuse_required(part, addition, label):
    """Positive-growth one-solid fusion; never discard detached pieces."""
    before = part.volume
    added = addition.volume
    volume_tol = max(0.05, (before + added) * 1e-6)
    combined = part.fuse(addition).clean()
    solids = list(combined.solids())
    if (combined.is_valid and len(solids) == 1
            and solids[0].volume > 0.01
            and combined.volume > before + min(0.05, added * 1e-4)
            and combined.volume <= before + added + volume_tol):
        return Part([solids[0]])
    raise RuntimeError(
        f"{label}: required fusion failed; valid={combined.is_valid} "
        f"volumes={[solid.volume for solid in combined.solids()]}")


def tweeter_crescent():
    """Rear-tapered acoustic crescent on two compact direct half-laps."""
    _require_guarded_build()
    from top_baffle_nd25fw4_v1 import v1_magnet_free_solid
    from top_baffle_nd25fw4_obiwan import UM_CORE_R

    # Obi-Wan uses M3 half-laps here, not V1's captive base magnets.  Crop from
    # the pre-cavity V1 authority so no sealed obsolete magnet void survives
    # inside the independently printed crescent.
    raw = v1_magnet_free_solid()
    crop = Pos(0.0, 434.75, 10.0) * Box(150.0, 37.5, 25.0)
    cropped = (raw & crop).clean()
    cropped_solids = list(cropped.solids())
    if (not cropped.is_valid or len(cropped_solids) != 1
            or cropped_solids[0].volume <= 0.01):
        raise RuntimeError(
            "tweeter crop must produce exactly one crescent; "
            f"valid={cropped.is_valid} volumes="
            f"{[solid.volume for solid in cropped.solids()]}")
    part = Part([cropped_solids[0]])

    # Establish the released R51.9 crescent clearance before adding the
    # complementary full-depth junction web.  Repeating this subtraction
    # afterwards would hollow the web and recreate the resonance-prone red
    # cusp islands that the web is intended to close.
    part -= _cylinder_at(
        UM_CUTOUT[0], UM_CUTOUT[1], UM_CORE_R + 0.20, 6.7, 20.0)
    # Assign any legacy B2 material inside the new cusp envelope before the
    # front half-lap ears are created, otherwise the mask can detach their
    # outboard tips.
    from top_baffle_nd25fw4_obiwan import _enforce_junction_plan_ownership
    part = _enforce_junction_plan_ownership(part, "t_um", "tweeter")
    part = _fuse_required(
        part, _junction_closure_web("t_um", "tweeter"),
        "tweeter-owned full-depth T--UM closure web")

    for x in TWEETER_JOINT_X:
        part = _subtract_plan_prisms(
            part,
            _owned_tweeter_joint_plan(
                "um", x, TWEETER_JOINT_CLEAR),
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_JOINT_Z[1] + TWEETER_JOINT_CLEAR)
        part = _fuse_required(
            part,
            _plan_prism(
                _owned_tweeter_joint_plan("tweeter", x),
                *TWEETER_ADDON_JOINT_Z),
            f"tweeter rounded ear {x:+.1f}")
        # The full-depth closure web also crosses the rear-driven M3 axis.
        # Recut the same core through-bore on this complementary owner before
        # opening the larger blind insert receiver.  Their 0.35-mm axial
        # overlap keeps one continuous service path without hollowing any
        # unrelated part of the solid web.
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_HOLE_D / 2.0,
            TWEETER_CORE_JOINT_Z[0] - 0.2,
            TWEETER_CORE_BORE_TOP_Z)
        part -= _cylinder_at(
            x, TWEETER_JOINT_Y, TWEETER_JOINT_INSERT_BORE_D / 2.0,
            TWEETER_ADDON_JOINT_Z[0] - 0.2,
            TWEETER_ADDON_JOINT_Z[0] + 4.0)

    # The cable leaves through the intentional central plan mouth and floats
    # behind this add-on.  There is no horn, trench or hidden suffix; only the
    # two solid, tangent-blended side webs bridge to the UM carrier.
    part = part.clean()
    solids = list(part.solids())
    if (not part.is_valid or len(solids) != 1
            or solids[0].volume <= 0.01):
        raise RuntimeError(
            "tweeter finalization must retain every required feature; "
            f"valid={part.is_valid} volumes="
            f"{[solid.volume for solid in part.solids()]}")
    return Part([solids[0]])


def obiwan_attachments():
    _require_guarded_build()
    return {"addon_tweeter_crescent": tweeter_crescent()}


def gen_step():
    _require_guarded_build()
    children = []
    for label, solid in obiwan_attachments().items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_obiwan_r6f_optional_addons"
    return assembly
