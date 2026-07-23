"""Exploded-in-place R6F review assembly: core, selected add-ons and fit proxy."""

from __future__ import annotations

from build123d import Compound

from ..base import STAND_FOOT

from ..cables import ROUTING_PROFILE
from ..um_fit import (
    faston_boot_proxy_parts,
    faston_pull_sweep_parts,
    faston_proxy_parts,
    mu10_body_keepout,
    rear_cable_envelope,
    removal_envelope,
    terminal_carrier_proxy,
    obiwan_lm_cable_envelope,
    obiwan_y_breakout_boot_parts,
    obiwan_ts_cable_envelope,
)
from .carriers import core_parts
from .attachments import obiwan_attachments
from .bridge import bridge_fastener_head_envelopes

if ROUTING_PROFILE != "obiwan":
    raise RuntimeError("assembled Obi-Wan review requires LX_ROUTING_PROFILE=obiwan")


def gen_step():
    children = []
    for label, solid in core_parts().items():
        solid.label = label
        children.append(solid)
    for label, solid in obiwan_attachments().items():
        solid.label = label
        children.append(solid)
    body = mu10_body_keepout(include_flange=True)
    body.label = (
        "REFERENCE_MU10_D98_D80_D60_BODY_TERMINALS_OMITTED_"
        "PHYSICAL_CHECK_REQUIRED")
    children.append(body)
    carrier = terminal_carrier_proxy()
    carrier.label = "REFERENCE_terminal_carrier_proxy_clock_283deg"
    children.append(carrier)
    env = removal_envelope()
    env.label = "KEEP_CLEAR_Faston_outboard_removal_envelope"
    children.append(env)
    for label, solid in faston_proxy_parts().items():
        solid.label = "REFERENCE_" + label
        children.append(solid)
    for label, solid in faston_boot_proxy_parts().items():
        solid.label = "KEEP_CLEAR_" + label
        children.append(solid)
    for label, solid in faston_pull_sweep_parts().items():
        solid.label = "KEEP_CLEAR_" + label + "_12mm"
        children.append(solid)
    um_cable = rear_cable_envelope("obiwan")
    um_cable.label = (
        "REFERENCE_UM_D7_LM_printed_cover_then_free_behind_UM_"
        "R15_R20_Faston_handoff")
    children.append(um_cable)
    for label, solid in obiwan_y_breakout_boot_parts().items():
        solid.label = "REFERENCE_" + label
        children.append(solid)
    lm_cable = obiwan_lm_cable_envelope()
    lm_cable.label = "REFERENCE_LM_D7p8_short_free_span_no_micro_duct"
    children.append(lm_cable)
    ts_cable = obiwan_ts_cable_envelope()
    ts_cable.label = (
        "REFERENCE_TS_D5p2_LM_UM_printed_then_free_behind_tweeter")
    children.append(ts_cable)
    if not STAND_FOOT:
        hardware = bridge_fastener_head_envelopes()
        hardware.label = "KEEP_CLEAR_four_stock_bridge_M5_heads"
        children.append(hardware)
    assembly = Compound(children=children)
    state = "floor" if STAND_FOOT else "no_floor_fused_solid_web"
    assembly.label = f"lx521_obiwan_r6f_assembled_{state}"
    return assembly
