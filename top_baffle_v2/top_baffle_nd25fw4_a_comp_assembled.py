"""Variant A-comp as actually built: the four B2 print pieces plus the
four shoulder attachments, shown in assembled position."""

from __future__ import annotations

from build123d import Compound

from top_baffle_nd25fw4_attachments import attachments
from top_baffle_nd25fw4_b2_split import pieces


def gen_step():
    children = []
    parts = dict(pieces())
    parts.update({k: v for k, v in attachments().items() if "a_shoulder" in k})
    for label, solid in parts.items():
        solid.label = label
        children.append(solid)
    assembly = Compound(children=children)
    assembly.label = "lx521_4_top_baffle_nd25fw4_a_comp_assembled"
    return assembly
