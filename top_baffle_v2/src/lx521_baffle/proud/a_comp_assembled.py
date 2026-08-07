"""Variant A-comp as actually built: the four B2 print pieces plus the
four shoulder attachments, shown in assembled position."""

from __future__ import annotations

from ..assembly import ordered_labeled_compound
from .attachments import attachments
from .b2_split import pieces


def gen_step():
    parts = dict(pieces())
    parts.update({k: v for k, v in attachments().items() if "a_shoulder" in k})
    return ordered_labeled_compound(
        parts, label="lx521_4_top_baffle_a_comp_assembled")
