"""Variant B1 as actually built: the four B2 print pieces plus the two
wing attachments, shown in assembled position."""

from __future__ import annotations

from ..assembly import ordered_labeled_compound
from .top_baffle_nd25fw4_attachments import attachments
from .top_baffle_nd25fw4_b2_split import pieces


def gen_step():
    parts = dict(pieces())
    parts.update({k: v for k, v in attachments().items() if "b1_wing" in k})
    return ordered_labeled_compound(
        parts, label="lx521_4_top_baffle_nd25fw4_b1_assembled")
