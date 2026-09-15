"""Reject separated footprints and loss of contact with the previous layer."""
import sys
from pathlib import Path
import pytest
from shapely.geometry import LineString, box
from shapely.ops import unary_union

sys.path[:0] = [str(Path(__file__).resolve().parents[1]/'scripts'),
                str(Path(__file__).resolve().parents[1]/'src')]
from captive_wall_audit import retaining_continuity


def test_connected_wall_with_previous_substrate():
    ring = box(0, 0, 10, .6)
    report = retaining_continuity(ring, LineString([(0, 0), (10, 0)]), box(0, 0, 10, .5))
    assert report['connected_components'] == 1
    assert report['interlayer_overlap_fraction'] == pytest.approx(5/6)


def test_break_in_retaining_wall_is_rejected():
    ring = unary_union([box(0, 0, 4, .6), box(6, 0, 10, .6)])
    with pytest.raises(AssertionError, match='disconnected retaining wall'):
        retaining_continuity(ring, LineString([(0, 0), (10, 0)]), box(0, 0, 10, 1))


def test_intact_but_unsupported_wall_is_rejected():
    ring = box(0, 0, 10, .6)
    with pytest.raises(AssertionError, match='insufficient retaining interlayer overlap'):
        retaining_continuity(ring, LineString([(0, 0), (10, 0)]), box(0, -.5, 10, .1))
