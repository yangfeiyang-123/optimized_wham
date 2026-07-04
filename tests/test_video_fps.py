"""Unit tests for fps detection / resolution (lib/video_fps.py)."""
from __future__ import annotations

import numpy as np
import pytest

from lib.video_fps import _parse_rate, fps_from_record, resolve_fps


def test_parse_rate_handles_fractions_and_junk():
    assert _parse_rate("30/1") == 30.0
    assert _parse_rate("60/1") == 60.0
    assert abs(_parse_rate("30000/1001") - 29.97) < 0.01
    assert _parse_rate("25") == 25.0
    assert _parse_rate("0/0") is None
    assert _parse_rate("N/A") is None
    assert _parse_rate("") is None


def test_fps_from_record_reads_known_keys():
    assert fps_from_record({"mocap_framerate": np.asarray(30.0)}) == 30.0
    assert fps_from_record({"mocap_frame_rate": np.asarray([60.0])}) == 60.0
    assert fps_from_record({"fps": 25}) == 25.0
    assert fps_from_record({"frame_rate": 50.0}) == 50.0
    assert fps_from_record({}) is None
    assert fps_from_record({"fps": 0}) is None  # non-positive ignored


def test_resolve_fps_priority_explicit_over_record():
    # explicit override wins even if the record disagrees
    assert resolve_fps(45.0, record={"fps": 30}) == 45.0


def test_resolve_fps_uses_record_when_no_explicit():
    assert resolve_fps(None, record={"mocap_framerate": np.asarray(30.0)}) == 30.0


def test_resolve_fps_raises_when_nothing_available():
    with pytest.raises(ValueError):
        resolve_fps(None, record={}, video_path=None)
