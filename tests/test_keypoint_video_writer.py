"""Tests for the extracted keypoint-overlay renderer (pure, no torch)."""

from __future__ import annotations

import numpy as np

from src.database.keypoint_video_writer import draw_keypoints_frame


def _frame(h=480, w=640):
    return np.full((h, w, 3), 50, dtype=np.uint8)


# The info panel occupies the top-left (rows 0..98, cols 0..340) and the legend
# the bottom strip, so test pixels are chosen clear of both overlays.
def test_returns_new_array_input_not_mutated():
    frame = _frame()
    original = frame.copy()
    kps = np.array([[400.0, 300.0], [500.0, 200.0]])
    out = draw_keypoints_frame(frame, kps, None, 0, 10)
    assert out is not frame
    assert np.array_equal(frame, original), "input frame must not be mutated"
    assert out.shape == frame.shape


def test_keypoints_are_drawn():
    frame = _frame()
    kps = np.array([[400.0, 300.0]])
    out = draw_keypoints_frame(frame, kps, None, 0, 10)
    # A green (0,255,0) dot must appear at the keypoint location.
    assert tuple(int(c) for c in out[300, 400]) == (0, 255, 0)


def test_static_mask_overlay_changes_dynamic_zone():
    frame = _frame()
    kps = np.zeros((0, 2))
    mask = np.ones((480, 640), dtype=np.uint8)
    mask[300:, :] = 0  # dynamic zone (mask == 0) in the bottom half
    out = draw_keypoints_frame(frame, kps, mask, 0, 10)
    # Tinted dynamic pixel, clear of the info panel and legend.
    assert not np.array_equal(out[350, 500], frame[350, 500])


def test_deterministic():
    frame = _frame()
    kps = np.array([[400.0, 300.0], [500.0, 200.0]])
    a = draw_keypoints_frame(frame, kps, None, 3, 10)
    b = draw_keypoints_frame(frame, kps, None, 3, 10)
    assert np.array_equal(a, b)
