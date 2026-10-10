"""Keypoint budget for query features (strongest-first slice).

FeatureExtractor returns query keypoints ordered by detector score, strongest
first (``FeatureExtractor._by_score``), so the first ``limit`` entries are the
set ALIKED itself keeps with ``max_keypoints = limit`` (its threshold mode
keeps the top ``n_limit`` by score). LightGlue and the MNN probe cost grow with
the keypoint count; a slice lets one extraction serve two budgets.
"""

from __future__ import annotations

# Per-keypoint arrays in a feature dict; everything else (image_size, ...) is
# per-image and must not be sliced.
PER_KEYPOINT_KEYS = ("keypoints", "descriptors", "coords_2d", "scores", "keypoint_scores")


def top_keypoints(features: dict | None, limit: int) -> dict | None:
    """The first ``limit`` keypoints of a score-ordered feature dict.

    ``limit <= 0`` or a set already within the limit returns ``features``
    itself; otherwise a shallow copy with the per-keypoint arrays sliced (the
    input, which may sit in a per-call cache, is not modified).
    """
    if not features or limit <= 0:
        return features
    keypoints = features.get("keypoints")
    if keypoints is None:
        return features
    n = len(keypoints)
    if n <= limit:
        return features
    out = dict(features)
    for key in PER_KEYPOINT_KEYS:
        value = out.get(key)
        if value is not None and len(value) == n:
            out[key] = value[:limit]
    return out
