"""Per-slot calibration origin; availability is distinct from measured support."""

from enum import IntEnum
from typing import Iterable

import numpy as np


class CalibrationOrigin(IntEnum):
    UNKNOWN = 0
    ANCHOR = 1
    OPTIMIZED = 2
    INTERPOLATED = 3
    EXTRAPOLATED = 4


class GeoreferenceStatus(IntEnum):
    """Whether a slot may be used to emit a trusted geographic fix."""

    UNKNOWN = 0
    SUPPORTED = 1
    PROVISIONAL = 2
    INVALID = 3


def anchored_graph_support(
    node_ids: Iterable[int], edges: Iterable[object], anchor_ids: Iterable[int]
) -> tuple[set[int], np.ndarray, dict[int, list[int]]]:
    """Return nodes supported by an anchor and deterministic component metadata."""
    nodes = sorted({int(node_id) for node_id in node_ids})
    if not nodes:
        return set(), np.empty(0, dtype=np.int32), {}
    node_set = set(nodes)
    adjacency = {node: set() for node in nodes}
    for edge in edges:
        a = int(edge.from_id)
        b = int(edge.to_id)
        weight = float(getattr(edge, "weight", 1.0))
        if a in node_set and b in node_set and np.isfinite(weight) and weight > 0:
            adjacency[a].add(b)
            adjacency[b].add(a)

    component_ids = np.full(max(nodes) + 1, -1, dtype=np.int32)
    components: dict[int, list[int]] = {}
    unseen = set(nodes)
    component = 0
    while unseen:
        seed = min(unseen)
        stack = [seed]
        members = []
        unseen.remove(seed)
        while stack:
            current = stack.pop()
            members.append(current)
            for neighbor in sorted(adjacency[current], reverse=True):
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        for node in members:
            component_ids[node] = component
        components[component] = sorted(members)
        component += 1

    anchors = {int(anchor) for anchor in anchor_ids}
    component_anchors = {
        cid: sorted(anchors.intersection(members)) for cid, members in components.items()
    }
    supported = {
        node
        for cid, members in components.items()
        if component_anchors[cid]
        for node in members
    }
    return supported, component_ids, component_anchors


def classify_calibration(valid, optimized, anchor_ids, invalid_ids=(), supported=None):
    """Return origin, support distance, and independent georeference status.

    Unknown legacy provenance is not reconstructed from frame_valid alone.
    This function requires the pre-interpolation support mask.
    """
    valid = np.asarray(valid, dtype=bool)
    optimized = np.asarray(optimized, dtype=bool)
    if valid.shape != optimized.shape:
        raise ValueError("Calibration masks must have the same shape")
    supported_mask = optimized if supported is None else np.asarray(supported, dtype=bool)
    if valid.shape != supported_mask.shape:
        raise ValueError("Calibration support mask must match availability")
    origins = np.zeros(len(valid), dtype=np.uint8)
    distances = np.full(len(valid), -1, dtype=np.int32)
    statuses = np.full(len(valid), GeoreferenceStatus.UNKNOWN, dtype=np.uint8)
    ids = np.flatnonzero(optimized & valid)
    if len(ids):
        slots = np.arange(len(valid))
        left = ids[np.clip(np.searchsorted(ids, slots, side="right") - 1, 0, len(ids) - 1)]
        right = ids[np.clip(np.searchsorted(ids, slots), 0, len(ids) - 1)]
        distances[valid] = np.minimum(abs(slots - left), abs(slots - right))[valid]
        origins[valid] = CalibrationOrigin.EXTRAPOLATED
        interior = valid & (slots >= ids[0]) & (slots <= ids[-1])
        origins[interior] = CalibrationOrigin.INTERPOLATED
        origins[ids] = CalibrationOrigin.OPTIMIZED
        statuses[valid] = GeoreferenceStatus.PROVISIONAL
        statuses[supported_mask & valid] = GeoreferenceStatus.SUPPORTED
        for fid in anchor_ids:
            if 0 <= fid < len(valid) and optimized[fid] and valid[fid]:
                origins[fid] = CalibrationOrigin.ANCHOR
                if supported_mask[fid]:
                    statuses[fid] = GeoreferenceStatus.SUPPORTED
    for fid in invalid_ids:
        if 0 <= fid < len(valid):
            statuses[fid] = GeoreferenceStatus.INVALID
    return origins, distances, statuses
