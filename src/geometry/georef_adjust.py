"""Re-fit an applied local-map georeference after the operator moves one frame.

A georeferenced local map is defined by its control points (GCPs: a point in
local X/Y plus its true lat/lon). Dragging frame ``F`` on the map to a new
place is the same statement as "the centre of frame F is at this lat/lon", so
it becomes one more control point and the whole frame graph is solved again
from the stored local solution. Nothing is edited in place: the same list of
control points always reproduces the same map.

Pure numpy, no Qt / h5py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.geometry.georef_fit import SIMILARITY, GeorefFit, fit_georef
from src.geometry.georef_fit import apply as apply_matrix
from src.geometry.georef_graph import (
    DEFAULT_STIFFNESS,
    GRAPH,
    MIN_GRAPH_POINTS,
    GraphGeorefFit,
    LocalGraph,
    fit_graph_georef,
)

MANUAL_WEIGHT = 10.0
# Control points closer than this (fraction of a frame footprint) to the moved
# frame's centre describe the same map point and are replaced by the move.
_SAME_POINT = 1e-3


@dataclass
class NodeMove:
    frame_id: int
    gcps: list[dict]  # control points after the move, with fresh residual_m
    fit: GeorefFit | GraphGeorefFit
    ground_k: float  # projected metres -> ground metres
    replaced: list[str]  # names of the control points the move superseded
    residual_m: float  # moved frame: distance from where it was dropped
    others_max_m: float  # worst residual among the other control points
    centers: np.ndarray  # (N, 2) new frame centres, projected metres (graph order)


def node_index(graph: LocalGraph, frame_id: int) -> int:
    hit = np.flatnonzero(np.asarray(graph.frame_ids) == int(frame_id))
    if len(hit) == 0:
        raise ValueError(f"frame {frame_id} is not part of the local map")
    return int(hit[0])


def with_moved_node(
    gcps: list[dict],
    graph: LocalGraph,
    frame_id: int,
    lat: float,
    lon: float,
    weight: float = MANUAL_WEIGHT,
) -> tuple[list[dict], list[str]]:
    """Control points with frame ``frame_id`` pinned to (lat, lon).

    Any existing point sitting on that frame's centre (an automatic point of
    the same frame, or an earlier move of it) is replaced, not contradicted.
    Returns the new list and the names of the replaced points.
    """
    if not (np.isfinite(lat) and np.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("invalid latitude/longitude")
    x, y = (float(v) for v in graph.centers[node_index(graph, frame_id)])
    tol = _SAME_POINT * graph.footprint
    kept, replaced = [], []
    for g in gcps:
        same = int(g.get("frame_id", -1)) == int(frame_id) and bool(g.get("manual"))
        if same or np.hypot(float(g["x"]) - x, float(g["y"]) - y) <= tol:
            replaced.append(str(g.get("name", "")))
        else:
            kept.append(dict(g))
    kept.append(
        {
            "name": f"Кадр {int(frame_id)} (вручну)",
            "x": x,
            "y": y,
            "lat": float(lat),
            "lon": float(lon),
            "frame_id": int(frame_id),
            "manual": True,
            "weight": float(weight),
        }
    )
    return kept, replaced


def frame_centers(fit, graph: LocalGraph) -> np.ndarray:
    """Frame centres (graph order) in projected metres under ``fit``."""
    if getattr(fit, "kind", None) == GRAPH:
        m = fit.frame_matrices
        return np.einsum("nij,nj->ni", m[:, :, :2], graph.centers) + m[:, :, 2]
    return apply_matrix(fit.matrix, graph.centers)


def refit(gcps: list[dict], graph: LocalGraph, converter, rotation_sign: int, stiffness=None):
    """Fit the control points again. Returns ``(fit, ground_k)`` and writes
    ``residual_m`` into every record.

    3+ points use the non-rigid graph model, so a move bends the map between
    the neighbouring control points instead of being averaged away by a single
    global similarity. The stiffness never exceeds the default: a stiffer field
    could only honour a hand-placed frame by dragging the other points off.
    """
    if len(gcps) < 2:
        raise ValueError("need at least 2 control points")
    src = np.array([[float(g["x"]), float(g["y"])] for g in gcps])
    dst = np.array([converter.gps_to_metric(float(g["lat"]), float(g["lon"])) for g in gcps])
    k = float(converter.ground_scale_factor(float(np.mean([g["lat"] for g in gcps]))))
    if len(gcps) >= MIN_GRAPH_POINTS:
        lam = DEFAULT_STIFFNESS if not stiffness else min(float(stiffness), DEFAULT_STIFFNESS)
        fit = fit_graph_georef(
            graph,
            src,
            dst,
            rotation_sign=rotation_sign,
            stiffness=lam,
            weights=[float(g.get("weight", 1.0)) for g in gcps],
        )
    else:
        fit = fit_georef(src, dst, kind=SIMILARITY, rotation_sign=rotation_sign)
    for g, r in zip(gcps, fit.residuals, strict=True):
        g["residual_m"] = float(r * k)
    return fit, k


def move_node(
    gcps: list[dict],
    graph: LocalGraph,
    frame_id: int,
    lat: float,
    lon: float,
    converter,
    rotation_sign: int,
    stiffness=None,
    weight: float = MANUAL_WEIGHT,
) -> NodeMove:
    """Pin frame ``frame_id`` to (lat, lon) and solve the whole graph again."""
    new_gcps, replaced = with_moved_node(gcps, graph, frame_id, lat, lon, weight)
    fit, k = refit(new_gcps, graph, converter, rotation_sign, stiffness)
    others = [g["residual_m"] for g in new_gcps[:-1]]
    return NodeMove(
        frame_id=int(frame_id),
        gcps=new_gcps,
        fit=fit,
        ground_k=k,
        replaced=replaced,
        residual_m=float(new_gcps[-1]["residual_m"]),
        others_max_m=float(max(others)) if others else 0.0,
        centers=frame_centers(fit, graph),
    )
