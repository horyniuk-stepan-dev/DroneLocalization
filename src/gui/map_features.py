"""User-drawn map features (points, lines, areas) and their measurements.

Pure Python (no Qt) so it can be unit-tested headless. The map page is only a
view: it reports created/edited/deleted features, this store keeps them and
computes the authoritative measurements.

Coordinates use the same transport convention as ``MapBridge``: each vertex is
``[lat, lon]`` in GEOGRAPHIC mode and ``[y, x]`` (arbitrary units) in LOCAL
mode. GEOGRAPHIC measurements are geodesic on WGS84 (``pyproj.Geod``); LOCAL
measurements are plain Euclidean and labelled as units, never metres.
"""

from __future__ import annotations

import itertools
import json
import math
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pyproj import Geod

GEOGRAPHIC = "GEOGRAPHIC"
LOCAL = "LOCAL"
FEATURE_TYPES = ("point", "line", "polygon", "rectangle", "circle")
_GEOD = Geod(ellps="WGS84")


@dataclass
class Measurement:
    length: float | None = None  # line length or polygon/circle perimeter
    area: float | None = None
    mode: str = GEOGRAPHIC

    def label(self) -> str:
        parts = []
        if self.area is not None:
            parts.append(format_area(self.area, self.mode))
        if self.length is not None:
            prefix = "P " if self.area is not None else ""
            parts.append(prefix + format_length(self.length, self.mode))
        return " · ".join(parts)


@dataclass
class MapFeature:
    type: str
    coords: list[list[float]]
    mode: str = GEOGRAPHIC
    name: str = ""
    color: str = "#ffb000"
    radius: float | None = None  # circle only: metres (GEOGRAPHIC) or units (LOCAL)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self):
        if self.type not in FEATURE_TYPES:
            raise ValueError(f"Unknown feature type: {self.type!r}")
        self.mode = LOCAL if self.mode == LOCAL else GEOGRAPHIC
        self.coords = [[float(a), float(b)] for a, b in self.coords]
        if not all(math.isfinite(v) for p in self.coords for v in p):
            raise ValueError("Non-finite coordinate")
        need = {"point": 1, "circle": 1, "line": 2, "polygon": 3, "rectangle": 2}[self.type]
        if len(self.coords) < need:
            raise ValueError(f"{self.type} needs at least {need} vertices")
        if self.type == "circle":
            if self.radius is None or not math.isfinite(self.radius) or self.radius <= 0:
                raise ValueError("circle needs a positive radius")
            self.radius = float(self.radius)

    def ring(self) -> list[list[float]]:
        """Closed-area vertices (rectangle expanded from its two corners)."""
        if self.type == "rectangle":
            lats = [p[0] for p in self.coords]
            lons = [p[1] for p in self.coords]
            s, n, w, e = min(lats), max(lats), min(lons), max(lons)
            return [[s, w], [n, w], [n, e], [s, e]]
        return self.coords

    def to_dict(self) -> dict[str, Any]:
        d = {
            "id": self.id,
            "type": self.type,
            "coords": self.coords,
            "mode": self.mode,
            "name": self.name,
            "color": self.color,
        }
        if self.radius is not None:
            d["radius"] = self.radius
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MapFeature:
        kwargs = {
            "type": d["type"],
            "coords": d["coords"],
            "mode": d.get("mode", GEOGRAPHIC),
            "name": str(d.get("name", "")),
            "color": str(d.get("color", "#ffb000")),
            "radius": d.get("radius"),
        }
        if d.get("id"):
            kwargs["id"] = str(d["id"])
        return cls(**kwargs)


# ── Measurement ──────────────────────────────────────────────────────────────


def path_length(coords: list[list[float]], mode: str) -> float:
    if len(coords) < 2:
        return 0.0
    if mode == LOCAL:
        return sum(math.dist(a, b) for a, b in itertools.pairwise(coords))
    lats = [p[0] for p in coords]
    lons = [p[1] for p in coords]
    return float(_GEOD.line_length(lons, lats))


def polygon_area_perimeter(ring: list[list[float]], mode: str) -> tuple[float, float]:
    if mode == LOCAL:
        n = len(ring)
        area = 0.5 * abs(
            sum(
                ring[i][1] * ring[(i + 1) % n][0] - ring[(i + 1) % n][1] * ring[i][0]
                for i in range(n)
            )
        )
        return area, path_length(ring + [ring[0]], mode)
    lats = [p[0] for p in ring]
    lons = [p[1] for p in ring]
    area, perim = _GEOD.polygon_area_perimeter(lons, lats)
    return abs(float(area)), float(perim)


def measure(feature: MapFeature) -> Measurement:
    m = feature.mode
    if feature.type == "point":
        return Measurement(mode=m)
    if feature.type == "line":
        return Measurement(length=path_length(feature.coords, m), mode=m)
    if feature.type == "circle":
        r = feature.radius
        return Measurement(length=2 * math.pi * r, area=math.pi * r * r, mode=m)
    area, perim = polygon_area_perimeter(feature.ring(), m)
    return Measurement(length=perim, area=area, mode=m)


def ruler_label(coords: list[list[float]], mode: str) -> str:
    """Total plus last-segment length for the ruler tool."""
    if len(coords) < 2:
        return ""
    total = path_length(coords, mode)
    last = path_length(coords[-2:], mode)
    if len(coords) == 2:
        return format_length(total, mode)
    return f"{format_length(total, mode)} (відрізок {format_length(last, mode)})"


def format_length(value: float, mode: str) -> str:
    if mode == LOCAL:
        return f"{value:.2f} од."
    if value < 1000:
        return f"{value:.1f} м"
    return f"{value / 1000:.3f} км"


def format_area(value: float, mode: str) -> str:
    if mode == LOCAL:
        return f"{value:.2f} од.²"
    if value < 10_000:
        return f"{value:.0f} м²"
    if value < 1_000_000:
        return f"{value / 10_000:.2f} га"
    return f"{value / 1_000_000:.3f} км²"


# ── Store ────────────────────────────────────────────────────────────────────


class MapFeatureStore:
    """Holds features of both coordinate modes; the map shows one mode at a time."""

    def __init__(self):
        self._features: dict[str, MapFeature] = {}

    def __len__(self) -> int:
        return len(self._features)

    def get(self, feature_id: str) -> MapFeature | None:
        return self._features.get(feature_id)

    def upsert(self, feature: MapFeature) -> MapFeature:
        self._features[feature.id] = feature
        return feature

    def remove(self, feature_id: str) -> bool:
        return self._features.pop(feature_id, None) is not None

    def clear(self, mode: str | None = None) -> None:
        if mode is None:
            self._features.clear()
        else:
            self._features = {k: f for k, f in self._features.items() if f.mode != mode}

    def features(self, mode: str | None = None) -> list[MapFeature]:
        return [f for f in self._features.values() if mode is None or f.mode == mode]

    def view_payload(self, mode: str) -> list[dict[str, Any]]:
        """Features of one mode, each with its measurement label, for the map page."""
        out = []
        for f in self.features(mode):
            d = f.to_dict()
            d["measure"] = measure(f).label()
            out.append(d)
        return out

    # GeoJSON: GEOGRAPHIC features use real [lon, lat] geometry; LOCAL ones are
    # stored with [x, y] geometry and properties.mode == "LOCAL" (not georeferenced).
    def to_geojson(self) -> dict[str, Any]:
        items = []
        for f in self._features.values():
            xy = [[p[1], p[0]] for p in f.coords]
            if f.type in ("point", "circle"):
                geom = {"type": "Point", "coordinates": xy[0]}
            elif f.type == "line":
                geom = {"type": "LineString", "coordinates": xy}
            else:
                ring = [[p[1], p[0]] for p in f.ring()]
                geom = {"type": "Polygon", "coordinates": [ring + [ring[0]]]}
            props = {"id": f.id, "kind": f.type, "name": f.name, "color": f.color, "mode": f.mode}
            if f.type == "rectangle":
                props["corners"] = f.coords
            if f.radius is not None:
                props["radius"] = f.radius
            items.append({"type": "Feature", "geometry": geom, "properties": props})
        return {"type": "FeatureCollection", "features": items}

    def load_geojson(self, data: dict[str, Any], *, replace: bool = False) -> int:
        """Import features; returns the count added. Skips items it cannot parse."""
        if replace:
            self.clear()
        added = 0
        for item in data.get("features", []):
            try:
                feature = _feature_from_geojson(item)
            except (KeyError, TypeError, ValueError, IndexError):
                continue
            self.upsert(feature)
            added += 1
        return added

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(self.to_geojson(), ensure_ascii=False, indent=1), encoding="utf-8"
        )
        tmp.replace(path)

    def load(self, path: Path, *, replace: bool = False) -> int:
        return self.load_geojson(json.loads(path.read_text(encoding="utf-8")), replace=replace)


def _feature_from_geojson(item: dict[str, Any]) -> MapFeature:
    geom = item["geometry"]
    props = item.get("properties") or {}
    gtype = geom["type"]
    coords = geom["coordinates"]
    if gtype == "Point":
        kind = props.get("kind") if props.get("kind") in ("point", "circle") else "point"
        pts = [coords]
    elif gtype == "LineString":
        kind, pts = "line", coords
    elif gtype == "Polygon":
        kind = "rectangle" if props.get("kind") == "rectangle" else "polygon"
        pts = coords[0]
        if len(pts) > 1 and pts[0] == pts[-1]:
            pts = pts[:-1]
    else:
        raise ValueError(f"Unsupported geometry {gtype}")
    latlon = [[p[1], p[0]] for p in pts]
    if kind == "rectangle" and props.get("corners"):
        latlon = props["corners"]
    d = {
        "type": kind,
        "coords": latlon,
        "mode": props.get("mode", GEOGRAPHIC),
        "name": props.get("name", ""),
        "color": props.get("color", "#ffb000"),
        "radius": props.get("radius"),
        "id": props.get("id"),
    }
    if kind == "circle" and d["radius"] is None:
        raise ValueError("circle without radius")
    return MapFeature.from_dict(d)
