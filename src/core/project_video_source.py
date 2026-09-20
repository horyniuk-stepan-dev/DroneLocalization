from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class ScaleLayerMetadata:
    """Versioned, source-level scale metadata used for layer discovery.

    Metric GSD is optional because older or externally produced databases may
    only be known to be a distinct layer.  Missing values remain unknown; they
    are never converted into an altitude estimate.
    """

    version: int = 1
    layer_id: str = ""
    nominal_gsd_m_per_px: float | None = None
    min_gsd_m_per_px: float | None = None
    max_gsd_m_per_px: float | None = None
    relative_scale: float | None = None
    scale_quality: str = "unknown"
    neighbor_layer_ids: tuple[str, ...] = ()
    descriptor_schema_fingerprint: str | None = None
    extra: dict[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        if self.version != 1:
            raise ValueError(f"Unsupported scale_layer version: {self.version}")
        if not self.layer_id.strip():
            raise ValueError("scale_layer.layer_id must be non-empty")
        if self.scale_quality not in {"unknown", "provisional", "verified"}:
            raise ValueError(
                "scale_layer.scale_quality must be unknown, provisional, or verified"
            )
        values = {
            "nominal_gsd_m_per_px": self.nominal_gsd_m_per_px,
            "min_gsd_m_per_px": self.min_gsd_m_per_px,
            "max_gsd_m_per_px": self.max_gsd_m_per_px,
            "relative_scale": self.relative_scale,
        }
        for name, value in values.items():
            if value is not None and (not isinstance(value, (int, float)) or value <= 0):
                raise ValueError(f"scale_layer.{name} must be a positive number")
        lo, nominal, hi = (
            self.min_gsd_m_per_px,
            self.nominal_gsd_m_per_px,
            self.max_gsd_m_per_px,
        )
        if lo is not None and hi is not None and lo > hi:
            raise ValueError("scale_layer min_gsd_m_per_px exceeds max_gsd_m_per_px")
        if nominal is not None and lo is not None and nominal < lo:
            raise ValueError("scale_layer nominal GSD is below its minimum")
        if nominal is not None and hi is not None and nominal > hi:
            raise ValueError("scale_layer nominal GSD is above its maximum")
        neighbors = tuple(self.neighbor_layer_ids)
        if any(not str(item).strip() for item in neighbors):
            raise ValueError("scale_layer neighbor IDs must be non-empty")
        if len(set(neighbors)) != len(neighbors):
            raise ValueError("scale_layer neighbor IDs must be unique")
        if self.layer_id in neighbors:
            raise ValueError("scale_layer cannot list itself as a neighbor")

    def to_dict(self) -> dict[str, Any]:
        data = dict(self.extra)
        data.update(
            {
                "version": self.version,
                "layer_id": self.layer_id,
                "nominal_gsd_m_per_px": self.nominal_gsd_m_per_px,
                "min_gsd_m_per_px": self.min_gsd_m_per_px,
                "max_gsd_m_per_px": self.max_gsd_m_per_px,
                "relative_scale": self.relative_scale,
                "scale_quality": self.scale_quality,
                "neighbor_layer_ids": list(self.neighbor_layer_ids),
                "descriptor_schema_fingerprint": self.descriptor_schema_fingerprint,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScaleLayerMetadata:
        known = {
            "version",
            "layer_id",
            "nominal_gsd_m_per_px",
            "min_gsd_m_per_px",
            "max_gsd_m_per_px",
            "relative_scale",
            "scale_quality",
            "neighbor_layer_ids",
            "descriptor_schema_fingerprint",
        }
        values = {key: value for key, value in data.items() if key in known}
        values["neighbor_layer_ids"] = tuple(values.get("neighbor_layer_ids", ()))
        values["extra"] = {key: value for key, value in data.items() if key not in known}
        return cls(**values)


@dataclass
class ProjectVideoSource:
    source_id: str
    area_id: str
    video_path: str
    database_file: str
    calibration_file: str
    description: str = ""
    enabled: bool = True
    priority: int = 0
    geo_bounds: tuple[float, float, float, float] | None = None
    camera_params: dict[str, Any] | None = None
    # Optional versioned layer description; legacy projects remain readable.
    # Unknown GSD/coverage is represented by absent fields, never invented height.
    scale_layer: ScaleLayerMetadata | None = None

    def __post_init__(self) -> None:
        if isinstance(self.scale_layer, dict):
            self.scale_layer = ScaleLayerMetadata.from_dict(self.scale_layer)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["geo_bounds"] is not None:
            d["geo_bounds"] = list(d["geo_bounds"])
        if self.scale_layer is not None:
            d["scale_layer"] = self.scale_layer.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProjectVideoSource:
        d = dict(data)
        gb = d.get("geo_bounds")
        if gb is not None:
            d["geo_bounds"] = tuple(gb)
        layer = d.get("scale_layer")
        if isinstance(layer, dict):
            d["scale_layer"] = ScaleLayerMetadata.from_dict(layer)
        import dataclasses

        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)

    def contains_point(self, lat: float, lon: float) -> bool:
        if self.geo_bounds is None:
            return True
        lat_min, lon_min, lat_max, lon_max = self.geo_bounds
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
