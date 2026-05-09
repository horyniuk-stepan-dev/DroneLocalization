from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any

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

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if d["geo_bounds"] is not None:
            d["geo_bounds"] = list(d["geo_bounds"])
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProjectVideoSource":
        d = dict(data)
        gb = d.get("geo_bounds")
        if gb is not None:
            d["geo_bounds"] = tuple(gb)
        import dataclasses
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        return cls(**d)


    def contains_point(self, lat: float, lon: float) -> bool:
        if self.geo_bounds is None:
            return True
        lat_min, lon_min, lat_max, lon_max = self.geo_bounds
        return lat_min <= lat <= lat_max and lon_min <= lon <= lon_max
