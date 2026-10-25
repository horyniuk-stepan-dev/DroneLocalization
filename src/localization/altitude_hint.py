"""Altitude hint for multi-layer search (``localization.altitude_hint``, default off).

The layers of an area are flown at different altitudes, so each has its own
ground pixel size (GSD). Knowing roughly how high the drone is gives the scale
at which the query matches each layer: r = query GSD / layer GSD (ScaleManager's
convention: r > 1 = flying higher than the layer). The layers nearest in scale
are then tried first instead of scanning the whole scale pyramid.

Where the query GSD comes from, in this order:

1. telemetry (``flight_data``: height above ground, else above take-off), times
   the camera's footprint factor (ground width per metre of altitude = sensor
   width / focal length), divided by the normalized frame width; the factor is
   refined from visual fixes, so a wrong camera setting corrects itself;
2. without telemetry: the last fix's own scale (scale prior x that layer's GSD),
   which still points to the other layers when the altitude drifts toward them;
3. neither: no hint, and the search is exactly the vision-only one.

A rough altitude is enough: layers are hundreds of metres apart, and a wrong
hint costs only the passes it adds before the usual scan.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from config import get_cfg

# DroneSimUE / FlightSimulator camera (8.8 mm sensor, 13.2 mm lens, HFOV 36.87 deg).
DEFAULT_FOOTPRINT_FACTOR = 8.8 / 13.2
# ScaleManager's clip range of the query/layer ratio.
MIN_RATIO = 0.3
MAX_RATIO = 3.5
# A visual fix may move the footprint factor at most this far from the camera model
# (a mismatch beyond it means wrong map units or a bad fix, not a camera error).
FACTOR_TRUST = 3.0


def _positive(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0.0 else None


def footprint_factor_from_settings(settings: Any) -> float:
    """Sensor width / focal length of the project camera, else the default camera's."""
    focal = _positive(getattr(settings, "focal_length_mm", None))
    sensor = _positive(getattr(settings, "sensor_width_mm", None))
    if focal is None or sensor is None:
        return DEFAULT_FOOTPRINT_FACTOR
    return sensor / focal


@dataclass
class AltitudeHint:
    enabled: bool = False
    max_layers: int = 2
    ema: float = 0.3
    footprint_factor: float = DEFAULT_FOOTPRINT_FACTOR
    # The camera model the learned factor may not leave by more than FACTOR_TRUST.
    model_factor: float = field(default=DEFAULT_FOOTPRINT_FACTOR)

    @classmethod
    def from_config(cls, config: Any, project_settings: Any = None) -> AltitudeHint:
        factor = footprint_factor_from_settings(project_settings)
        return cls(
            enabled=bool(get_cfg(config, "localization.altitude_hint.enabled", False)),
            max_layers=max(1, int(get_cfg(config, "localization.altitude_hint.max_layers", 2))),
            ema=float(get_cfg(config, "localization.altitude_hint.telemetry_scale_ema", 0.3)),
            footprint_factor=factor,
            model_factor=factor,
        )

    def query_gsd(
        self, altitude_m: float | None, norm_width_px: float | None, visual_gsd: float | None
    ) -> tuple[float | None, str]:
        """(query ground pixel size, "telemetry" | "visual" | "none" | "off")."""
        if not self.enabled:
            return None, "off"
        altitude = _positive(altitude_m)
        width = _positive(norm_width_px)
        if altitude is not None and width is not None:
            return altitude * self.footprint_factor / width, "telemetry"
        visual = _positive(visual_gsd)
        if visual is not None:
            return visual, "visual"
        return None, "none"

    def layer_scales(
        self, query_gsd: float | None, layer_gsds: Mapping[Any, float | None]
    ) -> list[tuple[Any, float]]:
        """(layer, query/layer ratio) of the layers nearest in scale, nearest first.

        Layers without a known GSD, or whose ratio is outside the range a frame can
        be normalized to, get no hint.
        """
        query = _positive(query_gsd)
        if not self.enabled or query is None:
            return []
        ratios = []
        for layer, gsd in layer_gsds.items():
            known = _positive(gsd)
            if known is None:
                continue
            ratio = query / known
            if MIN_RATIO <= ratio <= MAX_RATIO:
                ratios.append((layer, ratio))
        ratios.sort(key=lambda item: (abs(math.log(item[1])), str(item[0])))
        return ratios[: self.max_layers]

    def learn(
        self, altitude_m: float | None, norm_width_px: float | None, visual_gsd: float | None
    ) -> bool:
        """Refines the footprint factor from a fix made while telemetry was present."""
        altitude = _positive(altitude_m)
        width = _positive(norm_width_px)
        visual = _positive(visual_gsd)
        if not self.enabled or altitude is None or width is None or visual is None:
            return False
        measured = visual * width / altitude
        if not self.model_factor / FACTOR_TRUST <= measured <= self.model_factor * FACTOR_TRUST:
            return False
        self.footprint_factor = (1.0 - self.ema) * self.footprint_factor + self.ema * measured
        return True
