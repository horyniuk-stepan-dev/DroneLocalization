"""Download (or check) the open-data DEM for a project before propagation.

Propagation with ``graph_optimization.terrain_scale_prior`` fetches the DEM by
itself on first use; this script does it ahead of time, e.g. right after the
project is created and anchored, while the machine is online. Tiles land in
``<project>/terrain/terrarium/<zoom>/`` (AWS Terrain Tiles, open data), the
same folder the propagation reads.

    python scripts/fetch_dem.py --project "D:/My Projects/TEST/testtopboch_max"
    python scripts/fetch_dem.py --project P --bbox 48.39 26.10 48.45 26.26 --zoom 13

Without ``--bbox`` the area is the footprint of every anchor in every layer's
calibration.json, plus ``--margin-km``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def anchors_bbox(project: Path) -> tuple[float, float, float, float]:
    """(lat_min, lon_min, lat_max, lon_max) of all anchor footprints in the project."""
    import numpy as np

    from src.calibration.multi_anchor_calibration import MultiAnchorCalibration

    data = json.loads((project / "project.json").read_text(encoding="utf-8"))
    files = [s.get("calibration_file") for s in data.get("video_sources") or []]
    files = [f for f in files if f] or [data.get("calibration_filename")]
    lats: list[float] = []
    lons: list[float] = []
    for rel in files:
        path = project / rel
        if not path.is_file():
            continue
        cal = MultiAnchorCalibration()
        cal.load(str(path))
        frame = json.loads(path.read_text(encoding="utf-8")).get("frame_size") or [1920, 1080]
        w, h = float(frame[0]), float(frame[1])
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1], [w / 2, h / 2, 1]]).T
        for anchor in cal.anchors:
            pts = np.asarray(anchor.affine_matrix, dtype=np.float64) @ corners
            for x, y in pts.T:
                lat, lon = cal.converter.metric_to_gps(float(x), float(y))
                lats.append(lat)
                lons.append(lon)
    if not lats:
        raise SystemExit("no anchors found in the project's calibration files; pass --bbox")
    return min(lats), min(lons), max(lats), max(lons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--bbox", nargs=4, type=float, metavar=("LAT_MIN", "LON_MIN", "LAT_MAX", "LON_MAX")
    )
    parser.add_argument("--zoom", type=int, default=None, help="default: config terrain_dem_zoom")
    parser.add_argument("--margin-km", type=float, default=2.0)
    parser.add_argument("--offline", action="store_true", help="only report what the cache holds")
    args = parser.parse_args(argv)

    from config import APP_CONFIG, get_cfg
    from src.geometry.dem import load_terrarium

    zoom = args.zoom or int(get_cfg(APP_CONFIG, "graph_optimization.terrain_dem_zoom", 13))
    lat_min, lon_min, lat_max, lon_max = args.bbox or anchors_bbox(args.project)
    dlat = args.margin_km / 111.0
    dlon = args.margin_km / (111.0 * max(math.cos(math.radians((lat_min + lat_max) / 2)), 0.1))
    bbox = (lat_min - dlat, lon_min - dlon, lat_max + dlat, lon_max + dlon)
    cache = args.project / "terrain"
    print(
        f"bbox {bbox[0]:.5f},{bbox[1]:.5f} .. {bbox[2]:.5f},{bbox[3]:.5f}  zoom {zoom}  -> {cache}"
    )
    dem = load_terrarium(*bbox, zoom, cache, download=not args.offline)
    if dem is None:
        print("DEM unavailable (no tiles)")
        return 1
    import numpy as np

    finite = np.isfinite(dem.data)
    print(
        f"DEM ok: {dem.shape[1]}x{dem.shape[0]} px, {finite.mean() * 100:.0f}% filled, "
        f"elevation {np.nanmin(dem.data):.0f}..{np.nanmax(dem.data):.0f} m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
