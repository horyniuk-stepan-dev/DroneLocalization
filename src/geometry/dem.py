"""Digital elevation model (DEM): open-data download, local rasters, sampling.

Used by the terrain scale model of calibration propagation
(``graph_optimization.terrain_scale_prior``). Map building happens before the
flight, on a machine with internet, so the DEM is fetched once and cached in the
project; nothing here runs during GPS-denied localization.

Sources
-------
* ``terrarium`` - AWS Terrain Tiles (open data, registry.opendata.aws/terrain-tiles):
  256 px Web-Mercator PNG tiles, elevation = R*256 + G + B/256 - 32768 metres.
  The same source the FlightSimulator renders its relief from, so a simulator
  run and the propagation see the same terrain. Plain PNG: no GDAL needed.
* ``file`` - a local GeoTIFF (EPSG:3857, EPSG:4326 or any EPSG pyproj knows),
  one band in metres or an RGB(A) terrarium raster such as the simulator's
  ``.tile_cache/elevation_*.tif``. Read with rasterio when installed, else PIL.

Elevations are metres of the source's vertical datum. Only height differences
inside one mission matter here, so the datum cancels.
"""

from __future__ import annotations

import math
import urllib.request
from collections.abc import Callable
from pathlib import Path

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)

EARTH_RADIUS_M = 6378137.0
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_PX = 256
_MAX_TILES = 4096


# ── Web Mercator helpers (closed form, vectorised) ─────────────────────────────


def lonlat_to_mercator(lon, lat) -> tuple[np.ndarray, np.ndarray]:
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.clip(np.asarray(lat, dtype=np.float64), -85.05112878, 85.05112878)
    x = EARTH_RADIUS_M * np.radians(lon)
    y = EARTH_RADIUS_M * np.log(np.tan(np.pi / 4.0 + np.radians(lat) / 2.0))
    return x, y


def mercator_to_lonlat(x, y) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    lon = np.degrees(x / EARTH_RADIUS_M)
    lat = np.degrees(2.0 * np.arctan(np.exp(y / EARTH_RADIUS_M)) - np.pi / 2.0)
    return lon, lat


def tile_range(
    lat_min: float, lon_min: float, lat_max: float, lon_max: float, zoom: int
) -> tuple[int, int, int, int]:
    """Inclusive XYZ tile range (x0, y0, x1, y1) covering the bbox."""
    n = 2**zoom

    def tx(lon):
        return int(np.clip(math.floor((lon + 180.0) / 360.0 * n), 0, n - 1))

    def ty(lat):
        lat = max(min(lat, 85.05112878), -85.05112878)
        r = math.radians(lat)
        return int(
            np.clip(math.floor((1.0 - math.asinh(math.tan(r)) / math.pi) / 2.0 * n), 0, n - 1)
        )

    return tx(lon_min), ty(lat_max), tx(lon_max), ty(lat_min)


def terrarium_decode(rgb: np.ndarray) -> np.ndarray:
    """RGB(A) uint8 terrarium pixels -> float32 metres."""
    a = np.asarray(rgb)
    r = a[..., 0].astype(np.float64)
    g = a[..., 1].astype(np.float64)
    b = a[..., 2].astype(np.float64)
    return (r * 256.0 + g + b / 256.0 - 32768.0).astype(np.float32)


# ── Raster ─────────────────────────────────────────────────────────────────────


class RasterDEM:
    """Elevation raster with a north-up affine georeference.

    ``origin`` is the CRS coordinate of the top-left corner of pixel (0, 0) and
    ``pixel_size`` the (x, y) size of a pixel (y positive, rows go south).
    NaN marks no data. ``sample`` is bilinear; any NaN neighbour gives NaN.
    """

    def __init__(
        self,
        data: np.ndarray,
        origin: tuple[float, float],
        pixel_size: tuple[float, float],
        crs: str = "EPSG:3857",
        source: str = "",
    ) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        if self.data.ndim != 2:
            raise ValueError("DEM data must be 2-D")
        self.origin = (float(origin[0]), float(origin[1]))
        self.pixel_size = (float(pixel_size[0]), float(pixel_size[1]))
        self.crs = str(crs).upper()
        self.source = source
        self._transformer = None
        if self.crs not in ("EPSG:3857", "EPSG:4326"):
            from pyproj import Transformer

            self._transformer = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    def _to_crs(self, lat, lon) -> tuple[np.ndarray, np.ndarray]:
        if self.crs == "EPSG:3857":
            return lonlat_to_mercator(lon, lat)
        if self.crs == "EPSG:4326":
            return np.asarray(lon, dtype=np.float64), np.asarray(lat, dtype=np.float64)
        x, y = self._transformer.transform(np.asarray(lon), np.asarray(lat))
        return np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def sample(self, lat, lon) -> np.ndarray:
        """Bilinear elevation at WGS84 points; NaN outside the raster or on no-data."""
        from scipy.ndimage import map_coordinates

        lat = np.asarray(lat, dtype=np.float64)
        lon = np.asarray(lon, dtype=np.float64)
        x, y = self._to_crs(lat, lon)
        col = (x - self.origin[0]) / self.pixel_size[0] - 0.5
        row = (self.origin[1] - y) / self.pixel_size[1] - 0.5
        flat_r = np.ravel(row)
        flat_c = np.ravel(col)
        out = np.full(flat_r.shape, np.nan, dtype=np.float64)
        h, w = self.data.shape
        inside = (
            np.isfinite(flat_r)
            & np.isfinite(flat_c)
            & (flat_r >= -0.5)
            & (flat_r <= h - 0.5)
            & (flat_c >= -0.5)
            & (flat_c <= w - 0.5)
        )
        if np.any(inside):
            rr = np.clip(flat_r[inside], 0.0, h - 1.0)
            cc = np.clip(flat_c[inside], 0.0, w - 1.0)
            out[inside] = map_coordinates(
                self.data, np.vstack([rr, cc]), order=1, mode="nearest", prefilter=False
            )
        return out.reshape(np.shape(lat))

    def bounds_lonlat(self) -> tuple[float, float, float, float]:
        """(lat_min, lon_min, lat_max, lon_max) of the raster extent."""
        h, w = self.data.shape
        x0, y1 = self.origin
        x1 = x0 + w * self.pixel_size[0]
        y0 = y1 - h * self.pixel_size[1]
        if self.crs == "EPSG:4326":
            return y0, x0, y1, x1
        if self.crs == "EPSG:3857":
            lon0, lat0 = mercator_to_lonlat(x0, y0)
            lon1, lat1 = mercator_to_lonlat(x1, y1)
            return float(lat0), float(lon0), float(lat1), float(lon1)
        from pyproj import Transformer

        inv = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
        lons, lats = inv.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
        return float(min(lats)), float(min(lons)), float(max(lats)), float(max(lons))

    def covers(self, lat_min: float, lon_min: float, lat_max: float, lon_max: float) -> bool:
        b = self.bounds_lonlat()
        return b[0] <= lat_min and b[1] <= lon_min and b[2] >= lat_max and b[3] >= lon_max


# ── Terrarium tiles ────────────────────────────────────────────────────────────


def _default_fetch(url: str, timeout_s: float = 20.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "DroneLocalization-DEM/1.0"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 (fixed https URL)
        return response.read()


def _decode_png(payload: bytes) -> np.ndarray:
    import cv2

    image = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("not an RGB terrarium tile")
    return image[..., [2, 1, 0]]  # BGR(A) -> RGB


def load_terrarium(
    lat_min: float,
    lon_min: float,
    lat_max: float,
    lon_max: float,
    zoom: int,
    cache_dir: str | Path,
    *,
    download: bool = True,
    fetch: Callable[[str], bytes] | None = None,
    url_template: str = TERRARIUM_URL,
) -> RasterDEM | None:
    """Mosaic of terrarium tiles covering the bbox, cached as PNG files.

    Tiles are stored as ``<cache_dir>/terrarium/<z>/<x>/<y>.png`` exactly as
    downloaded. A tile that cannot be read or fetched stays NaN; returns None
    only when no tile is available at all.
    """
    zoom = int(zoom)
    x0, y0, x1, y1 = tile_range(lat_min, lon_min, lat_max, lon_max, zoom)
    n_tiles = (x1 - x0 + 1) * (y1 - y0 + 1)
    if n_tiles > _MAX_TILES:
        raise ValueError(f"DEM bbox needs {n_tiles} tiles at zoom {zoom}; lower the zoom")
    fetch = fetch or _default_fetch
    root = Path(cache_dir) / "terrarium" / str(zoom)
    mosaic = np.full(((y1 - y0 + 1) * TILE_PX, (x1 - x0 + 1) * TILE_PX), np.nan, np.float32)
    have = fetched = failed = 0
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            path = root / str(tx) / f"{ty}.png"
            payload = None
            if path.is_file():
                payload = path.read_bytes()
            elif download:
                try:
                    payload = fetch(url_template.format(z=zoom, x=tx, y=ty))
                    path.parent.mkdir(parents=True, exist_ok=True)
                    tmp = path.with_suffix(".tmp")
                    tmp.write_bytes(payload)
                    tmp.replace(path)
                    fetched += 1
                except Exception as exc:  # network / HTTP / disk
                    failed += 1
                    logger.warning(f"DEM tile z{zoom}/{tx}/{ty} unavailable: {exc}")
                    continue
            if payload is None:
                continue
            try:
                elev = terrarium_decode(_decode_png(payload))
            except Exception as exc:
                failed += 1
                logger.warning(f"DEM tile {path} unreadable: {exc}")
                continue
            r0 = (ty - y0) * TILE_PX
            c0 = (tx - x0) * TILE_PX
            mosaic[r0 : r0 + TILE_PX, c0 : c0 + TILE_PX] = elev[:TILE_PX, :TILE_PX]
            have += 1
    if have == 0:
        logger.warning(f"No DEM tiles available for zoom {zoom} ({n_tiles} needed)")
        return None
    tile_m = 2.0 * math.pi * EARTH_RADIUS_M / 2**zoom
    origin = (-math.pi * EARTH_RADIUS_M + x0 * tile_m, math.pi * EARTH_RADIUS_M - y0 * tile_m)
    px = tile_m / TILE_PX
    logger.info(
        f"DEM terrarium z{zoom}: {have}/{n_tiles} tiles ({fetched} downloaded, {failed} failed), "
        f"cache {root}"
    )
    return RasterDEM(mosaic, origin, (px, px), "EPSG:3857", source=f"terrarium z{zoom}")


# ── Local GeoTIFF ──────────────────────────────────────────────────────────────


def _geokey_epsg(geokeys) -> str | None:
    if not geokeys or len(geokeys) < 4:
        return None
    n = int(geokeys[3])
    keys = {}
    for i in range(n):
        key_id, location, _count, value = (int(v) for v in geokeys[4 + 4 * i : 8 + 4 * i])
        if location == 0:
            keys[key_id] = value
    for key_id in (3072, 2048):  # ProjectedCSTypeGeoKey, GeographicTypeGeoKey
        code = keys.get(key_id)
        if code and code != 32767:
            return f"EPSG:{code}"
    return None


def _band_to_metres(array: np.ndarray, elevation_format: str, nodata) -> np.ndarray:
    a = np.asarray(array)
    fmt = elevation_format
    if fmt == "auto":
        fmt = "terrarium" if (a.ndim == 3 and a.shape[2] >= 3 and a.dtype == np.uint8) else "meters"
    if fmt == "terrarium":
        if a.ndim != 3 or a.shape[2] < 3:
            raise ValueError("terrarium DEM needs at least three bands")
        out = terrarium_decode(a)
        if a.shape[2] >= 4:
            out[a[..., 3] == 0] = np.nan
        return out
    band = a[..., 0] if a.ndim == 3 else a
    out = band.astype(np.float32)
    if nodata is not None:
        out[band == nodata] = np.nan
    return out


def load_geotiff(path: str | Path, elevation_format: str = "auto") -> RasterDEM:
    """North-up GeoTIFF DEM (metres or terrarium RGB). rasterio if present, else PIL."""
    path = Path(path)
    if elevation_format not in ("auto", "terrarium", "meters"):
        raise ValueError("elevation_format must be auto, terrarium or meters")
    try:
        import rasterio  # type: ignore[import-not-found]
    except ImportError:
        rasterio = None
    if rasterio is not None:
        with rasterio.open(path) as src:
            t = src.transform
            if abs(t.b) > 1e-12 or abs(t.d) > 1e-12:
                raise ValueError(f"{path}: rotated rasters are not supported")
            bands = src.read()  # (bands, rows, cols)
            array = bands[0] if bands.shape[0] == 1 else np.moveaxis(bands, 0, -1)
            data = _band_to_metres(array, elevation_format, src.nodata)
            crs = src.crs.to_string() if src.crs else "EPSG:3857"
            return RasterDEM(data, (t.c, t.f), (t.a, -t.e), crs, source=str(path))

    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    with Image.open(path) as image:
        tags = image.tag_v2
        scale = tags.get(33550)
        tie = tags.get(33922)
        if not scale or not tie or len(tie) < 6:
            raise ValueError(f"{path}: no GeoTIFF georeference (ModelPixelScale/Tiepoint)")
        crs = _geokey_epsg(tags.get(34735))
        if crs is None:
            raise ValueError(f"{path}: CRS is not an EPSG code; reproject the DEM first")
        nodata_tag = tags.get(42113)
        nodata = float(str(nodata_tag).strip("\x00 ")) if nodata_tag else None
        array = np.asarray(image)
    origin = (tie[3] - tie[0] * scale[0], tie[4] + tie[1] * scale[1])
    data = _band_to_metres(array, elevation_format, nodata)
    return RasterDEM(data, origin, (scale[0], scale[1]), crs, source=str(path))


# ── Resolution from config ─────────────────────────────────────────────────────


def find_project_dir(start: str | Path) -> Path | None:
    """Closest ancestor of ``start`` that holds a project.json."""
    p = Path(start).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "project.json").is_file():
            return candidate
    return None


def resolve_dem(
    source: str,
    dem_path: str,
    zoom: int,
    bbox: tuple[float, float, float, float],
    default_cache_dir: str | Path,
    *,
    download: bool = True,
    fetch: Callable[[str], bytes] | None = None,
) -> RasterDEM | None:
    """DEM for ``bbox`` = (lat_min, lon_min, lat_max, lon_max) per config.

    ``file``: ``dem_path`` is a GeoTIFF. ``terrarium``: ``dem_path`` (if set) is
    the cache folder, else ``default_cache_dir``. Errors are logged; None means
    "no DEM", and the caller runs without the terrain model.
    """
    try:
        if source == "file":
            if not dem_path:
                logger.warning(
                    "terrain DEM source 'file' needs graph_optimization.terrain_dem_path"
                )
                return None
            dem = load_geotiff(dem_path)
            if not dem.covers(*bbox):
                logger.warning(f"DEM {dem_path} does not cover the whole mission bbox {bbox}")
            return dem
        if source == "terrarium":
            cache = Path(dem_path) if dem_path else Path(default_cache_dir)
            return load_terrarium(*bbox, zoom, cache, download=download, fetch=fetch)
        logger.warning(f"Unknown terrain DEM source: {source!r}")
    except Exception as exc:
        logger.warning(f"DEM unavailable ({source}): {exc}")
    return None
