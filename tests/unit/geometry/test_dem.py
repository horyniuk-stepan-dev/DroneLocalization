"""DEM loading and sampling (src/geometry/dem.py) without network access."""

import json
import math

import cv2
import numpy as np
import pytest

from src.geometry import dem as D

ZOOM = 13
LAT, LON = 48.42, 26.18


def _tile_png(elev: np.ndarray) -> bytes:
    """Encode metres as a terrarium PNG (BGR order for cv2)."""
    v = elev.astype(np.float64) + 32768.0
    r = np.floor(v / 256.0)
    g = np.floor(v - r * 256.0)
    b = np.round((v - r * 256.0 - g) * 256.0)
    rgb = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", rgb[..., ::-1])
    assert ok
    return buf.tobytes()


def _elev_for_tile(x: int, y: int) -> np.ndarray:
    """Elevation = 100 + (global pixel column mod 2560) / 10: a ramp across tiles."""
    cols = (x * 256 + np.arange(256)) % 2560
    return np.tile(100.0 + cols / 10.0, (256, 1))


class FakeFetch:
    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def __call__(self, url):
        self.calls.append(url)
        z, x, y = (int(p.split(".")[0]) for p in url.split("/")[-3:])
        assert z == ZOOM
        if (x, y) in self.fail:
            raise OSError("HTTP 404")
        return _tile_png(_elev_for_tile(x, y))


def test_terrarium_decode_roundtrip():
    elev = np.array([[-50.0, 0.0, 123.25, 2034.5]])
    png = _tile_png(np.tile(elev, (2, 1)))
    rgb = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)[..., ::-1]
    np.testing.assert_allclose(D.terrarium_decode(rgb)[0], elev[0], atol=1 / 256)


def test_mercator_roundtrip_and_tile_range():
    x, y = D.lonlat_to_mercator(LON, LAT)
    lon, lat = D.mercator_to_lonlat(x, y)
    assert abs(lon - LON) < 1e-9 and abs(lat - LAT) < 1e-9
    # the FlightSimulator's bochkivtsi bbox at z15 is a 15 x 7 tile mosaic (3840 x 1792 px)
    x0, y0, x1, y1 = D.tile_range(48.3995, 26.102, 48.4435, 26.257, 15)
    assert (x1 - x0 + 1, y1 - y0 + 1) == (15, 7)


def test_load_terrarium_samples_and_caches(tmp_path):
    fetch = FakeFetch()
    bbox = (LAT - 0.02, LON - 0.03, LAT + 0.02, LON + 0.03)
    dem = D.load_terrarium(*bbox, ZOOM, tmp_path, fetch=fetch)
    assert dem is not None and len(fetch.calls) > 0
    # exact at a pixel centre: global column -> elevation 100 + col/10
    n = 2**ZOOM * 256
    gx = (LON + 180.0) / 360.0 * n
    col = math.floor(gx) + 0.5
    lon_c = col / n * 360.0 - 180.0
    val = dem.sample(np.array([LAT]), np.array([lon_c]))[0]
    assert abs(val - (100.0 + (math.floor(gx) % 2560) / 10.0)) < 0.01
    assert dem.covers(*bbox)
    # second load reads the PNG cache only
    again = D.load_terrarium(*bbox, ZOOM, tmp_path, download=False)
    np.testing.assert_array_equal(again.data, dem.data)
    assert np.isnan(dem.sample(np.array([10.0]), np.array([10.0]))[0])


def test_failed_tiles_are_nan_and_all_failed_is_none(tmp_path):
    bbox = (LAT - 0.02, LON - 0.03, LAT + 0.02, LON + 0.03)
    x0, y0, _x1, _y1 = D.tile_range(*bbox, ZOOM)
    dem = D.load_terrarium(*bbox, ZOOM, tmp_path / "a", fetch=FakeFetch(fail={(x0, y0)}))
    assert np.isnan(dem.data[:256, :256]).all() and np.isfinite(dem.data[-1, -1])
    every = {(x, y) for x in range(x0, x0 + 10) for y in range(y0, y0 + 10)}
    assert D.load_terrarium(*bbox, ZOOM, tmp_path / "b", fetch=FakeFetch(fail=every)) is None
    assert D.load_terrarium(*bbox, ZOOM, tmp_path / "c", download=False) is None


def _write_geotiff(path, data, origin, pixel, epsg=3857, mode="F"):
    from PIL import Image, TiffImagePlugin, TiffTags

    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (float(pixel), float(pixel), 0.0)
    ifd.tagtype[33550] = TiffTags.DOUBLE
    ifd[33922] = (0.0, 0.0, 0.0, float(origin[0]), float(origin[1]), 0.0)
    ifd.tagtype[33922] = TiffTags.DOUBLE
    key = 3072 if epsg != 4326 else 2048
    ifd[34735] = (1, 1, 0, 1, key, 0, 1, epsg)
    ifd.tagtype[34735] = TiffTags.SHORT
    image = Image.fromarray(data, mode=mode) if mode == "F" else Image.fromarray(data)
    image.save(path, tiffinfo=ifd)


def test_geotiff_metres_and_terrarium(tmp_path, monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_rasterio(name, *args, **kwargs):  # exercise the PIL reader
        if name == "rasterio":
            raise ImportError
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_rasterio)
    x0, y0 = (float(v) for v in D.lonlat_to_mercator(LON - 0.05, LAT + 0.03))
    data = np.fromfunction(lambda r, c: 200.0 + r + 0.5 * c, (300, 400), dtype=np.float32)
    path = tmp_path / "dem_m.tif"
    _write_geotiff(path, data.astype(np.float32), (x0, y0), 20.0)
    dem = D.load_geotiff(path)
    assert dem.crs == "EPSG:3857" and dem.shape == (300, 400)
    r, c = 120, 250
    lon, lat = D.mercator_to_lonlat(x0 + (c + 0.5) * 20.0, y0 - (r + 0.5) * 20.0)
    assert abs(dem.sample(lat, lon) - data[r, c]) < 1e-3

    rgb = cv2.imdecode(np.frombuffer(_tile_png(data[:256, :256]), np.uint8), cv2.IMREAD_UNCHANGED)
    rgb = np.ascontiguousarray(rgb[..., ::-1])
    path2 = tmp_path / "dem_rgb.tif"
    _write_geotiff(path2, rgb, (x0, y0), 20.0, mode="RGB")
    dem2 = D.load_geotiff(path2)
    assert abs(dem2.sample(lat, lon) - data[r, c]) < 0.01  # (r, c) lies in the 256 px crop
    np.testing.assert_allclose(dem2.data, data[:256, :256], atol=1 / 256)


def test_geotiff_4326(tmp_path):
    data = np.full((50, 60), 321.0, dtype=np.float32)
    path = tmp_path / "dem_ll.tif"
    _write_geotiff(path, data, (LON - 0.1, LAT + 0.1), 0.004, epsg=4326)
    dem = D.load_geotiff(path)
    assert dem.crs == "EPSG:4326"
    assert abs(dem.sample(LAT, LON) - 321.0) < 1e-6


def test_resolve_dem_and_project_dir(tmp_path):
    project = tmp_path / "proj"
    (project / "sources" / "main").mkdir(parents=True)
    (project / "project.json").write_text(json.dumps({"project_name": "x"}))
    assert D.find_project_dir(project / "sources" / "main") == project
    assert D.find_project_dir(tmp_path) is None
    bbox = (LAT - 0.01, LON - 0.01, LAT + 0.01, LON + 0.01)
    dem = D.resolve_dem("terrarium", "", ZOOM, bbox, project / "terrain", fetch=FakeFetch())
    assert dem is not None and (project / "terrain" / "terrarium" / str(ZOOM)).is_dir()
    assert D.resolve_dem("file", "", ZOOM, bbox, project) is None
    assert D.resolve_dem("file", str(tmp_path / "missing.tif"), ZOOM, bbox, project) is None
    assert D.resolve_dem("other", "", ZOOM, bbox, project) is None
    with pytest.raises(ValueError):
        D.load_terrarium(-80, -170, 80, 170, 15, tmp_path, download=False)
