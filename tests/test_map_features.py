import math

import pytest

from src.gui.map_features import (
    GEOGRAPHIC,
    LOCAL,
    MapFeature,
    MapFeatureStore,
    format_area,
    format_length,
    measure,
    ruler_label,
)


def test_one_degree_latitude_at_equator():
    # WGS84 meridian arc 0..1 deg is ~110574 m
    f = MapFeature("line", [[0.0, 0.0], [1.0, 0.0]])
    assert measure(f).length == pytest.approx(110_574, rel=1e-3)


def test_km_square_area_at_50n():
    lat0, lon0 = 50.0, 30.0
    dlat = 1000 / 111_229  # ~1 km of latitude at 50N
    dlon = 1000 / (111_412.84 * math.cos(math.radians(lat0)))
    f = MapFeature(
        "polygon",
        [[lat0, lon0], [lat0 + dlat, lon0], [lat0 + dlat, lon0 + dlon], [lat0, lon0 + dlon]],
    )
    m = measure(f)
    assert m.area == pytest.approx(1_000_000, rel=0.01)
    assert m.length == pytest.approx(4_000, rel=0.01)


def test_rectangle_equals_polygon():
    r = MapFeature("rectangle", [[50.01, 30.02], [50.0, 30.0]])
    p = MapFeature("polygon", [[50.0, 30.0], [50.01, 30.0], [50.01, 30.02], [50.0, 30.02]])
    assert measure(r).area == pytest.approx(measure(p).area)


def test_circle():
    m = measure(MapFeature("circle", [[50, 30]], radius=100))
    assert m.area == pytest.approx(math.pi * 1e4)
    assert m.length == pytest.approx(200 * math.pi)


def test_local_mode_euclidean_units():
    f = MapFeature("polygon", [[0, 0], [0, 10], [5, 10], [5, 0]], mode=LOCAL)
    m = measure(f)
    assert m.area == pytest.approx(50)
    assert m.length == pytest.approx(30)
    assert "од." in m.label() and "м" not in m.label().replace("од.", "")


def test_validation():
    with pytest.raises(ValueError):
        MapFeature("polygon", [[0, 0], [1, 1]])
    with pytest.raises(ValueError):
        MapFeature("circle", [[0, 0]], radius=0)
    with pytest.raises(ValueError):
        MapFeature("blob", [[0, 0]])
    with pytest.raises(ValueError):
        MapFeature("point", [[float("nan"), 0]])


def test_formatting():
    assert format_length(12.34, GEOGRAPHIC) == "12.3 м"
    assert format_length(1234, GEOGRAPHIC) == "1.234 км"
    assert format_area(500, GEOGRAPHIC) == "500 м²"
    assert format_area(25_000, GEOGRAPHIC) == "2.50 га"
    assert format_area(2_500_000, GEOGRAPHIC) == "2.500 км²"


def test_ruler_label():
    assert ruler_label([[0, 0]], LOCAL) == ""
    assert ruler_label([[0, 0], [3, 4]], LOCAL) == "5.00 од."
    assert ruler_label([[0, 0], [3, 4], [3, 5]], LOCAL) == "6.00 од. (відрізок 1.00 од.)"


def test_store_modes_and_payload():
    s = MapFeatureStore()
    s.upsert(MapFeature("point", [[50, 30]], name="A"))
    s.upsert(MapFeature("line", [[0, 0], [3, 4]], mode=LOCAL))
    assert len(s.features(GEOGRAPHIC)) == 1
    payload = s.view_payload(LOCAL)
    assert payload[0]["measure"] == "5.00 од."
    s.clear(LOCAL)
    assert len(s) == 1


def test_geojson_roundtrip(tmp_path):
    s = MapFeatureStore()
    feats = [
        MapFeature("point", [[50.1, 30.2]], name="Точка"),
        MapFeature("line", [[50, 30], [50.1, 30.1]]),
        MapFeature("polygon", [[50, 30], [50.1, 30], [50.1, 30.1]]),
        MapFeature("rectangle", [[50, 30], [50.1, 30.1]]),
        MapFeature("circle", [[50, 30]], radius=250),
        MapFeature("polygon", [[0, 0], [0, 10], [5, 10]], mode=LOCAL),
    ]
    for f in feats:
        s.upsert(f)
    path = tmp_path / "f.geojson"
    s.save(path)
    s2 = MapFeatureStore()
    assert s2.load(path) == len(feats)
    for f in feats:
        g = s2.get(f.id)
        assert g is not None and g.to_dict() == f.to_dict()


def test_geojson_geometry_is_lonlat():
    s = MapFeatureStore()
    s.upsert(MapFeature("point", [[50.0, 30.0]]))
    assert s.to_geojson()["features"][0]["geometry"]["coordinates"] == [30.0, 50.0]


def test_load_skips_garbage():
    s = MapFeatureStore()
    n = s.load_geojson(
        {
            "features": [
                {"geometry": {"type": "Bogus"}},
                {"nope": 1},
                {"type": "Feature", "geometry": {"type": "Point", "coordinates": [30, 50]}},
            ]
        }
    )
    assert n == 1
