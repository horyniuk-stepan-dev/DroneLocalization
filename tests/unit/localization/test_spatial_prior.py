"""SpatialPrior: dead reckoning from the last fixes and the nearest-frame query."""

import math

import numpy as np
import pytest

from src.localization.spatial_prior import EARTH_RADIUS_M, SpatialPrior

LAT0, LON0 = 48.42, 26.12
M_LAT = math.degrees(1.0 / EARTH_RADIUS_M)  # degrees of latitude per metre
M_LON = M_LAT / math.cos(math.radians(LAT0))


def _at(north_m, east_m):
    return LAT0 + north_m * M_LAT, LON0 + east_m * M_LON


def _prior(**kw):
    return SpatialPrior(enabled=True, **kw)


def test_disabled_by_default_and_from_config():
    assert not SpatialPrior().enabled
    assert SpatialPrior.from_config({}).enabled is False
    cfg = {
        "localization": {
            "spatial_prior": True,
            "spatial_prior_max_age_s": 12,
            "spatial_prior_nearest_k": 3,
            "spatial_prior_radius_m": 800,
            "spatial_prior_verify": 1,
        },
        "tracking": {"max_speed_mps": 50},
    }
    sp = SpatialPrior.from_config(cfg)
    assert sp.enabled and sp.max_age_s == 12 and sp.nearest_k == 3
    assert sp.radius_m == 800 and sp.verify == 1 and sp.max_speed_mps == 50
    assert SpatialPrior.from_config(cfg, enabled=False).enabled is False  # relative maps


def test_constant_velocity_prediction():
    sp = _prior()
    sp.record_fix(sp.tick(10.0), *_at(0, 0), angle=33.0, scale=0.9)
    assert sp.predict(sp.tick(11.0)) == pytest.approx(_at(0, 0))  # one fix: stand still
    sp.record_fix(sp.tick(12.0), *_at(40, 30))  # 25 m/s; angle / scale kept
    assert sp.angle == 33.0 and sp.scale == 0.9
    lat, lon = sp.predict(sp.tick(16.0))  # 4 s after the last fix
    assert lat == pytest.approx(_at(120, 90)[0], abs=1e-9)
    assert lon == pytest.approx(_at(120, 90)[1], abs=1e-9)
    assert sp.age(16.0) == 4.0


def test_too_old_fix_and_implausible_speed():
    sp = _prior(max_age_s=30.0, max_speed_mps=100.0)
    sp.record_fix(sp.tick(0.0), *_at(0, 0))
    sp.record_fix(sp.tick(1.0), *_at(0, 500))  # 500 m/s: a wrong fix, no velocity
    assert sp.predict(sp.tick(5.0)) == pytest.approx(_at(0, 500))
    assert sp.predict(sp.tick(31.5)) is None  # 30.5 s since the last fix
    assert SpatialPrior(enabled=False).predict(0.0) is None


def test_time_without_timestamp_and_backwards():
    sp = _prior()
    assert sp.tick(None, dt=2.0) == 2.0 and sp.tick(None, dt=0.5) == 2.5
    sp.record_fix(2.5, *_at(0, 0), angle=10.0)
    assert sp.tick(1.0) == 1.0  # new video / seek: stored fixes dropped
    assert sp.predict(1.0) is None and sp.angle is None
    sp.reset()
    assert sp.age(0.0) is None


class _Map:
    def __init__(self, offsets_m, status=None):
        self.frame_gps = np.array([_at(n, e) for n, e in offsets_m])
        self.frame_georef_status = status


def test_nearest_ids_order_radius_exclude_and_usable():
    db = _Map([(0, 0), (900, 0), (50, 0), (-20, 10), (2000, 0), (float("nan"), 0), (30, 0)])
    sp = _prior(nearest_k=3, radius_m=1500.0)
    assert sp.nearest_ids(db, *_at(0, 0)) == [0, 3, 6]
    assert sp.nearest_ids(db, *_at(0, 0), exclude=[3]) == [0, 6, 2]
    assert sp.nearest_ids(db, *_at(0, 0), usable=lambda c: c not in (0, 6)) == [3, 2, 1]
    # 2000 m and NaN frames never qualify
    assert sp.nearest_ids(db, *_at(0, 0), usable=lambda c: c in (4, 5)) == []
    db.frame_georef_status = np.array([1, 1, 1, 2, 1, 1, 1])  # PROVISIONAL skipped
    assert sp.nearest_ids(db, *_at(0, 0)) == [0, 6, 2]


def test_nearest_ids_without_map_positions():
    sp = _prior()
    assert sp.nearest_ids(object(), *_at(0, 0)) == []
    empty = _Map([])
    empty.frame_gps = np.zeros((0, 2))
    assert sp.nearest_ids(empty, *_at(0, 0)) == []
