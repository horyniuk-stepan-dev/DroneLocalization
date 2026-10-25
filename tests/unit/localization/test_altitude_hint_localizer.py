"""Localizer + localization.altitude_hint: which source the hint uses, what it sets."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.localization.altitude_hint import DEFAULT_FOOTPRINT_FACTOR, AltitudeHint
from src.localization.localizer import Localizer
from src.localization.scale_manager import ScaleManager

# Relief project: "3" = 500 m, "main" = 1000 m, "2" = 2000 m layer (m/px).
GSD = {"main": 0.72, "2": 1.51, "3": 0.33}
WIDTH = 1280


class Manager:
    def __init__(self, gsds):
        self.gsds = gsds

    @property
    def active_source_ids(self):
        return set(self.gsds)

    def layer_gsd(self, sid):
        return self.gsds.get(sid)


def localizer(prior=None, active="main", altitude=None, enabled=True, relative=False):
    loc = Localizer.__new__(Localizer)
    loc.db_manager = Manager(GSD)
    loc._active_source_id = active
    loc._layer_gsd_cache = {}
    loc._scale_manager = ScaleManager({})
    loc._scale_manager._prior = prior
    loc._altitude_hint = AltitudeHint(enabled=enabled)
    loc._telemetry_altitude_m = altitude
    loc._altitude_norm_width = None
    loc._altitude_hint_logged = set()
    loc.relative_mode = relative
    loc._frame_query_gsd = loc._estimated_query_gsd()
    return loc


def telemetry_gsd(altitude):
    return altitude * DEFAULT_FOOTPRINT_FACTOR / WIDTH


def test_telemetry_picks_the_nearest_layers_and_the_query_scale():
    loc = localizer(altitude=700.0)  # 0.365 m/px: between the 500 m and 1000 m layers
    loc._apply_altitude_hint(WIDTH)
    gsd = telemetry_gsd(700.0)
    assert loc._frame_query_gsd == pytest.approx(gsd)
    sm = loc._scale_manager
    assert sm.altitude_primary == pytest.approx(gsd / 0.33)
    assert sm.candidates()[:2] == [pytest.approx(gsd / 0.33), pytest.approx(gsd / 0.72)]


def test_without_telemetry_the_last_fix_scale_points_to_the_layers():
    loc = localizer(prior=1.0, active="2")  # last fix: as high as the 2000 m layer
    loc._apply_altitude_hint(WIDTH)
    assert loc._frame_query_gsd == pytest.approx(1.51)  # the visual estimate is kept
    # The prior stays first; the 1000 m layer's scale backs up a failed rescan.
    assert loc._scale_manager.candidates() == [1.0, pytest.approx(1.51 / 0.72)]


def test_relative_projects_ignore_metric_telemetry():
    loc = localizer(prior=1.0, active="2", altitude=700.0, relative=True)
    loc._apply_altitude_hint(WIDTH)
    assert loc._frame_query_gsd == pytest.approx(1.51)


def test_no_telemetry_and_no_fix_keeps_the_vision_only_scan():
    loc = localizer()
    loc._apply_altitude_hint(WIDTH)
    assert loc._scale_manager.altitude_primary is None
    assert loc._scale_manager.candidates() == [0.5, 0.7, 1.0, 1.4, 2.0]


def test_disabled_changes_nothing():
    loc = localizer(altitude=700.0, enabled=False)
    loc._apply_altitude_hint(WIDTH)
    assert loc._frame_query_gsd is None
    assert loc._scale_manager.candidates() == [0.5, 0.7, 1.0, 1.4, 2.0]
    assert loc._altitude_layer_scales(WIDTH) == []


def with_layer_search(loc, active, log_ratio=None):
    beliefs = {} if log_ratio is None else {active: SimpleNamespace(log_ratio=log_ratio)}
    loc._layer_search = SimpleNamespace(handoff=SimpleNamespace(active=active), beliefs=beliefs)
    return loc


def test_layer_search_hint_from_telemetry_or_the_active_belief():
    loc = with_layer_search(localizer(altitude=1500.0), "3", log_ratio=0.0)
    gsd = telemetry_gsd(1500.0)  # 0.78 m/px: the 1000 m layer, then the 2000 m one
    assert loc._altitude_layer_scales(WIDTH) == [
        ("main", pytest.approx(gsd / 0.72)),
        ("2", pytest.approx(gsd / 1.51)),
    ]
    # No telemetry: the active layer's belief (as high as the 500 m layer).
    visual = with_layer_search(localizer(), "3", log_ratio=float(np.log(1.0)))
    assert visual._altitude_layer_scales(WIDTH) == [
        ("3", pytest.approx(1.0)),
        ("main", pytest.approx(0.33 / 0.72)),
    ]
    # Neither: no hint, the layer search runs as before.
    assert with_layer_search(localizer(), "3")._altitude_layer_scales(WIDTH) == []


def test_fix_with_telemetry_refines_the_footprint():
    loc = localizer(prior=1.0, active="main", altitude=1000.0)
    loc._altitude_norm_width = WIDTH
    before = loc._altitude_hint.footprint_factor
    loc._learn_altitude_factor()
    measured = 0.72 * WIDTH / 1000.0
    assert loc._altitude_hint.footprint_factor == pytest.approx(0.7 * before + 0.3 * measured)


def test_layer_search_takes_hinted_layers_right_after_the_active_one():
    from src.localization.layer_search import LayerSearch

    search = LayerSearch({"localization": {"layer_search": {"max_sources_per_frame": 3}}})
    search.handoff.active = "a"
    manager = SimpleNamespace(all_source_ids=["a", "b", "c", "d", "e"])
    for _ in range(4):
        selected = search._select_sources(manager, ["d", "e"])
        assert selected[:2] == ["a", "d"]  # cap 3: one probe slot stays free
        assert len(selected) == 3 and len(set(selected)) == 3
    assert search._select_sources(SimpleNamespace(all_source_ids=["a", "b"]), ["b"]) == [
        "a",
        "b",
    ]
