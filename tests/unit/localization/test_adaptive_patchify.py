"""Localizer: query scale estimate, adaptive patchify and the effective main layer."""

from types import SimpleNamespace

from src.localization.localizer import Localizer

GSD = {"main": 0.72, "2": 1.51, "3": 0.33}


class Manager:
    def __init__(self, gsds, steps=None):
        self.gsds = gsds
        self.steps = steps or {}

    @property
    def active_source_ids(self):
        return set(self.gsds)

    def layer_gsd(self, sid):
        return self.gsds.get(sid)

    def effective_priority_steps(self):
        return dict(self.steps)

    def get_source_config(self, sid):
        return SimpleNamespace(priority={"main": 0, "2": 1, "3": 2}.get(sid, 0))


def localizer(gsds, prior, active, adaptive=True, weight=0.0, steps=None):
    loc = Localizer.__new__(Localizer)
    loc.db_manager = Manager(gsds, steps)
    loc._active_source_id = active
    loc._scale_manager = SimpleNamespace(prior=prior)
    loc._patchify_adaptive = adaptive
    loc._patchify_gap_ratio = 1.5
    loc._fineness_weight = weight
    loc._layer_gsd_cache = {}
    loc.config = {"localization": {"patchify_grids": [[1, 1], [2, 2], [3, 3]]}}
    return loc


def test_query_scale_is_the_prior_times_the_active_layer_pixel():
    loc = localizer(GSD, 0.5, "2")

    assert abs(loc._estimated_query_gsd() - 0.755) < 1e-9
    assert localizer(GSD, None, "2")._estimated_query_gsd() is None


def test_three_layers_skip_patchify_two_layers_use_the_coarse_one():
    three = localizer(GSD, 1.0, "main")
    three._frame_query_gsd = three._estimated_query_gsd()
    assert not any(three._patchify_allowed(sid) for sid in GSD)

    two = localizer({"3": 0.33, "2": 1.51}, 0.5, "2")
    two._frame_query_gsd = two._estimated_query_gsd()
    assert two._patchify_allowed("2") and not two._patchify_allowed("3")


def test_adaptive_off_or_unknown_scale_keeps_patchify():
    off = localizer(GSD, 1.0, "main", adaptive=False)
    off._frame_query_gsd = 0.72
    assert off._patchify_allowed("main")

    unknown = localizer(GSD, None, "main")
    unknown._frame_query_gsd = None
    assert unknown._patchify_allowed("main")


def test_main_layer_follows_the_effective_priority():
    loc = localizer(GSD, 1.0, "main", weight=2.0, steps={"main": 2.3, "2": 5.4, "3": 2.0})
    assert loc._main_source_id() == "3"

    plain = localizer(GSD, 1.0, "main", weight=0.0)
    assert plain._main_source_id() == "main"
