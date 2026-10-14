"""Localizer: patchify-only candidates after the main layer's (patchify_main_rank_margin)."""

from types import SimpleNamespace

from src.localization.localizer import Localizer
from src.localization.rotation_selector import RotationResult


class Manager:
    databases = {"main": "DB_MAIN", "2": "DB_2"}
    priorities = {"main": 0, "2": 1}

    @property
    def active_source_ids(self):
        return {"main", "2"}

    def get_source_config(self, sid):
        return SimpleNamespace(priority=self.priorities[sid])

    def get_database(self, sid):
        return self.databases[sid]


def localizer(margin, verified_on=None):
    loc = Localizer.__new__(Localizer)
    loc._patchify_main_margin = margin
    loc.db_manager = Manager()
    loc.calib_manager = SimpleNamespace(get=lambda sid: f"CAL_{sid}")
    loc._active_source_id, loc.database, loc.calibration = "2", "DB_2", "CAL_2"
    loc._prepare_and_extract = lambda *args, **kwargs: ("view", "mask", "crop", {"kp": 1})
    loc._geometric_verifier = SimpleNamespace(
        verify=lambda feats, cands, db, tag=None: "VER" if db == verified_on and cands else None
    )
    return loc


def rotation(main_candidates):
    main = RotationResult(
        angle=90,
        score=main_candidates[0][1],
        candidates=main_candidates,
        source_id="main",
        best_scale=0.5,
        frame="main_view",
        crop_info=None,
    )
    return RotationResult(
        angle=0, score=0.9, candidates=[], source_id="2", per_source={"main": main}
    )


def test_main_layer_goes_before_another_layers_patchify_candidates():
    rot = rotation([(3, 0.85)])
    plan = localizer(3)._main_before_patchify(rot, [(7, 0.7)], [(50, 0.9), (7, 0.7)], "2")

    main_rot, own, patch_only = plan
    assert main_rot.source_id == "main"
    assert own == [(7, 0.7)]
    assert patch_only == [(50, 0.9)]


def test_rule_off_or_main_far_down_keeps_the_old_order():
    far = [(50, 0.99)] + [(frame, 0.98 - 0.01 * frame) for frame in range(8)]
    rot = rotation([(3, 0.10)])

    assert localizer(0)._main_before_patchify(rot, [(7, 0.7)], [(50, 0.9), (7, 0.7)], "2") is None
    assert localizer(3)._main_before_patchify(rot, [(f, 0.5) for f in range(8)], far, "2") is None


def test_on_the_main_layer_its_standard_candidates_go_first():
    rot = rotation([(3, 0.85)])
    plan = localizer(3)._main_before_patchify(rot, [(3, 0.85)], [(50, 0.9), (3, 0.85)], "main")

    assert plan == (None, [(3, 0.85)], [(50, 0.9)])


def test_verified_main_layer_becomes_the_active_layer():
    loc = localizer(3, verified_on="DB_MAIN")
    main_rot = rotation([(3, 0.85)]).per_source["main"]

    result = loc._verify_main_layer(main_rot, "frame", None, {})

    assert result[0] == "VER"
    assert result[-1] == "main"
    assert (loc._active_source_id, loc.database, loc.calibration) == ("main", "DB_MAIN", "CAL_main")


def test_failed_main_layer_leaves_the_chosen_layer_active():
    loc = localizer(3, verified_on=None)
    main_rot = rotation([(3, 0.85)]).per_source["main"]

    assert loc._verify_main_layer(main_rot, "frame", None, {}) is None
    assert (loc._active_source_id, loc.database) == ("2", "DB_2")
