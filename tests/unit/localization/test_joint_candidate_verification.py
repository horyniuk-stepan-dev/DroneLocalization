"""Localizer: joint candidate list of all layers (localization.joint_candidate_ranking)."""

from types import SimpleNamespace

from src.localization.localizer import Localizer
from src.localization.rotation_selector import RotationResult


class Database:
    def __init__(self, name, unsupported=()):
        self.name = name
        self.unsupported = set(unsupported)

    def is_frame_georef_supported(self, frame):
        return frame not in self.unsupported


class Manager:
    def __init__(self, databases, priorities, patch_priorities=None):
        self.databases = databases
        self.sources = {
            sid: SimpleNamespace(priority=p, patchify_priority=(patch_priorities or {}).get(sid))
            for sid, p in priorities.items()
        }

    @property
    def active_source_ids(self):
        return set(self.databases)

    def get_source_config(self, sid):
        return self.sources[sid]

    def get_database(self, sid):
        return self.databases[sid]


def localizer(patches, verified=(), step=2, penalty=2, top_k=5, unsupported=()):
    calls = []

    def verify(feats, cands, db, tag=None):
        calls.append((db.name, [int(c) for c, _ in cands], tag, feats["at"]))
        for frame, _ in cands:
            if (db.name, int(frame)) in verified:
                return SimpleNamespace(candidate_id=int(frame))
        return None

    loc = Localizer.__new__(Localizer)
    loc._joint_rank_step, loc._joint_patch_penalty, loc._joint_top_k = step, penalty, top_k
    loc.db_manager = Manager(
        {"main": Database("main", unsupported), "2": Database("2")}, {"main": 0, "2": 1}
    )
    loc.calib_manager = SimpleNamespace(get=lambda sid: f"CAL_{sid}")
    loc._active_source_id, loc.database, loc.calibration = "2", "DB_2", "CAL_2"
    loc._prepare_and_extract = lambda q, m, angle, scale, cache, prepared=None: (
        f"view{angle}",
        "mask",
        "crop",
        {"at": (angle, scale)},
    )
    loc._geometric_verifier = SimpleNamespace(verify=verify)
    loc._candidate_retriever = SimpleNamespace(
        patch_candidates=lambda frame, top_k, source_id=None, desc_cache=None: patches.get(
            source_id, []
        )
    )
    return loc, calls


def rotation(main, other):
    def layer(sid, candidates, angle, scale):
        return RotationResult(
            angle=angle,
            score=candidates[0][1],
            candidates=candidates,
            source_id=sid,
            best_scale=scale,
            frame=f"{sid}_view",
        )

    per_source = {"main": layer("main", main, 90, 0.5), "2": layer("2", other, 0, 1.0)}
    return RotationResult(
        angle=0, score=other[0][1], candidates=other, source_id="2", per_source=per_source
    )


def test_verification_order_follows_the_joint_list():
    # standard: 2/1 (0.90) 1st, main/7 (0.88) 2nd -> main first (within 2 places);
    # patch main/50 is 1st of the patches -> place 1 + 2 = 3
    loc, calls = localizer({"main": [(50, 0.4)]})
    rot = rotation([(7, 0.88)], [(1, 0.90), (2, 0.70)])

    assert loc._verify_joint(rot, "frame", None, {}, 12) is None
    assert [(db, frames) for db, frames, _tag, _at in calls] == [
        ("main", [7]),
        ("2", [1]),
        ("main", [50]),
        ("2", [2]),
    ]
    assert calls[2][2] == "joint_patchify"
    assert calls[0][3] == (90, 0.5) and calls[1][3] == (0, 1.0)


def test_verified_layer_becomes_active_with_its_own_angle_and_scale():
    loc, _calls = localizer({}, verified={("main", 7)})
    rot = rotation([(7, 0.88)], [(1, 0.90)])

    result = loc._verify_joint(rot, "frame", None, {}, 12)

    assert result[0].candidate_id == 7
    assert (result[1], result[2], result[-1]) == (90, 0.5, "main")
    assert (loc._active_source_id, loc.calibration) == ("main", "CAL_main")


def test_nothing_verified_leaves_the_chosen_layer_active():
    loc, _calls = localizer({})

    assert loc._verify_joint(rotation([(7, 0.88)], [(1, 0.9)]), "frame", None, {}, 12) is None
    assert (loc._active_source_id, loc.database) == ("2", "DB_2")


def test_patch_hits_of_standard_frames_and_unsupported_frames_are_dropped():
    loc, calls = localizer({"main": [(7, 2.0), (8, 1.5), (9, 1.0)]}, unsupported={8})
    rot = rotation([(7, 0.88)], [(1, 0.90)])

    loc._verify_joint(rot, "frame", None, {}, 12)

    verified = [(db, f) for db, frames, _tag, _at in calls for f in frames]
    assert verified.count(("main", 7)) == 1
    assert ("main", 8) not in verified
    assert ("main", 9) in verified


def test_only_the_first_top_k_are_verified():
    loc, calls = localizer({}, top_k=2)
    rot = rotation([(7, 0.88), (8, 0.87), (9, 0.86)], [(1, 0.90)])

    loc._verify_joint(rot, "frame", None, {}, 12)

    assert sum(len(frames) for _db, frames, _tag, _at in calls) == 2
