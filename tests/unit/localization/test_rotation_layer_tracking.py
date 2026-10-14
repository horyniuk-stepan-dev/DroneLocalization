"""RotationSelector keeps every layer's best (angle, scale) for the layer priority rules."""

from src.localization.rotation_selector import RotationSelector


class FakeExtractor:
    def extract_global_descriptor(self, frame, valid_mask=None):
        return frame

    def extract_global_descriptors_multi(self, frames, valid_masks=None):
        return list(frames)


class FakeRetriever:
    """Per rotation: main is best at 90 deg, layer 2 at 0 deg (and overall)."""

    TABLE = {
        0: {"main": [(1, 0.60)], "2": [(5, 0.90)]},
        90: {"main": [(2, 0.80)], "2": [(6, 0.70)]},
        180: {"main": [(3, 0.40)], "2": [(7, 0.30)]},
        270: {"main": [(4, 0.20)], "2": [(8, 0.10)]},
    }

    def __init__(self):
        self.last_by_source = None

    def retrieve(self, descriptor, top_k):
        by_source = self.TABLE[descriptor]
        self.last_by_source = by_source
        sid = max(by_source, key=lambda s: by_source[s][0][1])
        return sid, by_source[sid]


def _select(margin, joint=False):
    config = {
        "localization": {
            "rotation_mode": "quarter",
            "patchify_main_rank_margin": margin,
            "joint_candidate_ranking": joint,
        }
    }
    selector = RotationSelector(FakeExtractor(), FakeRetriever(), config)
    # quarter mode rotates with np.rot90; a fake frame keyed by angle stands in for it
    selector._prepare = lambda frame, angle, sc, manager, cache=None: (angle, None, None)
    return selector.select("frame", None, False, [0, 90, 180, 270], top_k=5)


def test_each_layer_keeps_its_best_rotation():
    rot = _select(margin=2)

    assert (rot.source_id, rot.angle, rot.score) == ("2", 0, 0.90)
    assert rot.per_source["main"].angle == 90
    assert rot.per_source["main"].candidates == [(2, 0.80)]
    assert rot.per_source["2"].angle == 0


def test_no_tracking_while_the_rule_is_off():
    assert _select(margin=0).per_source is None


def test_joint_candidate_list_tracks_the_layers_too():
    rot = _select(margin=0, joint=True)

    assert rot.per_source["main"].angle == 90
