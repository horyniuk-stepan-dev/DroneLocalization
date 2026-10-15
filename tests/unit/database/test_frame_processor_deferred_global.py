"""DB build: global and patchify descriptors are computed for saved frames only."""

import numpy as np

from src.database.frame_processor import FrameProcessor


class FakeExtractor:
    def __init__(self):
        self.local_calls = 0
        self.global_calls = 0

    def extract_local_features(self, rgb, mask=None):
        self.local_calls += 1
        kps = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        return {"keypoints": kps, "descriptors": np.ones((2, 4), dtype=np.float32)}

    def extract_global_descriptor(self, rgb):
        self.global_calls += 1
        return np.full(3, float(rgb[0, 0, 0]), dtype=np.float32)


class FakePatchify:
    def __init__(self):
        self.calls = 0

    def compute_patch_descriptors(self, rgb):
        self.calls += 1
        return np.full((14, 3), float(rgb[0, 0, 0]), dtype=np.float32)


class FakeWriter:
    def __init__(self):
        self.saved = {}
        self.poses = []

    def write_pose(self, idx, pose):
        self.poses.append(idx)

    def save_frame_data(self, idx, features, pose):
        self.saved[idx] = features


def test_skipped_frames_pay_no_global_or_patchify_pass():
    extractor, patchify, writer = FakeExtractor(), FakePatchify(), FakeWriter()
    keep = {0, 2, 4}  # frame 0 is always saved; motion gate keeps 2 and 4
    state = {"idx": 0}
    processor = FrameProcessor(
        feature_extractor=extractor,
        db_writer=writer,
        compute_inter_frame_h=lambda prev, cur: np.eye(3),
        is_significant_motion=lambda H, w, h: state["idx"] in keep,
        draw_keypoints=lambda *a: None,
        config={},
        width=8,
        height=8,
        num_frames=6,
        patchify=patchify,
        use_keyframe_selection=True,
        keyframe_criterion="step",
    )
    for idx in range(6):
        state["idx"] = idx
        frame = np.full((8, 8, 3), idx, dtype=np.uint8)
        processor.process(idx, frame, frame, None)

    assert sorted(writer.saved) == [0, 2, 4]
    assert writer.poses == list(range(6))  # pose chain still covers every frame
    assert extractor.local_calls == 6
    assert extractor.global_calls == 3
    assert patchify.calls == 3
    for idx, features in writer.saved.items():
        assert features["global_desc"][0] == idx  # descriptor of that very frame
        assert features["patch_descriptors"].shape == (14, 3)
        assert features["patch_descriptors"][0, 0] == idx


def test_without_patchify_only_global_is_added():
    extractor, writer = FakeExtractor(), FakeWriter()
    processor = FrameProcessor(
        feature_extractor=extractor,
        db_writer=writer,
        compute_inter_frame_h=lambda prev, cur: np.eye(3),
        is_significant_motion=lambda H, w, h: True,
        draw_keypoints=lambda *a: None,
        config={},
        width=8,
        height=8,
        num_frames=2,
        use_keyframe_selection=True,
        keyframe_criterion="step",
    )
    for idx in range(2):
        frame = np.full((8, 8, 3), idx, dtype=np.uint8)
        processor.process(idx, frame, frame, None)

    assert extractor.global_calls == 2
    assert all("patch_descriptors" not in f for f in writer.saved.values())
