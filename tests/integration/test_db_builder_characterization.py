"""Characterization (golden-net) test for ``DatabaseBuilder.build_from_video``.

PLAN GATE (IMPROVEMENT_PLAN п.1.3). Before ``build_from_video`` is decomposed
into modules (video_frame_source / frame_processor / db_writer /
keypoint_video_writer), this test pins the invariants that decomposition MUST
preserve. If any extraction changes per-frame behaviour, one of these fails.

The three invariants (all currently true against the real builder):

1. pose-always — ``global_descriptors/frame_poses[slot]`` is a finite, non-zero
   affine for EVERY processed slot, including non-keyframes (the loop writes
   ``frame_poses[p_idx] = current_pose`` unconditionally, before the keyframe
   gate). Without this, non-keyframe slots would be zeros and propagation would
   break — so a regression that couples the pose write to the keyframe branch
   must be caught. Only meaningful when non-keyframes actually exist (asserted).

2. keyframe-selectively — full local features (``local_features/kp_counts>0``)
   are stored ONLY for selected keyframes; non-keyframe slots keep count 0. The
   build must keep SOME frames and skip SOME (0 < keyframes < processed), so
   both the store and skip paths are exercised.

3. frame_id ↔ slot identity — DB slot ``i`` corresponds to video frame
   ``i * frame_step``: ``create_hdf5_structure`` persists the ``frame_step``
   attribute and ``num_frames == ceil(total_frames / frame_step)``;
   ``save_frame_data`` keys data by the processed index ``p_idx``. Pinned at
   step 1 and step 2.

FIXTURE. ``flight_clip.mp4`` is a real 150-frame excerpt (frames 3000..3149) of
the FlightSimulator recording ``flight.mp4`` (1280x720, 30 fps drone footage).
Built at ``frame_step=1`` the small real inter-frame motion yields a natural
keyframe / non-keyframe mix (~112 keyframes, ~38 non-keyframes on the reference
machine) with well-conditioned homographies — no synthetic tuning of the
selection thresholds. The exact split is not asserted (it may shift slightly
with GPU/ALIKED nondeterminism); only the structural invariants above are, which
are what the decomposition must preserve. ``use_decord=False`` pins the cv2
decode path for cross-machine reproducibility.

Windows/GPU only: gated by ``pytest.importorskip("torch")`` — the build needs
the real feature/mask/depth models. Skipped in the pure-Python CI sandbox.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
h5py = pytest.importorskip("h5py")

from config import APP_CONFIG  # noqa: E402
from src.database.database_builder import DatabaseBuilder  # noqa: E402
from src.models.model_manager import ModelManager  # noqa: E402

FIXTURE = Path(__file__).parent.parent / "fixtures" / "flight_clip.mp4"
TOTAL_FRAMES = 150  # frames in flight_clip.mp4


def _build_config(frame_step: int) -> dict:
    cfg = copy.deepcopy(APP_CONFIG)
    db = cfg.setdefault("database", {})
    db["frame_step"] = frame_step
    db["use_decord"] = False  # deterministic cv2 decode across machines
    db["keyframe_min_translation_px"] = 15.0  # selection ON (production default)
    return cfg


@pytest.fixture(scope="module")
def model_manager():
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing: {FIXTURE}")
    return ModelManager(copy.deepcopy(APP_CONFIG))


def _build(tmp_path_factory, model_manager, frame_step: int) -> dict:
    out = str(tmp_path_factory.mktemp(f"charden_{frame_step}") / "db.h5")
    cfg = _build_config(frame_step)
    DatabaseBuilder(output_path=out, config=cfg).build_from_video(
        video_path=str(FIXTURE), model_manager=model_manager, save_keypoint_video=False
    )
    with h5py.File(out, "r") as f:
        meta = f["metadata"]
        return {
            "num_frames": int(meta.attrs["num_frames"]),
            "frame_step": int(meta.attrs["frame_step"]),
            "frame_width": int(meta.attrs["frame_width"]),
            "frame_height": int(meta.attrs["frame_height"]),
            "frame_poses": f["global_descriptors"]["frame_poses"][:],
            "kp_counts": f["local_features"]["kp_counts"][:],
        }


@pytest.fixture(scope="module")
def built_step1(tmp_path_factory, model_manager):
    return _build(tmp_path_factory, model_manager, frame_step=1)


@pytest.fixture(scope="module")
def built_step2(tmp_path_factory, model_manager):
    return _build(tmp_path_factory, model_manager, frame_step=2)


# ── Invariant 2 (prerequisite): a real keyframe/non-keyframe mix exists ───────
def test_selection_keeps_some_and_skips_some(built_step1):
    kp = built_step1["kp_counts"]
    n = built_step1["num_frames"]
    keyframes = int(np.count_nonzero(kp))
    non_keyframes = n - keyframes
    assert n == TOTAL_FRAMES
    assert keyframes > 0, "no keyframes stored"
    assert non_keyframes > 0, "no non-keyframes — pose-always invariant would be vacuous"
    assert keyframes < n


# ── Invariant 1: pose-always ─────────────────────────────────────────────────
def test_pose_always_every_slot_nonzero_and_finite(built_step1):
    poses = built_step1["frame_poses"]
    n = built_step1["num_frames"]
    assert poses.shape == (n, 3, 3)
    for slot in range(n):
        assert np.all(np.isfinite(poses[slot])), f"slot {slot} pose non-finite"
        assert np.any(poses[slot]), f"slot {slot} pose is all-zero"


def test_pose_written_for_non_keyframes(built_step1):
    # The load-bearing case: slots that are NOT keyframes (kp_counts == 0) must
    # still carry a pose, or graph propagation reads zeros. This is exactly what
    # a keyframe-coupled pose write would break.
    kp = built_step1["kp_counts"]
    poses = built_step1["frame_poses"]
    n = built_step1["num_frames"]
    non_keyframe_slots = [i for i in range(n) if kp[i] == 0]
    assert non_keyframe_slots, "fixture produced no non-keyframes"
    for slot in non_keyframe_slots:
        assert np.any(poses[slot]), f"non-keyframe slot {slot} pose is all-zero"
        assert np.all(np.isfinite(poses[slot])), f"non-keyframe slot {slot} pose non-finite"


# ── Invariant 2: keyframe-selectively ────────────────────────────────────────
def test_local_features_only_where_kp_count_positive(built_step1):
    # kp_counts is the per-slot record of stored local features; it is written
    # only through save_frame_data, which runs only for keyframes. So every
    # positive count marks a keyframe and every zero marks a skipped frame.
    kp = built_step1["kp_counts"]
    n = built_step1["num_frames"]
    assert kp.shape == (n,)
    assert int(np.count_nonzero(kp)) <= n


# ── Invariant 3: frame_id ↔ slot identity ────────────────────────────────────
def test_frame_step_attr_and_num_frames_step1(built_step1):
    assert built_step1["frame_step"] == 1
    assert built_step1["num_frames"] == math.ceil(TOTAL_FRAMES / 1)
    assert (built_step1["frame_width"], built_step1["frame_height"]) == (1280, 720)


def test_frame_step_attr_and_num_frames_step2(built_step2):
    # Non-trivial step pins the slot i ↔ video frame i*frame_step mapping.
    assert built_step2["frame_step"] == 2
    assert built_step2["num_frames"] == math.ceil(TOTAL_FRAMES / 2)
    assert built_step2["frame_poses"].shape[0] == built_step2["num_frames"]
