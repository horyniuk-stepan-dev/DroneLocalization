"""Source sampling of scripts/build_vlad_vocab.py (no torch / DINO needed)."""

import cv2
import numpy as np
import pytest

import scripts.build_vlad_vocab as bvv


def test_split_budget_and_even_indices():
    assert bvv.split_budget(10, 3) == [4, 3, 3]
    assert bvv.split_budget(2, 3) == [1, 1, 0]
    assert bvv.even_indices(100, 4) == [0, 25, 50, 75]
    assert bvv.even_indices(3, 10) == [0, 1, 2]
    assert bvv.even_indices(0, 5) == []
    assert bvv.even_indices(7, 0) == []


def test_list_images_is_recursive_sorted_and_filtered(tmp_path):
    (tmp_path / "z17").mkdir()
    image = np.zeros((8, 8, 3), np.uint8)
    for name in ("z17/b.png", "a.JPG", "z17/c.tif"):
        assert cv2.imwrite(str(tmp_path / name), image)
    (tmp_path / "notes.txt").write_text("x")
    files = bvv.list_images(tmp_path)
    assert [f.relative_to(tmp_path).as_posix() for f in files] == [
        "a.JPG",
        "z17/b.png",
        "z17/c.tif",
    ]
    with pytest.raises(ValueError):
        bvv.list_images(tmp_path / "absent")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        bvv.list_images(tmp_path / "empty")


def test_image_source_yields_rgb_evenly(tmp_path):
    for i in range(6):
        bgr = np.zeros((4, 4, 3), np.uint8)
        bgr[..., 0] = 10 * i  # blue channel carries the file index
        cv2.imwrite(str(tmp_path / f"{i:02d}.png"), bgr)
    frames = list(bvv.iter_image_files(tmp_path, 3))
    assert [int(f[0, 0, 2]) for f in frames] == [0, 20, 40]  # RGB: blue is last
    assert all(int(f[0, 0, 0]) == 0 for f in frames)


def test_video_source_samples_the_whole_video(tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (16, 16))
    if not writer.isOpened():
        pytest.skip("no MJPG writer in this OpenCV build")
    for i in range(20):
        writer.write(np.full((16, 16, 3), 10 * i, np.uint8))
    writer.release()
    frames = list(bvv.iter_video_frames(path, 4))
    assert len(frames) == 4
    levels = [int(f.mean()) for f in frames]
    assert levels == sorted(levels) and levels[-1] >= 140  # reaches the last quarter
    with pytest.raises(ValueError):
        next(bvv.iter_video_frames(tmp_path / "absent.avi", 2))
