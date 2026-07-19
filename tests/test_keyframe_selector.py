"""Тести keyframe_selector — вибір keyframe-ів витягнуто з DatabaseBuilder.

is_significant_motion — чиста numpy-логіка; compute_inter_frame_homography
тестується з фейковим матчером (матчер інжектується). Обидва йдуть у пісочниці
(cv2/numpy), фіксуючи семантику «keyframe вибірково» перед розбиттям білдера.
"""

import numpy as np

from src.database import keyframe_selector as ks

W, H = 1920, 1080
CX, CY = W / 2.0, H / 2.0


def _translation_H(tx, ty):
    return np.array([[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]])


def _rotation_about_center_H(angle_deg):
    th = np.radians(angle_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    t = np.array([CX, CY]) - R @ np.array([CX, CY])
    return np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1]])


class TestIsSignificantMotion:
    def test_identity_is_not_significant(self):
        assert not ks.is_significant_motion(np.eye(3), W, H)

    def test_large_translation_is_significant(self):
        assert ks.is_significant_motion(_translation_H(20.0, 0.0), W, H)

    def test_small_translation_below_threshold(self):
        # 5 px < 15 px, без обертання → не keyframe
        assert not ks.is_significant_motion(_translation_H(5.0, 0.0), W, H)

    def test_translation_threshold_boundary(self):
        # рівно на порозі (>=) → significant
        assert ks.is_significant_motion(_translation_H(15.0, 0.0), W, H)

    def test_rotation_about_center_is_significant(self):
        # обертання навколо центру: центр не рухається, спрацьовує кутова гілка
        assert ks.is_significant_motion(_rotation_about_center_H(5.0), W, H)

    def test_small_rotation_below_threshold(self):
        assert not ks.is_significant_motion(_rotation_about_center_H(1.0), W, H)

    def test_degenerate_matrix_is_significant(self):
        A = np.array([[1e-4, 0.0], [0.0, 1e-4]])  # |det| = 1e-8 < 1e-6
        t = np.array([CX, CY]) - A @ np.array([CX, CY])  # центр лишається на місці
        Hd = np.array([[A[0, 0], A[0, 1], t[0]], [A[1, 0], A[1, 1], t[1]], [0, 0, 1]])
        assert ks.is_significant_motion(Hd, W, H)

    def test_returns_plain_bool(self):
        # контракт -> bool (не np.bool_), щоб `is`/серіалізація не дивували
        assert isinstance(ks.is_significant_motion(np.eye(3), W, H), bool)

    def test_custom_thresholds_respected(self):
        Ht = _translation_H(10.0, 0.0)
        assert ks.is_significant_motion(Ht, W, H, min_translation_px=8.0)
        assert not ks.is_significant_motion(Ht, W, H, min_translation_px=12.0)


class _FakeMatcher:
    """matcher.match(fa, fb) -> (mkpts_a, mkpts_b) — інжектується у функцію."""

    def __init__(self, a, b):
        self._a, self._b = a, b

    def match(self, fa, fb):
        return self._a, self._b


def _grid(n_x=6, n_y=5):
    xs = np.linspace(200, 1700, n_x)
    ys = np.linspace(150, 900, n_y)
    return np.array([[x, y] for y in ys for x in xs], dtype=np.float32)


class TestComputeInterFrameHomography:
    def test_recovers_homography_from_matches(self):
        a = _grid()  # 30 точок
        b = (a + np.array([10.0, 5.0], dtype=np.float32)).astype(np.float32)
        Hm = ks.compute_inter_frame_homography(_FakeMatcher(a, b), {}, {})
        assert Hm is not None
        assert Hm.shape == (3, 3)
        assert Hm.dtype == np.float64

    def test_too_few_matches_returns_none(self):
        a = _grid(3, 3)[:10]  # 10 < 15
        b = a.copy()
        assert ks.compute_inter_frame_homography(_FakeMatcher(a, b), {}, {}) is None

    def test_min_matches_param_gates(self):
        a = _grid()
        b = (a + np.array([10.0, 0.0], dtype=np.float32)).astype(np.float32)
        # 30 матчів, але поріг 999 → None
        assert (
            ks.compute_inter_frame_homography(_FakeMatcher(a, b), {}, {}, min_matches=999) is None
        )
