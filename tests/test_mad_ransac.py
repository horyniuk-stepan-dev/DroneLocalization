"""Регресія на MAD-RANSAC (аудит §1.1).

Історія бага. `estimate_homography` після RANSAC ПЕРЕРАХОВУВАЛА inlier-маску
порогом, отриманим із розподілу помилок по ВСІХ відповідностях:

    threshold = median(errors_усі) + k * 1.4826 * MAD(errors_усі)
    mask = errors < threshold          # ← маска могла РОЗШИРИТИСЬ

Коли аутлаєрів більше половини, median і MAD злітають, поріг стає сотнями
пікселів, і «інлаєрами» оголошується весь набір. Роздутий лічильник далі керує
вибором кандидата (`inliers > best_inliers`), спрацюванням `early_stop_inliers`
і `confidence` → шумом Калмана. Тобто хибний кадр БД міг обійти правильний.

Другий бік бага: під `backend="poselib"` (значення в user_config.json) функція
поверталась ДО MAD-блоку, тож `use_mad_ransac: true` не мав жодного ефекту —
конфіг казав ON, код робив OFF.

Тести нижче фіксують обидві властивості виправлення.
"""

import numpy as np
import pytest

from src.geometry.transformations import GeometryTransforms as G


def _split_matches(n_inliers: int, n_outliers: int, seed: int = 0):
    """Відповідності під H = I: `n_inliers` з ~1px шумом + грубі аутлаєри."""
    rng = np.random.default_rng(seed)
    src_in = rng.uniform(0.0, 1000.0, (n_inliers, 2))
    dst_in = src_in + rng.normal(0.0, 1.0, (n_inliers, 2))
    src_out = rng.uniform(0.0, 1000.0, (n_outliers, 2))
    dst_out = src_out + rng.uniform(50.0, 500.0, (n_outliers, 2))
    src = np.vstack([src_in, src_out])
    dst = np.vstack([dst_in, dst_out])
    ransac_mask = np.zeros((n_inliers + n_outliers, 1), dtype=np.uint8)
    ransac_mask[:n_inliers] = 1  # RANSAC знайшов саме справжні інлаєри
    return src, dst, ransac_mask


class TestMadThreshold:
    def test_threshold_over_inliers_only_is_tight(self):
        """Поріг по інлаєрах — одиниці пікселів; по всьому набору — сотні."""
        src, dst, mask = _split_matches(20, 40)
        H = np.eye(3)

        thr_all = G.compute_mad_threshold(src, dst, H, k=2.5)
        thr_in = G.compute_mad_threshold(
            src, dst, H, k=2.5, inlier_mask=mask.ravel().astype(bool)
        )

        assert thr_in < 10.0, f"поріг по інлаєрах має бути тісним, отримано {thr_in}"
        assert thr_all > 100.0, "контроль: по всьому набору поріг таки роздувається"

    def test_empty_mask_falls_back_to_all_points(self):
        """Порожня маска не має ділити на нуль — тихо беремо всі точки."""
        src, dst, _ = _split_matches(10, 10)
        empty = np.zeros(20, dtype=bool)
        thr = G.compute_mad_threshold(src, dst, np.eye(3), inlier_mask=empty)
        assert np.isfinite(thr) and thr > 0.0


class TestRefineMaskNeverGrows:
    @pytest.mark.parametrize(
        ("n_in", "n_out"),
        [(20, 40), (10, 50), (30, 30), (45, 15), (5, 55)],
    )
    def test_mask_never_grows(self, n_in, n_out):
        """Головна властивість: MAD не має права ДОДАВАТИ інлаєрів.

        Саме це давало 60 «інлаєрів» із 20 справжніх у старій реалізації.
        """
        src, dst, mask = _split_matches(n_in, n_out)
        refined = G._refine_mask_mad(src, dst, np.eye(3), mask, 2.5, 3.0)
        assert int(refined.sum()) <= int(mask.sum())

    def test_outlier_majority_does_not_inflate(self):
        """Регресія 1-в-1: 20 справжніх із 60 матчів не мають стати 60."""
        src, dst, mask = _split_matches(20, 40)
        refined = G._refine_mask_mad(src, dst, np.eye(3), mask, 2.5, 3.0)
        n = int(refined.sum())
        assert n <= 20, f"маска роздулась до {n} — баг §1.1 повернувся"
        # І жоден зі справжніх аутлаєрів не потрапив усередину
        assert not refined.ravel().astype(bool)[20:].any()

    def test_clean_set_is_preserved(self):
        """На чистому наборі уточнення не має викошувати справжні інлаєри."""
        src, dst, mask = _split_matches(45, 15)
        refined = G._refine_mask_mad(src, dst, np.eye(3), mask, 2.5, 3.0)
        assert int(refined.sum()) >= 40, "MAD не має різати здорові матчі"

    def test_never_drops_below_minimum_for_homography(self):
        """Агресивний k не має лишити менше 4 точок — гомографія стане виродженою."""
        src, dst, _ = _split_matches(30, 30)
        mask = np.zeros((60, 1), dtype=np.uint8)
        mask[:5] = 1
        refined = G._refine_mask_mad(src, dst, np.eye(3), mask, 0.0, 3.0)
        assert int(refined.sum()) >= G._MIN_HOMOGRAPHY_PTS

    def test_none_mask_is_passthrough(self):
        src, dst, _ = _split_matches(10, 10)
        assert G._refine_mask_mad(src, dst, np.eye(3), None, 2.5, 3.0) is None


class TestEstimateHomographyIntegration:
    def test_mad_does_not_inflate_end_to_end(self):
        """Через публічний API: use_mad_ransac не має завищувати лічильник."""
        src, dst, _ = _split_matches(40, 20, seed=7)

        _H_off, mask_off = G.estimate_homography(src, dst, use_mad_ransac=False)
        _H_on, mask_on = G.estimate_homography(src, dst, use_mad_ransac=True)

        assert mask_off is not None and mask_on is not None
        n_off, n_on = int(mask_off.sum()), int(mask_on.sum())
        assert n_on <= n_off, (
            f"MAD-уточнення додало інлаєрів ({n_off} → {n_on}) — воно має лише звужувати"
        )
        # І не має викосити все — інакше кадр даремно піде у фолбек
        assert n_on >= 4
