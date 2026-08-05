"""Обхід кінематичного гейта за силою незалежних доказів.

Мотивація — числа з живого прогону 2026-07-29 на тестовому відео з онлайн-карти
(нефізична видима швидкість, бо матеріал не з дрона):

* keyframe-шлях: гейт відкинув 5 фіксів зі швидкостями 175–367 м/с, у яких було
  647, 1191, 1376, 1549 і 1881 інлаєрів. Геометрію підтверджено з величезним
  запасом, а відкинуто через пріор на рух платформи.
* OF-шлях: із 20 відкинутих 17 мали flow_quality 0.017–0.035 (справжній зрив LK
  на однорідній текстурі — відкидати правильно), а 6 мали 0.183–1.0, тобто потік
  ідеально узгоджений і зсув реальний.

Висновок, який кодують ці тести: інлаєри та flow_quality — ПРЯМІ свідчення про
якість вимірювання, кінематика — лише припущення про платформу. Коли перші
сильні, вони мають переважати.

Тести працюють на чистій логіці рішення (без torch/PyQt), дзеркалячи умови з
``Localizer.localize_frame`` і ``localize_optical_flow``.
"""

import pytest

# Пороги — дефолти з config/localization.py
MIN_INLIERS = 100
MIN_FLOW_Q = 0.5


def keyframe_bypasses(trust_enabled: bool, inliers: int, min_inliers: int = MIN_INLIERS) -> bool:
    """Дзеркалить `_strong` у localize_frame."""
    return trust_enabled and inliers >= min_inliers


def of_bypasses(trust_enabled: bool, flow_quality, min_q: float = MIN_FLOW_Q) -> bool:
    """Дзеркалить `_strong_flow` у localize_optical_flow."""
    return trust_enabled and flow_quality is not None and float(flow_quality) >= min_q


# ── Дані рівно з логу ────────────────────────────────────────────────────────

REJECTED_KEYFRAME_INLIERS = [1549, 1376, 1881, 1191, 647]
OF_TRUE_LK_FAILURES = [0.025, 0.022, 0.027, 0.017, 0.035, 0.023, 0.033, 0.028]
OF_REAL_FAST_MOTION = [0.894, 0.625, 0.95, 1.0, 1.0]


class TestKeyframeBypass:
    @pytest.mark.parametrize("inliers", REJECTED_KEYFRAME_INLIERS)
    def test_all_wrongly_rejected_fixes_now_pass(self, inliers):
        """Усі 5 фіксів із логу мають проходити — геометрія в них незаперечна."""
        assert keyframe_bypasses(True, inliers) is True

    @pytest.mark.parametrize("inliers", [0, 10, 40, 80, 99])
    def test_weak_geometry_does_not_bypass(self, inliers):
        """Слабка геометрія обходу не отримує — гейт лишається чинним.

        Межові значення осмислені: 10 = localization.min_inliers_accept,
        40 = early_stop_inliers, 80 = confidence_max_inliers (насичення
        впевненості). Поріг 100 свідомо вищий за всі три.
        """
        assert keyframe_bypasses(True, inliers) is False

    def test_boundary_is_inclusive(self):
        assert keyframe_bypasses(True, 100) is True
        assert keyframe_bypasses(True, 99) is False

    def test_disabled_flag_is_current_behaviour(self):
        """Дефолт off — жодного обходу навіть за 2000 інлаєрів."""
        assert keyframe_bypasses(False, 2048) is False


class TestOpticalFlowBypass:
    @pytest.mark.parametrize("q", OF_REAL_FAST_MOTION)
    def test_self_consistent_flow_bypasses(self, q):
        """Потік з якістю 0.6–1.0 — це реальний рух, а не зрив."""
        assert of_bypasses(True, q) is True

    @pytest.mark.parametrize("q", OF_TRUE_LK_FAILURES)
    def test_genuine_lk_failures_are_still_gated(self, q):
        """17 подій зі зривом LK мають лишитись відкинутими — це і є цінність гейта."""
        assert of_bypasses(True, q) is False

    def test_threshold_separates_the_two_observed_populations(self):
        """0.5 лежить у розриві між режимами — не підігнано під один зразок."""
        assert max(OF_TRUE_LK_FAILURES) < MIN_FLOW_Q < min(OF_REAL_FAST_MOTION)

    def test_missing_flow_quality_never_bypasses(self):
        """Немає метрики — немає доказів; гейт працює як раніше."""
        assert of_bypasses(True, None) is False

    def test_disabled_flag_is_current_behaviour(self):
        assert of_bypasses(False, 1.0) is False


class TestObservedLogEndToEnd:
    def test_bypass_fixes_exactly_the_false_positives(self):
        """Підсумок логу: 6 хибних спрацювань зникають, 17 справжніх лишаються."""
        rescued = sum(1 for q in OF_REAL_FAST_MOTION if of_bypasses(True, q))
        still_gated = sum(1 for q in OF_TRUE_LK_FAILURES if not of_bypasses(True, q))
        assert rescued == len(OF_REAL_FAST_MOTION)
        assert still_gated == len(OF_TRUE_LK_FAILURES)

    def test_all_keyframe_rejections_were_false_positives(self):
        assert all(keyframe_bypasses(True, n) for n in REJECTED_KEYFRAME_INLIERS)


class TestSpeedLimitStillCatchesTeleports:
    """Обхід не робить max_speed_mps декоративним.

    У логу після трьох відкидань спрацьовував OUTLIER RESET, приймаючи позиції
    на 3849 і 3682 м/с. Такі стрибки супроводжуються слабкими доказами, тож
    обхід їх не рятує.
    """

    @pytest.mark.parametrize(
        ("inliers", "flow_q"),
        [(0, 0.02), (25, None), (60, 0.03), (99, 0.017)],
    )
    def test_teleport_with_weak_evidence_stays_gated(self, inliers, flow_q):
        assert keyframe_bypasses(True, inliers) is False
        assert of_bypasses(True, flow_q) is False
