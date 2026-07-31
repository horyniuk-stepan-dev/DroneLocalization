from collections import deque

import numpy as np

from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


class OutlierDetector:
    """Detect anomalous measurements (outliers) based on trajectory history using speeds"""

    def __init__(
        self,
        window_size=10,
        threshold_std=3.0,
        max_speed_mps=1000.0,
        max_consecutive=5,
        ground_scale=1.0,
        zscore_enabled=True,
    ):
        self.window = deque(maxlen=window_size)
        self.threshold_std = threshold_std
        self.max_speed_mps = max_speed_mps
        self._consecutive_outliers = 0
        self._max_consecutive = max_consecutive
        # Множник «проєкційні метри → справжні наземні» (Етап 6). 1.0 = стара
        # поведінка побітово. Без нього WebMercator-відстані на 48° завищені у
        # 1/cos(lat) ≈ 1.49×, тож max_speed_mps=120 реально гейтить на 80.6 м/с,
        # а ~39% відсіювань на живому прогоні були чистими хибними спрацюваннями.
        self._ground_scale = float(ground_scale)
        # Z-score гілка. Заміряно на місії top 2026-07-31 (of_stride=1,
        # of_local_speed=ON): з 70 спрацювань 47 дала вона, і ВСІ 23
        # спрацювання на ПРАВИЛЬНИХ вимірах (distance 2.3-4.5 м при нормі
        # 3.0 м) — теж її. Жодного справжнього зриву (>12 м) вона не спіймала:
        # їх усі ловить фізичний max_speed. Причина — самопідтримна петля:
        # гейт відкидає все, що вище mean_speed, тож у вікно потрапляють лише
        # повільні виміри, mean падає (11.9-78 при реальних 89.2 м/с) і
        # відкидається ще більше. Плюс std сідає на floor 1.0 і z сягає 159.
        self._zscore_enabled = bool(zscore_enabled)

        logger.info("Initializing OutlierDetector (Speed-based Z-score)")
        logger.info(
            f"Parameters: window_size={window_size}, threshold_std={threshold_std}, max_speed_mps={max_speed_mps}"
        )

    def set_ground_scale(self, scale: float) -> None:
        """Оновлює множник проєкція→наземні метри (cos(lat) для WebMercator).

        Викликається з локалізатора, коли є свіжа широта. Значення <= 0
        ігнорується — краще лишити попереднє, ніж занулити всі швидкості.
        """
        s = float(scale)
        if s > 0.0:
            self._ground_scale = s

    def reset(self) -> None:
        """Повне скидання стану (нова сесія трекінгу)."""
        self.window.clear()
        self._consecutive_outliers = 0

    def add_position(self, position: tuple, dt: float = 1.0, reset_consecutive: bool = True):
        # Тепер зберігаємо і позицію, і dt (час, за який ця позиція була досягнута)
        self.window.append((np.array(position, dtype=np.float64), max(dt, 0.01)))
        if reset_consecutive:
            self._consecutive_outliers = 0

    def is_outlier(
        self, new_position: tuple, dt: float = 1.0, ref_position: tuple | None = None
    ) -> bool:
        """Перевірка вимірювання на аномальність.

        ``ref_position`` — опорна точка для МИТТЄВОЇ швидкості. За замовчуванням
        береться остання ПРИЙНЯТА позиція з вікна, але на OF-шляху це неправильно:
        LK трекає завжди від keyframe (tracking_worker.py:486), тож зсув росте
        лінійно від keyframe, тоді як dt лишається кроком одного OF-кадру. Бази
        чисельника і знаменника різні -> швидкість завищується рівно в N разів,
        де N — номер OF-кадру після keyframe. Заміряно на місії top 2026-07-31:
        логовані 450-1271 м/с лягають на сітку N * 89.2 м/с при N=5..14, а між
        keyframe-ами якраз вміщається 10 OF-кадрів.

        Передача попередньої СИРОЇ OF-позиції як ref_position робить розрахунок
        локальним (кадр відносно кадру) і прибирає накопичення.
        """
        if len(self.window) < 3:
            return False

        new_pos_np = np.array(new_position, dtype=np.float64)
        if ref_position is not None:
            last_pos = np.array(ref_position, dtype=np.float64)
        else:
            last_pos, _ = self.window[-1]
        safe_dt = max(dt, 0.01)

        # 1. Перевірка максимально допустимої швидкості
        # Відстані переводимо в СПРАВЖНІ наземні метри до порівняння з порогом,
        # інакше поріг мовчки залежить від широти місії.
        distance = float(np.linalg.norm(new_pos_np - last_pos)) * self._ground_scale
        instantaneous_speed = distance / safe_dt

        is_speed_outlier = instantaneous_speed > self.max_speed_mps

        # 2. Статистичний Z-score тест (тепер за ШВИДКІСТЮ, а не за відстанню!)
        history = list(self.window)
        speeds = []
        for i in range(1, len(history)):
            p1, _ = history[i - 1]
            p2, hist_dt = history[i]
            dist = float(np.linalg.norm(p2 - p1)) * self._ground_scale
            speeds.append(dist / hist_dt)

        mean_speed = np.mean(speeds)
        std_speed = max(np.std(speeds), 1.0)

        z_score = abs(instantaneous_speed - mean_speed) / std_speed

        # 15.0 m/s - мінімальна дельта швидкості, при якій Z-score має сенс
        is_zscore_outlier = self._zscore_enabled and (
            z_score > self.threshold_std and abs(instantaneous_speed - mean_speed) > 15.0
        )

        if is_speed_outlier or is_zscore_outlier:
            self._consecutive_outliers += 1

            # Якщо забагато підряд — дрон реально перемістився, скидаємо вікно
            if self._consecutive_outliers >= self._max_consecutive:
                logger.warning(
                    f"OUTLIER RESET: {self._consecutive_outliers} consecutive outliers — "
                    f"accepting new position. "
                    f"Position: ({new_pos_np[0]:.1f}, {new_pos_np[1]:.1f}), "
                    f"speed={instantaneous_speed:.1f}m/s"
                )
                self.window.clear()
                self._consecutive_outliers = 0
                return False  # Приймаємо нову позицію

            if is_speed_outlier:
                logger.warning(
                    f"OUTLIER DETECTED (speed): {instantaneous_speed:.1f} m/s > {self.max_speed_mps} m/s | "
                    f"distance={distance:.1f}m, dt={safe_dt:.3f}s, "
                    f"consecutive={self._consecutive_outliers}/{self._max_consecutive}"
                )
            else:
                logger.warning(
                    f"OUTLIER DETECTED (z-score): z={z_score:.2f} > {self.threshold_std} | "
                    f"speed={instantaneous_speed:.1f}m/s, mean_speed={mean_speed:.1f}m/s, std={std_speed:.1f}m/s, "
                    f"distance={distance:.1f}m, dt={safe_dt:.3f}s, "
                    f"consecutive={self._consecutive_outliers}/{self._max_consecutive}"
                )
            return True

        self._consecutive_outliers = 0
        return False
