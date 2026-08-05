import os
import sys

sys.path.append(os.getcwd())

from src.geometry.coordinates import CoordinateConverter


def test_projection_difference():
    """Перевірка коректності WebMercator та UTM проєкцій."""
    test_points = [
        (48.45, 35.05),  # Дніпро
        (50.45, 30.52),  # Київ
        (46.48, 30.72),  # Одеса
    ]

    print("Testing Projection Difference: WebMercator vs UTM")
    print("-" * 60)

    for lat, lon in test_points:
        print(f"\nPoint: GPS({lat}, {lon})")

        # 1. WebMercator (екземпляр)
        converter_wm = CoordinateConverter("WEB_MERCATOR")
        mx_w, my_w = converter_wm.gps_to_metric(lat, lon)
        print(f"WEB_MERCATOR: ({mx_w:.2f}, {my_w:.2f})")

        # 2. UTM (екземпляр)
        converter_utm = CoordinateConverter("UTM", (lat, lon))
        mx_u, my_u = converter_utm.gps_to_metric(lat, lon)
        print(f"UTM (ref=this): ({mx_u:.2f}, {my_u:.2f})")

        # Тест репродуктивності (UTM з іншим референсом)
        ref_gps = (lat + 0.1, lon + 0.1)
        converter_utm2 = CoordinateConverter("UTM", ref_gps)
        mx_u2, my_u2 = converter_utm2.gps_to_metric(lat, lon)
        print(f"UTM (ref={ref_gps[0]:.2f}, {ref_gps[1]:.2f}): ({mx_u2:.2f}, {my_u2:.2f})")

        # Порівняємо відстані між точками
        p2_gps = (lat + 0.01, lon + 0.01)

        m2_w = converter_wm.gps_to_metric(*p2_gps)
        dist_w = ((mx_w - m2_w[0]) ** 2 + (my_w - m2_w[1]) ** 2) ** 0.5

        m2_u = converter_utm.gps_to_metric(*p2_gps)
        dist_u = ((mx_u - m2_u[0]) ** 2 + (my_u - m2_u[1]) ** 2) ** 0.5

        real_dist = CoordinateConverter.haversine_distance((lat, lon), p2_gps)

        print("Distance to +0.01,+0.01:")
        print(f"  Real (Haversine): {real_dist:.2f} m")
        print(f"  WebMercator:      {dist_w:.2f} m (Ratio: {dist_w / real_dist:.4f})")
        print(f"  UTM:              {dist_u:.2f} m (Ratio: {dist_u / real_dist:.4f})")

        # UTM повинна давати точніший результат ніж WebMercator
        assert abs(dist_u / real_dist - 1.0) < abs(dist_w / real_dist - 1.0) + 0.01


if __name__ == "__main__":
    test_projection_difference()
