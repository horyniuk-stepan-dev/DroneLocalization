import cv2
import sys
from pathlib import Path

# Додаємо шлях до нашого проекту
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.logging_utils import setup_logging, get_logger

setup_logging()
logger = get_logger("PanoramaBuilder")


def create_panorama_from_video(video_path, output_path, frame_step=30):
    """
    Створює панораму з відео, беручи кожен N-й кадр.
    frame_step=30 означає, що при 30 FPS ми беремо 1 кадр на секунду.
    """
    logger.info(f"Відкриття відео: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Не вдалося відкрити відео.")
        return

    frames_to_stitch = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Беремо лише кожен N-й кадр, щоб не перевантажити пам'ять
        # і забезпечити достатнє зміщення між кадрами
        if frame_count % frame_step == 0:
            # Можна зменшити розмір кадру для прискорення роботи
            # frame = cv2.resize(frame, (1280, 720))
            frames_to_stitch.append(frame)
            logger.info(f"Додано кадр {frame_count} для зшивання")

        frame_count += 1

    cap.release()
    logger.info(f"Всього зібрано {len(frames_to_stitch)} кадрів. Починаємо зшивання...")

    # Створюємо об'єкт Stitcher (режим SCANS краще підходить для дронів/сканерів)
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)

    status, panorama = stitcher.stitch(frames_to_stitch)

    if status == cv2.Stitcher_OK:
        cv2.imwrite(output_path, panorama)
        logger.success(f"Панораму успішно створено та збережено у: {output_path}")
    else:
        errors = {
            cv2.Stitcher_ERR_NEED_MORE_IMGS: "Потрібно більше зображень (або більше перекриття між ними).",
            cv2.Stitcher_ERR_HOMOGRAPHY_EST_FAIL: "Не вдалося розрахувати гомографію (замало спільних точок).",
            cv2.Stitcher_ERR_CAMERA_PARAMS_ADJUST_FAIL: "Не вдалося налаштувати параметри камери."
        }
        err_msg = errors.get(status, f"Невідома помилка (код {status})")
        logger.error(f"Помилка зшивання: {err_msg}")
        logger.warning("Спробуйте змінити параметр frame_step (наприклад, зробити його меншим: 15 або 20).")


if __name__ == "__main__":
    # Вкажи тут шлях до свого відео та куди зберегти результат
    VIDEO_FILE = "C:/Users/horyn/OneDrive/Desktop/Timeline1roadcat.mp4"
    OUTPUT_IMAGE = "C:/Users/horyn/OneDrive/Desktop/panorama_result2.jpg"

    create_panorama_from_video(VIDEO_FILE, OUTPUT_IMAGE, frame_step=20)