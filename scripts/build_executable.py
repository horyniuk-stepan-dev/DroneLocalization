import PyInstaller.__main__
import os
from pathlib import Path


def build_app():
    # Визначаємо кореневу директорію проєкту (на одну папку вище за scripts)
    root_dir = Path(__file__).parent.parent.absolute()
    main_script = str(root_dir / "main.py")

    # Створюємо абсолютний шлях до папки з ресурсами
    resources_dir = str(root_dir / "src" / "gui" / "resources")

    print(f"Починаємо збірку проєкту з кореня: {root_dir}")
    print("Це може зайняти кілька хвилин через великий розмір PyTorch...")

    # Переходимо в корінь
    os.chdir(root_dir)

    args = [
        main_script,
        '--name=DroneLocalization',
        '--noconfirm',

        # Режим папки (швидкий запуск, ідеально для AI додатків)
        '--onedir',

        # Вимикаємо консоль (змініть на --console для дебагу)
        '--windowed',

        # Абсолютний шлях до джерела ресурсів
        f'--add-data={resources_dir};src/gui/resources',

        # Примусово вказуємо PyInstaller запакувати ці бібліотеки
        '--hidden-import=PyQt6.QtWebEngineWidgets',
        '--hidden-import=PyQt6.QtWebEngineCore',
        '--hidden-import=torch',
        '--hidden-import=torchvision',
        '--hidden-import=ultralytics',
        '--hidden-import=lightglue',
        '--hidden-import=h5py',
        '--hidden-import=cv2',
        '--hidden-import=filterpy',
        '--hidden-import=pyproj',

        # --- ФІКС ПОМИЛКИ JARACO ---
        '--hidden-import=pkg_resources',
        '--hidden-import=jaraco.text',
        '--hidden-import=jaraco.functools',
        '--hidden-import=jaraco.context',
        '--hidden-import=pkg_resources._vendor.jaraco.text',
        '--hidden-import=pkg_resources._vendor.jaraco.functools',
        '--hidden-import=pkg_resources._vendor.jaraco.context',
        # ---------------------------

        # Очищуємо кеш попередніх збірок
        '--clean',

        # Явно вказуємо, куди класти готові файли (в корінь проєкту)
        f'--distpath={str(root_dir / "dist")}',
        f'--workpath={str(root_dir / "build")}'
    ]

    # Запускаємо процес збірки
    PyInstaller.__main__.run(args)
    print("========================================")
    print("✅ Збірка успішно завершена!")
    print(f"📂 Шукайте готову програму у папці: {root_dir / 'dist' / 'DroneLocalization'}")


if __name__ == "__main__":
    build_app()