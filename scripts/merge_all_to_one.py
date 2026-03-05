import sys
from pathlib import Path

def merge_files(source_dir: str, output_file: str) -> None:
    """
    Збирає всі .py файли з source_dir та об'єднує їх в один великий файл output_file,
    розділяючи коментарями з іменами оригінальних файлів.
    """
    source = Path(source_dir).resolve()
    output = Path(output_file).resolve()
    
    # Створюємо директорію для вихідного файлу, якщо її немає
    output.parent.mkdir(parents=True, exist_ok=True)
    
    merged_content = []
    copied = 0

    for file_path in source.rglob("*.py"):
        if not file_path.is_file():
            continue
        # Пропускаємо сам вихідний файл, якщо він знаходиться всередині source_dir
        if file_path.resolve() == output:
            continue

        relative = file_path.relative_to(source)
        
        # Додаємо красивий заголовок з назвою файлу
        merged_content.append(f"\n\n# {'='*80}\n# File: {relative}\n# {'='*80}\n")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                merged_content.append(f.read())
            print(f"[MERGED] {file_path.name} -> {output.name}")
            copied += 1
        except Exception as e:
            print(f"[ERROR] Не вдалося прочитати файл {file_path}: {e}")

    # Записуємо всі зібрані дані в один файл
    with open(output, 'w', encoding='utf-8') as out_f:
        out_f.write("".join(merged_content))

    print(f"\nГотово! {copied} файлів об'єднано в {output}.")


if __name__ == "__main__":
    # Змінюйте ці шляхи за потреби
    SOURCE = "E:/Dip/budyak/DroneLocalization/src"
    OUTPUT = "E:/Dip/gsdfg/New/DroneLocalization/scripts/allFiles/all_merged.py"
    
    merge_files(SOURCE, OUTPUT)
