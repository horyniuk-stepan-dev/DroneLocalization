
import shutil
from pathlib import Path


def collect_files(source_dir: str, output_dir: str) -> None:
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    copied = 0

    for file_path in source.rglob("*.py"):
        if not file_path.is_file():
            continue
        if file_path.resolve().is_relative_to(output):
            continue

        # Build new filename from relative path: a/b/c.py -> a__b__c.py
        relative = file_path.relative_to(source)
        new_name = "__".join(relative.parts[:-1] + (file_path.name,))
        dest = output / new_name

        shutil.copy2(file_path, dest)
        print(f"[COPIED] {file_path} -> {new_name}")
        copied += 1

    print(f"\nDone: {copied} copied.")


if __name__ == "__main__":
    collect_files("E:/Dip/budyak/DroneLocalization/src", "E:/Dip/gsdfg/New/DroneLocalization/scripts/allFiles")