import sys
from pathlib import Path
from typing import Sequence, Union

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

PathLike = Union[str, Path]


def merge_files(
    sources: Union[PathLike, Sequence[PathLike]],
    output_file: PathLike,
    base_dir: PathLike | None = None,
) -> None:
    """
    Збирає всі .py файли з директорій та/або окремих файлів, вказаних у sources,
    та об'єднує їх в один великий файл output_file, розділяючи коментарями
    з відносними шляхами оригінальних файлів.
    """
    output = Path(output_file).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(sources, (str, Path)):
        source_paths = [Path(sources).resolve()]
    else:
        source_paths = [Path(s).resolve() for s in sources]

    files_to_merge: list[Path] = []
    seen: set[Path] = set()

    for sp in source_paths:
        if not sp.exists():
            print(f"[WARN] Шлях не існує: {sp}")
            continue
        if sp.is_file():
            if sp.suffix == ".py" and sp != output and sp not in seen:
                files_to_merge.append(sp)
                seen.add(sp)
        elif sp.is_dir():
            for file_path in sorted(sp.rglob("*.py")):
                file_path_resolved = file_path.resolve()
                if (
                    file_path_resolved.is_file()
                    and file_path_resolved != output
                    and file_path_resolved not in seen
                ):
                    files_to_merge.append(file_path_resolved)
                    seen.add(file_path_resolved)

    if base_dir is not None:
        base = Path(base_dir).resolve()
    else:
        try:
            base = Path(Path.commonpath(source_paths))
            if base.is_file():
                base = base.parent
        except Exception:
            base = source_paths[0].parent if source_paths[0].is_dir() else source_paths[0].parent

    merged_content = []
    copied = 0

    for file_path in files_to_merge:
        try:
            relative = file_path.relative_to(base)
        except ValueError:
            relative = file_path.name

        merged_content.append(f"\n\n# {'=' * 80}\n# File: {relative}\n# {'=' * 80}\n")

        try:
            with open(file_path, encoding="utf-8") as f:
                merged_content.append(f.read())
            print(f"[MERGED] {relative} -> {output.name}")
            copied += 1
        except Exception as e:
            print(f"[ERROR] Не вдалося прочитати файл {file_path}: {e}")

    with open(output, "w", encoding="utf-8") as out_f:
        out_f.write("".join(merged_content))

    print(f"\nГотово! {copied} файлів об'єднано в {output}.")


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    SOURCES = [
        PROJECT_ROOT / "config",
        PROJECT_ROOT / "src",
        PROJECT_ROOT / "main.py",
    ]
    OUTPUT = PROJECT_ROOT / "scripts" / "allFiles" / "all_merged_new.py"

    merge_files(SOURCES, OUTPUT)
