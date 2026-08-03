#!/usr/bin/env python3
"""Порівняти дві бази на взаємозамінність ПЕРЕД A/B-експериментом.

Навіщо. A/B «CLS проти VLAD» має сенс лише тоді, коли бази відрізняються самим
VLAD-ом і більше нічим. Якщо базлайн збудовано раніше з іншим frame_step,
іншим локальним екстрактором чи іншим значенням ``dino_cpu_resize`` — різниця в
метриках припишеться VLAD-у, хоча її дала стороння зміна. Цей скрипт читає
fingerprint-компоненти обох баз (``/metadata`` в HDF5) і показує КОЖНЕ поле, що
розійшлося, розділяючи їх на очікувані (VLAD) і сторонні (конфаундери).

Read-only: нічого не пише і нічого не гейтить.

Запуск:
    python scripts/diagnostics/compare_db_schema.py ^
        --a "D:\\My Projects\\TEST\\topnew\\sources\\main\\database.h5" ^
        --b "D:\\My Projects\\TEST\\topvlad\\sources\\main\\database.h5"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Поля, розбіжність яких і Є предметом експерименту. Усе інше — конфаундер.
EXPECTED = {"vlad_enabled", "vlad_pca_dim", "descriptor_dim"}


def read_db(path: str) -> dict:
    import h5py

    out: dict = {"path": path}
    with h5py.File(path, "r") as f:
        if "metadata" not in f:
            raise SystemExit(f"ERROR: у {path} немає групи /metadata — це не база v2")
        a = f["metadata"].attrs
        for key in ("num_frames", "descriptor_dim", "frame_step", "hdf5_schema"):
            if key in a:
                out[key] = a[key]
        out["fingerprint"] = a.get("schema_fingerprint")
        raw = a.get("schema_components")
        out["components"] = json.loads(raw) if raw else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, help="database.h5 базлайна (без VLAD)")
    ap.add_argument("--b", required=True, help="database.h5 кандидата (з VLAD)")
    args = ap.parse_args()

    a, b = read_db(args.a), read_db(args.b)

    for tag, d in (("A", a), ("B", b)):
        print(f"\n[{tag}] {d['path']}")
        print(
            f"     frames={d.get('num_frames')}  descriptor_dim={d.get('descriptor_dim')}  "
            f"frame_step={d.get('frame_step')}  schema={d.get('hdf5_schema')}"
        )
        print(f"     fingerprint={d.get('fingerprint')}")

    if not a["components"] or not b["components"]:
        print(
            "\nВЕРДИКТ: НЕПОРІВНЯННО — принаймні в однієї бази немає "
            "schema_components (збудована старішим білдером, до появи "
            "fingerprint). Перебудуйте базлайн поточним кодом."
        )
        return 2

    from src.database.schema_fingerprint import compare

    diffs = compare(a["components"], b["components"])
    if not diffs:
        print(
            "\nВЕРДИКТ: бази ІДЕНТИЧНІ за схемою — VLAD не увімкнувся в жодній. "
            "Перевірте models.vlad.enabled і лог побудови."
        )
        return 2

    expected, confounders = [], []
    for line in diffs:
        (expected if line.split(":", 1)[0] in EXPECTED else confounders).append(line)

    print("\nОчікувані розбіжності (предмет експерименту):")
    for line in expected or ["  (жодної)"]:
        print(f"  {line}")

    if confounders:
        print("\nКОНФАУНДЕРИ — розбіжності, не пов'язані з VLAD:")
        for line in confounders:
            print(f"  {line}")
        print(
            "\nВЕРДИКТ: A/B СПЛУТАНИЙ. Різницю в метриках не можна приписати "
            "VLAD-у. Перебудуйте базлайн із тим самим конфігом, змінивши лише "
            "models.vlad.enabled."
        )
        return 1

    if a.get("num_frames") != b.get("num_frames"):
        print(
            f"\nУВАГА: різна кількість кадрів ({a.get('num_frames')} проти "
            f"{b.get('num_frames')}) при однаковій схемі — імовірно різне "
            f"вихідне відео або адаптивний вибір keyframe дав інший результат. "
            f"Порівнюйте частки, не абсолютні лічильники."
        )

    print("\nВЕРДИКТ: бази ПОРІВНЯННІ — відрізняються лише VLAD-ом. A/B валідний.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
