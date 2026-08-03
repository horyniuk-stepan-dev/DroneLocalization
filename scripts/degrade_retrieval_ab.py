#!/usr/bin/env python3
"""Стійкість retrieval до деградації зображення — стенд для A/B «CLS проти VLAD».

ЗАДАЧА. VLAD брали заради стійкості до зміни зовнішнього вигляду. Виміряти це
на одному обльоті неможливо: запит шукає сам себе і будь-який дескриптор дає
стелю. Цей стенд ламає саме ту вісь, що цікавить — фотометрію кадру — лишаючи
геометрію незмінною. Тоді відома правильна відповідь (кадр зі слота i мусить
знайти слот i), і єдине, що змінюється, — вигляд.

ЩО ВІН НЕ Є. Синтетична деградація — проксі сезону, а не сезон. Вона не
відтворює структурних змін (зібраний урожай, сніг, інші тіні, вирубка). Це
нижня оцінка й димовий тест, а не заміна різносезонній парі чи супутниковій
гілці (`RESEARCH_INTEGRATION_PLAN` §3.2). Результат «VLAD не виграв тут» слабко
свідчить проти VLAD; результат «VLAD програв тут» — сильно.

МЕТОДИКА. Пошук — точний косинус по ВСІХ векторах бази (не ANN-індекс), щоб
різниця між базами не змішувалася з різницею апроксимації індексу. Обидві
сторони L2-нормуються. Глобальний дескриптор рахується через
`FeatureExtractor.extract_global_descriptor`, тобто тим самим кодом, що будував
базу; CLAHE до цієї гілки не застосовується (він тільки на локальних фічах),
тож препроцес збігається за побудовою.

САМОПЕРЕВІРКА. Рівень 0 — чистий кадр. Якщо на ньому R@1 не ≈ 1.0, стенд
розійшовся з базою (не той відеофайл, не той frame_step, не той конфіг), і всі
інші числа безглузді. Скрипт про це кричить і повертає код 3.

ЗАПУСК (по одній базі за раз; конфіг models.vlad.enabled мусить відповідати базі —
скрипт це звіряє й падає, якщо ні):

    python scripts/degrade_retrieval_ab.py --video "D:/My Projects/FlightSimulator/flight.mp4" ^
        --db "D:/My Projects/TEST/topnew/sources/main/database.h5" --out logs/deg_cls.json

    (перемкнути models.vlad.enabled на true)

    python scripts/degrade_retrieval_ab.py --video "D:/My Projects/FlightSimulator/flight.mp4" ^
        --db "D:/My Projects/TEST/topvlad/sources/main/database.h5" --out logs/deg_vlad.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import zlib  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# ── Деградації ───────────────────────────────────────────────────────────────
# Рівень 0 у кожної = чистий кадр. Рівні 1..5 — зростаюча сила. Усі детерміновані
# за (slot, degradation, level), тож прогони відтворювані побайтово.

_GAMMA = [1.0, 1.3, 1.6, 2.0, 2.5, 3.0]
_BLUR_SIGMA = [0.0, 1.0, 2.0, 3.0, 5.0, 7.0]
_JPEG_Q = [100, 70, 50, 35, 20, 10]
_NOISE_SIGMA = [0.0, 5.0, 10.0, 20.0, 35.0, 50.0]
_HAZE_ALPHA = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
_SHADOW_STRENGTH = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75]
_SEASON_HUE = [0, 6, 12, 18, 24, 30]  # зсув відтінку в бік жовто-брунатного
_SEASON_SAT = [1.0, 0.85, 0.70, 0.55, 0.40, 0.25]


def deg_gamma(img: np.ndarray, lvl: int, rng: np.random.Generator) -> np.ndarray:
    g = _GAMMA[lvl]
    # round, не truncate: при g=1.0 i/255*255 подекуди дає i-eps, і .astype
    # зрізало б до i-1 — рівень 0 переставав би бути чистим кадром.
    lut = np.clip(np.round(np.linspace(0, 1, 256) ** g * 255.0), 0, 255).astype(np.uint8)
    return cv2.LUT(img, lut)


def deg_blur(img: np.ndarray, lvl: int, rng: np.random.Generator) -> np.ndarray:
    s = _BLUR_SIGMA[lvl]
    return img if s <= 0 else cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=s)


def deg_jpeg(img: np.ndarray, lvl: int, rng: np.random.Generator) -> np.ndarray:
    q = _JPEG_Q[lvl]
    if q >= 100:
        return img
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def deg_noise(img: np.ndarray, lvl: int, rng: np.random.Generator) -> np.ndarray:
    s = _NOISE_SIGMA[lvl]
    if s <= 0:
        return img
    n = rng.normal(0.0, s, size=img.shape)
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def deg_haze(img: np.ndarray, lvl: int, rng: np.random.Generator) -> np.ndarray:
    a = _HAZE_ALPHA[lvl]
    if a <= 0:
        return img
    white = np.full_like(img, 235)
    return cv2.addWeighted(img, 1.0 - a, white, a, 0.0)


def deg_shadow(img: np.ndarray, lvl: int, rng: np.random.Generator) -> np.ndarray:
    """Лінійний градієнт затемнення — груба імітація іншого кута сонця."""
    k = _SHADOW_STRENGTH[lvl]
    if k <= 0:
        return img
    h, w = img.shape[:2]
    ramp = np.linspace(1.0 - k, 1.0, w, dtype=np.float32)[None, :, None]
    return np.clip(img.astype(np.float32) * ramp, 0, 255).astype(np.uint8)


def deg_season(img: np.ndarray, lvl: int, rng: np.random.Generator) -> np.ndarray:
    """Зелень → жовто-брунатне + втрата насиченості. Головний сезонний сигнал
    на полях, і саме той, під який брали VLAD."""
    dh, sat = _SEASON_HUE[lvl], _SEASON_SAT[lvl]
    if dh == 0 and sat == 1.0:
        return img
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] - dh) % 180  # OpenCV hue 0..179
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


DEGRADATIONS = {
    "season": deg_season,
    "gamma": deg_gamma,
    "shadow": deg_shadow,
    "blur": deg_blur,
    "haze": deg_haze,
    "jpeg": deg_jpeg,
    "noise": deg_noise,
}
N_LEVELS = 6  # 0..5


# ── Робота з базою ───────────────────────────────────────────────────────────


def load_db_vectors(db_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """→ (frame_ids (M,), vectors (M, D) L2-нормовані, metadata)."""
    from src.database.database_loader import DatabaseLoader

    loader = DatabaseLoader(db_path)
    meta = dict(loader.metadata)

    if loader.lance_table is not None:
        tbl = loader.lance_table.to_arrow()
        frame_ids = np.asarray(tbl.column("frame_id").to_pylist(), dtype=np.int64)
        vecs = np.asarray(
            [np.asarray(v, dtype=np.float32) for v in tbl.column("vector").to_pylist()],
            dtype=np.float32,
        )
    elif loader.global_descriptors is not None:
        vecs = np.asarray(loader.global_descriptors, dtype=np.float32)
        frame_ids = np.arange(len(vecs), dtype=np.int64)
        keep = np.linalg.norm(vecs, axis=1) > 0  # незаповнені слоти — нулі
        frame_ids, vecs = frame_ids[keep], vecs[keep]
    else:
        raise SystemExit(f"ERROR: у {db_path} немає ані LanceDB, ані HDF5-дескрипторів")

    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-12)
    return frame_ids, vecs, meta


def check_config_matches_db(meta: dict) -> tuple[bool, int]:
    """Звірити models.vlad.enabled у конфігу з тим, чим збудована база."""
    from config import APP_CONFIG, get_cfg

    cfg_vlad = bool(get_cfg(APP_CONFIG, "models.vlad.enabled", False))
    raw = meta.get("schema_components")
    if not raw:
        print("УВАГА: база без schema_components — звірку конфігу пропущено")
        return cfg_vlad, int(meta.get("descriptor_dim", 0))
    comp = json.loads(raw)
    db_vlad = bool(comp.get("vlad_enabled", False))
    if cfg_vlad != db_vlad:
        raise SystemExit(
            f"ERROR: конфіг і база розійшлися. models.vlad.enabled={cfg_vlad}, "
            f"а база збудована з vlad_enabled={db_vlad}. Приведіть user_config.json "
            f"у відповідність до бази й перезапустіть."
        )
    return db_vlad, int(comp.get("descriptor_dim", 0))


def build_extractor():
    from config import APP_CONFIG, get_cfg
    from src.models.model_manager import ModelManager
    from src.models.wrappers.feature_extractor import FeatureExtractor

    # APP_CONFIG обов'язковий: ModelManager(None) дає self.config={}, а
    # load_dinov2() читає get_cfg(cfg, "global_descriptor.backend", "dinov2") —
    # тобто без конфігу тихо піднімає DINOv2 замість DINOv3, яким збудована база.
    # Дескриптор виходить тієї ж розмірності (1024), тож нічого не падає, а
    # retrieval стає випадковим.
    mm = ModelManager(APP_CONFIG)
    global_model = mm.load_dinov2()
    # local_model не потрібен: глобальна гілка його не торкається.
    extractor = FeatureExtractor(None, global_model, device=mm.device, config=APP_CONFIG)
    logger_name = type(global_model).__name__
    backend = get_cfg(APP_CONFIG, "global_descriptor.backend", "?")
    print(f"Бекбон: {logger_name} (global_descriptor.backend={backend})")
    return extractor


# ── Головний цикл ────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--video", required=True, help="референсне відео (те, з якого база)")
    ap.add_argument("--db", required=True, help="database.h5")
    ap.add_argument("--samples", type=int, default=150, help="скільки слотів пробувати")
    ap.add_argument("--degradations", default="all", help="через кому або 'all'")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="куди писати JSON-звіт")
    args = ap.parse_args()

    names = list(DEGRADATIONS) if args.degradations == "all" else args.degradations.split(",")
    for n in names:
        if n not in DEGRADATIONS:
            raise SystemExit(f"ERROR: невідома деградація {n!r}; доступні: {list(DEGRADATIONS)}")

    frame_ids, vecs, meta = load_db_vectors(args.db)
    vlad_on, db_dim = check_config_matches_db(meta)
    frame_step = int(meta.get("frame_step", 30))
    print(
        f"База: {len(frame_ids)} векторів, dim={vecs.shape[1]} (метадані: {db_dim}), "
        f"frame_step={frame_step}, VLAD={'ON' if vlad_on else 'OFF'}"
    )

    rng = np.random.default_rng(args.seed)
    pick = np.sort(
        rng.choice(len(frame_ids), size=min(args.samples, len(frame_ids)), replace=False)
    )
    slots = frame_ids[pick]
    row_of_slot = {int(s): i for i, s in enumerate(frame_ids)}

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"ERROR: не відкривається відео {args.video}")

    frames: dict[int, np.ndarray] = {}
    for s in slots:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s) * frame_step)
        ok, fr = cap.read()
        if ok:
            frames[int(s)] = fr
    cap.release()
    print(f"Прочитано {len(frames)}/{len(slots)} кадрів запиту")
    if not frames:
        raise SystemExit("ERROR: жодного кадру не прочитано — не той відеофайл?")

    extractor = build_extractor()

    def evaluate(name: str, lvl: int) -> dict:
        fn = DEGRADATIONS[name]
        ranks = []
        for slot, frame in frames.items():
            sub = np.random.default_rng([args.seed, slot, lvl, zlib.crc32(name.encode())])
            # Деградації працюють у BGR (deg_season покладається на
            # COLOR_BGR2HSV), а екстрактор чекає RGB: database_builder.py:400
            # робить cvtColor(BGR2RGB) перед extract. Без цієї конверсії
            # порядок каналів розходиться з базою і R@1 падає в нуль.
            deg = cv2.cvtColor(fn(frame, lvl, sub), cv2.COLOR_BGR2RGB)
            q = np.asarray(extractor.extract_global_descriptor(deg), dtype=np.float32).ravel()
            q /= max(float(np.linalg.norm(q)), 1e-12)
            sims = vecs @ q
            ranks.append(int((sims > sims[row_of_slot[slot]]).sum()))  # 0 = перший
        a = np.asarray(ranks)
        return {
            "n": int(len(a)),
            "r@1": float((a == 0).mean()),
            "r@5": float((a < 5).mean()),
            "median_rank": float(np.median(a)),
            "p95_rank": float(np.percentile(a, 95)),
        }

    def show(name: str, lvl: int, r: dict) -> None:
        print(
            f"  {name:<8} lvl={lvl}  R@1={r['r@1']:.3f}  R@5={r['r@5']:.3f}  "
            f"med_rank={r['median_rank']:.0f}  p95_rank={r['p95_rank']:.0f}"
        )

    # Рівень 0 у КОЖНОЇ деградації — тотожність, тож рахуємо його один раз і
    # падаємо ДО основного циклу: зламаний стенд коштує секунди, а не хвилини.
    clean = evaluate(names[0], 0)
    show("clean", 0, clean)
    if clean["r@1"] < 0.95:
        print(
            f"\nПОМИЛКА САМОПЕРЕВІРКИ: на чистому кадрі R@1 = {clean['r@1']:.3f} < 0.95.\n"
            f"Стенд розійшовся з базою. Перевірте по черзі:\n"
            f"  • бекбон у рядку 'Бекбон:' вище — має відповідати базі;\n"
            f"  • той самий відеофайл, з якого будувалася база;\n"
            f"  • frame_step у метаданих бази;\n"
            f"  • models.vlad.enabled = стан бази.\n"
            f"Основний цикл не запускався."
        )
        return 3

    results: dict[str, dict] = {}
    for name in names:
        results[name] = {0: dict(clean)}
        for lvl in range(1, N_LEVELS):
            results[name][lvl] = evaluate(name, lvl)
            show(name, lvl, results[name][lvl])

    clean_min = clean["r@1"]
    print(f"\nСамоперевірка (рівень 0, чистий кадр): R@1 = {clean_min:.3f}")

    report = {
        "db": args.db,
        "video": args.video,
        "vlad_enabled": vlad_on,
        "descriptor_dim": int(vecs.shape[1]),
        "db_vectors": int(len(frame_ids)),
        "samples": int(len(frames)),
        "frame_step": frame_step,
        "seed": args.seed,
        "clean_r@1": clean_min,
        "results": results,
    }
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(
            json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        print(f"Звіт збережено: {args.out}")

    # Хвостової перевірки більше немає: рівень 0 гейтиться ДО основного циклу,
    # тож сюди можна дійти лише з валідним стендом.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
