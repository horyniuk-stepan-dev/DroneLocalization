#!/usr/bin/env python3
"""A/B-компаратор ЯКОСТІ локалізації між двома прогонами місії (app.log).

PIPELINE_OPTIMIZATION_PLAN §A1: темпоральний prior міняє, ЯКІ кадри БД
перевіряються, тож швидкість може лишитись чудовою, а точність тихо просісти.
Цей інструмент читає два app.log (той самий детермінований відеофайл: baseline
з вимкненим прапорцем і кандидат із увімкненим) і числово порівнює якість:

  * скільки keyframe-ів локалізовано / провалено (і чому);
  * розподіл inliers і confidence;
  * позбіг matched-frame по keyframe-ах (головний сигнал для A1);
  * зсув позиції в метрах там, де обидва прогони успішні;
  * hit-rate темпорального prior (рядок [temporal-prior]).

Read-only: не гейт, нічого не пише в конфіг. Парсинг — чистий stdlib, тому
запускається будь-де, включно з цією пісочницею.

Використання:
  python scripts/ab_localization.py logs/app_baseline.log logs/app_a1.log
  python scripts/ab_localization.py logs/app.log            # лише зведення
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter
from statistics import median

# frame= може мати необовʼязковий " | source=..." одразу після номера.
_LOC = re.compile(
    r"Localized \(([-\d.]+), ([-\d.]+)\) \| frame=(\d+).*?\| inliers=(\d+) \| conf=([\d.]+)"
)
_TP = re.compile(r"\[temporal-prior\] tries=(\d+) hits=(\d+)")
_FAILS = (
    "Not enough valid inliers",
    "Localization failed",
    "Outlier filtered",
    "out_of_coverage",
    "No candidates found",
    "No propagated calibration",
)


def parse(path: str) -> dict:
    recs: list[tuple] = []          # (lat, lon, frame, inliers, conf)
    fails: Counter = Counter()
    tp = None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _LOC.search(line)
            if m:
                recs.append(
                    (float(m[1]), float(m[2]), int(m[3]), int(m[4]), float(m[5]))
                )
                continue
            for key in _FAILS:
                if key in line:
                    fails[key] += 1
                    break
            t = _TP.search(line)
            if t:
                tp = (int(t[1]), int(t[2]))  # остання зведена статистика
    return {"recs": recs, "fails": fails, "tp": tp}


def _haversine_m(a: tuple, b: tuple) -> float:
    """Відстань між (lat, lon) у метрах."""
    r = 6_371_000.0
    la1, lo1, la2, lo2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = (
        math.sin((la2 - la1) / 2) ** 2
        + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    )
    return 2 * r * math.asin(min(1.0, math.sqrt(h)))


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100.0 * len(s)))]


def summarize(name: str, d: dict) -> None:
    recs = d["recs"]
    inl = [r[3] for r in recs]
    conf = [r[4] for r in recs]
    n_ok = len(recs)
    n_fail = sum(d["fails"].values())
    total = n_ok + n_fail
    print(f"\n=== {name} ===")
    print(f"  keyframe успіхів : {n_ok}")
    print(f"  невдач           : {n_fail}"
          + (f"  ({total and 100 * n_ok / total:.1f}% success)" if total else ""))
    for k, v in d["fails"].most_common():
        print(f"      {k:<28} {v}")
    if inl:
        print(f"  inliers  median={median(inl):.0f}  p10={_pct(inl,10):.0f}  "
              f"min={min(inl)}")
        print(f"  conf     median={median(conf):.3f}  p10={_pct(conf,10):.3f}")
    if d["tp"]:
        tries, hits = d["tp"]
        rate = 100.0 * hits / tries if tries else 0.0
        print(f"  temporal-prior: tries={tries} hits={hits} ({rate:.0f}% hit-rate)")


def compare(a: dict, b: dict) -> None:
    """Порівняння якості. Головні сигнали — БЕЗ парування по індексу (лог не
    містить номера keyframe, тож зсув через різні провали ламає індексне
    вирівнювання). Індексне парування — лише коли кількість успіхів однакова.
    """
    ra, rb = a["recs"], b["recs"]
    print("\n" + "=" * 60)
    print("A/B  (A=baseline, B=candidate)")
    print("=" * 60)
    na, nb = len(ra), len(rb)
    fa, fb = sum(a["fails"].values()), sum(b["fails"].values())
    print(f"успіхів : A={na}  B={nb}  Δ={nb - na:+d}")
    print(f"невдач  : A={fa}  B={fb}  Δ={fb - fa:+d}")
    inl_a = [r[3] for r in ra]
    inl_b = [r[3] for r in rb]
    if inl_a and inl_b:
        print(f"inliers median: A={median(inl_a):.0f}  B={median(inl_b):.0f}  "
              f"Δ={median(inl_b) - median(inl_a):+.0f}")

    # ── Головний сигнал §A1: перетин МНОЖИН matched-frame (не залежить від
    #    вирівнювання). Якщо prior зводить локалізацію на ті самі кадри БД —
    #    Jaccard ≈ 1; якщо збиває на інші — падає.
    set_a = {r[2] for r in ra}
    set_b = {r[2] for r in rb}
    if set_a or set_b:
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        jac = inter / union if union else 1.0
        print(f"matched-frame множини: |A|={len(set_a)} |B|={len(set_b)} "
              f"спільних={inter}  Jaccard={jac:.3f}")
        only_b = sorted(set_b - set_a)
        if only_b:
            print(f"  кадри лише в B ({len(only_b)}): "
                  f"{only_b[:12]}{' …' if len(only_b) > 12 else ''}")

    # ── Вторинно: індексне парування лише за РІВНОЇ кількості успіхів ──────
    if na == nb and na > 0:
        same = sum(1 for i in range(na) if ra[i][2] == rb[i][2])
        dpos = [_haversine_m(ra[i][:2], rb[i][:2]) for i in range(na)]
        dinl = [rb[i][3] - ra[i][3] for i in range(na)]
        print("[рівні лічильники → парування по індексу валідне]")
        print(f"  збіг frame по парах: {same}/{na} ({100.0 * same / na:.1f}%)")
        print(f"  зсув позиції: median={median(dpos):.2f} м  p90={_pct(dpos,90):.2f} м  "
              f"max={max(dpos):.2f} м")
        print(f"  Δinliers (B−A): median={median(dinl):+.0f}  p10={_pct(dinl,10):+.0f}")
    else:
        print("[різні лічильники успіхів → парування по індексу пропущено "
              "(ненадійне); дивись Jaccard і агрегати вище]")

    # ── Вердикт з alignment-free сигналів ─────────────────────────────────
    v = []
    tot_a, tot_b = na + fa, nb + fb
    if tot_a and tot_b:
        sr_a, sr_b = na / tot_a, nb / tot_b
        if sr_b < sr_a - 0.02:
            v.append(f"success-rate впав {sr_a:.1%}→{sr_b:.1%}")
    if set_a or set_b:
        jac = len(set_a & set_b) / (len(set_a | set_b) or 1)
        if jac < 0.90:
            v.append(f"matched-frame Jaccard {jac:.2f} < 0.90 — prior зводить на інші кадри")
    if inl_a and inl_b and median(inl_b) < median(inl_a) - 50:
        v.append(f"медіанний inliers впав на {median(inl_a) - median(inl_b):.0f}")
    print("  вердикт: " + ("; ".join(v) if v else
                            "суттєвих розбіжностей якості не видно"))
    if b["tp"]:
        tries, hits = b["tp"]
        print(f"  temporal-prior у B: {hits}/{tries} "
              f"({100.0 * hits / tries if tries else 0:.0f}% hit-rate)")


def main(argv: list[str]) -> int:
    if not 1 <= len(argv) <= 2:
        print(__doc__)
        return 2
    a = parse(argv[0])
    summarize(argv[0], a)
    if len(argv) == 2:
        b = parse(argv[1])
        summarize(argv[1], b)
        compare(a, b)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
