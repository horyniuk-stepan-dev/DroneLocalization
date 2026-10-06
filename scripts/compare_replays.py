"""Compare replay reports with block-bootstrap confidence intervals.

Each report (``scripts/replay_multilayer_ground_truth.py --json``) is reduced
to: confirmed share of GT-valid slots, raw error median / p95 / max of confirmed
slots, false confirmations (raw error > ``--false-m``), median processing time.
Consecutive slots of one flight are correlated, so intervals come from a
moving-block bootstrap over slots (``--block`` slots per block). With
``--baseline`` every other report is also compared slot by slot against it
(paired: the same query slots), which is far more sensitive than comparing two
independent intervals.

Reports of several routes can be pooled per variant with ``--group``: the name
up to the first ``@`` is the variant (``layer@route3min.json``,
``layer@heading.json`` → variant ``layer``); pooled intervals resample blocks
within each route.

    python scripts/compare_replays.py D:/maps/reports/dino --baseline s224_legacy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_rows(path: Path) -> list[dict]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in report.get("rows", []) if r.get("gt_valid")]
    rows.sort(key=lambda r: r["slot"])
    return rows


def metrics(rows: list[dict], false_m: float) -> dict:
    n = len(rows)
    conf = [r for r in rows if r.get("confirmed")]
    err = np.array([r["raw_error_m"] for r in conf if r.get("raw_error_m") is not None], float)
    ms = np.array([r["processing_ms"] for r in rows if r.get("processing_ms") is not None], float)
    return {
        "n": n,
        "confirmed": len(conf) / n if n else float("nan"),
        "median_m": float(np.median(err)) if err.size else float("nan"),
        "p95_m": float(np.percentile(err, 95)) if err.size else float("nan"),
        "max_m": float(err.max()) if err.size else float("nan"),
        "false": int((err > false_m).sum()),
        "ms": float(np.median(ms)) if ms.size else float("nan"),
    }


def block_indices(n: int, block: int, rng) -> np.ndarray:
    """Moving-block bootstrap: n indices made of random contiguous blocks."""
    if n == 0:
        return np.zeros(0, dtype=int)
    block = max(1, min(block, n))
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]


def bootstrap(groups: list[list[dict]], stat, n_boot: int, block: int, seed: int = 0):
    """95 % percentile interval of stat(pooled rows), blocks resampled inside each group."""
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_boot):
        sample = []
        for rows in groups:
            sample.extend(rows[i] for i in block_indices(len(rows), block, rng))
        v = stat(sample)
        if v is not None and np.isfinite(v):
            values.append(v)
    if len(values) < max(10, n_boot // 10):
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def paired(base: list[dict], other: list[dict]) -> list[tuple[dict, dict]]:
    by_slot = {r["slot"]: r for r in base}
    return [(by_slot[r["slot"]], r) for r in other if r["slot"] in by_slot]


def paired_stats(pairs: list[tuple[dict, dict]]) -> dict:
    d_conf = [float(bool(b.get("confirmed"))) - float(bool(a.get("confirmed"))) for a, b in pairs]
    d_err = [
        b["raw_error_m"] - a["raw_error_m"]
        for a, b in pairs
        if a.get("confirmed")
        and b.get("confirmed")
        and a.get("raw_error_m") is not None
        and b.get("raw_error_m") is not None
    ]
    return {
        "pairs": len(pairs),
        "d_confirmed": float(np.mean(d_conf)) if d_conf else float("nan"),
        "d_median_err_m": float(np.median(d_err)) if d_err else float("nan"),
        "both_confirmed": len(d_err),
    }


def variant_of(name: str, group: bool) -> str:
    return name.split("@", 1)[0] if group else name


def compare(
    paths: list[Path], baseline: str | None, group: bool, n_boot: int, block: int, false_m: float
) -> dict:
    reports: dict[str, list[tuple[str, list[dict]]]] = {}
    for path in paths:
        reports.setdefault(variant_of(path.stem, group), []).append((path.stem, load_rows(path)))
    result = {}
    for variant, items in sorted(reports.items()):
        groups = [rows for _, rows in items]
        pooled = [r for rows in groups for r in rows]
        m = metrics(pooled, false_m)
        m["routes"] = [name for name, _ in items]
        m["confirmed_ci95"] = bootstrap(
            groups, lambda s: metrics(s, false_m)["confirmed"], n_boot, block
        )
        m["median_ci95"] = bootstrap(
            groups, lambda s: metrics(s, false_m)["median_m"], n_boot, block
        )
        m["p95_ci95"] = bootstrap(groups, lambda s: metrics(s, false_m)["p95_m"], n_boot, block)
        result[variant] = m
    if baseline is not None:
        if baseline not in reports:
            raise SystemExit(f"baseline {baseline!r} not among {sorted(reports)}")
        base_by_route = {name.split("@", 1)[-1]: rows for name, rows in reports[baseline]}
        for variant, items in reports.items():
            if variant == baseline:
                continue
            pair_groups = []
            for name, rows in items:
                base_rows = base_by_route.get(name.split("@", 1)[-1])
                if base_rows is None and len(reports[baseline]) == 1:
                    base_rows = reports[baseline][0][1]
                if base_rows is not None:
                    pair_groups.append(paired(base_rows, rows))
            pooled = [p for g in pair_groups for p in g]
            if not pooled:
                continue
            stats = paired_stats(pooled)
            stats["d_confirmed_ci95"] = bootstrap(
                pair_groups, lambda s: paired_stats(s)["d_confirmed"], n_boot, block
            )
            stats["d_median_err_ci95"] = bootstrap(
                pair_groups, lambda s: paired_stats(s)["d_median_err_m"], n_boot, block
            )
            result[variant]["vs_baseline"] = stats
    return result


def _ci(ci, fmt="{:.1f}"):
    return "-" if ci is None else "[" + ", ".join(fmt.format(v) for v in ci) + "]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("inputs", nargs="+", type=Path, help="report JSON files or folders")
    parser.add_argument("--baseline", help="report (or variant with --group) to pair against")
    parser.add_argument("--group", action="store_true", help="pool <variant>@<route>.json")
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--block", type=int, default=10, help="slots per bootstrap block")
    parser.add_argument("--false-m", type=float, default=10.0)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    paths: list[Path] = []
    for item in args.inputs:
        paths += sorted(item.glob("*.json")) if item.is_dir() else [item]
    paths = [p for p in paths if "rows" in json.loads(p.read_text(encoding="utf-8"))]
    if not paths:
        print("no replay reports found")
        return 1
    result = compare(paths, args.baseline, args.group, args.bootstrap, args.block, args.false_m)
    print(
        f"{'variant':<22} {'conf':>6} {'CI95':>15} {'med m':>6} {'CI95':>13} "
        f"{'p95':>6} {'max':>7} {'>10m':>4} {'ms':>6}"
    )
    for name, m in result.items():
        print(
            f"{name:<22} {m['confirmed']:6.1%} {_ci(m['confirmed_ci95'], '{:.2f}'):>15} "
            f"{m['median_m']:6.1f} {_ci(m['median_ci95']):>13} {m['p95_m']:6.1f} "
            f"{m['max_m']:7.1f} {m['false']:>4} {m['ms']:6.0f}"
        )
    if args.baseline:
        print(f"\npaired vs {args.baseline} (other - baseline):")
        for name, m in result.items():
            v = m.get("vs_baseline")
            if v:
                print(
                    f"{name:<22} slots {v['pairs']:>5}  confirmed {v['d_confirmed']:+.3f} "
                    f"{_ci(v['d_confirmed_ci95'], '{:+.3f}')}  median err {v['d_median_err_m']:+.1f} m "
                    f"{_ci(v['d_median_err_ci95'], '{:+.1f}')} ({v['both_confirmed']} both confirmed)"
                )
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
