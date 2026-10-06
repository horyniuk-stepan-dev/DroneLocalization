"""Calibrate the VLAD retrieval-score thresholds from replay reports.

Every retrieval call of a replay (``rows[].retrieval_calls``, written by
``scripts/replay_multilayer_ground_truth.py``) is scored against the query's
ground truth: the call's top-1 candidate is *correct* when that reference
frame's footprint (its propagated affine) contains the query's true ground
centre. From the score distributions of correct and wrong top-1 candidates the
script reports, for a grid of thresholds, how many calls pass, how many of those
are correct (precision) and how many correct calls pass (recall), and suggests:

* ``retrieval_only_min_score`` — lowest t whose precision is >= ``--precision``
  (an unverified fix is only acceptable when the top-1 frame is almost surely right);
* ``rotation_rescan_min_score`` / ``scale_rescan_min_score`` (keep the prior angle /
  scale without a full rescan) — the ``--keep``-quantile of correct top-1 scores of
  the first call of each frame (the prior view in steady state), so that share of
  good priors is kept; the share of wrong top-1 calls it would also keep is shown.

Rows are resampled (bootstrap) for confidence intervals because the calls of one
frame are correlated. Use replays of the path the app runs: with
``localization.layer_search.enabled = false`` in user_config.json, replay with
``--legacy-search``.

    python scripts/retrieval_threshold_report.py reports/r3m_legacy.json \\
        --out reports/thresholds_legacy.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CURRENT = {
    "retrieval_only_min_score": 0.90,
    "rotation_rescan_min_score": 0.70,
    "scale_rescan_min_score": 0.65,
}
GRID = np.round(np.arange(0.30, 1.0001, 0.01), 2)


@dataclass
class Footprints:
    """Reference frames of one source: affine (pixel -> metric) per slot, frame size."""

    affines: dict[int, np.ndarray]
    width: float
    height: float
    converter: object

    def contains(self, frame: int, lat: float, lon: float, margin: float = 0.0) -> bool | None:
        a = self.affines.get(int(frame))
        if a is None:
            return None
        x, y = self.converter.gps_to_metric(lat, lon)
        lin = a[:, :2]
        if abs(np.linalg.det(lin)) < 1e-12:
            return None
        u, v = np.linalg.solve(lin, np.array([x, y]) - a[:, 2])
        mx, my = margin * self.width, margin * self.height
        return bool(mx <= u <= self.width - mx and my <= v <= self.height - my)


def load_footprints(db_path: str | Path) -> Footprints:
    import h5py

    from src.geometry.coordinates import CoordinateConverter

    with h5py.File(db_path, "r") as db:
        meta = db["metadata"].attrs
        width, height = float(meta["frame_width"]), float(meta["frame_height"])
        grp = db["calibration"]
        affine = grp["frame_affine"][:]
        if "frame_georef_status" in grp:
            ok = grp["frame_georef_status"][:] == 1
        else:
            ok = grp["frame_valid"][:].astype(bool)
        raw = grp.attrs.get("projection_json")
        proj = json.loads(raw.decode() if isinstance(raw, bytes) else raw) if raw else {}
    affines = {int(i): affine[i].astype(np.float64) for i in np.flatnonzero(ok)}
    return Footprints(affines, width, height, CoordinateConverter.from_metadata(proj))


def score_calls(report: dict, footprints: dict[str, Footprints], margin: float = 0.0) -> list[dict]:
    """One record per retrieval call: row index, first-of-row flag, top-1 score/correct."""
    out = []
    for r_idx, row in enumerate(report.get("rows", [])):
        if not row.get("gt_valid") or row.get("gt_lat") is None:
            continue
        lat, lon = float(row["gt_lat"]), float(row["gt_lon"])
        for c_idx, call in enumerate(row.get("retrieval_calls") or []):
            best = None
            any_correct = False
            known = True
            for sid, cands in call.items():
                fp = footprints.get(sid)
                for rank, (frame, score) in enumerate(cands):
                    ok = fp.contains(frame, lat, lon, margin) if fp is not None else None
                    if rank == 0 and (best is None or score > best[2]):
                        best = (sid, int(frame), float(score), ok)
                    any_correct = any_correct or bool(ok)
            if best is None:
                continue
            if best[3] is None:
                known = False
            out.append(
                {
                    "row": r_idx,
                    "first": c_idx == 0,
                    "source": best[0],
                    "frame": best[1],
                    "score": best[2],
                    "correct": bool(best[3]) if known else None,
                    "any_correct_topk": any_correct,
                }
            )
    return out


def curve(scores: np.ndarray, correct: np.ndarray, grid=GRID) -> list[dict]:
    rows = []
    n_correct = max(int(correct.sum()), 1)
    for t in grid:
        passed = scores >= t
        n_pass = int(passed.sum())
        tp = int((passed & correct).sum())
        rows.append(
            {
                "t": float(t),
                "pass_rate": n_pass / max(len(scores), 1),
                "precision": tp / n_pass if n_pass else None,
                "recall": tp / n_correct,
                "n_pass": n_pass,
            }
        )
    return rows


def suggest_retrieval_only(scores, correct, precision: float, min_pass: int) -> float | None:
    for row in curve(scores, correct):
        if (
            row["n_pass"] >= min_pass
            and row["precision"] is not None
            and row["precision"] >= precision
        ):
            return row["t"]
    return None


def suggest_keep(scores, correct, keep: float) -> float | None:
    good = scores[correct]
    if good.size == 0:
        return None
    return float(np.floor(np.quantile(good, 1.0 - keep) * 100) / 100)


def bootstrap(records: list[dict], stat, n: int = 1000, seed: int = 0):
    """Percentile CI of ``stat(records)`` with rows (not calls) resampled."""
    by_row: dict[int, list[dict]] = {}
    for rec in records:
        by_row.setdefault(rec["row"], []).append(rec)
    rows = list(by_row)
    if len(rows) < 5:
        return None
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n):
        pick = rng.choice(len(rows), size=len(rows), replace=True)
        sample = [rec for i in pick for rec in by_row[rows[i]]]
        v = stat(sample)
        if v is not None:
            values.append(v)
    if not values:
        return None
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def _arrays(records):
    scores = np.array([r["score"] for r in records], dtype=np.float64)
    correct = np.array([bool(r["correct"]) for r in records], dtype=bool)
    return scores, correct


def precision_at(t: float):
    def stat(records):
        s, c = _arrays(records)
        passed = s >= t
        return float(c[passed].mean()) if passed.any() else None

    return stat


def recall_at(t: float):
    def stat(records):
        s, c = _arrays(records)
        return float((s[c] >= t).mean()) if c.any() else None

    return stat


def analyse(records: list[dict], precision: float, keep: float, min_pass: int, n_boot: int) -> dict:
    known = [r for r in records if r["correct"] is not None]
    first = [r for r in known if r["first"]]
    s_all, c_all = _arrays(known) if known else (np.zeros(0), np.zeros(0, bool))
    s_first, c_first = _arrays(first) if first else (np.zeros(0), np.zeros(0, bool))
    result = {
        "calls": len(records),
        "calls_with_known_footprint": len(known),
        "rows": len({r["row"] for r in known}),
        "top1_correct_rate": float(c_all.mean()) if known else None,
        "score_quantiles_correct": _quantiles(s_all[c_all]),
        "score_quantiles_wrong": _quantiles(s_all[~c_all]),
        "curve_all_calls": curve(s_all, c_all),
        "curve_first_call": curve(s_first, c_first),
        "current": {},
        "suggested": {},
    }
    for key, t in CURRENT.items():
        sample = first if key != "retrieval_only_min_score" else known
        s, c = _arrays(sample) if sample else (np.zeros(0), np.zeros(0, bool))
        passed = s >= t
        result["current"][key] = {
            "t": t,
            "pass_rate": float(passed.mean()) if s.size else None,
            "precision": float(c[passed].mean()) if passed.any() else None,
            "recall": float((s[c] >= t).mean()) if c.any() else None,
            "precision_ci95": bootstrap(sample, precision_at(t), n_boot),
            "recall_ci95": bootstrap(sample, recall_at(t), n_boot),
        }
    t_ro = suggest_retrieval_only(s_all, c_all, precision, min_pass) if known else None
    t_keep = suggest_keep(s_first, c_first, keep) if first else None
    if t_ro is not None:
        result["suggested"]["retrieval_only_min_score"] = {
            "t": t_ro,
            "rule": f"lowest t with precision >= {precision} and >= {min_pass} calls passing",
            "precision_ci95": bootstrap(known, precision_at(t_ro), n_boot),
            "recall": float((s_all[c_all] >= t_ro).mean()) if c_all.any() else None,
        }
    if t_keep is not None:
        passed = s_first >= t_keep
        result["suggested"]["rescan_min_score"] = {
            "t": t_keep,
            "rule": f"keeps {keep:.0%} of correct prior-view top-1 calls (first call per frame)",
            "wrong_kept_share": float((~c_first & passed).sum() / max((~c_first).sum(), 1)),
            "recall_ci95": bootstrap(first, recall_at(t_keep), n_boot),
        }
    return result


def _quantiles(values: np.ndarray) -> dict | None:
    if values.size == 0:
        return None
    q = np.percentile(values, [5, 25, 50, 75, 95])
    return {k: round(float(v), 4) for k, v in zip(("p5", "p25", "p50", "p75", "p95"), q)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("reports", nargs="+", type=Path, help="replay JSON reports")
    parser.add_argument("--out", type=Path, help="write the full analysis as JSON")
    parser.add_argument("--precision", type=float, default=0.99)
    parser.add_argument("--keep", type=float, default=0.95)
    parser.add_argument("--min-pass", type=int, default=30)
    parser.add_argument("--margin", type=float, default=0.0, help="footprint inset (fraction)")
    parser.add_argument("--bootstrap", type=int, default=1000)
    args = parser.parse_args(argv)

    records: list[dict] = []
    offset = 0
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        footprints = {
            src["source_id"]: load_footprints(src["database"])
            for src in report.get("inputs", {}).get("sources", [])
        }
        recs = score_calls(report, footprints, args.margin)
        for rec in recs:
            rec["row"] += offset
        offset += len(report.get("rows", []))
        records.extend(recs)
        print(f"{path.name}: {len(recs)} retrieval calls")
    if not records:
        print("no retrieval_calls in the reports (replay older than 2026-10-04?)")
        return 1
    result = analyse(records, args.precision, args.keep, args.min_pass, args.bootstrap)
    result["inputs"] = [str(p) for p in args.reports]
    print(
        f"calls {result['calls_with_known_footprint']} in {result['rows']} frames, "
        f"top-1 correct {result['top1_correct_rate']:.1%}"
    )
    print(f"score correct {result['score_quantiles_correct']}")
    print(f"score wrong   {result['score_quantiles_wrong']}")
    for key, cur in result["current"].items():
        print(
            f"current {key} = {cur['t']}: pass {cur['pass_rate']}, precision {cur['precision']} "
            f"{cur['precision_ci95']}, recall {cur['recall']} {cur['recall_ci95']}"
        )
    for key, sug in result["suggested"].items():
        print(f"suggested {key} = {sug['t']} ({sug['rule']}): {sug}")
    if args.out:
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
