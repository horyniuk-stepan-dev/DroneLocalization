"""scripts/compare_replays.py: metrics, block bootstrap, paired comparison, route pooling."""

import json

import numpy as np
import pytest

from scripts import compare_replays as C


def _report(path, errors, confirmed=None, ms=100.0):
    rows = []
    for i, e in enumerate(errors):
        ok = True if confirmed is None else confirmed[i]
        rows.append(
            {
                "slot": i,
                "gt_valid": True,
                "confirmed": ok,
                "raw_error_m": float(e) if ok else None,
                "processing_ms": ms,
            }
        )
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    return path


def test_metrics_and_block_indices():
    rows = [
        {"slot": i, "confirmed": i % 2 == 0, "raw_error_m": float(i), "processing_ms": 5.0}
        for i in range(10)
    ]
    m = C.metrics(rows, false_m=5.0)
    assert m["confirmed"] == 0.5 and m["median_m"] == 4.0 and m["false"] == 2 and m["ms"] == 5.0
    idx = C.block_indices(23, 5, np.random.default_rng(0))
    assert len(idx) == 23 and idx.max() < 23
    assert C.block_indices(0, 5, np.random.default_rng(0)).size == 0


def test_paired_comparison_detects_improvement(tmp_path):
    rng = np.random.default_rng(1)
    base_err = rng.uniform(5, 15, 200)
    _report(tmp_path / "base.json", base_err)
    _report(tmp_path / "better.json", base_err * 0.5)
    res = C.compare(sorted(tmp_path.glob("*.json")), "base", False, 300, 10, 10.0)
    v = res["better"]["vs_baseline"]
    assert v["pairs"] == 200 and v["d_median_err_m"] < 0
    lo, hi = v["d_median_err_ci95"]
    assert hi < 0  # interval excludes zero
    assert res["base"]["median_ci95"][0] <= res["base"]["median_m"] <= res["base"]["median_ci95"][1]


def test_group_pools_routes(tmp_path):
    for route in ("r1", "r2"):
        _report(tmp_path / f"a@{route}.json", np.full(50, 5.0))
        _report(
            tmp_path / f"b@{route}.json",
            np.full(50, 7.0),
            confirmed=[i % 5 != 0 for i in range(50)],
        )
    res = C.compare(sorted(tmp_path.glob("*.json")), "a", True, 200, 5, 10.0)
    assert res["a"]["n"] == 100 and res["a"]["routes"] == ["a@r1", "a@r2"]
    v = res["b"]["vs_baseline"]
    assert v["pairs"] == 100 and v["d_confirmed"] == pytest.approx(-0.2)
    assert v["d_median_err_m"] == pytest.approx(2.0)


def test_main(tmp_path, capsys):
    _report(tmp_path / "x.json", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    (tmp_path / "notareport.json").write_text("{}", encoding="utf-8")
    # retrieval_threshold_report.py output in the same folder ("rows" is a count)
    (tmp_path / "thresholds.json").write_text('{"calls": 3, "rows": 2}', encoding="utf-8")
    out = tmp_path / "out" / "c.json"
    out.parent.mkdir()
    assert C.main([str(tmp_path), "--bootstrap", "50", "--out", str(out)]) == 0
    assert "x" in json.loads(out.read_text(encoding="utf-8"))
    assert C.main([str(tmp_path / "out")]) == 1
