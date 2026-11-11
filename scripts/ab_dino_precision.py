#!/usr/bin/env python
"""A/B: DINOv3 global descriptors with FP32 weights, AMP off (A) vs AMP FP16 on (B).

Only ``models.performance.fp16_enabled`` differs. Input size, preprocessing,
VLAD/CLS aggregation, global_batch_max and the reference gallery (vectors.lance
of each source, built earlier and NOT rebuilt) stay as in user_config.json.
Each mode runs in its own fresh Python process (independent memory numbers).

Queries: GT frames of FlightSimulator route videos (independent of the map
videos), every --every-th frame, 4 quarter-turn views (0/90/180/270) like
rotation_mode=quarter at scale 1.0. No YOLO mask, no scale pyramid: this tests
the descriptor and retrieval, not the full localizer — that is the route replay
(tracking_routes.ps1 -Set models.performance.fp16_enabled=...).

Positives: gallery frames (any source) whose frame_gps lies within
--pos-radius-m of the query's GT ground centre. Key = (source_id, frame_id).

    python scripts/ab_dino_precision.py
    python scripts/ab_dino_precision.py --every 60 --routes winter base
    python scripts/ab_dino_precision.py --compare-only logs/dino_ab_<ts>   # re-analyse saved runs

Report: logs/dino_ab_<timestamp>/report.md (+ report.json, a.npz, b.npz)
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEF_MAPS = Path(r"D:\My Projects\TEST\testtopboch_max_dino336")
DEF_SIM = Path(r"D:\My Projects\FlightSimulator\output")
ANGLES = (0, 90, 180, 270)
ROT = {90: 0, 180: 1, 270: 2}  # cv2.ROTATE_90_CLOCKWISE, _180, _90_COUNTERCLOCKWISE


# ───────────────────────────── query set ─────────────────────────────
def route_dirs(sim: Path, names: list[str] | None):
    out = []
    for d in sorted(sim.glob("testtopboch_route3min*")):
        if not (d / "video.frames.jsonl").is_file():
            continue
        name = d.name.replace("testtopboch_route3min", "").lstrip("_") or "base"
        if name == "20260929":
            name = "base"
        if names and name not in names:
            continue
        out.append((name, d))
    return out


def query_list(routes, every):
    q = []
    for name, d in routes:
        with open(d / "video.frames.jsonl", encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["frame_index"] % every or not r.get("ground_center_valid"):
                    continue
                q.append((name, str(d / "video.mp4"), int(r["frame_index"]), r["ground_center_gps"]))
    return q


# ───────────────────────────── child: extract ─────────────────────────────
def child(mode: str, queries_json: Path, out: Path, warm: int):
    import cv2
    import torch

    from config import APP_CONFIG
    from src.models.model_manager import ModelManager
    from src.models.wrappers.feature_extractor import FeatureExtractor

    APP_CONFIG["models"]["performance"]["fp16_enabled"] = mode == "b"
    queries = json.loads(queries_json.read_text(encoding="utf-8"))

    torch.cuda.reset_peak_memory_stats()
    mm = ModelManager(config=APP_CONFIG)
    g = mm.load_dinov2()
    l = mm.load_local_extractor()
    fe = FeatureExtractor(l, g, mm.device, config=APP_CONFIG)
    mod = getattr(g, "model", g)
    params = list(mod.parameters())
    meta = {
        "mode": mode,
        "fp16_enabled": APP_CONFIG["models"]["performance"]["fp16_enabled"],
        "use_half": bool(getattr(fe, "use_half", None)),
        "amp_dtype": str(getattr(fe, "amp_dtype", None)),
        "param_dtypes": sorted({str(p.dtype) for p in params}),
        "param_bytes": int(sum(p.numel() * p.element_size() for p in params)),
        "input_size": APP_CONFIG["global_descriptor"]["dinov3"].get("input_size"),
        "global_batch_max": APP_CONFIG["models"]["performance"].get("global_batch_max"),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "alloc_after_load_mib": torch.cuda.memory_allocated() / 2**20,
    }

    caps: dict[str, cv2.VideoCapture] = {}

    def frame(video, idx):
        cap = caps.setdefault(video, cv2.VideoCapture(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"cannot read {video}#{idx}")
        # database builder / tracking worker hand the localizer RGB (see replay to_rgb)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def views(img):
        return [img if a == 0 else cv2.rotate(img, ROT[a]) for a in ANGLES]

    first = views(frame(queries[0][1], queries[0][2]))
    for _ in range(warm):
        fe.extract_global_descriptors_multi(first)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    descs, times = [], []
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    t_wall = time.perf_counter()
    for i, (_, video, idx, _) in enumerate(queries):
        v = views(frame(video, idx))
        s.record()
        d = fe.extract_global_descriptors_multi(v)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
        descs.append(np.asarray(d, dtype=np.float32))
        if i % 50 == 0:
            print(f"[{mode}] {i}/{len(queries)}", flush=True)
    meta["wall_s"] = time.perf_counter() - t_wall
    meta["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
    meta["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    np.savez(out, desc=np.stack(descs), ms=np.asarray(times), meta=json.dumps(meta))


# ───────────────────────────── gallery ─────────────────────────────
def load_gallery(maps: Path):
    from src.database.database_loader import DatabaseLoader

    pj = json.loads((maps / "project.json").read_text(encoding="utf-8"))
    vecs, keys, gps = [], [], []
    for s in pj["video_sources"]:
        if not s.get("enabled", True):
            continue
        db_path = maps / s["database_file"]
        if not db_path.is_file():
            continue
        db = DatabaseLoader(str(db_path))
        if db.lance_table is None or db.frame_gps is None:
            print(f"skip source {s['source_id']}: no lance table or frame_gps")
            continue
        t = db.lance_table.to_arrow()
        fid = np.asarray(t.column("frame_id").to_pylist())
        v = np.asarray(t.column("vector").to_pylist(), dtype=np.float32)
        vecs.append(v)
        keys += [(s["source_id"], int(f)) for f in fid]
        gps.append(db.frame_gps[fid])
    v = np.concatenate(vecs)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v, keys, np.concatenate(gps)


def haversine_m(lat1, lon1, lat2, lon2):
    r = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


# ───────────────────────────── compare ─────────────────────────────
def pct(x, q):
    return float(np.percentile(x, q)) if len(x) else float("nan")


def analyse(outdir: Path, maps: Path, pos_radius: float, top_k: int):
    queries = json.loads((outdir / "queries.json").read_text(encoding="utf-8"))
    A, B = np.load(outdir / "a.npz"), np.load(outdir / "b.npz")
    ma, mb = json.loads(str(A["meta"])), json.loads(str(B["meta"]))
    da, db = A["desc"], B["desc"]  # (N, V, D)
    rep = {"meta_a": ma, "meta_b": mb, "n_queries": len(queries), "views": list(ANGLES)}

    # 1. numerics
    num = {}
    for name, d in (("a", da), ("b", db)):
        num[name] = {"nan": int(np.isnan(d).sum()), "inf": int(np.isinf(d).sum()),
                     "norm_p50": pct(np.linalg.norm(d, axis=-1).ravel(), 50)}
    na = da / (np.linalg.norm(da, axis=-1, keepdims=True) + 1e-12)
    nb = db / (np.linalg.norm(db, axis=-1, keepdims=True) + 1e-12)
    cos = (na * nb).sum(-1).ravel()
    absd = np.abs(da - db)
    rel = (np.linalg.norm(da - db, axis=-1) / (np.linalg.norm(da, axis=-1) + 1e-12)).ravel()
    worst = np.argsort(cos)[:5]
    num["cos_ab"] = {"min": float(cos.min()), "p1": pct(cos, 1), "p50": pct(cos, 50)}
    num["abs_err"] = {"max": float(absd.max()), "mean": float(absd.mean())}
    num["rel_l2"] = {"p50": pct(rel, 50), "p99": pct(rel, 99), "max": float(rel.max())}
    num["worst"] = [{"query": queries[i // len(ANGLES)][:3], "angle": ANGLES[i % len(ANGLES)], "cos": float(cos[i])}
                    for i in worst]
    rep["numerics"] = num

    # 2. retrieval against the unchanged gallery
    gv, gkeys, ggps = load_gallery(maps)
    rep["gallery"] = {"size": len(gkeys), "pos_radius_m": pos_radius}
    res = {}
    per_q = {"a": [], "b": []}
    for name, nd in (("a", na), ("b", nb)):
        ranks, top1_angle, topk_sets, gaps = [], [], [], []
        for qi, (route, _, idx, gc) in enumerate(queries):
            sims = nd[qi] @ gv.T  # (V, G)
            best_v = int(np.argmax(sims.max(1)))  # RotationSelector: view with best top score
            s = sims[best_v]
            order = np.argsort(-s)
            dist = haversine_m(gc[0], gc[1], ggps[:, 0], ggps[:, 1])
            pos = dist <= pos_radius
            if not pos.any():
                ranks.append(None)
            else:
                ranks.append(int(np.nonzero(pos[order])[0][0]) + 1)
            top1_angle.append(ANGLES[best_v])
            topk_sets.append(set(order[:top_k].tolist()))
            gaps.append(float(s[order[0]] - s[order[1]]))
            per_q[name].append({"rank": ranks[-1], "angle": ANGLES[best_v], "top1": gkeys[order[0]],
                                "top1_score": float(s[order[0]])})
        valid = [r for r in ranks if r is not None]
        res[name] = {
            "with_positive": len(valid),
            **{f"R@{k}": sum(r <= k for r in valid) / max(1, len(valid)) for k in (1, 5, top_k)},
            "rank_p50": pct(valid, 50), "rank_p95": pct(valid, 95),
            "gap_p50": pct(gaps, 50),
        }
        res[name]["_sets"], res[name]["_angles"] = topk_sets, top1_angle
    sa, sb = res["a"].pop("_sets"), res["b"].pop("_sets")
    aa, ab = res["a"].pop("_angles"), res["b"].pop("_angles")
    jac = [len(x & y) / len(x | y) for x, y in zip(sa, sb)]
    res["diff"] = {
        "top1_changed": sum(p["top1"] != q["top1"] for p, q in zip(per_q["a"], per_q["b"])),
        "best_angle_changed": sum(x != y for x, y in zip(aa, ab)),
        f"top{top_k}_jaccard_p50": pct(jac, 50), f"top{top_k}_jaccard_min": float(min(jac)),
        "rank_worse": sum(1 for p, q in zip(per_q["a"], per_q["b"])
                          if p["rank"] and q["rank"] and q["rank"] > p["rank"]),
        "rank_better": sum(1 for p, q in zip(per_q["a"], per_q["b"])
                           if p["rank"] and q["rank"] and q["rank"] < p["rank"]),
        "lost_from_topk": sum(1 for p, q in zip(per_q["a"], per_q["b"])
                              if p["rank"] and p["rank"] <= top_k and (not q["rank"] or q["rank"] > top_k)),
        "gained_into_topk": sum(1 for p, q in zip(per_q["a"], per_q["b"])
                                if q["rank"] and q["rank"] <= top_k and (not p["rank"] or p["rank"] > top_k)),
    }
    per_route = {}
    for qi, q in enumerate(queries):
        r = per_route.setdefault(q[0], {"n": 0, "a_R1": 0, "b_R1": 0})
        r["n"] += 1
        r["a_R1"] += per_q["a"][qi]["rank"] == 1
        r["b_R1"] += per_q["b"][qi]["rank"] == 1
    res["per_route_R1"] = per_route
    rep["retrieval"] = res

    # 3. runtime / memory
    rt = {}
    for name, X, m in (("a", A, ma), ("b", B, mb)):
        ms = X["ms"][1:] if len(X["ms"]) > 1 else X["ms"]
        rt[name] = {"ms_per_query_4views_p50": pct(ms, 50), "p95": pct(ms, 95), "p99": pct(ms, 99),
                    "peak_alloc_mib": m["peak_alloc_mib"], "peak_reserved_mib": m["peak_reserved_mib"],
                    "param_mib": m["param_bytes"] / 2**20}
    rt["speedup_p50"] = rt["a"]["ms_per_query_4views_p50"] / rt["b"]["ms_per_query_4views_p50"]
    rep["runtime"] = rt
    (outdir / "report.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str), encoding="utf-8")

    L = [f"# DINO precision A/B — {datetime.now():%Y-%m-%d %H:%M}", "",
         f"A = fp16_enabled=false, B = fp16_enabled=true. Queries: {len(queries)} × {len(ANGLES)} views. "
         f"Gallery {len(gkeys)} frames, positive radius {pos_radius} m.", "",
         f"Setup A: {ma}", "", f"Setup B: {mb}", "",
         "## Numerics", "```", json.dumps(num, indent=1, ensure_ascii=False, default=str), "```",
         "## Retrieval (best of 4 views, cosine vs unchanged gallery)", "",
         "| | A fp32 | B amp |", "|---|---|---|"]
    for k in res["a"]:
        L.append(f"| {k} | {res['a'][k]:.4g} | {res['b'][k]:.4g} |" if isinstance(res["a"][k], float)
                 else f"| {k} | {res['a'][k]} | {res['b'][k]} |")
    L += ["", "Paired differences:", "```", json.dumps(res["diff"], indent=1), "```",
          "Per route R@1 counts:", "```", json.dumps(per_route, indent=1, ensure_ascii=False), "```",
          "## Runtime / memory (CUDA events, 1 query = 4 views)", "",
          "| | A fp32 | B amp |", "|---|---|---|"]
    for k in rt["a"]:
        L.append(f"| {k} | {rt['a'][k]:.1f} | {rt['b'][k]:.1f} |")
    L += ["", f"speedup p50 = **{rt['speedup_p50']:.2f}×**", "",
          "Not covered: YOLO mask, scale pyramid, local matching, geometry, filter. "
          "Route accuracy → tracking_routes.ps1 A/B."]
    (outdir / "report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", type=Path, default=DEF_MAPS)
    ap.add_argument("--sim", type=Path, default=DEF_SIM)
    ap.add_argument("--routes", nargs="*", help="base heading winter wayback_2019-04-03 yawsweep z17 (default all)")
    ap.add_argument("--every", type=int, default=30, help="take every N-th GT frame")
    ap.add_argument("--pos-radius-m", type=float, default=100.0)
    ap.add_argument("--top-k", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--compare-only", type=Path)
    ap.add_argument("--child", choices=["a", "b"], help=argparse.SUPPRESS)
    ap.add_argument("--queries", type=Path, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        child(args.child, args.queries, args.out, args.warmup)
        return
    if args.compare_only:
        analyse(args.compare_only, args.maps, args.pos_radius_m, args.top_k)
        return

    outdir = ROOT / "logs" / f"dino_ab_{datetime.now():%Y%m%d_%H%M%S}"
    outdir.mkdir(parents=True, exist_ok=True)
    routes = route_dirs(args.sim, args.routes)
    if not routes:
        sys.exit(f"no routes with video.frames.jsonl in {args.sim}")
    q = query_list(routes, args.every)
    (outdir / "queries.json").write_text(json.dumps(q), encoding="utf-8")
    print(f"{len(q)} queries from {[r[0] for r in routes]} -> {outdir}")
    for mode in ("a", "b"):
        rc = subprocess.call([sys.executable, __file__, "--child", mode, "--queries", str(outdir / "queries.json"),
                              "--out", str(outdir / f"{mode}.npz"), "--warmup", str(args.warmup)], cwd=ROOT)
        if rc:
            sys.exit(f"child {mode} failed with exit {rc}")
    analyse(outdir, args.maps, args.pos_radius_m, args.top_k)


if __name__ == "__main__":
    main()
