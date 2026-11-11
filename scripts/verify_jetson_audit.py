#!/usr/bin/env python
"""Verify the claims of the "DroneLocalization: Jetson optimization" audit
against THIS repository, and measure the GPU-side numbers it left open.

Run from the repo root (Windows, inside .venv):

    python scripts/verify_jetson_audit.py                       # static + GPU (synthetic image)
    python scripts/verify_jetson_audit.py --db path/to/db.h5 --video flight.mp4
    python scripts/verify_jetson_audit.py --db db.h5 --repack-test   # also writes chunk=1 copy to logs/
    python scripts/verify_jetson_audit.py --static-only

Sections
  S  static checks: code patterns + effective config (pydantic defaults + user_config.json)
  H  HDF5: chunk layout, chunk size in MiB, random single-frame read latency, optional lossless repack
  G  GPU: DINO dtype/params/batch timing (CUDA events), ALIKED timing, GPU->CPU->GPU roundtrip cost,
     LightGlue mp=False vs mp=True (matches + time), query keypoint caps 4096/2048/1024,
     MNN similarity-matrix memory, peak VRAM.

Nothing in the project is modified. Report: logs/jetson_audit_<timestamp>/report.{md,json}
Each check: CONFIRMED (audit claim holds here) / REFUTED (does not hold) / INFO (measured value) /
SKIP (precondition missing) / ERROR.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RESULTS: list[dict] = []


def rec(cid, status, claim, detail=""):
    RESULTS.append({"id": cid, "status": status, "claim": claim, "detail": str(detail)})
    print(f"[{status:9}] {cid:5} {claim}" + (f"\n            -> {detail}" if detail else ""))


def guarded(cid, claim):
    def deco(fn):
        def run(*a, **k):
            try:
                return fn(*a, **k)
            except Exception as e:  # noqa: BLE001
                rec(cid, "ERROR", claim, f"{type(e).__name__}: {e}")
                traceback.print_exc()
        return run
    return deco


def src(rel):
    p = ROOT / rel
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else None


def func_body(text, name):
    """Source of a def (first match) using AST; None if absent."""
    if text is None:
        return None
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return ast.get_source_segment(text, node)
    return None


# ───────────────────────────── S: effective config ─────────────────────────────
CONFIG_CLASSES = {
    # dotted prefix -> (file, class)
    "localization": ("config/localization.py", "LocalizationConfig"),
    "tracking": ("config/localization.py", "TrackingConfig"),
    "database": ("config/database.py", "DatabaseConfig"),
    "models.aliked": ("config/models.py", "AlikedSettings"),
    "models.lightglue": ("config/models.py", "LightGlueSettings"),
    "models.performance": ("config/models.py", "PerformanceConfig"),
    "global_descriptor.dinov3": ("config/models.py", "Dinov3ModelConfig"),
}

KEYS = [
    ("localization.temporal_candidate_prior", False),
    ("localization.temporal_prior_audit_every", 10),
    ("localization.temporal_prior_max_keypoints", 0),
    ("localization.temporal_prior_yaw_hint_tol_deg", None),
    ("localization.recovery_cascade", False),
    ("localization.candidate_prefilter", None),
    ("localization.prefilter_keep", 2),
    ("localization.prefilter_fallback_keep", None),
    ("localization.retrieval_top_k", 12),
    ("localization.max_local_edge", 1600),
    ("localization.scale_use_depth_hint", True),
    ("localization.layer_search.enabled", None),
    ("tracking.keyframe_interval", 30),
    ("tracking.keyframe_retry_every", 1),
    ("tracking.process_fps", None),
    ("database.hdf5_chunk_frames", 64),
    ("database.max_keypoints_stored", 2048),
    ("models.aliked.max_keypoints", 4096),
    ("models.lightglue.mixed_precision", False),
    ("models.lightglue.flash", True),
    ("models.performance.fp16_enabled", True),
    ("models.performance.global_batch_max", 0),
    ("models.performance.torch_compile", None),
    ("global_descriptor.backend", None),
    ("global_descriptor.dinov3.input_size", 224),
]


def pydantic_defaults(file, cls):
    text = src(file)
    out = {}
    if text is None:
        return out
    for node in ast.walk(ast.parse(text)):
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for st in node.body:
                if isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value:
                    try:
                        out[st.target.id] = ast.literal_eval(st.value)
                    except Exception:  # noqa: BLE001
                        out[st.target.id] = ast.unparse(st.value)
    return out


def default_for(key):
    best = None
    for pre, (f, c) in CONFIG_CLASSES.items():
        if key.startswith(pre + ".") and (best is None or len(pre) > len(best[0])):
            best = (pre, f, c)
    if not best:
        return "<unknown>"
    rest = key[len(best[0]) + 1:]
    d = pydantic_defaults(best[1], best[2])
    return d.get(rest, "<not a field>") if "." not in rest else "<nested>"


def user_value(cfg, key):
    cur = cfg
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return "<unset>"
        cur = cur[part]
    return cur


def section_config():
    print("\n=== S0 effective config ===")
    uc_path = ROOT / "user_config.json"
    uc = json.loads(uc_path.read_text(encoding="utf-8")) if uc_path.exists() else {}
    try:
        from config import APP_CONFIG, get_cfg  # real merged config if importable
    except Exception:  # noqa: BLE001
        APP_CONFIG, get_cfg = None, None
    rows = []
    for key, audit_default in KEYS:
        d = default_for(key)
        u = user_value(uc, key)
        eff = get_cfg(APP_CONFIG, key, None) if get_cfg else (d if u == "<unset>" else u)
        rows.append((key, audit_default, d, u, eff))
        if audit_default is not None and d not in ("<not a field>", "<nested>", "<unknown>"):
            st = "CONFIRMED" if d == audit_default else "REFUTED"
            rec("S0", st, f"default {key} == {audit_default!r}", f"code default={d!r}, user={u!r}, effective={eff!r}")
            if eff != d and eff is not None:
                rec("S0e", "INFO", f"EFFECTIVE {key} differs from code default (audit assumed default)",
                    f"effective={eff!r} vs default={d!r}")
        else:
            rec("S0", "INFO", f"{key}", f"code default={d!r}, user={u!r}, effective={eff!r}")
    return rows


# ───────────────────────────── S: code-pattern checks ─────────────────────────────
def section_static():
    print("\n=== S1 code-pattern checks ===")
    fe = src("src/models/wrappers/feature_extractor.py")
    mt = src("src/localization/matcher.py")
    gv = src("src/localization/geometric_verifier.py")
    rs = src("src/localization/rotation_selector.py")
    mm = src("src/models/model_manager.py")
    dl = src("src/database/database_loader.py")
    tw = src("src/workers/tracking_worker.py")
    vs = src("src/video/video_source.py")
    tel = src("src/utils/telemetry.py")
    trt = src("src/models/wrappers/trt_dinov2_wrapper.py")
    hp = src("src/utils/hardware_profile.py")
    loc = src("src/localization/localizer.py")

    b = func_body(fe, "extract_local_features") or ""
    rec("S1", "CONFIRMED" if ".cpu().numpy()" in b else "REFUTED",
        "extract_local_features returns query features via .cpu().numpy()")
    rec("S2", "CONFIRMED" if "nullcontext" in b and "autocast" not in b else "REFUTED",
        "ALIKED runs under nullcontext (no AMP)")

    b = func_body(mt, "_lightglue_match") or ""
    rec("S3", "CONFIRMED" if "from_numpy" in b and ".to(" in b else "REFUTED",
        "_lightglue_match re-creates tensors from NumPy and uploads per candidate")

    b = func_body(dl, "get_local_features") or ""
    m = re.search(r"len\(self\._feature_cache\)\s*>=\s*(\d+)", b)
    rec("S4", "CONFIRMED" if m and m.group(1) == "200" else "REFUTED",
        "CPU LRU cache = fixed 200 frames per DB (count, not bytes)", m.group(0) if m else "pattern absent")
    rec("S5", "CONFIRMED" if re.search(r"float32|astype\(np\.float32\)|\.float\(\)", b) else "REFUTED",
        "get_local_features converts descriptors to float32 in cache")

    b = func_body(dl, "open_maybe_encrypted_h5") or ""
    rec("S6", "CONFIRMED" if "BytesIO" in b else "REFUTED", "encrypted DB is fully decrypted into BytesIO")

    b = func_body(gv, "_prefilter") or ""
    has_remainder = bool(re.search(r"remainder|candidates\[self\.prefilter_keep", b))
    rec("S7", "CONFIRMED" if "scored[: self.prefilter_keep]" in b and not has_remainder else "REFUTED",
        "MNN prefilter truncates to prefilter_keep with no remainder fallback",
        "note: fallback_keep only applies when ALL MNN scores are 0" if "fallback_keep" in b else "")

    b = func_body(rs, "select") or ""
    gated = re.search(r"if best_global_score >= rescan_min:\s*\n\s*break", b)
    rec("S8", "CONFIRMED" if gated else "REFUTED",
        "recovery cascade stops on retrieval score BEFORE geometric verification")

    b = func_body(mm, "load_dinov2") or ""
    i3, ic = b.find('"dinov3"'), b.find("torch.compile")
    ret_between = i3 >= 0 and ic > i3 and "return self.models[name]" in b[i3:ic]
    rec("S9", "CONFIRMED" if ret_between else "REFUTED",
        "DINOv3 returns from load_dinov2 before TRT/torch.compile branch")
    rec("S10", "CONFIRMED" if ".half()" not in b[i3:ic] and "torch_dtype" not in b[i3:ic] else "REFUTED",
        "DINOv3 loader sets no explicit dtype / .half()")

    if trt:
        t = ast.parse(trt)
        cls = next((n for n in ast.walk(t) if isinstance(n, ast.ClassDef) and n.name == "TensorRTDINOv2Wrapper"), None)
        has_call = cls and any(isinstance(x, ast.FunctionDef) and x.name == "__call__" for x in cls.body)
        bases = [ast.unparse(x) for x in cls.bases] if cls else []
        rec("S11", "CONFIRMED" if cls and not has_call and not any("Module" in x for x in bases) else "REFUTED",
            "TensorRTDINOv2Wrapper has no __call__ and is not nn.Module", f"bases={bases}")
    else:
        rec("S11", "SKIP", "TensorRTDINOv2Wrapper", "file missing")

    b = func_body(mm, "load_lightglue") or ""
    rec("S12", "CONFIRMED" if "InferenceSession" in b else "REFUTED",
        "LightGlue .onnx path creates onnxruntime.InferenceSession (incompatible with model(data))")

    if tw:
        n = len(re.findall(r"self\.process_fps\b", tw))
        rec("S13", "CONFIRMED" if n <= 1 else "REFUTED",
            "tracking.process_fps is assigned but never used for scheduling", f"occurrences of self.process_fps={n}")
    rec("S14", "CONFIRMED" if vs and "cv2.VideoCapture(" in vs and "nvv4l2decoder" not in vs else "REFUTED",
        "VideoSource uses plain cv2.VideoCapture, no Jetson GStreamer pipeline")
    rec("S15", "CONFIRMED" if tel and "cuda" not in tel.lower() else "REFUTED",
        "Telemetry.profile is CPU wall-clock without CUDA events")
    rec("S16", "INFO", "HardwareProfile tiering uses GPU memory",
        "mentions jetson/tegra" if hp and re.search(r"jetson|tegra", hp, re.I) else "no jetson/tegra handling")
    if loc:
        rec("S17", "INFO", "layer_search branch exists (_localize_layers)",
            "_localize_layers present" if "_localize_layers" in loc else "absent")
    fm_init = ""
    for node in ast.walk(ast.parse(mt)):
        if isinstance(node, ast.ClassDef) and node.name == "FeatureMatcher":
            for x in node.body:
                if isinstance(x, ast.FunctionDef) and x.name == "__init__":
                    fm_init = ast.get_source_segment(mt, x)
    hit = re.search(r'"rdd" if local_extractor == "rdd" else "aliked"', fm_init)
    rec("S18", "CONFIRMED" if hit else "REFUTED",
        "FeatureMatcher.__init__ loads ALIKED-LightGlue for every extractor except rdd (incl. xfeat)")


# ───────────────────────────── H: HDF5 ─────────────────────────────
def section_hdf5(db, repack, n_reads, outdir):
    print("\n=== H HDF5 ===")
    try:
        import h5py
        import numpy as np
    except ImportError as e:
        rec("H0", "SKIP", "h5py not installed", e)
        return
    from src.database.database_loader import open_maybe_encrypted_h5

    f, _buf = open_maybe_encrypted_h5(str(db))
    big = []

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 2:
            ch = obj.chunks
            cmib = (np.prod(ch) * obj.dtype.itemsize / 2**20) if ch else None
            per = np.prod(obj.shape[1:]) * obj.dtype.itemsize / 2**20
            big.append((name, obj.shape, obj.dtype, ch, obj.compression, cmib, per))
    f.visititems(visit)
    for name, shp, dt, ch, comp, cmib, per in big:
        rec("H1", "INFO", f"dataset {name}",
            f"shape={shp} dtype={dt} chunks={ch} comp={comp} chunk={cmib and round(cmib,2)} MiB, one frame={per:.3f} MiB")
    desc = [b for b in big if b[0].endswith("descriptors") and len(b[1]) == 3]
    if not desc:
        rec("H2", "SKIP", "local descriptors dataset not found")
        return
    name, shp, dt, ch, comp, cmib, per = max(desc, key=lambda x: x[1][0])
    rec("H2", "CONFIRMED" if ch and ch[0] > 1 else "REFUTED",
        f"local descriptors chunked by >1 frame ({name})", f"chunk={ch}, {cmib:.2f} MiB vs frame {per:.3f} MiB")

    def bench(ds, label):
        N = ds.shape[0]
        ids = [random.randrange(N) for _ in range(n_reads)]
        t0 = time.perf_counter()
        for i in ids:
            _ = ds[i]
        dt_ = (time.perf_counter() - t0) / len(ids) * 1e3
        rec("H3", "INFO", f"random single-frame read [{label}]", f"{dt_:.2f} ms/frame (warm-ish, {len(ids)} reads)")
        return ids

    ids = bench(f[name], "original")
    if repack:
        out = outdir / f"repack_chunk1_{Path(db).parent.name}.h5"
        with h5py.File(out, "w") as g:
            def copy(n, o):
                if isinstance(o, h5py.Dataset):
                    chunks = (1,) + o.shape[1:] if o.ndim >= 2 and o.chunks else o.chunks
                    d = g.create_dataset(n, data=o[...], chunks=chunks, compression=o.compression,
                                         compression_opts=o.compression_opts, shuffle=o.shuffle)
                    for k, v in o.attrs.items():
                        d.attrs[k] = v
                elif isinstance(o, h5py.Group):
                    grp = g.require_group(n)
                    for k, v in o.attrs.items():
                        grp.attrs[k] = v
            for k, v in f.attrs.items():
                g.attrs[k] = v
            f.visititems(copy)
        with h5py.File(out, "r") as g:
            bad = []
            def cmp(n, o):
                if isinstance(o, h5py.Dataset) and not np.array_equal(o[...], g[n][...], equal_nan=o.dtype.kind == "f"):
                    bad.append(n)
            f.visititems(cmp)
            rec("H4", "CONFIRMED" if not bad else "REFUTED", "repack chunk=1 is lossless (all datasets equal)",
                f"mismatch: {bad}" if bad else str(out))
            t0 = time.perf_counter()
            for i in ids:
                _ = g[name][i]
            rec("H3", "INFO", "random single-frame read [chunk=1]",
                f"{(time.perf_counter()-t0)/len(ids)*1e3:.2f} ms/frame (file in OS cache — rerun cold for fair number)")
    f.close()


# ───────────────────────────── G: GPU ─────────────────────────────
def section_gpu(args, outdir):
    print("\n=== G GPU ===")
    import numpy as np
    import torch
    if not torch.cuda.is_available():
        rec("G0", "SKIP", "CUDA not available")
        return
    dev = torch.device("cuda")
    rec("G0", "INFO", "GPU", f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, "
        f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

    def cuda_time(fn, reps=args.reps, warm=3):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        ts = []
        for _ in range(reps):
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        ts.sort()
        return ts[len(ts) // 2], ts[int(len(ts) * 0.95) - 1 if len(ts) > 1 else 0]

    # image
    import cv2
    img = None
    if args.image:
        img = cv2.imread(str(args.image))
    elif args.video:
        cap = cv2.VideoCapture(str(args.video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
        ok, img = cap.read(); cap.release()
    if img is None:
        rng = np.random.default_rng(0)
        img = cv2.GaussianBlur((rng.random((1080, 1920, 3)) * 255).astype(np.uint8), (0, 0), 3)
        rec("G1", "INFO", "input", "SYNTHETIC noise image — timings OK, match counts meaningless; pass --video/--image")
    else:
        rec("G1", "INFO", "input", f"{img.shape}")

    from config import APP_CONFIG, get_cfg
    from src.models.model_manager import ModelManager
    from src.models.wrappers.feature_extractor import FeatureExtractor
    from src.localization.matcher import FeatureMatcher

    torch.cuda.reset_peak_memory_stats()
    mm = ModelManager(config=APP_CONFIG)
    t0 = time.perf_counter(); g = mm.load_dinov2(); t_g = time.perf_counter() - t0
    t0 = time.perf_counter(); l = mm.load_local_extractor(); t_l = time.perf_counter() - t0
    rec("G2", "INFO", "load time", f"global {t_g:.1f}s, local {t_l:.1f}s")

    mod = getattr(g, "model", g)
    try:
        ps = list(mod.parameters())
        dtypes = sorted({str(p.dtype) for p in ps})
        n = sum(p.numel() for p in ps)
        byt = sum(p.numel() * p.element_size() for p in ps)
        rec("G3", "INFO", "global model params / weight dtype", f"{n/1e6:.1f} M params, dtypes={dtypes}, {byt/2**30:.2f} GiB")
        rec("G3b", "CONFIRMED" if dtypes == ["torch.float32"] else "REFUTED",
            "DINOv3 weights stay FP32 (autocast only at runtime)", dtypes)
    except Exception as e:  # noqa: BLE001
        rec("G3", "SKIP", "global params", e)
    rec("G4", "INFO", "VRAM after model load",
        f"alloc {torch.cuda.memory_allocated()/2**20:.0f} MiB, reserved {torch.cuda.memory_reserved()/2**20:.0f} MiB")

    fe = FeatureExtractor(l, g, mm.device, config=APP_CONFIG)

    # DINO batch scaling (= rotation/scale hypotheses)
    for B in args.batches:
        imgs = [img] * B
        try:
            torch.cuda.reset_peak_memory_stats()
            p50, p95 = cuda_time(lambda: fe.extract_global_descriptors_multi(imgs))
            rec("G5", "INFO", f"DINO global, batch={B}",
                f"p50 {p50:.1f} ms ({p50/B:.1f} ms/img), p95 {p95:.1f} ms, peak {torch.cuda.max_memory_allocated()/2**20:.0f} MiB")
        except Exception as e:  # noqa: BLE001
            rec("G5", "ERROR", f"DINO batch={B}", e)
            torch.cuda.empty_cache()

    # ALIKED
    p50, p95 = cuda_time(lambda: fe.extract_local_features(img))
    q = fe.extract_local_features(img)
    nkp = len(q["keypoints"])
    rec("G6", "INFO", "ALIKED extract_local_features", f"p50 {p50:.1f} ms, p95 {p95:.1f}, kps={nkp}, desc dtype={q['descriptors'].dtype}")

    # GPU->CPU->GPU roundtrip that the audit wants to remove
    kp_t = torch.from_numpy(q["keypoints"]).float().to(dev)
    de_t = torch.from_numpy(q["descriptors"]).float().to(dev)
    def roundtrip():
        a = kp_t.cpu().numpy(); b = de_t.cpu().numpy()
        torch.from_numpy(a).to(dev); torch.from_numpy(b).to(dev)
    p50, _ = cuda_time(roundtrip)
    rec("G7", "INFO", "query GPU->NumPy->GPU roundtrip (per candidate re-upload cost)",
        f"{p50:.3f} ms; x top_k={get_cfg(APP_CONFIG,'localization.retrieval_top_k',12)} candidates")

    # reference features
    ref = None
    if args.db:
        try:
            from src.database.database_loader import DatabaseLoader
            db = DatabaseLoader(str(args.db))
            fid = args.ref_frame if args.ref_frame is not None else random.randrange(db.get_num_frames())
            t0 = time.perf_counter(); ref = db.get_local_features(fid); t_r = (time.perf_counter() - t0) * 1e3
            rec("G8", "INFO", f"DB get_local_features({fid}) cold", f"{t_r:.2f} ms, kps={len(ref['keypoints'])}, dtype={ref['descriptors'].dtype}")
        except Exception as e:  # noqa: BLE001
            rec("G8", "ERROR", "DB reference load", e)
    if ref is None:
        rot = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        ref = fe.extract_local_features(cv2.resize(rot, (rot.shape[1] * 3 // 4, rot.shape[0] * 3 // 4)))
        rec("G8", "INFO", "reference", "self-pair (query rotated 90° + scaled 0.75)")

    matcher = FeatureMatcher(model_manager=mm, config=APP_CONFIG)
    lg = mm.load_lightglue_aliked()
    conf = getattr(lg, "conf", None)

    def run_match(qf):
        return matcher._lightglue_match(qf, ref, model=lg)

    def nmatch(res):
        try:
            return len(res[0])
        except Exception:  # noqa: BLE001
            return "?"

    for mp in (False, True):
        if conf is not None and hasattr(conf, "mp"):
            conf.mp = mp
        elif mp:
            rec("G9", "SKIP", "LightGlue mp toggle", "model.conf.mp not found"); break
        for cap in args.kp_caps:
            if cap > nkp:
                continue
            qf = {k: (v[:cap] if hasattr(v, "__len__") and len(v) == nkp else v) for k, v in q.items()}
            p50, p95 = cuda_time(lambda: run_match(qf))
            rec("G9", "INFO", f"LightGlue mp={mp} query_kps={cap} ref_kps={len(ref['keypoints'])}",
                f"p50 {p50:.1f} ms, p95 {p95:.1f}, matches={nmatch(run_match(qf))}")
    if conf is not None and hasattr(conf, "mp"):
        conf.mp = bool(get_cfg(APP_CONFIG, "models.lightglue.mixed_precision", False))

    # MNN similarity memory
    try:
        Q = torch.from_numpy(q["descriptors"]).float().to(dev)
        R = torch.from_numpy(np.asarray(ref["descriptors"], np.float32)).to(dev)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        def mnn():
            S = Q @ R.T
            a = S.argmax(1); b = S.argmax(0)
            return (b[a] == torch.arange(len(a), device=dev)).sum()
        p50, _ = cuda_time(mnn)
        rec("G10", "INFO", f"MNN full sim matrix {len(Q)}x{len(R)}",
            f"p50 {p50:.2f} ms, extra peak {(torch.cuda.max_memory_allocated()-base)/2**20:.1f} MiB "
            f"(theory FP32 {len(Q)*len(R)*4/2**20:.1f} MiB)")
    except Exception as e:  # noqa: BLE001
        rec("G10", "ERROR", "MNN", e)

    rec("G11", "INFO", "peak VRAM whole run", f"{torch.cuda.max_memory_reserved()/2**20:.0f} MiB reserved")


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, help="HDF5 database (.h5) for H-section and real reference features")
    ap.add_argument("--video", type=Path); ap.add_argument("--frame", type=int, default=300)
    ap.add_argument("--image", type=Path)
    ap.add_argument("--ref-frame", type=int)
    ap.add_argument("--repack-test", action="store_true", help="write lossless chunk=1 copy into report dir")
    ap.add_argument("--reads", type=int, default=200)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 20, 40])
    ap.add_argument("--kp-caps", type=int, nargs="+", default=[4096, 2048, 1024])
    ap.add_argument("--static-only", action="store_true")
    ap.add_argument("--project", type=Path, help="DroneLocalization project dir (reads project.json, all source DBs)")
    ap.add_argument("--repack-max-mib", type=float, default=400,
                    help="with --project --repack-test: only repack DBs up to this size")
    args = ap.parse_args()
    random.seed(0)

    outdir = ROOT / "logs" / f"jetson_audit_{datetime.now():%Y%m%d_%H%M%S}"
    outdir.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)

    guarded("S0", "config")(section_config)()
    guarded("S1", "static")(section_static)()
    if args.project:
        pj = json.loads((args.project / "project.json").read_text(encoding="utf-8"))
        srcs = [s for s in pj.get("video_sources", []) if s.get("enabled", True)]
        dbs = []
        for s in srcs:
            p = args.project / s["database_file"]
            if p.exists():
                dbs.append((s["source_id"], s.get("description", ""), p, Path(s.get("video_path", ""))))
            else:
                rec("P0", "SKIP", f"source {s['source_id']}", f"missing {p}")
        rec("P0", "INFO", "project", f"{pj.get('project_name')}: {len(dbs)} source DBs")
        for sid, desc, p, _ in dbs:
            print(f"\n--- source {sid} ({desc}) {p.stat().st_size/2**20:.0f} MiB ---")
            RESULTS.append({"id": "P1", "status": "INFO", "claim": f"source {sid} ({desc})", "detail": str(p)})
            # lossless repack only for DBs below --repack-max-mib (copies the whole file)
            rp = args.repack_test and p.stat().st_size / 2**20 <= args.repack_max_mib
            guarded("H", "hdf5")(section_hdf5)(p, rp, args.reads, outdir)
        if not args.db and dbs:
            main_src = next((d for d in dbs if d[0] == "main"), dbs[0])
            args.db = main_src[2]
        if not args.video and not args.image:
            tv = args.project / "test_videos" / "video.mp4"
            cand = [tv] + [d[3] for d in dbs]
            args.video = next((c for c in cand if c and c.is_file()), None)
        rec("P2", "INFO", "GPU section inputs", f"db={args.db}, video={args.video}")
    elif args.db:
        guarded("H", "hdf5")(section_hdf5)(args.db, args.repack_test, args.reads, outdir)
    else:
        rec("H0", "SKIP", "HDF5 section", "pass --db")
    if not args.static_only:
        guarded("G", "gpu")(section_gpu)(args, outdir)

    (outdir / "report.json").write_text(json.dumps(RESULTS, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = ["# Jetson audit verification", "", f"{datetime.now():%Y-%m-%d %H:%M}  args: `{' '.join(sys.argv[1:])}`", "",
             "| id | status | claim | detail |", "|---|---|---|---|"]
    for r in RESULTS:
        lines.append(f"| {r['id']} | {r['status']} | {r['claim']} | {r['detail'].replace('|','/')} |")
    (outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    cnt = {s: sum(r["status"] == s for r in RESULTS) for s in ("CONFIRMED", "REFUTED", "INFO", "SKIP", "ERROR")}
    print(f"\nSummary: {cnt}\nReport: {outdir/'report.md'}")


if __name__ == "__main__":
    main()
