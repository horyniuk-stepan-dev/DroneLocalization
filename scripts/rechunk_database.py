"""Rewrite map databases so each frame's local features are one HDF5 chunk.

Databases built before 2026-10-10 store local_features/{keypoints,descriptors,
coords_2d} in LZF chunks of 64 frames (database.hdf5_chunk_frames). Reading one
reference frame then decompresses the whole 64-frame chunk (~32 MiB of
descriptors), and h5py's 1 MiB chunk cache cannot keep it, so every cache miss
pays that again. Retrieval and rescan frames read up to 12 new candidates:
measured 2026-10-10, the MNN prefilter (which does those reads) took 220-330 ms
per call on the RTX 5070 Ti and 500-760 ms on the GTX 1650, half of all replay
time. With one frame per chunk a read is ~1 ms (68 ms before, same file size).

Only the layout changes: every dataset keeps its values, dtype, shape, maxshape,
compression, fill value and attributes, so the schema fingerprint and all
results stay the same. Datasets with at least --min-frame-bytes per frame and a
first dimension equal to the frame slots are rechunked (local features and the
optional SIFT features); everything else is copied as is.

Each database is written to <name>.rechunk.tmp next to it, compared with the
original dataset by dataset, then the original is kept as <name>.chunk<N> and
the new file takes its place. Close the app first (Windows locks open files).

    python scripts/rechunk_database.py --project "D:/maps/testtopboch_max_dino336"
    python scripts/rechunk_database.py path/to/database.h5 --dry-run
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import h5py  # noqa: E402
import numpy as np  # noqa: E402

COPY_BLOCK_FRAMES = 64


def _frame_slots(f: h5py.File) -> int | None:
    lf = f.get("local_features")
    if lf is None or "descriptors" not in lf:
        return None
    return int(lf["descriptors"].shape[0])


def plan(f: h5py.File, min_frame_bytes: int) -> list[str]:
    """Names of the datasets that would get one-frame chunks."""
    slots = _frame_slots(f)
    names: list[str] = []
    if slots is None:
        return names

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset) or obj.ndim < 2 or obj.shape[0] != slots:
            return
        per_frame = int(np.prod(obj.shape[1:])) * obj.dtype.itemsize
        if per_frame >= min_frame_bytes and obj.chunks is not None and obj.chunks[0] > 1:
            names.append(name)

    f.visititems(visit)
    return names


def _copy_attrs(src, dst) -> None:
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def rechunk_file(src_path: Path, dst_path: Path, names: list[str]) -> None:
    """Write ``src_path`` to ``dst_path`` with ``names`` rechunked to one frame."""
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "w", libver="latest") as dst:
        _copy_attrs(src, dst)

        def copy_group(src_group, dst_group):
            for key, obj in src_group.items():
                if isinstance(obj, h5py.Group):
                    child = dst_group.create_group(key)
                    _copy_attrs(obj, child)
                    copy_group(obj, child)
                elif obj.name.lstrip("/") in names:
                    new = dst_group.create_dataset(
                        key,
                        shape=obj.shape,
                        maxshape=obj.maxshape,
                        dtype=obj.dtype,
                        chunks=(1,) + tuple(obj.shape[1:]),
                        compression=obj.compression,
                        compression_opts=obj.compression_opts,
                        shuffle=obj.shuffle,
                        fletcher32=obj.fletcher32,
                        fillvalue=obj.fillvalue,
                    )
                    _copy_attrs(obj, new)
                    step = max(COPY_BLOCK_FRAMES, obj.chunks[0])
                    fill = obj.fillvalue
                    for start in range(0, obj.shape[0], step):
                        block = obj[start : start + step]
                        # Pre-allocated slots never written read as the fill value;
                        # leaving them unwritten keeps them unallocated (and small).
                        if not np.all(block == fill):
                            new[start : start + step] = block
                else:
                    src_group.copy(obj, dst_group, name=key)

        copy_group(src, dst)


def _attrs_equal(a, b) -> bool:
    if set(a.attrs) != set(b.attrs):
        return False
    return all(np.array_equal(np.asarray(a.attrs[k]), np.asarray(b.attrs[k])) for k in a.attrs)


def verify(src_path: Path, dst_path: Path) -> list[str]:
    """Differences between the two files (empty list = identical content)."""
    problems: list[str] = []
    with h5py.File(src_path, "r") as src, h5py.File(dst_path, "r") as dst:
        if not _attrs_equal(src, dst):
            problems.append("file attributes differ")

        def visit(name, obj):
            other = dst.get(name)
            if other is None or type(other) is not type(obj):
                problems.append(f"{name}: missing or different kind")
                return
            if not _attrs_equal(obj, other):
                problems.append(f"{name}: attributes differ")
            if isinstance(obj, h5py.Dataset):
                if obj.shape != other.shape or obj.dtype != other.dtype:
                    problems.append(f"{name}: shape/dtype differ")
                    return
                if obj.shape == ():
                    same = np.array_equal(obj[()], other[()])
                else:
                    step = max(COPY_BLOCK_FRAMES, (obj.chunks or (1,))[0])
                    same = all(
                        np.array_equal(obj[s : s + step], other[s : s + step], equal_nan=True)
                        if obj.dtype.kind in "fc"
                        else np.array_equal(obj[s : s + step], other[s : s + step])
                        for s in range(0, obj.shape[0], step)
                    )
                if not same:
                    problems.append(f"{name}: values differ")

        src.visititems(visit)

        extra = []
        dst.visit(lambda name: extra.append(name) if name not in src else None)
        problems += [f"{name}: only in the new file" for name in extra]
    return problems


def random_read_ms(path: Path, n: int = 12, seed: int = 0) -> float:
    """Median ms to read the local features of ``n`` random frames (5 rounds)."""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        lf = f["local_features"]
        counts = lf["kp_counts"][:] if "kp_counts" in lf else None
        frames = (
            np.flatnonzero(counts > 0) if counts is not None else np.arange(len(lf["descriptors"]))
        )
        rounds = []
        for _ in range(5):
            ids = rng.choice(frames, min(n, len(frames)), replace=False)
            t = time.perf_counter()
            for i in ids:
                k = int(counts[i]) if counts is not None else lf["descriptors"].shape[1]
                for key in ("keypoints", "descriptors", "coords_2d"):
                    if key in lf:
                        lf[key][i, :k]
            rounds.append((time.perf_counter() - t) * 1000.0)
    return float(np.median(rounds))


def _is_encrypted(path: Path) -> bool:
    from src.security.at_rest import MAGIC

    with open(path, "rb") as stream:
        return stream.read(len(MAGIC)) == MAGIC


def process(path: Path, min_frame_bytes: int, dry_run: bool, bench: bool) -> str:
    if _is_encrypted(path):
        return "encrypted database: not supported (decrypt it first)"
    with h5py.File(path, "r") as f:
        names = plan(f, min_frame_bytes)
        old_chunk = f[names[0]].chunks[0] if names else None
    if not names:
        return "already one frame per chunk (or no local features): nothing to do"
    if dry_run:
        return f"would rechunk (from {old_chunk} frames per chunk): {', '.join(names)}"
    backup = path.with_name(f"{path.name}.chunk{old_chunk}")
    tmp = path.with_name(f"{path.name}.rechunk.tmp")
    if backup.exists():
        return f"backup {backup.name} already exists: move it away first"
    before = random_read_ms(path) if bench else None
    started = time.perf_counter()
    rechunk_file(path, tmp, names)
    problems = verify(path, tmp)
    if problems:
        return (
            "verification failed, original untouched, kept "
            + tmp.name
            + ": "
            + "; ".join(problems[:5])
        )
    os.replace(path, backup)
    os.replace(tmp, path)
    msg = (
        f"rechunked {len(names)} datasets in {time.perf_counter() - started:.0f} s; "
        f"original kept as {backup.name}"
    )
    if bench:
        msg += f"; 12 random frames {before:.0f} -> {random_read_ms(path):.0f} ms"
    return msg


def find_databases(project: Path) -> list[Path]:
    return sorted(project.glob("sources/*/database.h5")) or sorted(project.glob("**/database.h5"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("databases", nargs="*", type=Path, help="database.h5 files")
    parser.add_argument("--project", type=Path, help="project folder: every sources/*/database.h5")
    parser.add_argument("--dry-run", action="store_true", help="only report what would change")
    parser.add_argument(
        "--bench", action="store_true", help="time 12 random frame reads before/after"
    )
    parser.add_argument(
        "--min-frame-bytes",
        type=int,
        default=4096,
        help="rechunk datasets with at least this many bytes per frame (default 4096)",
    )
    args = parser.parse_args(argv)
    paths = list(args.databases)
    if args.project is not None:
        paths += find_databases(args.project)
    if not paths:
        parser.error("give database files or --project")
    failed = 0
    for path in paths:
        if not path.is_file():
            print(f"{path}: not found")
            failed += 1
            continue
        result = process(path, args.min_frame_bytes, args.dry_run, args.bench)
        print(f"{path}: {result}")
        failed += result.startswith(("verification failed", "backup", "encrypted"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
