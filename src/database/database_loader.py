import io
import json
import os
import shutil
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import h5py
import lancedb
import numpy as np

from src.geometry.coordinates import CoordinateConverter
from src.security.at_rest import (
    MAGIC,
    decrypt_bytes,
    get_passphrase,
    is_encrypted,
    wipe_file,
)
from src.utils.logging_utils import get_logger

logger = get_logger(__name__)


def open_maybe_encrypted_h5(path: str) -> tuple[h5py.File, io.BytesIO | None]:
    """Open an HDF5 database, transparently decrypting an at-rest-encrypted map.

    HARDENING P1-6 SP2: h5py needs a seekable source for the whole session. A
    plaintext DB is opened straight from the path (h5py reads lazily — no full
    load, zero overhead). An encrypted DB (`MAGIC` header) is decrypted whole into
    RAM and served from a ``BytesIO`` (67 MB fits comfortably); the buffer is
    returned so the caller can keep it alive for the file handle's lifetime.
    """
    with open(path, "rb") as f:
        head = f.read(len(MAGIC))
    if head != MAGIC:
        return h5py.File(path, "r"), None  # plaintext: unchanged lazy path
    plaintext = decrypt_bytes(Path(path).read_bytes(), get_passphrase())
    buf = io.BytesIO(plaintext)
    return h5py.File(buf, "r"), buf


_LANCE_TEMP_PREFIX = "droneloc_lance_"


def _owner_file(tmpdir: str) -> Path:
    """Sibling file holding the PID that owns a decrypted-index temp directory."""
    return Path(str(tmpdir) + ".owner")


def _pid_is_running(pid: int) -> bool | None:
    """True/False if the PID's liveness can be determined, None if it cannot.

    ``os.kill(pid, 0)`` is not usable here: on Windows it maps to TerminateProcess
    and would kill the very process we are probing. Without psutil we return None
    and the caller keeps the directory — never delete on a guess."""
    try:
        import psutil
    except ImportError:
        return None
    try:
        return psutil.pid_exists(pid)
    except Exception:
        return None


def sweep_stale_lance_tempdirs() -> int:
    """Wipe decrypted-index temp directories abandoned by crashed runs.

    ``DatabaseLoader.close`` wipes its own directory, but a hard kill or power
    loss skips it and leaves plaintext global descriptors on the temp disk. Called
    once at startup. Returns the number of directories wiped.

    Conservative by construction: a directory is removed only when its owner PID
    is known to be gone. An unreadable stamp, a live PID, or no way to check
    (psutil missing) all mean "leave it alone" — deleting a directory a
    concurrently running instance is reading from would break that instance."""
    wiped = 0
    for path in Path(tempfile.gettempdir()).glob(_LANCE_TEMP_PREFIX + "*"):
        if not path.is_dir():
            continue  # the .owner stamps themselves
        owner = _owner_file(str(path))
        try:
            pid = int(owner.read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue  # no readable stamp: not ours to judge
        if _pid_is_running(pid) is not False:
            continue  # alive, or undeterminable
        wipe_tree(str(path))
        wiped += 1
    if wiped:
        logger.warning(
            f"Wiped {wiped} decrypted LanceDB temp director(ies) left by a previous "
            f"run that did not shut down cleanly"
        )
    return wiped


def materialize_maybe_encrypted_lance(lance_path: Path) -> tuple[str, str | None]:
    """Return a path LanceDB can open, decrypting an at-rest-encrypted index first.

    HARDENING P1-6 SP3: LanceDB opens a *filesystem directory* and manages its own
    file handles, so there is no in-RAM route like the h5 one. A plaintext index is
    opened in place (unchanged path, zero overhead). An encrypted index is
    materialised into a temp directory, preserving the dataset layout.

    Returns ``(path_to_open, temp_dir_or_None)``; the caller must wipe the temp
    directory when closing. Fails closed: a failed decryption leaves no partial
    plaintext behind.
    """
    files = sorted(p for p in lance_path.rglob("*") if p.is_file())
    if not files:
        return str(lance_path), None
    if not is_encrypted(files[0].read_bytes()[: len(MAGIC)]):
        return str(lance_path), None  # plaintext: unchanged path

    passphrase = get_passphrase()
    tmpdir = tempfile.mkdtemp(prefix=_LANCE_TEMP_PREFIX)
    try:
        # Owner stamp, kept as a SIBLING file so LanceDB never sees a stray entry
        # in its dataset root. Lets a later run tell an abandoned directory (crash)
        # from one a concurrently running instance is still using.
        _owner_file(tmpdir).write_text(str(os.getpid()), encoding="utf-8")
        for src in files:
            target = Path(tmpdir) / src.relative_to(lance_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            data = src.read_bytes()
            target.write_bytes(decrypt_bytes(data, passphrase) if is_encrypted(data) else data)
    except Exception:
        wipe_tree(tmpdir)
        raise
    logger.info(f"Decrypted LanceDB index ({len(files)} files) into a temp directory")
    return tmpdir, tmpdir


def wipe_tree(dir_path: str) -> None:
    """Best-effort secure delete of a decrypted temp directory (see ``wipe_file``
    for the SSD/CoW caveat) — overwrite every file, then drop the tree and its
    owner stamp."""
    root = Path(dir_path)
    if root.exists():
        for path in root.rglob("*"):
            if path.is_file():
                wipe_file(str(path))
        shutil.rmtree(root, ignore_errors=True)
    _owner_file(dir_path).unlink(missing_ok=True)


def _synchronized(method):
    """Декоратор: виконує метод під self._lock (RLock — реентерабельний)."""
    import functools

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


class DatabaseLoader:
    """Loads and manages access to the HDF5 topometric database (XFeat + DINOv2)"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db_file: h5py.File | None = None
        # HARDENING P1-6 SP2: holds the decrypted-map RAM buffer (encrypted DBs
        # only), kept alive for the h5py handle's lifetime; None when plaintext.
        self._decrypted_buf: io.BytesIO | None = None
        # HARDENING P1-6 SP3: temp directory holding the decrypted LanceDB index
        # (encrypted projects only); wiped on close. None when plaintext.
        self._lance_tempdir: str | None = None
        self.global_descriptors: np.ndarray | None = None
        self.lance_table = None
        self.frame_poses: np.ndarray | None = None
        self.metadata: dict[str, Any] = {}
        self.converter: CoordinateConverter | None = None

        # Дані пропагації калібрування (заповнюються після калібрування)
        self.frame_affine: np.ndarray | None = None  # (N, 2, 3) — Metric Affine Matrices
        self.frame_valid: np.ndarray | None = None  # (N,)      — True якщо кадр має GPS
        self.frame_rmse: np.ndarray | None = None  # (N,)      — RMSE кожного кадру
        self.frame_disagreement: np.ndarray | None = None  # (N,)   — Розбіжність між гілками
        self.frame_matches: np.ndarray | None = None  # (N,)      — Кількість точок (inliers)
        self.depth_scales: np.ndarray | None = None  # (N,) — 1/median_depth per frame (GSD hint)

        # GPS-координати кадрів та просторовий індекс (мультиджерельна геолокалізація)
        self.frame_gps: np.ndarray | None = None  # (N, 2) — [lat, lon] per frame
        self.spatial_index = None  # SpatialIndex | None

        # Потокобезпека: h5py-хендл і кеші читаються з GUI-потоку та воркерів
        # одночасно (h5py не потокобезпечний, OrderedDict-кеш мутується).
        # RLock — бо публічні методи викликають один одного.
        # ЗОВНІШНІЙ код (напр. пропагація при перезаписі HDF5) може взяти
        # self.lock на весь цикл close → write → reload.
        self._lock = threading.RLock()

        # Каш для методів (заміна lru_cache для уникнення B019)
        self._size_cache: dict[int, tuple[int, int]] = {}
        self._feature_cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

        # Patchify: мультимасштабні дескриптори (None якщо БД не має їх)
        self.patch_descriptors: np.ndarray | None = None

        logger.info(f"Initializing DatabaseLoader | path={db_path}")
        self._load_hot_data()

    @property
    def lock(self) -> threading.RLock:
        """Публічний лок для зовнішніх критичних секцій (напр. перезапис HDF5)."""
        return self._lock

    @_synchronized
    def _load_hot_data(self) -> None:
        """Load global descriptors (DINOv2), poses and propagation data into RAM"""
        logger.info(f"Loading hot data into RAM from: {self.db_path}")

        try:
            self.db_file, self._decrypted_buf = open_maybe_encrypted_h5(self.db_path)
            logger.debug(f"HDF5 file opened | top-level groups: {list(self.db_file.keys())}")

            if "global_descriptors" not in self.db_file:
                raise KeyError(
                    f"HDF5 file is missing 'global_descriptors' group. "
                    f"Available groups: {list(self.db_file.keys())}. "
                    f"The database file may be corrupted or was created with an incompatible version."
                )

            lance_path = Path(self.db_path).parent / "vectors.lance"
            if lance_path.exists():
                logger.info(f"LanceDB index found at {lance_path}. Loading...")
                open_path, self._lance_tempdir = materialize_maybe_encrypted_lance(lance_path)
                db = lancedb.connect(open_path)
                self.lance_table = db.open_table("global_vectors")
                self.global_descriptors = None
            else:
                logger.info("LanceDB index not found. Falling back to HDF5 global descriptors.")
                self.global_descriptors = self.db_file["global_descriptors"]["descriptors"][:]

            self.frame_poses = self.db_file["global_descriptors"]["frame_poses"][:]

            if self.lance_table is not None:
                logger.info(f"Loaded LanceDB table with {self.lance_table.count_rows()} vectors.")
            else:
                logger.info(
                    f"Loaded global descriptors: shape={self.global_descriptors.shape}, "
                    f"dtype={self.global_descriptors.dtype}, "
                    f"mem={self.global_descriptors.nbytes / 1024**2:.1f} MB"
                )
            logger.info(f"Loaded frame poses: shape={self.frame_poses.shape}")

            for key, value in self.db_file["metadata"].attrs.items():
                self.metadata[key] = value
                logger.debug(f"Metadata — {key}: {value}")

            if "actual_num_frames" in self.metadata:
                actual_num = int(self.metadata["actual_num_frames"])
                total_slots = len(self.frame_poses)
                logger.info(
                    f"Database contains {actual_num} actual frames in {total_slots} pre-allocated slots"
                )
                # DO NOT SLICE with actual_num_frames! The arrays are sized to num_frames exactly,
                # and are indexed by absolute visual frame_id!

            if "frame_index_map" in self.db_file["metadata"]:
                self.frame_index_map = self.db_file["metadata"]["frame_index_map"][:]
                logger.debug(f"Frame index map loaded: {len(self.frame_index_map)} entries")
            else:
                total_len = len(self.frame_poses)
                self.frame_index_map = np.arange(total_len)
                logger.debug("No frame_index_map found — using sequential indices")

            # Завантажуємо патч-дескриптори якщо є (Patchify)
            if "patch_descriptors" in self.db_file:
                self.patch_descriptors = self.db_file["patch_descriptors"]["descriptors"][:]
                logger.info(f"Loaded patch descriptors: shape={self.patch_descriptors.shape}")
            else:
                self.patch_descriptors = None

            # Depth scales (1/median_depth per frame) — GSD hint for ScaleManager pyramid
            if "depth_scales" in self.db_file["metadata"]:
                self.depth_scales = self.db_file["metadata"]["depth_scales"][:]
            else:
                self.depth_scales = None

            # Завантажуємо frame_gps якщо є (мультиджерельна геолокалізація)
            if "frame_gps" in self.db_file:
                self.frame_gps = self.db_file["frame_gps"][:]
                # Перевіряємо чи є non-NaN значення
                valid_count = int(np.sum(~np.isnan(self.frame_gps[:, 0])))
                if valid_count > 0:
                    from src.database.spatial_index import SpatialIndex

                    self.spatial_index = SpatialIndex(self.frame_gps)
                    logger.info(
                        f"Loaded frame_gps: {valid_count}/{len(self.frame_gps)} "
                        f"frames with GPS. SpatialIndex built."
                    )
                else:
                    logger.info("frame_gps dataset exists but has no valid GPS values.")
                    self.frame_gps = None
            else:
                self.frame_gps = None

            # Завантажуємо дані пропагації якщо є
            self._load_propagation_data()

            logger.success(f"Hot data loaded successfully | {len(self.frame_poses)} frames")

        except KeyError as e:
            logger.error(
                f"Database structure error: {e} | path={self.db_path}. "
                f"This usually means the HDF5 file is incomplete or was created with a different version."
            )
            raise
        except OSError as e:
            logger.error(
                f"Cannot open database file: {e} | path={self.db_path}. "
                f"Check that the file exists and is not locked by another process."
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error loading database: {e} | path={self.db_path}", exc_info=True
            )
            raise

    def _load_propagation_data(self) -> None:
        if self.db_file is None or "calibration" not in self.db_file:
            logger.info("No propagation data in database (not calibrated yet)")
            self.frame_affine = None
            self.frame_valid = None
            return
        try:
            grp = self.db_file["calibration"]

            # 1. Відновлення проєкції (пріоритет)
            if "projection_json" in grp.attrs:
                try:
                    meta = json.loads(grp.attrs["projection_json"])
                    self.converter = CoordinateConverter.from_metadata(meta)
                    logger.success(f"Projection restored from HDF5: {meta['mode']}")
                except Exception as e:
                    logger.warning(
                        f"Could not load projection metadata: {e}. "
                        f"Raw value: {grp.attrs.get('projection_json', '<missing>')}. "
                        f"Falling back to default projection."
                    )
            elif "reference_gps" in grp.attrs:
                # Fallback для v2.0 (UTM)
                try:
                    ref_gps = json.loads(grp.attrs["reference_gps"])
                    self.converter = CoordinateConverter("UTM", tuple(ref_gps))
                    logger.success(f"UTM auto-initialized from legacy reference GPS: {ref_gps}")
                except Exception as e:
                    logger.warning(
                        f"Could not init UTM from legacy attribute: {e}. "
                        f"Raw reference_gps value: {grp.attrs.get('reference_gps', '<missing>')}. "
                        f"Defaulting to WEB_MERCATOR."
                    )
            else:
                # Fallback для v1.0 (WebMercator)
                logger.info("No projection metadata found. Defaulting to WEB_MERCATOR fallback.")
                self.converter = CoordinateConverter("WEB_MERCATOR")

            # 2. Завантаження датасетів
            if "frame_affine" in grp:
                self.frame_affine = grp["frame_affine"][:]
                self.frame_valid = grp["frame_valid"][:].astype(bool)

                # Метрики якості (QA)
                self.frame_rmse = grp["frame_rmse"][:] if "frame_rmse" in grp else None
                self.frame_disagreement = (
                    grp["frame_disagreement"][:] if "frame_disagreement" in grp else None
                )
                self.frame_matches = grp["frame_matches"][:] if "frame_matches" in grp else None

                valid_count = int(np.sum(self.frame_valid))
                logger.success(f"Propagation data loaded: {valid_count} frames valid")
            else:
                logger.warning("Found calibration group but no frame_affine dataset.")
                self.frame_affine = None
                self.frame_valid = None
        except Exception as e:
            logger.error(
                f"Failed to load propagation data: {e} | db_path={self.db_path}. "
                f"Calibration data may be corrupted — recalibration recommended.",
                exc_info=True,
            )
            self.frame_affine = None
            self.frame_valid = None

    @property
    def is_propagated(self) -> bool:
        return self.frame_affine is not None

    @property
    def median_depth_scale(self) -> float | None:
        """Median 1/median_depth across DB frames (GSD hint for ScaleManager), or None."""
        if self.depth_scales is None:
            return None
        vals = self.depth_scales[np.isfinite(self.depth_scales) & (self.depth_scales > 0)]
        return float(np.median(vals)) if vals.size else None

    @_synchronized
    def get_frame_affine(self, frame_id: int) -> np.ndarray | None:
        """Повертає афінну матрицю для конкретного кадру"""
        if not self.is_propagated or self.frame_affine is None or self.frame_valid is None:
            return None
        if frame_id < 0 or frame_id >= len(self.frame_valid):
            return None
        if not self.frame_valid[frame_id]:
            return None
        return self.frame_affine[frame_id]

    @_synchronized
    def get_frame_size(self, frame_id: int) -> tuple[int, int]:
        """Повертає (height, width) для вказаного кадру"""
        if frame_id in self._size_cache:
            return self._size_cache[frame_id]

        if self.db_file is None:
            return 1080, 1920

        # Нова схема v2: розміри збережені один раз в local_features.attrs
        schema = self.metadata.get("hdf5_schema", "v1")
        if schema == "v2" and "local_features" in self.db_file:
            lf_attrs = self.db_file["local_features"].attrs
            h = int(lf_attrs.get("frame_height", self.metadata.get("frame_height", 1080)))
            w = int(lf_attrs.get("frame_width", self.metadata.get("frame_width", 1920)))
            self._size_cache[frame_id] = (h, w)
            return h, w

        # Стара схема v1: fallback — читаємо з групи кадру (зворотня сумісність)
        group_name = f"local_features/frame_{frame_id}"
        if group_name in self.db_file:
            g = self.db_file[group_name]
            if "height" in g.attrs and "width" in g.attrs:
                h, w = int(g.attrs["height"]), int(g.attrs["width"])
            else:
                h = self.metadata.get("frame_height") or self.metadata.get("height") or 1080
                w = self.metadata.get("frame_width") or self.metadata.get("width") or 1920
        else:
            # Frame group not found — use global metadata fallback
            h = self.metadata.get("frame_height") or self.metadata.get("height") or 1080
            w = self.metadata.get("frame_width") or self.metadata.get("width") or 1920

        res = (int(h), int(w))
        self._size_cache[frame_id] = res
        return res

    @_synchronized
    def get_local_features(self, frame_id: int) -> dict[str, np.ndarray]:
        """Повертає локальні ознаки для вказаного кадру (сумісно з v1 і v2)"""
        if frame_id in self._feature_cache:
            self._feature_cache.move_to_end(frame_id)
            return self._feature_cache[frame_id]

        if self.db_file is None:
            raise RuntimeError("Database not opened")

        schema = self.metadata.get("hdf5_schema", "v1")
        if schema == "v2":
            lf = self.db_file["local_features"]
            n = int(lf["kp_counts"][frame_id])
            if n == 0:
                raise ValueError(f"Кадр {frame_id} не має keypoints (kp_count=0).")
            res = {
                "keypoints": lf["keypoints"][frame_id, :n],
                "descriptors": lf["descriptors"][frame_id, :n].astype("float32"),  # float16→32
                "coords_2d": lf["coords_2d"][frame_id, :n],
            }
        else:
            # Стара схема v1 — зворотня сумісність
            group_name = f"local_features/frame_{frame_id}"
            if group_name not in self.db_file:
                raise ValueError(f"Кадр {frame_id} не знайдено у базі даних.")
            g = self.db_file[group_name]
            res = {
                "keypoints": g["keypoints"][:],
                "descriptors": g["descriptors"][:],
                "coords_2d": g["coords_2d"][:],
            }

        # Додаємо image_size для коректної нормалізації у LightGlue
        h, w = self.get_frame_size(frame_id)
        res["image_size"] = np.array([h, w], dtype=np.int32)

        # LRU-витіснення
        if len(self._feature_cache) >= 200:
            self._feature_cache.popitem(last=False)

        self._feature_cache[frame_id] = res
        return res

    @property
    def has_sift_features(self) -> bool:
        """RESEARCH 2.2: чи містить БД SIFT-ознаки для аварійного фолбека."""
        try:
            return self.db_file is not None and "sift_features" in self.db_file
        except Exception:
            return False

    @_synchronized
    def get_sift_features(self, frame_id: int) -> dict[str, np.ndarray]:
        """RESEARCH 2.2: SIFT-ознаки кадру (rootSIFT, сумісні з LightGlue-sift).

        Без LRU-кешу: фолбек викликається рідко (лише при провалі ALIKED),
        кешування лише витісняло б гарячі ALIKED-ознаки з пам'яті.
        """
        if self.db_file is None:
            raise RuntimeError("Database not opened")
        if "sift_features" not in self.db_file:
            raise ValueError(
                "База не містить SIFT-ознак — перебудуйте з database.store_sift_features=True"
            )
        sf = self.db_file["sift_features"]
        n = int(sf["kp_counts"][frame_id])
        if n == 0:
            raise ValueError(f"Кадр {frame_id} не має SIFT keypoints (kp_count=0)")
        res = {
            "keypoints": sf["keypoints"][frame_id, :n],
            "descriptors": sf["descriptors"][frame_id, :n].astype("float32"),
            "image_size": np.array(self.get_frame_size(frame_id), dtype=np.int32),
        }
        return res

    def get_num_frames(self) -> int:
        """Повертає кількість кадрів у БД (pre-allocated slots для v2)."""
        return int(self.metadata.get("num_frames", 0))

    @_synchronized
    def close(self) -> None:
        if self.db_file is not None:
            self.db_file.close()
            self.db_file = None
            logger.info("Database file closed")

        # HARDENING P1-6 SP3: drop the LanceDB handle before wiping its files,
        # otherwise the open dataset keeps the decrypted temp copy on disk.
        if self._lance_tempdir is not None:
            self.lance_table = None
            wipe_tree(self._lance_tempdir)
            logger.info("Decrypted LanceDB temp directory wiped")
            self._lance_tempdir = None

        # Очищення кешу при закритті БД
        self._size_cache.clear()
        self._feature_cache.clear()
