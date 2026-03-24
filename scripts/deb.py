"""Діагностика: перевірити точність координат на якорях.
Якірний кадр повинен давати GPS = GPS якоря (бо affine відкалібрований саме для нього).
"""

import sys

sys.path.insert(0, r"e:\Dip\gsdfg\New\DroneLocalization")

import json

import h5py
import numpy as np

from src.geometry.coordinates import CoordinateConverter
from src.geometry.transformations import GeometryTransforms

DB_PATH = r"C:\Users\horyn\OneDrive\Desktop\big\newnew.h5"
CALIB_PATH = r"C:\Users\horyn\OneDrive\Desktop\big\newZap_calibration.json"

# 1. Завантажити калібрацію
with open(CALIB_PATH) as f:
    calib = json.load(f)

ref_gps = calib.get("reference_gps")
print(f"Reference GPS: {ref_gps}")
if ref_gps:
    CoordinateConverter.gps_to_metric(ref_gps[0], ref_gps[1])

# 2. Завантажити HDF5
with h5py.File(DB_PATH, "r") as db:
    frame_affine = db["calibration"]["frame_affine"][:]
    frame_valid = db["calibration"]["frame_valid"][:].astype(bool)

    meta = dict(db.attrs)
    frame_w = meta.get("frame_width", 1920)
    frame_h = meta.get("frame_height", 1080)

    print(f"\nFrame size: {frame_w}x{frame_h}")
    print(f"Valid frames: {np.sum(frame_valid)}/{len(frame_valid)}")

    # 3. Перевірити кожний якір
    anchors = calib.get("anchors", [])
    print(f"\n{'=' * 60}")
    print(f"Перевірка {len(anchors)} якорів:")
    print(f"{'=' * 60}")

    for i, anc in enumerate(anchors):
        fid = anc["frame_id"]
        M_anchor = np.array(anc["affine_matrix"], dtype=np.float32)

        # Центр кадру якоря в пікселях
        center_px = np.array([[frame_w / 2.0, frame_h / 2.0]], dtype=np.float32)

        # Через anchor.affine_matrix (pixel → metric)
        metric_via_anchor = GeometryTransforms.apply_affine(center_px, M_anchor)[0]
        lat_a, lon_a = CoordinateConverter.metric_to_gps(metric_via_anchor[0], metric_via_anchor[1])

        # Через frame_affine (pixel → metric)
        M_prop = frame_affine[fid]
        metric_via_prop = GeometryTransforms.apply_affine(center_px, M_prop)[0]
        lat_p, lon_p = CoordinateConverter.metric_to_gps(metric_via_prop[0], metric_via_prop[1])

        # Різниця
        diff_m = np.linalg.norm(metric_via_anchor - metric_via_prop)

        print(f"\nЯкір {i}: frame {fid}")
        print(f"  anchor.affine → GPS: ({lat_a:.6f}, {lon_a:.6f})")
        print(f"  frame_affine  → GPS: ({lat_p:.6f}, {lon_p:.6f})")
        print(f"  Δ metric: {diff_m:.2f} m")
        print(f"  anchor M: {M_anchor}")
        print(f"  propagated M: {M_prop}")

    # 4. Перевірити кілька non-anchor frames
    print(f"\n{'=' * 60}")
    print("Приклади non-anchor frames (кожен 500-й):")
    print(f"{'=' * 60}")
    for fid in range(0, len(frame_valid), 500):
        if frame_valid[fid]:
            M = frame_affine[fid]
            center_px = np.array([[frame_w / 2.0, frame_h / 2.0]], dtype=np.float32)
            metric_pt = GeometryTransforms.apply_affine(center_px, M)[0]
            lat, lon = CoordinateConverter.metric_to_gps(metric_pt[0], metric_pt[1])
            print(f"  frame {fid}: ({lat:.6f}, {lon:.6f})")
