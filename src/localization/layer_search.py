"""Geometry-first multi-source search and a separately testable handoff policy.

All scale estimates are image ratios in normalized query/reference coordinates.
Geographic comparisons use geodesic metres, never two sources' raw map units.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np
from pyproj import Geod

from config import get_cfg
from src.geometry.calibration_provenance import CalibrationOrigin, GeoreferenceStatus
from src.geometry.transformations import GeometryTransforms
from src.localization.rotation_selector import RotationSelector
from src.localization.scale_manager import crop_to_affine

_GEOD = Geod(ellps="WGS84")


def distance_m(a, b):
    return abs(float(_GEOD.inv(a[1], a[0], b[1], b[0])[2]))


@dataclass
class LayerObservation:
    source_id: str
    gps: tuple[float, float]
    quality: float
    prepared: tuple
    homography: np.ndarray
    dimensions: tuple[int, int]


@dataclass
class ScaleBelief:
    log_ratio: float
    sigma: float
    timestamp: float
    angle: int

    def candidates(self, now, drift):
        sigma = min(0.7, self.sigma + drift * max(0.0, now - self.timestamp))
        return [float(np.clip(np.exp(self.log_ratio + d), 0.3, 3.5)) for d in (0.0, -sigma, sigma)]


class LayerHandoff:
    """Selection is provisional until commit() follows downstream acceptance."""

    def __init__(
        self, confirmations=2, margin=0.15, agreement_m=30.0, lost_after_s=3.0, max_speed_mps=120.0
    ):
        self.confirmations = confirmations
        self.margin = margin
        self.agreement_m = agreement_m
        self.lost_after_s = lost_after_s
        self.max_speed_mps = max_speed_mps
        self.reset()

    def reset(self):
        self.active = None
        self.state = "SEARCHING"
        self.last_confirmed = None
        self.pending = None
        self.pending_count = 0
        self.pending_time = None
        self.pending_gps = None
        self.reason = None

    def choose(self, observations, now):
        self.reason = None
        if self.last_confirmed is not None and now - self.last_confirmed > self.lost_after_s:
            self.state = "LOST"
        was_lost = self.state == "LOST"
        if not observations:
            self.pending = None
            self.pending_count = 0
            self.reason = "no_verified_candidates"
            return None
        ranked = sorted(observations, key=lambda o: (-o.quality, o.source_id))
        best = ranked[0]
        current = next((o for o in ranked if o.source_id == self.active), None)
        # Similar geometric evidence at incompatible locations is not resolved
        # by retrieval score or by the previous source ID.
        for other in ranked[1:]:
            if (
                other.quality >= best.quality / (1.0 + self.margin)
                and distance_m(best.gps, other.gps) > self.agreement_m
            ):
                self.pending = None
                self.pending_count = 0
                self.reason = "ambiguous_geometry"
                return None
        if self.state != "LOST" and current is not None:
            if best.source_id == self.active or best.quality < current.quality * (1 + self.margin):
                self.pending = None
                self.pending_count = 0
                self.state = "TRACKING"
                return current
            if distance_m(best.gps, current.gps) > self.agreement_m:
                self.pending = None
                self.pending_count = 0
                self.reason = "layer_map_disagreement"
                return current
        # Bootstrap, reacquisition and switching require fresh observations.
        if (
            self.pending != best.source_id
            or (self.pending_time is not None and now - self.pending_time > self.lost_after_s)
            or (
                self.pending_gps is not None
                and self.pending_time is not None
                and distance_m(best.gps, self.pending_gps)
                > self.agreement_m + self.max_speed_mps * max(0.0, now - self.pending_time)
            )
        ):
            self.pending = best.source_id
            self.pending_count = 0
            self.pending_time = None
        if self.pending_time is None or now > self.pending_time:
            self.pending_count += 1
            self.pending_time = now
            self.pending_gps = best.gps
        self.state = "HANDOFF_PENDING" if self.active is not None else "SEARCHING"
        if self.pending_count >= self.confirmations:
            return best
        self.reason = "awaiting_confirmation"
        return None if was_lost else current

    def commit(self, observation, now):
        self.active = observation.source_id
        self.last_confirmed = now
        if self.pending == observation.source_id:
            self.pending = None
            self.pending_count = 0
        self.state = "HANDOFF_PENDING" if self.pending else "TRACKING"


class LayerSearch:
    def __init__(self, config):
        self.config = config
        self.beliefs = {}
        self.handoff = LayerHandoff(
            self.cfg("confirmations", 2),
            self.cfg("switch_margin", 0.15),
            self.cfg("agreement_m", 30.0),
            self.cfg("lost_after_s", 3.0),
            get_cfg(config, "tracking.max_speed_mps", 120.0),
        )
        self.last_diagnostics = {}
        self._source_cursor = 0

    def cfg(self, key, default):
        return get_cfg(self.config, "localization.layer_search." + key, default)

    def reset(self):
        self.beliefs.clear()
        self.handoff.reset()
        self._source_cursor = 0

    def search(self, localizer, frame, mask, now, yaw_hint=None):
        start = time.monotonic()
        deadline = start + self.cfg("budget_ms", 2000.0) / 1000.0
        max_verifications = self.cfg("max_verifications", 32)
        top_k = self.cfg("candidates_per_source", 4)
        angles = [0, 90, 180, 270] if localizer.enable_auto_rotation else [0]
        if yaw_hint is not None and localizer.enable_auto_rotation:
            angle = (int(round(yaw_hint / 90)) * 90) % 360
            angles = [angle] + [a for a in angles if a != angle]
        primary = [(a, 1.0) for a in angles]
        for belief in self.beliefs.values():
            primary = [
                (belief.angle, s)
                for s in belief.candidates(now, self.cfg("scale_drift_per_s", 0.1))
            ] + primary
        recovery = [(a, s) for s in localizer._scale_manager.full_candidates() for a in angles]
        seen = set()
        cache = {}
        observations = []
        verifications = 0
        exhausted = False
        for combos in (primary, recovery):
            by_source = {}
            for angle, scale in combos:
                key = (angle, round(scale, 6))
                if key in seen:
                    continue
                # Reserve roughly half the remaining stage budget for matching.
                # Inference is indivisible; the deadline is checked between calls.
                if by_source and time.monotonic() >= (start + deadline) / 2:
                    break
                if time.monotonic() >= deadline:
                    exhausted = True
                    break
                seen.add(key)
                prepared, crop = RotationSelector._prepare_frame(
                    frame, angle, scale, localizer._scale_manager
                )
                desc = localizer.feature_extractor.extract_global_descriptor(prepared)
                groups = localizer.db_manager.get_matches_by_source(
                    desc, top_k, require_schema=self.cfg("require_schema", False)
                )
                for sid, candidates in groups.items():
                    for candidate in candidates:
                        by_source.setdefault(sid, []).append(
                            (candidate, angle, scale, prepared, crop)
                        )
            for hypotheses in by_source.values():
                hypotheses.sort(key=lambda h: h[0][1], reverse=True)
            ids = sorted(by_source)
            if ids:
                offset = self._source_cursor % len(ids)
                ids = ids[offset:] + ids[:offset]
                by_source = {sid: by_source[sid] for sid in ids}
                self._source_cursor += 1
            # Round robin reserves verification opportunities for every source.
            while by_source and verifications < max_verifications:
                for sid in list(by_source):
                    if (
                        verifications > 0 and time.monotonic() >= deadline
                    ) or verifications >= max_verifications:
                        exhausted = True
                        break
                    candidate, angle, scale, prepared, crop = by_source[sid].pop(0)
                    if not by_source[sid]:
                        del by_source[sid]
                    database = localizer.db_manager.get_database(sid)
                    calibration = localizer.calib_manager.get(sid)
                    if database is None or getattr(calibration, "converter", None) is None:
                        continue
                    frm, msk, ci, features = localizer._prepare_and_extract(
                        frame, mask, angle, scale, cache, prepared=(prepared, crop)
                    )
                    verifications += 1
                    ver = localizer._geometric_verifier.verify(features, [candidate], database)
                    if ver is None:
                        continue
                    observation = self._observation(
                        localizer,
                        sid,
                        database,
                        calibration,
                        ver,
                        (ver, angle, scale, frm, msk, ci, features, [candidate]),
                        frame.shape,
                    )
                    if observation is not None:
                        observations.append(observation)
                if exhausted:
                    break
            if observations or exhausted or verifications >= max_verifications:
                break
        self.last_diagnostics = {
            "verifications": verifications,
            "hypotheses": len(seen),
            "budget_exhausted": exhausted,
            "elapsed_ms": (time.monotonic() - start) * 1000,
        }
        return observations

    def _observation(self, localizer, sid, database, calibration, ver, prepared, shape):
        _, angle, _, frm, _, crop, features, _ = prepared
        if (
            not np.isfinite(ver.rmse)
            or ver.rmse > self.cfg("max_rmse_px", 4.0)
            or ver.inliers / max(1, ver.total_matches) < self.cfg("min_inlier_ratio", 0.2)
        ):
            return None
        spread = localizer._inlier_spread(ver.mkpts_q_in, features)
        if spread is None or spread < self.cfg("min_spread", 0.015):
            return None
        affine = database.get_frame_affine(ver.candidate_id)
        if affine is None:
            return None
        statuses = getattr(database, "frame_georef_status", None)
        if statuses is not None and int(statuses[ver.candidate_id]) != int(
            GeoreferenceStatus.SUPPORTED
        ):
            return None
        origins = getattr(database, "frame_origin", None)
        if origins is not None and origins[ver.candidate_id] in (
            CalibrationOrigin.UNKNOWN,
            CalibrationOrigin.EXTRAPOLATED,
        ):
            return None
        ref_h, ref_w = database.get_frame_size(ver.candidate_id)
        ref_spread = localizer._inlier_spread(ver.mkpts_r_in, {"image_size": [ref_h, ref_w]})
        # Reference support is measured relative to its own extent: severe
        # near-collinearity is invalid even when the query occupies a small crop.
        points = np.asarray(ver.mkpts_r_in, dtype=float)
        extents = np.ptp(points, axis=0)
        if np.min(extents) <= 1e-6 or ref_spread is None:
            return None
        centered = (points - points.mean(axis=0)) / extents
        eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(points))
        if eigenvalues[0] < 1e-4:
            return None
        H = np.asarray(ver.H_query_to_ref, dtype=np.float64)
        if crop is not None:
            H = H @ crop_to_affine(crop, frm.shape[1], frm.shape[0])
        h, w = shape[:2]
        if angle in (90, 270):
            h, w = w, h
        # Matching a small off-centre patch does not support arbitrary projection
        # of the image centre. Measure distance in the actual matcher frame,
        # after the same crop/resize used to extract its keypoints.
        query_center = np.array([[w / 2, h / 2]], dtype=np.float64)
        if crop is not None:
            query_center = GeometryTransforms.apply_homography(
                query_center, crop_to_affine(crop, frm.shape[1], frm.shape[0])
            )
        hull = cv2.convexHull(np.asarray(ver.mkpts_q_in, dtype=np.float32))
        support_distance = cv2.pointPolygonTest(hull, tuple(query_center[0]), True)
        if support_distance < -self.cfg("max_center_extrapolation", 0.1) * np.hypot(*frm.shape[:2]):
            return None
        corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float)
        den = corners @ H[2]
        # A horizon/pole inside the image invalidates full-frame projection.
        if not np.isfinite(H).all() or not (np.all(den > 1e-9) or np.all(den < -1e-9)):
            return None
        center = GeometryTransforms.apply_homography(np.array([[w / 2, h / 2]]), H)
        metric = GeometryTransforms.apply_affine(center, affine)
        if metric is None or not np.isfinite(metric).all():
            return None
        gps = calibration.converter.metric_to_gps(*map(float, metric[0]))
        if not np.isfinite(gps).all() or abs(gps[0]) > 90 or abs(gps[1]) > 180:
            return None
        quality = (
            ver.inliers
            * (ver.inliers / max(1, ver.total_matches))
            * min(1.0, spread / 0.15)
            / (1.0 + ver.rmse)
        )
        return LayerObservation(sid, tuple(gps), quality, prepared, H, (w, h))

    def commit(self, observation, now):
        self.handoff.commit(observation, now)
        H = observation.homography
        w, h = observation.dimensions
        x, y = w / 2, h / 2
        den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
        projected = (H @ [x, y, 1])[:2] / den
        J = (H[:2, :2] - np.outer(projected, H[2, :2])) / den
        singular = np.linalg.svd(J, compute_uv=False)
        ratio = float(np.sqrt(abs(np.linalg.det(J))))
        if 0.3 <= ratio <= 3.5 and singular[-1] > 1e-9:
            sigma = min(0.7, 0.05 + 0.25 * np.log(singular[0] / singular[-1]))
            old = self.beliefs.get(observation.source_id)
            measured = np.log(ratio)
            mean = measured if old is None else 0.7 * measured + 0.3 * old.log_ratio
            self.beliefs[observation.source_id] = ScaleBelief(
                float(mean), float(sigma), now, observation.prepared[1]
            )
