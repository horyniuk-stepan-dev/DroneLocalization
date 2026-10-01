"""Geometry-first multi-source search and a separately testable handoff policy.

All scale estimates are image ratios in normalized query/reference coordinates.
Geographic comparisons use geodesic metres, never two sources' raw map units.
"""

from __future__ import annotations

import time
from collections import deque
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


def center_scale(H, dimensions):
    """Local query->reference scale sqrt|det J| and J's singular values at the image centre."""
    w, h = dimensions
    x, y = w / 2, h / 2
    den = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    projected = (H @ [x, y, 1])[:2] / den
    J = (H[:2, :2] - np.outer(projected, H[2, :2])) / den
    return float(np.sqrt(abs(np.linalg.det(J)))), np.linalg.svd(J, compute_uv=False)


@dataclass
class LayerObservation:
    source_id: str
    gps: tuple[float, float]
    quality: float
    prepared: tuple
    homography: np.ndarray
    dimensions: tuple[int, int]
    scale_ratio: float | None = None


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

    # Shortest baseline for a velocity estimate; shorter ones amplify fix noise.
    MIN_VELOCITY_BASELINE_S = 0.5

    def __init__(
        self,
        confirmations=2,
        margin=0.15,
        agreement_m=30.0,
        lost_after_s=3.0,
        max_speed_mps=120.0,
        motion_gate=False,
        motion_gate_growth_mps=30.0,
        motion_gate_max_gap_s=10.0,
    ):
        self.confirmations = confirmations
        self.margin = margin
        self.agreement_m = agreement_m
        self.lost_after_s = lost_after_s
        self.max_speed_mps = max_speed_mps
        # Motion gate: after a gap (LOST, or the active layer did not verify),
        # accept a single observation that lies where the confirmed track
        # predicts the drone to be, instead of waiting for fresh confirmations.
        self.motion_gate = bool(motion_gate)
        self.motion_gate_growth_mps = float(motion_gate_growth_mps)
        self.motion_gate_max_gap_s = float(motion_gate_max_gap_s)
        self.reset()

    def reset(self):
        self.active = None
        self.state = "SEARCHING"
        self.last_confirmed = None
        self.last_confirmed_gps = None
        self.track = deque(maxlen=8)
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
        # A gap in the active layer's fixes (LOST, or it did not verify this
        # frame) is bridged by one observation that agrees with the track's
        # motion. Bootstrap and discretionary switches still need confirmations.
        if self.motion_gate and (was_lost or current is None) and self.motion_consistent(best, now):
            self.pending = None
            self.pending_count = 0
            self.state = "TRACKING"
            self.reason = "motion_gated"
            return best
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

    def predict(self, now):
        """Predicted GPS and acceptance radius (m) from the confirmed track.

        Constant velocity over the longest recent baseline of confirmed fixes.
        Returns None (no gate) without such a velocity estimate or when the gap
        exceeds the gate window: a radius bounded only by the platform's
        maximum speed is too loose to stand in for fresh confirmations.
        """
        if self.last_confirmed is None or self.last_confirmed_gps is None:
            return None
        gap = now - self.last_confirmed
        if not 0.0 <= gap <= self.motion_gate_max_gap_s:
            return None
        lat, lon = self.last_confirmed_gps
        for t0, (lat0, lon0) in self.track:
            baseline = self.last_confirmed - t0
            if baseline > self.motion_gate_max_gap_s:
                continue
            if baseline < self.MIN_VELOCITY_BASELINE_S:
                break
            _, back_azimuth, dist = _GEOD.inv(lon0, lat0, lon, lat)
            speed = abs(float(dist)) / baseline
            if speed > self.max_speed_mps:
                break
            # Heading at the latest fix (reverse of the back azimuth).
            plon, plat, _ = _GEOD.fwd(lon, lat, back_azimuth + 180.0, speed * gap)
            return (float(plat), float(plon)), self.agreement_m + self.motion_gate_growth_mps * gap
        return None

    def motion_consistent(self, observation, now):
        prediction = self.predict(now)
        if prediction is None:
            return False
        center, radius = prediction
        return distance_m(observation.gps, center) <= radius

    def commit(self, observation, now):
        self.active = observation.source_id
        self.last_confirmed = now
        self.last_confirmed_gps = observation.gps
        self.track.append((now, tuple(observation.gps)))
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
            motion_gate=self.cfg("motion_gate", False),
            motion_gate_growth_mps=self.cfg("motion_gate_growth_mps", 30.0),
            motion_gate_max_gap_s=self.cfg("motion_gate_max_gap_s", 10.0),
        )
        self.last_diagnostics = {}
        self._source_cursor = 0
        self._probe_cursor = 0

    def cfg(self, key, default):
        return get_cfg(self.config, "localization.layer_search." + key, default)

    def reset(self):
        self.beliefs.clear()
        self.handoff.reset()
        self._source_cursor = 0
        self._probe_cursor = 0

    def _select_sources(self, manager):
        """Prefer the active/geographically plausible layers, probe the rest fairly.

        The last accepted GPS is only a routing hint. A rotating slot always
        reaches other loaded sources, including ones excluded by a stale active
        area or incomplete source coverage metadata.
        """
        all_ids = getattr(manager, "all_source_ids", None)
        if all_ids is None:
            return None  # legacy/test manager: preserve its own source routing
        ids = sorted(set(all_ids))
        cap = max(2, int(self.cfg("max_sources_per_frame", 8)))
        if len(ids) <= cap:
            return ids

        preferred = []
        active = self.handoff.active
        if active in ids:
            preferred.append(active)

        source_config = getattr(manager, "get_source_config", None)
        if callable(source_config):
            active_config = source_config(active) if active is not None else None
            layer = getattr(active_config, "scale_layer", None)
            neighbors = set(getattr(layer, "neighbor_layer_ids", ()) or ())
            for sid in ids:
                config = source_config(sid)
                candidate_layer = getattr(config, "scale_layer", None)
                if candidate_layer is not None and candidate_layer.layer_id in neighbors:
                    preferred.append(sid)

            gps = self.handoff.last_confirmed_gps
            if gps is not None:
                for sid in ids:
                    config = source_config(sid)
                    bounds = getattr(config, "geo_bounds", None)
                    if bounds is not None and config.contains_point(*gps):
                        preferred.append(sid)

        preferred = list(dict.fromkeys(preferred))[: cap - 1]
        remaining = [sid for sid in ids if sid not in preferred]
        probe_count = cap - len(preferred)
        start = self._probe_cursor % len(remaining)
        selected = preferred + [remaining[(start + i) % len(remaining)] for i in range(probe_count)]
        self._probe_cursor += probe_count
        return selected

    def search(self, localizer, frame, mask, now, yaw_hint=None):
        start = time.monotonic()
        deadline = start + self.cfg("budget_ms", 2000.0) / 1000.0
        max_verifications = self.cfg("max_verifications", 32)
        top_k = self.cfg("candidates_per_source", 4)
        source_ids = self._select_sources(localizer.db_manager)
        angles = [0, 90, 180, 270] if localizer.enable_auto_rotation else [0]
        if yaw_hint is not None and localizer.enable_auto_rotation:
            angle = (int(round(yaw_hint / 90)) * 90) % 360
            angles = [angle] + [a for a in angles if a != angle]
        primary = [(angle, 1.0) for angle in angles]
        for belief in self.beliefs.values():
            primary = [
                (belief.angle, scale)
                for scale in belief.candidates(now, self.cfg("scale_drift_per_s", 0.1))
            ] + primary
        recovery = [
            (angle, scale)
            for scale in localizer._scale_manager.full_candidates()
            for angle in angles
        ]
        seen = set()
        cache = {}
        observations = []
        verifications = 0
        verified_signatures = set()
        exhausted = False
        diverse = bool(self.cfg("diverse_candidates", False))
        early_stop = bool(self.cfg("early_stop", False))
        stopped_early = False
        for combos in (primary, recovery):
            by_source = {}
            queued_signatures = set()
            planned = []
            planned_keys = set()
            for angle, scale in combos:
                # ScaleManager leaves this band unchanged (possibly returning
                # an identity CropInfo). Treat those ratios as one image, so
                # identical descriptor/matcher calls do not exhaust the budget.
                effective_scale = 1.0 if 0.85 <= scale <= 1.18 else scale
                key = (angle, round(effective_scale, 6))
                if key in seen or key in planned_keys:
                    continue
                planned.append((angle, scale, key))
                planned_keys.add(key)
            batch_size = max(1, int(self.cfg("descriptor_batch_size", 4)))
            for offset in range(0, len(planned), batch_size):
                # Reserve roughly half the remaining stage budget for matching.
                # Inference is indivisible; the deadline is checked between calls.
                if by_source and time.monotonic() >= (start + deadline) / 2:
                    break
                if time.monotonic() >= deadline:
                    exhausted = True
                    break
                chunk = planned[offset : offset + batch_size]
                prepared_chunk = [
                    RotationSelector._prepare_frame(
                        frame, angle, scale, localizer._scale_manager
                    )
                    for angle, scale, _ in chunk
                ]
                frames = [item[0] for item in prepared_chunk]
                if len(frames) > 1 and hasattr(
                    localizer.feature_extractor, "extract_global_descriptors_multi"
                ):
                    descriptors = localizer.feature_extractor.extract_global_descriptors_multi(
                        frames
                    )
                else:
                    descriptors = [
                        localizer.feature_extractor.extract_global_descriptor(item)
                        for item in frames
                    ]
                for (angle, scale, key), (prepared, crop), desc in zip(
                    chunk, prepared_chunk, descriptors
                ):
                    seen.add(key)
                    groups = localizer.db_manager.get_matches_by_source(
                        desc,
                        top_k,
                        require_schema=self.cfg("require_schema", False),
                        **({"source_ids": source_ids} if source_ids is not None else {}),
                    )
                    crop_signature = (
                        int(getattr(crop, "crop_x", 0)),
                        int(getattr(crop, "crop_y", 0)),
                        int(getattr(crop, "crop_w", 0)),
                        int(getattr(crop, "crop_h", 0)),
                    )
                    image_signature = (
                        angle, prepared.shape[0], prepared.shape[1], crop_signature
                    )
                    for sid, candidates in groups.items():
                        for candidate in candidates:
                            signature = (sid, int(candidate[0]), image_signature)
                            if signature in queued_signatures or signature in verified_signatures:
                                continue
                            queued_signatures.add(signature)
                            by_source.setdefault(sid, []).append(
                                (candidate, angle, scale, prepared, crop, signature)
                            )
            for hypotheses in by_source.values():
                hypotheses.sort(key=lambda h: h[0][1], reverse=True)
                if diverse:
                    # Each retrieved frame gets one attempt (at its best-scoring
                    # angle/scale) before any frame is retried at another one.
                    attempts = {}
                    ranks = []
                    for hypothesis in hypotheses:
                        frame_id = int(hypothesis[0][0])
                        ranks.append(attempts.get(frame_id, 0))
                        attempts[frame_id] = ranks[-1] + 1
                    order = sorted(range(len(hypotheses)), key=ranks.__getitem__)
                    hypotheses[:] = [hypotheses[i] for i in order]
            ids = sorted(by_source)
            if ids:
                offset = self._source_cursor % len(ids)
                ids = ids[offset:] + ids[:offset]
                if early_stop and self.handoff.active in ids:
                    ids.remove(self.handoff.active)
                    ids.insert(0, self.handoff.active)
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
                    candidate, angle, scale, prepared, crop, signature = by_source[sid].pop(0)
                    verified_signatures.add(signature)
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
                        if early_stop and self._settles_frame(observation, now):
                            stopped_early = True
                            break
                if exhausted or stopped_early:
                    break
            if observations or exhausted or verifications >= max_verifications:
                break
        self.last_diagnostics = {
            "verifications": verifications,
            "hypotheses": len(seen),
            "budget_exhausted": exhausted,
            "early_stop": stopped_early,
            "elapsed_ms": (time.monotonic() - start) * 1000,
            "sources_probed": source_ids,
            "sources_available": len(getattr(localizer.db_manager, "all_source_ids", []) or []),
        }
        return observations

    def _settles_frame(self, observation, now):
        """True for a strong, near-native-scale fix from the active layer that
        agrees with the confirmed track: further candidates cannot improve on it."""
        handoff = self.handoff
        if (
            observation.source_id != handoff.active
            or handoff.last_confirmed is None
            or now - handoff.last_confirmed > handoff.lost_after_s
            or observation.quality < self.cfg("early_stop_min_quality", 150.0)
        ):
            return False
        max_ratio = self.cfg("early_stop_max_scale_ratio", 1.5)
        ratio = observation.scale_ratio
        if ratio is None or not 1.0 / max_ratio <= ratio <= max_ratio:
            return False
        return handoff.motion_consistent(observation, now)

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
        return LayerObservation(
            sid, tuple(gps), quality, prepared, H, (w, h), center_scale(H, (w, h))[0]
        )

    def commit(self, observation, now):
        self.handoff.commit(observation, now)
        ratio, singular = center_scale(observation.homography, observation.dimensions)
        if 0.3 <= ratio <= 3.5 and singular[-1] > 1e-9:
            sigma = min(0.7, 0.05 + 0.25 * np.log(singular[0] / singular[-1]))
            old = self.beliefs.get(observation.source_id)
            measured = np.log(ratio)
            mean = measured if old is None else 0.7 * measured + 0.3 * old.log_ratio
            self.beliefs[observation.source_id] = ScaleBelief(
                float(mean), float(sigma), now, observation.prepared[1]
            )
