"""Robust planar multi-view adjustment in original image pixel coordinates.

Transforms map each image into one fixed gauge. The objective measures symmetric
image transfer errors, not map distances which can be reduced by shrinking maps.
Geographic anchors, when supplied, remain exactly fixed at their input transforms.
"""

import heapq

import numpy as np
from scipy import sparse
from scipy.optimize import least_squares

from src.geometry.frame_transform import as_homography, project_points, valid_footprint


def connected_links(initial, links, fixed_ids):
    """Keep finite 2D constraints and only components supported by fixed frames."""
    valid, neighbors = [], {fid: set() for fid in initial}
    for link in links:
        p, q = np.asarray(link.first_points), np.asarray(link.second_points)
        if (
            link.first not in initial
            or link.second not in initial
            or link.first == link.second
            or not np.isfinite(link.weight)
            or link.weight <= 0
            or p.ndim != 2
            or p.shape[1] != 2
            or p.shape != q.shape
            or len(p) < 8
            or not np.isfinite(p).all()
            or not np.isfinite(q).all()
            or np.linalg.matrix_rank(p - p.mean(axis=0)) < 2
            or np.linalg.matrix_rank(q - q.mean(axis=0)) < 2
        ):
            continue
        valid.append(link)
        neighbors[link.first].add(link.second)
        neighbors[link.second].add(link.first)
    reached = set(fixed_ids) & initial.keys()
    pending = list(reached)
    while pending:
        for fid in neighbors[pending.pop()] - reached:
            reached.add(fid)
            pending.append(fid)
    return (
        {fid: initial[fid] for fid in sorted(reached)},
        [link for link in valid if link.first in reached and link.second in reached],
    )


def symmetric_transfer(H, ia, ib, p, q, size, dof=8, *, jacobian=False):
    """Symmetric pixel residuals and exact derivatives w.r.t. both frame matrices.

    For z=B^-1 A p, dz=d(B^-1) A p+B^-1 dA p and
    d(B^-1)=-B^-1 dB B^-1. H[2,2] is fixed in this parameterization.
    """
    inverse = np.linalg.inv(H)
    residual, blocks = [], []
    rows, cols = np.arange(dof) // 3, np.arange(dof) % 3
    for source, target, points, observed in ((ia, ib, p, q), (ib, ia, q, p)):
        inv = inverse[target]
        z = np.einsum("nij,nj->ni", inv @ H[source], points)
        denominator = z[:, 2:3]
        safe = np.where(
            np.abs(denominator) < 1e-9, np.where(denominator < 0, -1e-9, 1e-9), denominator
        )
        projected = z[:, :2] / safe
        raw = (projected - observed[:, :2]) * size
        residual.append(np.clip(raw, -1e6, 1e6))
        if jacobian:
            D = (inv[:, :2] - projected[:, :, None] * inv[:, 2:3]) / safe[:, :, None]
            # The clipped residual is locally constant beyond its safety bound.
            D *= (np.abs(raw) < 1e6)[:, :, None] * (np.abs(denominator) >= 1e-9)[:, :, None]
            blocks.append(
                (
                    size * D[:, :, rows] * points[:, None, cols],
                    -size * D[:, :, rows] * z[:, None, cols],
                )
            )
    errors = np.concatenate(residual, axis=1)
    if not jacobian:
        return errors
    return (
        errors,
        np.concatenate([blocks[0][0], blocks[1][1]], axis=1),
        np.concatenate([blocks[0][1], blocks[1][0]], axis=1),
    )


def initialize_from_pairs(pair_homographies, weighted_pairs, fixed, width, height):
    """Maximum-weight spanning forest, composing full H (source -> target).

    Only nodes connected to a fixed seed are returned; unsupported images are
    never given fabricated coordinates. Invalid accumulated horizons are skipped.
    """
    result = {fid: as_homography(H).copy() for fid, H in fixed.items()}
    adjacency = {}
    for first, second, weight in weighted_pairs:
        if weight <= 0:
            continue
        H = pair_homographies.get((second, first))
        if H is None:
            reverse = pair_homographies.get((first, second))
            if reverse is None:
                continue
            H = np.linalg.inv(reverse)
        adjacency.setdefault(first, []).append((second, weight, H))
        adjacency.setdefault(second, []).append((first, weight, np.linalg.inv(H)))
    queue, counter = [], 0

    def push(fid):
        nonlocal counter
        for other, weight, H in adjacency.get(fid, []):
            if other not in result:
                counter += 1
                heapq.heappush(queue, (-weight, counter, fid, other, H))

    for fid in sorted(result):
        push(fid)
    while queue:
        _, _, parent, child, H = heapq.heappop(queue)
        if child in result:
            continue
        composed = result[parent] @ H
        norm = np.linalg.norm(composed)
        if norm <= 0 or not valid_footprint(composed, width, height):
            continue
        # Prefer a center gauge; H[2,2] can be zero outside the actual footprint.
        denominator = composed[2] @ np.array([width / 2, height / 2, 1.0])
        result[child] = composed / denominator
        push(child)
    return result


def refine_projective_map(
    initial, links, width, height, fixed_ids, *, max_nfev=150, check_running=lambda: None
):
    """Select full-affine or projective geometry using held-out transfer error.

    The simpler full-affine model wins when its validation loss is within 5% of
    the projective candidate. This limits perspective extrapolation from noisy,
    narrow overlaps. Both models use the same points, gauge, weights and units.
    """
    candidates, failures = {}, {}
    for model in ("projective", "affine"):
        check_running()
        try:
            candidates[model] = _refine_map(
                initial,
                links,
                width,
                height,
                fixed_ids,
                model=model,
                max_nfev=max_nfev,
                check_running=check_running,
            )
        except ValueError as exc:
            failures[model] = str(exc)
        if set(initial) == set(fixed_ids):
            break
    if not candidates:
        raise ValueError(f"No valid relative model: {failures}")
    selected = "projective" if "projective" in candidates else "affine"
    if len(candidates) == 2:
        projective_qa = candidates["projective"][1]
        affine_qa = candidates["affine"][1]
        if (
            affine_qa["heldout_points"] > 0
            and affine_qa["heldout_robust_cost"]
            <= 1.05 * projective_qa["heldout_robust_cost"] + 1e-6
        ):
            selected = "affine"
    result, report, per_frame = candidates[selected]
    report["selected_model"] = selected
    report["model_selection"] = {
        model: {
            "heldout_robust_cost": qa[1]["heldout_robust_cost"],
            "heldout_p95_px": qa[1]["heldout_after_p95_px"],
        }
        for model, qa in candidates.items()
    }
    report["rejected_models"] = failures
    return result, report, per_frame


def _refine_map(initial, links, width, height, fixed_ids, *, model, max_nfev, check_running):
    """Return transforms, training/held-out QA and per-frame symmetric pixel RMS.

    Every fifth observation is withheld from the global solve (pairwise RANSAC
    may already have used it). The fixed first image removes the projective gauge.
    An analytic sparse Jacobian keeps derivatives local to each pair of cameras.
    """
    if not initial or not fixed_ids or not set(fixed_ids) <= initial.keys():
        raise ValueError("Projective adjustment needs existing fixed frames")
    if min(width, height) <= 0:
        raise ValueError("Projective adjustment needs image dimensions")
    ids = sorted(initial)
    index = {fid: i for i, fid in enumerate(ids)}
    free = [fid for fid in ids if fid not in fixed_ids]
    links = [
        link
        for link in links
        if link.first in initial and link.second in initial and link.weight > 0
    ]
    if not links:
        raise ValueError("Projective adjustment needs verified links")
    neighbors = {fid: set() for fid in ids}
    for link in links:
        p, q = np.asarray(link.first_points), np.asarray(link.second_points)
        if (
            p.ndim != 2
            or p.shape[1] != 2
            or p.shape != q.shape
            or len(p) < 8
            or not np.isfinite(p).all()
            or not np.isfinite(q).all()
            or not np.isfinite(link.weight)
            or np.linalg.matrix_rank(p - p.mean(axis=0)) < 2
            or np.linalg.matrix_rank(q - q.mean(axis=0)) < 2
        ):
            raise ValueError("Projective links need non-collinear finite observations")
        neighbors[link.first].add(link.second)
        neighbors[link.second].add(link.first)
    reached, pending = set(fixed_ids), list(fixed_ids)
    while pending:
        for fid in neighbors[pending.pop()] - reached:
            reached.add(fid)
            pending.append(fid)
    if reached != set(ids):
        raise ValueError("Projective links do not connect every frame to a fixed gauge")
    if len(free) > 64:
        return _refine_windowed(
            initial,
            links,
            width,
            height,
            fixed_ids,
            model=model,
            max_nfev=max_nfev,
            check_running=check_running,
        )

    # Normalize both image coordinates and the arbitrary map gauge. This also
    # removes large geographic translations before any solver arithmetic.
    size = float(max(width, height))
    N = np.array(
        [[1 / size, 0, -width / (2 * size)], [0, 1 / size, -height / (2 * size)], [0, 0, 1.0]]
    )
    inv_N = np.linalg.inv(N)
    gauge = as_homography(initial[min(fixed_ids)])
    to_local = N @ np.linalg.inv(gauge)
    to_map = gauge @ inv_N
    matrices = np.array([to_local @ as_homography(initial[fid]) @ inv_N for fid in ids])
    matrices /= matrices[:, 2:3, 2:3]
    if not np.isfinite(matrices).all():
        raise ValueError("Invalid projective initialization")
    free_idx = np.array([index[fid] for fid in free], dtype=int)
    dof = 6 if model == "affine" else 8
    columns = {fid: dof * i for i, fid in enumerate(free)}
    x0 = matrices[free_idx].reshape(-1, 9)[:, :dof].ravel()
    first_ids, second_ids, ps, qs, weights, training, sections = [], [], [], [], [], [], []
    validation = []
    start = 0
    median_weight = np.median([link.weight for link in links])
    for link in links:
        p, q = np.asarray(link.first_points), np.asarray(link.second_points)
        # Deterministic spatial ordering, rather than matcher confidence order.
        order = np.lexsort((p[:, 1], p[:, 0]))
        train = np.ones(len(p), dtype=bool)
        if len(p) >= 15:
            train[order[::5]] = False
        # Keep held-out observations independent of the training cap. Select
        # a spatially spread 96-point subset from the remaining observations.
        validation.extend(~train)
        candidates = np.flatnonzero(train)
        if len(candidates) > 96:
            from src.geometry.relative_alignment import sample_point_pair

            # sample_point_pair returns the paired indices in its second output.
            _, selected_ids = sample_point_pair(p[candidates], candidates[:, None], 96)
            train[:] = False
            train[selected_ids.ravel().astype(int)] = True
        first_ids.extend([index[link.first]] * len(p))
        second_ids.extend([index[link.second]] * len(p))
        ps.append(np.column_stack([project_points(N, p), np.ones(len(p))]))
        qs.append(np.column_stack([project_points(N, q), np.ones(len(q))]))
        # Graph weights are residual multipliers. Keep that convention here.
        weights.extend([link.weight / median_weight / np.sqrt(train.sum())] * len(p))
        training.extend(train)
        sections.append(slice(start, start + len(p)))
        start += len(p)
    ia, ib = np.array(first_ids), np.array(second_ids)
    p, q = np.concatenate(ps), np.concatenate(qs)
    train = np.array(training, dtype=bool)
    weight = np.array(weights)
    heldout = np.array(validation, dtype=bool)

    def unpack(x):
        H = matrices.copy()
        if len(free):
            values = np.zeros((len(free), 9))
            values[:, :dof] = x.reshape(-1, dof)
            values[:, 8] = 1.0
            H[free_idx] = values.reshape(-1, 3, 3)
        return H

    def errors(x, selected=slice(None), *, jacobian=False):
        check_running()
        try:
            H = unpack(x)
            if not jacobian:
                # Full QA can include millions of observations; avoid allocating
                # multiple 3x3 matrices per observation over the whole flight.
                indices = np.arange(len(p))[selected]
                chunks = [indices[i : i + 8192] for i in range(0, len(indices), 8192)]
                return np.concatenate(
                    [symmetric_transfer(H, ia[k], ib[k], p[k], q[k], size, dof) for k in chunks]
                )
            return symmetric_transfer(
                H, ia[selected], ib[selected], p[selected], q[selected], size, dof, jacobian=True
            )
        except np.linalg.LinAlgError:
            n = len(p[selected])
            if jacobian:
                return np.full((n, 4), 1e6), np.zeros((n, 4, dof)), np.zeros((n, 4, dof))
            return np.full((n, 4), 1e6)

    row_points = np.flatnonzero(train)
    row_ids, col_ids, active_indices = [], [], []
    camera_columns = np.array([columns.get(fid, -1) for fid in ids])
    for endpoints in (ia, ib):
        base_columns = camera_columns[endpoints[row_points]]
        active = np.flatnonzero(base_columns >= 0)
        active_indices.append(active)
        rows = (active[:, None] * 4 + np.arange(4)).reshape(-1)
        row_ids.append(np.repeat(rows, dof))
        col_ids.append(np.tile(base_columns[active, None] + np.arange(dof), (1, 4)).ravel())
    n_data = 4 * len(row_points)
    row_ids = np.concatenate(row_ids + [np.arange(n_data, n_data + len(x0))])
    col_ids = np.concatenate(col_ids + [np.arange(len(x0))])

    before = errors(x0)
    x = x0.copy()
    evaluations = 0
    converged = not len(free)
    for _ in range(3 if len(free) else 0):
        residual = errors(x, train)
        norms = np.sqrt(np.sum(residual**2, axis=1) / 2)
        robust = np.sqrt(np.minimum(1.0, 2.0 / np.maximum(norms, 1e-12)))
        per_point = weight[train] * robust

        def objective(candidate, per_point=per_point):
            data = errors(candidate, train) * per_point[:, None]
            # Tiny shape prior only stabilizes underconstrained directions.
            return np.r_[data.ravel(), 1e-3 * (candidate - x0)]

        def jac(candidate, per_point=per_point):
            _, first, second = errors(candidate, train, jacobian=True)
            values = [
                (block[active] * per_point[active, None, None]).ravel()
                for block, active in zip((first, second), active_indices)
            ]
            values.append(np.full(len(x0), 1e-3))
            return sparse.csr_matrix(
                (np.concatenate(values), (row_ids, col_ids)), shape=(n_data + len(x0), len(x0))
            )

        fit = least_squares(
            objective,
            x,
            jac=jac,
            method="trf",
            x_scale="jac",
            max_nfev=max_nfev,
            ftol=1e-8,
            xtol=1e-8,
            gtol=1e-8,
        )
        evaluations += fit.nfev
        converged = bool(fit.success)
        change = np.max(np.abs(fit.x - x))
        x = fit.x
        if change < 1e-8:
            break
    after = errors(x)

    def norms(residual):
        return np.sqrt(np.sum(residual**2, axis=1) / 2)

    def cost(values, selected):
        e = norms(values)[selected]
        return np.sum(weight[selected] ** 2 * np.where(e <= 2, e**2 / 2, 2 * e - 2))

    if cost(after, train) > cost(before, train) + 1e-5:
        raise ValueError("Projective refinement increased robust image error")
    if np.any(heldout) and cost(after, heldout) > cost(before, heldout) * 1.1 + 0.05:
        raise ValueError("Projective refinement worsened held-out observations")
    result = {}
    for fid, H in zip(ids, unpack(x)):
        matrix = to_map @ H @ N
        original = as_homography(initial[fid])
        orientation = np.sign(
            np.linalg.det(original) / (original[2] @ [width / 2, height / 2, 1]) ** 3
        )
        if not valid_footprint(matrix, width, height, orientation=orientation):
            raise ValueError("Projective refinement would fold a frame or cross a horizon")
        # Guard local scale relative to the seed at center and corners.
        corners = np.array([[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]])
        old = project_points(original, corners)
        new = project_points(matrix, corners)
        ratios = np.linalg.norm(new - np.roll(new, 1, axis=0), axis=1) / np.maximum(
            np.linalg.norm(old - np.roll(old, 1, axis=0), axis=1), 1e-12
        )
        if ratios.min() < 0.25 or ratios.max() > 4:
            raise ValueError("Projective refinement excessively distorted a frame")
        result[fid] = original.copy() if fid in fixed_ids else matrix / matrix[2, 2]
    b, a = norms(before), norms(after)
    per_frame = {fid: [] for fid in ids}
    for link, section in zip(links, sections):
        per_frame[link.first].extend(a[section] ** 2)
        per_frame[link.second].extend(a[section] ** 2)
    report = {
        "method": f"{model}_symmetric_irls_v1",
        "units": "original_image_pixels",
        "frames": len(ids),
        "links": len(links),
        "points": len(p),
        "training_points": int(train.sum()),
        "training_point_limit_per_link": 96,
        "jacobian": "analytic",
        "heldout_points": int((heldout).sum()),
        "before_median_px": float(np.median(b)),
        "after_median_px": float(np.median(a)),
        "before_p95_px": float(np.percentile(b, 95)),
        "after_p95_px": float(np.percentile(a, 95)),
        "heldout_before_p95_px": float(np.percentile(b[heldout], 95)) if np.any(heldout) else None,
        "heldout_after_p95_px": float(np.percentile(a[heldout], 95)) if np.any(heldout) else None,
        "heldout_scope": "global_adjustment_only",
        "evaluations": evaluations,
        "converged": converged,
        "heldout_robust_cost": float(cost(after, heldout)) if np.any(heldout) else None,
        "training_robust_cost": float(cost(after, train)),
        "frames_with_single_link": sum(len(n) == 1 for n in neighbors.values()),
        "gauge_warning": "Relative planar registration cannot establish metric ground accuracy or remove terrain parallax",
    }
    return (
        result,
        report,
        {fid: float(np.sqrt(np.mean(e))) if e else 0.0 for fid, e in per_frame.items()},
    )


def _refine_windowed(initial, links, width, height, fixed_ids, *, model, max_nfev, check_running):
    """Bounded block-coordinate adjustment; cross-window/loop neighbors stay fixed.

    Each solve includes every incident constraint, including distant loop closures.
    Forward and backward overlapping sweeps propagate corrections. This is a
    bounded approximation, not a claim of global convergence on long chains.
    """
    options = dict(model=model, max_nfev=max_nfev, check_running=check_running)
    _, before, _ = _refine_map(initial, links, width, height, set(initial), **options)
    result = {fid: as_homography(H).copy() for fid, H in initial.items()}
    free = sorted(set(initial) - set(fixed_ids))
    windows = [set(free[i : i + 32]) for i in range(0, len(free), 24)]
    incident = {fid: [] for fid in initial}
    for index, link in enumerate(links):
        incident[link.first].append(index)
        incident[link.second].append(index)
    evaluations, rejected, solved = 0, 0, 0
    all_converged, final_motion = True, 0.0
    probes = np.array(
        [[0.0, 0.0], [width, 0.0], [width, height], [0.0, height], [width / 2, height / 2]]
    )
    completed_sweeps = 0
    for sweep_index in range(8):
        sweep = windows if sweep_index % 2 == 0 else windows[::-1]
        completed_sweeps += 1
        final_motion = 0.0
        for active in sweep:
            check_running()
            selected = sorted({i for fid in active for i in incident[fid]})
            local_links = [links[i] for i in selected]
            nodes = {fid for link in local_links for fid in (link.first, link.second)}
            boundary = nodes - active
            if not boundary:
                raise ValueError("Window has no fixed boundary")
            try:
                adjusted, report, _ = _refine_map(
                    {fid: result[fid] for fid in sorted(nodes)},
                    local_links,
                    width,
                    height,
                    boundary,
                    **options,
                )
            except ValueError:
                rejected += 1
                all_converged = False
                continue
            solved += 1
            evaluations += report["evaluations"]
            all_converged &= report["converged"]
            for fid in active:
                movement = (
                    project_points(np.linalg.inv(adjusted[fid]) @ result[fid], probes) - probes
                )
                final_motion = max(final_motion, float(np.linalg.norm(movement, axis=1).max()))
                result[fid] = adjusted[fid]
        if completed_sweeps >= 2 and final_motion < 0.01:
            break
    _, after, frame_rmse = _refine_map(result, links, width, height, set(result), **options)
    if after["training_robust_cost"] > before["training_robust_cost"] + 1e-5:
        raise ValueError("Windowed refinement increased robust image error")
    if (
        before["heldout_points"]
        and after["heldout_robust_cost"] > before["heldout_robust_cost"] * 1.1 + 0.05
    ):
        raise ValueError("Windowed refinement worsened held-out observations")
    for key in ("before_median_px", "before_p95_px", "heldout_before_p95_px"):
        after[key] = before[key]
    after.update(
        solver="overlapping_block_coordinate",
        window_frames=32,
        window_overlap=8,
        sweeps=completed_sweeps,
        max_sweeps=8,
        windows_solved=solved,
        windows_rejected=rejected,
        evaluations=evaluations,
        converged=bool(all_converged and final_motion < 0.01),
        final_sweep_max_motion_px=final_motion,
        convergence_scope="bounded_window_sweeps; global stationarity not certified",
    )
    return result, after, frame_rmse
