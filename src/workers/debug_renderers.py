"""Рендер debug-каналів (вікна «очима моделей») — чистий cv2/numpy.

Викликається у worker-потоці ПІСЛЯ localize_frame. Кожна функція повертає
готове BGR-зображення (для opencv_to_qpixmap, який чекає BGR), вже
downscale-нуте до max_width. Жодних PyQt/torch-залежностей тут немає — модуль
можна тестувати ізольовано.

Увага: cv2.putText не рендерить кирилицю → усі підписи латиницею.
"""

import cv2
import numpy as np

# COCO-класи, які маскує YOLO (person, bicycle, car, motorcycle, bus, truck)
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# Кольори bbox у BGR
_CLASS_BGR = {
    0: (100, 100, 255),   # person — червоний
    1: (255, 200, 100),   # bicycle
    2: (255, 200, 100),   # car — блакитний
    3: (50, 200, 255),    # motorcycle — жовтогарячий
    5: (100, 255, 50),    # bus — зелений
    7: (50, 150, 255),    # truck — помаранчевий
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _downscale(img: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale до max_width зі збереженням співвідношення. Повертає contiguous."""
    h, w = img.shape[:2]
    if max_width and w > max_width:
        nh = max(1, int(round(h * max_width / float(w))))
        img = cv2.resize(img, (max_width, nh), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(img)


def _text(img, text, org, color=(255, 255, 255), scale=0.5, bg=(0, 0, 0)):
    """Текст з непрозорою підкладкою для читабельності на будь-якому фоні."""
    (tw, th), bl = cv2.getTextSize(text, _FONT, scale, 1)
    x, y = org
    cv2.rectangle(img, (x, y), (x + tw + 6, y + th + bl + 6), bg, -1)
    cv2.putText(img, text, (x + 3, y + th + 3), _FONT, scale, color, 1, cv2.LINE_AA)


def _panel(img, lines, scale=0.45):
    """Лівий-верхній багаторядковий блок (retrieval-панель тощо)."""
    y = 2
    for ln in lines:
        (tw, th), bl = cv2.getTextSize(ln, _FONT, scale, 1)
        cv2.rectangle(img, (2, y), (2 + tw + 6, y + th + bl + 4), (0, 0, 0), -1)
        cv2.putText(img, ln, (5, y + th + 2), _FONT, scale, (255, 255, 255), 1, cv2.LINE_AA)
        y += th + bl + 6


def render_yolo(frame_rgb, detections, static_mask, max_width) -> np.ndarray:
    """Кадр + напівпрозорий static_mask (динаміка) + bbox класу і confidence."""
    bgr = cv2.cvtColor(np.ascontiguousarray(frame_rgb), cv2.COLOR_RGB2BGR)
    if static_mask is not None:
        dyn = static_mask < 128  # 0 = динамічний об'єкт (замаскований)
        if bool(dyn.any()):
            overlay = bgr.copy()
            overlay[dyn] = (0, 0, 255)  # червоний BGR
            bgr = cv2.addWeighted(overlay, 0.35, bgr, 0.65, 0)
    n = 0
    for det in detections or []:
        box = det.get("bbox")
        if not box:
            continue
        cls = int(det.get("class_id", -1))
        conf = float(det.get("confidence", 0.0))
        x1, y1, x2, y2 = (int(round(v)) for v in box[:4])
        color = _CLASS_BGR.get(cls, (200, 200, 200))
        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, 2)
        label = f"{COCO_NAMES.get(cls, str(cls))} {conf:.0%}"
        ly = max(0, y1 - 18)
        _text(bgr, label, (x1, ly), color=(255, 255, 255), scale=0.45, bg=color)
        n += 1
    _text(bgr, f"YOLO  detections:{n}", (0, 0), scale=0.5)
    return _downscale(bgr, max_width)


def render_matches(collector, max_width) -> np.ndarray:
    """Query keypoints сірим, inliers зеленим, disparity-вектори q->r; лічильники."""
    bgr = cv2.cvtColor(np.ascontiguousarray(collector.rotated_frame), cv2.COLOR_RGB2BGR)
    qf = collector.query_features or {}
    kpts = qf.get("keypoints")
    n_kpts = 0
    if kpts is not None and len(kpts):
        n_kpts = len(kpts)
        for x, y in kpts:
            cv2.circle(bgr, (int(round(x)), int(round(y))), 1, (170, 170, 170), -1)
    mq = collector.mkpts_q_inliers
    mr = collector.mkpts_r_inliers
    n_in = 0
    if mq is not None and mr is not None and len(mq) == len(mr) and len(mq):
        n_in = len(mq)
        for (qx, qy), (rx, ry) in zip(mq, mr):
            p = (int(round(qx)), int(round(qy)))
            r = (int(round(rx)), int(round(ry)))
            cv2.line(bgr, p, r, (0, 180, 0), 1, cv2.LINE_AA)
            cv2.circle(bgr, p, 2, (0, 255, 0), -1)
    _text(
        bgr,
        f"kpts:{n_kpts}  matches:{collector.total_matches}  "
        f"inliers:{n_in}  rmse:{collector.rmse:.2f}",
        (0, 0),
        scale=0.45,
    )
    return _downscale(bgr, max_width)


def _pca_rgb(tokens, h_p, w_p) -> np.ndarray:
    """3 головні компоненти патч-токенів -> RGB (h_p, w_p, 3) uint8."""
    X = np.asarray(tokens, dtype=np.float32)
    X = X - X.mean(axis=0, keepdims=True)
    try:
        U, S, _ = np.linalg.svd(X, full_matrices=False)
        comps = U[:, :3] * S[:3]
    except np.linalg.LinAlgError:
        comps = X[:, :3]
    mn = comps.min(axis=0, keepdims=True)
    mx = comps.max(axis=0, keepdims=True)
    comps = (comps - mn) / (mx - mn + 1e-6)
    return (comps.reshape(h_p, w_p, 3) * 255.0).clip(0, 255).astype(np.uint8)


def render_dino(collector, max_width, pca_enabled) -> np.ndarray:
    """PCA патч-токенів поверх кадру + панель retrieval (top-k id/score, кут, масштаб)."""
    bgr = cv2.cvtColor(np.ascontiguousarray(collector.rotated_frame), cv2.COLOR_RGB2BGR)
    if pca_enabled and collector.patch_tokens is not None and collector.patch_grid is not None:
        try:
            h_p, w_p = collector.patch_grid
            if collector.patch_tokens.shape[0] == h_p * w_p:
                pca = _pca_rgb(collector.patch_tokens, h_p, w_p)
                pca_bgr = cv2.cvtColor(pca, cv2.COLOR_RGB2BGR)
                pca_big = cv2.resize(
                    pca_bgr, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST
                )
                bgr = cv2.addWeighted(pca_big, 0.6, bgr, 0.4, 0)
        except Exception:
            pass
    lines = [
        f"DINO  angle:{collector.global_angle}  scale:{collector.scale:.2f}"
        f"  score:{collector.global_score:.3f}",
        f"matched id:{collector.candidate_id}",
        "top-k retrieval:",
    ]
    for cid, sc in (collector.retrieval_candidates or [])[:8]:
        lines.append(f"  #{cid}: {sc:.3f}")
    _panel(bgr, lines)
    return _downscale(bgr, max_width)


def render_depth(collector, max_width) -> np.ndarray:
    """Colormap (INFERNO) відносної depth-мапи + значення relative scale."""
    d = np.asarray(collector.depth_map, dtype=np.float32)
    mn = float(np.nanmin(d))
    mx = float(np.nanmax(d))
    if mx - mn < 1e-6:
        norm = np.zeros(d.shape, dtype=np.uint8)
    else:
        norm = ((d - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)  # BGR
    scale = collector.depth_scale
    hdr = "Depth Anything" if scale is None else f"Depth Anything  rel.scale:{scale:.3f}"
    _text(color, hdr, (0, 0), scale=0.5)
    return _downscale(color, max_width)
