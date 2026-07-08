"""Frame-rotation geometry shared by localization and optical flow.

np.rot90(frame, k=K) rotates the frame counter-clockwise by K*90 degrees. These
helpers map a displacement vector / point measured in the ORIGINAL frame into the
rotated frame's coordinate system (where the homography H was built).

Values verified numerically against np.rot90 (pixel-centred convention):
  k=1 (90):  (x,y) -> (y, W-1-x)      vector (dx,dy) -> (dy, -dx)
  k=2 (180): (x,y) -> (W-1-x, H-1-y)  vector (dx,dy) -> (-dx, -dy)
  k=3 (270): (x,y) -> (H-1-y, x)      vector (dx,dy) -> (-dy, dx)
"""

from __future__ import annotations

# angle: (a, b, c, d) -> new_dx = a*dx + b*dy, new_dy = c*dx + d*dy
_ROTATION_VEC: dict[int, tuple[int, int, int, int]] = {
    0: (1, 0, 0, 1),
    90: (0, 1, -1, 0),
    180: (-1, 0, 0, -1),
    270: (0, -1, 1, 0),
}


def _rotate_point_np90(
    x: float, y: float, w: float, h: float, angle: int
) -> tuple[float, float]:
    """Map point (x, y) of a w×h frame into np.rot90(frame, k=angle//90) coords."""
    if angle == 90:
        return y, (w - 1.0) - x
    if angle == 180:
        return (w - 1.0) - x, (h - 1.0) - y
    if angle == 270:
        return (h - 1.0) - y, x
    return x, y
