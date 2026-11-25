"""Vectorised 2D geometry used by the simulators, the pedestrian model and the metrics.

All static obstacles are axis-aligned boxes stored as an (M, 4) array ``[xmin, ymin, xmax, ymax]``.
The robot footprint is an oriented rectangle; pedestrians are circles.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-12


def wrap_angle(a):
    """Wrap angle(s) to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def point_aabb_vector(px, py, boxes):
    """Vector from the closest point of every box to every point.

    Returns (dx, dy, dist) with shape (N, M). Points inside a box get distance 0.
    """
    px = np.atleast_1d(np.asarray(px, dtype=float))[:, None]
    py = np.atleast_1d(np.asarray(py, dtype=float))[:, None]
    cx = np.clip(px, boxes[None, :, 0], boxes[None, :, 2])
    cy = np.clip(py, boxes[None, :, 1], boxes[None, :, 3])
    dx = px - cx
    dy = py - cy
    return dx, dy, np.hypot(dx, dy)


def point_aabb_distance(px, py, boxes):
    """(N, M) distance from points to boxes (0 inside)."""
    if len(boxes) == 0:
        return np.full((np.size(px), 0), np.inf)
    return point_aabb_vector(px, py, boxes)[2]


def segment_clearance(p0, p1, boxes, step=0.1):
    """Minimum distance between the segment p0-p1 and the boxes (sampled every ``step`` m)."""
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / step)) + 1)
    s = np.linspace(0.0, 1.0, n)[:, None]
    pts = p0 + s * (p1 - p0)
    return float(point_aabb_distance(pts[:, 0], pts[:, 1], boxes).min())


def raycast(ox, oy, angles, max_range, boxes=None, circles=None, radii=None):
    """Cast rays from (ox, oy) at world-frame ``angles`` against boxes and circles.

    boxes:   (M, 4) axis-aligned boxes
    circles: (K, 2) centres, radii: (K,) radii
    Returns ranges (N,), clipped at ``max_range``.
    """
    angles = np.asarray(angles, dtype=float)
    dx = np.cos(angles)
    dy = np.sin(angles)
    ranges = np.full(angles.shape, max_range, dtype=float)

    if boxes is not None and len(boxes):
        sdx = np.where(np.abs(dx) < _EPS, _EPS, dx)[:, None]
        sdy = np.where(np.abs(dy) < _EPS, _EPS, dy)[:, None]
        tx1 = (boxes[None, :, 0] - ox) / sdx
        tx2 = (boxes[None, :, 2] - ox) / sdx
        ty1 = (boxes[None, :, 1] - oy) / sdy
        ty2 = (boxes[None, :, 3] - oy) / sdy
        tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
        tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
        hit = tmax >= np.maximum(tmin, 0.0)
        t = np.where(hit, np.maximum(tmin, 0.0), np.inf)
        ranges = np.minimum(ranges, t.min(axis=1))

    if circles is not None and len(circles):
        fx = ox - circles[None, :, 0]
        fy = oy - circles[None, :, 1]
        b = fx * dx[:, None] + fy * dy[:, None]
        c = fx * fx + fy * fy - np.asarray(radii)[None, :] ** 2
        disc = b * b - c
        sq = np.sqrt(np.maximum(disc, 0.0))
        t = -b - sq
        inside = c <= 0.0
        t = np.where(inside, 0.0, t)
        valid = (disc >= 0.0) & ((t >= 0.0) | inside)
        t = np.where(valid, t, np.inf)
        ranges = np.minimum(ranges, t.min(axis=1))

    return ranges


def rect_corners(x, y, yaw, half_l, half_w):
    """(4, 2) corners of an oriented rectangle centred at (x, y)."""
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array([[half_l, half_w], [half_l, -half_w], [-half_l, -half_w], [-half_l, half_w]])
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([x, y])


def to_local(x, y, yaw, px, py):
    """Transform world points into the frame (x, y, yaw)."""
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.asarray(px) - x
    dy = np.asarray(py) - y
    return c * dx + s * dy, -s * dx + c * dy


def rect_circle_distance(x, y, yaw, half_l, half_w, centers, radii):
    """Signed-ish distance from the oriented rectangle to circles (<= 0 means contact)."""
    if len(centers) == 0:
        return np.zeros(0)
    lx, ly = to_local(x, y, yaw, centers[:, 0], centers[:, 1])
    ddx = np.maximum(np.abs(lx) - half_l, 0.0)
    ddy = np.maximum(np.abs(ly) - half_w, 0.0)
    return np.hypot(ddx, ddy) - np.asarray(radii)


def rect_aabb_distance(x, y, yaw, half_l, half_w, boxes):
    """Exact distance from an oriented rectangle to each axis-aligned box (0 if overlapping)."""
    if len(boxes) == 0:
        return np.zeros(0)
    corners = rect_corners(x, y, yaw, half_l, half_w)
    # rectangle corners -> boxes
    d1 = point_aabb_distance(corners[:, 0], corners[:, 1], boxes).min(axis=0)
    # box corners -> rectangle (in rectangle frame)
    bx = boxes[:, [0, 2, 2, 0]]
    by = boxes[:, [1, 1, 3, 3]]
    lx, ly = to_local(x, y, yaw, bx, by)
    d2 = np.hypot(np.maximum(np.abs(lx) - half_l, 0.0), np.maximum(np.abs(ly) - half_w, 0.0)).min(axis=1)
    dist = np.minimum(d1, d2)
    # separating-axis test for the crossing case where no corner lies inside the other shape
    c, s = np.cos(yaw), np.sin(yaw)
    overlap = np.ones(len(boxes), dtype=bool)
    for ax, ay in ((1.0, 0.0), (0.0, 1.0), (c, s), (-s, c)):
        rp = corners[:, 0] * ax + corners[:, 1] * ay
        bp = bx * ax + by * ay
        overlap &= ~((rp.max() < bp.min(axis=1)) | (bp.max(axis=1) < rp.min()))
    return np.where(overlap, 0.0, dist)
