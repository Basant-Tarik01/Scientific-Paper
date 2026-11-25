"""Fixed-size LiDAR representations for learning (A7 step 2: "start with one")."""
from __future__ import annotations

import numpy as np


def scan_to_polar(ranges, range_max: float, n_bins: int = 360) -> np.ndarray:
    """Min-pooled polar range vector normalised to [0, 1]; no-return -> 1."""
    r = np.where(np.isfinite(ranges), ranges, range_max)
    r = np.asarray(r, np.float32).reshape(n_bins, -1).min(axis=1)
    return np.clip(r / range_max, 0.0, 1.0)


def scan_to_bev(ranges, angles, size: int = 64, resolution: float = 0.1, lidar_xy=(0.0, 0.0)) -> np.ndarray:
    """Robot-centred bird's-eye-view occupancy grid (size x size, robot at the centre, +x up).

    Cells containing a LiDAR return are 1, everything else 0. ``size * resolution`` metres per side.
    """
    grid = np.zeros((size, size), np.float32)
    r = np.asarray(ranges)
    ok = np.isfinite(r)
    px = lidar_xy[0] + r[ok] * np.cos(np.asarray(angles)[ok])
    py = lidar_xy[1] + r[ok] * np.sin(np.asarray(angles)[ok])
    half = size * resolution / 2
    row = ((half - px) / resolution).astype(int)       # +x (forward) is up
    col = ((half - py) / resolution).astype(int)       # +y (left) is left
    keep = (row >= 0) & (row < size) & (col >= 0) & (col < size)
    grid[row[keep], col[keep]] = 1.0
    return grid
