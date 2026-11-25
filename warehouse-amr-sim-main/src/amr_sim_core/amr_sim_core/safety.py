"""Reactive safety-layer baselines operating on the LiDAR scan.

* :class:`ProtectiveField` - industrial-style protective + warning field (A7 baseline "fixed protective
  field"). Field length follows the stopping-distance principle
      S = v * (t_scanner + t_control) + v^2 / (2 a_brake) + Z
  evaluated at the nominal maximum speed and nominal braking (``speed_dependent=False``) or at the
  current speed (``speed_dependent=True``, i.e. field switching). It always assumes the *nominal*
  braking deceleration: it does not know about payload or degradation.
* :class:`TTCRule` - reactive time-to-collision rule (A7 baseline).

Both expose ``filter(v_cmd, w_cmd, ranges, angles, v_now) -> (v, w, info)``.
"""
from __future__ import annotations

import numpy as np


def scan_points(ranges, angles, lidar_xy=(0.0, 0.0)):
    r = np.asarray(ranges, float)
    ok = np.isfinite(r)
    a = np.asarray(angles)[ok]
    r = r[ok]
    return np.column_stack([lidar_xy[0] + r * np.cos(a), lidar_xy[1] + r * np.sin(a)])


class ProtectiveField:
    def __init__(self, robot, t_response: float = 0.2, margin: float = 0.2, side_margin: float = 0.1,
                 warning_factor: float = 2.0, warning_speed: float = 0.3, speed_dependent: bool = False):
        self.robot = robot
        self.t_response = t_response
        self.margin = margin
        self.side_margin = side_margin
        self.warning_factor = warning_factor
        self.warning_speed = warning_speed
        self.speed_dependent = speed_dependent

    def field_length(self, v: float) -> float:
        a = self.robot.max_lin_acc                     # nominal braking assumption
        return v * self.t_response + v * v / (2 * a) + self.margin

    def filter(self, v_cmd, w_cmd, ranges, angles, v_now=0.0):
        r = self.robot
        v_ref = abs(v_now) if self.speed_dependent else r.max_lin_vel
        L = self.field_length(v_ref)
        pts = scan_points(ranges, angles, (r.lidar_x, r.lidar_y))
        hl, hw = r.half_length, r.half_width + self.side_margin
        ahead = pts[(np.abs(pts[:, 1]) <= hw) & (pts[:, 0] > hl)] if len(pts) else pts
        dist = float(ahead[:, 0].min() - hl) if len(ahead) else np.inf
        state = "clear"
        v = v_cmd
        if dist <= L:
            state, v = "protective_stop", min(v_cmd, 0.0)
        elif dist <= L * self.warning_factor:
            state, v = "warning", min(v_cmd, self.warning_speed)
        return v, w_cmd, {"state": state, "field_length": L, "front_distance": dist}


class TTCRule:
    def __init__(self, robot, t_stop: float = 1.0, t_slow: float = 2.5, side_margin: float = 0.1):
        self.robot = robot
        self.t_stop = t_stop
        self.t_slow = t_slow
        self.side_margin = side_margin

    def filter(self, v_cmd, w_cmd, ranges, angles, v_now=0.0):
        r = self.robot
        pts = scan_points(ranges, angles, (r.lidar_x, r.lidar_y))
        hl, hw = r.half_length, r.half_width + self.side_margin
        ahead = pts[(np.abs(pts[:, 1]) <= hw) & (pts[:, 0] > hl)] if len(pts) else pts
        dist = float(ahead[:, 0].min() - hl) if len(ahead) else np.inf
        v_ref = max(abs(v_now), abs(v_cmd), 1e-3)
        ttc = dist / v_ref
        v, state = v_cmd, "clear"
        if ttc < self.t_stop:
            v, state = min(v_cmd, 0.0), "stop"
        elif ttc < self.t_slow:
            v, state = min(v_cmd, dist / self.t_slow), "slow"
        return v, w_cmd, {"state": state, "ttc": ttc, "front_distance": dist}
