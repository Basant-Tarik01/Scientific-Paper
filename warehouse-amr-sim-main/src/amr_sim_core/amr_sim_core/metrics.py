"""Episode metrics shared by the 2D simulator and the ROS 2 ``metrics_monitor`` node.

Definitions (fixed in advance, as both proposals require):
  * distances are measured between the robot *footprint rectangle* and the pedestrian *body circle*
    (surface to surface) or the static obstacle boxes;
  * collision          : distance <= 0 to a pedestrian (``collision_ped``) or obstacle (``collision_static``);
  * near-miss event    : a pedestrian enters the band 0 < d < ``near_miss_distance`` (default 0.5 m);
                          counted once per pedestrian per entry (re-armed when it leaves the band
                          by more than 0.1 m hysteresis);
  * near-miss time     : total time with at least one pedestrian inside the band;
  * success            : goal reached within ``goal_tolerance`` without any collision before the timeout;
  * collision_speed    : robot speed at the first collision (about 0 = a worker walked into a stopped robot).
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np


class EpisodeMetrics:
    def __init__(self, near_miss_distance: float = 0.5, hysteresis: float = 0.1):
        self.near_miss_distance = float(near_miss_distance)
        self.hysteresis = float(hysteresis)
        self.reset(0.0, (0.0, 0.0), (0.0, 0.0))

    def reset(self, t: float, start_xy, goal_xy):
        self.t0 = self.t = float(t)
        self.start = np.asarray(start_xy, float)[:2]
        self.goal = np.asarray(goal_xy, float)[:2]
        self.last_xy = self.start.copy()
        self.path_length = 0.0
        self.min_ped_distance = np.inf
        self.min_obstacle_distance = np.inf
        self.near_miss_events = 0
        self.near_miss_time = 0.0
        self.ped_collisions = 0
        self.static_collisions = 0
        self.first_collision: Optional[str] = None
        self.collision_speed: Optional[float] = None
        self._in_band: Dict[int, bool] = {}
        self._in_contact: Dict[int, bool] = {}
        self._static_contact = False
        self.speed_sum = 0.0
        self.steps = 0

    def update(self, t: float, xy, ped_distances: Iterable[float], ped_ids: Iterable[int],
               obstacle_distance: float, speed: float = 0.0):
        dt = max(0.0, float(t) - self.t)
        self.t = float(t)
        xy = np.asarray(xy, float)[:2]
        self.path_length += float(np.hypot(*(xy - self.last_xy)))
        self.last_xy = xy
        self.speed_sum += abs(speed)
        self.steps += 1
        any_band = False
        for pid, d in zip(ped_ids, ped_distances):
            d = float(d)
            self.min_ped_distance = min(self.min_ped_distance, d)
            if d <= 0.0:
                if not self._in_contact.get(pid, False):
                    self.ped_collisions += 1
                    self._first("collision_ped", speed)
                self._in_contact[pid] = True
            elif d > self.hysteresis:
                self._in_contact[pid] = False
            if 0.0 < d < self.near_miss_distance:
                any_band = True
                if not self._in_band.get(pid, False):
                    self.near_miss_events += 1
                self._in_band[pid] = True
            elif d > self.near_miss_distance + self.hysteresis:
                self._in_band[pid] = False
        if any_band:
            self.near_miss_time += dt
        od = float(obstacle_distance)
        self.min_obstacle_distance = min(self.min_obstacle_distance, od)
        if od <= 0.0:
            if not self._static_contact:
                self.static_collisions += 1
                self._first("collision_static", speed)
            self._static_contact = True
        elif od > self.hysteresis:
            self._static_contact = False

    def _first(self, kind: str, speed: float):
        if self.first_collision is None:
            self.first_collision = kind
            self.collision_speed = abs(float(speed))

    @property
    def collided(self) -> bool:
        return self.first_collision is not None

    def goal_distance(self) -> float:
        return float(np.hypot(*(self.last_xy - self.goal)))

    def summary(self, outcome: str) -> dict:
        duration = self.t - self.t0
        straight = float(np.hypot(*(self.goal - self.start)))
        return {
            "outcome": outcome,
            "success": outcome == "success",
            "time": round(duration, 3),
            "path_length": round(self.path_length, 3),
            "straight_line_distance": round(straight, 3),
            "goal_distance_final": round(self.goal_distance(), 3),
            "mean_speed": round(self.speed_sum / max(self.steps, 1), 3),
            "min_ped_distance": None if np.isinf(self.min_ped_distance) else round(self.min_ped_distance, 3),
            "min_obstacle_distance": None if np.isinf(self.min_obstacle_distance) else round(self.min_obstacle_distance, 3),
            "near_miss_events": self.near_miss_events,
            "near_miss_time": round(self.near_miss_time, 3),
            "ped_collisions": self.ped_collisions,
            "static_collisions": self.static_collisions,
            "collision_speed": None if self.collision_speed is None else round(self.collision_speed, 3),
        }
