"""Scripted driver: global route on the waypoint graph + pure-pursuit tracking.

Used (a) as a simple classical baseline in the 2D simulator and (b) as the data-generation driver for
A7, where ``speed_scale``, ``speed_change_prob`` and ``stop_prob`` randomise the speed profile.
It is *not* human-aware: combine it with a safety layer from :mod:`amr_sim_core.safety`.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .geometry import wrap_angle


class PathFollower:
    def __init__(self, world, robot, lookahead: float = 0.9, speed_scale=(0.8, 0.8),
                 speed_change_prob: float = 0.0, stop_prob: float = 0.0, stop_time=(1.0, 3.0)):
        self.world = world
        self.robot = robot
        self.lookahead = lookahead
        self.speed_scale = tuple(speed_scale)
        self.speed_change_prob = speed_change_prob
        self.stop_prob = stop_prob
        self.stop_time = tuple(stop_time)
        self.path = np.zeros((0, 2))

    @classmethod
    def from_scenario(cls, world, robot, scenario: dict):
        c = scenario.get("controller", {})
        return cls(world, robot, speed_scale=c.get("speed_scale", (0.8, 0.8)),
                   speed_change_prob=c.get("speed_change_prob", 0.0), stop_prob=c.get("stop_prob", 0.0))

    def reset(self, start_pose, goal_xy, rng: Optional[np.random.Generator] = None):
        self.rng = rng or np.random.default_rng()
        g = self.world.graph
        a = g.nearest(start_pose[0], start_pose[1], boxes=self.world.aabbs)
        b = g.nearest(goal_xy[0], goal_xy[1], boxes=self.world.aabbs)
        nodes = g.nodes[g.shortest_path(a, b)] if a != b else g.nodes[[a]]
        self.path = np.vstack([np.asarray(start_pose[:2])[None], nodes, np.asarray(goal_xy)[None]])
        self.goal = np.asarray(goal_xy, float)
        self.progress = 0
        self.cruise = float(self.rng.uniform(*self.speed_scale))
        self.stop_until = -1.0

    def act(self, pose, t: float = 0.0):
        x, y, th = pose
        r = self.robot
        # randomised speed profile (A7 data generation)
        if self.speed_change_prob and self.rng.random() < self.speed_change_prob:
            self.cruise = float(self.rng.uniform(*self.speed_scale))
        if self.stop_prob and t > self.stop_until and self.rng.random() < self.stop_prob:
            self.stop_until = t + float(self.rng.uniform(*self.stop_time))
        # advance along the path
        d = np.hypot(self.path[:, 0] - x, self.path[:, 1] - y)
        window = slice(self.progress, min(self.progress + 6, len(self.path)))
        self.progress = self.progress + int(np.argmin(d[window]))
        target = self.path[-1]
        for k in range(self.progress, len(self.path)):
            if d[k] >= self.lookahead:
                target = self.path[k]
                break
        alpha = float(wrap_angle(np.arctan2(target[1] - y, target[0] - x) - th))
        dist_goal = float(np.hypot(*(self.goal - (x, y))))
        if abs(alpha) > 0.9:                       # turn in place
            return 0.0, float(np.clip(1.5 * alpha, -r.max_ang_vel, r.max_ang_vel))
        v = self.cruise * r.max_lin_vel * max(0.0, np.cos(alpha))
        v = min(v, 0.8 * dist_goal + 0.05)
        if t < self.stop_until:
            v = 0.0
        w = 2.0 * max(v, 0.2) * np.sin(alpha) / max(self.lookahead, 1e-3)
        return float(v), float(np.clip(w, -r.max_ang_vel, r.max_ang_vel))
