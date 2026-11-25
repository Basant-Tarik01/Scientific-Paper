"""Gymnasium environment for health-conditioned RL navigation (A1) on top of :class:`WarehouseSim2D`.

Observation (``gymnasium.spaces.Dict``)::

    scan        (n_beams,)  LiDAR ranges with faults, min-pooled to n_beams, /range_max, no-return = 1
    goal        (3,)        [distance / 20 (clipped to 1), cos(bearing), sin(bearing)] in the robot frame
    velocity    (2,)        measured (odometry, affected by encoder faults) v / v_max, w / w_max
    last_action (2,)        previous action in [-1, 1]
    peds        (5 * K,)    optional: K nearest workers within 6 m: rel. x, y, vx, vy (robot frame, /6) + valid mask
    payload     (1,)        optional context: payload / max payload
    health      (9,)        PRIVILEGED (teacher only, ``privileged=True``): degradation vector, see faults.HEALTH_VECTOR_NAMES

``info["health"]`` always contains the true degradation vector (for distillation / auxiliary losses),
``info["episode_metrics"]`` the metric summary at the end of the episode.

Action: Box([-1, 1]^2) -> v in [-max_rev_vel, max_lin_vel], w in [-max_ang_vel, max_ang_vel].
"""
from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .faults import HEALTH_VECTOR_NAMES
from .geometry import to_local
from .sim2d import WarehouseSim2D

DEFAULT_REWARD = {
    "progress": 1.0,          # per metre of progress towards the goal
    "goal": 10.0,
    "collision": -10.0,
    "near_miss": -0.5,        # per step, scaled by how deep the closest worker is in the near-miss band
    "smoothness": -0.05,      # per unit of |a_t - a_{t-1}|
    "time": -0.01,            # per step
}


class WarehouseNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 10}

    def __init__(self, scenario: str = "a1_train", world: Optional[str] = None, n_beams: int = 180,
                 privileged: bool = False, include_peds: bool = True, include_payload: bool = True,
                 k_peds: int = 6, reward: Optional[dict] = None, render_mode: Optional[str] = None,
                 seed: Optional[int] = None):
        super().__init__()
        self.sim = WarehouseSim2D(world=world, scenario=scenario, seed=seed)
        self.robot = self.sim.robot
        if self.robot.lidar_samples % n_beams:
            raise ValueError(f"n_beams must divide {self.robot.lidar_samples}")
        self.n_beams = n_beams
        self.privileged = privileged
        self.include_peds = include_peds
        self.include_payload = include_payload
        self.k_peds = k_peds
        self.reward_cfg = dict(DEFAULT_REWARD, **(reward or {}))
        self.render_mode = render_mode
        obs = {
            "scan": spaces.Box(0.0, 1.0, (n_beams,), np.float32),
            "goal": spaces.Box(-1.0, 1.0, (3,), np.float32),
            "velocity": spaces.Box(-2.0, 2.0, (2,), np.float32),
            "last_action": spaces.Box(-1.0, 1.0, (2,), np.float32),
        }
        if include_peds:
            obs["peds"] = spaces.Box(-5.0, 5.0, (5 * k_peds,), np.float32)
        if include_payload:
            obs["payload"] = spaces.Box(0.0, 1.0, (1,), np.float32)
        if privileged:
            obs["health"] = spaces.Box(-1.0, 1.0, (len(HEALTH_VECTOR_NAMES),), np.float32)
        self.observation_space = spaces.Dict(obs)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._fig = None

    # ------------------------------------------------------------------ helpers
    def _action_to_cmd(self, a):
        a = np.clip(np.asarray(a, dtype=float), -1.0, 1.0)
        r = self.robot
        v = a[0] * (r.max_lin_vel if a[0] >= 0 else r.max_rev_vel)
        return float(v), float(a[1] * r.max_ang_vel)

    def _goal_features(self, pose):
        dx, dy = self.sim.goal - pose[:2]
        dist = float(np.hypot(dx, dy))
        bearing = np.arctan2(dy, dx) - pose[2]
        return np.array([min(dist / 20.0, 1.0), np.cos(bearing), np.sin(bearing)], np.float32), dist

    def _obs(self, s):
        r = self.robot
        scan = np.where(np.isfinite(s["scan"]), s["scan"], r.lidar_range_max)
        scan = scan.reshape(self.n_beams, -1).min(axis=1) / r.lidar_range_max
        goal, _ = self._goal_features(s["pose"])
        obs = {
            "scan": np.clip(scan, 0, 1).astype(np.float32),
            "goal": goal,
            "velocity": np.array([s["odom_vel"][0] / r.max_lin_vel, s["odom_vel"][1] / r.max_ang_vel], np.float32),
            "last_action": self._last_action.astype(np.float32),
        }
        if self.include_peds:
            feat = np.zeros((self.k_peds, 5), np.float32)
            peds = s["peds"]
            if len(peds):
                x, y, th = s["pose"]
                lx, ly = to_local(x, y, th, peds[:, 0], peds[:, 1])
                c, sn = np.cos(th), np.sin(th)
                lvx = c * peds[:, 2] + sn * peds[:, 3]
                lvy = -sn * peds[:, 2] + c * peds[:, 3]
                d = np.hypot(lx, ly)
                order = [i for i in np.argsort(d) if d[i] < 6.0][: self.k_peds]
                for k, i in enumerate(order):
                    feat[k] = (lx[i] / 6, ly[i] / 6, lvx[i] / 6, lvy[i] / 6, 1.0)
            obs["peds"] = np.clip(feat.ravel(), -5.0, 5.0)
        if self.include_payload:
            obs["payload"] = np.array([s["payload"] / r.payload_max_mass], np.float32)
        if self.privileged:
            obs["health"] = s["health"].astype(np.float32)
        return obs

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        s = self.sim.reset(seed=seed, options=options)
        self._last_action = np.zeros(2, np.float32)
        self._prev_goal_dist = self._goal_features(s["pose"])[1]
        self._state = s
        return self._obs(s), self._info(s)

    def step(self, action):
        action = np.clip(np.asarray(action, np.float32), -1, 1)
        s = self.sim.step(*self._action_to_cmd(action))
        rc = self.reward_cfg
        _, dist = self._goal_features(s["pose"])
        reward = rc["progress"] * (self._prev_goal_dist - dist) + rc["time"]
        reward += rc["smoothness"] * float(np.abs(action - self._last_action).sum())
        nm = self.sim.metrics.near_miss_distance
        if len(s["ped_distances"]):
            dmin = float(np.min(s["ped_distances"]))
            if 0 < dmin < nm:
                reward += rc["near_miss"] * (1.0 - dmin / nm)
        outcome = s["outcome"]
        terminated = outcome in ("success", "collision_ped", "collision_static")
        truncated = outcome == "timeout"
        if outcome == "success":
            reward += rc["goal"]
        elif outcome in ("collision_ped", "collision_static"):
            reward += rc["collision"]
        self._prev_goal_dist = dist
        self._last_action = action
        self._state = s
        info = self._info(s)
        if terminated or truncated:
            info["episode_metrics"] = self.sim.metrics.summary(outcome)
        if self.render_mode == "human":
            self.render()
        return self._obs(s), float(reward), terminated, truncated, info

    def _info(self, s):
        return {"health": s["health"], "payload": s["payload"], "faults": list(self.sim.health.faults),
                "severities": list(self.sim.health.severities), "density": str(self.sim.density),
                "outcome": s["outcome"]}

    def render(self):
        from .render import render_sim
        if self.render_mode == "human":
            import matplotlib.pyplot as plt
            if self._fig is None:
                plt.ion()
                self._fig = plt.figure(figsize=(10, 7))
            self._fig.clf()
            render_sim(self.sim, ax=self._fig.gca())
            plt.pause(0.001)
            return None
        return render_sim(self.sim, as_array=True)

    def close(self):
        if self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
