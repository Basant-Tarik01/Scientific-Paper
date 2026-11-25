"""Warehouse workers: social-force model (Helbing & Molnar 1995; Helbing, Farkas & Vicsek 2000)
with route planning on the world's waypoint graph and simple task behaviour.

Behaviour of each worker
  * picks a goal (biased towards picking aisles, packing stations and the dock, see world YAML),
  * walks there along the shortest graph route (with lateral jitter so people do not walk in a line),
  * on arrival dwells (picking / packing) with probability ``dwell_prob`` for ``dwell_time`` seconds,
  * then chooses a new goal.
A fraction ``distracted_fraction`` of workers ignore the robot (no robot repulsion): they model
people who are looking at a scanner or carrying a box and are the main stress test for safety.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .geometry import point_aabb_vector


class Crowd:
    # social-force constants
    TAU = 0.5            # relaxation time (s)
    A_PED, B_PED = 2.0, 0.3
    LAMBDA = 0.35        # anisotropy (1 = isotropic)
    A_WALL, B_WALL = 5.0, 0.15
    A_ROBOT, B_ROBOT = 4.0, 0.5
    WAYPOINT_TOL = 0.6

    def __init__(self, world, spec: Optional[dict] = None):
        spec = spec or {}
        self.world = world
        self.speed_range = tuple(spec.get("speed", (0.9, 1.3)))
        self.radius_range = tuple(spec.get("radius", (0.22, 0.30)))
        self.dwell_prob = float(spec.get("dwell_prob", 0.4))
        self.dwell_time = tuple(spec.get("dwell_time", (2.0, 8.0)))
        self.distracted_fraction = float(spec.get("distracted_fraction", 0.0))
        self.zone_weights = dict(world.ped_spec.get("zone_goal_weights", {}))
        self.max_peds = int(world.ped_spec.get("max", 16))
        self.n = 0
        self.t = 0.0
        self._alloc(0)

    def _alloc(self, n):
        self.pos = np.zeros((n, 2))
        self.vel = np.zeros((n, 2))
        self.yaw = np.zeros(n)
        self.radius = np.zeros(n)
        self.v0 = np.zeros(n)
        self.aware = np.ones(n, dtype=bool)
        self.dwell_until = np.zeros(n)
        self.routes: List[np.ndarray] = [np.zeros((0, 2)) for _ in range(n)]
        self.wp = np.zeros(n, dtype=int)
        self.last_progress_t = np.zeros(n)
        self.best_dist = np.full(n, np.inf)

    # ------------------------------------------------------------------ setup
    def density_to_count(self, density) -> int:
        if isinstance(density, (int, np.integer)):
            return int(density)
        return int(self.world.ped_spec.get("densities", {}).get(str(density), 0))

    def reset(self, n: int, rng: np.random.Generator, keep_out: Optional[List[tuple]] = None):
        """Spawn ``n`` workers on graph nodes, at least 1 m apart and away from ``keep_out``
        points [(x, y, radius), ...] (e.g. the robot start and goal)."""
        n = int(min(n, self.max_peds))
        self.n, self.t = n, 0.0
        self._alloc(n)
        self.rng = rng
        nodes = self.world.graph.nodes
        keep_out = keep_out or []
        chosen: List[np.ndarray] = []
        for idx in rng.permutation(len(nodes)):
            if len(chosen) == n:
                break
            p = nodes[idx] + rng.uniform(-0.25, 0.25, 2)
            if any(np.hypot(p[0] - x, p[1] - y) < r for x, y, r in keep_out):
                continue
            if any(np.hypot(*(p - c)) < 1.0 for c in chosen):
                continue
            chosen.append(p)
        self.n = len(chosen)
        self._alloc(self.n)
        for i, p in enumerate(chosen):
            self.pos[i] = p
            self.radius[i] = rng.uniform(*self.radius_range)
            self.v0[i] = rng.uniform(*self.speed_range)
            self.aware[i] = rng.random() >= self.distracted_fraction
            if rng.random() < self.dwell_prob:          # some start while picking
                self.dwell_until[i] = rng.uniform(0.0, self.dwell_time[1])
            self._new_route(i)

    def _sample_goal_node(self) -> int:
        rng = self.rng
        zones = [z for z in self.zone_weights if z in self.world.zones]
        weights = np.array([self.zone_weights[z] for z in zones], dtype=float)
        r = rng.random()
        if zones and r < weights.sum():
            z = zones[int(np.searchsorted(np.cumsum(weights), r))]
            cand = self.world.nodes_in_zone(z)
            if len(cand):
                return int(rng.choice(cand))
        return int(rng.integers(len(self.world.graph.nodes)))

    def _new_route(self, i: int):
        g = self.world.graph
        start = g.nearest(*self.pos[i], boxes=self.world.aabbs)
        for _ in range(10):
            path = g.shortest_path(start, self._sample_goal_node())
            if len(path) >= 2:
                break
        pts = g.nodes[path[1:]] if len(path) >= 2 else g.nodes[[start]]
        # lateral jitter, limited by the local clearance
        jitter = self.rng.uniform(-1, 1, pts.shape) * np.minimum(0.35, g.clearance[path[1:] if len(path) >= 2 else [start]] - 0.4)[:, None]
        self.routes[i] = pts + jitter
        self.wp[i] = 0
        self.last_progress_t[i] = self.t
        self.best_dist[i] = np.inf

    # ------------------------------------------------------------------ dynamics
    def step(self, dt: float, robot=None):
        """Advance the crowd by ``dt``. ``robot`` = (x, y, vx, vy, radius) or None."""
        n = self.n
        self.t += dt
        if n == 0:
            return
        p, v = self.pos, self.vel
        targets = np.array([self.routes[i][min(self.wp[i], len(self.routes[i]) - 1)] for i in range(n)])
        dwelling = self.dwell_until > self.t
        to_t = targets - p
        dist_t = np.linalg.norm(to_t, axis=1) + 1e-9
        e = to_t / dist_t[:, None]
        desired = np.where(dwelling[:, None], 0.0, self.v0[:, None] * e)
        force = (desired - v) / self.TAU

        heading = np.where(np.linalg.norm(v, axis=1, keepdims=True) > 0.1,
                           v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9), e)

        # pedestrian-pedestrian repulsion
        if n > 1:
            d = p[:, None, :] - p[None, :, :]
            dist = np.linalg.norm(d, axis=2) + np.eye(n) * 1e6
            nij = d / dist[:, :, None]
            rij = self.radius[:, None] + self.radius[None, :]
            mag = self.A_PED * np.exp((rij - dist) / self.B_PED)
            cosphi = -np.einsum("ik,ijk->ij", heading, nij)
            w = self.LAMBDA + (1 - self.LAMBDA) * (1 + cosphi) / 2
            force += np.sum((mag * w)[:, :, None] * nij, axis=1)

        # obstacle repulsion
        dx, dy, dist = point_aabb_vector(p[:, 0], p[:, 1], self.world.aabbs)
        safe = np.maximum(dist, 1e-6)
        mag = np.where(dist < 3.0, self.A_WALL * np.exp((self.radius[:, None] - dist) / self.B_WALL), 0.0)
        force[:, 0] += np.sum(mag * dx / safe, axis=1)
        force[:, 1] += np.sum(mag * dy / safe, axis=1)

        # robot repulsion (aware workers only), anticipating the robot's motion by 0.5 s
        if robot is not None:
            rx, ry, rvx, rvy, rr = robot
            rel = p - np.array([rx + 0.5 * rvx, ry + 0.5 * rvy])
            dist = np.linalg.norm(rel, axis=1) + 1e-9
            nrm = rel / dist[:, None]
            mag = self.A_ROBOT * np.exp((self.radius + rr - dist) / self.B_ROBOT)
            cosphi = -np.sum(heading * nrm, axis=1)
            w = self.LAMBDA + (1 - self.LAMBDA) * (1 + cosphi) / 2
            force += np.where(self.aware[:, None], (mag * w)[:, None] * nrm, 0.0)

        v = v + force * dt
        speed = np.linalg.norm(v, axis=1)
        vmax = 1.3 * self.v0
        v = np.where((speed > vmax)[:, None], v * (vmax / (speed + 1e-9))[:, None], v)
        v = np.where(dwelling[:, None] & (speed < 0.05)[:, None], 0.0, v)
        p = p + v * dt

        # hard constraint: never inside static obstacles
        dx, dy, dist = point_aabb_vector(p[:, 0], p[:, 1], self.world.aabbs)
        pen = self.radius[:, None] - dist
        j = np.argmax(pen, axis=1)
        rows = np.arange(n)
        worst = pen[rows, j]
        fix = worst > 0
        if np.any(fix):
            ddx, ddy, dd = dx[rows, j], dy[rows, j], np.maximum(dist[rows, j], 1e-6)
            p[fix, 0] += (ddx / dd * worst)[fix]
            p[fix, 1] += (ddy / dd * worst)[fix]
            # remove the velocity component into the obstacle
            nx, ny = ddx / dd, ddy / dd
            vn = v[:, 0] * nx + v[:, 1] * ny
            into = fix & (vn < 0)
            v[into, 0] -= vn[into] * nx[into]
            v[into, 1] -= vn[into] * ny[into]

        self.pos, self.vel = p, v
        moving = np.linalg.norm(v, axis=1) > 0.1
        self.yaw = np.where(moving, np.arctan2(v[:, 1], v[:, 0]), self.yaw)

        # waypoint bookkeeping, dwelling, stuck detection
        for i in range(n):
            if dwelling[i]:
                self.last_progress_t[i] = self.t
                continue
            route = self.routes[i]
            d = float(np.hypot(*(route[self.wp[i]] - p[i])))
            if d < self.best_dist[i] - 0.2:
                self.best_dist[i] = d
                self.last_progress_t[i] = self.t
            if d < self.WAYPOINT_TOL:
                self.wp[i] += 1
                self.best_dist[i] = np.inf
                if self.wp[i] >= len(route):
                    if self.rng.random() < self.dwell_prob:
                        self.dwell_until[i] = self.t + self.rng.uniform(*self.dwell_time)
                    self._new_route(i)
            elif self.t - self.last_progress_t[i] > 6.0:    # stuck: re-plan
                self._new_route(i)

    # ------------------------------------------------------------------ export
    def states(self) -> np.ndarray:
        """(n, 6) array: x, y, vx, vy, radius, aware."""
        return np.column_stack([self.pos, self.vel, self.radius, self.aware.astype(float)]) if self.n else np.zeros((0, 6))
