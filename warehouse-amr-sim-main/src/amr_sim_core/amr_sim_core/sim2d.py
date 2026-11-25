"""Fast 2D warehouse simulator (no ROS). Same world, robot, faults, pedestrians and metrics as the
Gazebo setup, but ~100x faster than real time, so it can be used for RL training (A1) and for large
dataset generation (A7).

Step pipeline (control period ``dt``, physics sub-steps ``dt / substeps``)::

    policy cmd (v, w) --> ActuationModel: latency FIFO -> limits -> per-wheel gain -> accel limit (payload, torque)
                      --> unicycle integration (true pose)  --> OdometryModel (encoder faults) --> measured v, w
    pedestrians: social-force step (aware workers avoid the robot)
    LiDAR: ray casting against boxes + pedestrian circles --> nominal noise --> LidarFaultModel
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np

from .faults import ActuationModel, FaultSampler, HealthState, LidarFaultModel, OdometryModel
from .geometry import point_aabb_distance, raycast, rect_aabb_distance, rect_circle_distance
from .metrics import EpisodeMetrics
from .pedestrians import Crowd
from .robot import RobotParams, load_robot
from .world import World, load_scenario, load_world


class WarehouseSim2D:
    def __init__(self, world: Union[str, World] = "warehouse_small", scenario: Union[str, dict] = "nominal",
                 robot: Union[str, RobotParams] = "amr_w1", dt: float = 0.1, substeps: int = 4,
                 seed: Optional[int] = None, terminate_on_collision: bool = True):
        self.scenario = load_scenario(scenario) if isinstance(scenario, str) else scenario
        if world is None:
            world = self.scenario.get("world", "warehouse_small")
        self.world = load_world(world) if isinstance(world, str) else world
        self.robot = load_robot(robot) if isinstance(robot, str) else robot
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.terminate_on_collision = terminate_on_collision
        self.angles = self.robot.lidar_angles()
        self.crowd = Crowd(self.world, self.scenario.get("pedestrians", {}))
        self.fault_sampler = FaultSampler(self.scenario.get("faults", {}))
        self.actuation = ActuationModel(self.robot)
        self.lidar_faults = LidarFaultModel(self.angles)
        self.odom = OdometryModel(self.robot)
        self.metrics = EpisodeMetrics(self.scenario.get("metrics", {}).get("near_miss_distance", 0.5))
        self.rng = np.random.default_rng(seed)
        self.t = 0.0

    # ------------------------------------------------------------------ episode
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> dict:
        """Start a new episode. ``options`` may override: start (x, y, yaw), goal (x, y),
        health (HealthState or dict), payload (kg), density (name or int)."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        o = options or {}
        sc, ep, rng = self.scenario, self.scenario["episode"], self.rng
        if "start" in o and "goal" in o:
            start, goal = tuple(o["start"]), tuple(o["goal"])
        else:
            start, goal, _ = self.world.sample_start_goal(rng, ep["min_goal_distance"], ep["max_goal_distance"])
        health = o.get("health")
        if health is None:
            health = self.fault_sampler.sample(rng)
        elif isinstance(health, dict):
            health = HealthState.from_dict(health)
        payload = o.get("payload")
        if payload is None:
            lo, hi = sc.get("payload", [0.0, 0.0])
            payload = float(rng.uniform(lo, hi))
        density = o.get("density")
        if density is None:
            density = rng.choice(sc["pedestrians"].get("density", ["medium"]))
            density = str(density)
        self.density = density
        self.health = health
        self.payload = float(payload)
        self.goal = np.asarray(goal, float)
        self.pose = np.asarray(start, float)
        self.v = self.w = 0.0
        self.t = 0.0
        self.last_cmd = (0.0, 0.0)
        self.applied_cmd = (0.0, 0.0)
        self.actuation.reset(health, self.payload)
        self.lidar_faults.health = health
        self.odom.reset(self.pose, health)
        self.crowd.reset(self.crowd.density_to_count(density), rng,
                         keep_out=[(start[0], start[1], 2.5), (goal[0], goal[1], 1.5)])
        self.metrics.reset(0.0, start, goal)
        self.outcome: Optional[str] = None
        self._sense()
        self._measure_distances()
        return self.observe()

    def step(self, v_cmd: float, w_cmd: float) -> dict:
        r = self.robot
        self.last_cmd = (float(v_cmd), float(w_cmd))
        self.actuation.push(self.t, v_cmd, w_cmd)
        h = self.dt / self.substeps
        for _ in range(self.substeps):
            v_t, w_t = self.actuation.target(self.t)
            self.applied_cmd = (v_t, w_t)
            v_new, w_new = self.actuation.limit(self.t, h, self.v, self.w, v_t, w_t)
            # first-order actuator lag on top of the acceleration limit
            k = h / (r.response_time + h)
            self.v += k * (v_new - self.v)
            self.w += k * (w_new - self.w)
            x, y, th = self.pose
            thm = th + 0.5 * self.w * h
            new_pose = np.array([x + self.v * np.cos(thm) * h, y + self.v * np.sin(thm) * h,
                                 (th + self.w * h + np.pi) % (2 * np.pi) - np.pi])
            # static collision: do not penetrate, stop the robot
            if self._static_collision(new_pose):
                self.metrics.update(self.t + h, self.pose, [], [], 0.0, abs(self.v))
                self.v = self.w = 0.0
            else:
                self.pose = new_pose
            self.odom.update(self.t, h, self.v, self.w)
            vx, vy = self.v * np.cos(self.pose[2]), self.v * np.sin(self.pose[2])
            self.crowd.step(h, robot=(self.pose[0], self.pose[1], vx, vy, r.radius))
            self.t += h
        self._sense()
        self._measure_distances()
        self.metrics.update(self.t, self.pose, self.ped_distances, range(len(self.ped_distances)),
                            self.obstacle_distance, self.v)
        ep = self.scenario["episode"]
        if self.metrics.collided and self.terminate_on_collision:
            self.outcome = self.metrics.first_collision
        elif np.hypot(*(self.goal - self.pose[:2])) <= ep["goal_tolerance"]:
            self.outcome = "success"
        elif self.t >= ep["max_time"]:
            self.outcome = "timeout"
        return self.observe()

    @property
    def done(self) -> bool:
        return self.outcome is not None

    def _static_collision(self, pose) -> bool:
        r = self.robot
        near = point_aabb_distance(pose[0], pose[1], self.world.aabbs)[0] < r.radius   # broad phase
        if not near.any():
            return False
        return bool(rect_aabb_distance(*pose, r.half_length, r.half_width, self.world.aabbs[near]).min() <= 0.0)

    # ------------------------------------------------------------------ sensing
    def _sense(self):
        r = self.robot
        x, y, th = self.pose
        lx = x + r.lidar_x * np.cos(th) - r.lidar_y * np.sin(th)
        ly = y + r.lidar_x * np.sin(th) + r.lidar_y * np.cos(th)
        peds = self.crowd.states()
        ranges = raycast(lx, ly, self.angles + th, r.lidar_range_max, self.world.aabbs,
                         peds[:, :2] if len(peds) else None, peds[:, 4] if len(peds) else None)
        hit = ranges < r.lidar_range_max
        ranges = np.where(hit, ranges + self.rng.normal(0.0, r.lidar_noise_std, ranges.shape), np.inf)
        ranges = np.where(ranges < r.lidar_range_min, np.inf, ranges)
        self.scan_clean = ranges
        self.scan = self.lidar_faults.apply(ranges, self.t, self.rng, r.lidar_range_max)

    def _measure_distances(self):
        r = self.robot
        peds = self.crowd.states()
        self.ped_distances = rect_circle_distance(*self.pose, r.half_length, r.half_width,
                                                  peds[:, :2], peds[:, 4]) if len(peds) else np.zeros(0)
        self.obstacle_distance = float(rect_aabb_distance(*self.pose, r.half_length, r.half_width,
                                                          self.world.aabbs).min())

    def observe(self) -> dict:
        """Full simulator state. Keys marked (privileged) must not be fed to a deployable policy."""
        return {
            "t": self.t,
            "scan": self.scan.astype(np.float32),                   # with faults, inf = no return
            "scan_clean": self.scan_clean.astype(np.float32),       # (privileged) without sensor faults
            "angles": self.angles.astype(np.float32),
            "odom_pose": self.odom.pose.copy(),                     # drifts with encoder faults
            "odom_vel": np.array([self.odom.v, self.odom.w], np.float32),
            "cmd": np.array(self.last_cmd, np.float32),
            "goal": self.goal.copy(),
            "pose": self.pose.copy(),                               # (privileged) ground truth
            "true_vel": np.array([self.v, self.w], np.float32),     # (privileged)
            "applied_cmd": np.array(self.applied_cmd, np.float32),  # (privileged) after actuation faults
            "peds": self.crowd.states().astype(np.float32),         # (privileged) x, y, vx, vy, r, aware
            "health": self.health.to_vector(self.t),                # (privileged) degradation vector
            "payload": self.payload,                                # context (known to the robot)
            "ped_distances": self.ped_distances.copy(),
            "obstacle_distance": self.obstacle_distance,
            "outcome": self.outcome,
        }

    def braking_distance(self) -> float:
        return self.actuation.braking_distance(self.t, self.v)
