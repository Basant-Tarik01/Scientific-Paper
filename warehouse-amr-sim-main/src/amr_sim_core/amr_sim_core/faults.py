"""Robot health state, fault sampling and fault models.

The same classes are used by the 2D simulator and by the ROS 2 ``fault_injector`` node, so a fault
of a given severity has the same effect in both simulators.

Fault interfaces (as required by the A1 proposal, step 2 "inject each fault at the right interface"):

=============== ===================================== ==================================================
fault type      parameter(s)                           interface
=============== ===================================== ==================================================
wheel_gain      wheel_gain_left / wheel_gain_right     actuation: per-wheel speed gain (cmd -> wheels)
torque_loss     torque_gain                            actuation: max drive force -> accel/brake limit
cmd_latency     cmd_latency                            command processing: FIFO delay of cmd_vel
lidar_dropout   lidar_dropout_rate                     LiDAR stream: random beams return no echo (inf)
lidar_sector    lidar_sector_center / _width           LiDAR stream: contiguous blind sector (dirt/occl.)
lidar_noise     lidar_noise_std                        LiDAR stream: extra Gaussian range noise
encoder_drift   encoder_scale_left / _right            odometry: wheel-encoder scale error
=============== ===================================== ==================================================

Severity s in [0, 1] maps to parameters with :func:`apply_fault` (documented in docs/FAULTS.md).
"""
from __future__ import annotations

import collections
from dataclasses import asdict, dataclass, field, replace
from typing import Dict, List, Optional, Sequence

import numpy as np

FAULT_TYPES = ("wheel_gain", "torque_loss", "cmd_latency", "lidar_dropout", "lidar_sector",
               "lidar_noise", "encoder_drift")

MAX_LATENCY = 0.5          # s, at severity 1
MAX_NOISE = 0.15           # m, at severity 1
MAX_ENCODER_ERROR = 0.2    # relative, at severity 1

# Order of the normalised health ("degradation") vector given to the privileged teacher.
HEALTH_VECTOR_NAMES = (
    "wheel_loss_left",      # 1 - wheel_gain_left
    "wheel_loss_right",     # 1 - wheel_gain_right
    "torque_loss",          # 1 - torque_gain
    "cmd_latency",          # latency / MAX_LATENCY
    "lidar_dropout",        # dropout rate
    "lidar_sector",         # blind sector width / (2 pi)
    "lidar_noise",          # extra noise std / MAX_NOISE
    "encoder_err_left",     # (scale - 1) / MAX_ENCODER_ERROR, in [-1, 1]
    "encoder_err_right",
)


@dataclass
class HealthState:
    wheel_gain_left: float = 1.0
    wheel_gain_right: float = 1.0
    torque_gain: float = 1.0
    cmd_latency: float = 0.0
    lidar_dropout_rate: float = 0.0
    lidar_sector_center: float = 0.0
    lidar_sector_width: float = 0.0
    lidar_noise_std: float = 0.0
    encoder_scale_left: float = 1.0
    encoder_scale_right: float = 1.0
    onset_time: float = 0.0                       # faults become active at this sim time
    faults: List[str] = field(default_factory=list)
    severities: List[float] = field(default_factory=list)

    @property
    def is_nominal(self) -> bool:
        return not self.faults

    def effective(self, t: float) -> "HealthState":
        """Health state actually in effect at time t (nominal before the onset)."""
        if t >= self.onset_time or self.is_nominal:
            return self
        return HealthState(onset_time=self.onset_time)

    def to_vector(self, t: Optional[float] = None) -> np.ndarray:
        h = self if t is None else self.effective(t)
        return np.array([
            1.0 - h.wheel_gain_left, 1.0 - h.wheel_gain_right, 1.0 - h.torque_gain,
            h.cmd_latency / MAX_LATENCY, h.lidar_dropout_rate, h.lidar_sector_width / (2 * np.pi),
            h.lidar_noise_std / MAX_NOISE,
            np.clip((h.encoder_scale_left - 1.0) / MAX_ENCODER_ERROR, -1, 1),
            np.clip((h.encoder_scale_right - 1.0) / MAX_ENCODER_ERROR, -1, 1),
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "HealthState":
        return HealthState(**{k: v for k, v in d.items() if k in HealthState.__dataclass_fields__})


def apply_fault(h: HealthState, ftype: str, s: float, rng: np.random.Generator) -> HealthState:
    """Return a copy of ``h`` with fault ``ftype`` at severity ``s`` in [0, 1] applied."""
    s = float(np.clip(s, 0.0, 1.0))
    h = replace(h, faults=list(h.faults) + [ftype], severities=list(h.severities) + [s])
    if ftype == "wheel_gain":               # one wheel loses up to 70 % of its speed gain
        gain = 1.0 - 0.7 * s
        if rng.random() < 0.5:
            h.wheel_gain_left = min(h.wheel_gain_left, gain)
        else:
            h.wheel_gain_right = min(h.wheel_gain_right, gain)
    elif ftype == "torque_loss":            # both motors lose up to 80 % of their torque
        h.torque_gain = min(h.torque_gain, 1.0 - 0.8 * s)
    elif ftype == "cmd_latency":
        h.cmd_latency = max(h.cmd_latency, MAX_LATENCY * s)
    elif ftype == "lidar_dropout":          # up to 60 % of beams lost at random
        h.lidar_dropout_rate = max(h.lidar_dropout_rate, 0.6 * s)
    elif ftype == "lidar_sector":           # up to 180 deg contiguous blind sector
        h.lidar_sector_width = max(h.lidar_sector_width, np.pi * s)
        h.lidar_sector_center = float(rng.uniform(-np.pi, np.pi))
    elif ftype == "lidar_noise":
        h.lidar_noise_std = max(h.lidar_noise_std, MAX_NOISE * s)
    elif ftype == "encoder_drift":          # one encoder mis-scaled by up to +-20 %
        err = MAX_ENCODER_ERROR * s * (1 if rng.random() < 0.5 else -1)
        if rng.random() < 0.5:
            h.encoder_scale_left = 1.0 + err
        else:
            h.encoder_scale_right = 1.0 + err
    else:
        raise ValueError(f"unknown fault type '{ftype}', expected one of {FAULT_TYPES}")
    return h


class FaultSampler:
    """Samples a :class:`HealthState` per episode from a scenario ``faults`` block::

        faults:
          p_nominal: 0.2            # probability of a healthy robot
          types: [wheel_gain, cmd_latency]
          max_simultaneous: 1       # 1..k faults combined
          severity: [0.0, 0.6]      # uniform severity range
          onset: [0.0, 0.0]         # uniform onset-time range (s)
    """

    def __init__(self, spec: Optional[dict] = None):
        spec = spec or {}
        self.p_nominal = float(spec.get("p_nominal", 1.0))
        self.types: List[str] = list(spec.get("types", []))
        for t in self.types:
            if t not in FAULT_TYPES:
                raise ValueError(f"unknown fault type '{t}'")
        self.max_simultaneous = int(spec.get("max_simultaneous", 1))
        self.severity = tuple(spec.get("severity", (0.0, 1.0)))
        self.onset = tuple(spec.get("onset", (0.0, 0.0)))

    def sample(self, rng: np.random.Generator) -> HealthState:
        if not self.types or rng.random() < self.p_nominal:
            return HealthState()
        k = int(rng.integers(1, min(self.max_simultaneous, len(self.types)) + 1))
        h = HealthState(onset_time=float(rng.uniform(*self.onset)))
        for ftype in rng.choice(self.types, size=k, replace=False):
            h = apply_fault(h, str(ftype), float(rng.uniform(*self.severity)), rng)
        return h


# ---------------------------------------------------------------------------------------------
# Fault models
# ---------------------------------------------------------------------------------------------
class ActuationModel:
    """Command path from the policy's (v, w) to what the wheels actually do.

    cmd -> [latency FIFO] -> [velocity limits] -> [per-wheel gain] -> target (v, w)
    and, when integrating the motion, [drive-force-limited acceleration] which depends on the
    payload mass and the torque gain: a_max = min(a_nominal, torque_gain * 2 * tau_max / r / m_total).
    """

    def __init__(self, robot, health: Optional[HealthState] = None, payload_mass: float = 0.0):
        self.robot = robot
        self.health = health or HealthState()
        self.payload_mass = float(payload_mass)
        self._queue: "collections.deque" = collections.deque()
        self._last = (0.0, 0.0)

    def reset(self, health: Optional[HealthState] = None, payload_mass: Optional[float] = None):
        if health is not None:
            self.health = health
        if payload_mass is not None:
            self.payload_mass = float(payload_mass)
        self._queue.clear()
        self._last = (0.0, 0.0)

    def push(self, t: float, v: float, w: float):
        r = self.robot
        v = min(max(float(v), -r.max_rev_vel), r.max_lin_vel)
        w = min(max(float(w), -r.max_ang_vel), r.max_ang_vel)
        self._queue.append((t, v, w))

    def target(self, t: float):
        """Wheel-level target velocity at time t (after latency and wheel-gain faults)."""
        h = self.health.effective(t)
        while self._queue and self._queue[0][0] <= t - h.cmd_latency + 1e-9:
            _, v, w = self._queue.popleft()
            self._last = (v, w)
        v, w = self._last
        vl, vr = self.robot.to_wheels(v, w)
        return self.robot.from_wheels(vl * h.wheel_gain_left, vr * h.wheel_gain_right)

    def accel_limits(self, t: float):
        r = self.robot
        h = self.health.effective(t)
        m = r.empty_mass + self.payload_mass
        a_force = h.torque_gain * 2.0 * r.max_wheel_torque / r.wheel_radius / m
        a_lin = min(r.max_lin_acc, a_force)
        a_ang = r.max_ang_acc * a_lin / r.max_lin_acc
        return a_lin, a_ang

    def limit(self, t: float, dt: float, v: float, w: float, v_tgt: float, w_tgt: float):
        """Move (v, w) towards the target respecting the (fault- and payload-dependent) accel limits."""
        a_lin, a_ang = self.accel_limits(t)
        dv, dw = a_lin * dt, a_ang * dt
        v = v + min(max(v_tgt - v, -dv), dv)
        w = w + min(max(w_tgt - w, -dw), dw)
        return v, w

    def braking_distance(self, t: float, v: float) -> float:
        a_lin, _ = self.accel_limits(t)
        return v * v / (2 * a_lin) + abs(v) * self.health.effective(t).cmd_latency


class LidarFaultModel:
    """Applies sensor faults to a range array. Lost returns are set to ``inf`` (no echo)."""

    def __init__(self, angles: np.ndarray, health: Optional[HealthState] = None):
        self.angles = np.asarray(angles)
        self.health = health or HealthState()

    def apply(self, ranges: np.ndarray, t: float, rng: np.random.Generator, range_max: float) -> np.ndarray:
        h = self.health.effective(t)
        out = np.array(ranges, dtype=float, copy=True)
        if h.lidar_noise_std > 0:
            finite = np.isfinite(out) & (out < range_max)
            out[finite] += rng.normal(0.0, h.lidar_noise_std, int(finite.sum()))
            out = np.maximum(out, 0.0)
        if h.lidar_sector_width > 0:
            d = np.abs((self.angles - h.lidar_sector_center + np.pi) % (2 * np.pi) - np.pi)
            out[d <= h.lidar_sector_width / 2] = np.inf
        if h.lidar_dropout_rate > 0:
            out[rng.random(out.shape) < h.lidar_dropout_rate] = np.inf
        return out


class OdometryModel:
    """Wheel odometry with encoder-scale faults. Integrates the *measured* velocities."""

    def __init__(self, robot, health: Optional[HealthState] = None):
        self.robot = robot
        self.health = health or HealthState()
        self.pose = np.zeros(3)
        self.v = 0.0
        self.w = 0.0

    def reset(self, pose: Sequence[float], health: Optional[HealthState] = None):
        self.pose = np.array(pose, dtype=float)
        if health is not None:
            self.health = health
        self.v = self.w = 0.0

    def measure(self, t: float, v_true: float, w_true: float):
        h = self.health.effective(t)
        vl, vr = self.robot.to_wheels(v_true, w_true)
        return self.robot.from_wheels(vl * h.encoder_scale_left, vr * h.encoder_scale_right)

    def update(self, t: float, dt: float, v_true: float, w_true: float):
        self.v, self.w = self.measure(t, v_true, w_true)
        x, y, th = self.pose
        th_mid = th + 0.5 * self.w * dt
        self.pose = np.array([x + self.v * np.cos(th_mid) * dt, y + self.v * np.sin(th_mid) * dt,
                              (th + self.w * dt + np.pi) % (2 * np.pi) - np.pi])
        return self.pose, (self.v, self.w)
