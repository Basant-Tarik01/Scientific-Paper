"""Robot parameters (loaded from data/robot/<name>.yaml) and differential-drive kinematics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import yaml

from .world import resolve_data_file


@dataclass
class RobotParams:
    name: str
    length: float
    width: float
    base_mass: float
    deck_mass: float
    wheel_radius: float
    wheel_separation: float
    wheel_x_offset: float
    max_wheel_torque: float
    max_lin_vel: float
    max_rev_vel: float
    max_ang_vel: float
    max_lin_acc: float
    max_ang_acc: float
    response_time: float
    lidar_x: float
    lidar_y: float
    lidar_fov: float
    lidar_samples: int
    lidar_rate: float
    lidar_range_min: float
    lidar_range_max: float
    lidar_noise_std: float
    payload_max_mass: float
    raw: dict

    @property
    def half_length(self) -> float:
        return self.length / 2

    @property
    def half_width(self) -> float:
        return self.width / 2

    @property
    def radius(self) -> float:
        """Radius of the circumscribed circle (used by pedestrians to perceive the robot)."""
        return float(np.hypot(self.half_length, self.half_width))

    @property
    def empty_mass(self) -> float:
        return self.base_mass + self.deck_mass

    def lidar_angles(self) -> np.ndarray:
        """Beam angles in the robot frame (same convention as a Gazebo ray sensor)."""
        fov = self.lidar_fov
        if np.isclose(fov, 2 * np.pi):
            return -np.pi + np.arange(self.lidar_samples) * (2 * np.pi / self.lidar_samples)
        return np.linspace(-fov / 2, fov / 2, self.lidar_samples)

    def footprint(self):
        hl, hw = self.half_length, self.half_width
        return [[hl, hw], [hl, -hw], [-hl, -hw], [-hl, hw]]

    # differential-drive kinematics
    def to_wheels(self, v, w):
        half = self.wheel_separation / 2
        return v - w * half, v + w * half

    def from_wheels(self, vl, vr):
        return (vl + vr) / 2, (vr - vl) / self.wheel_separation


def load_robot(name_or_path: str = "amr_w1") -> RobotParams:
    path = resolve_data_file(name_or_path, "robot")
    with open(path) as f:
        r = yaml.safe_load(f)
    b, d, wh, dr, li = r["base"], r["deck"], r["wheels"], r["drive"], r["lidar"]
    return RobotParams(
        name=r["name"], length=b["length"], width=b["width"], base_mass=b["mass"], deck_mass=d["mass"],
        wheel_radius=wh["radius"], wheel_separation=wh["separation"], wheel_x_offset=wh.get("x_offset", 0.0),
        max_wheel_torque=dr["max_wheel_torque"], max_lin_vel=dr["max_lin_vel"], max_rev_vel=dr["max_rev_vel"],
        max_ang_vel=dr["max_ang_vel"], max_lin_acc=dr["max_lin_acc"], max_ang_acc=dr["max_ang_acc"],
        response_time=dr.get("response_time", 0.05),
        lidar_x=li["x"], lidar_y=li["y"], lidar_fov=np.deg2rad(li["fov_deg"]), lidar_samples=int(li["samples"]),
        lidar_rate=li["rate"], lidar_range_min=li["range_min"], lidar_range_max=li["range_max"],
        lidar_noise_std=li["noise_std"], payload_max_mass=r["payload"]["max_mass"], raw=r,
    )
