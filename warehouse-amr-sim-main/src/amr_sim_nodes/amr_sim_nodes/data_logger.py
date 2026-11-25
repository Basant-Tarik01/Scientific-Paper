"""data_logger: records synchronised samples (one per /scan message) for world-model training (A7)
and offline analysis (A1). One .npz file per episode (a new episode starts on every goal) plus one
on shutdown or on `ros2 service call /data_logger/save std_srvs/srv/Trigger`.

Arrays in each file (T = number of scans):
    t (T,)                     sim time
    scan (T, N)                LiDAR with faults (inf = no return)      scan_raw (T, N)  before faults
    angles (N,)
    cmd (T, 2)                 policy command v, w (/cmd_vel)           cmd_hw (T, 2)    after actuation faults
    odom_pose (T, 3), odom_twist (T, 2)                                 measured (encoder faults)
    gt_pose (T, 3), gt_twist (T, 3)                                     ground truth x, y, yaw / vx, vy, w
    health (T, 9)              degradation vector                       payload (T,)
    peds (T, K, 7)             x, y, vx, vy, radius, aware, active
The same layout is written by examples/generate_a7_dataset.py from the 2D simulator.
"""
import json
import os
import time

import numpy as np
import rclpy
from amr_interfaces.msg import HealthState, PedestrianArray
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger

from .common import stamp_to_sec, yaw_from_quat


class DataLogger(Node):
    def __init__(self):
        super().__init__("data_logger")
        self.declare_parameter("output_dir", os.path.expanduser("~/amr_datasets/gazebo"))
        self.declare_parameter("max_peds", 16)
        self.out = os.path.expanduser(self.get_parameter("output_dir").value)
        os.makedirs(self.out, exist_ok=True)
        self.K = self.get_parameter("max_peds").value
        self.latest = {"scan_raw": None, "cmd": (0.0, 0.0), "cmd_hw": (0.0, 0.0), "odom": None, "gt": None,
                       "health": np.zeros(9, np.float32), "payload": 0.0, "faults": [],
                       "peds": np.zeros((self.K, 7), np.float32)}
        self.buf = []
        self.episode = 0
        self.meta = {}
        self.create_subscription(LaserScan, "scan", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(LaserScan, "scan_raw", lambda m: self.latest.__setitem__("scan_raw", m.ranges),
                                 qos_profile_sensor_data)
        self.create_subscription(Twist, "cmd_vel", lambda m: self.latest.__setitem__("cmd", (m.linear.x, m.angular.z)), 10)
        self.create_subscription(Twist, "cmd_vel_hw", lambda m: self.latest.__setitem__("cmd_hw", (m.linear.x, m.angular.z)), 10)
        self.create_subscription(Odometry, "odom", lambda m: self.latest.__setitem__("odom", m), 20)
        self.create_subscription(Odometry, "ground_truth/odom", lambda m: self.latest.__setitem__("gt", m), 20)
        self.create_subscription(HealthState, "amr/health", self._on_health, 10)
        self.create_subscription(PedestrianArray, "pedestrians", self._on_peds, 10)
        self.create_subscription(PoseStamped, "amr/episode_goal", self._on_goal, 10)
        self.create_subscription(PoseStamped, "goal_pose", self._on_goal, 10)
        self.create_service(Trigger, "~/save", self._srv_save)
        self.get_logger().info(f"logging to {self.out}")

    def _on_health(self, m):
        self.latest["health"] = np.asarray(m.vector, np.float32) if len(m.vector) else np.zeros(9, np.float32)
        self.latest["payload"] = m.payload_mass
        self.latest["faults"] = list(m.faults)

    def _on_peds(self, m):
        arr = np.zeros((self.K, 7), np.float32)
        for p in m.pedestrians[: self.K]:
            arr[p.id % self.K] = (p.pose.position.x, p.pose.position.y, p.twist.linear.x, p.twist.linear.y,
                                  p.radius, float(p.aware), float(p.active))
        self.latest["peds"] = arr

    def _on_goal(self, m):
        self._save()
        self.meta = {"goal": [m.pose.position.x, m.pose.position.y]}

    def _on_scan(self, m):
        L = self.latest
        if L["odom"] is None or L["gt"] is None:
            return
        if not hasattr(self, "angles") or len(self.angles) != len(m.ranges):
            self.angles = m.angle_min + np.arange(len(m.ranges)) * m.angle_increment
        o, g = L["odom"], L["gt"]
        self.buf.append({
            "t": stamp_to_sec(m.header.stamp),
            "scan": np.asarray(m.ranges, np.float32),
            "scan_raw": np.asarray(L["scan_raw"] if L["scan_raw"] is not None else m.ranges, np.float32),
            "cmd": L["cmd"], "cmd_hw": L["cmd_hw"],
            "odom_pose": (o.pose.pose.position.x, o.pose.pose.position.y, yaw_from_quat(o.pose.pose.orientation)),
            "odom_twist": (o.twist.twist.linear.x, o.twist.twist.angular.z),
            "gt_pose": (g.pose.pose.position.x, g.pose.pose.position.y, yaw_from_quat(g.pose.pose.orientation)),
            "gt_twist": (g.twist.twist.linear.x, g.twist.twist.linear.y, g.twist.twist.angular.z),
            "health": L["health"].copy(), "payload": L["payload"], "peds": L["peds"].copy(),
        })
        self.meta["faults"] = L["faults"]

    def _save(self):
        if len(self.buf) < 2:
            self.buf = []
            return None
        b = self.buf
        path = os.path.join(self.out, f"gz_{time.strftime('%Y%m%d_%H%M%S')}_ep{self.episode:04d}.npz")
        np.savez_compressed(
            path, t=np.array([x["t"] for x in b]), scan=np.stack([x["scan"] for x in b]),
            scan_raw=np.stack([x["scan_raw"] for x in b]), angles=self.angles.astype(np.float32),
            cmd=np.array([x["cmd"] for x in b], np.float32), cmd_hw=np.array([x["cmd_hw"] for x in b], np.float32),
            odom_pose=np.array([x["odom_pose"] for x in b], np.float32),
            odom_twist=np.array([x["odom_twist"] for x in b], np.float32),
            gt_pose=np.array([x["gt_pose"] for x in b], np.float32),
            gt_twist=np.array([x["gt_twist"] for x in b], np.float32),
            health=np.stack([x["health"] for x in b]), payload=np.array([x["payload"] for x in b], np.float32),
            peds=np.stack([x["peds"] for x in b]), meta=json.dumps(self.meta))
        self.get_logger().info(f"saved {len(b)} samples -> {path}")
        self.buf = []
        self.episode += 1
        return path

    def _srv_save(self, _req, res):
        path = self._save()
        res.success, res.message = path is not None, path or "nothing to save"
        return res


def main():
    rclpy.init()
    node = DataLogger()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._save()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
