"""metrics_monitor: live safety/efficiency metrics in Gazebo, with exactly the same definitions as the
2D simulator (amr_sim_core.metrics.EpisodeMetrics): footprint-to-body distances from ground truth.

An episode starts when a goal arrives on /goal_pose (RViz "2D Goal Pose") or /amr/episode_goal
(episode_runner), and ends at success (goal within tolerance). Running metrics are published as JSON
on /amr/metrics (2 Hz); the final summary on /amr/episode_summary.
"""
import json

import numpy as np
import rclpy
from amr_interfaces.msg import PedestrianArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from amr_sim_core.geometry import rect_aabb_distance, rect_circle_distance
from amr_sim_core.metrics import EpisodeMetrics
from amr_sim_core.robot import load_robot
from amr_sim_core.world import load_world

from .common import yaw_from_quat


class MetricsMonitor(Node):
    def __init__(self):
        super().__init__("metrics_monitor")
        self.declare_parameter("world", "warehouse_small")
        self.declare_parameter("robot", "amr_w1")
        self.declare_parameter("near_miss_distance", 0.5)
        self.declare_parameter("goal_tolerance", 0.3)
        self.declare_parameter("rate", 20.0)
        self.world = load_world(self.get_parameter("world").value)
        self.robot = load_robot(self.get_parameter("robot").value)
        self.metrics = EpisodeMetrics(self.get_parameter("near_miss_distance").value)
        self.pose = None
        self.speed = 0.0
        self.peds = None
        self.active = False
        self.goal = None
        self.pub = self.create_publisher(String, "amr/metrics", 10)
        self.pub_summary = self.create_publisher(String, "amr/episode_summary", 10)
        self.create_subscription(Odometry, "ground_truth/odom", self._on_odom, 20)
        self.create_subscription(PedestrianArray, "pedestrians", self._on_peds, 10)
        self.create_subscription(PoseStamped, "goal_pose", self._on_goal, 10)
        self.create_subscription(PoseStamped, "amr/episode_goal", self._on_goal, 10)
        self.create_service(Trigger, "~/finish", self._srv_finish)
        self.create_timer(1.0 / self.get_parameter("rate").value, self._tick)
        self.create_timer(0.5, self._publish)

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_odom(self, msg):
        p = msg.pose.pose
        self.pose = np.array([p.position.x, p.position.y, yaw_from_quat(p.orientation)])
        self.speed = float(np.hypot(msg.twist.twist.linear.x, msg.twist.twist.linear.y))

    def _on_peds(self, msg):
        act = [p for p in msg.pedestrians if p.active]
        self.peds = (np.array([[p.pose.position.x, p.pose.position.y] for p in act]).reshape(-1, 2),
                     np.array([p.radius for p in act]), [p.id for p in act])

    def _on_goal(self, msg):
        if self.pose is None:
            self.get_logger().warn("goal received before ground truth odometry; ignoring")
            return
        self.goal = np.array([msg.pose.position.x, msg.pose.position.y])
        self.metrics.reset(self._now(), self.pose[:2], self.goal)
        self.active = True
        self.get_logger().info(f"episode started: goal=({self.goal[0]:.2f}, {self.goal[1]:.2f})")

    def _tick(self):
        if self.pose is None:
            return
        r = self.robot
        if self.peds is not None and len(self.peds[2]):
            d_ped = rect_circle_distance(*self.pose, r.half_length, r.half_width, self.peds[0], self.peds[1])
            ids = self.peds[2]
        else:
            d_ped, ids = [], []
        d_obs = float(rect_aabb_distance(*self.pose, r.half_length, r.half_width, self.world.aabbs).min())
        self.current = {"min_ped_distance_now": float(np.min(d_ped)) if len(d_ped) else None,
                        "obstacle_distance_now": d_obs}
        if not self.active:
            return
        self.metrics.update(self._now(), self.pose[:2], d_ped, ids, d_obs, self.speed)
        if self.metrics.goal_distance() <= self.get_parameter("goal_tolerance").value:
            self._finish("success")

    def _finish(self, outcome):
        if not self.active:
            return None
        if outcome == "success" and self.metrics.collided:
            outcome = self.metrics.first_collision
        summary = self.metrics.summary(outcome)
        self.active = False
        self.pub_summary.publish(String(data=json.dumps(summary)))
        self.get_logger().info(f"episode finished: {summary}")
        return summary

    def _srv_finish(self, _req, res):
        """Finish the running episode (used by episode_runner on timeout / abort)."""
        outcome = self.metrics.first_collision if self.metrics.collided else "timeout"
        summary = self._finish(outcome)
        res.success = summary is not None
        res.message = json.dumps(summary) if summary else "no active episode"
        return res

    def _publish(self):
        if self.pose is None:
            return
        data = dict(getattr(self, "current", {}))
        data["active"] = self.active
        data.update(self.metrics.summary("running" if self.active else "idle"))
        self.pub.publish(String(data=json.dumps(data)))


def main():
    rclpy.init()
    node = MetricsMonitor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
