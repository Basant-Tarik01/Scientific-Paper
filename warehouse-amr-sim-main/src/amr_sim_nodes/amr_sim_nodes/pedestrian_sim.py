"""pedestrian_sim: runs the warehouse-worker social-force model (amr_sim_core.pedestrians.Crowd) and
publishes the workers on /pedestrians. The Gazebo plugin libamr_pedestrian_pose_plugin.so moves the
ped_<i> models accordingly, so they are seen by the LiDAR.

    ros2 param set /pedestrian_sim density high && ros2 service call /pedestrian_sim/reset std_srvs/srv/Trigger
"""
import numpy as np
import rclpy
from amr_interfaces.msg import Pedestrian, PedestrianArray
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from amr_sim_core.pedestrians import Crowd
from amr_sim_core.robot import load_robot
from amr_sim_core.world import load_scenario, load_world

from .common import quat_from_yaw, yaw_from_quat


class PedestrianSim(Node):
    def __init__(self):
        super().__init__("pedestrian_sim")
        self.declare_parameter("world", "warehouse_small")
        self.declare_parameter("scenario", "nominal")
        self.declare_parameter("density", "")          # '' = sample from the scenario; or none/low/medium/high/<int>
        self.declare_parameter("seed", 0)
        self.declare_parameter("rate", 20.0)
        self.declare_parameter("fixed_radius", 0.25)   # radius of the Gazebo worker collision cylinder
        self.declare_parameter("frame_id", "map")
        self.world = load_world(self.get_parameter("world").value)
        self.scenario = load_scenario(self.get_parameter("scenario").value)
        spec = dict(self.scenario.get("pedestrians", {}))
        r = self.get_parameter("fixed_radius").value
        if r > 0:
            spec["radius"] = (r, r)
        self.crowd = Crowd(self.world, spec)
        self.robot = load_robot("amr_w1")
        self.rng = np.random.default_rng(self.get_parameter("seed").value)
        self.robot_state = None
        self.last_t = None
        self.pub = self.create_publisher(PedestrianArray, "pedestrians", 10)
        self.pub_markers = self.create_publisher(MarkerArray, "pedestrians/markers", 10)
        self.create_subscription(Odometry, "ground_truth/odom", self._on_robot, 10)
        self.create_service(Trigger, "~/reset", self._srv_reset)
        self.create_timer(1.0 / self.get_parameter("rate").value, self._tick)
        self._reset()

    def _density(self):
        d = str(self.get_parameter("density").value).strip()
        if not d:
            d = str(self.rng.choice(self.scenario["pedestrians"].get("density", ["medium"])))
        return int(d) if d.isdigit() else d

    def _reset(self):
        keep = []
        if self.robot_state is not None:
            keep.append((self.robot_state[0], self.robot_state[1], 2.5))
        else:
            sp = self.world.robot_spawn
            keep.append((sp["x"], sp["y"], 2.5))
        density = self._density()
        self.crowd.reset(self.crowd.density_to_count(density), self.rng, keep_out=keep)
        self.get_logger().info(f"spawned {self.crowd.n} workers (density={density})")

    def _srv_reset(self, _req, res):
        self._reset()
        res.success, res.message = True, f"{self.crowd.n} workers"
        return res

    def _on_robot(self, msg: Odometry):
        p = msg.pose.pose
        yaw = yaw_from_quat(p.orientation)
        v = msg.twist.twist.linear
        self.robot_state = (p.position.x, p.position.y, v.x, v.y, self.robot.radius, yaw)

    def _tick(self):
        t = self.get_clock().now().nanoseconds * 1e-9
        if self.last_t is None:
            self.last_t = t
            return
        dt = t - self.last_t
        if dt <= 0.0:          # simulation paused
            return
        self.last_t = t
        rs = self.robot_state[:5] if self.robot_state is not None else None
        remaining = min(dt, 0.2)
        while remaining > 1e-6:
            h = min(remaining, 0.05)
            self.crowd.step(h, robot=rs)
            remaining -= h
        self._publish()

    def _publish(self):
        stamp = self.get_clock().now().to_msg()
        frame = self.get_parameter("frame_id").value
        arr = PedestrianArray()
        arr.header.stamp, arr.header.frame_id = stamp, frame
        markers = MarkerArray()
        n = self.crowd.n
        for i in range(self.crowd.max_peds):
            p = Pedestrian()
            p.id = i
            p.active = i < n
            if p.active:
                p.pose.position.x, p.pose.position.y = float(self.crowd.pos[i, 0]), float(self.crowd.pos[i, 1])
                p.pose.orientation = quat_from_yaw(float(self.crowd.yaw[i]))
                p.twist.linear.x, p.twist.linear.y = float(self.crowd.vel[i, 0]), float(self.crowd.vel[i, 1])
                p.radius = float(self.crowd.radius[i])
                p.aware = bool(self.crowd.aware[i])
                p.dwelling = bool(self.crowd.dwell_until[i] > self.crowd.t)
            else:
                p.pose.orientation.w = 1.0
            arr.pedestrians.append(p)
            m = Marker()
            m.header.stamp, m.header.frame_id = stamp, frame
            m.ns, m.id = "workers", i
            if not p.active:
                m.action = Marker.DELETE
                markers.markers.append(m)
                continue
            m.type, m.action = Marker.CYLINDER, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = p.pose.position.x, p.pose.position.y, 0.9
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = 2 * p.radius
            m.scale.z = 1.8
            m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.55, 0.0, 0.8) if p.aware else (0.85, 0.1, 0.1, 0.8)
            markers.markers.append(m)
            a = Marker()
            a.header = m.header
            a.ns, a.id, a.type = "velocity", i, Marker.ARROW
            a.scale.x, a.scale.y, a.scale.z = 0.05, 0.1, 0.1
            a.color.r, a.color.g, a.color.b, a.color.a = 0.1, 0.1, 0.1, 1.0
            a.points = [Point(x=p.pose.position.x, y=p.pose.position.y, z=1.9),
                        Point(x=p.pose.position.x + p.twist.linear.x, y=p.pose.position.y + p.twist.linear.y, z=1.9)]
            markers.markers.append(a)
        self.pub.publish(arr)
        self.pub_markers.publish(markers)


def main():
    rclpy.init()
    node = PedestrianSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
