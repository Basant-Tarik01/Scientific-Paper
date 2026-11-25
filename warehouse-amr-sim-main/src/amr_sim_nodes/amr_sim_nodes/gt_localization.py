"""gt_localization: "perfect localisation" for runs without AMCL.

Publishes TF map->odom = T(map->base, ground truth) * T(odom->base, odometry)^-1, the way AMCL would
but without error. The robot, the scan and the workers then line up in RViz in the map frame after
spawning anywhere, after teleports (episode resets) and under encoder drift. In that last case the
drift shows up as a growing map->odom correction, exactly as with a real localiser.
"""
import math

import rclpy
from geometry_msgs.msg import TransformStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from .common import quat_from_yaw, yaw_from_quat


def _pose(msg):
    p = msg.pose.pose
    return p.position.x, p.position.y, yaw_from_quat(p.orientation)


class GtLocalization(Node):
    def __init__(self):
        super().__init__("gt_localization")
        self.declare_parameter("map_frame", "map")
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("rate", 20.0)
        self.gt = None
        self.odom = None
        self.tf = TransformBroadcaster(self)
        self.create_subscription(Odometry, "ground_truth/odom", lambda m: setattr(self, "gt", m), 20)
        self.create_subscription(Odometry, "odom", lambda m: setattr(self, "odom", m), 20)
        self.create_timer(1.0 / self.get_parameter("rate").value, self._tick)

    def _tick(self):
        if self.gt is None or self.odom is None:
            return
        gx, gy, gth = _pose(self.gt)
        ox, oy, oth = _pose(self.odom)
        th = gth - oth                                   # map->odom rotation
        c, s = math.cos(th), math.sin(th)
        tx, ty = gx - (c * ox - s * oy), gy - (s * ox + c * oy)
        t = TransformStamped()
        t.header.stamp = self.odom.header.stamp
        t.header.frame_id = self.get_parameter("map_frame").value
        t.child_frame_id = self.get_parameter("odom_frame").value
        t.transform.translation.x, t.transform.translation.y = tx, ty
        t.transform.rotation = quat_from_yaw(th)
        self.tf.sendTransform(t)


def main():
    rclpy.init()
    node = GtLocalization()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
