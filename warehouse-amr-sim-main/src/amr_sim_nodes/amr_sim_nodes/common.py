"""Small helpers shared by the nodes."""
import math

from geometry_msgs.msg import Quaternion


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def quat_from_yaw(yaw: float) -> Quaternion:
    q = Quaternion()
    q.z = math.sin(yaw / 2.0)
    q.w = math.cos(yaw / 2.0)
    return q


def stamp_to_sec(stamp) -> float:
    return stamp.sec + stamp.nanosec * 1e-9
