"""fault_injector: injects robot degradation between the navigation stack and Gazebo.

    /cmd_vel   --[latency, per-wheel gain, payload/torque accel limit]-->  /cmd_vel_hw  (Gazebo diff drive)
    /scan_raw  --[dropout, blind sector, extra noise]------------------->  /scan
    /odom_raw  --[encoder scale errors, re-integration]----------------->  /odom  + TF odom->base_footprint
    publishes the privileged true health on /amr/health (amr_interfaces/HealthState)

The fault models are the same Python classes the 2D simulator uses (amr_sim_core.faults).

Set faults at runtime, e.g.:
    ros2 param set /fault_injector wheel_gain_left 0.5
    ros2 param set /fault_injector cmd_latency 0.3
    ros2 service call /fault_injector/resample std_srvs/srv/Trigger      # random, from the scenario
    ros2 service call /fault_injector/set_nominal std_srvs/srv/Trigger
"""
import numpy as np
import rclpy
from amr_interfaces.msg import HealthState as HealthStateMsg
from geometry_msgs.msg import TransformStamped, Twist
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

from amr_sim_core.faults import (ActuationModel, FaultSampler, HealthState, LidarFaultModel,
                                 OdometryModel)
from amr_sim_core.robot import load_robot
from amr_sim_core.world import load_scenario

from .common import quat_from_yaw, stamp_to_sec

NOMINAL = {
    "wheel_gain_left": 1.0, "wheel_gain_right": 1.0, "torque_gain": 1.0, "cmd_latency": 0.0,
    "lidar_dropout_rate": 0.0, "lidar_sector_center": 0.0, "lidar_sector_width": 0.0, "lidar_noise_std": 0.0,
    "encoder_scale_left": 1.0, "encoder_scale_right": 1.0,
}
PARAM_TO_FAULT = {
    "wheel_gain_left": "wheel_gain", "wheel_gain_right": "wheel_gain", "torque_gain": "torque_loss",
    "cmd_latency": "cmd_latency", "lidar_dropout_rate": "lidar_dropout", "lidar_sector_width": "lidar_sector",
    "lidar_noise_std": "lidar_noise", "encoder_scale_left": "encoder_drift", "encoder_scale_right": "encoder_drift",
}


class FaultInjector(Node):
    def __init__(self):
        super().__init__("fault_injector")
        self.declare_parameter("robot", "amr_w1")
        self.declare_parameter("scenario", "nominal")
        self.declare_parameter("seed", 0)
        self.declare_parameter("payload_mass", 0.0)
        self.declare_parameter("onset_time", 0.0)
        self.declare_parameter("cmd_rate", 50.0)
        self.declare_parameter("cmd_timeout", 0.5)
        self.declare_parameter("publish_odom_tf", True)
        self.declare_parameter("odom_frame", "odom")
        self.declare_parameter("base_frame", "base_footprint")
        for name, val in NOMINAL.items():
            self.declare_parameter(name, val)

        self.robot = load_robot(self.get_parameter("robot").value)
        scenario = load_scenario(self.get_parameter("scenario").value)
        self.sampler = FaultSampler(scenario.get("faults", {}))
        self.rng = np.random.default_rng(self.get_parameter("seed").value)
        self.actuation = ActuationModel(self.robot, payload_mass=self.get_parameter("payload_mass").value)
        self.odom_model = OdometryModel(self.robot)
        self.lidar_model = None
        self.t0 = self._now()
        self._set_health(self._health_from_params({}))

        self.cmd_out = (0.0, 0.0)
        self.last_cmd_time = -1e9
        self.last_odom_t = None
        self.odom_frame = self.get_parameter("odom_frame").value
        self.base_frame = self.get_parameter("base_frame").value
        self.tf_pub = TransformBroadcaster(self) if self.get_parameter("publish_odom_tf").value else None

        self.pub_cmd = self.create_publisher(Twist, "cmd_vel_hw", 10)
        self.pub_scan = self.create_publisher(LaserScan, "scan", qos_profile_sensor_data)
        self.pub_odom = self.create_publisher(Odometry, "odom", 10)
        self.pub_health = self.create_publisher(HealthStateMsg, "amr/health", 10)
        self.create_subscription(Twist, "cmd_vel", self._on_cmd, 10)
        self.create_subscription(LaserScan, "scan_raw", self._on_scan, qos_profile_sensor_data)
        self.create_subscription(Odometry, "odom_raw", self._on_odom, 20)
        self.create_service(Trigger, "~/resample", self._srv_resample)
        self.create_service(Trigger, "~/set_nominal", self._srv_nominal)
        self.add_on_set_parameters_callback(self._on_params)
        self.dt_cmd = 1.0 / self.get_parameter("cmd_rate").value
        self.create_timer(self.dt_cmd, self._cmd_tick)
        self.create_timer(0.1, self._publish_health)
        self.get_logger().info("fault_injector ready: /cmd_vel->/cmd_vel_hw, /scan_raw->/scan, /odom_raw->/odom")

    # ------------------------------------------------------------------ health management
    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _t(self) -> float:
        return self._now() - self.t0

    def _health_from_params(self, overrides: dict) -> HealthState:
        vals = {k: self.get_parameter(k).value for k in NOMINAL}
        vals.update({k: v for k, v in overrides.items() if k in NOMINAL})
        faults = sorted({PARAM_TO_FAULT[k] for k, v in vals.items()
                         if k in PARAM_TO_FAULT and abs(float(v) - NOMINAL[k]) > 1e-6})
        onset = overrides.get("onset_time", self.get_parameter("onset_time").value)
        return HealthState(**{k: float(v) for k, v in vals.items()}, onset_time=float(onset), faults=faults)

    def _set_health(self, h: HealthState):
        self.health = h
        self.t0 = self._now()
        self.actuation.health = h
        self.odom_model.health = h
        if self.lidar_model is not None:
            self.lidar_model.health = h
        self.get_logger().info(f"health set: faults={h.faults or 'none'} severities={h.severities} onset={h.onset_time:.1f}s")

    def _on_params(self, params):
        overrides = {p.name: p.value for p in params}
        if "payload_mass" in overrides:
            self.actuation.payload_mass = float(overrides["payload_mass"])
        if any(k in NOMINAL or k == "onset_time" for k in overrides):
            self._set_health(self._health_from_params(overrides))
        return SetParametersResult(successful=True)

    def _srv_resample(self, _req, res):
        self._set_health(self.sampler.sample(self.rng))
        res.success, res.message = True, f"faults={self.health.faults} severities={self.health.severities}"
        return res

    def _srv_nominal(self, _req, res):
        self._set_health(HealthState())
        res.success, res.message = True, "nominal"
        return res

    # ------------------------------------------------------------------ command path
    def _on_cmd(self, msg: Twist):
        self.last_cmd_time = self._now()
        self.actuation.push(self._t(), msg.linear.x, msg.angular.z)

    def _cmd_tick(self):
        t = self._t()
        if self._now() - self.last_cmd_time > self.get_parameter("cmd_timeout").value:
            self.actuation.push(t, 0.0, 0.0)            # command timeout -> stop
            self.last_cmd_time = self._now()
        v_t, w_t = self.actuation.target(t)
        self.cmd_out = self.actuation.limit(t, self.dt_cmd, *self.cmd_out, v_t, w_t)
        msg = Twist()
        msg.linear.x, msg.angular.z = float(self.cmd_out[0]), float(self.cmd_out[1])
        self.pub_cmd.publish(msg)

    # ------------------------------------------------------------------ sensors
    def _on_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        if self.lidar_model is None or len(self.lidar_model.angles) != n:
            angles = msg.angle_min + np.arange(n) * msg.angle_increment
            self.lidar_model = LidarFaultModel(angles, self.health)
        ranges = np.asarray(msg.ranges, dtype=float)
        out = self.lidar_model.apply(ranges, self._t(), self.rng, msg.range_max)
        msg.ranges = [float(r) for r in out]
        self.pub_scan.publish(msg)

    def _on_odom(self, msg: Odometry):
        t_msg = stamp_to_sec(msg.header.stamp)
        dt = 0.0 if self.last_odom_t is None else max(0.0, min(t_msg - self.last_odom_t, 0.2))
        self.last_odom_t = t_msg
        v, w = msg.twist.twist.linear.x, msg.twist.twist.angular.z
        pose, (vm, wm) = self.odom_model.update(self._t(), dt, v, w)
        out = Odometry()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.odom_frame
        out.child_frame_id = self.base_frame
        out.pose.pose.position.x, out.pose.pose.position.y = float(pose[0]), float(pose[1])
        out.pose.pose.orientation = quat_from_yaw(float(pose[2]))
        out.pose.covariance = msg.pose.covariance
        out.twist.twist.linear.x, out.twist.twist.angular.z = float(vm), float(wm)
        out.twist.covariance = msg.twist.covariance
        self.pub_odom.publish(out)
        if self.tf_pub is not None:
            tf = TransformStamped()
            tf.header = out.header
            tf.child_frame_id = self.base_frame
            tf.transform.translation.x, tf.transform.translation.y = out.pose.pose.position.x, out.pose.pose.position.y
            tf.transform.rotation = out.pose.pose.orientation
            self.tf_pub.sendTransform(tf)

    def _publish_health(self):
        t = self._t()
        h = self.health.effective(t)
        m = HealthStateMsg()
        m.header.stamp = self.get_clock().now().to_msg()
        for k in NOMINAL:
            setattr(m, k, float(getattr(h, k)))
        m.payload_mass = float(self.actuation.payload_mass)
        m.onset_time = float(self.health.onset_time)
        m.fault_active = bool(self.health.faults) and t >= self.health.onset_time
        m.faults = list(self.health.faults)
        m.severities = [float(s) for s in self.health.severities]
        m.vector = [float(x) for x in self.health.to_vector(t)]
        self.pub_health.publish(m)


def main():
    rclpy.init()
    node = FaultInjector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
