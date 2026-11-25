"""episode_runner: automatic evaluation of the Nav2 baseline in Gazebo (A1 "Nav2 baseline results").

For every episode it samples start/goal, faults, payload and pedestrian density from a scenario
(the same sampling code as the 2D simulator), teleports the robot, configures fault_injector and
pedestrian_sim, re-localises AMCL, sends the goal to Nav2 and records the metrics_monitor summary
to a CSV file.

    ros2 launch amr_bringup nav2_sim.launch.py gui:=false            # terminal 1
    ros2 run amr_sim_nodes episode_runner --ros-args -p scenario:=a1_train -p episodes:=20 \
        -p output:=~/amr_results/nav2_a1_train.csv                    # terminal 2
"""
import csv
import json
import math
import os
import sys
import threading
import time

import numpy as np
import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import SetEntityState
from geometry_msgs.msg import PoseStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rcl_interfaces.srv import SetParameters
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import String
from std_srvs.srv import Trigger

from amr_sim_core.faults import FaultSampler, HealthState
from amr_sim_core.world import load_scenario, load_world

from .common import quat_from_yaw

HEALTH_PARAMS = ("wheel_gain_left", "wheel_gain_right", "torque_gain", "cmd_latency", "lidar_dropout_rate",
                 "lidar_sector_center", "lidar_sector_width", "lidar_noise_std", "encoder_scale_left",
                 "encoder_scale_right")


class RunnerHelper(Node):
    def __init__(self):
        super().__init__("episode_runner")
        self.declare_parameter("scenario", "nominal")
        self.declare_parameter("episodes", 10)
        self.declare_parameter("seed", 0)
        self.declare_parameter("output", "~/amr_results/nav2_episodes.csv")
        self.declare_parameter("robot_name", "amr_w1")
        self.declare_parameter("stop_on_collision", True)
        self.metrics = {}
        self.summary = None
        self.create_subscription(String, "amr/metrics", lambda m: setattr(self, "metrics", json.loads(m.data)), 10)
        self.create_subscription(String, "amr/episode_summary", lambda m: setattr(self, "summary", json.loads(m.data)), 10)
        self.goal_pub = self.create_publisher(PoseStamped, "amr/episode_goal", 10)
        self.cli_state = self.create_client(SetEntityState, "/gazebo/set_entity_state")
        self.cli_faults = self.create_client(SetParameters, "/fault_injector/set_parameters")
        self.cli_peds_params = self.create_client(SetParameters, "/pedestrian_sim/set_parameters")
        self.cli_peds_reset = self.create_client(Trigger, "/pedestrian_sim/reset")
        self.cli_finish = self.create_client(Trigger, "/metrics_monitor/finish")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def call(self, client, req, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f"service {client.srv_name} not available")
        fut = client.call_async(req)
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                raise RuntimeError(f"service {client.srv_name} timed out")
            time.sleep(0.01)
        return fut.result()

    def set_params(self, client, values: dict):
        req = SetParameters.Request()
        req.parameters = [Parameter(k, value=v).to_parameter_msg() for k, v in values.items()]
        return self.call(client, req)

    def teleport(self, name, x, y, yaw):
        req = SetEntityState.Request()
        req.state = EntityState()
        req.state.name = name
        req.state.pose.position.x, req.state.pose.position.y, req.state.pose.position.z = x, y, 0.02
        req.state.pose.orientation = quat_from_yaw(yaw)
        req.state.reference_frame = "world"
        return self.call(self.cli_state, req)

    def sleep_sim(self, seconds):
        t0 = self.now()
        while self.now() - t0 < seconds:
            time.sleep(0.02)


def pose_stamped(nav, x, y, yaw):
    p = PoseStamped()
    p.header.frame_id = "map"
    p.header.stamp = nav.get_clock().now().to_msg()
    p.pose.position.x, p.pose.position.y = float(x), float(y)
    p.pose.orientation = quat_from_yaw(float(yaw))
    return p


def main():
    args = sys.argv + ["--ros-args", "-p", "use_sim_time:=true"]
    rclpy.init(args=args)
    helper = RunnerHelper()
    executor = MultiThreadedExecutor()
    executor.add_node(helper)
    threading.Thread(target=executor.spin, daemon=True).start()
    nav = BasicNavigator()

    scenario_name = helper.get_parameter("scenario").value
    sc = load_scenario(scenario_name)
    world = load_world(sc.get("world", "warehouse_small"))
    rng = np.random.default_rng(helper.get_parameter("seed").value)
    sampler = FaultSampler(sc["faults"])
    ep = sc["episode"]
    out = os.path.expanduser(helper.get_parameter("output").value)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    robot_name = helper.get_parameter("robot_name").value
    stop_on_collision = helper.get_parameter("stop_on_collision").value

    sp = world.robot_spawn
    nav.setInitialPose(pose_stamped(nav, sp["x"], sp["y"], sp["yaw"]))
    nav.waitUntilNav2Active()

    write_header = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        writer = None
        for k in range(helper.get_parameter("episodes").value):
            start, goal, _ = world.sample_start_goal(rng, ep["min_goal_distance"], ep["max_goal_distance"])
            health: HealthState = sampler.sample(rng)
            payload = float(rng.uniform(*sc.get("payload", [0.0, 0.0])))
            density = str(rng.choice(sc["pedestrians"].get("density", ["medium"])))
            helper.get_logger().info(f"[{k + 1}] start={np.round(start, 2)} goal={np.round(goal, 2)} "
                                     f"faults={health.faults} payload={payload:.0f} density={density}")
            # 1) nominal robot while repositioning, 2) teleport, 3) workers, 4) localisation
            helper.set_params(helper.cli_faults, {**{p: 1.0 if "gain" in p or "scale" in p else 0.0
                                                     for p in HEALTH_PARAMS}, "onset_time": 0.0})
            helper.teleport(robot_name, *start)
            helper.set_params(helper.cli_peds_params, {"density": density})
            helper.call(helper.cli_peds_reset, Trigger.Request())
            nav.setInitialPose(pose_stamped(nav, *start))
            helper.sleep_sim(1.5)
            nav.clearAllCostmaps()
            helper.sleep_sim(0.5)
            helper.set_params(helper.cli_faults, {**{p: float(getattr(health, p)) for p in HEALTH_PARAMS},
                                                  "onset_time": float(health.onset_time), "payload_mass": payload})
            helper.summary = None
            goal_msg = pose_stamped(nav, goal[0], goal[1], math.atan2(goal[1] - start[1], goal[0] - start[0]))
            helper.goal_pub.publish(goal_msg)
            helper.sleep_sim(0.2)
            t0 = helper.now()
            nav.goToPose(goal_msg)
            aborted = None
            while not nav.isTaskComplete():
                time.sleep(0.05)
                if helper.now() - t0 > ep["max_time"]:
                    aborted = "timeout"
                elif stop_on_collision and (helper.metrics.get("ped_collisions", 0) or
                                            helper.metrics.get("static_collisions", 0)):
                    aborted = "collision"
                if aborted:
                    nav.cancelTask()
                    break
            # BasicNavigator keeps the previous task's status after cancelTask(), so label aborts ourselves
            nav_result = f"CANCELED_{aborted.upper()}" if aborted else nav.getResult()
            helper.sleep_sim(0.3)
            summary = helper.summary
            if summary is None:
                res = helper.call(helper.cli_finish, Trigger.Request())
                summary = json.loads(res.message) if res.success else {"outcome": "unknown"}
                if summary.get("outcome") == "timeout" and aborted is None and nav_result != TaskResult.SUCCEEDED:
                    summary["outcome"] = "nav_failed"
            row = {"episode": k, "scenario": scenario_name, "density": density, "payload": round(payload, 2),
                   "faults": "+".join(health.faults) or "none",
                   "severities": "+".join(f"{s:.2f}" for s in health.severities),
                   "onset_time": round(health.onset_time, 2),
                   **{p: round(float(getattr(health, p)), 4) for p in HEALTH_PARAMS},
                   "start_x": round(start[0], 2), "start_y": round(start[1], 2),
                   "goal_x": round(goal[0], 2), "goal_y": round(goal[1], 2),
                   "nav2_result": str(nav_result).split(".")[-1], **summary}
            if writer is None:
                writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
            writer.writerow(row)
            f.flush()
            helper.get_logger().info(f"[{k + 1}] outcome={row['outcome']} time={row.get('time')} "
                                     f"near_misses={row.get('near_miss_events')}")
    helper.get_logger().info(f"results written to {out}")
    nav.destroy_node()
    helper.destroy_node()
    rclpy.try_shutdown()


if __name__ == "__main__":
    main()
