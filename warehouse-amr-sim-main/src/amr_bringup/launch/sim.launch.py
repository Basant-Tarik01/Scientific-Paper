"""Gazebo warehouse + AMR-W1 + fault injection + workers + metrics (no navigation stack).

    ros2 launch amr_bringup sim.launch.py                       # GUI + RViz, nominal robot, medium density
    ros2 launch amr_bringup sim.launch.py gui:=false rviz:=false scenario:=a1_train density:=high payload_mass:=40
    ros2 run teleop_twist_keyboard teleop_twist_keyboard       # drive it (publishes /cmd_vel)

Arguments: world, scenario, density, payload_mass, seed, gui, rviz, log_data, static_map_tf, x, y, yaw
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from amr_sim_core.world import load_world


def _spawn(context):
    world = load_world(LaunchConfiguration("world").perform(context))
    sp = world.robot_spawn
    x = LaunchConfiguration("x").perform(context) or str(sp["x"])
    y = LaunchConfiguration("y").perform(context) or str(sp["y"])
    yaw = LaunchConfiguration("yaw").perform(context) or str(sp["yaw"])
    return [Node(package="gazebo_ros", executable="spawn_entity.py", output="screen",
                 arguments=["-entity", "amr_w1", "-topic", "robot_description", "-x", x, "-y", y, "-z", "0.02",
                            "-Y", yaw])]


def _gazebo(context):
    world_name = LaunchConfiguration("world").perform(context)
    world_file = os.path.join(get_package_share_directory("warehouse_world"), "worlds", f"{world_name}.world")
    gz = get_package_share_directory("gazebo_ros")
    actions = [IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gz, "launch", "gzserver.launch.py")),
                                        launch_arguments={"world": world_file, "verbose": "false"}.items())]
    if LaunchConfiguration("gui").perform(context).lower() == "true":
        actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(gz, "launch", "gzclient.launch.py"))))
    # scoped: gzserver.launch.py declares its own 'world' argument, which must not leak into ours
    return [GroupAction(actions, scoped=True)]


def generate_launch_description():
    desc = get_package_share_directory("amr_description")
    bringup = get_package_share_directory("amr_bringup")
    xacro_file = os.path.join(desc, "urdf", "amr_w1.urdf.xacro")
    lc = LaunchConfiguration
    robot_description = ParameterValue(Command(["xacro ", xacro_file, " payload_mass:=", lc("payload_mass")]), value_type=str)
    sim_time = {"use_sim_time": True}
    args = [
        DeclareLaunchArgument("world", default_value="warehouse_small"),
        DeclareLaunchArgument("scenario", default_value="nominal", description="scenario for fault/worker sampling"),
        DeclareLaunchArgument("density", default_value="", description="none|low|medium|high|<int>; '' = from scenario"),
        DeclareLaunchArgument("payload_mass", default_value="0.0"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=os.path.join(bringup, "rviz", "sim.rviz")),
        DeclareLaunchArgument("log_data", default_value="false"),
        DeclareLaunchArgument("static_map_tf", default_value="true", description="ground-truth map->odom (false when AMCL runs)"),
        DeclareLaunchArgument("x", default_value=""),
        DeclareLaunchArgument("y", default_value=""),
        DeclareLaunchArgument("yaw", default_value=""),
    ]
    nodes = [
        OpaqueFunction(function=_gazebo),
        Node(package="robot_state_publisher", executable="robot_state_publisher", output="screen",
             parameters=[{"robot_description": robot_description}, sim_time]),
        OpaqueFunction(function=_spawn),
        Node(package="amr_sim_nodes", executable="fault_injector", output="screen",
             parameters=[sim_time, {"scenario": lc("scenario"), "seed": lc("seed"),
                                    "payload_mass": ParameterValue(lc("payload_mass"), value_type=float)}]),
        Node(package="amr_sim_nodes", executable="pedestrian_sim", output="screen",
             parameters=[sim_time, {"world": lc("world"), "scenario": lc("scenario"), "seed": lc("seed"),
                                    "density": ParameterValue(lc("density"), value_type=str)}]),
        Node(package="amr_sim_nodes", executable="metrics_monitor", output="screen",
             parameters=[sim_time, {"world": lc("world")}]),
        Node(package="amr_sim_nodes", executable="data_logger", output="screen", parameters=[sim_time],
             condition=IfCondition(lc("log_data"))),
        Node(package="amr_sim_nodes", executable="gt_localization", output="screen", parameters=[sim_time],
             condition=IfCondition(lc("static_map_tf"))),
        Node(package="rviz2", executable="rviz2", arguments=["-d", lc("rviz_config")], parameters=[sim_time],
             condition=IfCondition(lc("rviz"))),
    ]
    return LaunchDescription(args + nodes)
