"""Full stack: Gazebo warehouse + AMR + faults + workers + metrics + Nav2 (AMCL, map from the world YAML).

    ros2 launch amr_bringup nav2_sim.launch.py                     # then send a "2D Goal Pose" in RViz
    ros2 launch amr_bringup nav2_sim.launch.py gui:=false scenario:=a1_train
    ros2 run amr_sim_nodes episode_runner --ros-args -p scenario:=a1_train -p episodes:=20
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _nav2(context):
    world = LaunchConfiguration("world").perform(context)
    map_yaml = os.path.join(get_package_share_directory("warehouse_world"), "maps", f"{world}.yaml")
    nav2 = get_package_share_directory("nav2_bringup")
    return [TimerAction(period=5.0, actions=[GroupAction(scoped=True, actions=[IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(nav2, "launch", "bringup_launch.py")),
        launch_arguments={"map": map_yaml, "use_sim_time": "true", "autostart": "true",
                          "params_file": LaunchConfiguration("params_file").perform(context),
                          "use_composition": "False"}.items())])])]


def generate_launch_description():
    bringup = get_package_share_directory("amr_bringup")
    lc = LaunchConfiguration
    passthrough = ["world", "scenario", "density", "payload_mass", "seed", "gui", "rviz", "log_data"]
    return LaunchDescription([
        DeclareLaunchArgument("world", default_value="warehouse_small"),
        DeclareLaunchArgument("scenario", default_value="nominal"),
        DeclareLaunchArgument("density", default_value=""),
        DeclareLaunchArgument("payload_mass", default_value="0.0"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("gui", default_value="true"),
        DeclareLaunchArgument("rviz", default_value="true"),
        DeclareLaunchArgument("log_data", default_value="false"),
        DeclareLaunchArgument("params_file", default_value=os.path.join(bringup, "config", "nav2_params.yaml")),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(bringup, "launch", "sim.launch.py")),
            launch_arguments={**{k: lc(k) for k in passthrough}, "static_map_tf": "false",
                              "rviz_config": os.path.join(bringup, "rviz", "nav2.rviz")}.items()),
        OpaqueFunction(function=_nav2),
    ])
