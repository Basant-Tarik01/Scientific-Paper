"""Inspect the robot model in RViz: ros2 launch amr_description display.launch.py payload_mass:=40.0"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory("amr_description")
    xacro_file = os.path.join(share, "urdf", "amr_w1.urdf.xacro")
    payload = LaunchConfiguration("payload_mass")
    robot_description = ParameterValue(Command(["xacro ", xacro_file, " payload_mass:=", payload]), value_type=str)
    return LaunchDescription([
        DeclareLaunchArgument("payload_mass", default_value="0.0"),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": robot_description}]),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui"),
        Node(package="rviz2", executable="rviz2", arguments=["-d", os.path.join(share, "rviz", "robot.rviz")]),
    ])
