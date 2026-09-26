import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            RegisterEventHandler, Shutdown)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from webots_ros2_driver.webots_controller import WebotsController
from webots_ros2_driver.wait_for_controller_connection import (
    WaitForControllerConnection,
)


def generate_launch_description():
    pkg_turtlebot = get_package_share_directory("webots_ros2_turtlebot")
    pkg_slam = get_package_share_directory("slam_toolbox")
    pkg_mine = get_package_share_directory("slam_webots_pkg")

    robot_description_path = os.path.join(
        pkg_mine, "resource", "turtlebot_webots_ope.urdf")
    ros2_control_params = os.path.join(
        pkg_turtlebot, "resource", "ros2control.yml")

    args = [
        DeclareLaunchArgument("episode", default_value="1"),
        DeclareLaunchArgument("seed", default_value="0"),
        DeclareLaunchArgument("output_dir", default_value=os.path.expanduser("~/ope_data")),
        DeclareLaunchArgument("max_timesteps", default_value="500"),
        DeclareLaunchArgument("policy_name", default_value="behaviour_a"),
        DeclareLaunchArgument("sigma", default_value="0.3"),
        DeclareLaunchArgument("gain", default_value="1.5"),
        DeclareLaunchArgument("goal_x", default_value="3.0"),
        DeclareLaunchArgument("goal_y", default_value="1.5"),
    ]

    with open(os.path.join(pkg_mine, "resource", "bootstrap.urdf")) as f:
        bootstrap_urdf = f.read()

    use_sim_time = False

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[{
            "robot_description": bootstrap_urdf,
            "use_sim_time": use_sim_time,
        }],
    )

    footprint_publisher = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        output="screen",
        arguments=["0", "0", "0", "0", "0", "0", "base_link", "base_footprint"],
    )

    turtlebot_driver = WebotsController(
        robot_name="TurtleBot3Burger",
        parameters=[
            {
                "robot_description": robot_description_path,
                "use_sim_time": use_sim_time,
                "set_robot_state_publisher": True,
            },
            ros2_control_params,
        ],
        remappings=[
            ("/diffdrive_controller/cmd_vel", "/cmd_vel"),
            ("/diffdrive_controller/odom", "/odom"),
        ],
        respawn=False,
    )

    diffdrive_spawner = Node(
        package="controller_manager", executable="spawner", output="screen",
        arguments=["diffdrive_controller", "--controller-manager-timeout", "50"],
    )
    joint_state_spawner = Node(
        package="controller_manager", executable="spawner", output="screen",
        arguments=["joint_state_broadcaster", "--controller-manager-timeout", "50"],
    )
    waiting_spawners = WaitForControllerConnection(
        target_driver=turtlebot_driver,
        nodes_to_start=[diffdrive_spawner, joint_state_spawner],
    )

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_slam, "launch", "online_sync_launch.py")),
        launch_arguments={
            "slam_params_file": os.path.join(pkg_mine, "config", "slam_params.yaml"),
            "use_sim_time": "false",
        }.items(),
    )

    policy_node = Node(
        package="slam_webots_pkg",
        executable="behaviour_policy",
        name="behaviour_policy",
        output="screen",
        parameters=[{
            "use_sim_time": use_sim_time,
            "episode": LaunchConfiguration("episode"),
            "seed": LaunchConfiguration("seed"),
            "output_dir": LaunchConfiguration("output_dir"),
            "max_timesteps": LaunchConfiguration("max_timesteps"),
            "policy_name": LaunchConfiguration("policy_name"),
            "sigma": LaunchConfiguration("sigma"),
            "gain": LaunchConfiguration("gain"),
            "goal_x": LaunchConfiguration("goal_x"),
            "goal_y": LaunchConfiguration("goal_y"),
        }],
    )

    shutdown_on_policy_exit = RegisterEventHandler(
        OnProcessExit(
            target_action=policy_node,
            on_exit=[Shutdown(reason="episode finished")],
        )
    )

    return LaunchDescription(args + [
        robot_state_publisher,
        footprint_publisher,
        turtlebot_driver,
        waiting_spawners,
        slam_toolbox,
        policy_node,
        shutdown_on_policy_exit,
    ])
