import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
from webots_ros2_driver.webots_controller import WebotsController

def generate_launch_description():
    pkg_turtlebot = get_package_share_directory('webots_ros2_turtlebot')
    pkg_slam      = get_package_share_directory('slam_toolbox')
    pkg_mine      = get_package_share_directory('slam_webots_pkg')

    robot_description_path = os.path.join(pkg_turtlebot, 'resource', 'turtlebot_webots.urdf')
    ros2_control_params    = os.path.join(pkg_turtlebot, 'resource', 'ros2control.yml')

    turtlebot_driver = WebotsController(
        robot_name='TurtleBot3Burger',
        parameters=[
            {'robot_description': robot_description_path,
             'use_sim_time': False,
             'set_robot_state_publisher': True},
            ros2_control_params
        ],
        remappings=[
            ('/diffdrive_controller/cmd_vel', '/cmd_vel'),
            ('/diffdrive_controller/odom', '/odom')
        ],
        respawn=True
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': '<robot name=""><link name=""/></robot>'
        }]
    )

    footprint_publisher = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        output='screen',
        arguments=['0', '0', '0', '0', '0', '0', 'base_link', 'base_footprint']
    )

    diffdrive_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['diffdrive_controller', '--controller-manager-timeout', '50']
    )

    joint_state_spawner = Node(
        package='controller_manager',
        executable='spawner',
        output='screen',
        arguments=['joint_state_broadcaster', '--controller-manager-timeout', '50']
    )

    slam_toolbox = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_slam, 'launch', 'online_sync_launch.py')
        ),
        launch_arguments={
            'slam_params_file': os.path.join(pkg_mine, 'config', 'slam_params.yaml'),
            'use_sim_time': 'false'
        }.items()
    )

    policy_node = Node(
        package='slam_webots_pkg',
        executable='behaviour_policy',
        name='behaviour_policy',
        output='screen',
        parameters=[{'use_sim_time': False}]
    )

    delayed_spawners = TimerAction(
        period=15.0,
        actions=[diffdrive_spawner, joint_state_spawner]
    )

    return LaunchDescription([
        robot_state_publisher,
        footprint_publisher,
        turtlebot_driver,
        slam_toolbox,
        delayed_spawners,
        policy_node,
    ])
