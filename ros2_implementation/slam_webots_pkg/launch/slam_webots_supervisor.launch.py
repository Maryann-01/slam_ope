from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'world_file',
            default_value='/mnt/c/Users/amara/Documents/WebotsProject/worlds/turtlebot_slam.wbt',
            description='Full path to your .wbt world file'
        ),

        # ROS 2 Supervisor node (connects to already-running Webots)
        Node(
            package='webots_ros2_driver',
            executable='ros2_supervisor.py',          # ← This was the missing .py
            name='ros2_supervisor',
            output='screen',
            parameters=[{
                'world_file': LaunchConfiguration('world_file'),
                'use_sim_time': True,
            }],
            arguments=['--robot-name', 'TurtleBot3Burger']
        ),
    ])
