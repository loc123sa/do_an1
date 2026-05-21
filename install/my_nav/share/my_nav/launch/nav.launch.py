from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():

    config = '/home/loc/ros2_ws/src/my_nav/config/nav2_params.yaml'

    return LaunchDescription([

        Node(
            package='nav2_controller',
            executable='controller_server',
            output='screen',
            parameters=[config]
        ),

        Node(
            package='nav2_planner',
            executable='planner_server',
            output='screen',
            parameters=[config]
        ),

        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            output='screen',
            parameters=[config]
        ),

    ])
