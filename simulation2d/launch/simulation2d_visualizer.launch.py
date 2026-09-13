"""Launch simulation2d with its self-contained OpenCV visualizer."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from simulation2d.resources import checkpoint_path


def generate_launch_description() -> LaunchDescription:
    simulation_share = Path(get_package_share_directory("simulation2d"))
    battlefield_topic = "/simulation2d/battlefield/state"
    return LaunchDescription([
        DeclareLaunchArgument("policy_mode", default_value="ppo"),
        DeclareLaunchArgument("checkpoint_path", default_value=str(checkpoint_path())),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("sparring_backend", default_value="offline"),
        DeclareLaunchArgument("update_rate_hz", default_value="1.0"),
        DeclareLaunchArgument("show_battlefield", default_value="true"),
        DeclareLaunchArgument("snapshot_path", default_value=""),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                str(simulation_share / "launch" / "simulation2d.launch.py")),
            launch_arguments={
                "policy_mode": LaunchConfiguration("policy_mode"),
                "checkpoint_path": LaunchConfiguration("checkpoint_path"),
                "device": LaunchConfiguration("device"),
                "sparring_backend": LaunchConfiguration("sparring_backend"),
                "update_rate_hz": LaunchConfiguration("update_rate_hz"),
                "battlefield_topic": battlefield_topic,
            }.items(),
        ),
        Node(
            package="simulation2d",
            executable="simulation2d_visualizer",
            name="simulation2d_visualizer",
            output="screen",
            parameters=[
                {
                    "map_image": str(simulation_share / "assets" / "semantic_map_aligned.png"),
                    "battlefield_topic": battlefield_topic,
                    "goal_topic": "/rl/policy_goal_pose",
                    "target_topic": "/simulation2d/selected_target_pose",
                    "target_label_topic": "/simulation2d/selected_target_label",
                    "gazebo_pose_topic": "/simulation2d/gazebo_sentry_pose",
                    "goal_feedback_topic": "/simulation2d/gazebo_goal_feedback",
                    "show_window": ParameterValue(
                        LaunchConfiguration("show_battlefield"), value_type=bool),
                    "snapshot_path": LaunchConfiguration("snapshot_path"),
                },
            ],
        ),
    ])
