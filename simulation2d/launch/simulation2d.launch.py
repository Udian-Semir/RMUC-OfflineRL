"""Launch the real-time interactive simulation2d ROS node."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

from simulation2d.resources import checkpoint_path


def generate_launch_description() -> LaunchDescription:
    share = Path(get_package_share_directory("simulation2d"))
    config = LaunchConfiguration("world_config")
    checkpoint = LaunchConfiguration("checkpoint_path")
    policy_mode = LaunchConfiguration("policy_mode")
    return LaunchDescription([
        DeclareLaunchArgument(
            "world_config", default_value=str(share / "config" / "live_ppo_00155_offline.yaml")),
        DeclareLaunchArgument("semantic_map_json", default_value=str(share / "assets" / "semantic_map_aligned.json")),
        DeclareLaunchArgument("obstacle_map", default_value=str(share / "assets" / "blackwhite_map.png")),
        DeclareLaunchArgument("policy_mode", default_value="ppo"),
        DeclareLaunchArgument("checkpoint_path", default_value=str(checkpoint_path())),
        DeclareLaunchArgument("device", default_value="cpu"),
        DeclareLaunchArgument("sparring_backend", default_value="offline"),
        DeclareLaunchArgument("seed", default_value="7"),
        DeclareLaunchArgument("update_rate_hz", default_value="1.0"),
        DeclareLaunchArgument("start_paused", default_value="false"),
        DeclareLaunchArgument("battlefield_topic", default_value="/simulation2d/battlefield/state"),
        DeclareLaunchArgument("goal_topic", default_value="/rl/policy_goal_pose"),
        DeclareLaunchArgument("action_topic", default_value="/rl/tactical_action"),
        DeclareLaunchArgument("state_topic", default_value="/simulation2d/state"),
        DeclareLaunchArgument("markers_topic", default_value="/simulation2d/markers"),
        DeclareLaunchArgument("target_topic", default_value="/simulation2d/selected_target_pose"),
        DeclareLaunchArgument("target_label_topic", default_value="/simulation2d/selected_target_label"),
        DeclareLaunchArgument("gazebo_pose_topic", default_value="/simulation2d/gazebo_sentry_pose"),
        DeclareLaunchArgument("goal_feedback_topic", default_value="/simulation2d/gazebo_goal_feedback"),
        DeclareLaunchArgument("controller_status_topic", default_value="/navigation/controller_status"),
        DeclareLaunchArgument("gazebo_base_frame", default_value="base_footprint"),
        DeclareLaunchArgument("gazebo_goal_tolerance_m", default_value="0.35"),
        DeclareLaunchArgument("enable_gazebo_feedback", default_value="true"),
        Node(
            package="simulation2d",
            executable="simulation2d_node",
            name="simulation2d",
            output="screen",
            parameters=[{
                "world_config": config,
                "semantic_map_json": LaunchConfiguration("semantic_map_json"),
                "obstacle_map": LaunchConfiguration("obstacle_map"),
                "policy_mode": policy_mode,
                "checkpoint_path": checkpoint,
                "device": LaunchConfiguration("device"),
                "sparring_backend": LaunchConfiguration("sparring_backend"),
                "seed": ParameterValue(LaunchConfiguration("seed"), value_type=int),
                "update_rate_hz": ParameterValue(LaunchConfiguration("update_rate_hz"), value_type=float),
                "start_paused": ParameterValue(LaunchConfiguration("start_paused"), value_type=bool),
                "battlefield_topic": LaunchConfiguration("battlefield_topic"),
                "goal_topic": LaunchConfiguration("goal_topic"),
                "action_topic": LaunchConfiguration("action_topic"),
                "state_topic": LaunchConfiguration("state_topic"),
                "markers_topic": LaunchConfiguration("markers_topic"),
                "target_topic": LaunchConfiguration("target_topic"),
                "target_label_topic": LaunchConfiguration("target_label_topic"),
                "gazebo_pose_topic": LaunchConfiguration("gazebo_pose_topic"),
                "goal_feedback_topic": LaunchConfiguration("goal_feedback_topic"),
                "controller_status_topic": LaunchConfiguration("controller_status_topic"),
                "gazebo_base_frame": LaunchConfiguration("gazebo_base_frame"),
                "gazebo_goal_tolerance_m": ParameterValue(
                    LaunchConfiguration("gazebo_goal_tolerance_m"), value_type=float),
                "enable_gazebo_feedback": ParameterValue(
                    LaunchConfiguration("enable_gazebo_feedback"), value_type=bool),
            }],
        ),
    ])
