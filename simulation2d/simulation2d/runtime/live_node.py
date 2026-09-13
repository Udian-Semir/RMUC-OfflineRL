"""ROS 2 node that runs and publishes the interactive simulation2d world."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml

from geometry_msgs.msg import Point, PoseStamped
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from ..resources import asset_path, checkpoint_path, config_path
from ..world.environment import SentryTacticalEnv, Unit
from ..world.semantic_map import SemanticMap
from .live_engine import LiveSimulationEngine, PpoPolicy, PursuitPolicy


ACTION_SCHEMA = "simulation2d_action/v1"
STATE_SCHEMA = "simulation2d_state/v1"
BATTLEFIELD_SCHEMA = "simulation2d_battlefield/v1"


class Simulation2dNode(Node):
    def __init__(self) -> None:
        super().__init__("simulation2d")
        self._declare_parameters()
        self._paused = bool(self.get_parameter("start_paused").value)
        self._sequence = 0
        self._last_goal_xy: tuple[float, float] | None = None
        self._controller_status = "unavailable"
        self._controller_goal_reached = False
        self._gazebo_feedback: dict[str, Any] = {"available": False}

        config = self._load_config(str(self.get_parameter("world_config").value))
        env_config = dict(config.get("env") or {})
        env_config["sparring_backend"] = str(self.get_parameter("sparring_backend").value)
        env_config["horizon"] = int(self.get_parameter("horizon").value)
        env_config["decision_seconds"] = float(self.get_parameter("decision_seconds").value)
        seed = int(self.get_parameter("seed").value)
        env_config["seed"] = seed

        semantic_map = SemanticMap.from_aligned_json(
            self.get_parameter("semantic_map_json").value,
            obstacle_path=self.get_parameter("obstacle_map").value,
        )
        self._env = SentryTacticalEnv(semantic_map=semantic_map, **env_config)
        self._policy = self._create_policy()
        self._engine = LiveSimulationEngine(
            self._env,
            self._policy,
            seed=seed,
            loop_episodes=bool(self.get_parameter("loop_episodes").value),
        )

        self._battlefield_pub = self.create_publisher(
            String, str(self.get_parameter("battlefield_topic").value), 10)
        self._goal_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("goal_topic").value), 10)
        self._action_pub = self.create_publisher(
            String, str(self.get_parameter("action_topic").value), 10)
        self._state_pub = self.create_publisher(
            String, str(self.get_parameter("state_topic").value), 10)
        self._markers_pub = self.create_publisher(
            MarkerArray, str(self.get_parameter("markers_topic").value), 10)
        self._target_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("target_topic").value), 10)
        self._target_label_pub = self.create_publisher(
            String, str(self.get_parameter("target_label_topic").value), 10)
        self._gazebo_pose_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("gazebo_pose_topic").value), 10)
        self._goal_feedback_pub = self.create_publisher(
            String, str(self.get_parameter("goal_feedback_topic").value), 10)
        self._controller_status_sub = self.create_subscription(
            String,
            str(self.get_parameter("controller_status_topic").value),
            self._controller_status_callback,
            10,
        )
        self._tf_buffer = Buffer(cache_time=Duration(seconds=5.0))
        self._tf_listener = TransformListener(self._tf_buffer, self, spin_thread=False)
        self.create_service(Trigger, "~/reset", self._reset_callback)
        self.create_service(SetBool, "~/pause", self._pause_callback)

        update_rate_hz = float(self.get_parameter("update_rate_hz").value)
        if update_rate_hz <= 0.0:
            raise ValueError("update_rate_hz must be positive")
        self._timer = self.create_timer(1.0 / update_rate_hz, self._tick)
        self._publish_snapshot(None, None, 0.0, False)
        self.get_logger().info(
            f"simulation2d ready: policy={self._policy.name} "
            f"sparring={env_config['sparring_backend']} rate={update_rate_hz:.2f} Hz "
            f"battlefield={self._battlefield_pub.topic_name} goal={self._goal_pub.topic_name}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("world_config", str(config_path("live_ppo_00155_offline.yaml")))
        self.declare_parameter("semantic_map_json", str(asset_path("semantic_map_aligned.json")))
        self.declare_parameter("obstacle_map", str(asset_path("blackwhite_map.png")))
        self.declare_parameter("policy_mode", "ppo")
        self.declare_parameter("checkpoint_path", str(checkpoint_path()))
        self.declare_parameter("device", "cpu")
        self.declare_parameter("deterministic", True)
        self.declare_parameter("sparring_backend", "offline")
        self.declare_parameter("seed", 7)
        self.declare_parameter("horizon", 420)
        self.declare_parameter("decision_seconds", 1.0)
        self.declare_parameter("update_rate_hz", 1.0)
        self.declare_parameter("loop_episodes", True)
        self.declare_parameter("start_paused", False)
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("battlefield_topic", "/simulation2d/battlefield/state")
        self.declare_parameter("goal_topic", "/rl/policy_goal_pose")
        self.declare_parameter("action_topic", "/rl/tactical_action")
        self.declare_parameter("state_topic", "/simulation2d/state")
        self.declare_parameter("markers_topic", "/simulation2d/markers")
        self.declare_parameter("target_topic", "/simulation2d/selected_target_pose")
        self.declare_parameter("target_label_topic", "/simulation2d/selected_target_label")
        self.declare_parameter("gazebo_pose_topic", "/simulation2d/gazebo_sentry_pose")
        self.declare_parameter("goal_feedback_topic", "/simulation2d/gazebo_goal_feedback")
        self.declare_parameter("controller_status_topic", "/navigation/controller_status")
        self.declare_parameter("gazebo_base_frame", "base_footprint")
        self.declare_parameter("gazebo_goal_tolerance_m", 0.35)
        self.declare_parameter("enable_gazebo_feedback", True)

    @staticmethod
    def _load_config(path: str) -> dict[str, Any]:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        if not isinstance(payload, dict):
            raise ValueError("world_config must contain a YAML mapping")
        return payload

    def _create_policy(self):
        mode = str(self.get_parameter("policy_mode").value).lower()
        if mode == "pursuit":
            return PursuitPolicy(self._env)
        if mode == "ppo":
            checkpoint = str(self.get_parameter("checkpoint_path").value)
            if not checkpoint:
                raise ValueError("checkpoint_path is required when policy_mode=ppo")
            return PpoPolicy(
                self._env,
                checkpoint,
                device=str(self.get_parameter("device").value),
                deterministic=bool(self.get_parameter("deterministic").value),
            )
        raise ValueError("policy_mode must be 'pursuit' or 'ppo'")

    def _tick(self) -> None:
        if self._paused or self._engine.finished:
            return
        try:
            self._sync_external_sentry_state()
            result = self._engine.step()
            self._sequence += 1
            self._publish_snapshot(result.action, result.info, result.reward, result.done)
        except Exception as exc:
            self._paused = True
            self.get_logger().error(
                f"simulation2d tick failed and was paused: {type(exc).__name__}: {exc}")

    def _sync_external_sentry_state(self) -> None:
        """Feed the real Gazebo pose into PPO before it selects the next action."""
        if not bool(self.get_parameter("enable_gazebo_feedback").value):
            self._engine.clear_external_sentry_state()
            return
        transform = self._lookup_gazebo_transform()
        if transform is None:
            self._engine.clear_external_sentry_state()
            return
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        yaw = math.atan2(
            2.0 * (rotation.w * rotation.z + rotation.x * rotation.y),
            1.0 - 2.0 * (rotation.y * rotation.y + rotation.z * rotation.z),
        )
        self._engine.set_external_sentry_state(
            time_s=float(self._env.match.time_s),
            x_m=float(translation.x),
            y_m=float(translation.y),
            hp=float(self._env.sentry.hp),
            max_hp=float(self._env.sentry.max_hp),
            yaw_deg=math.degrees(yaw),
            heat=float(self._env.sentry.heat),
            ammo17_fired=float(self._env.sentry.shots_fired),
            power_w=float(self._env.sentry.power_w),
            simulate_fire=True,
            goal_reached=self._controller_goal_reached,
        )

    def _controller_status_callback(self, message: String) -> None:
        self._controller_status = message.data
        state = message.data.split(":", 1)[0]
        if state in {"goal_reached", "goal_reached_position"}:
            self._controller_goal_reached = True
        elif state.startswith("tracking_") or state.startswith("recovery_"):
            self._controller_goal_reached = False

    def _reset_callback(self, _request: Trigger.Request, response: Trigger.Response):
        self._engine.reset()
        self._sequence = 0
        self._last_goal_xy = None
        self._controller_goal_reached = False
        self._gazebo_feedback = {"available": False, "reason": "episode_reset"}
        stamp = self.get_clock().now().to_msg()
        invalid_goal = PoseStamped()
        invalid_goal.header.stamp = stamp
        self._goal_pub.publish(invalid_goal)
        invalid_target = PoseStamped()
        invalid_target.header.stamp = stamp
        self._target_pub.publish(invalid_target)
        self._target_label_pub.publish(self._string_message("TARGET NONE"))
        self._goal_feedback_pub.publish(self._string_message("GAZEBO waiting for new goal"))
        self._publish_snapshot(None, None, 0.0, False)
        response.success = True
        response.message = f"reset episode={self._engine.episode}"
        return response

    def _pause_callback(self, request: SetBool.Request, response: SetBool.Response):
        self._paused = bool(request.data)
        response.success = True
        response.message = "paused" if self._paused else "running"
        return response

    def _publish_snapshot(
        self,
        action: tuple[int, int, int] | None,
        info: dict[str, Any] | None,
        reward: float,
        done: bool,
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        self._battlefield_pub.publish(self._json_message(self._battlefield_payload(stamp)))
        if action is not None and info is not None:
            goal = self._goal_message(stamp, info)
            if goal is not None:
                self._goal_pub.publish(goal)
                self._track_goal(goal.pose.position.x, goal.pose.position.y)
            target_pose, target_label = self._target_message(stamp, info)
            self._target_pub.publish(target_pose)
            self._target_label_pub.publish(self._string_message(target_label))
            self._action_pub.publish(self._json_message(self._action_payload(action, info, stamp)))
        self._update_gazebo_feedback(stamp)
        self._markers_pub.publish(self._marker_message(stamp, info))
        self._state_pub.publish(
            self._json_message(self._state_payload(action, info, reward, done, stamp)))

    def _track_goal(self, x_m: float, y_m: float) -> None:
        goal = (float(x_m), float(y_m))
        if self._last_goal_xy is None or math.dist(goal, self._last_goal_xy) > 0.02:
            self._controller_goal_reached = False
        self._last_goal_xy = goal

    def _target_message(
        self, stamp, info: dict[str, Any],
    ) -> tuple[PoseStamped, str]:
        target_index = int(info.get("executed_target_idx", self._env.NONE_TARGET))
        message = PoseStamped()
        message.header.stamp = stamp
        target = self._selected_target(target_index)
        if target is None:
            return message, "TARGET NONE"
        x_m, y_m, label = target
        message.header.frame_id = str(self.get_parameter("frame_id").value)
        message.pose.position.x = x_m
        message.pose.position.y = y_m
        message.pose.orientation.w = 1.0
        return message, label

    def _selected_target(self, target_index: int) -> tuple[float, float, str] | None:
        if 0 <= target_index < self._env.ROBOT_TARGETS:
            unit = self._env.enemies[target_index]
            x_m, y_m = self._env.map.cell_to_meters(unit.cell)
            return x_m, y_m, f"TARGET {unit.team} {unit.role} HP {unit.hp:.0f}/{unit.max_hp:.0f}"
        if target_index == self._env.BLUE_OUTPOST_TARGET:
            x_m, y_m = self._env.map.cell_to_meters(self._env.map.blue_outpost)
            return x_m, y_m, f"TARGET blue outpost HP {self._env.match.blue.outpost_hp:.0f}"
        if target_index == self._env.BLUE_BASE_TARGET:
            x_m, y_m = self._env.map.cell_to_meters(self._env.map.blue_base)
            return x_m, y_m, f"TARGET blue base HP {self._env.match.blue.base_hp:.0f}"
        return None

    def _update_gazebo_feedback(self, stamp) -> None:
        invalid_pose = PoseStamped()
        invalid_pose.header.stamp = stamp
        if not bool(self.get_parameter("enable_gazebo_feedback").value):
            self._gazebo_feedback = {"available": False, "reason": "disabled"}
            self._gazebo_pose_pub.publish(invalid_pose)
            self._goal_feedback_pub.publish(self._string_message("GAZEBO feedback disabled"))
            return
        frame_id = str(self.get_parameter("frame_id").value)
        base_frame = str(self.get_parameter("gazebo_base_frame").value)
        transform = self._lookup_gazebo_transform()
        if transform is None:
            self._gazebo_feedback = {"available": False, "reason": "tf_unavailable"}
            self._gazebo_pose_pub.publish(invalid_pose)
            self._goal_feedback_pub.publish(self._string_message("GAZEBO TF unavailable"))
            return

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = frame_id
        pose.pose.position.x = transform.transform.translation.x
        pose.pose.position.y = transform.transform.translation.y
        pose.pose.position.z = transform.transform.translation.z
        pose.pose.orientation = transform.transform.rotation
        self._gazebo_pose_pub.publish(pose)

        distance_m = None
        geometric_reached = False
        if self._last_goal_xy is not None:
            distance_m = math.dist(
                (pose.pose.position.x, pose.pose.position.y), self._last_goal_xy)
            geometric_reached = distance_m <= float(
                self.get_parameter("gazebo_goal_tolerance_m").value)
        reached = geometric_reached or self._controller_goal_reached
        self._gazebo_feedback = {
            "available": True,
            "x_m": pose.pose.position.x,
            "y_m": pose.pose.position.y,
            "distance_to_goal_m": distance_m,
            "geometric_reached": geometric_reached,
            "controller_reached": self._controller_goal_reached,
            "reached": reached,
            "controller_status": self._controller_status,
        }
        state = "ARRIVED" if reached else "MOVING"
        distance_text = "--" if distance_m is None else f"{distance_m:.2f}m"
        self._goal_feedback_pub.publish(self._string_message(
            f"GAZEBO {state} goal_distance={distance_text} controller={self._controller_status}"))

    def _lookup_gazebo_transform(self):
        frame_id = str(self.get_parameter("frame_id").value)
        base_frame = str(self.get_parameter("gazebo_base_frame").value)
        try:
            return self._tf_buffer.lookup_transform(frame_id, base_frame, Time())
        except TransformException:
            return None

    def _battlefield_payload(self, stamp) -> dict[str, Any]:
        units = (self._env.sentry, *self._env.allies, *self._env.enemies)
        sparring_units = (*self._env.allies, *self._env.enemies)
        active_commands = sum(
            unit.alive and unit.unit_id in self._env._sparring_commands
            for unit in sparring_units
        )
        return {
            "schema": BATTLEFIELD_SCHEMA,
            "sequence": self._sequence,
            "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
            "frame_id": str(self.get_parameter("frame_id").value),
            "field": {
                "x_m": self._env.map.width * self._env.map.resolution_m,
                "y_m": self._env.map.height * self._env.map.resolution_m,
            },
            "sparring": {
                "backend": self._env.sparring_backend,
                "profile": self._env.blue_sparring_profile,
                "commands": active_commands,
                "expected_commands": sum(unit.alive for unit in sparring_units),
                "roster_size": len(sparring_units),
            },
            "units": [self._unit_payload(unit) for unit in units],
        }

    def _unit_payload(self, unit: Unit) -> dict[str, Any]:
        x_m, y_m = self._env.map.cell_to_meters(unit.cell)
        return {
            "unit_id": unit.unit_id,
            "team": unit.team,
            "role": unit.role,
            "x_m": x_m,
            "y_m": y_m,
            "hp": unit.hp,
            "max_hp": unit.max_hp,
            "alive": unit.alive,
            "respawn_remaining_s": unit.respawn_remaining_s,
        }

    def _goal_message(self, stamp, info: dict[str, Any]) -> PoseStamped | None:
        position = info.get("execution_goal_xy_m")
        if position is None and self._env.active_goal_cell is not None:
            position = self._env.map.cell_to_meters(self._env.active_goal_cell)
        if position is None:
            return None
        message = PoseStamped()
        message.header.stamp = stamp
        message.header.frame_id = str(self.get_parameter("frame_id").value)
        message.pose.position.x = float(position[0])
        message.pose.position.y = float(position[1])
        message.pose.orientation.w = 1.0
        return message

    def _marker_message(self, stamp, info: dict[str, Any] | None) -> MarkerArray:
        markers = MarkerArray()
        clear = Marker()
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)
        units = (self._env.sentry, *self._env.allies, *self._env.enemies)
        for index, unit in enumerate(units):
            x_m, y_m = self._env.map.cell_to_meters(unit.cell)
            body = Marker()
            body.header.stamp = stamp
            body.header.frame_id = str(self.get_parameter("frame_id").value)
            body.ns = "simulation2d_units"
            body.id = index
            body.type = Marker.CYLINDER
            body.action = Marker.ADD
            body.pose.position.x = x_m
            body.pose.position.y = y_m
            body.pose.position.z = 0.12
            body.pose.orientation.w = 1.0
            body.scale.x = body.scale.y = 0.42 if unit.role == "sentry" else 0.32
            body.scale.z = 0.24
            if unit.team == "red":
                body.color.r, body.color.g, body.color.b = 0.9, 0.15, 0.12
            else:
                body.color.r, body.color.g, body.color.b = 0.12, 0.3, 0.95
            body.color.a = 1.0 if unit.alive else 0.2
            markers.markers.append(body)

            label = Marker()
            label.header = body.header
            label.ns = "simulation2d_labels"
            label.id = index
            label.type = Marker.TEXT_VIEW_FACING
            label.action = Marker.ADD
            label.pose.position.x = x_m
            label.pose.position.y = y_m
            label.pose.position.z = 0.55
            label.pose.orientation.w = 1.0
            label.scale.z = 0.22
            label.color.r = label.color.g = label.color.b = label.color.a = 1.0
            label.text = f"{unit.team} {unit.role} {unit.hp:.0f}/{unit.max_hp:.0f}"
            markers.markers.append(label)

        if self._env.active_goal_cell is not None:
            x_m, y_m = self._env.map.cell_to_meters(self._env.active_goal_cell)
            goal = Marker()
            goal.header.stamp = stamp
            goal.header.frame_id = str(self.get_parameter("frame_id").value)
            goal.ns = "simulation2d_goal"
            goal.id = 0
            goal.type = Marker.SPHERE
            goal.action = Marker.ADD
            goal.pose.position.x = x_m
            goal.pose.position.y = y_m
            goal.pose.position.z = 0.18
            goal.pose.orientation.w = 1.0
            goal.scale.x = goal.scale.y = goal.scale.z = 0.32
            goal.color.r, goal.color.g, goal.color.b, goal.color.a = 1.0, 0.85, 0.05, 1.0
            markers.markers.append(goal)

            goal_label = Marker()
            goal_label.header = goal.header
            goal_label.ns = "simulation2d_goal_label"
            goal_label.id = 0
            goal_label.type = Marker.TEXT_VIEW_FACING
            goal_label.action = Marker.ADD
            goal_label.pose.position.x = x_m
            goal_label.pose.position.y = y_m
            goal_label.pose.position.z = 0.65
            goal_label.pose.orientation.w = 1.0
            goal_label.scale.z = 0.28
            goal_label.color.r, goal_label.color.g = 1.0, 0.85
            goal_label.color.a = 1.0
            goal_label.text = f"PPO GOAL ({x_m:.2f}, {y_m:.2f})"
            markers.markers.append(goal_label)

        target_index = int((info or {}).get("executed_target_idx", self._env.NONE_TARGET))
        target = self._selected_target(target_index)
        if target is not None:
            target_x, target_y, target_label_text = target
            target_marker = Marker()
            target_marker.header.stamp = stamp
            target_marker.header.frame_id = str(self.get_parameter("frame_id").value)
            target_marker.ns = "simulation2d_target"
            target_marker.id = 0
            target_marker.type = Marker.CYLINDER
            target_marker.action = Marker.ADD
            target_marker.pose.position.x = target_x
            target_marker.pose.position.y = target_y
            target_marker.pose.position.z = 0.16
            target_marker.pose.orientation.w = 1.0
            target_marker.scale.x = target_marker.scale.y = 0.72
            target_marker.scale.z = 0.08
            target_marker.color.r, target_marker.color.b, target_marker.color.a = 1.0, 1.0, 0.85
            markers.markers.append(target_marker)

            target_label = Marker()
            target_label.header = target_marker.header
            target_label.ns = "simulation2d_target_label"
            target_label.id = 0
            target_label.type = Marker.TEXT_VIEW_FACING
            target_label.action = Marker.ADD
            target_label.pose.position.x = target_x
            target_label.pose.position.y = target_y
            target_label.pose.position.z = 0.85
            target_label.pose.orientation.w = 1.0
            target_label.scale.z = 0.26
            target_label.color.r, target_label.color.b, target_label.color.a = 1.0, 1.0, 1.0
            target_label.text = target_label_text
            markers.markers.append(target_label)

            intent_line = Marker()
            intent_line.header = target_marker.header
            intent_line.ns = "simulation2d_target_intent"
            intent_line.id = 0
            intent_line.type = Marker.LINE_LIST
            intent_line.action = Marker.ADD
            intent_line.scale.x = 0.04
            intent_line.color.r, intent_line.color.b, intent_line.color.a = 1.0, 1.0, 0.7
            intent_line.points = [self._unit_point(self._env.sentry), self._point(target_x, target_y, 0.25)]
            markers.markers.append(intent_line)
        elif info is not None:
            sentry_x, sentry_y = self._env.map.cell_to_meters(self._env.sentry.cell)
            no_target_label = Marker()
            no_target_label.header.stamp = stamp
            no_target_label.header.frame_id = str(self.get_parameter("frame_id").value)
            no_target_label.ns = "simulation2d_target_label"
            no_target_label.id = 0
            no_target_label.type = Marker.TEXT_VIEW_FACING
            no_target_label.action = Marker.ADD
            no_target_label.pose.position.x = sentry_x
            no_target_label.pose.position.y = sentry_y
            no_target_label.pose.position.z = 1.05
            no_target_label.pose.orientation.w = 1.0
            no_target_label.scale.z = 0.26
            no_target_label.color.r = 1.0
            no_target_label.color.b = 1.0
            no_target_label.color.a = 1.0
            no_target_label.text = "PPO TARGET NONE"
            markers.markers.append(no_target_label)

        if bool(self._gazebo_feedback.get("available")):
            actual_x = float(self._gazebo_feedback["x_m"])
            actual_y = float(self._gazebo_feedback["y_m"])
            actual = Marker()
            actual.header.stamp = stamp
            actual.header.frame_id = str(self.get_parameter("frame_id").value)
            actual.ns = "simulation2d_gazebo_sentry"
            actual.id = 0
            actual.type = Marker.CUBE
            actual.action = Marker.ADD
            actual.pose.position.x = actual_x
            actual.pose.position.y = actual_y
            actual.pose.position.z = 0.2
            actual.pose.orientation.w = 1.0
            actual.scale.x = actual.scale.y = actual.scale.z = 0.42
            actual.color.g, actual.color.a = 1.0, 1.0
            markers.markers.append(actual)

            actual_label = Marker()
            actual_label.header = actual.header
            actual_label.ns = "simulation2d_gazebo_sentry_label"
            actual_label.id = 0
            actual_label.type = Marker.TEXT_VIEW_FACING
            actual_label.action = Marker.ADD
            actual_label.pose.position.x = actual_x
            actual_label.pose.position.y = actual_y
            actual_label.pose.position.z = 0.8
            actual_label.pose.orientation.w = 1.0
            actual_label.scale.z = 0.28
            actual_label.color.g, actual_label.color.a = 1.0, 1.0
            state = "ARRIVED" if self._gazebo_feedback.get("reached") else "MOVING"
            distance = self._gazebo_feedback.get("distance_to_goal_m")
            distance_text = "--" if distance is None else f"{float(distance):.2f}m"
            actual_label.text = f"GAZEBO SENTRY {state} d={distance_text}"
            markers.markers.append(actual_label)

            if self._last_goal_xy is not None:
                feedback_line = Marker()
                feedback_line.header = actual.header
                feedback_line.ns = "simulation2d_gazebo_goal_error"
                feedback_line.id = 0
                feedback_line.type = Marker.LINE_LIST
                feedback_line.action = Marker.ADD
                feedback_line.scale.x = 0.06
                if self._gazebo_feedback.get("reached"):
                    feedback_line.color.g = 1.0
                else:
                    feedback_line.color.r, feedback_line.color.g = 1.0, 0.55
                feedback_line.color.a = 0.9
                feedback_line.points = [
                    self._point(actual_x, actual_y, 0.3),
                    self._point(*self._last_goal_xy, 0.3),
                ]
                markers.markers.append(feedback_line)

        for index, event in enumerate((info or {}).get("combat_events", ())):
            attacker = self._unit_by_id(int(event["attacker_id"]))
            target_id = event.get("target_id")
            target = self._unit_by_id(int(target_id)) if target_id is not None else None
            if attacker is None or target is None:
                continue
            line = Marker()
            line.header.stamp = stamp
            line.header.frame_id = str(self.get_parameter("frame_id").value)
            line.ns = "simulation2d_combat"
            line.id = index
            line.type = Marker.LINE_LIST
            line.action = Marker.ADD
            line.scale.x = 0.06
            line.color.r, line.color.g, line.color.b, line.color.a = 1.0, 0.7, 0.05, 0.9
            line.lifetime.sec = 1
            line.points = [self._unit_point(attacker), self._unit_point(target)]
            markers.markers.append(line)
        return markers

    def _unit_by_id(self, unit_id: int) -> Unit | None:
        return next(
            (unit for unit in (self._env.sentry, *self._env.allies, *self._env.enemies)
             if unit.unit_id == unit_id),
            None,
        )

    def _unit_point(self, unit: Unit) -> Point:
        x_m, y_m = self._env.map.cell_to_meters(unit.cell)
        return self._point(x_m, y_m, 0.25)

    @staticmethod
    def _point(x_m: float, y_m: float, z_m: float) -> Point:
        point = Point()
        point.x, point.y, point.z = float(x_m), float(y_m), float(z_m)
        return point

    def _action_payload(
        self, action: tuple[int, int, int], info: dict[str, Any], stamp,
    ) -> dict[str, Any]:
        goal = info.get("execution_goal_xy_m")
        if goal is None and self._env.active_goal_cell is not None:
            goal = self._env.map.cell_to_meters(self._env.active_goal_cell)
        return {
            "schema": ACTION_SCHEMA,
            "sequence": self._sequence,
            "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
            "frame_id": str(self.get_parameter("frame_id").value),
            "policy": self._policy.name,
            "requested": {"goal_index": action[0], "target_index": action[1], "fire_mode": action[2]},
            "executed": {
                "goal_index": int(info.get("executed_goal_idx", action[0])),
                "target_index": int(info.get("executed_target_idx", action[1])),
                "fire_mode": int(info.get("executed_fire_mode", action[2])),
                "goal_x_m": float(goal[0]) if goal is not None else None,
                "goal_y_m": float(goal[1]) if goal is not None else None,
                "goal_reached": bool(info.get("goal_reached", False)),
                "route_switch_blocked": bool(info.get("goal_switch_blocked", False)),
                "target_replan_deferred": bool(
                    info.get("goal_target_replan_deferred", False)),
                "route_switch_reason": str(info.get("goal_switch_reason", "")),
            },
        }

    def _state_payload(
        self,
        action: tuple[int, int, int] | None,
        info: dict[str, Any] | None,
        reward: float,
        done: bool,
        stamp,
    ) -> dict[str, Any]:
        sentry_x, sentry_y = self._env.map.cell_to_meters(self._env.sentry.cell)
        return {
            "schema": STATE_SCHEMA,
            "sequence": self._sequence,
            "stamp": {"sec": stamp.sec, "nanosec": stamp.nanosec},
            "episode": self._engine.episode,
            "paused": self._paused,
            "done": done,
            "policy": self._policy.name,
            "sparring_backend": self._env.sparring_backend,
            "blue_sparring_profile": self._env.blue_sparring_profile,
            "sparring_commands": len(self._env._sparring_commands),
            "match_time_s": self._env.match.time_s,
            "reward": reward,
            "sentry": {"x_m": sentry_x, "y_m": sentry_y, "hp": self._env.sentry.hp},
            "buildings": {
                "red_base_hp": self._env.match.red.base_hp,
                "red_outpost_hp": self._env.match.red.outpost_hp,
                "blue_base_hp": self._env.match.blue.base_hp,
                "blue_outpost_hp": self._env.match.blue.outpost_hp,
            },
            "action": list(action) if action is not None else None,
            "outcome": (info or {}).get("outcome"),
            "gazebo": dict(self._gazebo_feedback),
        }

    @staticmethod
    def _json_message(payload: dict[str, Any]) -> String:
        message = String()
        message.data = json.dumps(payload, separators=(",", ":"), allow_nan=False)
        return message

    @staticmethod
    def _string_message(value: str) -> String:
        message = String()
        message.data = value
        return message


def main(args=None) -> None:
    rclpy.init(args=args)
    node = Simulation2dNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.try_shutdown()


if __name__ == "__main__":
    main()
