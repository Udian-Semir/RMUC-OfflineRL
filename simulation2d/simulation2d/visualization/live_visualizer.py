"""Self-contained OpenCV visualizer for the simulation2d ROS topics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import cv2
from geometry_msgs.msg import PoseStamped
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from ..resources import asset_path


ROLE_ORDER = {
    "hero": 0,
    "engineer": 1,
    "infantry3": 2,
    "infantry4": 3,
    "aerial": 4,
    "sentry": 5,
}
ROLE_LABEL = {
    "hero": "Hero",
    "engineer": "Engineer",
    "infantry3": "Infantry 3",
    "infantry4": "Infantry 4",
    "aerial": "Aerial",
    "sentry": "Sentry",
}
ROLE_CODE = {
    "hero": "H",
    "engineer": "E",
    "infantry3": "I3",
    "infantry4": "I4",
    "aerial": "A",
    "sentry": "S",
}
TEAM_COLOR = {
    "red": (55, 70, 230),
    "blue": (225, 120, 35),
}


class LiveBattlefieldVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("simulation2d_visualizer")
        self._declare_parameters()
        image_path = Path(str(self.get_parameter("map_image").value)).expanduser()
        source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(f"cannot load battlefield image: {image_path}")
        display_width = int(self.get_parameter("map_display_width").value)
        if display_width <= 0:
            display_width = source.shape[1]
        display_height = max(1, round(source.shape[0] * display_width / source.shape[1]))
        self._map_image = cv2.resize(
            source, (display_width, display_height), interpolation=cv2.INTER_AREA)
        self._field_x_m = 28.0
        self._field_y_m = 15.0
        self._units: list[dict[str, Any]] = []
        self._goal: tuple[float, float] | None = None
        self._target: tuple[float, float] | None = None
        self._target_label = "TARGET waiting"
        self._gazebo_pose: tuple[float, float] | None = None
        self._gazebo_feedback = "GAZEBO feedback waiting"
        self._last_sequence = -1
        self._sparring_backend = "waiting"
        self._sparring_profile = "--"
        self._sparring_commands = 0
        self._expected_commands = 11
        self._show_window = bool(self.get_parameter("show_window").value)
        self._snapshot_path = str(self.get_parameter("snapshot_path").value)
        self._window_name = "simulation2D Interactive World"

        self._battlefield_sub = self.create_subscription(
            String,
            str(self.get_parameter("battlefield_topic").value),
            self._battlefield_callback,
            10,
        )
        self._goal_sub = self.create_subscription(
            PoseStamped,
            str(self.get_parameter("goal_topic").value),
            self._goal_callback,
            10,
        )
        self._target_sub = self.create_subscription(
            PoseStamped,
            str(self.get_parameter("target_topic").value),
            self._target_callback,
            10,
        )
        self._target_label_sub = self.create_subscription(
            String,
            str(self.get_parameter("target_label_topic").value),
            self._target_label_callback,
            10,
        )
        self._gazebo_pose_sub = self.create_subscription(
            PoseStamped,
            str(self.get_parameter("gazebo_pose_topic").value),
            self._gazebo_pose_callback,
            10,
        )
        self._gazebo_feedback_sub = self.create_subscription(
            String,
            str(self.get_parameter("goal_feedback_topic").value),
            self._gazebo_feedback_callback,
            10,
        )
        if self._show_window:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            cv2.moveWindow(
                self._window_name,
                int(self.get_parameter("window_x").value),
                int(self.get_parameter("window_y").value),
            )
        self.get_logger().info(
            f"visualizer ready: battlefield={self._battlefield_sub.topic_name} "
            f"map={image_path} show_window={self._show_window}"
        )

    def _declare_parameters(self) -> None:
        self.declare_parameter("map_image", str(asset_path("semantic_map_aligned.png")))
        self.declare_parameter("map_display_width", 1120)
        self.declare_parameter("show_window", True)
        self.declare_parameter("window_x", 80)
        self.declare_parameter("window_y", 80)
        self.declare_parameter("snapshot_path", "")
        self.declare_parameter("battlefield_topic", "/simulation2d/battlefield/state")
        self.declare_parameter("goal_topic", "/rl/policy_goal_pose")
        self.declare_parameter("target_topic", "/simulation2d/selected_target_pose")
        self.declare_parameter("target_label_topic", "/simulation2d/selected_target_label")
        self.declare_parameter("gazebo_pose_topic", "/simulation2d/gazebo_sentry_pose")
        self.declare_parameter("goal_feedback_topic", "/simulation2d/gazebo_goal_feedback")

    def destroy_node(self) -> bool:
        if self._show_window:
            cv2.destroyWindow(self._window_name)
        return super().destroy_node()

    def _battlefield_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"invalid battlefield JSON: {exc}")
            return
        if payload.get("schema") != "simulation2d_battlefield/v1":
            self.get_logger().warning("ignoring unsupported battlefield schema")
            return
        units = payload.get("units")
        if not isinstance(units, list):
            self.get_logger().warning("battlefield state has no units list")
            return
        field = payload.get("field") or {}
        sparring = payload.get("sparring") or {}
        self._field_x_m = max(float(field.get("x_m", 28.0)), 0.1)
        self._field_y_m = max(float(field.get("y_m", 15.0)), 0.1)
        self._units = [unit for unit in units if isinstance(unit, dict)]
        self._sparring_backend = str(sparring.get("backend", "unknown"))
        self._sparring_profile = str(sparring.get("profile", "--"))
        self._sparring_commands = int(sparring.get("commands", 0))
        self._expected_commands = int(sparring.get("expected_commands", 11))
        self._last_sequence = int(payload.get("sequence", self._last_sequence + 1))
        self._render()

    def _goal_callback(self, message: PoseStamped) -> None:
        self._goal = self._pose_xy(message)

    def _target_callback(self, message: PoseStamped) -> None:
        self._target = self._pose_xy(message)

    def _target_label_callback(self, message: String) -> None:
        self._target_label = message.data or "TARGET NONE"

    def _gazebo_pose_callback(self, message: PoseStamped) -> None:
        self._gazebo_pose = self._pose_xy(message)

    def _gazebo_feedback_callback(self, message: String) -> None:
        self._gazebo_feedback = message.data or "GAZEBO feedback waiting"

    @staticmethod
    def _pose_xy(message: PoseStamped) -> tuple[float, float] | None:
        if not message.header.frame_id:
            return None
        return float(message.pose.position.x), float(message.pose.position.y)

    def _world_to_map(self, x_m: float, y_m: float) -> tuple[int, int]:
        height, width = self._map_image.shape[:2]
        x = int(round(x_m / self._field_x_m * (width - 1)))
        y = int(round((1.0 - y_m / self._field_y_m) * (height - 1)))
        return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))

    def _render(self) -> None:
        map_frame = self._map_image.copy()
        self._draw_units(map_frame)
        self._draw_tactical_overlay(map_frame)
        canvas = self._compose_canvas(map_frame)
        if self._snapshot_path:
            cv2.imwrite(self._snapshot_path, canvas)
        if self._show_window:
            cv2.imshow(self._window_name, canvas)
            cv2.waitKey(1)

    def _draw_tactical_overlay(self, image: np.ndarray) -> None:
        if self._goal is not None:
            point = self._world_to_map(*self._goal)
            cv2.drawMarker(image, point, (0, 235, 255), cv2.MARKER_DIAMOND, 30, 3)
            self._outlined_text(
                image, "GOAL",
                (point[0] + 14, max(22, point[1] - 12)), (0, 235, 255), 0.56,
            )
        if self._target is not None:
            point = self._world_to_map(*self._target)
            cv2.circle(image, point, 22, (220, 55, 220), 4)
            self._outlined_text(
                image, "TARGET", (point[0] + 14, point[1] + 28),
                (220, 55, 220), 0.50,
            )
        if self._gazebo_pose is not None:
            point = self._world_to_map(*self._gazebo_pose)
            arrived = "ARRIVED" in self._gazebo_feedback
            color = (70, 220, 70) if arrived else (0, 165, 255)
            cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 34, 4)
            if self._goal is not None:
                cv2.line(image, point, self._world_to_map(*self._goal), color, 2, cv2.LINE_AA)

    def _draw_units(self, image: np.ndarray) -> None:
        ordered = sorted(
            self._units,
            key=lambda unit: (str(unit.get("team")), ROLE_ORDER.get(str(unit.get("role")), 99)),
        )
        entries = [
            (unit, self._world_to_map(
                float(unit.get("x_m", 0.0)), float(unit.get("y_m", 0.0))))
            for unit in ordered
        ]
        clusters: list[list[tuple[dict[str, Any], tuple[int, int]]]] = []
        for entry in entries:
            for cluster in clusters:
                if any(math.dist(entry[1], member[1]) < 38.0 for member in cluster):
                    cluster.append(entry)
                    break
            else:
                clusters.append([entry])

        for cluster in clusters:
            count = len(cluster)
            for index, (unit, center) in enumerate(cluster):
                display = center
                role = str(unit.get("role", "unknown"))
                if count > 1:
                    spread_radius = 27.0 + min(count, 6) * 3.0
                    angle = -math.pi / 2.0 + index * 2.0 * math.pi / count
                    display = (
                        int(np.clip(center[0] + math.cos(angle) * spread_radius, 14, image.shape[1] - 14)),
                        int(np.clip(center[1] + math.sin(angle) * spread_radius, 14, image.shape[0] - 14)),
                    )
                    cv2.circle(image, center, 3, (35, 35, 35), -1, cv2.LINE_AA)
                    cv2.line(image, center, display, (75, 75, 75), 1, cv2.LINE_AA)

            alive = bool(unit.get("alive"))
            team = str(unit.get("team", ""))
            color = TEAM_COLOR.get(team, (160, 160, 160)) if alive else (75, 75, 75)
            radius = 13 if role == "sentry" else 10
            cv2.circle(image, display, radius, color, -1, cv2.LINE_AA)
            cv2.circle(image, display, radius, (250, 250, 250), 2, cv2.LINE_AA)
            if role == "sentry":
                cv2.circle(image, display, radius + 4, (0, 180, 255), 2, cv2.LINE_AA)
            label = ROLE_CODE.get(role, role[:2].upper())
            if not alive:
                label = "X"
            size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1)[0]
            origin = (display[0] - size[0] // 2, display[1] + size[1] // 2)
            cv2.putText(
                image, label, origin, cv2.FONT_HERSHEY_SIMPLEX,
                0.34, (255, 255, 255), 1, cv2.LINE_AA,
            )

    def _compose_canvas(self, map_frame: np.ndarray) -> np.ndarray:
        top_height = 92
        panel_width = 440
        height, width = map_frame.shape[:2]
        canvas = np.full((height + top_height, width + panel_width, 3), 238, dtype=np.uint8)
        canvas[top_height:, :width] = map_frame
        canvas[:top_height, :] = (30, 34, 39)
        canvas[top_height:, width:] = (245, 245, 245)

        goal_text = "PPO GOAL waiting"
        if self._goal is not None:
            goal_text = f"PPO GOAL ({self._goal[0]:.2f}, {self._goal[1]:.2f}) m"
        self._put_fit(canvas, goal_text, (18, 27), (0, 235, 255), width - 36)
        self._put_fit(canvas, self._target_label, (18, 54), (220, 55, 220), width - 36)
        gazebo_color = (70, 220, 70) if "ARRIVED" in self._gazebo_feedback else (0, 165, 255)
        self._put_fit(canvas, self._gazebo_feedback, (18, 81), gazebo_color, width - 36)
        backend_text = (
            f"{self._sparring_backend.upper()} | {self._sparring_profile} | "
            f"commands {self._sparring_commands}/{self._expected_commands}"
        )
        cv2.putText(
            canvas, backend_text, (width + 18, 27),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 225, 225), 1,
        )
        cv2.putText(
            canvas, f"seq {self._last_sequence} | units {len(self._units)}/12",
            (width + 18, 53), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (225, 225, 225), 2,
        )
        self._draw_roster(canvas, width, top_height)
        return canvas

    def _draw_roster(self, canvas: np.ndarray, panel_x: int, top_y: int) -> None:
        by_team = {
            team: sorted(
                [unit for unit in self._units if unit.get("team") == team],
                key=lambda unit: ROLE_ORDER.get(str(unit.get("role")), 99),
            )
            for team in ("red", "blue")
        }
        row_y = top_y + 28
        for team in ("red", "blue"):
            cv2.putText(
                canvas, f"{team.upper()} TEAM  {len(by_team[team])}/6 slots",
                (panel_x + 18, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                TEAM_COLOR[team], 2,
            )
            row_y += 28
            for unit in by_team[team]:
                alive = bool(unit.get("alive"))
                role = str(unit.get("role", "unknown"))
                color = TEAM_COLOR[team] if alive else (90, 90, 90)
                cv2.circle(canvas, (panel_x + 25, row_y - 5), 7, color, -1, cv2.LINE_AA)
                hp = float(unit.get("hp", 0.0))
                max_hp = max(float(unit.get("max_hp", 1.0)), 1.0)
                status = "ALIVE" if alive else f"RESPAWN {float(unit.get('respawn_remaining_s', 0.0)):.0f}s"
                text = (
                    f"{ROLE_LABEL.get(role, role):<10} "
                    f"({float(unit.get('x_m', 0.0)):5.2f}, {float(unit.get('y_m', 0.0)):5.2f}) "
                    f"HP {hp:.0f}/{max_hp:.0f} {status}"
                )
                self._put_fit(canvas, text, (panel_x + 40, row_y), color, 385, scale=0.42)
                row_y += 30
            row_y += 18

    @staticmethod
    def _outlined_text(
        image: np.ndarray, text: str, origin: tuple[int, int],
        color: tuple[int, int, int], scale: float,
    ) -> None:
        cv2.putText(
            image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
            scale, (20, 20, 20), 4, cv2.LINE_AA,
        )
        cv2.putText(
            image, text, origin, cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, 1, cv2.LINE_AA,
        )

    @staticmethod
    def _put_fit(
        image: np.ndarray, text: str, origin: tuple[int, int],
        color: tuple[int, int, int], max_width: int, *, scale: float = 0.55,
    ) -> None:
        output = str(text)
        while output and cv2.getTextSize(
            output, cv2.FONT_HERSHEY_SIMPLEX, scale, 1,
        )[0][0] > max_width:
            output = output[:-1]
        if output != text and len(output) > 3:
            output = output[:-3] + "..."
        cv2.putText(
            image, output, origin, cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, 1, cv2.LINE_AA,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LiveBattlefieldVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
