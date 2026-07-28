"""Publish this project's black/white map and semantic polygons for Foxglove.

Run from the RMUC-OfflineRL repository after sourcing ROS 2 Humble::

    /usr/bin/python3 -m sentry_tactical_rl.tools.foxglove_semantic_preview

The node publishes a separate preview map and marker topic.  It never writes
to ``/map`` and therefore cannot interfere with a navigation stack.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

try:  # Keep importing this file harmless in the non-ROS training environment.
    import rclpy
    from geometry_msgs.msg import Point, PointStamped, TransformStamped
    from nav_msgs.msg import OccupancyGrid
    from rclpy.executors import ExternalShutdownException
    from rclpy.node import Node
    from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
    from tf2_ros import StaticTransformBroadcaster
    from visualization_msgs.msg import Marker, MarkerArray
except ImportError as exc:  # pragma: no cover - depends on ROS system Python
    raise RuntimeError(
        "ROS 2 messages are unavailable. Run with /usr/bin/python3 after "
        "sourcing /opt/ros/humble/setup.bash."
    ) from exc

from ..foxglove_preview import SemanticPreview, load_semantic_preview


class SemanticPreviewNode(Node):
    def __init__(self, preview: SemanticPreview, *, frame_id: str, parent_frame: str,
                 map_topic: str, marker_topic: str, click_topic: str, labels: bool, publish_period_s: float,
                 origin_m: tuple[float, float]) -> None:
        super().__init__("rm_offline_rl_semantic_preview")
        self.preview = preview
        self.frame_id = frame_id
        self.parent_frame = parent_frame
        self.labels = labels
        self.origin_m = origin_m
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.map_pub = self.create_publisher(OccupancyGrid, map_topic, qos)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, qos)
        self.click_sub = self.create_subscription(PointStamped, click_topic, self._on_clicked_point, 10)
        self.frame_broadcaster = StaticTransformBroadcaster(self)
        self._publish_frame_transform(parent_frame)
        self._publish()
        self.timer = self.create_timer(max(0.1, publish_period_s), self._publish)
        self.get_logger().info(
            f"preview map={map_topic} regions={marker_topic} frame={frame_id} "
            f"image={preview.width}x{preview.height} regions={len(preview.regions)}"
        )

    def _on_clicked_point(self, message: PointStamped) -> None:
        # Foxglove's 3D "Publish point" tool emits in its selected display
        # frame.  The preview's map -> semantic_preview transform is identity,
        # so either frame has the same coordinates here.
        if message.header.frame_id not in {self.parent_frame, self.frame_id}:
            self.get_logger().warning(
                f"ignored click in frame {message.header.frame_id!r}; use Display frame {self.parent_frame!r}"
            )
            return
        point_in_preview = (
            message.point.x - self.origin_m[0],
            message.point.y - self.origin_m[1],
        )
        x_m, y_m = self.preview.transform.source_point(point_in_preview)
        self.get_logger().info(
            f"clicked {message.header.frame_id}: ({message.point.x:.3f}, {message.point.y:.3f}) "
            f"-> polygon_xy_m: [{x_m:.3f}, {y_m:.3f}]"
        )

    def _publish_frame_transform(self, parent_frame: str) -> None:
        if parent_frame == self.frame_id:
            return
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = parent_frame
        transform.child_frame_id = self.frame_id
        transform.transform.rotation.w = 1.0
        self.frame_broadcaster.sendTransform(transform)

    def _publish(self) -> None:
        stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self._grid_message(stamp))
        self.marker_pub.publish(self._marker_message(stamp))

    def _grid_message(self, stamp) -> OccupancyGrid:
        msg = OccupancyGrid()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.info.resolution = float(self.preview.resolution_m)
        msg.info.width = self.preview.width
        msg.info.height = self.preview.height
        msg.info.origin.position.x, msg.info.origin.position.y = self.origin_m
        msg.info.origin.orientation.w = 1.0
        msg.data = self.preview.occupancy.reshape(-1).tolist()
        return msg

    def _marker_message(self, stamp) -> MarkerArray:
        markers = MarkerArray()
        for index, region in enumerate(self.preview.regions):
            outline = Marker()
            outline.header.stamp = stamp
            outline.header.frame_id = self.frame_id
            outline.ns = "semantic_outline"
            outline.id = index
            outline.type = Marker.LINE_STRIP
            outline.action = Marker.ADD
            outline.pose.orientation.w = 1.0
            outline.scale.x = max(0.03, self.preview.resolution_m * 2.0)
            outline.color.r, outline.color.g, outline.color.b, outline.color.a = region.color
            outline.points = [
                _point(x + self.origin_m[0], y + self.origin_m[1], 0.04)
                for x, y in (*region.points_m, region.points_m[0])
            ]
            markers.markers.append(outline)
            if self.labels:
                label = Marker()
                label.header.stamp = stamp
                label.header.frame_id = self.frame_id
                label.ns = "semantic_label"
                label.id = index
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose.orientation.w = 1.0
                label.pose.position.x = region.label_position_m[0] + self.origin_m[0]
                label.pose.position.y = region.label_position_m[1] + self.origin_m[1]
                label.pose.position.z = 0.08
                label.scale.z = 0.28
                label.color.r, label.color.g, label.color.b, label.color.a = region.color
                label.text = f"{region.region_id} ({region.kind})"
                markers.markers.append(label)
        return markers


def _point(x: float, y: float, z: float) -> Point:
    point = Point()
    point.x, point.y, point.z = float(x), float(y), float(z)
    return point


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Publish RMUC-OfflineRL semantic map preview to Foxglove via ROS 2")
    parser.add_argument("--json", default=str(root / "sentry_tactical_rl/assets/semantic_map_aligned.json"))
    parser.add_argument("--obstacle-map", default=None, help="override the black/white PNG recorded by the JSON")
    parser.add_argument("--frame-id", default="semantic_preview")
    parser.add_argument("--parent-frame", default="map", help="parent for the preview-only static transform")
    parser.add_argument("--origin-x", type=float, default=0.0, help="preview-map lower-left x in its parent frame")
    parser.add_argument("--origin-y", type=float, default=0.0, help="preview-map lower-left y in its parent frame")
    parser.add_argument("--map-topic", default="/sentry/semantic_map")
    parser.add_argument("--marker-topic", default="/sentry/semantic_regions")
    parser.add_argument("--click-topic", default="/clicked_point", help="Foxglove clicked-point input topic")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--publish-period", type=float, default=1.0)
    parser.add_argument(
        "--ros-domain-id",
        type=int,
        default=int(os.environ.get("ROS_DOMAIN_ID", "42")),
        help="DDS domain; default is this project's isolated domain 42",
    )
    args = parser.parse_args()
    if not 0 <= args.ros_domain_id <= 232:
        parser.error("--ros-domain-id must be in [0, 232]")
    # Domain selection must happen before the rclpy context is initialised.
    os.environ["ROS_DOMAIN_ID"] = str(args.ros_domain_id)
    preview = load_semantic_preview(args.json, args.obstacle_map)
    # argparse has already consumed this process' user-facing flags.  Do not
    # feed them into rclpy again as if they were ROS remapping arguments.
    rclpy.init(args=[])
    node = SemanticPreviewNode(
        preview,
        frame_id=args.frame_id,
        parent_frame=args.parent_frame,
        map_topic=args.map_topic,
        marker_topic=args.marker_topic,
        click_topic=args.click_topic,
        labels=not args.no_labels,
        publish_period_s=args.publish_period,
        origin_m=(args.origin_x, args.origin_y),
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
