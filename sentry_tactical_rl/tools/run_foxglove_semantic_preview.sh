#!/usr/bin/env bash
# Start the RMUC-OfflineRL semantic preview and its dedicated Foxglove bridge.
#
# The preview intentionally uses domain 42 and port 8766 by default so it
# cannot leak into or consume an existing robot/Gazebo ROS graph on domain 0.
set -euo pipefail

DOMAIN_ID="${1:-${ROS_DOMAIN_ID:-42}}"
PORT="${FOXGLOVE_PORT:-8766}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! [[ "$DOMAIN_ID" =~ ^[0-9]+$ ]] || (( DOMAIN_ID < 0 || DOMAIN_ID > 232 )); then
  echo "ROS domain must be an integer in [0, 232], got: $DOMAIN_ID" >&2
  exit 2
fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Foxglove port must be an integer in [1, 65535], got: $PORT" >&2
  exit 2
fi
if ss -ltnH "sport = :$PORT" | grep -q .; then
  echo "Foxglove port $PORT is already in use." >&2
  echo "Stop the existing preview/bridge, or choose another port: FOXGLOVE_PORT=8767 $0 $DOMAIN_ID" >&2
  exit 3
fi

# ROS Humble's setup script reads optional environment variables before
# defining them, so source it with nounset temporarily disabled.
set +u
source /opt/ros/humble/setup.bash
set -u
export ROS_DOMAIN_ID="$DOMAIN_ID"
cd "$ROOT_DIR"

echo "RMUC-OfflineRL semantic preview"
echo "  ROS_DOMAIN_ID=$ROS_DOMAIN_ID"
echo "  Foxglove WebSocket=ws://<this-machine-ip>:$PORT"

# Bring up the bridge first.  If it cannot bind its port, do not briefly
# launch a map publisher that has no client transport.
ros2 run foxglove_bridge foxglove_bridge --ros-args \
  -p address:=0.0.0.0 \
  -p port:="$PORT" &
BRIDGE_PID=$!

for _ in {1..20}; do
  if ss -ltnH "sport = :$PORT" | grep -q .; then
    break
  fi
  if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    wait "$BRIDGE_PID" || true
    echo "Foxglove bridge exited before listening on port $PORT." >&2
    exit 1
  fi
  sleep 0.1
done
if ! ss -ltnH "sport = :$PORT" | grep -q .; then
  kill "$BRIDGE_PID" 2>/dev/null || true
  wait "$BRIDGE_PID" 2>/dev/null || true
  echo "Foxglove bridge did not listen on port $PORT." >&2
  exit 1
fi

/usr/bin/python3 -m sentry_tactical_rl.tools.foxglove_semantic_preview \
  --ros-domain-id "$ROS_DOMAIN_ID" &
PREVIEW_PID=$!

cleanup() {
  kill "$PREVIEW_PID" 2>/dev/null || true
  kill "$BRIDGE_PID" 2>/dev/null || true
  wait "$PREVIEW_PID" 2>/dev/null || true
  wait "$BRIDGE_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait "$BRIDGE_PID"
