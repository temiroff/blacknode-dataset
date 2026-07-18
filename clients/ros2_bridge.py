"""Bridge a Blacknode replay stream onto a ROS 2 topic.

Subscribes to a ReplayStreamPublisher WebSocket and republishes each frame as a
``sensor_msgs/JointState`` on the chosen topic. Run it inside any environment
where ``rclpy`` is importable (a sourced ROS 2 install or WSL):

    source /opt/ros/jazzy/setup.bash
    python3 ros2_bridge.py --url ws://127.0.0.1:8765 --topic /joint_commands

Safety: this only publishes the recorded values it receives. Whether that topic
actually drives a robot is your controller's decision — keep motion disarmed
until you have confirmed the values look right in RViz or an echo.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make this folder importable so `blacknode_ws` resolves when run by path.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
except NameError:
    pass

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from blacknode_ws import connect


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765", help="StreamPublisher stream_url")
    parser.add_argument("--topic", default="/joint_commands", help="JointState topic to publish")
    args = parser.parse_args()

    rclpy.init()
    node: Node = rclpy.create_node("blacknode_replay_bridge")
    publisher = node.create_publisher(JointState, args.topic, 10)
    node.get_logger().info(f"streaming {args.url} -> {args.topic}")

    stream = connect(args.url)
    try:
        while rclpy.ok():
            frame = stream.recv_json()
            if frame is None:
                break
            if frame.get("kind") == "blacknode.stream-schema":
                node.get_logger().info(f"loaded {len(frame.get('joint_names') or [])} replay joints")
                continue
            msg = JointState()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.name = list(frame.get("joint_names") or [])
            msg.position = [float(value) for value in (frame.get("positions") or [])]
            publisher.publish(msg)
    finally:
        stream.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
