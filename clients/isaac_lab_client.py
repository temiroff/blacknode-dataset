"""Drive an Isaac Lab / Isaac Sim articulation from a Blacknode replay stream.

Subscribes to a ReplayStreamPublisher WebSocket and writes each frame's joint
values as position targets on an articulation. Run it inside the Isaac Lab
Python environment (which already has the isaaclab / omni packages) alongside
blacknode_ws.py:

    ./isaaclab.sh -p isaac_lab_client.py --url ws://127.0.0.1:8765

This intentionally keeps the transport (blacknode_ws) separate from the scene so
you can adapt ``build_scene`` / ``set_targets`` to your own robot. The example
uses the standard ArticulationView API; swap in your asset path and joint order.
"""
from __future__ import annotations

import argparse

import torch  # Isaac Lab ships torch
from blacknode_ws import connect


def build_articulation():
    """Return an object exposing set_joint_position_target(values, joint_ids=...).

    Replace this stub with your scene. For an Isaac Lab ``Articulation``:

        from isaaclab.assets import Articulation, ArticulationCfg
        robot = Articulation(ArticulationCfg(prim_path="/World/Robot", ...))
        return robot

    The returned object must also expose ``joint_names`` (list[str]) and
    ``write_data_to_sim()`` if your workflow requires it.
    """
    raise NotImplementedError("wire build_articulation() to your Isaac Lab scene")


def order_positions(robot, frame: dict) -> list[float]:
    """Reorder streamed joints to the articulation's own joint order."""
    stream_names = frame.get("joint_names") or []
    positions = frame.get("positions") or []
    lookup = {name: float(value) for name, value in zip(stream_names, positions)}
    return [lookup.get(name, 0.0) for name in robot.joint_names]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765", help="ReplayStreamPublisher stream_url")
    args = parser.parse_args()

    robot = build_articulation()
    stream = connect(args.url)
    print(f"blacknode: streaming {args.url} into Isaac Lab articulation")
    try:
        while True:
            frame = stream.recv_json()
            if frame is None:
                break
            targets = torch.tensor([order_positions(robot, frame)], dtype=torch.float32)
            robot.set_joint_position_target(targets)
            robot.write_data_to_sim()
    finally:
        stream.close()


if __name__ == "__main__":
    main()
