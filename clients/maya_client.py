"""Drive a Maya rig from a Blacknode replay stream.

Subscribes to a ReplayStreamPublisher WebSocket and applies each frame's joint
values to Maya attributes on a background thread. Run it inside Maya (Script
Editor) or with mayapy alongside blacknode_ws.py:

    mayapy maya_client.py --url ws://127.0.0.1:8765 --map joint_map.json

joint_map.json maps each streamed joint name to a Maya attribute and axis, e.g.:

    {
      "shoulder_pan":  {"attr": "arm_shoulder.rotateY"},
      "shoulder_lift": {"attr": "arm_shoulder.rotateX", "scale": -1.0},
      "gripper":       {"attr": "arm_gripper.translateZ", "scale": 0.01}
    }

Streamed values are radians by default (set units=degrees on the node to change);
Maya rotate attributes are degrees, so rotate.* targets are converted for you.
"""
from __future__ import annotations

import argparse
import json
import math
import threading

import maya.cmds as cmds  # available inside Maya / mayapy
import maya.utils

from blacknode_ws import connect


def _apply(mapping: dict, frame: dict) -> None:
    names = frame.get("joint_names") or []
    positions = frame.get("positions") or []
    radians = str(frame.get("units") or "radians").startswith("rad")
    for name, value in zip(names, positions):
        target = mapping.get(name)
        if not target:
            continue
        attr = target["attr"]
        out = float(value) * float(target.get("scale", 1.0)) + float(target.get("offset", 0.0))
        if radians and ".rotate" in attr.lower():
            out = math.degrees(out)
        try:
            cmds.setAttr(attr, out)
        except Exception as exc:  # noqa: BLE001 - keep streaming past a bad attr
            cmds.warning(f"blacknode: could not set {attr}: {exc}")


def _run(url: str, mapping: dict) -> None:
    stream = connect(url)
    try:
        while True:
            frame = stream.recv_json()
            if frame is None:
                break
            # Maya is not thread-safe; marshal the edit onto the main thread.
            maya.utils.executeInMainThreadWithResult(_apply, mapping, frame)
    finally:
        stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="ws://127.0.0.1:8765", help="ReplayStreamPublisher stream_url")
    parser.add_argument("--map", required=True, help="JSON file mapping joint names to Maya attributes")
    args = parser.parse_args()
    with open(args.map, encoding="utf-8") as handle:
        mapping = json.load(handle)
    threading.Thread(target=_run, args=(args.url, mapping), daemon=True, name="blacknode-replay").start()
    print(f"blacknode: streaming {args.url} into Maya on a background thread")


if __name__ == "__main__":
    main()
