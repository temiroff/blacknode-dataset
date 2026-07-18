"""Drive a Maya rig from a Blacknode stream.

Subscribes to a StreamPublisher WebSocket and applies each frame's joint values
to Maya attributes on a background thread.

Recommended (inside Maya's Script Editor — Python tab): add this folder to the
path and import it, then call start() with your joint map. Do NOT use
exec(open(...)) — that leaves this folder off sys.path so blacknode_ws can't be
found and gives no argv.

    import sys
    sys.path.append(r"<repo>/packages/blacknode-dataset/clients")
    import maya_client                     # or: importlib.reload(maya_client)
    maya_client.start("ws://127.0.0.1:8765", {
        "shoulder_pan":  {"attr": "arm_shoulder.rotateY"},
        "shoulder_lift": {"attr": "arm_shoulder.rotateX", "scale": -1.0},
        "gripper":       {"attr": "arm_gripper.translateZ", "scale": 0.01},
    })
    # later: maya_client.stop()

Each map entry supports "attr" (required), "scale", and "offset". Streamed values
are radians by default; rotate.* targets are converted to degrees for you.

From a shell you can also run: mayapy maya_client.py --url ... --map joint_map.json
"""
from __future__ import annotations

import math
import os
import sys
import threading

# Make this folder importable so `blacknode_ws` resolves even when this file is
# imported by path or read into Maya. Under exec(open(...)) __file__ is undefined,
# which is exactly why importing (not exec) is the supported path.
try:
    _HERE = os.path.dirname(os.path.abspath(__file__))
    if _HERE not in sys.path:
        sys.path.insert(0, _HERE)
except NameError:
    pass

from blacknode_ws import connect  # noqa: E402

import maya.cmds as cmds  # noqa: E402 - available inside Maya / mayapy
import maya.utils  # noqa: E402

_stream = None
_thread = None


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
    global _stream
    _stream = connect(url)
    try:
        while True:
            frame = _stream.recv_json()
            if frame is None:
                break
            if frame.get("kind") == "blacknode.stream-schema":
                continue
            # Maya is not thread-safe; marshal each edit onto the main thread.
            maya.utils.executeInMainThreadWithResult(_apply, mapping, frame)
    finally:
        try:
            _stream.close()
        except Exception:  # noqa: BLE001
            pass
        _stream = None


def start(url: str = "ws://127.0.0.1:8765", joint_map: dict | None = None) -> None:
    """Start streaming in a background thread. Call stop() to end it."""
    global _thread
    if _thread and _thread.is_alive():
        print("blacknode: stream already running; call maya_client.stop() first")
        return
    _thread = threading.Thread(target=_run, args=(url, dict(joint_map or {})), daemon=True,
                               name="blacknode-replay")
    _thread.start()
    print(f"blacknode: streaming {url} into Maya on a background thread")


def stop() -> None:
    if _stream is not None:
        _stream.close()
    print("blacknode: stream stopped")


def main() -> None:
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Drive a Maya rig from a Blacknode stream")
    parser.add_argument("--url", default="ws://127.0.0.1:8765", help="StreamPublisher stream_url")
    parser.add_argument("--map", required=True, help="JSON file mapping joint names to Maya attributes")
    args = parser.parse_args()
    with open(args.map, encoding="utf-8") as handle:
        start(args.url, json.load(handle))
    if _thread:
        _thread.join()


if __name__ == "__main__":
    main()
