"""Stream a saved episode replay to external apps over a plain WebSocket.

``ReplayStreamPublisher`` is the transport-neutral producer: it walks the frames
of the episode currently selected in the Dataset Browser and broadcasts each one
as JSON to every connected subscriber. It is strictly read-only — it re-reads
recorded frames and never opens a robot connection or commands motion. Receiving
apps (ROS 2, Maya, Isaac Lab, ...) run a small subscriber that maps the joint
values into their own rig/topic/prim; example clients ship under ``clients/``.
"""
from __future__ import annotations

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Image, Int, Text, node

from . import runtime

_CATEGORY = "Dataset"


@node(name="ReplayStreamPublisher", live=True, category=_CATEGORY,
      description="Broadcast the selected Dataset Browser replay episode frame-by-frame to any app over a plain "
                  "WebSocket. Read-only: it streams recorded joint data and never commands hardware. Wire "
                  "replay_token from a DatasetBrowser node, set action=start, and connect subscribers to stream_url.",
      inputs={"trigger": AnyPort,
              "action": Enum(["status", "start", "stop"], default="status"),
              "run_id": Text(default="replay_stream"),
              "replay_token": Text(default=""),
              "host": Text(default="127.0.0.1"),
              "port": Int(default=8765),
              "source": Enum(["action", "observation", "leader"], default="action"),
              "units": Enum(["radians", "degrees"], default="radians"),
              "fps": Float(default=0),
              "rate": Float(default=1.0),
              "loop": Bool(default=True)},
      outputs={"stream_url": Text, "streaming": Bool, "clients": Int,
               "status": Dict, "dashboard": Image, "report": Text})
def replay_stream_publisher(ctx: dict) -> dict:
    action = str(ctx.get("action") or "status").strip().lower()
    run_id = str(ctx.get("run_id") or "replay_stream").strip() or "replay_stream"
    try:
        if action == "start":
            status = runtime.start_replay_stream(
                run_id=run_id,
                token=str(ctx.get("replay_token") or ""),
                host=str(ctx.get("host") or "127.0.0.1"),
                port=int(ctx.get("port") or 8765),
                fps=float(ctx.get("fps") or 0),
                rate=float(ctx.get("rate") or 1.0),
                loop=bool(ctx.get("loop")),
                source=str(ctx.get("source") or "action"),
                units=str(ctx.get("units") or "radians"),
            )
        else:
            status = runtime.control_replay_stream(run_id, action)
        return runtime.stream_outputs(status)
    except Exception as exc:  # noqa: BLE001 - surfaced in node report
        return runtime.stream_outputs({
            "run_id": run_id, "running": False, "streaming": False, "clients": 0,
            "sent": 0, "frames": 0, "stream_url": "",
            "last_error": f"{type(exc).__name__}: {exc}",
        })
