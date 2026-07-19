"""Broadcast a stream to external apps over a plain WebSocket.

``StreamPublisher`` is the transport-neutral producer. Connect any stream handle
into its ``stream`` input — a recorded replay from a DatasetBrowser, a derived
trajectory such as a smoothed or policy-predicted replay, or a live
``blacknode.sample-stream`` — and it broadcasts each frame as JSON to every
connected subscriber. It is strictly read-only: it never
opens a robot connection or commands motion. Receiving apps (ROS 2, Maya, Isaac
Sim, ...) run a small subscriber that maps the joint values into their own rig /
topic / prim; example clients ship under ``clients/``.
"""
from __future__ import annotations

from blacknode.node import Any as AnyPort
from blacknode.node import Bool, Dict, Enum, Float, Image, Int, Text, node

from . import runtime

_CATEGORY = "Dataset"


@node(name="StreamPublisher", live=True, category=_CATEGORY,
      description="Broadcast a stream frame-by-frame to any app over a plain WebSocket. Connect a 'stream' handle "
                  "(from DatasetBrowser, TrajectorySmoother, policy replay, or a live sample-stream), set action=start, and "
                  "connect subscribers to stream_url. Recorded replay follows Dataset Browser play and seek by "
                  "default. Read-only: it never commands hardware.",
      inputs={"trigger": AnyPort,
              "action": Enum(["status", "start", "stop"], default="status"),
              "run_id": Text(default="stream"),
              "stream": Dict(default={}),
              "host": Text(default="127.0.0.1"),
              "port": Int(default=8765),
              "source": Enum(["action", "observation", "leader"], default="action"),
              "units": Enum(["radians", "degrees"], default="radians"),
              "fps": Float(default=0),
              "rate": Float(default=1.0),
              "loop": Bool(default=True),
              "sync_to_browser": Bool(default=True)},
      outputs={"stream_url": Text, "streaming": Bool, "clients": Int,
               "status": Dict, "dashboard": Image, "report": Text})
def stream_publisher(ctx: dict) -> dict:
    action = str(ctx.get("action") or "status").strip().lower()
    run_id = str(ctx.get("run_id") or "stream").strip() or "stream"
    try:
        if action == "start":
            status = runtime.start_stream(
                run_id=run_id,
                stream=ctx.get("stream") or {},
                host=str(ctx.get("host") or "127.0.0.1"),
                port=int(ctx.get("port") or 8765),
                fps=float(ctx.get("fps") or 0),
                rate=float(ctx.get("rate") or 1.0),
                loop=bool(ctx.get("loop")),
                source=str(ctx.get("source") or "action"),
                units=str(ctx.get("units") or "radians"),
                sync_to_browser=bool(ctx.get("sync_to_browser", True)),
            )
        else:
            status = runtime.control_stream(run_id, action)
        return runtime.stream_outputs(status)
    except Exception as exc:  # noqa: BLE001 - surfaced in node report
        return runtime.stream_outputs({
            "run_id": run_id, "running": False, "streaming": False, "clients": 0,
            "sent": 0, "frames": 0, "stream_url": "",
            "last_error": f"{type(exc).__name__}: {exc}",
        })
