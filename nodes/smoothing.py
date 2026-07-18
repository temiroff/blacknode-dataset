"""Smooth a recorded episode's trajectories and hand out a new stream handle.

``TrajectorySmoother`` is a connectable filter: wire a DatasetBrowser's ``stream``
output into it, pick a method, and connect its ``stream`` output into
``StreamPublisher`` (or anything that consumes a replay stream). Because replay has
the whole episode available, the default filters are non-causal and zero-lag — a
cubic smoothing B-spline (SciPy) or a zero-phase Gaussian — so the streamed motion
is smoother than the shaky recording with no added latency. It is read-only and
never commands hardware.
"""
from __future__ import annotations

from blacknode.node import Any as AnyPort
from blacknode.node import Dict, Enum, Float, Image, Text, node

from . import runtime

_CATEGORY = "Dataset"


@node(name="TrajectorySmoother", category=_CATEGORY,
      description="Smooth a recorded episode's joint trajectories offline (zero-lag) and emit a new 'stream' handle. "
                  "Wire DatasetBrowser.stream in and this node's stream into StreamPublisher to broadcast smoothed "
                  "motion while preserving the exact first and last episode poses. spline (cubic B-spline) and "
                  "savgol need SciPy; gaussian, moving_average, and one_euro are numpy-only. Read-only; never "
                  "commands hardware.",
      inputs={"trigger": AnyPort,
              "stream": Dict(default={}),
              "method": Enum(["spline", "gaussian", "savgol", "moving_average", "one_euro", "none"], default="spline"),
              "strength": Float(default=1.0),
              "preview_source": Enum(["action", "observation", "leader"], default="action"),
              "preview_joint": Text(default="")},
      outputs={"stream": Dict, "episode": Dict, "preview": Image, "report": Text})
def trajectory_smoother(ctx: dict) -> dict:
    try:
        mode, _token, _ = runtime.parse_stream(ctx.get("stream") or {})
        if mode != "replay":
            raise ValueError("TrajectorySmoother needs a recorded replay stream (offline smoothing "
                             "cannot run on a live sample-stream)")
        return runtime.apply_configured_smoother(
            str(ctx.get("__node_id__") or "trajectory_smoother"),
            str(ctx.get("method") or "spline"),
            float(ctx.get("strength") if ctx.get("strength") is not None else 1.0),
            preview_source=str(ctx.get("preview_source") or "action"),
            preview_joint=str(ctx.get("preview_joint") or ""),
            stream=ctx.get("stream") or {},
        )
    except Exception as exc:  # noqa: BLE001 - surfaced in node report
        return {"stream": {}, "episode": {}, "preview": "",
                "report": f"trajectory smoother FAILED: {exc}"}
