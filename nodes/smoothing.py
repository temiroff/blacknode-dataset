"""Smooth a recorded episode's trajectories and hand out a new replay token.

``ReplayTrajectorySmoother`` is a connectable filter: wire a DatasetBrowser's
``replay_token`` into it, pick a method, and connect its ``replay_token`` output
into ``ReplayStreamPublisher`` (or anything that consumes a replay token). Because
replay has the whole episode available, the default filters are non-causal and
zero-lag — a cubic smoothing B-spline (SciPy) or a zero-phase Gaussian — so the
streamed motion is smoother than the shaky recording with no added latency. It is
read-only and never commands hardware.
"""
from __future__ import annotations

from blacknode.node import Any as AnyPort
from blacknode.node import Dict, Enum, Float, Image, Text, node

from . import runtime

_CATEGORY = "Dataset"


@node(name="ReplayTrajectorySmoother", category=_CATEGORY,
      description="Smooth a recorded episode's joint trajectories offline (zero-lag) and emit a new replay_token. "
                  "Wire DatasetBrowser.replay_token in and this node's replay_token into ReplayStreamPublisher to "
                  "stream smoothed motion. spline (cubic B-spline) and savgol need SciPy; gaussian, moving_average, "
                  "and one_euro are numpy-only. Read-only; never commands hardware.",
      inputs={"trigger": AnyPort,
              "replay_token": Text(default=""),
              "method": Enum(["spline", "gaussian", "savgol", "moving_average", "one_euro", "none"], default="spline"),
              "strength": Float(default=1.0),
              "preview_source": Enum(["action", "observation", "leader"], default="action"),
              "preview_joint": Text(default="")},
      outputs={"replay_token": Text, "episode": Dict, "preview": Image, "report": Text})
def replay_trajectory_smoother(ctx: dict) -> dict:
    try:
        info = runtime.register_smoothed_replay(
            str(ctx.get("replay_token") or ""),
            str(ctx.get("method") or "spline"),
            float(ctx.get("strength") or 1.0),
            preview_source=str(ctx.get("preview_source") or "action"),
            preview_joint=str(ctx.get("preview_joint") or ""),
        )
        return runtime.smoother_outputs(info)
    except Exception as exc:  # noqa: BLE001 - surfaced in node report
        return {"replay_token": "", "episode": {}, "preview": "",
                "report": f"trajectory smoother FAILED: {exc}"}
