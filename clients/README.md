# Replay stream clients

`ReplayStreamPublisher` (Dataset category) broadcasts the episode selected in the
**Dataset Browser** frame-by-frame over a plain WebSocket. These scripts are
example **subscribers** — one small file per app. Each maps the joint values into
that app's own rig, topic, or prim, which is knowledge only the app has, so the
mapping lives here on the receiving side rather than in a Blacknode node.

## Run the publisher

1. Add a **DatasetBrowser** node, choose a dataset/episode.
2. Add a **ReplayStreamPublisher** node and wire `DatasetBrowser.replay_token`
   into its `replay_token`.
3. Set `action = start` and cook it. Its `stream_url` is `ws://127.0.0.1:8765`
   by default. `source` selects which recorded signal to send (`action`,
   `observation`, or `leader`).

The node is **read-only** — it replays recorded data and never commands
hardware. What a subscriber does with the values is the subscriber's decision.

To smooth shaky recordings before they reach your app, drop a
**ReplayTrajectorySmoother** between the browser and the publisher
(`DatasetBrowser → ReplayTrajectorySmoother → ReplayStreamPublisher`). It filters
the whole episode offline with zero lag; subscribers need no changes. Smoothed
frames additionally carry a `"smoothing": {"method": ..., "strength": ...}` field.

## Wire schema

Each WebSocket message is one JSON object per frame:

```json
{
  "kind": "blacknode.episode-frame",
  "schema_version": 1,
  "frame_index": 42,
  "frames": 300,
  "timestamp": 1.4,
  "joint_names": ["shoulder_pan", "shoulder_lift", "elbow", "gripper"],
  "positions": [0.12, -0.44, 1.02, 0.0],
  "source": "action",
  "units": "radians",
  "action": {"shoulder_pan": 0.12, "...": 0.0},
  "observation": {"shoulder_pan": 0.11, "...": 0.0},
  "leader": {"shoulder_pan": 0.12, "...": 0.0},
  "fps": 30.0,
  "seq": 1289
}
```

`positions` is `joint_names` resolved against the selected `source`, so most
clients only need those two arrays plus `units`.

## Included clients

| Script | Runs in | Maps to |
| --- | --- | --- |
| `ros2_bridge.py` | a sourced ROS 2 / WSL Python (`rclpy`) | `sensor_msgs/JointState` on a topic |
| `maya_client.py` | Maya / `mayapy` | rig attributes via a `joint -> attr` JSON map |
| `isaac_lab_client.py` | Isaac Lab Python | articulation joint position targets |

All three import `blacknode_ws.py` (also in this folder), a dependency-free
WebSocket client, so they run without `pip install` inside those environments.

## Add another app

```python
from blacknode_ws import connect

stream = connect("ws://127.0.0.1:8765")
while True:
    frame = stream.recv_json()
    if frame is None:
        break
    for name, value in zip(frame["joint_names"], frame["positions"]):
        your_app.set_joint(name, value)  # units are frame["units"]
```

That is the whole seam: one subscriber, one mapping. No Blacknode changes are
needed to support a new target.
