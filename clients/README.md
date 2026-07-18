# Stream clients

`StreamPublisher` (Dataset category) broadcasts a stream frame-by-frame over a
plain WebSocket. These scripts are example **subscribers** — one small file per
app. Each maps the joint values into that app's own rig, topic, or prim, which is
knowledge only the app has, so the mapping lives here on the receiving side rather
than in a Blacknode node.

## Run the publisher

1. Add a **DatasetBrowser** node, choose a dataset/episode.
2. Add a **StreamPublisher** node and wire `DatasetBrowser.stream` into its
   `stream` input.
3. Set `action = start` and cook it. Its `stream_url` is `ws://127.0.0.1:8765`
   by default. `source` selects which recorded signal to send (`action`,
   `observation`, or `leader`). Once streaming, the node shows a **STREAMING**
   badge with a **Stop stream** button.

The node is **read-only** — it never commands hardware. What a subscriber does
with the values is the subscriber's decision. The `stream` input also accepts a
live `blacknode.sample-stream` handle, so `StreamPublisher` can stream a live
source, not only recorded replay.

To smooth shaky recordings before they reach your app, drop a
**TrajectorySmoother** between the browser and the publisher
(`DatasetBrowser.stream → TrajectorySmoother.stream → StreamPublisher.stream`). It
filters the whole episode offline with zero lag; subscribers need no changes.
Smoothed frames additionally carry a `"smoothing": {"method": ..., "strength": ...}`
field.

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

### Running inside Maya

In Maya's Script Editor, **import** the client — do not `exec(open(...).read())`,
which leaves this folder off `sys.path` (so `blacknode_ws` is not found) and
passes no arguments:

```python
import sys
sys.path.append(r"<repo>/packages/blacknode-dataset/clients")
import maya_client                         # importlib.reload(maya_client) to re-import
maya_client.start("ws://127.0.0.1:8765", {
    "shoulder_pan":  {"attr": "arm_shoulder.rotateY"},
    "shoulder_lift": {"attr": "arm_shoulder.rotateX", "scale": -1.0},
    "gripper":       {"attr": "arm_gripper.translateZ", "scale": 0.01},
})
# maya_client.stop() to end it
```

**Prefer a paste-and-run window?** `maya_so101_stream.py` is fully self-contained
(the WebSocket client is inlined), so it works with plain `exec` and opens a small
Start/Stop window — no `sys.path` setup:

```python
exec(open(r"<repo>/packages/blacknode-dataset/clients/maya_so101_stream.py").read())
show_so101_stream_window()
```

Edit its `JOINT_MAP` for your rig. `maya_client.py` (imported) is the leaner
option when you already manage a joint map elsewhere.

For Isaac Sim, the simplest path is often not a Python client at all: run
`ros2_bridge.py` to put the stream on a ROS 2 topic and let Isaac Sim's ROS 2
bridge (OmniGraph: *ROS2 Subscribe JointState → Articulation Controller*) consume
it. Use `isaac_lab_client.py` when you want to drive an Isaac Lab articulation
directly in Python.

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
