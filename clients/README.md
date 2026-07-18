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
3. Keep `sync_to_browser = true`, set `action = start`, and cook it. Its
   `stream_url` is `ws://127.0.0.1:8765` by default. `source` selects which
   recorded signal to send (`action`, `observation`, or `leader`). The publisher
   opens the connection but does not advance the episode itself: Dataset Browser
   playback emits each displayed frame, and dragging or clicking its timeline
   immediately emits the selected pose. Set `sync_to_browser = false` only when
   an independent looping replay is explicitly wanted.

The node is **read-only** — it never commands hardware. What a subscriber does
with the values is the subscriber's decision. The `stream` input also accepts a
live `blacknode.sample-stream` handle, so `StreamPublisher` can stream a live
source, not only recorded replay.

To smooth shaky recordings before they reach your app, drop a
**TrajectorySmoother** between the browser and the publisher
(`DatasetBrowser.stream → TrajectorySmoother.stream → StreamPublisher.stream`). It
filters the whole episode offline with zero lag; subscribers need no changes.
All smoothing methods retain the exact first and last joint samples.
Smoothed frames additionally carry a `"smoothing": {"method": ..., "strength": ...}`
field. After one initial graph run resolves its replay input, changing smoother
parameters recomputes only that smoother and hot-swaps the running publisher;
the WebSocket URL and connected clients remain active. The publisher immediately
re-emits the current Browser frame with the new smoothing result.

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

For every recorded replay, including independent looping mode, a new subscriber first receives a
`blacknode.stream-schema` message containing `joint_names`, `source`, and
`units`, with an empty `positions` list. It also carries `trajectory`, the
complete ordered position arrays for the selected source and episode. It is
configuration data and must not be applied as a pose. Smoother changes publish
an updated schema with a replacement full trajectory. The included clients
handle it automatically. Live sample streams remain frame-only because they do
not have a finite episode trajectory.

## Included clients

| Script | Runs in | Maps to |
| --- | --- | --- |
| `ros2_bridge.py` | a sourced ROS 2 / WSL Python (`rclpy`) | `sensor_msgs/JointState` on a topic |
| `maya_client.py` | Maya / `mayapy` | rig attributes via a `joint -> attr` JSON map |
| `isaac_sim_stream.py` | Isaac Sim Script Editor | articulation DOF position targets |
| `isaac_lab_client.py` | Isaac Lab Python | articulation joint position targets |

The importable clients use `blacknode_ws.py` (also in this folder), a dependency-free
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

**Prefer a paste-and-run window?** `maya_stream.py` is fully self-contained (the
WebSocket client is inlined), so it works with plain `exec` — no `sys.path` setup —
and opens a small window. It reads the **joint names from the stream (the
dataset)**, so it is robot-agnostic: it adds one row per streamed joint for you to
map to a Maya attribute.

```python
exec(open(r"<repo>/packages/blacknode-dataset/clients/maya_stream.py").read())
show_blacknode_stream_window()
```

Click **Get joints / Connect**. The publisher sends the joint schema immediately,
while the Maya rig remains unchanged until Dataset Browser plays or seeks. For
each joint, enter the Maya `node.attr`, select X/Y/Z and +1/-1 direction, and set
an optional scale magnitude. Changes are saved automatically in Maya preferences
and restored the next time the window loads. Blank rows are ignored.

Enable **Path** on any joint row to visualize that mapped Maya node's world-space
motion as a thick cubic spline. Each joint receives a stable, high-contrast color,
and all segments belonging to that joint keep the same color. The synchronized publisher sends the complete
filtered episode range when Maya connects, so path creation does not depend on
playback. Changing the smoother rebuilds the full paths automatically. The path
builder rejects non-finite samples and isolated discontinuity spikes before fitting
the spline. Maya waits until the complete trajectory payload is present, repairs
non-finite or isolated extreme joint values for visualization only, and evaluates
every episode frame exactly once. If adjacent world-space samples are unusually
far apart, Maya cuts the path there instead of drawing a misleading connector.
Each sufficiently long continuous section becomes its own cubic curve. Short
fragments around a gap are discarded, so an invalid origin sample cannot create
a stray linear path. Maya never substitutes a linear curve. Maya forces
each mapped node's dependency graph and world matrix to evaluate before sampling;
viewport refresh is never suspended, so an unsolved origin pose cannot be inserted
by deferred rig evaluation. Curves are created explicitly in world space. The
original dataset remains unchanged. The
generated curves are parented under
`blacknodeDatasetDebugPaths`; **Clear debug paths** deletes them. Editing a joint
mapping evaluates the cached full trajectory again for the new rig target.
The status line confirms when a curve is ready and identifies a mapped node that
has no world-space movement, which is common for a root or fixed rotation control.
Live Maya playback keeps only the newest received WebSocket frame and applies it
from a Maya idle callback, dropping stale frames when Maya is busy instead of
building latency. Status text is repainted at most four times per second so it
does not slow pose application. The complete trajectory is evaluated only when at least one
**Path** checkbox is enabled, once when that path is requested or when its
trajectory/mapping changes; individual streamed frames never rebuild the path.
Executing `maya_stream.py` again closes the previous receiver automatically, so
reloading the client does not leave duplicate WebSocket threads in Maya.
**Stop** sends a WebSocket close frame and closes the socket. StreamPublisher
removes the subscriber immediately, and the graph's `clients`, status, report,
and dashboard outputs reflect the disconnect on the next editor status poll.

### Running inside Isaac Sim

For a persistent menu entry, open **Window → Extensions**, add this Extension
Manager search path, and enable **Blacknode Dataset Replay**:

```text
E:\F\PROJECTS\NVDIA\Blacknode\packages\blacknode-dataset\clients\isaac_extension\exts
```

Isaac then provides **Window → Blacknode Dataset Replay** on current and future
runs after the extension is enabled.

Open **Window → Script Editor** and run these two lines:

```python
exec(open(r"<repo>/packages/blacknode-dataset/clients/isaac_sim_stream.py").read())
show_blacknode_isaac_window()
```

Enter the `StreamPublisher.stream_url` and the USD path of the robot's
articulation root, or select a prim in the Stage and click **Use selected prim**.
Then click **Connect**. The client starts the simulation
timeline when needed and matches dataset joints to articulation DOFs by exact
name. Unmatched DOFs retain their current targets. Dataset Browser play and seek
poses drive the simulated articulation directly. This path uses the WebSocket
directly and does not require ROS 2 or a terminal.

After the articulation loads, the window lists every dataset joint. Select the
matching Isaac DOF or `(ignore)`, and choose `+1` or `-1` to reverse its direction.
Outside calibration mode, **Joint Angle** follows the articulation's measured
position in real time, including motion caused by streaming and physics. During
calibration it changes to **Angle Nudge** and retains the relative adjustment.
Each angle row includes a numeric field beside the slider. Type an exact value
and press Enter to apply it; the field uses the same units, limits, and
calibration behavior as the slider.
Use **Motion Scale** when a simulated joint has a different usable range than
the recorded robot joint. A value above `1` increases travel and a value below
`1` decreases it. This is especially useful for grippers whose finger spacing
does not map one-to-one to the recorded actuator angle. Scale changes reapply
the current replay frame immediately and are saved with the joint mapping.
The joint axis is shown as `USD`: Isaac position commands are scalar DOF targets,
so the physical X/Y/Z axis comes from the USD joint definition and is not changed
by this mapping panel.

To calibrate a simulator model against the real robot's home pose, move each
joint with its **Joint Angle** slider. Use the **Angle units** switch to work in
degrees (default) or radians. Slider movement commands that Isaac DOF immediately.
The slider limits come from the Isaac USD articulation limits, so values outside
those limits cannot be commanded. Stop or pause Browser playback while posing
the robot. When the
complete simulator pose matches the real robot's home, click **Set Home Pose**.
If a commanded slider value is held back by a collision or other physical
constraint, the client detects the stall, clamps that side of the slider to the
reached pose, snaps the slider and numeric field back to the measured angle,
and reports the condition in red. Detection uses lack of progress toward the
target, so small contact jitter does not allow the control to keep increasing.
The client records every mapped simulator value against the dataset's first
trajectory frame as the dataset home. Subsequent streamed targets use:
`sim_home + direction * (dataset_value - dataset_home)`. Stop or pause Browser
playback while applying homes so the next streamed frame does not immediately
override the calibration pose. URL, prim path, unit choice, DOF mappings,
directions, slider values, and home calibration are saved in Isaac's persistent
settings and restored next time.

Use **Go Home** to apply the saved calibration pose again.
**Go Home** also exits calibration-nudge mode and resets all angle sliders to
zero. For incremental calibration, click **Calibrate**: the current simulator
pose becomes the nudge baseline and every angle slider resets to zero without
moving the robot. Adjust only the required joints, then click **Add to Home
Pose**. The reached simulator positions are added to the saved home pose and the
nudge sliders reset to zero, ready for another small adjustment.
Only joints whose measured pose changed are updated, using
`new_home = previous_home + (current_pose - nudge_baseline)`. Incoming replay
frames are held while **CALIBRATING**, so streamed movement cannot be folded
into the home pose accidentally.
Press the orange **CALIBRATING** button again to cancel nudge mode. Cancelling
keeps the robot at its current pose, restores absolute **Joint Angle** values,
and does not modify the saved home pose.

Calibration is also written as a portable Blacknode artifact at
`~/.blacknode/calibrations/isaac/<articulation-name>.json`. **Save Calibration
File** writes it explicitly and **Load Calibration File** restores its mappings,
home pose, display units, and drive settings without moving the robot. The
versioned `blacknode.isaac-articulation-calibration` JSON stores simulator home
positions in radians and can be loaded by later Isaac or reinforcement-learning
code independently of Isaac's private user settings.

Click **Discover Joints** after selecting or typing the articulation root. The
client scans its descendants and matches USD joint prims to the selected Isaac
DOFs. Each joint row provides **Stiffness**, **Damping**, and **Max Force**
fields. Changes apply immediately to the joint's angular or linear USD drive;
**Apply Drive Settings** reapplies every saved value. Drive settings persist in
Isaac settings with the mappings and home calibration. A green `found` label
confirms the joint prim mapping; a red `missing` label requires correcting the
articulation root or Isaac DOF selection.

Use `ros2_bridge.py` when ROS 2 is already part of the simulation graph. Use
`isaac_lab_client.py` for an Isaac Lab application that owns its articulation and
simulation loop.

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
