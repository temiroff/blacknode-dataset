# blacknode-dataset

`blacknode-dataset` records synchronized robot demonstrations as recoverable
episodes. It samples live teleoperation and camera streams at a fixed dataset
FPS, journals every captured frame, atomically saves Parquet and MP4 artifacts,
validates dataset consistency, and provides HDF5 and structured Parquet/MP4
export profiles.

## Install

```powershell
blacknode packages install https://github.com/temiroff/blacknode-dataset.git
blacknode packages setup blacknode-dataset
```

The recorder consumes two Blacknode stream contracts:

- `blacknode.sample-stream` carries synchronized leader state, follower
  observation, commanded action, joint order, units, sequence, and capture time.
- `blacknode.frame-stream` carries a camera identity plus a current JPEG
  snapshot endpoint with sequence and capture-time headers.

Start with `templates/teleoperation-episode-recording.json`. The template
provides a calibrated SO-ARM101 leader/follower flow, a local camera preview,
dynamic multi-camera collection, a dataset, and an inline recorder dashboard.
Motion starts disarmed and recording starts in `status` mode.

`DatasetCreate` stores data under `~/.blacknode/datasets/<dataset-id>` by
default. The editor shows that resolved location and provides **Choose folder**
to select a visible storage root such as `E:\RobotData`; the dataset ID remains
its own subfolder. **Use default** restores the application-data location. The
leading dot follows the conventional application-data directory name and does
not affect dataset contents or portability.

## Nodes

| Node | Purpose |
| --- | --- |
| `DatasetCreate` | Create or reopen a dataset with a stable task, FPS, robot type, and metadata. |
| `DatasetBrowser` | Pick a storage root, dataset, episode, and camera, replay or trim it, and inspect synchronized robot and timing data. |
| `TrajectorySmoother` | Smooth a recorded episode's joint trajectories offline (zero-lag B-spline / Gaussian / Savitzky-Golay / One-Euro) and emit a new `stream` handle. Read-only. |
| `StreamPublisher` | Publish Browser-synchronized replay poses or a live sample stream over a plain WebSocket for Maya, ROS 2, Isaac Sim, and other subscribers. Read-only; never commands hardware. |
| `DatasetCameraStreamList` | Collect any number of camera stream handles through dynamic sockets. |
| `EpisodeRecorder` | Start, pause, resume, save/finalize, stop, or discard a recording and render its live camera and capture-health dashboard. |
| `EpisodeReplay` | Play any saved episode camera video and expose its task, timing, joints, camera, and exact artifact paths. |
| `EpisodeDatasetSummary` | Inspect saved episodes, frame totals, duration, cameras, joints, and recoverable journals. |
| `EpisodeDatasetValidate` | Validate manifests, Parquet rows, videos, timestamps, and feature consistency. |
| `HDF5EpisodeExport` | Check or export one Blacknode HDF5 file per saved episode. |
| `LeRobotV3Export` | Write the v3 structured Parquet/MP4 repository profile. |
| `HuggingFaceDatasetUpload` | Check or explicitly publish a prepared repository export. |

## Recording lifecycle

1. Add a `Camera` with `selection: 0` and connect `frame_stream` to the camera
   list's dashed **connect to add** socket.
2. Duplicate the camera with `selection: 1`, `2`, and so on when the task needs
   additional views. The camera list creates `camera_1`, `camera_2`, and further
   sockets dynamically.
3. Run `DatasetCreate` with a stable dataset ID, task, FPS, and robot type.
4. Confirm the robot and camera dashboards are fresh, arm teleoperation, and
   click **Record** on `EpisodeRecorder`.
5. Use **Pause** and **Resume** while positioning the scene.
6. Use **Save episode** for a successful demonstration or **Discard** for an
   unsuccessful attempt.
7. Run `EpisodeDatasetValidate`, then select the export profile required by the
   training workflow.

While recording, the dashboard shows the first camera stream, captured frame
count, dataset time, effective capture rate, robot/camera source age, and dropped
frames. A freshness failure names the exact source and reports its measured age
and configured limit.

Recorder buttons control the managed recorder directly after the graph has
resolved its inputs once. Pause, resume, save, stop, and discard do not recook
the robot/camera network. Save stops capture before encoding the episode, so the
frame count is final as soon as saving begins. The recorder displays its active
journal path and the final saved episode directory.

Connect `EpisodeReplay.video` to an `Output` node to play a saved camera video
with browser seek and playback controls. Select `episode_index` and optionally a
camera name. Replay is read-only and never publishes robot commands. Its
`episode_path`, `video_path`, and `replay` outputs identify the exact saved
artifacts and recorded metadata.

For interactive review, open `templates/dataset-browser.json`. Its
`DatasetBrowser` panel lists every valid dataset beneath the chosen root and
provides dataset, episode, and camera selectors. Playing or seeking the video
updates the current frame index, dataset timestamp, sample sequence, camera
sequence, and the complete ordered leader/observation/action joint table. The
panel also shows the task, save time, episode directory, MP4 path, and Parquet
robot-data path. Dataset browsing and replay are read-only and never command a
robot. The replay toolbar provides Replay/Pause, restart, previous/next frame,
0.25×–2× playback speed, looping controls, and radians/degrees display switching.
Stored values remain unchanged; the unit switch only converts the review table.
Pause or seek to a frame, then use **Cut before** or **Cut after** to remove the
unwanted beginning or ending. The selected frame is retained. Blacknode shows
the exact number of frames to remove and requires confirmation before changing
the dataset. The edit atomically trims every camera video and the matching
Parquet rows, resets the remaining frame indexes and dataset timestamps, and
updates episode metadata. Trimming is disabled while that dataset has an active
recorder, and it never sends robot commands.

New episode videos are encoded as browser-compatible H.264 (`avc1`) with a
fast-start MP4 index. When browsing an older `mp4v` episode, Blacknode creates
an H.264 playback copy in the operating system's temporary cache and preserves
the original dataset artifact unchanged.

Connect `DatasetCreate.dataset` to `DatasetBrowser.dataset` to open the same
dataset automatically. With no connection and an empty `root`, both nodes use
the same `~/.blacknode/datasets` default. An explicit browser root or dataset ID
still allows reviewing another collection.

`stop` preserves the current journal under `incomplete/<run-id>`. The same run
can later be saved/finalized or discarded. After three consecutive source
errors, the recorder pauses and reports the last error. The saved timeline uses
`frame_index / fps`; source capture and wall-clock timestamps remain attached
to the robot and camera samples.

## Stream to external apps

`StreamPublisher` fans a stream out to any number of apps over a plain WebSocket,
so the same source can drive ROS 2, Maya, Isaac Sim, or anything else at once. It
is transport-neutral and **read-only**: it never opens a robot connection or
commands motion. What a subscriber does with the values is the subscriber's
responsibility. Its `stream` input accepts a `stream` handle from any producer —
a recorded replay (`DatasetBrowser`/`TrajectorySmoother`) or a live
`blacknode.sample-stream` — so it streams anything, not just replay.

1. Load `templates/replay-stream.json`, or wire `DatasetBrowser.stream`
   into a `StreamPublisher`.
2. Select a dataset and episode in the browser.
3. Keep `sync_to_browser=true`, set the publisher's `action` to `start`, and cook
   it. `stream_url` is `ws://127.0.0.1:8765` by default. `source` chooses which
   recorded signal to send (`action`, `observation`, or `leader`). The server is
   now ready, but it does not advance the replay on its own.
4. Start a subscriber, then play or seek in Dataset Browser. Every displayed
   playback frame and timeline scrub pose is published. Each broadcast carries `joint_names`,
   an ordered `positions` array, `units`, timestamps, and the full per-joint
   dictionaries.

Set `sync_to_browser=false` for an explicit independent replay; in that mode,
`rate` scales its playback speed and `loop` repeats the episode.

`action=stop`, **Stop all**, or restarting the server ends the stream and closes
the port.

The self-contained `clients/maya_stream.py` window provides **Get joints /
Connect**. It receives joint names without changing the rig, lets each dataset
joint target a Maya attribute with X/Y/Z axis, +1/-1 direction, and scale, then
stores that mapping in Maya preferences. Rig values change only when Browser
playback or timeline seeking publishes a pose.

### Smooth shaky trajectories before streaming

Insert a `TrajectorySmoother` between the browser and the publisher
(`DatasetBrowser.stream → TrajectorySmoother.stream → StreamPublisher.stream`) to
calm jittery recordings before they drive an app. Because replay has the whole
episode available, the filters are **non-causal and zero-lag** — the smoothed
motion has no added latency:

| `method` | Filter | Needs |
| --- | --- | --- |
| `spline` (default) | Cubic smoothing B-spline (control points + knots) | SciPy |
| `savgol` | Savitzky-Golay, peak-preserving | SciPy |
| `gaussian` | Zero-phase Gaussian | numpy only |
| `moving_average` | Zero-phase box | numpy only |
| `one_euro` | Causal One-Euro (for the low-latency / live case) | numpy only |
| `none` | Pass through unchanged | — |

`strength` is the single tuning knob (larger = smoother). The node reports the
measured jerk reduction and renders a raw-vs-smoothed sparkline of one joint on
its `preview` output, so a shaky recording and its smoothed version are visible
side by side on the canvas. When SciPy is absent, `spline`/`savgol` fall back to
`gaussian` and say so.

After the graph resolves the smoother input once, changing `method`, `strength`,
`preview_source`, or `preview_joint` recomputes only the in-memory smoother.
Dataset Browser and other upstream nodes do not cook again. The preview updates
directly, and every running StreamPublisher connected to the previous smoother
output is hot-swapped to the new token while retaining its URL and subscribers.
If Browser playback or seeking has already sent a pose, the newly smoothed value
for that same frame is published immediately so connected apps update in place;
no additional timeline movement is required. The recorded dataset remains unchanged.

The B-spline representation is inspired by
[B-spline Policy (arXiv:2607.09648)](https://arxiv.org/abs/2607.09648), which
predicts continuous B-spline actions in-policy for faster execution; here the
same curve is fit offline to recorded trajectories purely to smooth them for
replay and streaming.

Example subscribers ship in [`clients/`](clients/README.md), one small file per
app, all built on a dependency-free WebSocket client that runs unchanged inside
mayapy, an Isaac Lab environment, or a ROS 2 node:

| Client | Runs in | Maps the stream to |
| --- | --- | --- |
| `clients/ros2_bridge.py` | a sourced ROS 2 / WSL Python | `sensor_msgs/JointState` on a topic |
| `clients/maya_client.py` | Maya / `mayapy` | rig attributes via a joint→attribute map |
| `clients/isaac_lab_client.py` | Isaac Lab Python | articulation joint position targets |

Adding another target is one more subscriber file — no Blacknode change is
needed. See [`clients/README.md`](clients/README.md) for the wire schema and a
40-line template for a new app.

## Native dataset layout

```text
dataset.json
episodes/episode-000000/
  episode.json
  data.parquet
  cameras/<camera>.mp4
incomplete/<run-id>/
  episode.json
  frames.jsonl
  cameras/<camera>/frame-000000.jpg
```

The native Parquet table stores observation state, leader state, action,
dataset time, frame and episode indexes, task, sample sequence, source capture
time, wall-clock record time, and per-camera sequence/capture time. The episode
manifest stores joint order, units, FPS, robot identity, calibration references,
camera shapes, codec information, duration, and save time.

## HDF5 export profile

`HDF5EpisodeExport` writes one `episode_<index>.hdf5` file per saved episode:

```text
/observations/qpos                 float32 [T, joints]
/observations/leader               float32 [T, joints]
/observations/images/<camera>      uint8   [T, height, width, 3] RGB
/observations/camera_metadata/*    int64   [T]
/action                            float32 [T, joints]
/timestamp                         float64 [T]
/frame_index                       int64   [T]
/sample_sequence                   int64   [T]
/captured_at_ns                    int64   [T]
/recorded_at_ns                    int64   [T]
/metadata/joint_names              UTF-8   [joints]
```

Root and metadata attributes preserve FPS, task, robot type, units, hardware
IDs, calibration references, source paths, image color space, and export time.
Images support `gzip`, `lzf`, or uncompressed storage. Export happens through a
temporary directory and becomes visible after every episode and camera passes
frame-count validation.

## Structured Parquet/MP4 export profile

`LeRobotV3Export` writes:

```text
meta/info.json
meta/stats.json
meta/tasks.parquet
meta/episodes/chunk-000/file-000.parquet
data/chunk-000/file-<episode>.parquet
videos/observation.images.<camera>/chunk-000/file-<episode>.mp4
blacknode-export.json
```

The export includes typed feature definitions, task and episode metadata,
normalization statistics, global frame indexes, camera video metadata, robot
type, dataset split information, and Blacknode export provenance.

`HuggingFaceDatasetUpload` starts in `check` mode. Publishing occurs only after
choosing `action=upload`. Authentication uses the configured account or
`HF_TOKEN`; credentials are never written into dataset manifests or node
outputs.

## Before-training smoothing

`HDF5EpisodeExport` and `LeRobotV3Export` take an optional `smoothing` method
(`none` by default, plus `spline`, `gaussian`, `savgol`, `moving_average`) and a
`smoothing_strength`. When set, each episode's `observation.state`, `leader`, and
`action` trajectories are filtered per-episode (zero-lag, never across episode
boundaries) before writing, so a policy trains on the intended motion rather than
teleoperation jitter. `spline`/`savgol` need SciPy; the numpy-only methods always
work. The choice is recorded in the export (`smoothing` attribute in HDF5,
`smoothing` field in `blacknode-export.json`). With `smoothing=none` the export is
byte-for-byte unchanged. This is the same zero-lag filter family as the
`TrajectorySmoother` streaming node, applied here at export time instead.

## Test

```powershell
$env:PYTHONPATH="python"
python -m pytest packages/blacknode-dataset/tests
```

## License

Apache-2.0, same as Blacknode.
