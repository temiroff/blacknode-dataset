# blacknode-dataset

Blacknode-native robot episode recording with no LeRobot runtime dependency.
It samples a live teleoperation stream and one or more camera streams at a
fixed dataset FPS, journals every frame for crash recovery, then atomically
saves Parquet and MP4 episode artifacts. Separate exporters produce an
ACT-style HDF5 episode set or a LeRobot v3-compatible Hugging Face dataset tree.

## Install

```powershell
blacknode packages install https://github.com/temiroff/blacknode-dataset.git
blacknode packages setup blacknode-dataset
```

The package expects stream handles from:

- `ROS2LeaderFollower.sample_stream` (`blacknode.sample-stream`), containing
  synchronized leader state, follower observation, and commanded action in
  radians.
- `Camera.frame_stream` (`blacknode.frame-stream`), containing a
  latest JPEG snapshot endpoint with capture sequence and timestamp headers.

It uses these generic contracts rather than importing either package, so other
robot and camera packages can implement the same handles later.

Start with `templates/teleoperation-episode-recording.json`. It includes the
SO-ARM101 leader/follower setup and local camera, starts motion disarmed, and
starts the recorder in non-mutating `status` mode.

## Nodes

| Node | Purpose |
| --- | --- |
| `DatasetCreate` | Create or reopen a dataset with fixed task and FPS. |
| `DatasetCameraStreamList` | Append camera handles into a chainable list for any number of cameras. |
| `EpisodeRecorder` | Start, pause, resume, save/finalize, stop, or discard a recording; its `dashboard` Image renders the current state and counters. |
| `EpisodeDatasetSummary` | Inspect saved and incomplete episodes. |
| `EpisodeDatasetValidate` | Check manifests, Parquet, videos, timestamps, and feature consistency. |
| `HDF5EpisodeExport` | Check or explicitly export one ACT-style HDF5 file per episode. |
| `LeRobotV3Export` | Produce the LeRobot v3 Parquet/MP4 repository layout without importing LeRobot. |
| `HuggingFaceDatasetUpload` | Check or explicitly upload an export to a Hugging Face dataset repository. |

## Recording lifecycle

1. Add a `Camera` with `selection: 0`, then connect its stream directly to the
   camera list's dashed **connect to add** socket. Duplicate `Camera` with
   `selection: 1`, `2`, and so on for more cameras. The list creates
   `camera_1`, `camera_2`, and further inputs dynamically with no fixed limit.
2. Run `DatasetCreate` once with a stable dataset ID, task, and FPS.
3. Click **Record** on `EpisodeRecorder`. Recording requires fresh robot and
   camera timestamps; it can additionally require teleoperation to be armed.
4. Use the recorder's **Pause**, **Resume**, and **Save episode** controls.
   Saving atomically commits the episode. The `action` property remains
   available for automated workflows and agents.
5. `stop` intentionally leaves the journal under `incomplete/` for recovery.
   A later `save`/`finalize` commits that journal; `discard` removes it.
6. Validate, choose `action=export` on the HDF5 exporter if needed, then use
   LeRobot export and set the upload node to `upload` only when ready.

After three consecutive source errors the recorder pauses instead of silently
writing stale or misaligned data. Timestamps in the final dataset are the
regular `frame_index / fps` timeline expected by training pipelines; original
source and wall-clock timestamps remain in the native journal/Parquet data.

## On-disk native format

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

The LeRobot exporter writes `meta/info.json`, `meta/stats.json`, task and
episode Parquet metadata, chunked episode Parquet data, and per-camera MP4
files. This makes the exported directory readable by LeRobot v3 tooling while
keeping LeRobot optional. Format compatibility is validated structurally in
this package; consumers should still pin and test the LeRobot release used for
training because its dataset schema may evolve.

## HDF5 episode layout

HDF5 does not have one universal robotics schema. `HDF5EpisodeExport` uses the
widely recognizable ACT/ALOHA convention where practical and writes one
`episode_<index>.hdf5` file per saved episode:

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

The exporter never invents unavailable velocity or force fields. Source
hardware IDs, calibration references, units, FPS, task, and image color space
are stored as attributes. Images support `gzip`, `lzf`, or no HDF5 compression.
The export is written through a temporary directory and becomes visible only
after every episode and camera passes frame-count validation.

NVIDIA Isaac Lab-Arena also records HDF5, but its converter expects an
Arena-specific layout (for example `observations/camera_obs/robot_head_cam_rgb`)
inside a consolidated recording. This ACT-style export is not advertised as a
drop-in Arena recording. For GR00T, use `LeRobotV3Export` as the direct data
bridge and validate the embodiment modality mapping required by the exact
GR00T release.

Authentication uses the standard Hugging Face login or `HF_TOKEN`. Tokens are
never written into dataset manifests or node outputs.

## Test

```powershell
$env:PYTHONPATH="python"
python -m pytest packages/blacknode-dataset/tests
```
