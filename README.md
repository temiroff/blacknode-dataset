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

## Nodes

| Node | Purpose |
| --- | --- |
| `DatasetCreate` | Create or reopen a dataset with a stable task, FPS, robot type, and metadata. |
| `DatasetCameraStreamList` | Collect any number of camera stream handles through dynamic sockets. |
| `EpisodeRecorder` | Start, pause, resume, save/finalize, stop, or discard a recording and render its live dashboard. |
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

`stop` preserves the current journal under `incomplete/<run-id>`. The same run
can later be saved/finalized or discarded. After three consecutive source
errors, the recorder pauses and reports the last error. The saved timeline uses
`frame_index / fps`; source capture and wall-clock timestamps remain attached
to the robot and camera samples.

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

## Test

```powershell
$env:PYTHONPATH="python"
python -m pytest packages/blacknode-dataset/tests
```

## License

Apache-2.0, same as Blacknode.
