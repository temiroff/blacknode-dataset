# blacknode-dataset

`blacknode-dataset` records synchronized robot demonstrations as recoverable episodes, validates them, replays them, and exports explicit training or repository profiles.

## Components

| Component | Default | Purpose |
|---|---:|---|
| `recording` | On | Dataset creation, camera collection, journaling, and episode recording |
| `replay` | On | Episode browsing, playback, trimming, streaming, and trajectory smoothing |
| `validation` | On | Dataset summaries, statistics, and consistency checks |
| `evaluation` | Off | Episode scoring |
| `export` | Off | HDF5 and LeRobot Parquet/MP4 export |
| `publishing` | Off | Blacknode Hub export, WebSocket streaming, and explicit repository upload |

## Typical workflow

1. Install the package and open `teleoperation-episode-recording.json`.
2. Connect one or more `blacknode.frame-stream` camera handles and a synchronized `blacknode.sample-stream`.
3. Create or reopen the dataset with `DatasetCreate`.
4. Confirm sources are fresh, then record, pause, resume, save, or discard from `EpisodeRecorder`.
5. Review with `DatasetBrowser` or `EpisodeReplay`.
6. Run `EpisodeDatasetValidate` before selecting an export or publishing profile.

The `teleoperation-episode-recording` template includes a **Collect episodes**
Workflow App. Opening it from the editor shortcut presents camera and motion
views, dataset fields, live status, and episode controls. **Edit workflow**
reveals the same underlying nodes for advanced configuration. Motion remains
disarmed until the operator explicitly confirms arming.

Recording journals frames before final conversion and preserves incomplete runs for recovery. Saved episodes contain a manifest, Parquet robot data, and one MP4 per camera under `~/.blacknode/datasets/<dataset-id>` by default.

Key optional nodes include `TrajectorySmoother`, `StreamPublisher`, `HDF5EpisodeExport`, `LeRobotV3Export`, `BlacknodeHubExport`, and `HuggingFaceDatasetUpload`. Export and upload are separate actions; credentials are read from explicit configuration or environment and are never written into dataset artifacts.

## Safety

- Recording, browsing, smoothing, replay, and export never command motion.
- Stale or malformed samples are rejected; joint order and units are preserved.
- Save and trim operations are atomic.
- Network publishing requires an explicit upload action.

## Install and verify

```powershell
blacknode packages install https://github.com/temiroff/blacknode-dataset.git
blacknode packages setup blacknode-dataset
python -m pytest packages/blacknode-dataset/tests
Get-ChildItem packages\blacknode-dataset\templates\*.json | ForEach-Object { blacknode validate $_.FullName }
```

Wire schemas and external replay clients are documented in [clients/README.md](clients/README.md).
